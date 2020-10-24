#ifndef RKDG_SWE_PROC_OMPI_STEP_HPP
#define RKDG_SWE_PROC_OMPI_STEP_HPP

#include "rkdg_swe_kernels_processor.hpp"
#include "problem/SWE/problem_slope_limiter/swe_CS_sl_ompi.hpp"
#include "problem/SWE/seabed_update/swe_seabed_update.hpp"

namespace SWE {
namespace RKDG {
template <template <typename> class OMPISimUnitType, typename ProblemType>
void Problem::step_ompi(std::vector<std::unique_ptr<OMPISimUnitType<ProblemType>>>& sim_units,
                        typename ProblemType::ProblemGlobalDataType& global_data,
                        ProblemStepperType& stepper,
                        const uint begin_sim_id,
                        const uint end_sim_id) {
    for (uint stage = 0; stage < stepper.GetNumStages(); ++stage) {
        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            if (sim_units[su_id]->parser.ParsingInput()) {
                sim_units[su_id]->parser.ParseInput(stepper, sim_units[su_id]->discretization.mesh);
            }
        }

        Problem::stage_ompi(sim_units, global_data, stepper, begin_sim_id, end_sim_id);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        if (sim_units[su_id]->writer.WritingOutput()) {
            sim_units[su_id]->writer.WriteOutput(stepper, sim_units[su_id]->discretization.mesh);
        }
    }
}

template <template <typename> class OMPISimUnitType, typename ProblemType>
void Problem::stage_ompi(std::vector<std::unique_ptr<OMPISimUnitType<ProblemType>>>& sim_units,
                         typename ProblemType::ProblemGlobalDataType& global_data,
                         ProblemStepperType& stepper,
                         const uint begin_sim_id,
                         const uint end_sim_id) {
    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile() << "Current (time, stage): (" << stepper.GetTimeAtCurrentStage()
                                                  << ',' << stepper.GetStage() << ')' << std::endl;

            sim_units[su_id]->writer.GetLogFile() << "Exchanging data" << std::endl;
        }

        sim_units[su_id]->communicator.ReceiveAll(CommTypes::bound_state, stepper.GetTimestamp());

        sim_units[su_id]->discretization.mesh.CallForEachDistributedBoundary(
            [&stepper](auto& dbound) { Problem::distributed_boundary_send_kernel(stepper, dbound); });

        sim_units[su_id]->communicator.SendAll(CommTypes::bound_state, stepper.GetTimestamp());
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile() << "Starting work before receive" << std::endl;
        }

        sim_units[su_id]->discretization.mesh.CallForEachElement(
            [&stepper](auto& elt) { Problem::volume_kernel(stepper, elt); });

        sim_units[su_id]->discretization.mesh.CallForEachElement(
            [&stepper](auto& elt) { Problem::source_kernel(stepper, elt); });

        sim_units[su_id]->discretization.mesh.CallForEachInterface(
            [&stepper](auto& intface) { Problem::interface_kernel(stepper, intface); });

        sim_units[su_id]->discretization.mesh.CallForEachBoundary(
            [&stepper](auto& bound) { Problem::boundary_kernel(stepper, bound); });

        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile() << "Finished work before receive" << std::endl;
        }
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile()
                << "Starting to wait on receive with timestamp: " << stepper.GetTimestamp() << std::endl;
        }

        sim_units[su_id]->communicator.WaitAllReceives(CommTypes::bound_state, stepper.GetTimestamp());

        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile() << "Starting work after receive" << std::endl;
        }

        sim_units[su_id]->discretization.mesh.CallForEachDistributedBoundary(
            [&stepper](auto& dbound) { Problem::distributed_boundary_kernel(stepper, dbound); });

        sim_units[su_id]->discretization.mesh.CallForEachElement([&stepper](auto& elt) {
            const uint stage = stepper.GetStage();
            auto& state      = elt.data.state;
            auto& curr_state = elt.data.state[stage];
            auto& next_state = elt.data.state[stage + 1];

            curr_state.solution = elt.ApplyMinv(curr_state.rhs);
            stepper.UpdateState(elt);

            if (SWE::SedimentTransport::bed_update) {
                if (stage + 1 == stepper.GetNumStages()) {
                    // swap back if we are at the last stage
                    std::swap(state[0].q, state[stepper.GetNumStages()].q);
                }

                curr_state.b_solution = elt.ApplyMinv(curr_state.b_rhs);
                set_constant(row(next_state.aux, SWE::Auxiliaries::bath), 0.0);
                set_constant(row(next_state.q, SWE::Variables::ze), 0.0);
                for (uint s = 0; s <= stage; ++s) {
                    row(next_state.aux, SWE::Auxiliaries::bath) +=
                        stepper.ark[stage][s] * row(state[s].aux, SWE::Auxiliaries::bath) +
                        stepper.GetDT() * stepper.brk[stage][s] * state[s].b_solution;

                    row(next_state.q, SWE::Variables ::ze) +=
                        stepper.ark[stage][s] * row(state[s].q, SWE::Variables::ze) +
                        stepper.GetDT() * stepper.brk[stage][s] * row(state[s].solution, SWE::Variables::ze) -
                        stepper.GetDT() * stepper.brk[stage][s] * state[s].b_solution;
                }

                if (stage + 1 == stepper.GetNumStages()) {
                    std::swap(state[0].q, state[stepper.GetNumStages()].q);
                    std::swap(state[0].aux, state[stepper.GetNumStages()].aux);
                }
            }
        });

        if (sim_units[su_id]->writer.WritingVerboseLog()) {
            sim_units[su_id]->writer.GetLogFile() << "Finished work after receive" << std::endl << std::endl;
        }
    }

#pragma omp barrier
#pragma omp master
    { ++(stepper); }
#pragma omp barrier

    if (SWE::SedimentTransport::bed_slope_limiting) {
        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            sim_units[su_id]->communicator.ReceiveAll(CommTypes::baryctr_state, stepper.GetTimestamp());
            sim_units[su_id]->discretization.mesh.CallForEachElement([&stepper](auto& elt) {
                auto& state              = elt.data.state[stepper.GetStage()];
                auto& sl_state           = elt.data.slope_limit_state;
                sl_state.bath_lin        = elt.ProjectBasisToLinear(row(state.aux, SWE::Auxiliaries::bath));
                sl_state.bath_at_baryctr = elt.ComputeLinearUbaryctr(sl_state.bath_lin);
            });
            sim_units[su_id]->discretization.mesh.CallForEachDistributedBoundary([](auto& dbound) {
                std::vector<double> message(2);
                message[0] = (double)dbound.data.wet_dry_state.wet;
                if (dbound.data.wet_dry_state.wet) {
                    message[1] = dbound.data.slope_limit_state.bath_at_baryctr;
                }
                dbound.boundary_condition.exchanger.SetToSendBuffer(CommTypes::baryctr_state, message);
            });
            sim_units[su_id]->communicator.SendAll(CommTypes::baryctr_state, stepper.GetTimestamp());
        }

        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            sim_units[su_id]->communicator.WaitAllReceives(CommTypes::baryctr_state, stepper.GetTimestamp());
            sim_units[su_id]->discretization.mesh.CallForEachDistributedBoundary([](auto& dbound) {
                std::vector<double> message(2);
                dbound.boundary_condition.exchanger.GetFromReceiveBuffer(CommTypes::baryctr_state, message);
                dbound.data.slope_limit_state.wet_neigh[dbound.bound_id]             = (bool)message[0];
                dbound.data.slope_limit_state.bath_at_baryctr_neigh[dbound.bound_id] = message[1];
            });
            SWE::CS_seabed_slope_limiter(stepper, sim_units[su_id]->discretization);
        }

        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            sim_units[su_id]->communicator.WaitAllSends(CommTypes::baryctr_state, stepper.GetTimestamp());
        }
    }

    if (SWE::PostProcessing::wetting_drying) {
        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            sim_units[su_id]->discretization.mesh.CallForEachElement(
                [&stepper](auto& elt) { wetting_drying_kernel(stepper, elt); });
        }
    }

    if (SWE::PostProcessing::slope_limiting) {
        CS_slope_limiter_ompi(stepper, sim_units, begin_sim_id, end_sim_id, CommTypes::baryctr_state);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->discretization.mesh.CallForEachElement([&stepper](auto& elt) {
            bool nan_found = SWE::scrutinize_solution(stepper, elt);

            if (nan_found)
                MPI_Abort(MPI_COMM_WORLD, 0);
        });
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->communicator.WaitAllSends(CommTypes::bound_state, stepper.GetTimestamp());
    }
}
}
}

#endif