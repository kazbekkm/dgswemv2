#ifndef RKDG_SWE_PROC_SERIAL_STEP_HPP
#define RKDG_SWE_PROC_SERIAL_STEP_HPP

#include "rkdg_swe_kernels_processor.hpp"
#include "problem/SWE/problem_slope_limiter/swe_CS_sl_serial.hpp"
#include "problem/SWE/seabed_update/swe_seabed_update.hpp"

namespace SWE {
namespace RKDG {
template <template <typename> class DiscretizationType, typename ProblemType>
void Problem::step_serial(DiscretizationType<ProblemType>& discretization,
                          typename ProblemType::ProblemGlobalDataType& global_data,
                          ProblemStepperType& stepper,
                          typename ProblemType::ProblemWriterType& writer,
                          typename ProblemType::ProblemParserType& parser) {
    for (uint stage = 0; stage < stepper.GetNumStages(); ++stage) {
        if (parser.ParsingInput()) {
            parser.ParseInput(stepper, discretization.mesh);
        }

        Problem::stage_serial(discretization, global_data, stepper);
    }

    if (writer.WritingOutput()) {
        writer.WriteOutput(stepper, discretization.mesh);
    }
}

template <template <typename> class DiscretizationType, typename ProblemType>
void Problem::stage_serial(DiscretizationType<ProblemType>& discretization,
                           typename ProblemType::ProblemGlobalDataType& global_data,
                           ProblemStepperType& stepper) {
    discretization.mesh.CallForEachElement([&stepper](auto& elt) { Problem::volume_kernel(stepper, elt); });

    discretization.mesh.CallForEachElement([&stepper](auto& elt) { Problem::source_kernel(stepper, elt); });

    discretization.mesh.CallForEachInterface(
        [&stepper](auto& intface) { Problem::interface_kernel(stepper, intface); });

    discretization.mesh.CallForEachBoundary([&stepper](auto& bound) { Problem::boundary_kernel(stepper, bound); });

    discretization.mesh.CallForEachElement([&stepper](auto& elt) {
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

    ++stepper;

    if (SWE::SedimentTransport::bed_slope_limiting)
        SWE::CS_seabed_slope_limiter(stepper, discretization);

    if (SWE::PostProcessing::wetting_drying) {
        discretization.mesh.CallForEachElement([&stepper](auto& elt) { wetting_drying_kernel(stepper, elt); });
    }

    if (SWE::PostProcessing::slope_limiting) {
        CS_slope_limiter_serial(stepper, discretization);
    }

    discretization.mesh.CallForEachElement([&stepper](auto& elt) {
        bool nan_found = SWE::scrutinize_solution(stepper, elt);

        if (nan_found) {
            std::cerr << "Fatal Error: NaN found at element " << elt.GetID() << std::endl;
            abort();
        }
    });
}
}
}

#endif
