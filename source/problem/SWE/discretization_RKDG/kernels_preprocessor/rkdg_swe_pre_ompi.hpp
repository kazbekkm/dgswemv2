#ifndef RKDG_SWE_PRE_OMPI_HPP
#define RKDG_SWE_PRE_OMPI_HPP

#include "problem/SWE/problem_preprocessor/swe_pre_init_data.hpp"

namespace SWE {
namespace RKDG {
template <template <typename> class OMPISimUnitType, typename ProblemType>
void Problem::preprocessor_ompi(std::vector<std::unique_ptr<OMPISimUnitType<ProblemType>>>& sim_units,
                                typename ProblemType::ProblemGlobalDataType& global_data,
                                const ProblemStepperType& stepper,
                                const uint begin_sim_id,
                                const uint end_sim_id) {
    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        SWE::initialize_data_parallel_pre_send(
            sim_units[su_id]->discretization.mesh, sim_units[su_id]->problem_input, CommTypes::baryctr_coord);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->communicator.ReceiveAll(CommTypes::baryctr_coord, 0);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->communicator.SendAll(CommTypes::baryctr_coord, 0);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->communicator.WaitAllReceives(CommTypes::baryctr_coord, 0);

        SWE::initialize_data_parallel_post_receive(sim_units[su_id]->discretization.mesh, CommTypes::baryctr_coord);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->communicator.WaitAllSends(CommTypes::baryctr_coord, 0);
    }

    for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
        sim_units[su_id]->discretization.mesh.CallForEachElement(
            [&stepper](auto& elt) { elt.data.resize(stepper.GetNumStages() + 1); });
    }

#ifdef HAS_PETSC
    if (SWE::SedimentTransport::bed_update) {
#pragma omp barrier
#pragma omp master
        {
            std::set<uint> nodeIDs;
            for (uint su_id = 0; su_id < sim_units.size(); ++su_id) {
                sim_units[su_id]->discretization.mesh.CallForEachElement(
                    [&nodeIDs](auto& elt) { nodeIDs.insert(elt.GetNodeID().begin(), elt.GetNodeID().end()); });
            }
            global_data.local_bath_nodeIDs = std::vector<int>(nodeIDs.begin(), nodeIDs.end());
            std::map<uint, uint> nodeIDmap;
            uint it = 0;
            for (auto nodeID : nodeIDs) {
                nodeIDmap[nodeID] = it++;
            }

            uint local_max_nodeID  = *std::max_element(nodeIDs.begin(), nodeIDs.end());
            uint global_max_nodeID = 0;
            MPI_Allreduce(&local_max_nodeID, &global_max_nodeID, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

            VecCreateMPI(MPI_COMM_WORLD, global_max_nodeID + 1, PETSC_DECIDE, &(global_data.global_bath_at_node));
            VecCreateSeq(MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), &(global_data.local_bath_at_node));
            global_data.bath_at_node = DynVector<double>(global_data.local_bath_nodeIDs.size());

            ISCreateGeneral(MPI_COMM_SELF,
                            global_data.local_bath_nodeIDs.size(),
                            global_data.local_bath_nodeIDs.data(),
                            PETSC_COPY_VALUES,
                            &(global_data.bath_from));
            ISCreateStride(MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), 0, 1, &(global_data.bath_to));
            VecScatterCreate(global_data.global_bath_at_node,
                             global_data.bath_from,
                             global_data.local_bath_at_node,
                             global_data.bath_to,
                             &(global_data.bath_scatter));

            VecCreateMPI(MPI_COMM_WORLD, global_max_nodeID + 1, PETSC_DECIDE, &(global_data.global_bath_node_mult));
            VecCreateSeq(MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), &(global_data.local_bath_node_mult));

            ISCreateGeneral(MPI_COMM_SELF,
                            global_data.local_bath_nodeIDs.size(),
                            global_data.local_bath_nodeIDs.data(),  // dangerous cast!
                            PETSC_COPY_VALUES,
                            &(global_data.bath_node_mult_from));
            ISCreateStride(
                MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), 0, 1, &(global_data.bath_node_mult_to));
            VecScatterCreate(global_data.global_bath_node_mult,
                             global_data.bath_node_mult_from,
                             global_data.local_bath_node_mult,
                             global_data.bath_node_mult_to,
                             &(global_data.bath_node_mult_scatter));

            VecSet(global_data.global_bath_node_mult, 0.0);
            DynVector<double> node_mult(global_data.local_bath_nodeIDs.size());
            set_constant(node_mult, 0.0);

            for (uint su_id = 0; su_id < sim_units.size(); ++su_id) {
                sim_units[su_id]->discretization.mesh.CallForEachElement([&node_mult, &nodeIDmap](auto& elt) {
                    auto& sl_state = elt.data.slope_limit_state;
                    sl_state.area  = elt.GetShape().GetArea();
                    for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
                        sl_state.local_nodeID[node] = nodeIDmap[elt.GetNodeID()[node]];
                        node_mult[sl_state.local_nodeID[node]] += sl_state.area;
                    }
                });
            }

            VecSetValues(global_data.global_bath_node_mult,
                         global_data.local_bath_nodeIDs.size(),
                         global_data.local_bath_nodeIDs.data(),
                         node_mult.data(),
                         ADD_VALUES);

            VecAssemblyBegin(global_data.global_bath_node_mult);
            VecAssemblyEnd(global_data.global_bath_node_mult);

            VecScatterBegin(global_data.bath_node_mult_scatter,
                            global_data.global_bath_node_mult,
                            global_data.local_bath_node_mult,
                            INSERT_VALUES,
                            SCATTER_FORWARD);
            VecScatterEnd(global_data.bath_node_mult_scatter,
                          global_data.global_bath_node_mult,
                          global_data.local_bath_node_mult,
                          INSERT_VALUES,
                          SCATTER_FORWARD);

            double* m_ptr;
            VecGetArray(global_data.local_bath_node_mult, &m_ptr);

            for (uint su_id = 0; su_id < sim_units.size(); ++su_id) {
                sim_units[su_id]->discretization.mesh.CallForEachElement([m_ptr](auto& elt) {
                    auto& sl_state = elt.data.slope_limit_state;
                    for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
                        sl_state.node_mult[node] = m_ptr[sl_state.local_nodeID[node]];
                    }
                });
            }

            VecRestoreArray(global_data.local_bath_node_mult, &m_ptr);
        }
#pragma omp barrier
    }
#endif
}
}
}

#endif