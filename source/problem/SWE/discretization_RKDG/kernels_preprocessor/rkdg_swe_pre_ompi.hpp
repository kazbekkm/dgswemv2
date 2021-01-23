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

            VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, global_max_nodeID + 1, &(global_data.global_bath_node_mult));
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

            VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, global_max_nodeID + 1, &(global_data.global_bath_mpi_rank));
            VecCreateSeq(MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), &(global_data.local_bath_mpi_rank));

            ISCreateGeneral(MPI_COMM_SELF,
                            global_data.local_bath_nodeIDs.size(),
                            global_data.local_bath_nodeIDs.data(),  // dangerous cast!
                            PETSC_COPY_VALUES,
                            &(global_data.bath_mpi_rank_from));
            ISCreateStride(MPI_COMM_SELF, global_data.local_bath_nodeIDs.size(), 0, 1, &(global_data.bath_mpi_rank_to));
            VecScatterCreate(global_data.global_bath_mpi_rank,
                             global_data.bath_mpi_rank_from,
                             global_data.local_bath_mpi_rank,
                             global_data.bath_mpi_rank_to,
                             &(global_data.bath_mpi_rank_scatter));

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

            int locality_id;
            MPI_Comm_rank(MPI_COMM_WORLD, &locality_id);

            VecSet(global_data.global_bath_mpi_rank, 0.0);
            DynVector<double> mpi_rank(global_data.local_bath_nodeIDs.size());
            set_constant(mpi_rank, locality_id);

            VecSetValues(global_data.global_bath_mpi_rank,
                         global_data.local_bath_nodeIDs.size(),
                         global_data.local_bath_nodeIDs.data(),
                         mpi_rank.data(),
                         INSERT_VALUES);

            VecAssemblyBegin(global_data.global_bath_mpi_rank);
            VecAssemblyEnd(global_data.global_bath_mpi_rank);

            VecScatterBegin(global_data.bath_mpi_rank_scatter,
                            global_data.global_bath_mpi_rank,
                            global_data.local_bath_mpi_rank,
                            INSERT_VALUES,
                            SCATTER_FORWARD);
            VecScatterEnd(global_data.bath_mpi_rank_scatter,
                          global_data.global_bath_mpi_rank,
                          global_data.local_bath_mpi_rank,
                          INSERT_VALUES,
                          SCATTER_FORWARD);

            double* r_ptr;
            VecGetArray(global_data.local_bath_mpi_rank, &r_ptr);

            std::set<uint> dboundNodeIDs;
            for (uint su_id = 0; su_id < sim_units.size(); ++su_id) {
                sim_units[su_id]->discretization.mesh.CallForEachDistributedBoundary([&dboundNodeIDs](auto& dbound) {
                    for (const auto& nodeID : dbound.GetNodeID()) {
                        dboundNodeIDs.erase(nodeID);
                        dboundNodeIDs.insert(nodeID);
                    }
                });
            }

            uint nodes_to_number = 0;
            for (const auto& node : dboundNodeIDs) {
                if (r_ptr[nodeIDmap[node]] == locality_id)
                    ++nodes_to_number;
            }

            int n_localities;
            MPI_Comm_size(MPI_COMM_WORLD, &n_localities);

            std::vector<uint> global_nodes_to_number;
            if (locality_id == 0) {
                global_nodes_to_number.resize(n_localities);
            }
            MPI_Gather(
                &nodes_to_number, 1, MPI_UINT32_T, &global_nodes_to_number.front(), 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

            uint n_global_nodes_to_number = 0;
            if (locality_id == 0) {
                n_global_nodes_to_number =
                    std::accumulate(global_nodes_to_number.begin(), global_nodes_to_number.end(), 0);
            }
            MPI_Bcast(&n_global_nodes_to_number, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

            if (locality_id == 0) {
                std::rotate(
                    global_nodes_to_number.begin(), global_nodes_to_number.end() - 1, global_nodes_to_number.end());
                global_nodes_to_number.front() = 0;
                for (int locality_id = 1; locality_id < n_localities; ++locality_id) {
                    global_nodes_to_number[locality_id] += global_nodes_to_number[locality_id - 1];
                }
            }
            MPI_Scatter(
                &global_nodes_to_number.front(), 1, MPI_UINT32_T, &nodes_to_number, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

            VecSet(global_data.global_bath_mpi_rank, 0.0);
            set_constant(mpi_rank, 0.0);
            for (const auto& node : dboundNodeIDs) {
                if (r_ptr[nodeIDmap[node]] == locality_id)
                    mpi_rank[nodeIDmap[node]] = nodes_to_number++;
            }

            VecRestoreArray(global_data.local_bath_mpi_rank, &r_ptr);

            VecSetValues(global_data.global_bath_mpi_rank,
                         global_data.local_bath_nodeIDs.size(),
                         global_data.local_bath_nodeIDs.data(),
                         mpi_rank.data(),
                         ADD_VALUES);

            VecAssemblyBegin(global_data.global_bath_mpi_rank);
            VecAssemblyEnd(global_data.global_bath_mpi_rank);

            VecScatterBegin(global_data.bath_mpi_rank_scatter,
                            global_data.global_bath_mpi_rank,
                            global_data.local_bath_mpi_rank,
                            INSERT_VALUES,
                            SCATTER_FORWARD);
            VecScatterEnd(global_data.bath_mpi_rank_scatter,
                          global_data.global_bath_mpi_rank,
                          global_data.local_bath_mpi_rank,
                          INSERT_VALUES,
                          SCATTER_FORWARD);

            VecGetArray(global_data.local_bath_mpi_rank, &r_ptr);

            std::set<uint> dbound_local_bath_nodeIDs;
            for (const auto& node : dboundNodeIDs) {
                dbound_local_bath_nodeIDs.insert(nodeIDmap[node]);
            }
            global_data.dbound_local_bath_nodeIDs =
                std::vector<int>(dbound_local_bath_nodeIDs.begin(), dbound_local_bath_nodeIDs.end());

            for (const auto& node : global_data.dbound_local_bath_nodeIDs) {
                global_data.dbound_loc_to_glob_nodeIDs.push_back(r_ptr[node]);
            }

            VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, n_global_nodes_to_number, &(global_data.global_bath_at_node));
            VecCreateSeq(
                MPI_COMM_SELF, global_data.dbound_local_bath_nodeIDs.size(), &(global_data.local_bath_at_node));
            global_data.bath_at_node        = DynVector<double>(global_data.local_bath_nodeIDs.size());
            global_data.dbound_bath_at_node = DynVector<double>(global_data.dbound_local_bath_nodeIDs.size());

            ISCreateGeneral(MPI_COMM_SELF,
                            global_data.dbound_loc_to_glob_nodeIDs.size(),
                            global_data.dbound_loc_to_glob_nodeIDs.data(),
                            PETSC_COPY_VALUES,
                            &(global_data.bath_from));
            ISCreateStride(MPI_COMM_SELF, global_data.dbound_loc_to_glob_nodeIDs.size(), 0, 1, &(global_data.bath_to));
            VecScatterCreate(global_data.global_bath_at_node,
                             global_data.bath_from,
                             global_data.local_bath_at_node,
                             global_data.bath_to,
                             &(global_data.bath_scatter));

            VecRestoreArray(global_data.local_bath_mpi_rank, &r_ptr);

            PetscLogStageRegister("ContinuityLimiter", &global_data.continuity_limiter_stage);
        }
#pragma omp barrier
    }
#endif
}
}
}

#endif
