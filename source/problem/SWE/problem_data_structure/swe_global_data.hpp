#ifndef SWE_GLOBAL_DATA_HPP
#define SWE_GLOBAL_DATA_HPP

#ifdef HAS_PETSC
#include <petscksp.h>
#endif

namespace SWE {
struct GlobalData {
#ifndef HAS_PETSC
    SparseMatrix<double> delta_hat_global;
    DynVector<double> rhs_global;
#endif

#ifdef HAS_PETSC
    Mat delta_hat_global;
    Vec rhs_global;
    KSP ksp;
    PC pc;

    IS from, to;
    VecScatter scatter;
    Vec sol;

    DynVector<double> solution;
    bool converged = false;

    Vec global_bath_at_node;
    Vec global_bath_node_mult;
    Vec global_bath_mpi_rank;
    DynVector<double> bath_at_node;
    std::vector<int> local_bath_nodeIDs;
    DynVector<double> dbound_bath_at_node;
    std::vector<int> dbound_local_bath_nodeIDs;
    std::vector<int> dbound_loc_to_glob_nodeIDs;

    IS bath_from, bath_to;
    VecScatter bath_scatter;
    Vec local_bath_at_node;

    IS bath_node_mult_from, bath_node_mult_to;
    VecScatter bath_node_mult_scatter;
    Vec local_bath_node_mult;

    IS bath_mpi_rank_from, bath_mpi_rank_to;
    VecScatter bath_mpi_rank_scatter;
    Vec local_bath_mpi_rank;

    PetscLogStage continuity_limiter_stage;

    void destroy() {
        MatDestroy(&delta_hat_global);
        VecDestroy(&rhs_global);
        KSPDestroy(&ksp);

        ISDestroy(&from);
        ISDestroy(&to);
        VecScatterDestroy(&scatter);
        VecDestroy(&sol);

        if (SWE::SedimentTransport::bed_update) {
            VecDestroy(&global_bath_at_node);
            VecDestroy(&global_bath_node_mult);
            VecDestroy(&global_bath_mpi_rank);

            ISDestroy(&bath_from);
            ISDestroy(&bath_to);
            VecScatterDestroy(&bath_scatter);
            VecDestroy(&local_bath_at_node);

            ISDestroy(&bath_node_mult_from);
            ISDestroy(&bath_node_mult_to);
            VecScatterDestroy(&bath_node_mult_scatter);
            VecDestroy(&local_bath_node_mult);

            ISDestroy(&bath_mpi_rank_from);
            ISDestroy(&bath_mpi_rank_to);
            VecScatterDestroy(&bath_mpi_rank_scatter);
            VecDestroy(&local_bath_mpi_rank);
        }
    }
#endif
};
}

#endif
