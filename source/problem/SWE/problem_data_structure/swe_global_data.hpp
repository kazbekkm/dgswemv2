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
    DynVector<double> bath_at_node;
    std::vector<int> local_bath_nodeIDs;

    IS bath_from, bath_to;
    VecScatter bath_scatter;
    Vec local_bath_at_node;

    IS bath_node_mult_from, bath_node_mult_to;
    VecScatter bath_node_mult_scatter;
    Vec local_bath_node_mult;

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

            ISDestroy(&bath_from);
            ISDestroy(&bath_to);
            VecScatterDestroy(&bath_scatter);
            VecDestroy(&local_bath_at_node);

            ISDestroy(&bath_node_mult_from);
            ISDestroy(&bath_node_mult_to);
            VecScatterDestroy(&bath_node_mult_scatter);
            VecDestroy(&local_bath_node_mult);
        }
    }
#endif
};
}

#endif
