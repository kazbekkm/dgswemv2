#ifndef RECON_AVG_DBATH_HPP
#define RECON_AVG_DBATH_HPP

namespace GN {
namespace EHDG {
template <typename ProblemDiscretizationType, typename ProblemGlobalDataType>
void reconstruct_dbath(ProblemDiscretizationType& discretization,
                       ProblemGlobalDataType& global_data,
                       const uint stage) {
    set_constant(global_data.derivatives_at_node, 0.0);

    discretization.mesh.CallForEachElement([&global_data](auto& elt) {
        auto& derivative = elt.data.derivative;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            subvector(
                global_data.derivatives_at_node, derivative.local_nodeID[node] * GN::n_dimensions, GN::n_dimensions) +=
                derivative.dbath_at_baryctr;
        }
    });

    discretization.mesh.CallForEachElement([stage, &global_data](auto& elt) {
        auto& state      = elt.data.state[stage];
        auto& derivative = elt.data.derivative;
        auto& internal   = elt.data.internal;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            column(derivative.dbath_lin, node) = subvector(global_data.derivatives_at_node,
                                                           derivative.local_nodeID[node] * GN::n_dimensions,
                                                           GN::n_dimensions) /
                                                 derivative.node_mult[node];
        }
        state.dbath          = elt.ProjectLinearToBasis(derivative.dbath_lin);
        internal.dbath_at_gp = elt.ComputeUgp(state.dbath);
    });
}

template <typename ProblemDiscretizationType, typename ProblemGlobalDataType>
void reconstruct_ddbath(ProblemDiscretizationType& discretization,
                        ProblemGlobalDataType& global_data,
                        const uint stage) {
    set_constant(global_data.derivatives_at_node, 0.0);

    discretization.mesh.CallForEachElement([&global_data](auto& elt) {
        auto& derivative = elt.data.derivative;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            subvector(global_data.derivatives_at_node,
                      derivative.local_nodeID[node] * GN::n_ddbath_terms,
                      GN::n_ddbath_terms) += derivative.ddbath_at_baryctr;
        }
    });

    discretization.mesh.CallForEachElement([stage, &global_data](auto& elt) {
        auto& state      = elt.data.state[stage];
        auto& derivative = elt.data.derivative;
        auto& internal   = elt.data.internal;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            column(derivative.ddbath_lin, node) = subvector(global_data.derivatives_at_node,
                                                            derivative.local_nodeID[node] * GN::n_ddbath_terms,
                                                            GN::n_ddbath_terms) /
                                                  derivative.node_mult[node];
        }
        state.ddbath          = elt.ProjectLinearToBasis(derivative.ddbath_lin);
        internal.ddbath_at_gp = elt.ComputeUgp(state.ddbath);
    });
}

template <typename ProblemDiscretizationType, typename ProblemGlobalDataType>
void reconstruct_dddbath(ProblemDiscretizationType& discretization,
                         ProblemGlobalDataType& global_data,
                         const uint stage) {
    set_constant(global_data.derivatives_at_node, 0.0);

    discretization.mesh.CallForEachElement([&global_data](auto& elt) {
        auto& derivative = elt.data.derivative;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            subvector(global_data.derivatives_at_node,
                      derivative.local_nodeID[node] * GN::n_dddbath_terms,
                      GN::n_dddbath_terms) += derivative.dddbath_at_baryctr;
        }
    });

    discretization.mesh.CallForEachElement([stage, &global_data](auto& elt) {
        auto& state      = elt.data.state[stage];
        auto& derivative = elt.data.derivative;
        auto& internal   = elt.data.internal;
        for (uint node = 0; node < elt.GetNodeID().size(); ++node) {
            column(derivative.dddbath_lin, node) = subvector(global_data.derivatives_at_node,
                                                             derivative.local_nodeID[node] * GN::n_dddbath_terms,
                                                             GN::n_dddbath_terms) /
                                                   derivative.node_mult[node];
        }
        state.dddbath          = elt.ProjectLinearToBasis(derivative.dddbath_lin);
        internal.dddbath_at_gp = elt.ComputeUgp(state.dddbath);
    });
}
}
}

#endif
