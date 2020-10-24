#ifndef EHDG_SWE_PROC_DBOUND_HPP
#define EHDG_SWE_PROC_DBOUND_HPP

namespace SWE {
namespace EHDG {
template <typename DistributedBoundaryType>
void Problem::global_distributed_boundary_kernel(const ProblemStepperType& stepper, DistributedBoundaryType& dbound) {
    auto& state    = dbound.data.state[stepper.GetStage()];
    auto& boundary = dbound.data.boundary[dbound.bound_id];

    boundary.q_at_gp = dbound.ComputeUgp(state.q);

    // Construct message to exterior state
    std::vector<double> message;

    message.reserve(1 + SWE::n_variables * dbound.data.get_ngp_boundary(dbound.bound_id));

    message.push_back(dbound.data.wet_dry_state.wet);

    for (uint gp = 0; gp < dbound.data.get_ngp_boundary(dbound.bound_id); ++gp) {
        for (uint var = 0; var < SWE::n_variables; ++var) {
            message.push_back(boundary.q_at_gp(var, gp));
        }
    }

    // Set message to send buffer
    dbound.boundary_condition.exchanger.SetToSendBuffer(CommTypes::bound_state, message);
}

template <typename DistributedBoundaryType>
void Problem::local_distributed_boundary_kernel(const ProblemStepperType& stepper, DistributedBoundaryType& dbound) {
    // Get message from exterior state
    std::vector<double> message;

    message.resize(1);  // just wet/dry state info

    dbound.boundary_condition.exchanger.GetFromReceiveBuffer(CommTypes::bound_state, message);

    bool wet_ex = (bool)message[0];

    if (dbound.data.wet_dry_state.wet || wet_ex) {
        auto& state    = dbound.data.state[stepper.GetStage()];
        auto& boundary = dbound.data.boundary[dbound.bound_id];

        // now compute contributions to the righthand side
        state.rhs -= dbound.IntegrationPhi(boundary.F_hat_at_gp);
    }
}
}
}

#endif
