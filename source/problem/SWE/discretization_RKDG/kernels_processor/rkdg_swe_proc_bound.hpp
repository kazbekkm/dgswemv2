#ifndef RKDG_SWE_PROC_BOUND_HPP
#define RKDG_SWE_PROC_BOUND_HPP

namespace SWE {
namespace RKDG {
template <typename BoundaryType>
void Problem::boundary_kernel(const ProblemStepperType& stepper, BoundaryType& bound) {
    auto& state    = bound.data.state[stepper.GetStage()];
    auto& boundary = bound.data.boundary[bound.bound_id];

    if (SWE::SedimentTransport::bed_update) {
        row(boundary.aux_at_gp, SWE::Auxiliaries::bath) = bound.ComputeUgp(row(state.aux, SWE::Auxiliaries::bath));
    }

    if (bound.data.wet_dry_state.wet) {
        boundary.q_at_gp = bound.ComputeUgp(state.q);
        row(boundary.aux_at_gp, SWE::Auxiliaries::h) =
            row(boundary.q_at_gp, SWE::Variables::ze) + row(boundary.aux_at_gp, SWE::Auxiliaries::bath);

        bound.boundary_condition.ComputeFlux(stepper, bound);
        state.rhs -= bound.IntegrationPhi(boundary.F_hat_at_gp);
    }

    if (SWE::SedimentTransport::bed_update) {
        if (bound.data.wet_dry_state.wet) {
            bound.boundary_condition.ComputeBedFlux(stepper, bound);
            state.b_rhs -= bound.IntegrationPhi(boundary.qb_hat_at_gp);
        }
    }
}
}
}

#endif
