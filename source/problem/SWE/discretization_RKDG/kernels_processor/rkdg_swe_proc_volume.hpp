#ifndef RKDG_SWE_PROC_VOLUME_HPP
#define RKDG_SWE_PROC_VOLUME_HPP

#include "problem/SWE/problem_flux/swe_flux.hpp"

namespace SWE {
namespace RKDG {
template <typename ElementType>
void Problem::volume_kernel(const ProblemStepperType& stepper, ElementType& elt) {
    auto& state    = elt.data.state[stepper.GetStage()];
    auto& internal = elt.data.internal;

    if (SWE::SedimentTransport::bed_update) {
        row(internal.aux_at_gp, SWE::Auxiliaries::bath) = elt.ComputeUgp(row(state.aux, SWE::Auxiliaries::bath));
        row(internal.db_at_gp, GlobalCoord::x) =
            elt.ComputeDUgp(GlobalCoord::x, row(state.aux, SWE::Auxiliaries::bath));
        row(internal.db_at_gp, GlobalCoord::y) =
            elt.ComputeDUgp(GlobalCoord::y, row(state.aux, SWE::Auxiliaries::bath));

        if (SWE::PostProcessing::wetting_drying) {
            auto& wd_state                = elt.data.wet_dry_state;
            DynRowVector<double> bath_lin = elt.ProjectBasisToLinear(row(state.aux, SWE::Auxiliaries::bath));
            for (uint vrtx = 0; vrtx < elt.data.get_nvrtx(); ++vrtx) {
                wd_state.bath_at_vrtx[vrtx] = bath_lin[vrtx];
            }
            wd_state.bath_min = *std::min_element(wd_state.bath_at_vrtx.begin(), wd_state.bath_at_vrtx.end());
        }

        if (SWE::PostProcessing::slope_limiting || SWE::SedimentTransport::bed_slope_limiting) {
            auto& sl_state           = elt.data.slope_limit_state;
            sl_state.bath_lin        = elt.ProjectBasisToLinear(row(state.aux, SWE::Auxiliaries::bath));
            sl_state.bath_at_baryctr = elt.ComputeLinearUbaryctr(sl_state.bath_lin);
            sl_state.bath_at_vrtx    = sl_state.bath_lin;
            sl_state.bath_at_midpts  = elt.ComputeLinearUmidpts(sl_state.bath_lin);
        }
    }

    set_constant(state.rhs, 0.0);
    if (elt.data.wet_dry_state.wet) {
        internal.q_at_gp = elt.ComputeUgp(state.q);
        row(internal.aux_at_gp, SWE::Auxiliaries::h) =
            row(internal.q_at_gp, SWE::Variables::ze) + row(internal.aux_at_gp, SWE::Auxiliaries::bath);

        SWE::get_F(internal.q_at_gp, internal.aux_at_gp, internal.Fx_at_gp, internal.Fy_at_gp);
        state.rhs = elt.IntegrationDPhi(GlobalCoord::x, internal.Fx_at_gp) +
                    elt.IntegrationDPhi(GlobalCoord::y, internal.Fy_at_gp);
    }

    if (SWE::SedimentTransport::bed_update) {
        set_constant(state.b_rhs, 0.0);
        if (elt.data.wet_dry_state.wet) {
            for (uint gp = 0; gp < elt.data.get_ngp_internal(); ++gp) {
                column(internal.qb_at_gp, gp) = bed_flux(column(internal.q_at_gp, gp), column(internal.aux_at_gp, gp));
            }
            state.b_rhs = elt.IntegrationDPhi(GlobalCoord::x, row(internal.qb_at_gp, GlobalCoord::x)) +
                          elt.IntegrationDPhi(GlobalCoord::y, row(internal.qb_at_gp, GlobalCoord::y));
        }
    }
}
}
}

#endif
