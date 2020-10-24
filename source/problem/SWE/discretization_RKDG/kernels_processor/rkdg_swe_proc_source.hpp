#ifndef RKDG_SWE_PROC_SOURCE_HPP
#define RKDG_SWE_PROC_SOURCE_HPP

#include "problem/SWE/problem_source/swe_source.hpp"

namespace SWE {
namespace RKDG {
template <typename ElementType>
void Problem::source_kernel(const ProblemStepperType& stepper, ElementType& elt) {
    auto& state    = elt.data.state[stepper.GetStage()];
    auto& internal = elt.data.internal;

    if (elt.data.wet_dry_state.wet) {
        SWE::get_source(stepper, elt);
        state.rhs += elt.IntegrationPhi(internal.source_at_gp);
    }

    if (SWE::SedimentTransport::bed_update) {
        if (elt.data.wet_dry_state.wet) {
            state.b_rhs +=
                elt.IntegrationPhi(1.0 / (1.0 - Global::sat_sediment) * (internal.E_at_gp - internal.D_at_gp));
        }
    }
}
}
}

#endif
