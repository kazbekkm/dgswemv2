#ifndef SWE_POST_SEDIMENT_HPP
#define SWE_POST_SEDIMENT_HPP

namespace SWE {
template <typename StepperType, typename ElementType>
void sediment_kernel(const StepperType& stepper, ElementType& elt) {
    const uint stage = stepper.GetStage();
    auto& state      = elt.data.state[stage];
    auto& source     = elt.data.source;

    source.hc_lin     = elt.ProjectBasisToLinear(row(state.q, SWE::Variables::hc));
    source.hc_at_vrtx = elt.ComputeLinearUvrtx(source.hc_lin);

    double hc_avg = std::accumulate(source.hc_at_vrtx.begin(), source.hc_at_vrtx.end(), 0.0) / elt.data.get_nvrtx();
    if (hc_avg <= 0) {
        state.aux(SWE::Auxiliaries::bath, 0) -= hc_avg;
        set_constant(source.hc_at_vrtx, 0.0);
    } else {
        double hc_min = *std::min_element(source.hc_at_vrtx.begin(), source.hc_at_vrtx.end());
        double theta  = std::min(1.0, std::abs(hc_avg / (hc_avg - hc_min)));
        for (uint vrtx = 0; vrtx < elt.data.get_nvrtx(); ++vrtx) {
            source.hc_at_vrtx[vrtx] = theta * source.hc_at_vrtx[vrtx] + (1.0 - theta) * hc_avg;
        }
    }
    row(state.q, SWE::Variables::hc) = elt.ProjectLinearToBasis(source.hc_at_vrtx);
}
}

#endif