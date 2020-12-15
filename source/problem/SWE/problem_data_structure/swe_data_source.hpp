#ifndef SWE_DATA_SOURCE_HPP
#define SWE_DATA_SOURCE_HPP

namespace SWE {
struct Source {
    Source() = default;
    Source(const uint nnode, const uint nbound, const uint nvrtx)
        : parsed_meteo_data(nnode),
          tau_s(nnode),
          p_atm(nnode),
          tide_pot(nnode),
          manning_n(nnode),
          wet_neigh(nbound, false),
          hc_lin(nvrtx),
          hc_at_vrtx(nvrtx) {}

    double coriolis_f = 0.0;

    bool manning          = false;
    double g_manning_n_sq = 0.0;

    std::vector<std::vector<double>*> parsed_meteo_data;

    AlignedVector<StatVector<double, SWE::n_dimensions>> tau_s;
    std::vector<double> p_atm;

    std::vector<double> tide_pot;
    std::vector<double> manning_n;

    std::vector<bool> wet_neigh;
    StatVector<double, SWE::n_variables> q_avg;
    StatVector<double, SWE::n_auxiliaries> aux_avg;
    DynRowVector<double> hc_lin;
    DynRowVector<double> hc_at_vrtx;

    double total_entrainment = 0.0;
};
}

#endif