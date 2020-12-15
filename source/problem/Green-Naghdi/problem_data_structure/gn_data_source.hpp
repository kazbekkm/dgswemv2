#ifndef GN_DATA_SOURCE_HPP
#define GN_DATA_SOURCE_HPP

namespace GN {
struct Source : SWE::Source {
    Source() = default;
    Source(const uint nnode, const uint nbound, const uint nvrtx) : SWE::Source(nnode, nbound, nvrtx) {}

    bool dispersive_correction = true;
};
}

#endif