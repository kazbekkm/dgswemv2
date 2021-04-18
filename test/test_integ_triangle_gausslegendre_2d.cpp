#include "integration/integrations_2D.hpp"
#include "utilities/almost_equal.hpp"

int main() {
    using Utilities::almost_equal;

    bool any_error = false;
    Integration::Triangle_GaussLegendre_2D tri_gl_2d;
    std::pair<DynVector<double>, AlignedVector<Point<2>>> rule;

    for (uint p = 1; p < 65; ++p) {
        double exact_integration =
            1 / ((double)p + 1) *
            ((1 - pow(-1.0, p)) / ((double)p + 2) + 2 * pow(-1.0, p));  // S(x^p)dxdy over triangle

        rule = tri_gl_2d.GetRule(p);

        uint num_gp = tri_gl_2d.GetNumGP(p);

        double num_integration = 0;
        for (uint gp = 0; gp < rule.first.size(); ++gp) {
            num_integration += pow(rule.second[gp][GlobalCoord::x], p) * rule.first[gp];
        }

        if (!almost_equal(num_integration, exact_integration, 1.e+03)) {
            any_error = true;
            std::cerr << "Error found in Triangle Gauss-Legendre 2D at " << std::to_string(p)
                      << " - integration true value: " << exact_integration
                      << ", integration computed value: " << num_integration << std::endl;
        }

        if (num_gp != rule.first.size()) {
            any_error = true;
            std::cerr << "Error found in Triangle Gauss-Legendre 2D at " << std::to_string(p) << " gp_vector has size "
                      << rule.first.size() << " and return num_gp value " << num_gp << std::endl;
        }
    }

    if (any_error) {
        return 1;
    }

    return 0;
}