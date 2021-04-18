#include "../integrations_2D.hpp"
#include "../integrations_1D.hpp"

namespace Integration {
std::pair<DynVector<double>, AlignedVector<Point<2>>> Triangle_GaussLegendre_2D::GetRule(const uint p) {
    if (p < 0 || p > 64) {
        printf("\n");
        printf("TRIANLE GAUSS LEGENDRE 2D - Fatal error!\n");
        printf("Illegal P = %d\n", p);
        exit(1);
    }

    // p+1 due to the linear det(J)
    const std::pair<DynVector<double>, AlignedVector<Point<1>>> rule_1D = GaussLegendre_1D{}.GetRule(p+1);
    const uint ngp_1D = GaussLegendre_1D{}.GetNumGP(p+1);

    std::pair<DynVector<double>, AlignedVector<Point<2>>> rule;
    rule.first.resize(ngp_1D * ngp_1D);
    rule.second.resize(ngp_1D * ngp_1D);

    for (uint gp_i = 0; gp_i < ngp_1D; ++gp_i) {
        for (uint gp_j = 0; gp_j < ngp_1D; ++gp_j) {
            double n1 = rule_1D.second[gp_i][LocalCoordLin::l1];
            double n2 =  rule_1D.second[gp_j][LocalCoordLin::l1];
            rule.first[gp_i * ngp_1D + gp_j]  = rule_1D.first[gp_i] * rule_1D.first[gp_j] * (1-n2)/2;
            rule.second[gp_i * ngp_1D + gp_j][LocalCoordTri::z1] = (1+n1)*(1-n2)/2 - 1;
            rule.second[gp_i * ngp_1D + gp_j][LocalCoordTri::z2] = n2;
        }
    }

    return rule;
}

uint Triangle_GaussLegendre_2D::GetNumGP(const uint p) {
    // p+1 due to the linear det(J)
    return std::pow(GaussLegendre_1D{}.GetNumGP(p+1), 2);
}
}