#pragma once

#include "LieGroup.h"

namespace liepp {

template <typename Grp> class TangentGroup {
  protected:
    static_assert(isLieGroup<G>);
    using Alg = Grp::VectorDS;

  public:
    using Scalar = Grp::Scalar;
    constexpr static int CDim = 2 * Grp::CDim;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;

    TangentGroup() = default;
    TangentGroup(const Grp& X_G, const Alg& X_g) : X_G(X_G), X_g(X_g){};

    static TangentGroup Identity() { return TangentGroup(Grp::Identity(), Alg::Zero()); }

    static TangentGroup Random() { return TangentGroup(Grp::Random(), Alg::Random()); }

    TangentGroup inverse() const { return TangentGroup(X_G.inverse(), -Grp::Adjoint(X_G.inverse()) * X_g); }

    static MatrixDS Adjoint(const TangentGroup& X) {
        MatrixDS Ad = MatrixDS::Identity();
        const auto& Ad_G = Grp::Adjoint(X.X_G);
        Ad.block<Grp::CDim, Grp::CDim>(0,0) = Ad_G;
        Ad.block<Grp::CDim, Grp::CDim>(Grp::CDim,0) = Grp::adjoint(X.X_g) * Ad_G;
        Ad.block<Grp::CDim, Grp::CDim>(Grp::CDim,Grp::CDim) = Ad_G;
        return Ad;
    }

    static MatrixDS adjoint(const VectorDS& U) {
        const auto& U_G = U.segment<Grp::CDim>(0);
        const auto& U_g = U.segment<Grp::CDim>(Grp::CDim);
        MatrixDS ad = MatrixDS::Zero();
        ad.block<Grp::CDim, Grp::CDim>(0,0) = Grp::adjoint(U_G);
        ad.block<Grp::CDim, Grp::CDim>(Grp::CDim,0) = Grp::adjoint(U_g);
        ad.block<Grp::CDim, Grp::CDim>(Grp::CDim,Grp::CDim) = Grp::adjoint(U_G);
        return ad;
    }

    operator*(const TangentGroup& other) const {
        TangentGroup result;
        result.X_G = this->X_G * other.X_G;
        result.X_g = this->X_g + Grp::Adjoint(this->X_G) * other.X_g;
        return result;
    }

    Grp X_G;
    Alg X_g;
};
}; // namespace liepp
