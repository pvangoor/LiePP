#pragma once

#include "LieGroup.h"

#include <tuple>

namespace liepp {

template <typename G, typename H, auto action> class SemiDirectProductGroup {
  protected:
    static_assert((isLieGroup<G> && isLieGroup<H>));

  public:
    using Scalar = G::Scalar;
    static_assert(std::is_same_v<Scalar, H::Scalar>);

    constexpr static int CDim = G::CDim + H::CDim;
    using VectorDS = Eigen::Matrix<_Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<_Scalar, CDim, CDim>;

    SemiDirectProductGroup() = default;
    SemiDirectProductGroup(const G& X_G, const H& X_H) : X_G(X_G), X_H(X_H){};

    static SemiDirectProductGroup Identity() { return SemiDirectProductGroup(G::Identity(), H::Identity()); }

    static SemiDirectProductGroup Random() { return SemiDirectProductGroup(G::Random(), H::Random()); }

    SemiDirectProductGroup inverse() const {
        return SemiDirectProductGroup(X_G.inverse(), action(X_G.inverse(), X_H).inverse());
    }

    // The adjoints are hard due to the need to differentiate the action of G on H. See
    // https://math.stackexchange.com/questions/3378416/describing-the-lie-algebra-structure-of-a-semi-direct-product-of-lie-groups
    static MatrixDS Adjoint(const SemiDirectProductGroup& X) // TODO
        static MatrixDS adjoint(const VectorDS& U)           // TODO

        SemiDirectProductGroup
        operator*(const SemiDirectProductGroup& other) const {
        SemiDirectProductGroup result;
        result.X_G = this->X_G * other.X_G;
        result.X_H = this->X_H * action(this->X_G, other.X_H);
        return result;
    }

    G X_G;
    H X_H;
};
}; // namespace liepp
