/*
    This file is part of LiePP.

    LiePP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LiePP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LiePP.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "eigen3/Eigen/Dense"
#include "eigen3/unsupported/Eigen/MatrixFunctions"

template <int n, typename _Scalar = double> class SLn {
    // The special linear group of n dimensions.
    // n by n matrices with determinant 1.
  public:
    constexpr grpDim = n*n-1;
    using MatrixAlgS = Eigen::Matrix<_Scalar, grpDim, grpDim>;
    using VectorAlgS = Eigen::Matrix<_Scalar, grpDim, 1>;
    using VectorNS = Eigen::Matrix<_Scalar, n, 1>;

    static MatrixAlgS wedge(const VectorAlgS& u) {
        MatrixAlgS M;
        M(n-1,n-1) = 0.0;
        for (int i=0;i<n;++i) {
            for (int j=0;j<n;++j) {
                M(i,j) = u(n*i+j);

                if (i==j) {
                    M(n-1,n-1) -= u(n*i+j);
                }
            }
        }
        assert(abs(M.trace()) < 1e-8);
        return M;
    }

    static VectorAlgS vex(const MatrixAlgS& M) { 
        VectorAlgS u;
        for (int i=0;i<n;++i) {
            for (int j=0;j<n;++j) {
                if (n*i+j < grpDim)
                    u(n*i+j) = M(i,j);
            }
        }
        return u;
    }

    static MatrixAlgS adjoint(const VectorAlgS& u);

    static SLn exp(const VectorAlgS& u) {
        return SLn(wedge(u).exp(u));
    }

    static VectorAlgS log(const SLn& X) {
        return vee(X.asMatrix().log());
    }

    static SLn Identity() { return SLn(MatrixAlgS::Identity()); }
    static SLn Random() {
        MatrixAlgS M;
        _Scalar d;
        do {
            M.setRandom();
            d = M.determinant();
        } while (d == 0);
        return SLn(M / d);
    }

    SLn() = default;
    SLn(const MatrixAlgS& mat) { H = mat; }
    SLn inverse() const { return SLn(H.inverse()); }

    MatrixAlgS Adjoint() const;

    void setIdentity() { H = MatrixAlgS::Identity(); }
    VectorAlgS operator*(const VectorNS& point) const { return H * point; }
    SLn operator*(const SLn& other) const { return SLn(H * other.H); }
    VectorAlgS applyInverse(const VectorAlgS& point) const { return H.inverse() * point; }

    void invert() { H = H.inverse(); }

    // Set and get
    MatrixAlgS asMatrix() const { return H; }
    void fromMatrix(const MatrixAlgS& mat) { H = mat; }

  private:
    MatrixAlgS H;
};

typedef SLn<3, double> SL3d;
typedef SLn<3, float> SL3f;
typedef SLn<3, Eigen::dcomplex> SL3cd;
typedef SLn<3, Eigen::scomplex> SL3cf;