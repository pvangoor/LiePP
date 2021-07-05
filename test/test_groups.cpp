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

#include "liepp/SE3.h"
#include "liepp/SO3.h"
#include "liepp/SOT3.h"
#include "liepp/SEn3.h"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "gtest/gtest.h"

using namespace std;
using namespace Eigen;

void testMatrixEquality(const MatrixXd& M1, const MatrixXd& M2, const double abs_error = 1e-8) {
    ASSERT_EQ(M1.rows(), M2.rows());
    ASSERT_EQ(M1.cols(), M2.cols());
    for (int i=0; i<M1.rows(); ++i) {
        for (int j=0; j<M1.cols(); ++j) {
            EXPECT_NEAR(M1(i,j), M2(i,j), abs_error);
        }
    }
}

TEST(TestGroups, SO3FromVectors) {
    // Test generating an SO(3) matrix between two vectors
    for (int i = 0; i < 100; ++i) {
        Vector3d v = Vector3d::Random();
        Vector3d w = Vector3d::Random();

        v = v.normalized();
        w = w.normalized();

        Matrix3d R = SO3d::SO3FromVectors(v, w).asMatrix();
        Vector3d w2 = R * v;

        testMatrixEquality(w, w2);
        testMatrixEquality(R.transpose() * R, Matrix3d::Identity());
    }
}

template <typename T>
class MatrixGroupTest : public testing::Test {};

using testing::Types;
typedef Types<SO3d, SE3d, SOT3d, SE23d> MatrixGroups;

TYPED_TEST_SUITE(MatrixGroupTest, MatrixGroups);

TYPED_TEST(MatrixGroupTest, TestExpLog) {
    // Test the matrix group exponential and logarithm
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorAlgS v = TypeParam::VectorAlgS::Random();

        auto X1 = TypeParam::exp(v).asMatrix();
        decltype(X1) X2 = TypeParam::wedge(v).exp();

        testMatrixEquality(X1, X2);

        auto v11 = TypeParam::log(TypeParam(X1));
        auto v12 = TypeParam::vee(X1.log());
        auto v21 = TypeParam::log(TypeParam(X2));
        auto v22 = TypeParam::vee(X2.log());

        testMatrixEquality(v, v11);
        testMatrixEquality(v, v12);
        testMatrixEquality(v, v21);
        testMatrixEquality(v, v22);
    }
}

TYPED_TEST(MatrixGroupTest, TestWedgeVee) {
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorAlgS v = TypeParam::VectorAlgS::Random();
        typename TypeParam::MatrixAlgS vWedge = TypeParam::wedge(v);
        typename TypeParam::VectorAlgS vWedgeVee = TypeParam::vee(vWedge);
        testMatrixEquality(vWedgeVee, v);
    }
}

TYPED_TEST(MatrixGroupTest, TestAssociativity) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X1 = TypeParam::Random();
        TypeParam X2 = TypeParam::Random();
        TypeParam X3 = TypeParam::Random();

        TypeParam Z1 = (X1 * X2) * X3;
        TypeParam Z2 = X1 * (X2 * X3);
        
        testMatrixEquality(Z1.asMatrix(), Z2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestIdentity) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        TypeParam I = TypeParam::Identity();

        TypeParam X1 = X * I;
        TypeParam X2 = I * X;
        
        testMatrixEquality(X.asMatrix(), X1.asMatrix());
        testMatrixEquality(X.asMatrix(), X2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestInverse) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        TypeParam XInv = X.inverse();
        TypeParam I = TypeParam::Identity();

        TypeParam I1 = X * XInv;
        TypeParam I2 = XInv * X;
        
        testMatrixEquality(I.asMatrix(), I1.asMatrix());
        testMatrixEquality(I.asMatrix(), I2.asMatrix());
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixGroupAdjoint) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();
        typename TypeParam::VectorAlgS U = TypeParam::VectorAlgS::Random();
        
        typename TypeParam::MatrixAlgS Ad_XU1 = TypeParam::wedge(X.Adjoint() * U);
        typename TypeParam::MatrixAlgS Ad_XU2 = X.asMatrix() * TypeParam::wedge(U) * X.inverse().asMatrix();
        
        testMatrixEquality(Ad_XU1, Ad_XU2);
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixAlgebraAdjoint) {
    for (int i = 0; i < 100; ++i) {
        typename TypeParam::VectorAlgS V = TypeParam::VectorAlgS::Random();
        typename TypeParam::VectorAlgS U = TypeParam::VectorAlgS::Random();
        
        typename TypeParam::MatrixAlgS ad_VU1 = TypeParam::wedge(TypeParam::adjoint(V) * U);
        typename TypeParam::MatrixAlgS ad_VU2 = TypeParam::wedge(V) * TypeParam::wedge(U) - TypeParam::wedge(U) * TypeParam::wedge(V);
        
        testMatrixEquality(ad_VU1, ad_VU2);
    }
}


TYPED_TEST(MatrixGroupTest, TestMatrixProduct) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X1 = TypeParam::Random();
        TypeParam X2 = TypeParam::Random();

        typename TypeParam::MatrixAlgS Z1 = X1.asMatrix() * X2.asMatrix();
        typename TypeParam::MatrixAlgS Z2 = (X1 * X2).asMatrix();
        
        testMatrixEquality(Z1, Z2);
    }
}

TYPED_TEST(MatrixGroupTest, TestMatrixIdentity) {
    typename TypeParam::MatrixAlgS I1 = TypeParam::MatrixAlgS::Identity();
    typename TypeParam::MatrixAlgS I2 = TypeParam::Identity().asMatrix();
    
    testMatrixEquality(I1, I2);
}

TYPED_TEST(MatrixGroupTest, TestMatrixInverse) {
    for (int i = 0; i < 100; ++i) {
        TypeParam X = TypeParam::Random();

        typename TypeParam::MatrixAlgS XInv1 = X.inverse().asMatrix();
        typename TypeParam::MatrixAlgS XInv2 = X.asMatrix().inverse();
        
        testMatrixEquality(XInv1, XInv2);
    }
}