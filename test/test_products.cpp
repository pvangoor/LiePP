#include "liepp/ProductGroup.h"
#include "liepp/SO3.h"
#include "liepp/SLn.h"
#include "liepp/SE3.h"

#include "gtest/gtest.h"

#include "testing_utilities.h"

using SO3dxSO3d = liepp::ProductGroup<liepp::SO3d, liepp::SO3d>;

TEST(TestGroups, ProductGroupInstantiations) {
    // Test SO(3) x SO(3) product
    liepp::SO3d R1 = liepp::SO3d::Random();
    liepp::SL3d H1 = liepp::SL3d::Random();
    liepp::SE3d P1 = liepp::SE3d::Random();

    liepp::SO3d R2 = liepp::SO3d::Random();
    liepp::SL3d H2 = liepp::SL3d::Random();
    liepp::SE3d P2 = liepp::SE3d::Random();

    liepp::ProductGroup X1 = liepp::ProductGroup(R1, H1, P1);
    liepp::ProductGroup X2 = liepp::ProductGroup(R2, H2, P2);

    liepp::SO3d R3 = R1 * R2;
    liepp::SL3d H3 = H1 * H2;
    liepp::SE3d P3 = P1 * P2;
    liepp::ProductGroup X3 = X1 * X2;

    testMatrixEquality(std::get<0>(X3.X).asMatrix(), R3.asMatrix());
    testMatrixEquality(std::get<1>(X3.X).asMatrix(), H3.asMatrix());
    testMatrixEquality(std::get<2>(X3.X).asMatrix(), P3.asMatrix());

    decltype(X1)::MatrixDS Ad1 = X1.Adjoint();
    testMatrixEquality<double, liepp::SO3d::CDim,liepp::SO3d::CDim>(Ad1.block<liepp::SO3d::CDim,liepp::SO3d::CDim>(0,0), R1.Adjoint());
    testMatrixEquality<double, liepp::SL3d::CDim,liepp::SL3d::CDim>(Ad1.block<liepp::SL3d::CDim,liepp::SL3d::CDim>(liepp::SO3d::CDim,liepp::SO3d::CDim), H1.Adjoint());
    testMatrixEquality<double, liepp::SE3d::CDim,liepp::SE3d::CDim>(Ad1.block<liepp::SE3d::CDim,liepp::SE3d::CDim>(liepp::SO3d::CDim+liepp::SL3d::CDim,liepp::SO3d::CDim+liepp::SL3d::CDim),P1.Adjoint());
}