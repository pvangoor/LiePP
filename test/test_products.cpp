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

    liepp::SO3d::VectorDS U_R = liepp::SO3d::VectorDS::Random();
    liepp::SL3d::VectorDS U_H = liepp::SL3d::VectorDS::Random();
    liepp::SE3d::VectorDS U_P = liepp::SE3d::VectorDS::Random();
    decltype(X1)::VectorDS U_X;
    U_X << U_R, U_H, U_P;

    liepp::SO3d::MatrixDS ad_R = liepp::SO3d::adjoint(U_R);
    liepp::SL3d::MatrixDS ad_H = liepp::SL3d::adjoint(U_H);
    liepp::SE3d::MatrixDS ad_P = liepp::SE3d::adjoint(U_P);
    decltype(X1)::MatrixDS ad_X = decltype(X1)::adjoint(U_X);
    testMatrixEquality<double, liepp::SO3d::CDim,liepp::SO3d::CDim>(ad_X.block<liepp::SO3d::CDim,liepp::SO3d::CDim>(0,0), ad_R);
    testMatrixEquality<double, liepp::SL3d::CDim,liepp::SL3d::CDim>(ad_X.block<liepp::SL3d::CDim,liepp::SL3d::CDim>(liepp::SO3d::CDim,liepp::SO3d::CDim), ad_H);
    testMatrixEquality<double, liepp::SE3d::CDim,liepp::SE3d::CDim>(ad_X.block<liepp::SE3d::CDim,liepp::SE3d::CDim>(liepp::SO3d::CDim+liepp::SL3d::CDim,liepp::SO3d::CDim+liepp::SL3d::CDim),ad_P);

    // Exponential and Logarithm
    const auto exp_U_R = decltype(R1)::exp(U_R);
    const auto exp_U_H = decltype(H1)::exp(U_H);
    const auto exp_U_P = decltype(P1)::exp(U_P);
    const auto exp_U_X = decltype(X1)::exp(U_X);

    testMatrixEquality(std::get<0>(exp_U_X.X).asMatrix(), exp_U_R.asMatrix());
    testMatrixEquality(std::get<1>(exp_U_X.X).asMatrix(), exp_U_H.asMatrix());
    testMatrixEquality(std::get<2>(exp_U_X.X).asMatrix(), exp_U_P.asMatrix());

    const auto log_exp_U_R = decltype(R1)::log(exp_U_R);
    const auto log_exp_U_H = decltype(H1)::log(exp_U_H);
    const auto log_exp_U_P = decltype(P1)::log(exp_U_P);
    const auto log_exp_U_X = decltype(X1)::log(exp_U_X);
    
    testMatrixEquality<double, liepp::SO3d::CDim, 1>(log_exp_U_X.segment<liepp::SO3d::CDim>(0), log_exp_U_R);
    testMatrixEquality<double, liepp::SL3d::CDim, 1>(log_exp_U_X.segment<liepp::SL3d::CDim>(liepp::SO3d::CDim), log_exp_U_H);
    testMatrixEquality<double, liepp::SE3d::CDim, 1>(log_exp_U_X.segment<liepp::SE3d::CDim>(liepp::SO3d::CDim+liepp::SL3d::CDim),log_exp_U_P);

} 