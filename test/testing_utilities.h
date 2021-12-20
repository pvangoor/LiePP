#pragma once

#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include "gtest/gtest.h"

#include <type_traits>
template <class T> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename _Scalar, std::enable_if_t<is_complex<_Scalar>::value, bool> = true>
void expectNear(const _Scalar& a, const _Scalar& b, const double abs_error = 1e-8) {
    EXPECT_NEAR(a.real(), b.real(), abs_error);
    EXPECT_NEAR(a.imag(), b.imag(), abs_error);
}

template <typename _Scalar, std::enable_if_t<!is_complex<_Scalar>::value, bool> = true>
void expectNear(const _Scalar& a, const _Scalar& b, const double abs_error = 1e-8) {
    EXPECT_NEAR(a, b, abs_error);
}

template <typename _Scalar, int rows, int cols>
void testMatrixEquality(
    const Eigen::Matrix<_Scalar, rows, cols>& M1, const Eigen::Matrix<_Scalar, rows, cols>& M2,
    const double abs_error = 1e-8) {
    for (int i = 0; i < M1.rows(); ++i) {
        for (int j = 0; j < M1.cols(); ++j) {
            expectNear(M1(i, j), M2(i, j), abs_error);
        }
    }
}