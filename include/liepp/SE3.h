#pragma once

#include "SO3.h"
#include "eigen3/Eigen/Dense"

template <typename _Scalar = double> class SE3 {
    using Vector3S = Eigen::Matrix<_Scalar, 3, 1>;
    using Matrix3S = Eigen::Matrix<_Scalar, 3, 3>;
    using Matrix4S = Eigen::Matrix<_Scalar, 4, 4>;
    using Vector6S = Eigen::Matrix<_Scalar, 6, 1>;
    using Matrix6S = Eigen::Matrix<_Scalar, 6, 6>;
    using SO3S = SO3<_Scalar>;

  public:
    static Matrix4S wedge(const Vector6S& u) {
        // u is in the format (omega, v)
        Matrix4S result;
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template block<3, 1>(0, 0));
        result.template block<3, 1>(0, 3) = u.template block<3, 1>(3, 0);
        result.template block<1, 4>(3, 0) = Eigen::Matrix<_Scalar, 1, 4>::Zero();
        return result;
    }
    static Vector6S vee(const Matrix4S& U) {
        // u is in the format (omega, v)
        Vector6S result;
        result.template block<3, 1>(0, 0) = SO3S::vex(U.template block<3, 3>(0, 0));
        result.template block<3, 1>(3, 0) = U.template block<3, 1>(0, 3);
        return result;
    }
    static Matrix6S adjoint(const Vector6S& u) {
        // u is in the format (omega, v)
        Matrix6S result = Matrix6S::Zero();
        result.template block<3, 3>(0, 0) = SO3S::skew(u.template segment<3>(0));
        result.template block<3, 3>(3, 3) = SO3S::skew(u.template segment<3>(0));
        result.template block<3, 3>(3, 0) = SO3S::skew(u.template segment<3>(3));
        return result;
    }
    static SE3 exp(const Vector6S& u) {
        Vector3S w = u.template block<3, 1>(0, 0);
        Vector3S v = u.template block<3, 1>(3, 0);

        _Scalar th = w.norm();
        _Scalar A, B, C;
        if (abs(th) >= 1e-12) {
            A = sin(th) / th;
            B = (1 - cos(th)) / pow(th, 2);
            C = (1 - A) / pow(th, 2);
        } else {
            A = 1.0;
            B = 1.0 / 2.0;
            C = 1.0 / 6.0;
        }

        Matrix3S wx = SO3S::skew(w);
        Matrix3S R = Matrix3S::Identity() + A * wx + B * wx * wx;
        Matrix3S V = Matrix3S::Identity() + B * wx + C * wx * wx;

        Matrix4S expMat = Matrix4S::Identity();
        expMat.template block<3, 3>(0, 0) = R;
        expMat.template block<3, 1>(0, 3) = V * v;

        return SE3(expMat);
    }
    static Vector6S log(const SE3& P) {
        SO3S R = P.R;
        Vector3S x = P.x;

        Matrix3S Omega = SO3S::skew(SO3S::log(R));

        _Scalar theta = SO3S::vex(Omega).norm();
        _Scalar coefficient = 1.0 / 12.0;
        if (abs(theta) > 1e-8) {
            coefficient = 1 / (theta * theta) * (1 - (theta * sin(theta)) / (2 * (1 - cos(theta))));
        }

        Matrix3S VInv = Matrix3S::Identity() - 0.5 * Omega + coefficient * Omega * Omega;
        Vector3S v = VInv * x;

        Matrix4S U = Matrix4S::Zero();
        U.template block<3, 3>(0, 0) = Omega;
        U.template block<3, 1>(0, 3) = v;

        return SE3::vee(U);
    }
    static SE3 Identity() { return SE3(SO3S::Identity(), Vector3S::Zero()); }

    SE3() = default;
    SE3(const SE3& other) {
        R = other.R;
        x = other.x;
    }
    SE3(const Matrix4S& mat) {
        R = SO3S(mat.template block<3, 3>(0, 0));
        x = mat.template block<3, 1>(0, 3);
    }
    SE3(const SO3S& R, const Vector3S& x) {
        this->R = R;
        this->x = x;
    }

    void setIdentity() {
        R.setIdentity();
        x.setZero();
    }
    Vector3S operator*(const Vector3S& point) const { return R * point + x; }
    SE3 operator*(const SE3& other) const { return SE3(R * other.R, x + R * other.x); }

    void invert() {
        x = -R.inverse() * x;
        R = R.inverse();
    }
    SE3 inverse() const { return SE3(R.inverse(), -(R.inverse() * x)); }
    Matrix6S Adjoint() const {
        Matrix6S AdMat;
        Matrix3S Rmat = R.asMatrix();
        AdMat.template block<3, 3>(0, 0) = Rmat;
        AdMat.template block<3, 3>(0, 3) = Matrix3S::Zero();
        AdMat.template block<3, 3>(3, 0) = SO3S::skew(x) * Rmat;
        AdMat.template block<3, 3>(3, 3) = Rmat;
        return AdMat;
    }

    // Set and get
    Matrix4S asMatrix() const {
        Matrix4S result;
        result.setIdentity();
        result.template block<3, 3>(0, 0) = R.asMatrix();
        result.template block<3, 1>(0, 3) = x;
        return result;
    }
    void fromMatrix(const Matrix4S& mat) {
        R.fromMatrix(mat.template block<3, 3>(0, 0));
        x = mat.template block<3, 1>(0, 3);
    }

    SO3S R;
    Vector3S x;
};

typedef SE3<double> SE3d;
typedef SE3<float> SE3f;
typedef SE3<Eigen::dcomplex> SE3cd;
typedef SE3<Eigen::scomplex> SE3cf;

typedef Eigen::Matrix<double, 6, 1> se3d;
typedef Eigen::Matrix<float, 6, 1> se3f;
typedef Eigen::Matrix<Eigen::dcomplex, 6, 1> se3cd;
typedef Eigen::Matrix<Eigen::scomplex, 6, 1> se3cf;