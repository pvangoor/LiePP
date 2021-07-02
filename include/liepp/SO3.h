#pragma once

#include "eigen3/Eigen/Dense"

template <typename _Scalar = double> class SO3 {
    using Matrix3S = Eigen::Matrix<_Scalar, 3, 3>;
    using Vector3S = Eigen::Matrix<_Scalar, 3, 1>;
    using QuaternionS = Eigen::Quaternion<_Scalar>;

  public:
    static Matrix3S skew(const Vector3S& v) {
        return (Matrix3S() << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0).finished();
    }

    static Vector3S vex(const Matrix3S& M) { return (Vector3S() << M(2, 1), M(0, 2), M(1, 0)).finished(); }

    static SO3 exp(const Vector3S& w) {
        _Scalar theta = w.norm() / 2.0;
        QuaternionS result;
        result.w() = cos(theta);
        result.vec() = sin(theta) * w.normalized();
        return SO3(result);
    }

    static Vector3S log(const SO3& rotation) {
        Matrix3S R = rotation.asMatrix();
        _Scalar theta = acos((R.trace() - 1.0) / 2.0);
        _Scalar coefficient = (abs(theta) >= 1e-6) ? theta / (2.0 * sin(theta)) : 0.5;

        Matrix3S Omega = coefficient * (R - R.transpose());
        return vex(Omega);

        // _Scalar s = rotation.quaternion.w();
        // _Scalar c = rotation.quaternion.vec().norm();
        // _Scalar theta = atan2(s, c);
        // Vector3S result = abs(theta) * rotation.quaternion.vec() / c;
        // return result;
    }

    static SO3 SO3FromVectors(const Vector3S& origin, const Vector3S& dest) {
        SO3 result;
        result.quaternion.setFromTwoVectors(origin, dest);
        return result;
    }

    static SO3 Identity() { return SO3(QuaternionS::Identity()); }

    SO3() = default;
    SO3(const Matrix3S& mat) { quaternion = mat; }
    SO3(const QuaternionS& quat) { quaternion = quat; }
    SO3 inverse() const { return SO3(quaternion.inverse()); }

    void setIdentity() { quaternion = QuaternionS::Identity(); }
    Vector3S operator*(const Vector3S& point) const { return quaternion * point; }
    SO3 operator*(const SO3& other) const { return SO3(quaternion * other.quaternion); }
    Vector3S applyInverse(const Vector3S& point) const { return quaternion.inverse() * point; }

    void invert() { quaternion = quaternion.inverse(); }

    // Set and get
    Matrix3S asMatrix() const { return quaternion.toRotationMatrix(); }
    QuaternionS asQuaternion() const { return quaternion; }
    void fromMatrix(const Matrix3S& mat) { quaternion = mat; }
    void fromQuaternion(const QuaternionS& quat) { quaternion = quat; }

  private:
    QuaternionS quaternion;
};

typedef SO3<double> SO3d;
typedef SO3<float> SO3f;
typedef SO3<Eigen::dcomplex> SO3cd;
typedef SO3<Eigen::scomplex> SO3cf;