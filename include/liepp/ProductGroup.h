#pragma once

#include "LieGroup.h"

#include <tuple>
#include <utility>
#include <array>

namespace liepp {

template <typename... Groups> class ProductGroup {
    protected:
    static_assert((isLieGroup<Groups> && ...));
    template<size_t N>
    using GroupTypeN = typename std::tuple_element<N, std::tuple<Groups...>>::type;

    template<size_t n>
    static constexpr int ZeroIndexDim() {
        if constexpr (n == 0) {
            return 0;
        } else {
            constexpr int CDimN = GroupTypeN<n-1>::CDim;
            return CDimN + ZeroIndexDim<n-1>();
        }
    }

    public:
    using Scalar = std::tuple_element<0, std::tuple<Groups...>>::type::Scalar;
    static_assert((std::is_same_v<Scalar, typename Groups::Scalar> && ...));
    constexpr static int CDim = (Groups::CDim + ...);


    using VectorDS = Eigen::Matrix<Scalar, CDim, 1>;
    using MatrixDS = Eigen::Matrix<Scalar, CDim, CDim>;
    static_assert(((Groups::CDim >= 0) && ...));

    std::tuple<Groups...> X;
    static ProductGroup Identity() {
        ProductGroup X;
        std::apply([](Groups&... args) {((args = Groups::Identity()), ...);}, X.X);
        return X;
    }
    static ProductGroup Random() {
        ProductGroup X;
        std::apply([](Groups&... args) {((args = Groups::Random()), ...);}, X.X);
        return X;
    }
    ProductGroup inverse() const {
        ProductGroup result;
        result.X = [this]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            return std::make_tuple(std::get<Idx>(this->X).inverse() ...); }(std::make_index_sequence<sizeof...(Groups)>());
        return result;
    }
    ProductGroup operator*(const ProductGroup& other) const {
        ProductGroup result;
        result.X = [this, other]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            return std::make_tuple(std::get<Idx>(this->X) * std::get<Idx>(other.X) ...); }(std::make_index_sequence<sizeof...(Groups)>());
        return result;
    }

    ProductGroup() = default;
    ProductGroup(const Groups&... x) {X = std::make_tuple(x...);};


    static ProductGroup exp(const VectorDS& U) {
        ProductGroup result;
        constexpr size_t n = sizeof...(Groups);
        result.X = [U]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            return std::make_tuple(GroupTypeN<Idx>::exp(U.template segment<GroupTypeN<Idx>::CDim>(ZeroIndexDim<Idx>())) ...);
            }(std::make_index_sequence<sizeof...(Groups)>());
        return result;
    }

    static VectorDS log(const ProductGroup& X) {
        VectorDS U;
        [&U, &X]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            ((U.template segment<GroupTypeN<Idx>::CDim>(ZeroIndexDim<Idx>())
                = GroupTypeN<Idx>::log(std::get<Idx>(X.X))), ...);
            }(std::make_index_sequence<sizeof...(Groups)>());
        return U;
    }

    MatrixDS Adjoint() const {
        MatrixDS Ad = MatrixDS::Zero();
        [&Ad, this]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            ((Ad.template block<GroupTypeN<Idx>::CDim, GroupTypeN<Idx>::CDim>(ZeroIndexDim<Idx>(), ZeroIndexDim<Idx>())
                = std::get<Idx>(this->X).Adjoint()), ...);
            }(std::make_index_sequence<sizeof...(Groups)>());
        return Ad;
    }

    static MatrixDS adjoint(const VectorDS& U) {
        MatrixDS ad = MatrixDS::Zero();
        [&ad, &U]<std::size_t ... Idx>(std::integer_sequence<size_t, Idx...>) {
            ((ad.template block<GroupTypeN<Idx>::CDim, GroupTypeN<Idx>::CDim>(ZeroIndexDim<Idx>(), ZeroIndexDim<Idx>())
                = GroupTypeN<Idx>::adjoint(U.template segment<GroupTypeN<Idx>::CDim>(ZeroIndexDim<Idx>()))), ...);
            }(std::make_index_sequence<sizeof...(Groups)>());
        return ad;
    }


        static_assert(isLieGroup<ProductGroup<Groups...>>);
    };

}; // namespace liepp