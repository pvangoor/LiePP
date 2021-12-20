#pragma once

#include "LieGroup.h"

#include <tuple>
#include <utility>

namespace liepp {

template <typename... Groups> class ProductGroup {
    public:
    constexpr static int CDim = (Groups::CDim + ...);

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
    
    };

}; // namespace liepp