#pragma once

#include <cstdint>
#include <concepts>
#include <typeinfo>
#include <cassert>
#include <utility>
#include <typeindex>
#include <functional>
#include <type_traits>

#include <boost/container_hash/hash.hpp>

namespace rg {

    using Hash64 = std::uint64_t;

    namespace detail {
        template<class>
        inline constexpr bool always_false_v = false;
    }

    constexpr Hash64 HashCombine(Hash64 seed, Hash64 value) noexcept {
        seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        return seed;
    }

    template<class T>
    concept HasAdlHashValue = requires(const T& value) {
        { HashValue(value) } -> std::convertible_to<Hash64>;
    };

    template<class T>
    concept HasStdHash = requires(const T& value) {
        { std::hash<std::remove_cvref_t<T>>{}(value) } -> std::convertible_to<std::size_t>;
    };

    template<class T>
    concept HasBoostHash = requires(const T& value) {
        { boost::hash<std::remove_cvref_t<T>>{}(value) } -> std::convertible_to<std::size_t>;
    };

    template<class T>
    Hash64 FieldHash(const T& value) {
        using ValueType = std::remove_cvref_t<T>;
        if constexpr (HasAdlHashValue<ValueType>) {
            return HashValue(value);
        }
        else if constexpr (std::is_enum_v<ValueType>) {
            return static_cast<Hash64>(static_cast<std::underlying_type_t<ValueType>>(value));
        }
        else if constexpr (HasStdHash<ValueType>) {
            return static_cast<Hash64>(std::hash<ValueType>{}(value));
        }
        else if constexpr (HasBoostHash<ValueType>) {
            return static_cast<Hash64>(boost::hash<ValueType>{}(value));
        }
        else {
            static_assert(detail::always_false_v<ValueType>, "Field type is not hashable for RG_DEFINE_PASS_INPUTS");
        }
    }

    template<class T, auto... Members>
    Hash64 AutoHashMembers(const T& value) {
        Hash64 seed = 0;
        ((seed = HashCombine(seed, FieldHash(value.*Members))), ...);
        return seed;
    }

    template<class T, auto... Members>
    bool AutoEqualMembers(const T& a, const T& b) {
        return (((a.*Members) == (b.*Members)) && ...);
    }

    // ADL hook: user defines `HashValue(const T&) -> Hash64` in T's namespace.
    template<class T>
    concept PassInputs =
        requires(const T & a, const T & b) {
            { HashValue(a) } -> std::convertible_to<Hash64>;
            { a == b } -> std::convertible_to<bool>;
    };

    struct NoInputs {
        friend bool operator==(const NoInputs&, const NoInputs&) = default;
    };

    inline Hash64 HashValue(const NoInputs&) noexcept { return 0; }

} // namespace rg

#define RG_DEFINE_PASS_INPUTS(Type, ...) \
    friend bool operator==(const Type& a, const Type& b) { return ::rg::AutoEqualMembers<Type, __VA_ARGS__>(a, b); } \
    friend ::rg::Hash64 HashValue(const Type& value) { return ::rg::AutoHashMembers<Type, __VA_ARGS__>(value); }

struct AnyPassInputs {
    const std::type_info* type = nullptr;

    using HashFn = rg::Hash64(*)(const void*) noexcept;
    using EqFn = bool(*)(const void*, const void*) noexcept;
    using DtorFn = void(*)(void*) noexcept;

    HashFn hashFn = nullptr;
    EqFn   eqFn = nullptr;
    DtorFn dtorFn = nullptr;

	// Simple version: heap allocate always. TODO: Will this ever be a performance issue?
    void* ptr = nullptr;

    template<rg::PassInputs T>
    void set(T value) {
        reset();
        ptr = new T(std::move(value));
        type = &typeid(T);

        hashFn = [](const void* p) noexcept -> rg::Hash64 {
            return HashValue(*static_cast<const T*>(p));
            };
        eqFn = [](const void* a, const void* b) noexcept -> bool {
            return *static_cast<const T*>(a) == *static_cast<const T*>(b);
            };
        dtorFn = [](void* p) noexcept { delete static_cast<T*>(p); };
    }

    void reset() noexcept {
        if (ptr) { dtorFn(ptr); }
        ptr = nullptr; type = nullptr;
        hashFn = nullptr; eqFn = nullptr; dtorFn = nullptr;
    }

    template<class T>
    const T& get() const {
        assert(type && *type == typeid(T));
        return *static_cast<const T*>(ptr);
    }

    rg::Hash64 hash() const noexcept { return hashFn(ptr); }
    bool equals(const AnyPassInputs& o) const noexcept {
        if (type != o.type) return false;
        return eqFn(ptr, o.ptr);
    }

    ~AnyPassInputs() { reset(); }
};

class RenderGraphPassBase {
public:
    rg::Hash64 CompileKey() const noexcept { return compileKey_; }
    bool ConsumeCompileDirty() noexcept { return std::exchange(compileDirty_, false); }

    template<rg::PassInputs T>
    void SetInputs(T in) {
        rg::Hash64 newKey = Mix(TypeHash<T>(), HashValue(in));

        // If types differ or keys differ, still confirm with == (collision safety).
        bool changed = (!inputs_.type) ||
            (*inputs_.type != typeid(T)) ||
            (newKey != compileKey_);

        if (!changed) {
            // same key, same type: still verify (rare collision defense)
            const T& cur = inputs_.get<T>();
            if (!(cur == in)) changed = true;
        }

        if (changed) {
            inputs_.set<T>(std::move(in));
            compileKey_ = newKey;
            compileDirty_ = true;
        }
    }

protected:
    template<class T> const T& Inputs() const { return inputs_.get<T>(); }

private:
    AnyPassInputs inputs_;
    rg::Hash64 compileKey_ = 0;
    bool compileDirty_ = true;

    template<class T> static rg::Hash64 TypeHash() noexcept {
        // Use a stable per-run type hash. If you want cross-run stability, supply your own ID.
        return rg::Hash64(std::type_index(typeid(T)).hash_code());
    }

    static rg::Hash64 Mix(rg::Hash64 a, rg::Hash64 b) noexcept {
        // any decent 64-bit mix (splitmix64/xor-fold/etc.)
        a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
        return a;
    }
};