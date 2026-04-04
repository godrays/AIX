//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

namespace aix::metal::shaders
{

// Requires Metal Language Version 2_2 and higher.
const char* aixDeviceMetalShaders = R"(

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

#define BATCH_PROCESS_SIZE_PER_THREAD       1

struct MatrixSize
{
    size_t rows;
    size_t cols;
};

METAL_FUNC size_t layoutOffset(const constant size_t* layout)
{
    return layout[0];
}

METAL_FUNC size_t layoutRank(const constant size_t* layout)
{
    return layout[1];
}

METAL_FUNC const constant size_t* layoutShape(const constant size_t* layout)
{
    return layout + 2;
}

METAL_FUNC const constant size_t* layoutStrides(const constant size_t* layout)
{
    return layout + 2 + layoutRank(layout);
}

METAL_FUNC size_t physicalIndex(size_t flatIndex, const constant size_t* layout)
{
    size_t idx = flatIndex;
    size_t ofs = layoutOffset(layout);
    const constant size_t* shape = layoutShape(layout);
    const constant size_t* strides = layoutStrides(layout);
    size_t rank = layoutRank(layout);
    for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim)
    {
        size_t dimIndex = idx % shape[dim];
        idx /= shape[dim];
        ofs += dimIndex * strides[dim];
    }
    return ofs;
}

METAL_FUNC size_t physicalIndex2D(size_t row, size_t col, const constant size_t* layout)
{
    const constant size_t* strides = layoutStrides(layout);
    return layoutOffset(layout) + row * strides[0] + col * strides[1];
}

template <typename T>
struct NegateOp
{
    T operator()(T a) const { return -a; }
};

template <typename T>
struct SqrtOp
{
    T operator()(T a) const { return static_cast<T>(metal::sqrt(static_cast<float>(a))); }
};

template <typename T>
struct SinOp
{
    T operator()(T a) const { return static_cast<T>(metal::sin(static_cast<float>(a))); }
};

template <typename T>
struct CosOp
{
    T operator()(T a) const { return static_cast<T>(metal::cos(static_cast<float>(a))); }
};

template <typename T>
struct TanhOp
{
    T operator()(T a) const { return static_cast<T>(metal::tanh(static_cast<float>(a))); }
};

template <typename T>
struct LogOp
{
    T operator()(T a) const { return static_cast<T>(metal::log(static_cast<float>(a))); }
};

template <typename T>
struct ExpOp
{
    T operator()(T a) const { return static_cast<T>(metal::exp(static_cast<float>(a))); }
};

template <typename T>
struct AddOp
{
    T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct SubOp
{
    T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct MulOp
{
    T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct DivOp
{
    T operator()(T a, T b) const { return a / b; }
};

template <typename T>
struct PowOp
{
    T operator()(T a, T b) const { return static_cast<T>(metal::pow(static_cast<float>(a), static_cast<float>(b))); }
};

METAL_FUNC size_t translateLogicalIndex(size_t index,
                                        const device size_t* shape,
                                        const device size_t* newShape,
                                        size_t shapeSize,
                                        size_t newShapeSize)
{
    size_t originalIndex = 0;
    size_t targetStride = 1;
    size_t originalStride = 1;

    for (int64_t i = static_cast<int64_t>(newShapeSize) - 1, j = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimIndex = (index / targetStride) % newShape[i];
        if (j >= 0 && shape[j] == newShape[i])
        {
            originalIndex += dimIndex * originalStride;
            originalStride *= shape[--j + 1];
        }
        else if (j >= 0 && shape[j] == 1)
        {
            originalStride *= shape[--j + 1];
        }
        targetStride *= newShape[i];
    }

    return originalIndex;
}

METAL_FUNC size_t translateLogicalIndex(size_t index,
                                        const device size_t* shape,
                                        const constant size_t* newShape,
                                        size_t shapeSize,
                                        size_t newShapeSize)
{
    size_t originalIndex = 0;
    size_t targetStride = 1;
    size_t originalStride = 1;

    for (int64_t i = static_cast<int64_t>(newShapeSize) - 1, j = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimIndex = (index / targetStride) % newShape[i];
        if (j >= 0 && shape[j] == newShape[i])
        {
            originalIndex += dimIndex * originalStride;
            originalStride *= shape[--j + 1];
        }
        else if (j >= 0 && shape[j] == 1)
        {
            originalStride *= shape[--j + 1];
        }
        targetStride *= newShape[i];
    }

    return originalIndex;
}

METAL_FUNC size_t translateLogicalIndex(size_t index,
                                        const constant size_t* shape,
                                        const device size_t* newShape,
                                        size_t shapeSize,
                                        size_t newShapeSize)
{
    size_t originalIndex = 0;
    size_t targetStride = 1;
    size_t originalStride = 1;

    for (int64_t i = static_cast<int64_t>(newShapeSize) - 1, j = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimIndex = (index / targetStride) % newShape[i];
        if (j >= 0 && shape[j] == newShape[i])
        {
            originalIndex += dimIndex * originalStride;
            originalStride *= shape[--j + 1];
        }
        else if (j >= 0 && shape[j] == 1)
        {
            originalStride *= shape[--j + 1];
        }
        targetStride *= newShape[i];
    }

    return originalIndex;
}

METAL_FUNC size_t reducedSourceBaseOffset(size_t reducedIndex,
                                          const device size_t* shape,
                                          const device size_t* strides,
                                          size_t shapeSize,
                                          size_t dim)
{
    size_t srcBaseOffset = 0;
    size_t index = reducedIndex;

    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimSize = static_cast<size_t>(i) == dim ? 1 : shape[i];
        size_t coord = index % dimSize;
        index /= dimSize;
        srcBaseOffset += coord * strides[i];
    }

    return srcBaseOffset;
}

METAL_FUNC size_t reducedSourceBaseOffset(size_t reducedIndex,
                                          const constant size_t* shape,
                                          const constant size_t* strides,
                                          size_t shapeSize,
                                          size_t dim)
{
    size_t srcBaseOffset = 0;
    size_t index = reducedIndex;

    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimSize = static_cast<size_t>(i) == dim ? 1 : shape[i];
        size_t coord = index % dimSize;
        index /= dimSize;
        srcBaseOffset += coord * strides[i];
    }

    return srcBaseOffset;
}

METAL_FUNC size_t reducedLogicalBaseIndex(size_t reducedIndex,
                                          const constant size_t* shape,
                                          size_t shapeSize,
                                          size_t dim)
{
    size_t logicalIndex = 0;
    size_t logicalStride = 1;
    size_t index = reducedIndex;

    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i >= 0; --i)
    {
        size_t dimSize = static_cast<size_t>(i) == dim ? 1 : shape[i];
        size_t coord = index % dimSize;
        index /= dimSize;
        logicalIndex += coord * logicalStride;
        logicalStride *= shape[i];
    }

    return logicalIndex;
}

METAL_FUNC size_t logicalStrideForDimension(const constant size_t* shape, size_t shapeSize, size_t dim)
{
    size_t logicalStride = 1;
    for (int64_t i = static_cast<int64_t>(shapeSize) - 1; i > static_cast<int64_t>(dim); --i)
    {
        logicalStride *= shape[i];
    }
    return logicalStride;
}

// -----------------------------------------------------------------
// ATOMIC UTILS
// -----------------------------------------------------------------

#pragma METAL internals : enable
template <typename T>
constexpr constant bool is_metal_atomic = _disjunction<is_same<T, int>,
                                                        is_same<T, uint>,
                                                        is_same<T, ulong>,
                                                        is_same<T, float>>::value;

#pragma METAL internals : disable

template <typename T, typename = void>
struct aix_atomic
{
    atomic<uint> val;
};

template <typename T>
struct aix_atomic<T, enable_if_t<is_metal_atomic<T>>>
{
    atomic<T> val;
};

// -----------------------------------------------------------------
// NATIVE METAL ATOMICS
// -----------------------------------------------------------------

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC T aix_atomic_load_explicit(device aix_atomic<T>* object, uint offset)
{
    return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void
aix_atomic_store_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_store_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_and_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_and_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_or_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_or_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_min_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_min_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_max_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_max_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_add_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_mul_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    T expected = aix_atomic_load_explicit(object, offset);
    while (!aix_atomic_compare_exchange_weak_explicit(object, &expected, val * expected, offset)) { }
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC bool aix_atomic_compare_exchange_weak_explicit(device aix_atomic<T>* object,
                                                          thread T* expected,
                                                          T val,
                                                          uint offset)
{
    return atomic_compare_exchange_weak_explicit(&(object[offset].val), expected, val,
                                                 memory_order_relaxed, memory_order_relaxed);
}

// Specialization for float since it does not atomic_fetch_min_explicit
template <>
METAL_FUNC void aix_atomic_fetch_min_explicit<float>(device aix_atomic<float>* object, float val, uint offset)
{
    float expected = aix_atomic_load_explicit(object, offset);
    while (val < expected)
    {
        if (aix_atomic_compare_exchange_weak_explicit(object, &expected, val, offset)) { return; }
    }
}

// Specialization for float since it does not atomic_fetch_max_explicit
template <>
METAL_FUNC void aix_atomic_fetch_max_explicit<float>(device aix_atomic<float>* object, float val, uint offset)
{
    float expected = aix_atomic_load_explicit(object, offset);
    while (val > expected)
    {
        if (aix_atomic_compare_exchange_weak_explicit(object, &expected, val, offset)) { return; }
    }
}

// -----------------------------------------------------------------
// CUSTOM ATOMICS
// -----------------------------------------------------------------

namespace
{

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <typename T>
union uint_or_packed
{
    T val[packing_size<T>];
    uint bits;
};

template <typename T, typename Op>
struct aix_atomic_update_helper
{
    uint operator()(uint_or_packed<T> init, T update, uint elem_offset)
    {
        Op op;
        init.val[elem_offset] = op(update, init.val[elem_offset]);
        return init.bits;
    }
};

template <typename T, typename Op>
METAL_FUNC void aix_atomic_update_and_store(device aix_atomic<T>* object, T update, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;

    aix_atomic_update_helper<T, Op> helper;
    uint_or_packed<T> expected;
    expected.bits = atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);

    while (Op::condition(update, expected.val[elem_offset]) &&
           !aix_atomic_compare_exchange_weak_explicit(object, &(expected.bits),helper(expected, update, elem_offset),
                                                      pack_offset))
    { }
}

template <typename T>
struct __None
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        #pragma unused(b)
        return true;
    }

    T operator()(T a, T b)
    {
        #pragma unused(b)
        return a;
    }
};

template <typename T>
struct __Add
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        #pragma unused(b)
        return true;
    }

    T operator()(T a, T b)
    {
        return a + b;
    }
};

template <typename T>
struct __Mul
{
    static bool condition(T a, T b)
    {
        #pragma unused(a)
        return b != 0;
    }

    T operator()(T a, T b)
    {
        return a * b;
    }
};

template <typename T>
struct __Max
{
    static bool condition(T a, T b)
    {
        return a > b;
    }

    T operator()(T a, T b)
    {
        return max(a, b);
    }
};

template <typename T>
struct __Min
{
    static bool condition(T a, T b)
    {
        return a < b;
    }

    T operator()(T a, T b)
    {
        return min(a, b);
    }
};

} // namespace

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC T aix_atomic_load_explicit(device aix_atomic<T>* object, uint offset)
{
    uint pack_offset = offset / sizeof(T);
    uint elem_offset = offset % sizeof(T);
    uint_or_packed<T> packed_val;
    packed_val.bits = atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
    return packed_val.val[elem_offset];
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_store_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __None<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_and_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;
    uint_or_packed<T> identity;
    identity.bits = __UINT32_MAX__;
    identity.val[elem_offset] = val;

    atomic_fetch_and_explicit(&(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_or_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    uint pack_offset = offset / packing_size<T>;
    uint elem_offset = offset % packing_size<T>;
    uint_or_packed<T> identity;
    identity.bits = 0;
    identity.val[elem_offset] = val;

    atomic_fetch_or_explicit(&(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_min_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Min<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_max_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Max<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_add_explicit(device aix_atomic<T>* object, T val, uint offset)
{
      aix_atomic_update_and_store<T, __Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void aix_atomic_fetch_mul_explicit(device aix_atomic<T>* object, T val, uint offset)
{
    aix_atomic_update_and_store<T, __Mul<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC bool aix_atomic_compare_exchange_weak_explicit(device aix_atomic<T>* object,
                                                          thread uint* expected,
                                                          uint val,
                                                          uint offset)
{
    return atomic_compare_exchange_weak_explicit(&(object[offset].val), expected, val,
                                                 memory_order_relaxed, memory_order_relaxed);
}


// -----------------------------------------------------------------
// TEMPLATES
// -----------------------------------------------------------------


// Add - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void add(const device T* inA     [[buffer(0)]],
                    const device T* inB     [[buffer(1)]],
                    device T* result        [[buffer(2)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] + inB[index + i];
}


// Sub - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void sub(const device T* inA     [[buffer(0)]],
                    const device T* inB     [[buffer(1)]],
                    device T* result        [[buffer(2)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] - inB[index + i];
}


// Mul - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void mul(const device T* inA     [[buffer(0)]],
                    const device T* inB     [[buffer(1)]],
                    device T* result        [[buffer(2)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] * inB[index + i];
}


// Div - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void div(const device T* inA     [[buffer(0)]],
                    const device T* inB     [[buffer(1)]],
                    device T* result        [[buffer(2)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = inA[index + i] / inB[index + i];
}

template<typename T>
[[kernel]] void addStrided(const device T* inA         [[buffer(0)]],
                           const device T* inB         [[buffer(1)]],
                           device T* result            [[buffer(2)]],
                           const constant size_t* layoutA [[buffer(3)]],
                           const constant size_t* layoutB [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = inA[physicalIndex(index, layoutA)] + inB[physicalIndex(index, layoutB)];
}

template<typename T>
[[kernel]] void subStrided(const device T* inA         [[buffer(0)]],
                           const device T* inB         [[buffer(1)]],
                           device T* result            [[buffer(2)]],
                           const constant size_t* layoutA [[buffer(3)]],
                           const constant size_t* layoutB [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = inA[physicalIndex(index, layoutA)] - inB[physicalIndex(index, layoutB)];
}

template<typename T>
[[kernel]] void mulStrided(const device T* inA         [[buffer(0)]],
                           const device T* inB         [[buffer(1)]],
                           device T* result            [[buffer(2)]],
                           const constant size_t* layoutA [[buffer(3)]],
                           const constant size_t* layoutB [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = inA[physicalIndex(index, layoutA)] * inB[physicalIndex(index, layoutB)];
}

template<typename T>
[[kernel]] void divStrided(const device T* inA         [[buffer(0)]],
                           const device T* inB         [[buffer(1)]],
                           device T* result            [[buffer(2)]],
                           const constant size_t* layoutA [[buffer(3)]],
                           const constant size_t* layoutB [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = inA[physicalIndex(index, layoutA)] / inB[physicalIndex(index, layoutB)];
}


// Sqrt - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void sqrt(const device T* inA    [[buffer(0)]],
                     device T* result       [[buffer(1)]],
                     uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(sqrt(static_cast<float4>(inA[index + i])));
}


// Sin - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void sin(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(sin(static_cast<float4>(inA[index + i])));
}


// Cos - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void cos(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(cos(static_cast<float4>(inA[index + i])));
}


// Tanh - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void tanh(const device T* inA    [[buffer(0)]],
                     device T* result       [[buffer(1)]],
                     uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(tanh(static_cast<float4>(inA[index + i])));
}


// Log - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void log(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(log(static_cast<float4>(inA[index + i])));
}


// Exp - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void exp(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(exp(static_cast<float4>(inA[index + i])));
}

template<typename T>
[[kernel]] void unaryStrided(const device T* inA         [[buffer(0)]],
                             device T* result            [[buffer(1)]],
                             const constant size_t* layoutA [[buffer(2)]],
                             uint index [[thread_position_in_grid]])
{
    result[index] = -inA[physicalIndex(index, layoutA)];
}

template<typename T>
[[kernel]] void sqrtStrided(const device T* inA         [[buffer(0)]],
                            device T* result            [[buffer(1)]],
                            const constant size_t* layoutA [[buffer(2)]],
                            uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::sqrt(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}

template<typename T>
[[kernel]] void sinStrided(const device T* inA          [[buffer(0)]],
                           device T* result             [[buffer(1)]],
                           const constant size_t* layoutA [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::sin(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}

template<typename T>
[[kernel]] void cosStrided(const device T* inA          [[buffer(0)]],
                           device T* result             [[buffer(1)]],
                           const constant size_t* layoutA [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::cos(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}

template<typename T>
[[kernel]] void tanhStrided(const device T* inA         [[buffer(0)]],
                            device T* result            [[buffer(1)]],
                            const constant size_t* layoutA [[buffer(2)]],
                            uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::tanh(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}

template<typename T>
[[kernel]] void logStrided(const device T* inA          [[buffer(0)]],
                           device T* result             [[buffer(1)]],
                           const constant size_t* layoutA [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::log(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}

template<typename T>
[[kernel]] void expStrided(const device T* inA          [[buffer(0)]],
                           device T* result             [[buffer(1)]],
                           const constant size_t* layoutA [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::exp(static_cast<float>(inA[physicalIndex(index, layoutA)])));
}


// Pow - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void pow(const device T* inA     [[buffer(0)]],
                    const device T* expA    [[buffer(1)]],
                    device T* result        [[buffer(2)]],
                    uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = static_cast<T>(pow(static_cast<float4>(inA[index + i]), static_cast<float4>(expA[index + i])));
}

template<typename T>
[[kernel]] void powStrided(const device T* inA         [[buffer(0)]],
                           const device T* inB         [[buffer(1)]],
                           device T* result            [[buffer(2)]],
                           const constant size_t* layoutA [[buffer(3)]],
                           const constant size_t* layoutB [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    result[index] = static_cast<T>(metal::pow(static_cast<float>(inA[physicalIndex(index, layoutA)]),
                                              static_cast<float>(inB[physicalIndex(index, layoutB)])));
}


// Matrix Mul Tiled with boundary checks
// -----------------------------------------------------------------
template<typename T, uint BM, uint BN, uint BK, uint TM, uint TN>
[[kernel]] void matrixMulTiledBC(const device T* inA,
                                 const device T* inB,
                                 device T* result,
                                 constant MatrixSize& matASize,
                                 constant MatrixSize& matBSize,
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint2 lid  [[thread_position_in_threadgroup]])
{
    // Constants defining tile and thread sizes.
    // BM: Tile size in M dimension.
    // BN: Tile size in N dimension.
    // BK: Tile size in K dimension.
    // TM: Elements per thread along M.
    // TN: Elements per thread along N.

    // Thread indices within a block.
    constexpr uint bRowThread = BN / TN;            // Block row thread.
    constexpr uint bColThread = BM / TM;            // Block col thread.
    constexpr uint numThreads = bRowThread * bColThread;

    // Matrix dimensions.
    uint M = matASize.rows;
    uint K = matASize.cols;
    uint N = matBSize.cols;

    uint tx = (lid.x % bRowThread) * TN;
    uint ty = (lid.x / bRowThread) * TM;

    // Shared memory for tiles.
    threadgroup T A[BM * BK];
    threadgroup T B[BK * BN];

    // Indices for loading tiles into thread group memory.
    uint tileRowA = lid.x / BK;
    uint tileColA = lid.x % BK;
    constexpr uint tileStrideA = numThreads / BK;

    uint tileRowB = lid.x / BN;
    uint tileColB = lid.x % BN;
    constexpr uint tileStrideB = numThreads / BN;

    T tmp[TM][TN] = {{0}};      // Temporary accumulation buffer.

    // Main loop over tiles of K dimension.
    for (uint k=0; k<K; k+=BK)
    {
        // Load tiles from inA into shared memory with boundary checks.
        for (uint i=0; i<BM; i+=tileStrideA)
        {
            uint globalRow = tgid.y * BM + tileRowA + i;
            uint globalCol = k + tileColA;
            A[(tileRowA + i) * BK + tileColA] = globalRow < M && globalCol < K ? inA[globalRow * K + globalCol] : 0;
        }

        // Load tiles from inB into shared memory with boundary checks.
        for (uint i=0; i<BK; i+=tileStrideB)
        {
            uint globalRow = k + tileRowB + i;
            uint globalCol = tgid.x * BN + tileColB;
            B[(tileRowB + i) * BN + tileColB] = globalRow < K && globalCol < N ? inB[globalRow * N + globalCol] : 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial results.
        for (uint i=0; i<BK; i++)
        {
            for (uint j=0; j<TM; j++)
            {
                #pragma unroll(TN)
                for (uint l=0; l<TN; l++)
                {
                    tmp[j][l] += A[(ty + j) * BK + i] * B[tx + l + i * BN];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the final results with boundary checks.
    for (uint j=0; j<TM; j++)
    {
        #pragma unroll(TN)
        for (uint l=0; l<TN; l++)
        {
            uint globalRow = tgid.y * BM + ty + j;
            uint globalCol = tgid.x * BN + tx + l;
            if (globalRow < M && globalCol < N)
            {
                result[globalRow * N + globalCol] = tmp[j][l];
            }
        }
    }
}

template<typename T>
[[kernel]] void matrixMulStrided(const device T* inA,
                                 const device T* inB,
                                 device T* result,
                                 constant MatrixSize& matASize,
                                 constant MatrixSize& matBSize,
                                 const constant size_t* layoutA,
                                 const constant size_t* layoutB,
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= matBSize.cols || gid.y >= matASize.rows)
    {
        return;
    }

    // Fetch layout information ONCE per thread, outside the loop.
    const constant size_t* stridesA = layoutStrides(layoutA);
    const constant size_t* stridesB = layoutStrides(layoutB);

    // The amount to advance our pointers per k iteration.
    size_t strideA_k = stridesA[1];
    size_t strideB_k = stridesB[0];

    // Calculate starting physical positions for this thread.
    size_t startA = layoutOffset(layoutA) + gid.y * stridesA[0];
    size_t startB = layoutOffset(layoutB) + gid.x * stridesB[1];

    const device T* ptrA = inA + startA;
    const device T* ptrB = inB + startB;

    T sum = 0;
    size_t K = matASize.cols;
    size_t k = 0;

    // Manually unroll the loop for better Instruction Level Parallelism (ILP).
    for (; k + 4 <= K; k += 4)
    {
        sum += ptrA[0] * ptrB[0];
        ptrA += strideA_k; ptrB += strideB_k;

        sum += ptrA[0] * ptrB[0];
        ptrA += strideA_k; ptrB += strideB_k;

        sum += ptrA[0] * ptrB[0];
        ptrA += strideA_k; ptrB += strideB_k;

        sum += ptrA[0] * ptrB[0];
        ptrA += strideA_k; ptrB += strideB_k;
    }

    // Handle remaining elements.
    for (; k < K; ++k)
    {
        sum += ptrA[0] * ptrB[0];
        ptrA += strideA_k; ptrB += strideB_k;
    }

    result[gid.y * matBSize.cols + gid.x] = sum;
}

template<typename T, uint N>
[[kernel]] void matrixMulStridedTiled(const device T* inA,
                                      const device T* inB,
                                      device T* result,
                                      constant MatrixSize& matASize,
                                      constant MatrixSize& matBSize,
                                      const constant size_t* layoutA,
                                      const constant size_t* layoutB,
                                      uint2 gid [[thread_position_in_grid]],
                                      uint2 tid [[thread_position_in_threadgroup]])
{
    constexpr int TILE_SIZE = N;

    // Fast on chip memory to hold matrix blocks.
    threadgroup T tileA[TILE_SIZE][TILE_SIZE];
    threadgroup T tileB[TILE_SIZE][TILE_SIZE];

    size_t rowA = gid.y;
    size_t colB = gid.x;
    size_t colA = tid.x;
    size_t rowB = tid.y;

    const constant size_t* stridesA = layoutStrides(layoutA);
    const constant size_t* stridesB = layoutStrides(layoutB);

    // Prevent out of bounds pointer calculation (cap at max valid row/col).
    size_t safeRowA = rowA < matASize.rows ? rowA : 0;
    size_t safeColB = colB < matBSize.cols ? colB : 0;

    const device T* ptrA = inA + layoutOffset(layoutA) + safeRowA * stridesA[0] + colA * stridesA[1];
    const device T* ptrB = inB + layoutOffset(layoutB) + rowB * stridesB[0] + safeColB * stridesB[1];

    size_t strideA_tile = TILE_SIZE * stridesA[1];
    size_t strideB_tile = TILE_SIZE * stridesB[0];

    T sum = 0;
    size_t numTiles = (matASize.cols + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t t = 0; t < numTiles; ++t)
    {
        // Collaboratively load tiles into threadgroup memory.
        if (rowA < matASize.rows && colA < matASize.cols)
        {
            tileA[tid.y][tid.x] = *ptrA;
        }
        else
        {
            tileA[tid.y][tid.x] = 0;
        }

        if (rowB < matASize.cols && colB < matBSize.cols)
        {
            tileB[tid.y][tid.x] = *ptrB;
        }
        else
        {
            tileB[tid.y][tid.x] = 0;
        }

        // Wait for all threads to finish loading the tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the math entirely using fast on chip memory.
        #pragma unroll
        for (uint k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }

        // Wait for math to finish before overwriting the tile in the next loop.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Advance pointers to the next tile block.
        ptrA += strideA_tile;
        ptrB += strideB_tile;
        colA += TILE_SIZE;
        rowB += TILE_SIZE;
    }

    // Finally, write the output safely.
    if (rowA < matASize.rows && colB < matBSize.cols)
    {
        result[rowA * matBSize.cols + colB] = sum;
    }
}

// Matrix Mul Tiled
// -----------------------------------------------------------------
template<typename T, uint TSX, uint TSY>
[[kernel]] void matrixMulTiled(const device T* inA,
                               const device T* inB,
                               device T* result,
                               constant MatrixSize& matASize,
                               constant MatrixSize& matBSize,
                               uint2 tgid [[threadgroup_position_in_grid]],
                               uint2 lid  [[thread_position_in_threadgroup]])
{
    const uint N = matASize.cols;
    const uint K = matBSize.cols;

    auto xOffset = tgid.x * TSX;
    auto yOffset = tgid.y * TSY + lid.y * TSX;
    inA += yOffset * N;
    inB += xOffset;
    result += yOffset * K + xOffset;

    // Local tile buffers.
    simdgroup_matrix<T,8,8>  A[4];
    simdgroup_matrix<T,8,8>  B[4];
    simdgroup_matrix<T,8,8>  C[4][4] = { { simdgroup_matrix<T,8,8>(0) } };

    // Iterate over tiles.
    for (uint k=0; k<N; k+=8)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load tiles of A.
        #pragma unroll(4)
        for (uint i=0; i<4; ++i)
        {
            simdgroup_load(A[i], inA + k + i * 8 * N, N);
        }

        // Load tiles of B.
        #pragma unroll(4)
        for (uint i=0; i<4; ++i)
        {
            simdgroup_load(B[i], inB + i * 8 + k * K, K);
        }

        // Multiply and accumulate.
        #pragma unroll(4)
        for (int i=0; i<4; ++i)
        {
            #pragma unroll(4)
            for (int j=0; j<4; ++j)
            {
                simdgroup_multiply_accumulate(C[i][j], A[j], B[i], C[i][j]);
            }
        }
    }

    // Store the results.
    #pragma unroll(4)
    for (int i=0; i<4; ++i)
    {
        #pragma unroll(4)
        for (int j=0; j<4; ++j)
        {
            simdgroup_store(C[j][i], result + j * 8 + i * 8 * K, K);
        }
    }
}


// Transpose2D - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void transpose2D(const device T* mat          [[buffer(0)]],
                            device T* result             [[buffer(1)]],
                            constant MatrixSize& matSize [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]])
{
    uint ofs1 = gid.y * matSize.rows + gid.x;
    uint ofs2 = gid.x * matSize.cols + gid.y;
    result[ofs1] = mat[ofs2];
}


// Transpose2D Tiled
// -----------------------------------------------------------------
template<typename T, uint TILE_SIZE, uint BATCH_SIZE>
[[kernel]] void transpose2DTiled(const device T* mat          [[buffer(0)]],
                                 device T* result             [[buffer(1)]],
                                 constant MatrixSize& matSize [[buffer(2)]],
                                 uint2 gid  [[thread_position_in_grid]],
                                 uint2 tgid [[threadgroup_position_in_grid]],
                                 uint2 tid  [[thread_position_in_threadgroup]])
{
    threadgroup T tile[TILE_SIZE][TILE_SIZE + 1];

    const uint inputHeight = matSize.rows;    // M
    const uint inputWidth  = matSize.cols;    // N
    const uint outputWidth = inputHeight;     // M since transposing.

    uint x  = tgid.x * TILE_SIZE + tid.x;      // Global x position.
    uint y  = tgid.y * TILE_SIZE + tid.y;      // Global y position.

    #pragma unroll(TILE_SIZE/BATCH_SIZE)
    for (uint j=0; j<TILE_SIZE; j+=BATCH_SIZE)
    {
        tile[tid.y + j][tid.x] = mat[(y + j) * inputWidth + x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tX = tgid.y * TILE_SIZE + tid.x;     // Transposed x.
    uint tY = tgid.x * TILE_SIZE + tid.y;     // Transposed y.

    #pragma unroll(TILE_SIZE/BATCH_SIZE)
    for (uint j=0; j<TILE_SIZE; j+=BATCH_SIZE)
    {
        result[(tY + j) * outputWidth + tX] = tile[tid.x][tid.y + j];
    }
}


// Transpose - Naive Implementation
// -----------------------------------------------------------------
size_t flattenIndex(thread size_t* indices, size_t indicesSize, device const size_t* strides)
{
    size_t index = 0;
    for (size_t i = 0; i < indicesSize; ++i)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

void unflattenIndex(size_t index, device const size_t* strides, size_t stridesSize, thread size_t* outIndices)
{
    for (size_t i = 0; i < stridesSize; ++i)
    {
        outIndices[i] = index / strides[i];
        index %= strides[i];
    }
}

void swap(thread size_t& a, thread size_t& b)
{
    size_t temp = a;
    a = b;
    b = temp;
}

template<typename T>
[[kernel]] void transpose(const device T* data            [[buffer(0)]],
                          device T* result                [[buffer(1)]],
                          constant size_t& dim0           [[buffer(2)]],
                          constant size_t& dim1           [[buffer(3)]],
                          const constant size_t* srcLayout [[buffer(4)]],
                          constant size_t& size           [[buffer(5)]],
                          uint index [[thread_position_in_grid]])
{
    thread size_t coords[16];
    const constant size_t* srcShape = layoutShape(srcLayout);
    const constant size_t* srcStrides = layoutStrides(srcLayout);
    size_t rank = layoutRank(srcLayout);
    size_t remaining = index;

    for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i)
    {
        size_t dimSize = srcShape[i];
        if (static_cast<size_t>(i) == dim0)
            dimSize = srcShape[dim1];
        else if (static_cast<size_t>(i) == dim1)
            dimSize = srcShape[dim0];

        coords[i] = remaining % dimSize;
        remaining /= dimSize;
    }

    swap(coords[dim0], coords[dim1]);

    size_t srcIndex = layoutOffset(srcLayout);
    for (size_t i = 0; i < rank; ++i)
    {
        srcIndex += coords[i] * srcStrides[i];
    }

    result[index] = data[srcIndex];
}


// Copy - Naive Implementation
// -----------------------------------------------------------------
template<typename ST, typename DT>
[[kernel]] void copy(const device ST* src [[buffer(0)]],
                     device DT* dst       [[buffer(1)]],
                     uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        dst[index + i] = static_cast<DT>(src[index + i]);
}


// Unary - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void unary(const device T* inA   [[buffer(0)]],
                      device T* result      [[buffer(1)]],
                      uint index [[thread_position_in_grid]])
{
    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = -inA[index + i];
}


// Fill - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
[[kernel]] void fill(const device T* scalar [[buffer(0)]],
                     device T2* result      [[buffer(1)]],
                     uint index [[thread_position_in_grid]])
{
    T2 scalarVector = static_cast<T2>(scalar[0].xxxx);

    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = scalarVector;
}


// FillMin - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void fillMin(device T* result   [[buffer(0)]],
                        uint index [[thread_position_in_grid]])
{
    T minVal = static_cast<T>(numeric_limits<T>::lowest().xxxx);

    index *= BATCH_PROCESS_SIZE_PER_THREAD;
    #pragma unroll(BATCH_PROCESS_SIZE_PER_THREAD)
    for (size_t i=0; i<BATCH_PROCESS_SIZE_PER_THREAD; ++i)
        result[index + i] = minVal;
}


template<typename T>
struct ReductionSumIdentity
{
    T operator()() const { return static_cast<T>(0); }
};


template<typename T>
struct ReductionMaxIdentity
{
    T operator()() const { return numeric_limits<T>::lowest(); }
};


template<typename T, typename IdentityOp, typename ReduceOp>
METAL_FUNC void stagedScalarReduce(const device T* inA,
                                   device T* result,
                                   threadgroup T* sharedData,
                                   constant size_t& elementCount,
                                   constant size_t& useLayout,
                                   const constant size_t* layout,
                                   IdentityOp identityOp,
                                   ReduceOp reduceOp,
                                   uint li,
                                   uint tgi,
                                   uint threadsPerThreadgroup)
{
    const size_t baseOffset = static_cast<size_t>(tgi) * static_cast<size_t>(threadsPerThreadgroup);
    const size_t inputIndex = baseOffset + static_cast<size_t>(li);
    sharedData[li] = inputIndex < elementCount
        ? inA[useLayout != 0 ? physicalIndex(inputIndex, layout) : inputIndex]
        : identityOp();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    size_t size = threadsPerThreadgroup;
    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
        if (size % 2 == 1 && li == 0)
        {
            sharedData[0] = reduceOp(sharedData[0], sharedData[size - 1]);
        }
        size >>= 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < stride)
        {
            sharedData[li] = reduceOp(sharedData[li], sharedData[li + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (li == 0)
    {
        result[tgi] = sharedData[0];
    }
}


template<typename T>
struct ReductionSumOp
{
    T operator()(T a, T b) const { return a + b; }
};


template<typename T>
struct ReductionMaxOp
{
    T operator()(T a, T b) const { return a > b ? a : b; }
};


// Sum - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void sum(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    constant size_t& elementCount [[buffer(2)]],
                    constant size_t& useLayout [[buffer(3)]],
                    const constant size_t* layout [[buffer(4)]],
                    uint li [[thread_position_in_threadgroup]],
                    uint tgi [[threadgroup_position_in_grid]],
                    uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    threadgroup T sharedData[MAX_THREADS];
    stagedScalarReduce(inA, result, sharedData, elementCount, useLayout, layout,
                       ReductionSumIdentity<T>{}, ReductionSumOp<T>{},
                       li, tgi, threadsPerThreadgroup);
}


// Max - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
[[kernel]] void max(const device T* inA     [[buffer(0)]],
                    device T* result        [[buffer(1)]],
                    constant size_t& elementCount [[buffer(2)]],
                    constant size_t& useLayout [[buffer(3)]],
                    const constant size_t* layout [[buffer(4)]],
                    uint li  [[thread_position_in_threadgroup]],
                    uint tgi [[threadgroup_position_in_grid]],
                    uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    threadgroup T sharedData[MAX_THREADS];
    stagedScalarReduce(inA, result, sharedData, elementCount, useLayout, layout,
                       ReductionMaxIdentity<T>{}, ReductionMaxOp<T>{},
                       li, tgi, threadsPerThreadgroup);
}


// ArgMax - Naive Implementation
// -----------------------------------------------------------------
template<typename T>
METAL_FUNC bool argmaxIsBetter(T candidateValue, int candidateIndex, T currentValue, int currentIndex)
{
    return candidateValue > currentValue || (candidateValue == currentValue && candidateIndex < currentIndex);
}


template<typename T>
[[kernel]] void argmaxInit(const device T* src       [[buffer(0)]],
                           device T* values          [[buffer(1)]],
                           device int* indices       [[buffer(2)]],
                           constant size_t& useLayout [[buffer(3)]],
                           const constant size_t* layout [[buffer(4)]],
                           uint index [[thread_position_in_grid]])
{
    values[index] = src[useLayout != 0 ? physicalIndex(index, layout) : index];
    indices[index] = static_cast<int>(index);
}


template<typename T>
[[kernel]] void argmaxReduce(const device T* inValues     [[buffer(0)]],
                             const device int* inIndices  [[buffer(1)]],
                             device T* outValues          [[buffer(2)]],
                             device int* outIndices       [[buffer(3)]],
                             uint li  [[thread_position_in_threadgroup]],
                             uint tgi [[threadgroup_position_in_grid]],
                             uint threadsPerThreadgroup [[threads_per_threadgroup]])
{
    const size_t MAX_THREADS = 1024;
    const size_t baseOffset = static_cast<size_t>(tgi) * static_cast<size_t>(threadsPerThreadgroup);

    threadgroup T sharedValues[MAX_THREADS];
    threadgroup int sharedIndices[MAX_THREADS];
    sharedValues[li] = inValues[baseOffset + li];
    sharedIndices[li] = inIndices[baseOffset + li];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    size_t size = threadsPerThreadgroup;
    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
        if (size % 2 == 1 && li == 0 &&
            argmaxIsBetter(sharedValues[size - 1], sharedIndices[size - 1], sharedValues[0], sharedIndices[0]))
        {
            sharedValues[0] = sharedValues[size - 1];
            sharedIndices[0] = sharedIndices[size - 1];
        }
        size >>= 1;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (li < stride &&
            argmaxIsBetter(sharedValues[li + stride], sharedIndices[li + stride], sharedValues[li], sharedIndices[li]))
        {
            sharedValues[li] = sharedValues[li + stride];
            sharedIndices[li] = sharedIndices[li + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (li == 0)
    {
        outValues[tgi] = sharedValues[0];
        outIndices[tgi] = sharedIndices[0];
    }
}


template<typename T>
[[kernel]] void argmaxTo(const device T* src            [[buffer(0)]],
                         device int* dst                [[buffer(1)]],
                         const constant size_t* layout  [[buffer(2)]],
                         constant size_t& dim           [[buffer(3)]],
                         uint dstIndex [[thread_position_in_grid]])
{
    const constant size_t* shape = layoutShape(layout);
    const constant size_t* strides = layoutStrides(layout);
    size_t shapeSize = layoutRank(layout);
    size_t srcBaseOffset = layoutOffset(layout) + reducedSourceBaseOffset(dstIndex, shape, strides, shapeSize, dim);
    size_t baseLogicalIndex = reducedLogicalBaseIndex(dstIndex, shape, shapeSize, dim);
    size_t logicalStride = logicalStrideForDimension(shape, shapeSize, dim);

    T bestValue = src[srcBaseOffset];
    int bestIndex = static_cast<int>((baseLogicalIndex / strides[dim]) % shape[dim]);
    for (size_t i = 1; i < shape[dim]; ++i)
    {
        T candidateValue = src[srcBaseOffset + i * strides[dim]];
        if (argmaxIsBetter(candidateValue, static_cast<int>(i), bestValue, bestIndex))
        {
            bestValue = candidateValue;
            size_t candidateLogicalIndex = baseLogicalIndex + i * logicalStride;
            bestIndex = static_cast<int>((candidateLogicalIndex / strides[dim]) % shape[dim]);
        }
    }

    dst[dstIndex] = bestIndex;
}


[[kernel]] void argmaxIndicesSet(const device int* winningIndex [[buffer(0)]],
                                 device int* result             [[buffer(1)]],
                                 constant size_t& resultSize    [[buffer(2)]],
                                 uint index [[thread_position_in_grid]])
{
    if (index == 0)
    {
        auto winner = static_cast<size_t>(winningIndex[0]);
        if (winner < resultSize)
        {
            result[winner] = 1;
        }
    }
}


[[kernel]] void argmaxIndicesToSet(const device int* winningIndices [[buffer(0)]],
                                   device int* result              [[buffer(1)]],
                                   const device size_t* shape      [[buffer(2)]],
                                   const device size_t* strides    [[buffer(3)]],
                                   constant size_t& shapeSize      [[buffer(4)]],
                                   constant size_t& dim            [[buffer(5)]],
                                   uint reducedIndex [[thread_position_in_grid]])
{
    size_t baseOffset = reducedSourceBaseOffset(reducedIndex, shape, strides, shapeSize, dim);

    auto winner = static_cast<size_t>(winningIndices[reducedIndex]);
    if (winner < shape[dim])
    {
        result[baseOffset + winner * strides[dim]] = 1;
    }
}


// TranslationIndex - Naive Implementation
// -----------------------------------------------------------------
size_t translationIndex(size_t index, device const size_t* shape, device const size_t* newShape,
                        size_t shapeSize, size_t newShapeSize)
{
    return translateLogicalIndex(index, shape, newShape, shapeSize, newShapeSize);
}


// Contiguous - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
[[kernel]] void contiguous(const device T* src       [[buffer(0)]],
                           device       T* dst       [[buffer(1)]],
                           const constant T2* layout [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    dst[index] = src[physicalIndex(index, layout)];
}


// ReduceTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
[[kernel]] void reduceTo(const device T* src       [[buffer(0)]],
                         device       T* dst       [[buffer(1)]],
                         const device T2* newShape [[buffer(2)]],
                         constant T2& newShapeSize [[buffer(3)]],
                         const constant T2* layout [[buffer(4)]],
                         uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translateLogicalIndex(index, newShape, layoutShape(layout), newShapeSize, layoutRank(layout));
    atomic_fetch_add_explicit((device atomic<T>*)&(dst[originalIndex]), src[physicalIndex(index, layout)], memory_order_relaxed);

    // NOTE: Metal Framework supports add and sub operations for only atomic_float, atomic_uint and atomic_int.
}


// MaxTo - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
[[kernel]] void maxTo(const device T* src       [[buffer(0)]],
                      device       T* dst       [[buffer(1)]],
                      const device T2* newShape [[buffer(2)]],
                      constant T2& newShapeSize [[buffer(3)]],
                      const constant T2* layout [[buffer(4)]],
                      uint index [[thread_position_in_grid]])
{
    size_t originalIndex = translateLogicalIndex(index, newShape, layoutShape(layout), newShapeSize, layoutRank(layout));
    aix_atomic_fetch_max_explicit((device aix_atomic<T>*)&(dst[originalIndex]), src[physicalIndex(index, layout)], memory_order_relaxed);
}


// SliceSet - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2>
[[kernel]] void sliceSet(const device T* src       [[buffer(0)]],
                         device       T* dst       [[buffer(1)]],
                         const constant T2* srcLayout [[buffer(2)]],
                         const constant T2* dstLayout [[buffer(3)]],
                         constant T2& dim             [[buffer(4)]],
                         constant T2& start           [[buffer(5)]],
                         constant T2& step            [[buffer(6)]],
                         uint index [[thread_position_in_grid]])
{
    size_t logicalIndex = index;
    size_t srcIndex = layoutOffset(srcLayout);
    size_t dstIndex = layoutOffset(dstLayout);
    const constant size_t* srcShape = layoutShape(srcLayout);
    const constant size_t* srcStrides = layoutStrides(srcLayout);
    const constant size_t* dstStrides = layoutStrides(dstLayout);
    size_t rank = layoutRank(srcLayout);

    for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i)
    {
        size_t coordinate = logicalIndex % srcShape[i];
        logicalIndex /= srcShape[i];
        srcIndex += coordinate * srcStrides[i];

        if (i == static_cast<int64_t>(dim))
            dstIndex += (start + coordinate * step) * dstStrides[i];
        else
            dstIndex += coordinate * dstStrides[i];
    }

    dst[dstIndex] = src[srcIndex];
}


// Tril - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2, typename T3>
[[kernel]] void tril(device T* dst             [[buffer(1)]],
                     const device T2* shape    [[buffer(2)]],
                     const device T2* strides  [[buffer(3)]],
                     constant T2& shapeSize    [[buffer(4)]],
                     constant T2& stridesSize  [[buffer(5)]],
                     constant T3& diagonal     [[buffer(6)]],
                     constant T2& size         [[buffer(7)]],
                     uint index [[thread_position_in_grid]])
{
    size_t rows = shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
    size_t cols = shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

    for (size_t i = 0; i < size; ++i)
    {
        // Calculate the row and column indices for the last 2-dim slice.
        size_t row = (i / strides[shapeSize - 2]) % rows;
        size_t col = (i / strides[shapeSize - 1]) % cols;

        // Zero out the elements above the specified diagonal.
        if (static_cast<int64_t>(col) > static_cast<int64_t>(row) + diagonal)
        {
            dst[i] = 0;
        }
    }
}


// Triu - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2, typename T3>
[[kernel]] void triu(device T* dst             [[buffer(1)]],
                     const device T2* shape    [[buffer(2)]],
                     const device T2* strides  [[buffer(3)]],
                     constant T2& shapeSize    [[buffer(4)]],
                     constant T2& stridesSize  [[buffer(5)]],
                     constant T3& diagonal     [[buffer(6)]],
                     constant T2& size         [[buffer(7)]],
                     uint index [[thread_position_in_grid]])
{
    size_t rows = shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
    size_t cols = shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

    for (size_t i = 0; i < size; ++i)
    {
        // Calculate the row and column indices for the last 2-dim slice.
        size_t row = (i / strides[shapeSize - 2]) % rows;
        size_t col = (i / strides[shapeSize - 1]) % cols;

        // Zero out the elements above the specified diagonal.
        if (static_cast<int64_t>(col) < static_cast<int64_t>(row) + diagonal)
        {
            dst[i] = 0;
        }
    }
}


// IndexSelect - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2, typename T3>
[[kernel]] void indexSelect(const device T* src       [[buffer(0)]],
                            device T* dst             [[buffer(1)]],
                            const device T2* indices  [[buffer(2)]],
                            constant T3& indicesSize  [[buffer(3)]],
                            constant T3& dimSize      [[buffer(4)]],
                            constant T3& sliceSize    [[buffer(5)]],
                            const constant T3* srcLayout [[buffer(6)]],
                            const constant T3* indicesLayout [[buffer(7)]],
                            uint index [[thread_position_in_grid]])
{
    size_t elementWithinSlice = index % sliceSize;
    size_t idx = (index / sliceSize) % indicesSize;
    size_t outer = index / (indicesSize * sliceSize);
    size_t selectedIndex = static_cast<size_t>(indices[physicalIndex(idx, indicesLayout)]);
    size_t srcIndex  = selectedIndex * sliceSize + elementWithinSlice;
    size_t srcOffset = outer * dimSize + srcIndex;
    size_t dstOffset = outer * indicesSize * sliceSize + idx * sliceSize + elementWithinSlice;

    dst[dstOffset] = src[physicalIndex(srcOffset, srcLayout)];
}


// IndexAdd - Naive Implementation
// -----------------------------------------------------------------
template<typename T, typename T2, typename T3>
[[kernel]] void indexAdd(const device T* src       [[buffer(0)]],
                         device T* dst             [[buffer(1)]],
                         const device T2* indices  [[buffer(2)]],
                         constant T3& indicesSize  [[buffer(3)]],
                         constant T3& dimSize      [[buffer(4)]],
                         constant T3& sliceSize    [[buffer(5)]],
                         const constant T3* srcLayout [[buffer(6)]],
                         const constant T3* dstLayout [[buffer(7)]],
                         const constant T3* indicesLayout [[buffer(8)]],
                         uint index [[thread_position_in_grid]])
{
    size_t elementWithinSlice = index % sliceSize;
    size_t idx = (index / sliceSize) % indicesSize;
    size_t outer = index / (indicesSize * sliceSize);
    size_t selectedIndex = static_cast<size_t>(indices[physicalIndex(idx, indicesLayout)]);
    size_t dstIndex = selectedIndex * sliceSize + elementWithinSlice;
    size_t dstOffset = outer * dimSize + dstIndex;
    size_t srcOffset = outer * indicesSize * sliceSize + idx * sliceSize + elementWithinSlice;
    atomic_fetch_add_explicit((device atomic<T>*)&(dst[physicalIndex(dstOffset, dstLayout)]),
                              src[physicalIndex(srcOffset, srcLayout)],
                              memory_order_relaxed);
}


// nullKernel
// -----------------------------------------------------------------
[[kernel]] void nullKernel(uint index [[thread_position_in_grid]])
{
}


// -----------------------------------------------------------------
// TEMPLATE SPECIALIZATIONS
// -----------------------------------------------------------------


// Add
// -----------------------------------------------------------------
#define SpecializeAdd(tname, type)  \
    template [[ host_name("add_" tname) ]]  \
    [[kernel]] void add(const device type* inA  [[buffer(0)]], \
                        const device type* inB  [[buffer(1)]], \
                        device type* result     [[buffer(2)]], \
                        uint index [[thread_position_in_grid]])

#define SpecializeAddStrided(tname, type)  \
    template [[ host_name("add_strided_" tname) ]]  \
    [[kernel]] void addStrided(const device type* inA           [[buffer(0)]], \
                               const device type* inB           [[buffer(1)]], \
                               device type* result              [[buffer(2)]], \
                               const constant size_t* layoutA   [[buffer(3)]], \
                               const constant size_t* layoutB   [[buffer(4)]], \
                               uint index [[thread_position_in_grid]])

SpecializeAdd("f32",  float4);
SpecializeAdd("f16",  half4);
SpecializeAdd("bf16", bfloat4);
SpecializeAdd("i64",  long4);
SpecializeAdd("i32",  int4);
SpecializeAdd("i16",  short4);
SpecializeAdd("i8",   char4);
SpecializeAdd("ui8",  uchar4);
SpecializeAddStrided("f32",  float);
SpecializeAddStrided("f16",  half);
SpecializeAddStrided("bf16", bfloat);
SpecializeAddStrided("i64",  long);
SpecializeAddStrided("i32",  int);
SpecializeAddStrided("i16",  short);
SpecializeAddStrided("i8",   char);
SpecializeAddStrided("ui8",  uchar);


// Sub
// -----------------------------------------------------------------
#define SpecializeSub(tname, type)  \
    template [[ host_name("sub_" tname) ]]  \
    [[kernel]] void sub(const device type* inA  [[buffer(0)]], \
                        const device type* inB  [[buffer(1)]], \
                        device type* result     [[buffer(2)]], \
                        uint index [[thread_position_in_grid]])

#define SpecializeSubStrided(tname, type)  \
    template [[ host_name("sub_strided_" tname) ]]  \
    [[kernel]] void subStrided(const device type* inA           [[buffer(0)]], \
                               const device type* inB           [[buffer(1)]], \
                               device type* result              [[buffer(2)]], \
                               const constant size_t* layoutA   [[buffer(3)]], \
                               const constant size_t* layoutB   [[buffer(4)]], \
                               uint index [[thread_position_in_grid]])

SpecializeSub("f32",  float4);
SpecializeSub("f16",  half4);
SpecializeSub("bf16", bfloat4);
SpecializeSub("i64",  long4);
SpecializeSub("i32",  int4);
SpecializeSub("i16",  short4);
SpecializeSub("i8",   char4);
SpecializeSub("ui8",  uchar4);
SpecializeSubStrided("f32",  float);
SpecializeSubStrided("f16",  half);
SpecializeSubStrided("bf16", bfloat);
SpecializeSubStrided("i64",  long);
SpecializeSubStrided("i32",  int);
SpecializeSubStrided("i16",  short);
SpecializeSubStrided("i8",   char);
SpecializeSubStrided("ui8",  uchar);


// Mul
// -----------------------------------------------------------------
#define SpecializeMul(tname, type)  \
    template [[ host_name("mul_" tname) ]]  \
    [[kernel]] void mul(const device type* inA  [[buffer(0)]], \
                        const device type* inB  [[buffer(1)]], \
                        device type* result     [[buffer(2)]], \
                        uint index [[thread_position_in_grid]])

#define SpecializeMulStrided(tname, type)  \
    template [[ host_name("mul_strided_" tname) ]]  \
    [[kernel]] void mulStrided(const device type* inA           [[buffer(0)]], \
                               const device type* inB           [[buffer(1)]], \
                               device type* result              [[buffer(2)]], \
                               const constant size_t* layoutA   [[buffer(3)]], \
                               const constant size_t* layoutB   [[buffer(4)]], \
                               uint index [[thread_position_in_grid]])

SpecializeMul("f32",  float4);
SpecializeMul("f16",  half4);
SpecializeMul("bf16", bfloat4);
SpecializeMul("i64",  long4);
SpecializeMul("i32",  int4);
SpecializeMul("i16",  short4);
SpecializeMul("i8",   char4);
SpecializeMul("ui8",  uchar4);
SpecializeMulStrided("f32",  float);
SpecializeMulStrided("f16",  half);
SpecializeMulStrided("bf16", bfloat);
SpecializeMulStrided("i64",  long);
SpecializeMulStrided("i32",  int);
SpecializeMulStrided("i16",  short);
SpecializeMulStrided("i8",   char);
SpecializeMulStrided("ui8",  uchar);


// Div
// -----------------------------------------------------------------
#define SpecializeDiv(tname, type)  \
    template [[ host_name("div_" tname) ]]  \
    [[kernel]] void div(const device type* inA  [[buffer(0)]], \
                        const device type* inB  [[buffer(1)]], \
                        device type* result     [[buffer(2)]], \
                        uint index [[thread_position_in_grid]])

#define SpecializeDivStrided(tname, type)  \
    template [[ host_name("div_strided_" tname) ]]  \
    [[kernel]] void divStrided(const device type* inA           [[buffer(0)]], \
                               const device type* inB           [[buffer(1)]], \
                               device type* result              [[buffer(2)]], \
                               const constant size_t* layoutA   [[buffer(3)]], \
                               const constant size_t* layoutB   [[buffer(4)]], \
                               uint index [[thread_position_in_grid]])

SpecializeDiv("f32",  float4);
SpecializeDiv("f16",  half4);
SpecializeDiv("bf16", bfloat4);
SpecializeDiv("i64",  long4);
SpecializeDiv("i32",  int4);
SpecializeDiv("i16",  short4);
SpecializeDiv("i8",   char4);
SpecializeDiv("ui8",  uchar4);
SpecializeDivStrided("f32",  float);
SpecializeDivStrided("f16",  half);
SpecializeDivStrided("bf16", bfloat);
SpecializeDivStrided("i64",  long);
SpecializeDivStrided("i32",  int);
SpecializeDivStrided("i16",  short);
SpecializeDivStrided("i8",   char);
SpecializeDivStrided("ui8",  uchar);


// Sqrt
// -----------------------------------------------------------------
#define SpecializeSqrt(tname, type)  \
    template [[ host_name("sqrt_" tname) ]]  \
    [[kernel]] void sqrt(const device type* inA   [[buffer(0)]],  \
                         device type* result      [[buffer(1)]],  \
                         uint index [[thread_position_in_grid]])

#define SpecializeSqrtStrided(tname, type)  \
    template [[ host_name("sqrt_strided_" tname) ]]  \
    [[kernel]] void sqrtStrided(const device type* inA          [[buffer(0)]], \
                                device type* result             [[buffer(1)]], \
                                const constant size_t* layoutA  [[buffer(2)]], \
                                uint index [[thread_position_in_grid]])

SpecializeSqrt("f32",  float4);
SpecializeSqrt("f16",  half4);
SpecializeSqrt("bf16", bfloat4);
SpecializeSqrt("i64",  long4);
SpecializeSqrt("i32",  int4);
SpecializeSqrt("i16",  short4);
SpecializeSqrt("i8",   char4);
SpecializeSqrt("ui8",  uchar4);
SpecializeSqrtStrided("f32",  float);
SpecializeSqrtStrided("f16",  half);
SpecializeSqrtStrided("bf16", bfloat);
SpecializeSqrtStrided("i64",  long);
SpecializeSqrtStrided("i32",  int);
SpecializeSqrtStrided("i16",  short);
SpecializeSqrtStrided("i8",   char);
SpecializeSqrtStrided("ui8",  uchar);


// Sin
// -----------------------------------------------------------------
#define SpecializeSin(tname, type)  \
    template [[ host_name("sin_" tname) ]]  \
    [[kernel]] void sin(const device type* inA   [[buffer(0)]],  \
                        device type* result      [[buffer(1)]],  \
                        uint index [[thread_position_in_grid]])

#define SpecializeSinStrided(tname, type)  \
    template [[ host_name("sin_strided_" tname) ]]  \
    [[kernel]] void sinStrided(const device type* inA           [[buffer(0)]], \
                               device type* result              [[buffer(1)]], \
                               const constant size_t* layoutA   [[buffer(2)]], \
                               uint index [[thread_position_in_grid]])

SpecializeSin("f32",  float4);
SpecializeSin("f16",  half4);
SpecializeSin("bf16", bfloat4);
SpecializeSin("i64",  long4);
SpecializeSin("i32",  int4);
SpecializeSin("i16",  short4);
SpecializeSin("i8",   char4);
SpecializeSin("ui8",  uchar4);
SpecializeSinStrided("f32",  float);
SpecializeSinStrided("f16",  half);
SpecializeSinStrided("bf16", bfloat);
SpecializeSinStrided("i64",  long);
SpecializeSinStrided("i32",  int);
SpecializeSinStrided("i16",  short);
SpecializeSinStrided("i8",   char);
SpecializeSinStrided("ui8",  uchar);


// Cos
// -----------------------------------------------------------------
#define SpecializeCos(tname, type)  \
    template [[ host_name("cos_" tname) ]]  \
    [[kernel]] void cos(const device type* inA   [[buffer(0)]],  \
                        device type* result      [[buffer(1)]],  \
                        uint index [[thread_position_in_grid]])

#define SpecializeCosStrided(tname, type)  \
    template [[ host_name("cos_strided_" tname) ]]  \
    [[kernel]] void cosStrided(const device type* inA           [[buffer(0)]], \
                               device type* result              [[buffer(1)]], \
                               const constant size_t* layoutA   [[buffer(2)]], \
                               uint index [[thread_position_in_grid]])

SpecializeCos("f32",  float4);
SpecializeCos("f16",  half4);
SpecializeCos("bf16", bfloat4);
SpecializeCos("i64",  long4);
SpecializeCos("i32",  int4);
SpecializeCos("i16",  short4);
SpecializeCos("i8",   char4);
SpecializeCos("ui8",  uchar4);
SpecializeCosStrided("f32",  float);
SpecializeCosStrided("f16",  half);
SpecializeCosStrided("bf16", bfloat);
SpecializeCosStrided("i64",  long);
SpecializeCosStrided("i32",  int);
SpecializeCosStrided("i16",  short);
SpecializeCosStrided("i8",   char);
SpecializeCosStrided("ui8",  uchar);


// Tanh
// -----------------------------------------------------------------
#define SpecializeTanh(tname, type)  \
    template [[ host_name("tanh_" tname) ]]  \
    [[kernel]] void tanh(const device type* inA   [[buffer(0)]],  \
                         device type* result      [[buffer(1)]],  \
                         uint index [[thread_position_in_grid]])

#define SpecializeTanhStrided(tname, type)  \
    template [[ host_name("tanh_strided_" tname) ]]  \
    [[kernel]] void tanhStrided(const device type* inA          [[buffer(0)]], \
                                device type* result             [[buffer(1)]], \
                                const constant size_t* layoutA  [[buffer(2)]], \
                                uint index [[thread_position_in_grid]])

SpecializeTanh("f32",  float4);
SpecializeTanh("f16",  half4);
SpecializeTanh("bf16", bfloat4);
SpecializeTanh("i64",  long4);
SpecializeTanh("i32",  int4);
SpecializeTanh("i16",  short4);
SpecializeTanh("i8",   char4);
SpecializeTanh("ui8",  uchar4);
SpecializeTanhStrided("f32",  float);
SpecializeTanhStrided("f16",  half);
SpecializeTanhStrided("bf16", bfloat);
SpecializeTanhStrided("i64",  long);
SpecializeTanhStrided("i32",  int);
SpecializeTanhStrided("i16",  short);
SpecializeTanhStrided("i8",   char);
SpecializeTanhStrided("ui8",  uchar);


// Log
// -----------------------------------------------------------------
#define SpecializeLog(tname, type)  \
    template [[ host_name("log_" tname) ]]  \
    [[kernel]] void log(const device type* inA   [[buffer(0)]],  \
                        device type* result      [[buffer(1)]],  \
                        uint index [[thread_position_in_grid]])

#define SpecializeLogStrided(tname, type)  \
    template [[ host_name("log_strided_" tname) ]]  \
    [[kernel]] void logStrided(const device type* inA           [[buffer(0)]], \
                               device type* result              [[buffer(1)]], \
                               const constant size_t* layoutA   [[buffer(2)]], \
                               uint index [[thread_position_in_grid]])

SpecializeLog("f32",  float4);
SpecializeLog("f16",  half4);
SpecializeLog("bf16", bfloat4);
SpecializeLog("i64",  long4);
SpecializeLog("i32",  int4);
SpecializeLog("i16",  short4);
SpecializeLog("i8",   char4);
SpecializeLog("ui8",  uchar4);
SpecializeLogStrided("f32",  float);
SpecializeLogStrided("f16",  half);
SpecializeLogStrided("bf16", bfloat);
SpecializeLogStrided("i64",  long);
SpecializeLogStrided("i32",  int);
SpecializeLogStrided("i16",  short);
SpecializeLogStrided("i8",   char);
SpecializeLogStrided("ui8",  uchar);


// Exp
// -----------------------------------------------------------------
#define SpecializeExp(tname, type)  \
    template [[ host_name("exp_" tname) ]]  \
    [[kernel]] void exp(const device type* inA   [[buffer(0)]],  \
                        device type* result      [[buffer(1)]],  \
                        uint index [[thread_position_in_grid]])

#define SpecializeExpStrided(tname, type)  \
    template [[ host_name("exp_strided_" tname) ]]  \
    [[kernel]] void expStrided(const device type* inA           [[buffer(0)]], \
                               device type* result              [[buffer(1)]], \
                               const constant size_t* layoutA   [[buffer(2)]], \
                               uint index [[thread_position_in_grid]])

SpecializeExp("f32",  float4);
SpecializeExp("f16",  half4);
SpecializeExp("bf16", bfloat4);
SpecializeExp("i64",  long4);
SpecializeExp("i32",  int4);
SpecializeExp("i16",  short4);
SpecializeExp("i8",   char4);
SpecializeExp("ui8",  uchar4);
SpecializeExpStrided("f32",  float);
SpecializeExpStrided("f16",  half);
SpecializeExpStrided("bf16", bfloat);
SpecializeExpStrided("i64",  long);
SpecializeExpStrided("i32",  int);
SpecializeExpStrided("i16",  short);
SpecializeExpStrided("i8",   char);
SpecializeExpStrided("ui8",  uchar);


// Pow
// -----------------------------------------------------------------
#define SpecializePow(tname, type)  \
    template [[ host_name("pow_" tname) ]]  \
    [[kernel]] void pow(const device type* inA      [[buffer(0)]],  \
                        const device type* expA     [[buffer(1)]],  \
                        device type* result         [[buffer(2)]],  \
                        uint index [[thread_position_in_grid]])

#define SpecializePowStrided(tname, type)  \
    template [[ host_name("pow_strided_" tname) ]]  \
    [[kernel]] void powStrided(const device type* inA           [[buffer(0)]], \
                               const device type* inB           [[buffer(1)]], \
                               device type* result              [[buffer(2)]], \
                               const constant size_t* layoutA   [[buffer(3)]], \
                               const constant size_t* layoutB   [[buffer(4)]], \
                               uint index [[thread_position_in_grid]])

SpecializePow("f32",  float4);
SpecializePow("f16",  half4);
SpecializePow("bf16", bfloat4);
SpecializePow("i64",  long4);
SpecializePow("i32",  int4);
SpecializePow("i16",  short4);
SpecializePow("i8",   char4);
SpecializePow("ui8",  uchar4);
SpecializePowStrided("f32",  float);
SpecializePowStrided("f16",  half);
SpecializePowStrided("bf16", bfloat);
SpecializePowStrided("i64",  long);
SpecializePowStrided("i32",  int);
SpecializePowStrided("i16",  short);
SpecializePowStrided("i8",   char);
SpecializePowStrided("ui8",  uchar);


// Sum
// -----------------------------------------------------------------
#define SpecializeSum(tname, type)  \
    template [[ host_name("sum_" tname) ]]  \
    [[kernel]] void sum(const device type* inA      [[buffer(0)]],   \
                         device type* result         [[buffer(1)]],   \
                         constant size_t& elementCount [[buffer(2)]], \
                         constant size_t& useLayout [[buffer(3)]], \
                         const constant size_t* layout [[buffer(4)]], \
                         uint li  [[thread_position_in_threadgroup]], \
                         uint tgi [[threadgroup_position_in_grid]],   \
                         uint threadsPerThreadgroup [[threads_per_threadgroup]])

SpecializeSum("f32",  float);
SpecializeSum("f16",  half);
SpecializeSum("bf16", bfloat);
SpecializeSum("i64",  long);
SpecializeSum("i32",  int);
SpecializeSum("i16",  short);
SpecializeSum("i8",   char);
SpecializeSum("ui8",  uchar);


// Max
// -----------------------------------------------------------------
#define SpecializeMax(tname, type)  \
    template [[ host_name("max_" tname) ]]  \
    [[kernel]] void max(const device type* inA      [[buffer(0)]],   \
                         device type* result         [[buffer(1)]],   \
                         constant size_t& elementCount [[buffer(2)]], \
                         constant size_t& useLayout [[buffer(3)]], \
                         const constant size_t* layout [[buffer(4)]], \
                         uint li  [[thread_position_in_threadgroup]], \
                         uint tgi [[threadgroup_position_in_grid]],   \
                         uint threadsPerThreadgroup [[threads_per_threadgroup]])

SpecializeMax("f32",  float);
SpecializeMax("f16",  half);
SpecializeMax("bf16", bfloat);
SpecializeMax("i64",  long);
SpecializeMax("i32",  int);
SpecializeMax("i16",  short);
SpecializeMax("i8",   char);
SpecializeMax("ui8",  uchar);


// ArgMax
// -----------------------------------------------------------------
#define SpecializeArgmaxInit(tname, type)  \
    template [[ host_name("argmaxInit_" tname) ]]  \
    [[kernel]] void argmaxInit(const device type* src   [[buffer(0)]], \
                                device type* values      [[buffer(1)]], \
                                device int* indices      [[buffer(2)]], \
                                constant size_t& useLayout [[buffer(3)]], \
                                const constant size_t* layout [[buffer(4)]], \
                                uint index [[thread_position_in_grid]])

#define SpecializeArgmaxReduce(tname, type)  \
    template [[ host_name("argmaxReduce_" tname) ]]  \
    [[kernel]] void argmaxReduce(const device type* inValues     [[buffer(0)]], \
                                 const device int* inIndices     [[buffer(1)]], \
                                 device type* outValues          [[buffer(2)]], \
                                 device int* outIndices          [[buffer(3)]], \
                                 uint li  [[thread_position_in_threadgroup]], \
                                 uint tgi [[threadgroup_position_in_grid]],   \
                                 uint threadsPerThreadgroup [[threads_per_threadgroup]])

#define SpecializeArgmaxTo(tname, type)  \
    template [[ host_name("argmaxTo_" tname) ]]  \
    [[kernel]] void argmaxTo(const device type* src           [[buffer(0)]], \
                             device int* dst                  [[buffer(1)]], \
                             const constant size_t* layout    [[buffer(2)]], \
                             constant size_t& dim             [[buffer(3)]], \
                             uint dstIndex [[thread_position_in_grid]])

SpecializeArgmaxInit("f32",  float);
SpecializeArgmaxInit("f16",  half);
SpecializeArgmaxInit("bf16", bfloat);
SpecializeArgmaxInit("i64",  long);
SpecializeArgmaxInit("i32",  int);
SpecializeArgmaxInit("i16",  short);
SpecializeArgmaxInit("i8",   char);
SpecializeArgmaxInit("ui8",  uchar);

SpecializeArgmaxReduce("f32",  float);
SpecializeArgmaxReduce("f16",  half);
SpecializeArgmaxReduce("bf16", bfloat);
SpecializeArgmaxReduce("i64",  long);
SpecializeArgmaxReduce("i32",  int);
SpecializeArgmaxReduce("i16",  short);
SpecializeArgmaxReduce("i8",   char);
SpecializeArgmaxReduce("ui8",  uchar);

SpecializeArgmaxTo("f32",  float);
SpecializeArgmaxTo("f16",  half);
SpecializeArgmaxTo("bf16", bfloat);
SpecializeArgmaxTo("i64",  long);
SpecializeArgmaxTo("i32",  int);
SpecializeArgmaxTo("i16",  short);
SpecializeArgmaxTo("i8",   char);
SpecializeArgmaxTo("ui8",  uchar);


// Matrix_Mul
// -----------------------------------------------------------------
#define SpecializeMatrixMulStrided(tname, type) \
    template [[ host_name("matrixMulStrided_" tname) ]] \
    [[kernel]] void matrixMulStrided(const device type* inA, \
                                     const device type* inB, \
                                     device type* result, \
                                     constant MatrixSize& matASize, \
                                     constant MatrixSize& matBSize, \
                                     const constant size_t* layoutA, \
                                     const constant size_t* layoutB, \
                                     uint2 gid [[thread_position_in_grid]])

SpecializeMatrixMulStrided("f32",  float);
SpecializeMatrixMulStrided("f16",  half);
SpecializeMatrixMulStrided("bf16", bfloat);
SpecializeMatrixMulStrided("i64",  long);
SpecializeMatrixMulStrided("i32",  int);
SpecializeMatrixMulStrided("i16",  short);
SpecializeMatrixMulStrided("i8",   char);
SpecializeMatrixMulStrided("ui8",  uchar);

#define SpecializeMatrixMulStridedTiled(tname, n, type) \
    template [[ host_name("matrixMulStridedTiled_" #n "_" #n "_" tname) ]] \
    [[kernel]] void matrixMulStridedTiled<type, n>(const device type* inA, \
                                                   const device type* inB, \
                                                   device type* result, \
                                                   constant MatrixSize& matASize, \
                                                   constant MatrixSize& matBSize, \
                                                   const constant size_t* layoutA, \
                                                   const constant size_t* layoutB, \
                                                   uint2 gid [[thread_position_in_grid]], \
                                                   uint2 tid [[thread_position_in_threadgroup]])

SpecializeMatrixMulStridedTiled("f32",  16, float);
SpecializeMatrixMulStridedTiled("f16",  16, half);
SpecializeMatrixMulStridedTiled("bf16", 16, bfloat);
SpecializeMatrixMulStridedTiled("i64",  16, long);
SpecializeMatrixMulStridedTiled("i32",  16, int);
SpecializeMatrixMulStridedTiled("i16",  16, short);
SpecializeMatrixMulStridedTiled("i8",   16, char);
SpecializeMatrixMulStridedTiled("ui8",  16, uchar);

#define SpecializeMatrixMulTiledBC(tname, bm, bn, bk, tm, tn, type)  \
    template [[ host_name("matrixMulTiledBC_" #bm "_" #bn "_" #bk "_" #tm "_" #tn "_" tname) ]]  \
    [[kernel]] void matrixMulTiledBC<type,bm,bn,bk,tm,tn>(const device type* inA,  \
                                                          const device type* inB,  \
                                                          device type* result,     \
                                                          constant MatrixSize& matASize,  \
                                                          constant MatrixSize& matBSize,  \
                                                          uint2 tgid [[threadgroup_position_in_grid]],  \
                                                          uint2 lid  [[thread_position_in_threadgroup]])

#define DeclareConfigMatrixMulTiledBC(bm, bn, bk, tm, tn) \
    SpecializeMatrixMulTiledBC("f32",  bm, bn, bk, tm, tn, float);  \
    SpecializeMatrixMulTiledBC("f16",  bm, bn, bk, tm, tn, half);   \
    SpecializeMatrixMulTiledBC("bf16", bm, bn, bk, tm, tn, bfloat); \
    SpecializeMatrixMulTiledBC("i64",  bm, bn, bk, tm, tn, long);   \
    SpecializeMatrixMulTiledBC("i32",  bm, bn, bk, tm, tn, int);    \
    SpecializeMatrixMulTiledBC("i16",  bm, bn, bk, tm, tn, short);  \
    SpecializeMatrixMulTiledBC("i8",   bm, bn, bk, tm, tn, char);   \
    SpecializeMatrixMulTiledBC("ui8",  bm, bn, bk, tm, tn, unsigned char)

DeclareConfigMatrixMulTiledBC(64, 64, 8, 8, 8);

// clang-format off
// Matrix Mul Tiled
// -----------------------------------------------------------------
#define SpecializeMatrixMulTiled(tname, tsx, tsy, type)  \
    template [[ host_name("matrixMulTiled_" #tsx "_" #tsy "_" tname) ]]  \
    [[kernel]] void matrixMulTiled<type,tsx,tsy>(const device type* inA,  \
                                                 const device type* inB,  \
                                                 device type* result,     \
                                                 constant MatrixSize& matSize1,  \
                                                 constant MatrixSize& matSize2,  \
                                                 uint2 tgid [[threadgroup_position_in_grid]],  \
                                                 uint2 lid  [[thread_position_in_threadgroup]])

#define ImplementSpecializedMatrixMulTiled(tname, tsx, tsy, type)  \
    template <> [[ host_name("matrixMulTiled_" #tsx "_" #tsy "_" tname) ]]  \
    [[kernel]] void matrixMulTiled<type,tsx,tsy>(const device type* inA,  \
                                                 const device type* inB,  \
                                                 device type* result,     \
                                                 constant MatrixSize& matSize1,  \
                                                 constant MatrixSize& matSize2,  \
                                                 uint2 tgid [[threadgroup_position_in_grid]],  \
                                                 uint2 lid  [[thread_position_in_threadgroup]])

#define DeclareConfigMatrixMulTiled(tsx, tsy) \
    SpecializeMatrixMulTiled("f32",  tsx, tsy, float); \
    SpecializeMatrixMulTiled("f16",  tsx, tsy, half); \
    SpecializeMatrixMulTiled("bf16", tsx, tsy, bfloat); \
    ImplementSpecializedMatrixMulTiled("i64", tsx, tsy, long)  { } \
    ImplementSpecializedMatrixMulTiled("i32", tsx, tsy, int)   { } \
    ImplementSpecializedMatrixMulTiled("i16", tsx, tsy, short) { } \
    ImplementSpecializedMatrixMulTiled("i8",  tsx, tsy, char)  { } \
    ImplementSpecializedMatrixMulTiled("ui8", tsx, tsy, unsigned char)  { }

DeclareConfigMatrixMulTiled(32, 32);
DeclareConfigMatrixMulTiled(32, 64);
DeclareConfigMatrixMulTiled(32, 128);
// clang-format on


// Transpose2D
// -----------------------------------------------------------------
#define SpecializeTranspose2D(tname, type)  \
    template [[ host_name("transpose2D_" tname) ]]  \
    [[kernel]] void transpose2D(const device type* mat          [[buffer(0)]],  \
                                device type* result             [[buffer(1)]],  \
                                constant MatrixSize& matSize    [[buffer(2)]],  \
                                uint2 gid [[thread_position_in_grid]],  \
                                uint2 tid [[thread_position_in_threadgroup]])

SpecializeTranspose2D("f32",  float);
SpecializeTranspose2D("f16",  half);
SpecializeTranspose2D("bf16", bfloat);
SpecializeTranspose2D("i64",  long);
SpecializeTranspose2D("i32",  int);
SpecializeTranspose2D("i16",  short);
SpecializeTranspose2D("i8",   char);
SpecializeTranspose2D("ui8",  uchar);


// Transpose2D Tiled
// -----------------------------------------------------------------
#define SpecializeTranspose2DTiled(tname, type, ts, bs)  \
    template [[ host_name("transpose2DTiled_" #ts "_" #ts "_" #bs "_" tname) ]]  \
    [[kernel]] void transpose2DTiled<type,ts,bs>(const device type* mat          [[buffer(0)]],  \
                                                 device type* result             [[buffer(1)]],  \
                                                 constant MatrixSize& matSize    [[buffer(2)]],  \
                                                 uint2 gid  [[thread_position_in_grid]],         \
                                                 uint2 tgid [[threadgroup_position_in_grid]],    \
                                                 uint2 tid  [[thread_position_in_threadgroup]])

#define DeclareConfigTranspose2DTiled(ts, bs)  \
    SpecializeTranspose2DTiled("f32",  float , ts, bs);  \
    SpecializeTranspose2DTiled("f16",  half  , ts, bs);  \
    SpecializeTranspose2DTiled("bf16", bfloat, ts, bs);  \
    SpecializeTranspose2DTiled("i64",  long  , ts, bs);  \
    SpecializeTranspose2DTiled("i32",  int   , ts, bs);  \
    SpecializeTranspose2DTiled("i16",  short , ts, bs);  \
    SpecializeTranspose2DTiled("i8",   char  , ts, bs);  \
    SpecializeTranspose2DTiled("ui8",  uchar , ts, bs);

DeclareConfigTranspose2DTiled(16, 8);
DeclareConfigTranspose2DTiled(32, 8);


// Transpose
// -----------------------------------------------------------------
#define SpecializeTranspose(tname, type)  \
    template [[ host_name("transpose_" tname) ]]  \
    [[kernel]] void transpose(const device type* data         [[buffer(0)]], \
                               device type* result             [[buffer(1)]], \
                               constant size_t& dim0           [[buffer(2)]], \
                               constant size_t& dim1           [[buffer(3)]], \
                               const constant size_t* srcLayout [[buffer(4)]], \
                               constant size_t& size           [[buffer(5)]], \
                               uint index [[thread_position_in_grid]])

SpecializeTranspose("f32",  float);
SpecializeTranspose("f16",  half);
SpecializeTranspose("bf16", bfloat);
SpecializeTranspose("i64",  long);
SpecializeTranspose("i32",  int);
SpecializeTranspose("i16",  short);
SpecializeTranspose("i8",   char);
SpecializeTranspose("ui8",  uchar);


// Copy
// -----------------------------------------------------------------
#define SpecializeCopy(tname1, tname2, type1, type2)  \
    template [[ host_name("copy_" tname1 "_" tname2) ]]  \
    [[kernel]] void copy(const device type1* src    [[buffer(0)]], \
                         device type2* dst          [[buffer(1)]], \
                         uint index [[thread_position_in_grid]])

#define SpecializeCopySet(tname2, type2)   \
    SpecializeCopy("f32",  tname2, float4,  type2);   \
    SpecializeCopy("f16",  tname2, half4,   type2);   \
    SpecializeCopy("bf16", tname2, bfloat4, type2);   \
    SpecializeCopy("i64",  tname2, long4,   type2);   \
    SpecializeCopy("i32",  tname2, int4,    type2);   \
    SpecializeCopy("i16",  tname2, short4,  type2);   \
    SpecializeCopy("i8",   tname2, char4,   type2);   \
    SpecializeCopy("ui8",  tname2, uchar4,  type2);

SpecializeCopySet("f32",  float4 );
SpecializeCopySet("f16",  half4  );
SpecializeCopySet("bf16", bfloat4);
SpecializeCopySet("i64",  long4  );
SpecializeCopySet("i32",  int4   );
SpecializeCopySet("i16",  short4 );
SpecializeCopySet("i8",   char4  );
SpecializeCopySet("ui8",  uchar4 );


// Unary
// -----------------------------------------------------------------
#define SpecializeUnary(tname, type)  \
    template [[ host_name("unary_" tname) ]]  \
    [[kernel]] void unary(const device type* inA    [[buffer(0)]], \
                          device type* result       [[buffer(1)]], \
                          uint index [[thread_position_in_grid]])

#define SpecializeUnaryStrided(tname, type)  \
    template [[ host_name("unary_strided_" tname) ]]  \
    [[kernel]] void unaryStrided(const device type* inA        [[buffer(0)]], \
                                 device type* result           [[buffer(1)]], \
                                 const constant size_t* layoutA [[buffer(2)]], \
                                 uint index [[thread_position_in_grid]])

SpecializeUnary("f32",  float4);
SpecializeUnary("f16",  half4);
SpecializeUnary("bf16", bfloat4);
SpecializeUnary("i64",  long4);
SpecializeUnary("i32",  int4);
SpecializeUnary("i16",  short4);
SpecializeUnary("i8",   char4);
SpecializeUnary("ui8",  uchar4);
SpecializeUnaryStrided("f32",  float);
SpecializeUnaryStrided("f16",  half);
SpecializeUnaryStrided("bf16", bfloat);
SpecializeUnaryStrided("i64",  long);
SpecializeUnaryStrided("i32",  int);
SpecializeUnaryStrided("i16",  short);
SpecializeUnaryStrided("i8",   char);
SpecializeUnaryStrided("ui8",  uchar);


// Fill
// -----------------------------------------------------------------
#define SpecializeFill(tname1, tname2, type1, type2)  \
    template [[ host_name("fill_" tname1 "_" tname2) ]]  \
    [[kernel]] void fill(const device type1* scalar     [[buffer(0)]], \
                         device type2* result           [[buffer(1)]], \
                         uint index [[thread_position_in_grid]])

#define SpecializeFillSet(tname2, type2)   \
    SpecializeFill("f32",  tname2, float4,  type2);   \
    SpecializeFill("f16",  tname2, half4,   type2);   \
    SpecializeFill("bf16", tname2, bfloat4, type2);   \
    SpecializeFill("i64",  tname2, long4,   type2);   \
    SpecializeFill("i32",  tname2, int4,    type2);   \
    SpecializeFill("i16",  tname2, short4,  type2);   \
    SpecializeFill("i8",   tname2, char4,   type2);   \
    SpecializeFill("ui8",  tname2, uchar4,  type2);

SpecializeFillSet("f32",  float4 );
SpecializeFillSet("f16",  half4  );
SpecializeFillSet("bf16", bfloat4);
SpecializeFillSet("i64",  long4  );
SpecializeFillSet("i32",  int4   );
SpecializeFillSet("i16",  short4 );
SpecializeFillSet("i8",   char4  );
SpecializeFillSet("ui8",  uchar4 );


// FillMin
// -----------------------------------------------------------------
#define SpecializeFillMin(tname, type)  \
    template [[ host_name("fillMin_" tname) ]]  \
    [[kernel]] void fillMin(device type* result     [[buffer(0)]], \
                            uint index [[thread_position_in_grid]])

SpecializeFillMin("f32",  float4);
SpecializeFillMin("f16",  half4);
SpecializeFillMin("bf16", bfloat4);
SpecializeFillMin("i64",  long4);
SpecializeFillMin("i32",  int4);
SpecializeFillMin("i16",  short4);
SpecializeFillMin("i8",   char4);
SpecializeFillMin("ui8",  uchar4);


// Contiguous
// -----------------------------------------------------------------
#define SpecializeContiguous(tname, type1, type2)  \
    template [[ host_name("contiguous_" tname) ]]  \
    [[kernel]] void contiguous(const device type1* src       [[buffer(0)]], \
                               device       type1* dst       [[buffer(1)]], \
                               const constant type2* layout  [[buffer(2)]], \
                               uint index [[thread_position_in_grid]])

SpecializeContiguous("f32",  float , size_t);
SpecializeContiguous("f16",  half  , size_t);
SpecializeContiguous("bf16", bfloat, size_t);
SpecializeContiguous("i64",  long  , size_t);
SpecializeContiguous("i32",  int   , size_t);
SpecializeContiguous("i16",  short , size_t);
SpecializeContiguous("i8",   char  , size_t);
SpecializeContiguous("ui8",  uchar , size_t);


// ReduceTo
// -----------------------------------------------------------------
#define SpecializeReduceTo(tname, type1, type2)  \
    template [[ host_name("reduceTo_" tname) ]]  \
    [[kernel]] void reduceTo(const device type1* src       [[buffer(0)]], \
                             device       type1* dst       [[buffer(1)]], \
                             const device type2* newShape  [[buffer(2)]], \
                             constant type2& newShapeSize  [[buffer(3)]], \
                             const constant type2* layout  [[buffer(4)]], \
                             uint index [[thread_position_in_grid]])

#define ImplementSpecializedReduceTo(tname, type1, type2)  \
    template <> [[ host_name("reduceTo_" tname) ]]  \
    [[kernel]] void reduceTo<type1,type2>(const device type1* src       [[buffer(0)]], \
                                          device       type1* dst       [[buffer(1)]], \
                                          const device type2* newShape  [[buffer(2)]], \
                                          constant type2& newShapeSize  [[buffer(3)]], \
                                          const constant type2* layout  [[buffer(4)]], \
                                          uint index [[thread_position_in_grid]]) { }

SpecializeReduceTo("f32",  float , size_t);
SpecializeReduceTo("i32",  int   , size_t);
ImplementSpecializedReduceTo("f16",  half  , size_t);
ImplementSpecializedReduceTo("bf16", bfloat, size_t);
ImplementSpecializedReduceTo("i64",  long  , size_t);
ImplementSpecializedReduceTo("i16",  short , size_t);
ImplementSpecializedReduceTo("i8",   char  , size_t);
ImplementSpecializedReduceTo("ui8",  uchar , size_t);


// MaxTo
// -----------------------------------------------------------------
#define SpecializeMaxTo(tname, type1, type2)  \
    template [[ host_name("maxTo_" tname) ]]  \
    [[kernel]] void maxTo(const device type1* src       [[buffer(0)]], \
                          device       type1* dst       [[buffer(1)]], \
                          const device type2* newShape  [[buffer(2)]], \
                          constant type2& newShapeSize  [[buffer(3)]], \
                          const constant type2* layout  [[buffer(4)]], \
                          uint index [[thread_position_in_grid]])

#define ImplementSpecializedMaxTo(tname, type1, type2)  \
    template <> [[ host_name("maxTo_" tname) ]]  \
    [[kernel]] void maxTo<type1,type2>(const device type1* src       [[buffer(0)]], \
                                       device       type1* dst       [[buffer(1)]], \
                                       const device type2* newShape  [[buffer(2)]], \
                                       constant type2& newShapeSize  [[buffer(3)]], \
                                       const constant type2* layout  [[buffer(4)]], \
                                       uint index [[thread_position_in_grid]]) { }

SpecializeMaxTo("f32",  float , size_t);
SpecializeMaxTo("f16",  half  , size_t);
SpecializeMaxTo("i32",  int   , size_t);
SpecializeMaxTo("i16",  short , size_t);
SpecializeMaxTo("i8",   char  , size_t);
SpecializeMaxTo("ui8",  uchar , size_t);
ImplementSpecializedMaxTo("bf16", bfloat, size_t);
ImplementSpecializedMaxTo("i64",  long  , size_t);


// SliceSet
// -----------------------------------------------------------------
#define SpecializeSliceSet(tname, type1, type2)  \
    template [[ host_name("sliceSet_" tname) ]]  \
    [[kernel]] void sliceSet(const device type1* src      [[buffer(0)]],  \
                              device       type1* dst      [[buffer(1)]],  \
                              const constant type2* srcLayout [[buffer(2)]],  \
                              const constant type2* dstLayout [[buffer(3)]],  \
                              constant type2& dim             [[buffer(4)]],  \
                              constant type2& start           [[buffer(5)]],  \
                              constant type2& step            [[buffer(6)]], \
                              uint index [[thread_position_in_grid]])

SpecializeSliceSet("f32",  float , size_t);
SpecializeSliceSet("f16",  half  , size_t);
SpecializeSliceSet("bf16", bfloat, size_t);
SpecializeSliceSet("i64",  long  , size_t);
SpecializeSliceSet("i32",  int   , size_t);
SpecializeSliceSet("i16",  short , size_t);
SpecializeSliceSet("i8",   char  , size_t);
SpecializeSliceSet("ui8",  uchar , size_t);


// Tril
// -----------------------------------------------------------------
#define SpecializeTril(tname, type1, type2, type3)  \
    template [[ host_name("tril_" tname) ]]  \
    [[kernel]] void tril(device type1* dst            [[buffer(1)]], \
                         const device type2* shape    [[buffer(2)]], \
                         const device type2* strides  [[buffer(3)]], \
                         constant type2& shapeSize    [[buffer(4)]], \
                         constant type2& stridesSize  [[buffer(5)]], \
                         constant type3& diagonal     [[buffer(6)]], \
                         constant type2& size         [[buffer(7)]], \
                         uint index [[thread_position_in_grid]])

SpecializeTril("f32",  float , size_t, int64_t);
SpecializeTril("f16",  half  , size_t, int64_t);
SpecializeTril("bf16", bfloat, size_t, int64_t);
SpecializeTril("i64",  long  , size_t, int64_t);
SpecializeTril("i32",  int   , size_t, int64_t);
SpecializeTril("i16",  short , size_t, int64_t);
SpecializeTril("i8",   char  , size_t, int64_t);
SpecializeTril("ui8",  uchar , size_t, int64_t);


// Triu
// -----------------------------------------------------------------
#define SpecializeTriu(tname, type1, type2, type3)  \
    template [[ host_name("triu_" tname) ]]  \
    [[kernel]] void triu(device type1* dst            [[buffer(1)]], \
                         const device type2* shape    [[buffer(2)]], \
                         const device type2* strides  [[buffer(3)]], \
                         constant type2& shapeSize    [[buffer(4)]], \
                         constant type2& stridesSize  [[buffer(5)]], \
                         constant type3& diagonal     [[buffer(6)]], \
                         constant type2& size         [[buffer(7)]], \
                         uint index [[thread_position_in_grid]])

SpecializeTriu("f32",  float , size_t, int64_t);
SpecializeTriu("f16",  half  , size_t, int64_t);
SpecializeTriu("bf16", bfloat, size_t, int64_t);
SpecializeTriu("i64",  long  , size_t, int64_t);
SpecializeTriu("i32",  int   , size_t, int64_t);
SpecializeTriu("i16",  short , size_t, int64_t);
SpecializeTriu("i8",   char  , size_t, int64_t);
SpecializeTriu("ui8",  uchar , size_t, int64_t);


// Index Select
// -----------------------------------------------------------------
#define SpecializeIndexSelect(tname, type1, type2, type3)  \
    template [[ host_name("indexSelect_" tname) ]]  \
    [[kernel]] void indexSelect(const device type1* src      [[buffer(0)]], \
                                  device type1* dst            [[buffer(1)]], \
                                  const device type2* indices  [[buffer(2)]], \
                                  constant type3& indicesSize  [[buffer(3)]], \
                                  constant type3& dimSize      [[buffer(4)]], \
                                  constant type3& sliceSize    [[buffer(5)]], \
                                  const constant type3* srcLayout [[buffer(6)]], \
                                  const constant type3* indicesLayout [[buffer(7)]], \
                                  uint index [[thread_position_in_grid]])

SpecializeIndexSelect("f32",  float , int, size_t);
SpecializeIndexSelect("f16",  half  , int, size_t);
SpecializeIndexSelect("bf16", bfloat, int, size_t);
SpecializeIndexSelect("i64",  long  , int, size_t);
SpecializeIndexSelect("i32",  int   , int, size_t);
SpecializeIndexSelect("i16",  short , int, size_t);
SpecializeIndexSelect("i8",   char  , int, size_t);
SpecializeIndexSelect("ui8",  uchar , int, size_t);


// Index Add
// -----------------------------------------------------------------
#define SpecializeIndexAdd(tname, type1, type2, type3)  \
    template [[ host_name("indexAdd_" tname) ]]  \
    [[kernel]] void indexAdd(const device type1* src      [[buffer(0)]], \
                               device type1* dst            [[buffer(1)]], \
                               const device type2* indices  [[buffer(2)]], \
                               constant type3& indicesSize  [[buffer(3)]], \
                               constant type3& dimSize      [[buffer(4)]], \
                               constant type3& sliceSize    [[buffer(5)]], \
                               const constant type3* srcLayout [[buffer(6)]], \
                               const constant type3* dstLayout [[buffer(7)]], \
                               const constant type3* indicesLayout [[buffer(8)]], \
                               uint index [[thread_position_in_grid]])

#define ImplementSpecializedIndexAdd(tname, type1, type2, type3)  \
    template <> [[ host_name("indexAdd_" tname) ]]  \
    [[kernel]] void indexAdd<type1,type2,type3>(const device type1* src      [[buffer(0)]], \
                                                   device type1* dst            [[buffer(1)]], \
                                                   const device type2* indices  [[buffer(2)]], \
                                                   constant type3& indicesSize  [[buffer(3)]], \
                                                   constant type3& dimSize      [[buffer(4)]], \
                                                   constant type3& sliceSize    [[buffer(5)]], \
                                                   const constant type3* srcLayout [[buffer(6)]], \
                                                   const constant type3* dstLayout [[buffer(7)]], \
                                                   const constant type3* indicesLayout [[buffer(8)]], \
                                                   uint index [[thread_position_in_grid]]) { }

SpecializeIndexAdd("f32",  float , int, size_t);
SpecializeIndexAdd("i32",  int   , int, size_t);
ImplementSpecializedIndexAdd("f16",  half  , int, size_t);
ImplementSpecializedIndexAdd("bf16", bfloat, int, size_t);
ImplementSpecializedIndexAdd("i64",  long  , int, size_t);
ImplementSpecializedIndexAdd("i16",  short , int, size_t);
ImplementSpecializedIndexAdd("i8",   char  , int, size_t);
ImplementSpecializedIndexAdd("ui8",  uchar , int, size_t);

)";


}   // namespace
