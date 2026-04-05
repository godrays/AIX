//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "aix.hpp"
// External includes
// System includes
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace aix
{

static size_t translationIndex(size_t index, const Shape& shape, const Shape& newShape);
static size_t flattenIndex(const Stride& indices, const Stride& strides);
static Stride unflattenIndex(size_t index, const Stride& strides);
static size_t physicalIndex(size_t flatIndex, size_t offset, const Shape& shape, const Stride& strides);

Device::~Device() = default;

template <typename T>
static void addGeneric(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a1.data);
    auto t2  = static_cast<const T*>(a2.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < result.size; ++i)
    {
        res[i] = t1[physicalIndex(i, a1.offset, a1.shape, a1.strides)]
               + t2[physicalIndex(i, a2.offset, a2.shape, a2.strides)];
    }
}

template <typename T>
static void subGeneric(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a1.data);
    auto t2  = static_cast<const T*>(a2.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < result.size; ++i)
    {
        res[i] = t1[physicalIndex(i, a1.offset, a1.shape, a1.strides)]
               - t2[physicalIndex(i, a2.offset, a2.shape, a2.strides)];
    }
}

template <typename T>
static void mulGeneric(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a1.data);
    auto t2  = static_cast<const T*>(a2.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < result.size; ++i)
    {
        res[i] = t1[physicalIndex(i, a1.offset, a1.shape, a1.strides)]
               * t2[physicalIndex(i, a2.offset, a2.shape, a2.strides)];
    }
}

template <typename T>
static void divGeneric(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a1.data);
    auto t2  = static_cast<const T*>(a2.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < result.size; ++i)
    {
        res[i] = t1[physicalIndex(i, a1.offset, a1.shape, a1.strides)]
               / t2[physicalIndex(i, a2.offset, a2.shape, a2.strides)];
    }
}

template <typename T>
static void unaryGeneric(const DeviceTensorParams& a1, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a1.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a1.size; ++i)
    {
        res[i] = -t1[physicalIndex(i, a1.offset, a1.shape, a1.strides)];
    }
}

template <typename SrcType, typename DstType>
static DstType convertGenericValue(SrcType value)
{
    constexpr bool isFloatLikeSrc = std::is_floating_point_v<SrcType>  ||
                                    std::is_same_v<SrcType, float16_t> ||
                                    std::is_same_v<SrcType, bfloat16_t>;

    if constexpr (isFloatLikeSrc && std::is_integral_v<DstType>)
    {
        auto converted = static_cast<long double>(value);
        if (std::isnan(converted))
        {
            return 0;
        }

        converted = std::clamp(converted, static_cast<long double>(std::numeric_limits<DstType>::lowest()),
                                          static_cast<long double>(std::numeric_limits<DstType>::max()));
        return static_cast<DstType>(converted);
    }

    return static_cast<DstType>(value);
}

template <typename SrcType, typename DstType>
static void fillGeneric(const void* scalar, const DeviceTensorParams& result)
{
    auto tSrc = static_cast<const SrcType*>(scalar);
    auto tDst = static_cast<DstType*>(result.data);
    for (size_t i=0; i<result.size; ++i)
    {
        tDst[i] = convertGenericValue<SrcType, DstType>(tSrc[0]);
    }
}

template <typename T>
static void fillMinGeneric(const DeviceTensorParams& result)
{
    auto tDst = static_cast<T*>(result.data);
    for (size_t i=0; i<result.size; ++i)
    {
        tDst[i] = std::numeric_limits<T>::lowest();
    }
}

template <typename T>
static void sumGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    T sum = 0;
    for (size_t i = 0; i < a.size; ++i)
    {
        sum += t1[physicalIndex(i, a.offset, a.shape, a.strides)];
    }
    *res = sum;
}

template <typename T>
static void sqrtGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        auto value = std::sqrt(static_cast<long double>(t1[physicalIndex(i, a.offset, a.shape, a.strides)]));
        res[i] = convertGenericValue<long double, T>(value);
    }
}

template <typename T>
static void sinGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        res[i] = std::sin(t1[physicalIndex(i, a.offset, a.shape, a.strides)]);
    }
}

template <typename T>
static void cosGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        res[i] = std::cos(t1[physicalIndex(i, a.offset, a.shape, a.strides)]);
    }
}

template <typename T>
static void tanhGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        res[i] = std::tanh(t1[physicalIndex(i, a.offset, a.shape, a.strides)]);
    }
}

template <typename T>
static void logGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        auto value = std::log(static_cast<long double>(t1[physicalIndex(i, a.offset, a.shape, a.strides)]));
        res[i] = convertGenericValue<long double, T>(value);
    }
}

template <typename T>
static void expGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        auto value = std::exp(static_cast<long double>(t1[physicalIndex(i, a.offset, a.shape, a.strides)]));
        res[i] = convertGenericValue<long double, T>(value);
    }
}

template <typename T>
static void powGeneric(const DeviceTensorParams& a, const DeviceTensorParams& exp, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto t2  = static_cast<const T*>(exp.data);
    auto res = static_cast<T*>(result.data);

    for (size_t i = 0; i < a.size; ++i)
    {
        auto value = std::pow(static_cast<long double>(t1[physicalIndex(i, a.offset, a.shape, a.strides)]),
                              static_cast<long double>(t2[physicalIndex(i, exp.offset, exp.shape, exp.strides)]));
        res[i] = convertGenericValue<long double, T>(value);
    }
}

template <typename T>
static void maxGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    res[0] = t[physicalIndex(0, a.offset, a.shape, a.strides)];
    for (size_t i = 1; i < a.size; ++i)
    {
        res[0] = std::max<T>(res[0], t[physicalIndex(i, a.offset, a.shape, a.strides)]);
    }
}

template <typename T, typename T2>
static void argmaxGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t  = static_cast<const T*>(a.data);
    auto res = static_cast<T2*>(result.data);

    T max = t[physicalIndex(0, a.offset, a.shape, a.strides)];
    res[0] = 0;
    for (size_t i = 1; i < a.size; ++i)
    {
        if (t[physicalIndex(i, a.offset, a.shape, a.strides)] > max)
        {
            max = t[physicalIndex(i, a.offset, a.shape, a.strides)];
            res[0] = i;
        }
    }
}

template <typename T, typename T2>
static void argmaxToGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T2*>(dst.data);
    auto tmaxTemp = new T[dst.size];         // Temporary helper buffer to store max values for comparison.
    auto tInitialized = new bool[dst.size]{};

    // Initialize the temp buffer with the lowest value of the data type, T.
    fillMinGeneric<T>({ .data=tmaxTemp, .size=dst.size });

    for (size_t index = 0; index < src.size; ++index)
    {
        auto transIndex = translationIndex(index, dst.shape, src.shape);
        if (!tInitialized[transIndex] || tSrc[physicalIndex(index, src.offset, src.shape, src.strides)] > tmaxTemp[transIndex])
        {
            tInitialized[transIndex] = true;
            tmaxTemp[transIndex] = tSrc[physicalIndex(index, src.offset, src.shape, src.strides)];
            tDst[transIndex] = (index / src.strides[dim]) % src.shape[dim];
        }
    }
    delete [] tmaxTemp;
    delete [] tInitialized;
}

template <typename T, typename T2>
static void argmaxIndicesGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto t  = static_cast<const T*>(a.data);
    auto res = static_cast<T2*>(result.data);

    T max = t[physicalIndex(0, a.offset, a.shape, a.strides)];
    T2 index = res[0] = 0;
    for (size_t i = 1; i < a.size; ++i)
    {
        res[i] = 0;
        if (t[physicalIndex(i, a.offset, a.shape, a.strides)] > max)
        {
            max = t[physicalIndex(i, a.offset, a.shape, a.strides)];
            index = i;
        }
    }
    res[index] = 1;
}

template <typename T, typename T2>
static void argmaxIndicesToGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T2*>(dst.data);
    auto dstShape = dst.shape;
    dstShape[dim] = 1;
    size_t maxElementCount = 1;
    for (auto i : dstShape)
    {
        maxElementCount *= i;
    }

    auto tmaxTemp = new T[maxElementCount];     // Temporary helper buffer to store max values for comparison.
    auto tInitialized = new bool[maxElementCount]{};

    // Initialize the temp buffer with the lowest value of the data type, T.
    fillMinGeneric<T>({ .data=tmaxTemp, .size=maxElementCount });

    auto tDstTemp = new T2[maxElementCount];   // Temporary helper buffer to store index of max elements.

    for (size_t index = 0; index < src.size; ++index)
    {
        auto transIndex = translationIndex(index, dstShape, src.shape);
        if (!tInitialized[transIndex] || tSrc[physicalIndex(index, src.offset, src.shape, src.strides)] > tmaxTemp[transIndex])
        {
            tInitialized[transIndex] = true;
            tmaxTemp[transIndex] = tSrc[physicalIndex(index, src.offset, src.shape, src.strides)];
            tDstTemp[transIndex] = index;
        }
    }

    for (size_t i = 0; i < maxElementCount; ++i)
    {
        tDst[tDstTemp[i]] = 1;
    }

    delete [] tmaxTemp;
    delete [] tInitialized;
    delete [] tDstTemp;
}

template <typename T>
static void matmulGeneric(const DeviceTensorParams& a, const DeviceTensorParams& b, const DeviceTensorParams& result)
{
    auto t1  = static_cast<const T*>(a.data);
    auto t2  = static_cast<const T*>(b.data);
    auto res = static_cast<T*>(result.data);

    // NOTE: Since TensorValue validated the parameters, device method do not validate again.
    size_t m = a.shape[0];      // Rows of the first matrix
    size_t n = b.shape[1];      // Columns of the second matrix
    size_t inner = a.shape[1];  // Inner dimension

    // Perform matrix multiplication
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                size_t aIdx = physicalIndex(i * inner + k, a.offset, a.shape, a.strides);
                size_t bIdx = physicalIndex(k * n + j, b.offset, b.shape, b.strides);
                sum += t1[aIdx] * t2[bIdx];
            }
            res[i * n + j] = sum;
        }
    }
}

template <typename T>
static void transposeGeneric(const DeviceTensorParams& a, const DeviceTensorParams& result, size_t dim0, size_t dim1)
{
    auto t1  = static_cast<const T*>(a.data);
    auto res = static_cast<T*>(result.data);

    // Perform the generalized transpose operation.
    for (size_t i=0; i<a.size; ++i)
    {
        auto oldIndices = unflattenIndex(i, a.strides);
        std::swap(oldIndices[dim0], oldIndices[dim1]);
        size_t newIndex = flattenIndex(oldIndices, result.strides);
        res[newIndex] = t1[i];
    }
}

template <typename SrcType, typename DstType>
static void copyGeneric(const void* src, void* dst, size_t size)
{
    if constexpr (std::is_same_v<SrcType, DstType>)
    {
        std::memcpy(dst, src, size * sizeof(SrcType));
    }
    else
    {
        auto tSrc = static_cast<const SrcType*>(src);
        auto tDst = static_cast<DstType*>(dst);
        for (size_t i=0; i<size; ++i)
        {
            tDst[i] = convertGenericValue<SrcType, DstType>(tSrc[i]);
        }
    }
}

template <typename T>
static void contiguousGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);

    for (size_t i=0; i<dst.size; ++i)
    {
        size_t ofs = physicalIndex(i, src.offset, src.shape, src.strides);

        // Copy the element from non-contiguous source to contiguous destination.
        tDst[i] = tSrc[ofs];
    }
}

template <typename T>
static void reduceToGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);

    // Sum the values from the broadcasted tensor to the original tensor shape. The reduction involves summation
    // because each element of the original tensor is used multiple times in the broadcasted operation.
    // Summing the gradients correctly aggregates these contributions.
    for (size_t index = 0; index < src.size; ++index)
    {
        tDst[translationIndex(index, dst.shape, src.shape)] += tSrc[physicalIndex(index, src.offset, src.shape, src.strides)];
    }
}

template <typename T>
static void maxToGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);

    for (size_t index = 0; index < src.size; ++index)
    {
        auto transIndex = translationIndex(index, dst.shape, src.shape);
        tDst[transIndex] = std::max<T>(tDst[transIndex], tSrc[physicalIndex(index, src.offset, src.shape, src.strides)]);
    }
}

template <typename T>
static void sliceSetGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                            size_t dim, size_t start, size_t end, size_t step)
{
    auto tSrc = static_cast<T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);
    auto newShape = dst.shape;
    newShape[dim] = (end - start + step - 1) / step;    // This computes the size along the slicing dimension.

    for (size_t index = 0; index < src.size; ++index)
    {
        // Translate the flat index into multi-dimensional indices.
        size_t dstIndex = index;
        size_t srcIndex = 0;

        for (ssize_t i = static_cast<ssize_t>(dst.shape.size()) - 1; i >= 0; --i)
        {
            size_t coordinate = dstIndex % newShape[i];
            dstIndex /= newShape[i];

            if (i == static_cast<ssize_t>(dim))   // Handle the slicing dimension.
                srcIndex += (start + coordinate * step) * dst.strides[i];
            else
                srcIndex += coordinate * dst.strides[i];
        }

        tDst[srcIndex] = tSrc[physicalIndex(index, src.offset, src.shape, src.strides)];
    }
}

template <typename T, typename T2>
static void indexSelectGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                               const DeviceTensorParams& indices, size_t dim)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);
    auto tIndices = static_cast<const T2*>(indices.data);

    // Calculate the number of elements in one slice after the specified dimension.
    size_t sliceSize = 1;
    for (size_t i = dim + 1; i < src.shape.size(); ++i)
    {
        sliceSize *= src.shape[i];
    }

    // Calculate the size of one entire slice for the dimension in question.
    size_t dimSize = !src.shape.empty() ? src.shape[dim] * sliceSize : 0;

    for (size_t index=0; index<dst.size; ++index)
    {
        // Calculate the outer loop index, index position, and element within the slice.
        size_t elementWithinSlice = index % sliceSize;
        size_t idx = (index / sliceSize) % indices.size;
        size_t outer = index / (indices.size * sliceSize);

        size_t srcIndex = tIndices[idx] * sliceSize + elementWithinSlice;
        size_t srcOffset = outer * dimSize + srcIndex;
        size_t dstOffset = outer * indices.size * sliceSize + idx * sliceSize + elementWithinSlice;

        // Perform the copy operation.
        tDst[dstOffset] = tSrc[physicalIndex(srcOffset, src.offset, src.shape, src.strides)];
    }
}

template <typename T, typename T2>
static void indexAddGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                            const DeviceTensorParams& indices, size_t dim)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);
    auto tIndices = static_cast<const T2*>(indices.data);

    // Calculate the number of elements in one slice after the specified dimension.
    size_t sliceSize = 1;
    for (size_t i = dim + 1; i < dst.shape.size(); ++i)
    {
        sliceSize *= dst.shape[i];
    }

    // Calculate the size of one entire slice for the dimension in question.
    size_t dimSize = !dst.shape.empty() ? dst.shape[dim] * sliceSize : 0;

    for (size_t index = 0; index < src.size; ++index)
    {
        // Calculate the outer loop index, index position, and element within the slice.
        size_t elementWithinSlice = index % sliceSize;
        size_t idx = (index / sliceSize) % indices.size;
        size_t outer = index / (indices.size * sliceSize);

        size_t dstIndex = tIndices[idx] * sliceSize + elementWithinSlice;
        size_t dstOffset = outer * dimSize + dstIndex;
        size_t srcOffset = outer * indices.size * sliceSize + idx * sliceSize + elementWithinSlice;

        // Perform the addition operation.
        tDst[dstOffset] += tSrc[physicalIndex(srcOffset, src.offset, src.shape, src.strides)];
    }
}

template <typename T>
static void trilGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst, ssize_t diagonal)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);

    size_t shapeSize = src.shape.size();
    size_t rows = src.shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
    size_t cols = src.shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

    for (size_t i = 0; i < dst.size; ++i)
    {
        // Calculate the row and column indices for the last 2-dim slice.
        size_t row = (i / cols) % rows;
        size_t col = i % cols;
        size_t dstOffset = physicalIndex(i, dst.offset, dst.shape, dst.strides);

        // Zero out the elements above the specified diagonal.
        if (static_cast<ssize_t>(col) > static_cast<ssize_t>(row) + diagonal)
        {
            tDst[dstOffset] = 0;
        }
        else
        {
            tDst[dstOffset] = tSrc[physicalIndex(i, src.offset, src.shape, src.strides)];
        }
    }
}

template <typename T>
static void triuGeneric(const DeviceTensorParams& src, const DeviceTensorParams& dst, ssize_t diagonal)
{
    auto tSrc = static_cast<const T*>(src.data);
    auto tDst = static_cast<T*>(dst.data);

    size_t shapeSize = src.shape.size();
    size_t rows = src.shape[shapeSize - 2];      // Rows in the last 2-dim tensor.
    size_t cols = src.shape[shapeSize - 1];      // Columns in the last 2-dim tensor.

    for (size_t i = 0; i < dst.size; ++i)
    {
        // Calculate the row and column indices for the last 2-dim slice.
        size_t row = (i / cols) % rows;
        size_t col = i % cols;
        size_t dstOffset = physicalIndex(i, dst.offset, dst.shape, dst.strides);

        // Zero out the elements above the specified diagonal.
        if (static_cast<ssize_t>(col) < static_cast<ssize_t>(row) + diagonal)
        {
            tDst[dstOffset] = 0;
        }
        else
        {
            tDst[dstOffset] = tSrc[physicalIndex(i, src.offset, src.shape, src.strides)];
        }
    }
}

// Helper Methods

// Calculate the translation index, originalIndex, to copy data from the original index to the new index.
static size_t translationIndex(size_t index, const Shape& shape, const Shape& newShape)
{
    size_t originalIndex  = 0;
    size_t targetStride   = 1;
    size_t originalStride = 1;

    for (ssize_t i = newShape.size() - 1, j = shape.size() - 1; i >= 0; --i)
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

static size_t flattenIndex(const Stride& indices, const Stride& strides)
{
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        index += indices[i] * strides[i];
    }
    return index;
}

static Stride unflattenIndex(size_t index, const Stride& strides)
{
    Stride indices(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
    {
        indices[i] = index / strides[i];
        index %= strides[i];
    }
    return indices;
}

static size_t physicalIndex(size_t flatIndex, size_t offset, const Shape& shape, const Stride& strides)
{
    size_t idx = flatIndex;
    size_t ofs = offset;
    for (ssize_t dim = static_cast<ssize_t>(shape.size()) - 1; dim >= 0; --dim)
    {
        auto dimIndex = idx % shape[dim];
        idx /= shape[dim];
        ofs += dimIndex * strides[dim];
    }
    return ofs;
}

size_t Device::dataTypeSize(DataType dtype)
{
    static const size_t dTypeSizeTable[DataTypeCount]
    {
        sizeof(double    ),   // kFloat64
        sizeof(float     ),   // kFloat32
        sizeof(float16_t ),   // kFloat16
        sizeof(bfloat16_t),   // kBFloat16
        sizeof(int64_t   ),   // kInt64
        sizeof(int32_t   ),   // kInt32
        sizeof(int16_t   ),   // kInt16
        sizeof(int8_t    ),   // kInt8
        sizeof(uint8_t   ),   // kUInt8
    };
    return dTypeSizeTable[static_cast<size_t>(dtype)];
}


DeviceCPU::DeviceCPU([[maybe_unused]] size_t deviceIndex)
{
}

DeviceType DeviceCPU::type() const
{
    return DeviceType::kCPU;
}

std::string DeviceCPU::name() const
{
    return "CPU";
}

void* DeviceCPU::allocate(size_t size)
{
    return std::malloc(size);
}

void* DeviceCPU::allocate(size_t size, DataType dtype)
{
    return allocate(size * dataTypeSize(dtype));
}

void DeviceCPU::deallocate(void * memory)
{
    return std::free(memory);
}

void DeviceCPU::add(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        addGeneric<double    >,
        addGeneric<float     >,
        addGeneric<float16_t >,
        addGeneric<bfloat16_t>,
        addGeneric<int64_t   >,
        addGeneric<int32_t   >,
        addGeneric<int16_t   >,
        addGeneric<int8_t    >,
        addGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a1, a2, result);
}

void DeviceCPU::sub(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        subGeneric<double    >,
        subGeneric<float     >,
        subGeneric<float16_t >,
        subGeneric<bfloat16_t>,
        subGeneric<int64_t   >,
        subGeneric<int32_t   >,
        subGeneric<int16_t   >,
        subGeneric<int8_t    >,
        subGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a1, a2, result);
}

void DeviceCPU::mul(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        mulGeneric<double    >,
        mulGeneric<float     >,
        mulGeneric<float16_t >,
        mulGeneric<bfloat16_t>,
        mulGeneric<int64_t   >,
        mulGeneric<int32_t   >,
        mulGeneric<int16_t   >,
        mulGeneric<int8_t    >,
        mulGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a1, a2, result);
}

void DeviceCPU::div(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        divGeneric<double    >,
        divGeneric<float     >,
        divGeneric<float16_t >,
        divGeneric<bfloat16_t>,
        divGeneric<int64_t   >,
        divGeneric<int32_t   >,
        divGeneric<int16_t   >,
        divGeneric<int8_t    >,
        divGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a1, a2, result);
}

void DeviceCPU::unary(const DeviceTensorParams& a1, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        unaryGeneric<double    >,
        unaryGeneric<float     >,
        unaryGeneric<float16_t >,
        unaryGeneric<bfloat16_t>,
        unaryGeneric<int64_t   >,
        unaryGeneric<int32_t   >,
        unaryGeneric<int16_t   >,
        unaryGeneric<int8_t    >,
        unaryGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a1, result);
}

void DeviceCPU::fill(const void* scalar, DataType scalarDType, const DeviceTensorParams& result)
{
    // Define a function pointer type for the conversion copy functions.
    using fillFunc = void (*)(const void*, const DeviceTensorParams&);

    // Create a lookup table of the functions.
    static const fillFunc funcTable[DataTypeCount][DataTypeCount] =
    {
        { fillGeneric<double, double>,     fillGeneric<double, float>,     fillGeneric<double, float16_t>,     fillGeneric<double, bfloat16_t>,     fillGeneric<double, int64_t>,     fillGeneric<double, int32_t>,     fillGeneric<double, int16_t>,     fillGeneric<double, int8_t>,     fillGeneric<double, uint8_t>     },
        { fillGeneric<float, double>,      fillGeneric<float, float>,      fillGeneric<float, float16_t>,      fillGeneric<float, bfloat16_t>,      fillGeneric<float, int64_t>,      fillGeneric<float, int32_t>,      fillGeneric<float, int16_t>,      fillGeneric<float, int8_t>,      fillGeneric<float, uint8_t>      },
        { fillGeneric<float16_t, double>,  fillGeneric<float16_t, float>,  fillGeneric<float16_t, float16_t>,  fillGeneric<float16_t, bfloat16_t>,  fillGeneric<float16_t, int64_t>,  fillGeneric<float16_t, int32_t>,  fillGeneric<float16_t, int16_t>,  fillGeneric<float16_t, int8_t>,  fillGeneric<float16_t, uint8_t>  },
        { fillGeneric<bfloat16_t, double>, fillGeneric<bfloat16_t, float>, fillGeneric<bfloat16_t, float16_t>, fillGeneric<bfloat16_t, bfloat16_t>, fillGeneric<bfloat16_t, int64_t>, fillGeneric<bfloat16_t, int32_t>, fillGeneric<bfloat16_t, int16_t>, fillGeneric<bfloat16_t, int8_t>, fillGeneric<bfloat16_t, uint8_t> },
        { fillGeneric<int64_t, double>,    fillGeneric<int64_t, float>,    fillGeneric<int64_t, float16_t>,    fillGeneric<int64_t, bfloat16_t>,    fillGeneric<int64_t, int64_t>,    fillGeneric<int64_t, int32_t>,    fillGeneric<int64_t, int16_t>,    fillGeneric<int64_t, int8_t>,    fillGeneric<int64_t, uint8_t>    },
        { fillGeneric<int32_t, double>,    fillGeneric<int32_t, float>,    fillGeneric<int32_t, float16_t>,    fillGeneric<int32_t, bfloat16_t>,    fillGeneric<int32_t, int64_t>,    fillGeneric<int32_t, int32_t>,    fillGeneric<int32_t, int16_t>,    fillGeneric<int32_t, int8_t>,    fillGeneric<int32_t, uint8_t>    },
        { fillGeneric<int16_t, double>,    fillGeneric<int16_t, float>,    fillGeneric<int16_t, float16_t>,    fillGeneric<int16_t, bfloat16_t>,    fillGeneric<int16_t, int64_t>,    fillGeneric<int16_t, int32_t>,    fillGeneric<int16_t, int16_t>,    fillGeneric<int16_t, int8_t>,    fillGeneric<int16_t, uint8_t>    },
        { fillGeneric<int8_t, double>,     fillGeneric<int8_t, float>,     fillGeneric<int8_t,  float16_t>,    fillGeneric<int8_t,  bfloat16_t>,    fillGeneric<int8_t, int64_t>,     fillGeneric<int8_t, int32_t>,     fillGeneric<int8_t, int16_t>,     fillGeneric<int8_t, int8_t>,     fillGeneric<int8_t, uint8_t>     },
        { fillGeneric<uint8_t, double>,    fillGeneric<uint8_t, float>,    fillGeneric<uint8_t, float16_t>,    fillGeneric<uint8_t, bfloat16_t>,    fillGeneric<uint8_t, int64_t>,    fillGeneric<uint8_t, int32_t>,    fillGeneric<uint8_t, int16_t>,    fillGeneric<uint8_t, int8_t>,    fillGeneric<uint8_t, uint8_t>    },
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(scalarDType)][static_cast<size_t>(result.dtype)](scalar, result);
}

void DeviceCPU::fillMin(const DeviceTensorParams& result)
{
    // Create a lookup table of the functions.
    static const auto funcTable = std::array
    {
        fillMinGeneric<double    >,
        fillMinGeneric<float     >,
        fillMinGeneric<float16_t >,
        fillMinGeneric<bfloat16_t>,
        fillMinGeneric<int64_t   >,
        fillMinGeneric<int32_t   >,
        fillMinGeneric<int16_t   >,
        fillMinGeneric<int8_t    >,
        fillMinGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](result);
}

void DeviceCPU::sum(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        sumGeneric<double    >,
        sumGeneric<float     >,
        sumGeneric<float16_t >,
        sumGeneric<bfloat16_t>,
        sumGeneric<int64_t   >,
        sumGeneric<int32_t   >,
        sumGeneric<int16_t   >,
        sumGeneric<int8_t    >,
        sumGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::sqrt(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        sqrtGeneric<double    >,
        sqrtGeneric<float     >,
        sqrtGeneric<float16_t >,
        sqrtGeneric<bfloat16_t>,
        sqrtGeneric<int64_t   >,
        sqrtGeneric<int32_t   >,
        sqrtGeneric<int16_t   >,
        sqrtGeneric<int8_t    >,
        sqrtGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::sin(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        sinGeneric<double    >,
        sinGeneric<float     >,
        sinGeneric<float16_t >,
        sinGeneric<bfloat16_t>,
        sinGeneric<int64_t   >,
        sinGeneric<int32_t   >,
        sinGeneric<int16_t   >,
        sinGeneric<int8_t    >,
        sinGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::cos(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        cosGeneric<double    >,
        cosGeneric<float     >,
        cosGeneric<float16_t >,
        cosGeneric<bfloat16_t>,
        cosGeneric<int64_t   >,
        cosGeneric<int32_t   >,
        cosGeneric<int16_t   >,
        cosGeneric<int8_t    >,
        cosGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::tanh(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        tanhGeneric<double    >,
        tanhGeneric<float     >,
        tanhGeneric<float16_t >,
        tanhGeneric<bfloat16_t>,
        tanhGeneric<int64_t   >,
        tanhGeneric<int32_t   >,
        tanhGeneric<int16_t   >,
        tanhGeneric<int8_t    >,
        tanhGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::log(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        logGeneric<double    >,
        logGeneric<float     >,
        logGeneric<float16_t >,
        logGeneric<bfloat16_t>,
        logGeneric<int64_t   >,
        logGeneric<int32_t   >,
        logGeneric<int16_t   >,
        logGeneric<int8_t    >,
        logGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::exp(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        expGeneric<double    >,
        expGeneric<float     >,
        expGeneric<float16_t >,
        expGeneric<bfloat16_t>,
        expGeneric<int64_t   >,
        expGeneric<int32_t   >,
        expGeneric<int16_t   >,
        expGeneric<int8_t    >,
        expGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::pow(const DeviceTensorParams& a, const DeviceTensorParams& exp, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        powGeneric<double    >,
        powGeneric<float     >,
        powGeneric<float16_t >,
        powGeneric<bfloat16_t>,
        powGeneric<int64_t   >,
        powGeneric<int32_t   >,
        powGeneric<int16_t   >,
        powGeneric<int8_t    >,
        powGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, exp, result);
}

void DeviceCPU::max(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        maxGeneric<double    >,
        maxGeneric<float     >,
        maxGeneric<float16_t >,
        maxGeneric<bfloat16_t>,
        maxGeneric<int64_t   >,
        maxGeneric<int32_t   >,
        maxGeneric<int16_t   >,
        maxGeneric<int8_t    >,
        maxGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result);
}

void DeviceCPU::argmax(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    if (result.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceCPU::argmax supports only int32 data type for its result.");
    }

    static const auto funcTable = std::array
    {
        argmaxGeneric<double    , int32_t>,
        argmaxGeneric<float     , int32_t>,
        argmaxGeneric<float16_t , int32_t>,
        argmaxGeneric<bfloat16_t, int32_t>,
        argmaxGeneric<int64_t   , int32_t>,
        argmaxGeneric<int32_t   , int32_t>,
        argmaxGeneric<int16_t   , int32_t>,
        argmaxGeneric<int8_t    , int32_t>,
        argmaxGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(a.dtype)](a, result);
}

void DeviceCPU::argmaxIndices(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    if (result.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceCPU::argmaxIndices supports only int32 data type for its result.");
    }

    static const auto funcTable = std::array
    {
        argmaxIndicesGeneric<double    , int32_t>,
        argmaxIndicesGeneric<float     , int32_t>,
        argmaxIndicesGeneric<float16_t , int32_t>,
        argmaxIndicesGeneric<bfloat16_t, int32_t>,
        argmaxIndicesGeneric<int64_t   , int32_t>,
        argmaxIndicesGeneric<int32_t   , int32_t>,
        argmaxIndicesGeneric<int16_t   , int32_t>,
        argmaxIndicesGeneric<int8_t    , int32_t>,
        argmaxIndicesGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(a.dtype)](a, result);
}

void DeviceCPU::matmul(const DeviceTensorParams& a, const DeviceTensorParams& b, const DeviceTensorParams& result)
{
    static const auto funcTable = std::array
    {
        matmulGeneric<double    >,
        matmulGeneric<float     >,
        matmulGeneric<float16_t >,
        matmulGeneric<bfloat16_t>,
        matmulGeneric<int64_t   >,
        matmulGeneric<int32_t   >,
        matmulGeneric<int16_t   >,
        matmulGeneric<int8_t    >,
        matmulGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, b, result);
}

void DeviceCPU::transpose(const DeviceTensorParams& a, const DeviceTensorParams& result, size_t dim0, size_t dim1)
{
    static const auto funcTable = std::array
    {
        transposeGeneric<double    >,
        transposeGeneric<float     >,
        transposeGeneric<float16_t >,
        transposeGeneric<bfloat16_t>,
        transposeGeneric<int64_t   >,
        transposeGeneric<int32_t   >,
        transposeGeneric<int16_t   >,
        transposeGeneric<int8_t    >,
        transposeGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(result.dtype)](a, result, dim0, dim1);
}

void DeviceCPU::copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    // Define a function pointer type for the conversion copy functions.
    using copyFunc = void (*)(const void*, void*, size_t);

    // Create a lookup table of the functions.
    static const copyFunc funcTable[DataTypeCount][DataTypeCount] =
    {
        { copyGeneric<double, double>,     copyGeneric<double, float>,     copyGeneric<double, float16_t>,     copyGeneric<double, bfloat16_t>,     copyGeneric<double, int64_t>,     copyGeneric<double, int32_t>,     copyGeneric<double, int16_t>,     copyGeneric<double, int8_t>,     copyGeneric<double, uint8_t>     },
        { copyGeneric<float, double>,      copyGeneric<float, float>,      copyGeneric<float, float16_t>,      copyGeneric<float, bfloat16_t>,      copyGeneric<float, int64_t>,      copyGeneric<float, int32_t>,      copyGeneric<float, int16_t>,      copyGeneric<float, int8_t>,      copyGeneric<float, uint8_t>      },
        { copyGeneric<float16_t, double>,  copyGeneric<float16_t, float>,  copyGeneric<float16_t, float16_t>,  copyGeneric<float16_t, bfloat16_t>,  copyGeneric<float16_t, int64_t>,  copyGeneric<float16_t, int32_t>,  copyGeneric<float16_t, int16_t>,  copyGeneric<float16_t, int8_t>,  copyGeneric<float16_t, uint8_t>  },
        { copyGeneric<bfloat16_t, double>, copyGeneric<bfloat16_t, float>, copyGeneric<bfloat16_t, float16_t>, copyGeneric<bfloat16_t, bfloat16_t>, copyGeneric<bfloat16_t, int64_t>, copyGeneric<bfloat16_t, int32_t>, copyGeneric<bfloat16_t, int16_t>, copyGeneric<bfloat16_t, int8_t>, copyGeneric<bfloat16_t, uint8_t> },
        { copyGeneric<int64_t, double>,    copyGeneric<int64_t, float>,    copyGeneric<int64_t, float16_t>,    copyGeneric<int64_t, bfloat16_t>,    copyGeneric<int64_t, int64_t>,    copyGeneric<int64_t, int32_t>,    copyGeneric<int64_t, int16_t>,    copyGeneric<int64_t, int8_t>,    copyGeneric<int64_t, uint8_t>    },
        { copyGeneric<int32_t, double>,    copyGeneric<int32_t, float>,    copyGeneric<int32_t, float16_t>,    copyGeneric<int32_t, bfloat16_t>,    copyGeneric<int32_t, int64_t>,    copyGeneric<int32_t, int32_t>,    copyGeneric<int32_t, int16_t>,    copyGeneric<int32_t, int8_t>,    copyGeneric<int32_t, uint8_t>    },
        { copyGeneric<int16_t, double>,    copyGeneric<int16_t, float>,    copyGeneric<int16_t, float16_t>,    copyGeneric<int16_t, bfloat16_t>,    copyGeneric<int16_t, int64_t>,    copyGeneric<int16_t, int32_t>,    copyGeneric<int16_t, int16_t>,    copyGeneric<int16_t, int8_t>,    copyGeneric<int16_t, uint8_t>    },
        { copyGeneric<int8_t, double>,     copyGeneric<int8_t, float>,     copyGeneric<int8_t,  float16_t>,    copyGeneric<int8_t,  bfloat16_t>,    copyGeneric<int8_t, int64_t>,     copyGeneric<int8_t, int32_t>,     copyGeneric<int8_t, int16_t>,     copyGeneric<int8_t, int8_t>,     copyGeneric<int8_t, uint8_t>     },
        { copyGeneric<uint8_t, double>,    copyGeneric<uint8_t, float>,    copyGeneric<uint8_t, float16_t>,    copyGeneric<uint8_t, bfloat16_t>,    copyGeneric<uint8_t, int64_t>,    copyGeneric<uint8_t, int32_t>,    copyGeneric<uint8_t, int16_t>,    copyGeneric<uint8_t, int8_t>,    copyGeneric<uint8_t, uint8_t>    },
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(srcDType)][static_cast<size_t>(dstDType)](src, dst, size);
}

void DeviceCPU::copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    copy(src, srcDType, dst, dstDType, size);
    synchronize();    // This call has no effect, but it shows the difference between copy and copyImmediate.
}

void DeviceCPU::contiguous(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    static const auto funcTable = std::array
    {
        contiguousGeneric<double    >,
        contiguousGeneric<float     >,
        contiguousGeneric<float16_t >,
        contiguousGeneric<bfloat16_t>,
        contiguousGeneric<int64_t   >,
        contiguousGeneric<int32_t   >,
        contiguousGeneric<int16_t   >,
        contiguousGeneric<int8_t    >,
        contiguousGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst);
}

void DeviceCPU::reduceTo(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    static const auto funcTable = std::array
    {
        reduceToGeneric<double    >,
        reduceToGeneric<float     >,
        reduceToGeneric<float16_t >,
        reduceToGeneric<bfloat16_t>,
        reduceToGeneric<int64_t   >,
        reduceToGeneric<int32_t   >,
        reduceToGeneric<int16_t   >,
        reduceToGeneric<int8_t    >,
        reduceToGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst);
}

void DeviceCPU::maxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    static const auto funcTable = std::array
    {
        maxToGeneric<double    >,
        maxToGeneric<float     >,
        maxToGeneric<float16_t >,
        maxToGeneric<bfloat16_t>,
        maxToGeneric<int64_t   >,
        maxToGeneric<int32_t   >,
        maxToGeneric<int16_t   >,
        maxToGeneric<int8_t    >,
        maxToGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst);
}

void DeviceCPU::argmaxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    if (dst.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceCPU::argmaxTo supports only int32 data type for its result.");
    }

    static const auto funcTable = std::array
    {
        argmaxToGeneric<double    , int32_t>,
        argmaxToGeneric<float     , int32_t>,
        argmaxToGeneric<float16_t , int32_t>,
        argmaxToGeneric<bfloat16_t, int32_t>,
        argmaxToGeneric<int64_t   , int32_t>,
        argmaxToGeneric<int32_t   , int32_t>,
        argmaxToGeneric<int16_t   , int32_t>,
        argmaxToGeneric<int8_t    , int32_t>,
        argmaxToGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, dim);
}

void DeviceCPU::argmaxIndicesTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    if (dst.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceCPU::argmaxIndicesTo supports only int32 data type for its result.");
    }

    static const auto funcTable = std::array
    {
        argmaxIndicesToGeneric<double    , int32_t>,
        argmaxIndicesToGeneric<float     , int32_t>,
        argmaxIndicesToGeneric<float16_t , int32_t>,
        argmaxIndicesToGeneric<bfloat16_t, int32_t>,
        argmaxIndicesToGeneric<int64_t   , int32_t>,
        argmaxIndicesToGeneric<int32_t   , int32_t>,
        argmaxIndicesToGeneric<int16_t   , int32_t>,
        argmaxIndicesToGeneric<int8_t    , int32_t>,
        argmaxIndicesToGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, dim);
}

void DeviceCPU::sliceSet(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                      size_t dim, size_t start, size_t end, size_t step)
{
    static const auto funcTable = std::array
    {
        sliceSetGeneric<double    >,
        sliceSetGeneric<float     >,
        sliceSetGeneric<float16_t >,
        sliceSetGeneric<bfloat16_t>,
        sliceSetGeneric<int64_t   >,
        sliceSetGeneric<int32_t   >,
        sliceSetGeneric<int16_t   >,
        sliceSetGeneric<int8_t    >,
        sliceSetGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, dim, start, end, step);
}

void DeviceCPU::tril(const DeviceTensorParams& src, const DeviceTensorParams& dst, ssize_t diagonal)
{
    static const auto funcTable = std::array
    {
        trilGeneric<double    >,
        trilGeneric<float     >,
        trilGeneric<float16_t >,
        trilGeneric<bfloat16_t>,
        trilGeneric<int64_t   >,
        trilGeneric<int32_t   >,
        trilGeneric<int16_t   >,
        trilGeneric<int8_t    >,
        trilGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, diagonal);
}

void DeviceCPU::triu(const DeviceTensorParams& src, const DeviceTensorParams& dst, ssize_t diagonal)
{
    static const auto funcTable = std::array
    {
        triuGeneric<double    >,
        triuGeneric<float     >,
        triuGeneric<float16_t >,
        triuGeneric<bfloat16_t>,
        triuGeneric<int64_t   >,
        triuGeneric<int32_t   >,
        triuGeneric<int16_t   >,
        triuGeneric<int8_t    >,
        triuGeneric<uint8_t   >,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, diagonal);
}

void DeviceCPU::indexSelect(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                         const DeviceTensorParams& indices, size_t dim)
{
    static const auto funcTable = std::array
    {
        indexSelectGeneric<double    , int32_t>,
        indexSelectGeneric<float     , int32_t>,
        indexSelectGeneric<float16_t , int32_t>,
        indexSelectGeneric<bfloat16_t, int32_t>,
        indexSelectGeneric<int64_t   , int32_t>,
        indexSelectGeneric<int32_t   , int32_t>,
        indexSelectGeneric<int16_t   , int32_t>,
        indexSelectGeneric<int8_t    , int32_t>,
        indexSelectGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, indices, dim);
}

void DeviceCPU::indexAdd(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                         const DeviceTensorParams& indices, size_t dim)
{
    static const auto funcTable = std::array
    {
        indexAddGeneric<double    , int32_t>,
        indexAddGeneric<float     , int32_t>,
        indexAddGeneric<float16_t , int32_t>,
        indexAddGeneric<bfloat16_t, int32_t>,
        indexAddGeneric<int64_t   , int32_t>,
        indexAddGeneric<int32_t   , int32_t>,
        indexAddGeneric<int16_t   , int32_t>,
        indexAddGeneric<int8_t    , int32_t>,
        indexAddGeneric<uint8_t   , int32_t>,
    };
    // Call the appropriate function from the table.
    funcTable[static_cast<size_t>(src.dtype)](src, dst, indices, dim);
}

void DeviceCPU::emptyCache()
{
}

void DeviceCPU::synchronize()
{
}

}   // aix namespace
