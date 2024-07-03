//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// Project includes
// External includes
// System includes
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <stack>
#include <utility>


namespace aix
{

enum class DataType : size_t
{
    kFloat64 = 0,
    kFloat32 = 1,
};

enum class DeviceType
{
    kCPU,
    kGPU_METAL,
};

// Primary template (default case)
template <typename T>
constexpr DataType getDataType()            { throw std::runtime_error("Unknown format found."); return DataType::kFloat32; }
template <>
constexpr DataType getDataType<double>()    { return DataType::kFloat64; }
template <>
constexpr DataType getDataType<float>()     { return DataType::kFloat32; }

static DataType promoteDataType(DataType dtype1, DataType dtype2)
{
    assert(static_cast<size_t>(dtype1) < 2 && static_cast<size_t>(dtype2) < 2);
    static_assert(static_cast<size_t>(DataType::kFloat64) == 0);
    static_assert(static_cast<size_t>(DataType::kFloat32) == 1);

    static const DataType promotionTable[2][2] =
    {
        //         0                  1
        { DataType::kFloat64, DataType::kFloat64 },     // 0
        { DataType::kFloat64, DataType::kFloat32 },     // 1
    };
    return promotionTable[static_cast<size_t>(dtype1)][static_cast<size_t>(dtype2)];
}

// Forward declarations
class Tensor;

// Tensor Index, Shape and Stride Types
using Index  = std::vector<size_t>;
using Shape  = std::vector<size_t>;
using Stride = std::vector<size_t>;


class Device
{
public:
    virtual ~Device() = default;

    virtual DeviceType type() const { return DeviceType::kCPU; }

    static size_t dataTypeSize(DataType dtype)
    {
        static const size_t dTypeSizeTable[]
        {
            sizeof(double),     // kFloat64
            sizeof(float),      // kFloat32
        };
        return dTypeSizeTable[static_cast<size_t>(dtype)];
    }

    virtual void* allocate(size_t size)
    {
        return std::malloc(size);
    }

    virtual void* allocate(size_t size, DataType dtype)
    {
        return allocate(size * dataTypeSize(dtype));
    }

    virtual void deallocate(void * memory)
    {
        return std::free(memory);
    }

    virtual void add(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            addGeneric<double>,
            addGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void sub(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            subGeneric<double>,
            subGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void mul(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            mulGeneric<double>,
            mulGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void div(const void* a1, const void* a2, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            divGeneric<double>,
            divGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, a2, size, result);
    }

    virtual void addAS(const void* a1, const void* scalar, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            addASGeneric<double>,
            addASGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, scalar, size, result);
    }

    virtual void subAS(const void* a1, const void* scalar, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            subASGeneric<double>,
            subASGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, scalar, size, result);
    }

    virtual void subSA(const void* scalar, const void* a1, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            subSAGeneric<double>,
            subSAGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](scalar, a1, size, result);
    }

    virtual void mulAS(const void* a1, const void* scalar, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            mulASGeneric<double>,
            mulASGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, scalar, size, result);
    }

    virtual void divAS(const void* a1, const void* scalar, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            divASGeneric<double>,
            divASGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, scalar, size, result);
    }

    virtual void divSA(const void* scalar, const void* a1, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            divSAGeneric<double>,
            divSAGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](scalar, a1, size, result);
    }

    virtual void unary(const void* a1, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            unaryGeneric<double>,
            unaryGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, size, result);
    }

    virtual void fill(const void* scalar, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            fillGeneric<double>,
            fillGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](scalar, size, result);
    }

    virtual void sum(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sumGeneric<double>,
            sumGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void mean(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            meanGeneric<double>,
            meanGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void sqrt(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sqrtGeneric<double>,
            sqrtGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void sin(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            sinGeneric<double>,
            sinGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void cos(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            cosGeneric<double>,
            cosGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void tanh(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            tanhGeneric<double>,
            tanhGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void log(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            logGeneric<double>,
            logGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void exp(const void* a, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            expGeneric<double>,
            expGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, size, result);
    }

    virtual void pow(const void* a, const void* exp, const size_t size, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            powGeneric<double>,
            powGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a, exp, size, result);
    }

    virtual void matmul(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result, DataType dtype)
    {
        static const auto funcTable = std::array
        {
            matmulGeneric<double>,
            matmulGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](a1, s1, a2, s2, result);
    }

    virtual void transpose(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                           const Stride& strides, const Stride& newStrides, const size_t size, void* result,
                           DataType dtype)
    {
        static const auto funcTable = std::array
        {
            transposeGeneric<double>,
            transposeGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](dim0, dim1, data, shape, strides, newStrides, size, result);
    }

    virtual void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
    {
        if (srcDType == dstDType)
        {
            std::memcpy(dst, src, size * dataTypeSize(srcDType));
        }
        else
        {
            // Define a function pointer type for the conversion copy functions.
            using ConversionCopyFunc = void (*)(const void*, void*, size_t);

            // Create a lookup table of the functions.
            static const ConversionCopyFunc funcTable[2][2] =
            {
                { nullptr, conversionCopyGeneric<double, float> },
                { conversionCopyGeneric<float, double>, nullptr }
            };
            // Call the appropriate function from the table.
            funcTable[static_cast<size_t>(srcDType)][static_cast<size_t>(dstDType)](src, dst, size);
        }
    }

    virtual void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
    {
        copy(src, srcDType, dst, dstDType, size);
        commitAndWait();    // This call has no effect, but it shows the difference between copy and copyImmediate.
    }

    virtual void broadcastTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                             DataType dtype)
    {
        static const auto funcTable = std::array
        {
            broadcastToGeneric<double>,
            broadcastToGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape);
    }

    virtual void reduceTo(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape,
                          DataType dtype)
    {
        static const auto funcTable = std::array
        {
            reduceToGeneric<double>,
            reduceToGeneric<float>,
        };
        // Call the appropriate function from the table.
        funcTable[static_cast<size_t>(dtype)](src, dst, size, shape, newShape);
    }

    virtual void commitAndWait()
    {
    }

protected:
    template <typename T>
    static void addGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] + t2[i];
        }
    }

    template <typename T>
    static void subGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] - t2[i];
        }
    }

    template <typename T>
    static void mulGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] * t2[i];
        }
    }

    template <typename T>
    static void divGeneric(const void* a1, const void* a2, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] / t2[i];
        }
    }

    template <typename T>
    static void addASGeneric(const void* a1, const void* scalar, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] + scalarValue;
        }
    }

    template <typename T>
    static void subASGeneric(const void* a1, const void* scalar, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] - scalarValue;
        }
    }

    template <typename T>
    static void subSAGeneric(const void* scalar, const void* a1, const size_t size, void* result)
    {
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto t1  = static_cast<const T*>(a1);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = scalarValue - t1[i];
        }
    }

    template <typename T>
    static void mulASGeneric(const void* a1, const void* scalar, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] * scalarValue;
        }
    }

    template <typename T>
    static void divASGeneric(const void* a1, const void* scalar, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = t1[i] / scalarValue;
        }
    }

    template <typename T>
    static void divSAGeneric(const void* scalar, const void* a1, const size_t size, void* result)
    {
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto t1  = static_cast<const T*>(a1);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = scalarValue / t1[i];
        }
    }

    template <typename T>
    static void unaryGeneric(const void* a1, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = -t1[i];
        }
    }

    template <typename T>
    static void fillGeneric(const void* scalar, const size_t size, void* result)
    {
        auto scalarValue = *static_cast<const float*>(scalar);    // TODO: Handle multiple scalar types
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = scalarValue;
        }
    }

    template <typename T>
    static void sumGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        T sum = 0;
        for (size_t i = 0; i < size; ++i)
        {
            sum += t1[i];
        }
        *res = sum;
    }

    template <typename T>
    static void meanGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        T sum = 0;
        for (size_t i = 0; i < size; ++i)
        {
            sum += t1[i];
        }
        *res = sum / size;
    }

    template <typename T>
    static void sqrtGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::sqrt(t1[i]);
        }
    }

    template <typename T>
    static void sinGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::sin(t1[i]);
        }
    }

    template <typename T>
    static void cosGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::cos(t1[i]);
        }
    }

    template <typename T>
    static void tanhGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::tanh(t1[i]);
        }
    }

    template <typename T>
    static void logGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::log(t1[i]);
        }
    }

    template <typename T>
    static void expGeneric(const void* a, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::exp(t1[i]);
        }
    }

    template <typename T>
    static void powGeneric(const void* a1, const void* exp, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(exp);
        auto res = static_cast<T*>(result);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = std::pow(t1[i], t2[i]);
        }
    }

    template <typename T>
    static void matmulGeneric(const void* a1, const Shape & s1, const void* a2, const Shape & s2, void* result)
    {
        auto t1  = static_cast<const T*>(a1);
        auto t2  = static_cast<const T*>(a2);
        auto res = static_cast<T*>(result);

        // NOTE: Since TensorValue validated the parameters, device method do not validate again.
        size_t m = s1[0];        // Rows of the first matrix
        size_t n = s2[1];        // Columns of the second matrix
        size_t inner = s1[1];    // Inner dimension

        // Perform matrix multiplication
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    sum += t1[i * s1[1] + k] * t2[k * n + j];
                }
                res[i * n + j] = sum;
            }
        }
    }

    template <typename T>
    static void transposeGeneric(size_t dim0, size_t dim1, const void* data, [[maybe_unused]] const Shape& shape,
                                 const Stride& strides, const Stride& newStrides, const size_t size, void* result)
    {
        auto t1  = static_cast<const T*>(data);
        auto res = static_cast<T*>(result);

        // Perform the generalized transpose operation.
        for (size_t i=0; i<size; ++i)
        {
            auto oldIndices = unflattenIndex(i, strides);
            std::swap(oldIndices[dim0], oldIndices[dim1]);
            size_t newIndex = flattenIndex(oldIndices, newStrides);
            res[newIndex] = t1[i];
        }
    }

    template <typename SrcType, typename DstType>
    static void conversionCopyGeneric(const void* src, void* dst, size_t size)
    {
        auto tSrc = static_cast<const SrcType*>(src);
        auto tDst = static_cast<DstType*>(dst);
        for (size_t i=0; i<size; ++i)
        {
            tDst[i] = static_cast<DstType>(tSrc[i]);
        }
    }

    template <typename T>
    static void broadcastToGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        for (size_t index = 0; index < size; ++index)
        {
            // Copy value from original index to the new index.
            tDst[index] = tSrc[translationIndex(index, shape, newShape)];
        }
    }

    template <typename T>
    static void reduceToGeneric(const void* src, void* dst, size_t size, const Shape& shape, const Shape& newShape)
    {
        auto tSrc = static_cast<const T*>(src);
        auto tDst = static_cast<T*>(dst);

        // Sum the values from the broadcasted tensor to the original tensor shape. The reduction involves summation
        // because each element of the original tensor is used multiple times in the broadcasted operation.
        // Summing the gradients correctly aggregates these contributions.
        for (size_t index = 0; index < size; ++index)
        {
            tDst[translationIndex(index, shape, newShape)] += tSrc[index];
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
};


// Default device.
static Device defaultDevice;      // TODO: default CPU device needs to move to a global context.
static std::random_device randomDevice;
static std::mt19937 randGen(randomDevice());


class TensorValue
{
public:
    // Constructor
    TensorValue(const void* data, size_t size, DataType srcDType, Shape shape, Device* device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_data = device->allocate(size, dType);
        device->copyImmediate(data, srcDType, m_data, dType, size);
        m_size = size;
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    template<typename T>
    TensorValue(const std::initializer_list<T> & data, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_data = device->allocate(data.size(), dType);
        device->copyImmediate(data.begin(), getDataType<T>(), m_data, dType, data.size());
        m_size = data.size();
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(float value, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
        // initialize data.
        device->fill(&value, m_size, m_data, dType);
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device, size_t size, Stride strides, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_size(size), m_shape(std::move(shape)), m_strides(std::move(strides)), m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_data = device->allocate(m_size, dType);
    }

    // Constructor
    TensorValue(float value, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape{}, m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_size = 1;
        m_data = device->allocate(m_size, dType);
        device->fill(&value, m_size, m_data, dType);
        m_strides = computeStrides();
    }

    // Destructor
    ~TensorValue()
    {
        if (m_data) m_device->deallocate(m_data);
        m_data   = nullptr;
        m_device = nullptr;
    }

    // Copy constructor
    TensorValue(const TensorValue& other) noexcept
    {
        m_dType   = other.m_dType;
        m_size    = other.m_size;
        m_shape   = other.m_shape;
        m_strides = other.m_strides;
        m_device  = other.m_device;
        m_data    = m_device->allocate(other.m_size, other.m_dType);
        m_device->copyImmediate(other.m_data, other.m_dType, m_data, other.m_dType, other.m_size);
    }

    // Copy assignment operator
    TensorValue& operator=(const TensorValue& other) noexcept
    {
        if (this != &other)     // Protect against self-assignment
        {
            m_device->deallocate(m_data);
            m_dType   = other.m_dType;
            m_size    = other.m_size;
            m_shape   = other.m_shape;
            m_strides = other.m_strides;
            m_device  = other.m_device;
            m_data    = m_device->allocate(other.m_size, other.m_dType);
            m_device->copyImmediate(other.m_data, other.m_dType, m_data, other.m_dType, other.m_size);
        }

        return *this;
    }

    // Move constructor
    TensorValue(TensorValue&& other) noexcept
    {
        m_dType   = other.m_dType;
        m_data    = other.m_data;
        m_size    = other.m_size;
        m_shape   = other.m_shape;
        m_strides = other.m_strides;
        m_device  = other.m_device;
        other.m_size   = 0;
        other.m_data   = nullptr;           // Avoid double deletion
        other.m_device = nullptr;
    }

    // Move assignment operator
    TensorValue& operator=(TensorValue&& other) noexcept
    {
        if (this != &other)
        {
            m_device->deallocate(m_data);   // Free existing resource
            m_dType   = other.m_dType;
            m_data    = other.m_data;
            m_size    = other.m_size;
            m_shape   = other.m_shape;
            m_strides = other.m_strides;
            m_device  = other.m_device;
            other.m_size   = 0;
            other.m_data   = nullptr;       // Avoid double deletion
            other.m_device = nullptr;
        }

        return *this;
    }

    // Access element at a specific index (non-const version).
    template<typename T>
    T & getValueAt(const Index & indices)     { return static_cast<T*>(m_data)[getIndex(indices)]; }

    // Access element at a specific index (const version).
    template<typename T>
    T getValueAt(const Index & indices) const { return static_cast<T*>(m_data)[getIndex(indices)]; }

    // Get the data type of the tensor.
    DataType dataType() const      { return m_dType; }

    // Get the shape of the tensor
    const Shape & shape() const    { return m_shape; }

    // Get the strides of the tensor
    const Stride & strides() const  { return m_strides; }

    // Get the raw data of the tensor.
    const void* data() const    { return m_data; }
    void* data()                { return m_data; }

    // Get the raw data of the tensor.
    template<typename T>
    const T* data() const       { return static_cast<T*>(m_data); }
    template<typename T>
    T* data()                   { return static_cast<T*>(m_data); }

    // Get the size of the data
    size_t size() const         { return m_size; }

    // Get the device
    Device * device() const     { return m_device; }

    // Set the device
    void device(Device * device)
    {
        if (m_device == device) return;
        // Move data to the new device. Create a new data with new device and copy the data. Deallocate the old data.
        // Create a new array from the new device.
        auto newData = device->allocate(m_size, m_dType);
        // Copy old data to the new array.
        device->copyImmediate(m_data, m_dType, newData, m_dType, m_size);
        // Delete old data from old device.
        m_device->deallocate(m_data);
        // Set new data and the new device.
        m_data = newData;
        m_device = device;
    }

    template<typename T>
    T item() const
    {
        if (!m_shape.empty())    // Scalar value must have no dimension.
        {
            throw std::invalid_argument("Tensor is not a scalar.");
        }
        return static_cast<T*>(m_data)[0];
    }

    // Returns a new TensorValue with a new shape.
    TensorValue reshape(const Shape & newShape) const
    {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
        if (m_size != newSize)
        {
            throw std::invalid_argument("Reshape error: element count mismatch (" +
                                        std::to_string(m_size) + " vs " + std::to_string(newSize) + ").");
        }
        return {m_data, m_size, m_dType, newShape, m_device, m_dType};
    }

    // Equalize tensor data types by promoting data type of tensors.
    TensorValue to(DataType newDataType) const
    {
        if (dataType() != newDataType)
        {
            return {data(), size(), dataType(), shape(), device(), newDataType};
        }
        return *this;
    }

    // Returns true if two TensorValue shapes are compatible for a broadcast operation.
    static bool checkBroadcastShapes(const Shape& shape1, const Shape& shape2)
    {
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend())
        {
            size_t dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            size_t dim2 = (it2 != shape2.rend()) ? *it2++ : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                return false;
            }
        }
        return true;
    }

    // Returns final shape of a broadcast operation.
    static Shape broadcastShapes(const Shape& shape1, const Shape& shape2)
    {
        Shape resultShape;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend())
        {
            size_t dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            size_t dim2 = (it2 != shape2.rend()) ? *it2++ : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            {
                throw std::invalid_argument("Shapes are not compatible for broadcasting.");
            }
            resultShape.push_back(std::max(dim1, dim2));
        }
        std::reverse(resultShape.begin(), resultShape.end());
        return resultShape;
    }

    static bool checkBroadcastTo(const Shape& sourceShape, const Shape& targetShape)
    {
        if (sourceShape.size() > targetShape.size()) return false;

        auto itSrc = sourceShape.rbegin();
        auto itTgt = targetShape.rbegin();

        while (itTgt != targetShape.rend())
        {
            size_t dimSrc = (itSrc != sourceShape.rend()) ? *itSrc++ : 1;
            size_t dimTgt = *itTgt++;

            if (dimSrc != dimTgt && dimSrc != 1)
            {
                return false;
            }
        }

        return true;
    }

    // Returns a broadcasted TensorValue with a new shape.
    TensorValue broadcastTo(const Shape& newShape) const
    {
        if (!checkBroadcastTo(shape(), newShape))
        {
            throw std::invalid_argument("Target TensorValue shape is not broadcastable.");
        }
        Shape resultShape = broadcastShapes(shape(), newShape);
        TensorValue result(resultShape, device(), m_dType);
        device()->broadcastTo(m_data, result.data(), result.size(), shape(), resultShape, m_dType);
        return result;
    }

    // Reduces the TensorValue back to the original shape.
    TensorValue reduceTo(const Shape & originalShape) const
    {
        // Ensure tensor values are initialized to zero, as the reduction operation performs a summation.
        TensorValue result(0, originalShape, device(), m_dType);
        device()->reduceTo(m_data, result.data(), m_size, m_shape, originalShape, m_dType);
        return result;
    }

    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            auto result = prepareTensors(lhs, rhs);
            result.device()->add(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->add(m_data, other.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            auto result = prepareTensors(lhs, rhs);
            result.device()->sub(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }
        TensorValue result(m_shape, m_device, m_dType);
        m_device->sub(m_data, other.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            auto result = prepareTensors(lhs, rhs);
            result.device()->mul(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }
        TensorValue result(m_shape, m_device, m_dType);
        m_device->mul(m_data, other.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            auto result = prepareTensors(lhs, rhs);
            result.device()->div(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }
        TensorValue result(m_shape, m_device, m_dType);
        m_device->div(m_data, other.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Overload the += operator - In-place operation.
    TensorValue & operator+=(const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            prepareTensors(lhs, rhs);
            m_device->add(lhs.data(), rhs.data(), lhs.size(), lhs.data(), lhs.dataType());
            *this = TensorValue(lhs.data(), lhs.size(), lhs.dataType(), lhs.shape(), lhs.device(), dataType());
            return *this;
        }
        else
        {
            m_device->add(m_data, other.m_data, m_size, m_data, m_dType);
        }
        return *this;
    }

    // Overload the -= operator - In-place operation.
    TensorValue & operator-=(const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            prepareTensors(lhs, rhs);
            m_device->sub(lhs.data(), rhs.data(), lhs.size(), lhs.data(), lhs.dataType());
            *this = TensorValue(lhs.data(), lhs.size(), lhs.dataType(), lhs.shape(), lhs.device(), dataType());
            return *this;
        }
        else
        {
            m_device->sub(m_data, other.m_data, m_size, m_data, m_dType);
        }
        return *this;
    }

    // Overload the *= operator - In-place operation.
    TensorValue & operator*=(const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            prepareTensors(lhs, rhs);
            m_device->mul(lhs.data(), rhs.data(), lhs.size(), lhs.data(), lhs.dataType());
            *this = TensorValue(lhs.data(), lhs.size(), lhs.dataType(), lhs.shape(), lhs.device(), dataType());
            return *this;
        }
        else
        {
            m_device->mul(m_data, other.m_data, m_size, m_data, m_dType);
        }
        return *this;
    }

    // Overload the /= operator - In-place operation.
    TensorValue & operator/=(const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = other;
            prepareTensors(lhs, rhs);
            m_device->div(lhs.data(), rhs.data(), lhs.size(), lhs.data(), lhs.dataType());
            *this = TensorValue(lhs.data(), lhs.size(), lhs.dataType(), lhs.shape(), lhs.device(), dataType());
            return *this;
        }
        else
        {
            m_device->div(m_data, other.m_data, m_size, m_data, m_dType);
        }
        return *this;
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->unary(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue operator+(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->addAS(m_data, &scalar, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue operator-(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->subAS(m_data, &scalar, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue& operator+=(float scalar)
    {
        // Perform element-wise.
        m_device->addAS(m_data, &scalar, m_size, m_data, m_dType);
        return *this;
    }

    TensorValue& operator-=(float scalar)
    {
        // Perform element-wise.
        m_device->subAS(m_data, &scalar, m_size, m_data, m_dType);
        return *this;
    }

    TensorValue& operator*=(float scalar)
    {
        // Perform element-wise.
        m_device->mulAS(m_data, &scalar, m_size, m_data, m_dType);
        return *this;
    }

    TensorValue& operator/=(float scalar)
    {
        // Perform element-wise.
        m_device->divAS(m_data, &scalar, m_size, m_data, m_dType);
        return *this;
    }

    TensorValue operator*(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->mulAS(m_data, &scalar, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue operator/(float scalar) const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->divAS(m_data, &scalar, m_size, result.m_data, m_dType);
        return result;
    }

    friend TensorValue operator*(float scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device, tensor.m_dType);
        tensor.m_device->mulAS(tensor.m_data, &scalar, tensor.m_size, result.m_data, result.m_dType);
        return result;
    }

    friend TensorValue operator/(float scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device, tensor.m_dType);
        tensor.m_device->divSA(&scalar, tensor.m_data, tensor.m_size, result.m_data, result.m_dType);
        return result;
    }

    friend TensorValue operator+(float scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device, tensor.m_dType);
        tensor.m_device->addAS(tensor.m_data, &scalar, tensor.m_size, result.m_data, result.m_dType);
        return result;
    }

    friend TensorValue operator-(float scalar, const TensorValue & tensor)
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(tensor.m_shape, tensor.m_device, tensor.m_dType);
        tensor.m_device->subSA(&scalar, tensor.m_data, tensor.m_size, result.m_data, result.m_dType);
        return result;
    }

    void fill(float value) const
    {
        m_device->fill(&value, m_size, m_data, m_dType);
    }

    TensorValue sum() const
    {
        TensorValue result({}, device(), m_dType);
        m_device->sum(m_data, m_size, result.data(), m_dType);
        return result;
    }

    TensorValue mean() const
    {
        TensorValue result({}, device(), m_dType);
        m_device->mean(m_data, m_size, result.data(), m_dType);
        return result;
    }

    TensorValue sqrt() const
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->sqrt(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue sin() const
    {
        // Perform element-wise sin.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->sin(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue cos() const
    {
        // Perform element-wise cos.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->cos(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue tanh() const
    {
        // Perform element-wise tanh.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->tanh(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue log() const
    {
        // Perform element-wise tanh.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->log(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue exp() const
    {
        // Perform element-wise exp.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->exp(m_data, m_size, result.m_data, m_dType);
        return result;
    }

    TensorValue pow(const TensorValue & exp) const
    {
        if (shape() != exp.shape() || dataType() != exp.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = exp;
            auto result = prepareTensors(lhs, rhs);
            result.device()->pow(lhs.data(), rhs.data(), lhs.size(), result.data(), result.dataType());
            return result;
        }

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->pow(m_data, exp.m_data, m_size, result.m_data, m_dType);
        return result;
    }

    // Matrix multiplication for 2D tensors.
    TensorValue matmul(const TensorValue & b) const
    {
        // Ensure both tensors are 2D or can be treated as such.
        if (m_shape.size() != 2 || b.shape().size() != 2)
        {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }

        // Check if the inner dimensions match.
        if (m_shape[1] != b.shape()[0])
        {
            throw std::invalid_argument("The inner dimensions of the tensors do not match.");
        }

        Shape resultShape{m_shape[0], b.shape()[1]};

        // Convert tensors to the promoted data type if necessary
        if (dataType() != b.dataType())
        {
            TensorValue lhs = *this;
            TensorValue rhs = b;
            auto promotedDType = promoteDataType(lhs.dataType(), rhs.dataType());
            lhs = lhs.to(promotedDType);
            rhs = rhs.to(promotedDType);
            TensorValue result(resultShape, lhs.device(), promotedDType);
            result.device()->matmul(lhs.data(), lhs.shape(), rhs.data(), rhs.shape(), result.data(), result.dataType());
            return result;
        }

        // Resultant tensor shape
        TensorValue result(resultShape, m_device, m_dType);
        m_device->matmul(m_data, m_shape, b.m_data, b.m_shape, result.m_data, m_dType);
        return result;
    }

    // Generalized transpose function.
    TensorValue transpose(size_t dim0, size_t dim1) const
    {
        // Check dimensions
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size())
        {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        Shape newShape = m_shape;
        std::swap(newShape[dim0], newShape[dim1]);
        TensorValue result(newShape, device(), m_dType);
        m_device->transpose(dim0, dim1, m_data, m_shape, m_strides, result.strides(), result.size(), result.data(), m_dType);
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream& operator<<(std::ostream & os, const TensorValue & tensor);

private:
    // Compute the strides based on the shape of the tensor
    Stride computeStrides()
    {
        Stride strides(m_shape.size());
        size_t stride = 1;
        for (int64_t i = strides.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= m_shape[i];
        }
        return strides;
    }

    // Get the flat index from a vector of indices
    size_t getIndex(const Index & indices) const
    {
        assert(indices.size() == m_shape.size());
        return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), 0);
    }

    // Promotes data types and applies broadcasting if necessary.
    static TensorValue prepareTensors(TensorValue & lhs, TensorValue & rhs)
    {
        // TODO: Minimize copy operations.
        auto promotedDType = lhs.dataType();

        if (lhs.dataType() != rhs.dataType())
        {
            // Convert tensors to the promoted data type if necessary
            promotedDType = promoteDataType(lhs.dataType(), rhs.dataType());
            lhs = lhs.to(promotedDType);
            rhs = rhs.to(promotedDType);
        }

        // If shapes are different then try broadcasting.
        if (lhs.shape() != rhs.shape())
        {
            Shape bcShape = broadcastShapes(lhs.shape(), rhs.shape());
            lhs = lhs.broadcastTo(bcShape);
            rhs = rhs.broadcastTo(bcShape);
        }

        return {lhs.shape(), lhs.device(), promotedDType};
    }

    // Print Tensor data
    template<typename T>
    void print(std::ostream & os) const
    {
        // Print scalar value, a tensor with no dimension.
        if (m_shape.empty())
        {
            os << item<T>() << "\n\n";
        }
        else if (m_shape.size() == 1)
        {
            // Print tensor that has only one dimension.
            for (size_t i = 0; i < m_shape[0]; ++i)
            {
                os << "  " << getValueAt<T>({i}) << "\n";
            }
            os << "\n";
        }
        else
        {
            // Print tensor that has at least two dimensions.
            std::stack<std::pair<Index, size_t>> stack;
            stack.push({Index(), 0});

            while (!stack.empty())
            {
                auto [indices, dim] = stack.top();
                stack.pop();

                if (dim == m_shape.size() - 2)
                {
                    bool isOverTwo = m_shape.size() > 2;
                    if (isOverTwo)
                    {
                        os << "(";
                    }

                    for (size_t i = 0; i < indices.size(); ++i)
                    {
                        os << indices[i];
                        if (i < indices.size() - 1)
                        {
                            os << ",";
                        }
                    }

                    if (isOverTwo)
                    {
                        os << ",.,.) =\n";
                    }

                    size_t rows = m_shape[dim];
                    size_t cols = m_shape[dim + 1];

                    for (size_t i = 0; i < rows; ++i)
                    {
                        for (size_t j = 0; j < cols; ++j)
                        {
                            Index subIndices = indices;
                            subIndices.push_back(i);
                            subIndices.push_back(j);
                            os << "  " << getValueAt<T>(subIndices);
                        }
                        os << '\n';
                    }

                    os << '\n';
                }
                else
                {
                    for (size_t i = m_shape[dim]; i-- > 0;) // Process in reverse order
                    {
                        Index subIndices = indices;
                        subIndices.push_back(i);
                        stack.push({subIndices, dim + 1});
                    }
                }
            }
        }

        // Print shape
        switch (dataType())
        {
            case DataType::kFloat32:  os << "[ Float{";    break;
            case DataType::kFloat64:  os << "[ Double{";   break;
            default:                  os << "[ Unknown{";  break;
        }

        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            os << m_shape[i];
            if (i < m_shape.size() - 1)
            {
                os << ",";
            }
        }
        os << "} ]\n";
    }

private:
    DataType  m_dType{DataType::kFloat32};
    void*     m_data{nullptr};  // The flat array of tensor elements.
    size_t    m_size{0};        // Number of elements in DataType.
    Shape     m_shape;          // The shape of the tensor.
    Stride    m_strides;        // The strides for indexing the tensor.
    Device *  m_device{nullptr};
};


class TensorNode
{
public:
    // Constructor
    explicit TensorNode(const TensorValue & value, bool requireGrad = false) :
        m_value{value}, m_grad{value.shape(), value.device(), value.size(), value.strides(), value.dataType()},
        m_requireGrad{requireGrad}
    {
    }

    // Constructor
    explicit TensorNode(const Shape & shape, Device * device, bool requireGrad = false, DataType dType = DataType::kFloat32) :
        m_value{shape, device, dType}, m_grad{shape, device, m_value.size(), m_value.strides(), dType},
        m_requireGrad{requireGrad}
    {
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)
    {
        if (m_retainGrad)
        {
            // TODO: Do not create the gradient vector until it's required to improve performance.
            m_grad += seed;
        }
        m_backwardFunc(this, seed);
    }

    Device * device() const          { return m_value.device(); }
    void device(Device * device)     { m_value.device(device); m_grad.device(device); }

    std::string  m_name;
    TensorValue  m_value;
    TensorValue  m_grad;
    bool  m_requireGrad;
    bool  m_retainGrad{false};
    std::shared_ptr<TensorNode>  m_a{nullptr};
    std::shared_ptr<TensorNode>  m_b{nullptr};
    size_t m_dim0{0};
    size_t m_dim1{0};
    std::function<void(TensorNode * tensor, const TensorValue & seed)>  m_backwardFunc{nullptr};
};


class Tensor
{
public:
    // Constructor.
    Tensor() = default;

    // Constructor.
    explicit Tensor(const void* data, size_t size, DataType srcDType, const Shape & shape, bool requireGrad = false,
                    DataType dType = DataType::kFloat32, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{data, size, srcDType, shape, device, dType}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(float value, const Shape & shape, bool requireGrad = false,
                    DataType dType = DataType::kFloat32, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{value, shape, device, dType}, requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const Shape & shape, bool requireGrad = false,
                    DataType dType = DataType::kFloat32, Device * device = &defaultDevice)
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(shape, device, requireGrad, dType);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(float value=1)  { m_data->backward(TensorValue{value, m_data->m_a->m_grad.shape(), device(), dataType()}); }
    void backward(float value, const Shape & gradShape)  { m_data->backward(TensorValue{value, gradShape, device(), dataType()}); }

    // Getters and setters for the tensor's value.
    inline const TensorValue & value() const    { return m_data->m_value; }
    inline TensorValue & value()                { return m_data->m_value; }
    inline const Shape & shape() const          { return m_data->m_value.shape(); }
    inline DataType dataType() const            { return m_data->m_value.dataType(); }

    // Gradient-related methods.
    inline const TensorValue & grad() const
    {
        if (!m_data->m_requireGrad && !m_data->m_retainGrad)
        {
            throw std::runtime_error("Gradients for non-leaf tensors won’t be populated during automatic gradient"  \
                                     " calculation. Use .retainGrad() on the non-leaf tensor if needed, or access"  \
                                     " the leaf tensor instead.");
        }
        return m_data->m_grad;
    }

    inline void zeroGrad()                      { m_data->m_grad.fill(0); }
    inline bool isRequireGrad() const           { return m_data->m_requireGrad; }
    inline void retainGrad() const              { m_data->m_retainGrad = true; m_data->m_grad.fill(0); }

    // Set operation device for the tensor.
    inline Tensor & to(Device & device)         { m_data->device(&device); return *this; }
    inline Device * device() const              { return m_data->device(); }

    inline void name(const std::string& name) const  { m_data->m_name = name; }
    inline const std::string& name() const           { return m_data->m_name; }

    // Returns a new Tensor with a new shape.
    Tensor reshape(const Shape & newShape) const
    {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
        if (value().size() != newSize)
        {
            throw std::invalid_argument("Reshape error: element count mismatch (" +
                                        std::to_string(value().size()) + " vs " + std::to_string(newSize) + ").");
        }
        return Tensor{value().data(), value().size(), dataType(), newShape, isRequireGrad(), dataType(), device()};
    }

    Tensor broadcastTo(const Shape & newShape) const
    {
        TensorValue tValue = m_data->m_value.broadcastTo(newShape);
        Tensor result{tValue.data(), tValue.size(), tValue.dataType(), tValue.shape(), isRequireGrad(), dataType(), device()};
        result.m_data->m_a = m_data;            // Keep the reference to the original tensor node
        result.m_data->m_backwardFunc = broadcastBackwardFunc;
        return result;
    }

    Tensor to(DataType newDataType) const
    {
        if (dataType() != newDataType)
        {
            Tensor result{value().data(), value().size(), value().dataType(), value().shape(), isRequireGrad(),
                          newDataType, device()};
            result.m_data->m_a = m_data;
            result.m_data->m_backwardFunc = toBackwardFunc;
            return result;
        }
        return *this;
    }

    static void defaultBackward(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_requireGrad && !node->m_retainGrad)
        {
            assert(node->m_grad.dataType() == seed.dataType());
            node->m_grad += seed;
        }
    }

    static void broadcastBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Accumulate the gradient to the original node by reducing the gradient from the broadcasted shape to the
        // original shape. Summation is used for gradient accumulation when reducing dimensions because each element
        // of the original tensor contributes to multiple elements of the resulting tensor after broadcasting.
        node->m_a->backward(seed.reduceTo(node->m_a->m_value.shape()));
    }

    static void toBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Ensure that the seed gradient is converted back to the data type of the original tensor.
        node->m_a->backward(seed.to(node->m_a->m_value.dataType()));
    }

    static void addBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(seed);
    }

    static void subBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed);
        node->m_b->backward(-seed);
    }

    static void mulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(node->m_b->m_value * seed);
        node->m_b->backward(node->m_a->m_value * seed);
    }

    static void divBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Calculate gradients.
        node->m_a->backward(seed / node->m_b->m_value);                                               // ∂f/∂a = 1 / b
        node->m_b->backward(-node->m_a->m_value * seed / (node->m_b->m_value * node->m_b->m_value));  // ∂f/∂b = -a / b^2
    }

    static void unaryBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // Calculate gradients.
        node->m_a->backward(-seed);
    }

    static void sqrtBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sqrt(a) with respect to 'a' is 0.5/sqrt(a).
        // Therefore, the gradient of the input is multiplied by 0.5/sqrt(a).
        node->m_a->backward(0.5 / node->m_a->m_value.sqrt() * seed);   // ∂f/∂a = 0.5/sqrt(a)
    }

    static void sinBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        node->m_a->backward(node->m_a->m_value.cos() * seed);   // ∂f/∂a = cos(a)
    }

    static void cosBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of cos(a) with respect to 'a' is -sin(a).
        // Therefore, the gradient of the input is multiplied by -sin(a).
        node->m_a->backward(-node->m_a->m_value.sin() * seed);   // ∂f/∂a = -sin(a)
    }

    static void tanhBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        const auto & tanhValue = node->m_a->m_value.tanh();
        node->m_a->backward((float(1) - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    static void logBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // TODO: Handle division by zero case.
        // The derivative of log(a) with respect to 'a' is 1/a.
        node->m_a->backward(seed / node->m_a->m_value);  // ∂f/∂a = 1/a
    }

    static void expBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The derivative of exp(a) with respect to 'a' is exp(a), itself.
        node->m_a->backward(seed * node->m_a->m_value.exp());  // ∂f/∂a = exp(a)
    }

    static void powBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // The derivative of pow(a, b) with respect to 'a' is b * a^(b-1).
        // ∂f/∂a = b * pow(a, b-1)
        node->m_a->backward(seed * node->m_b->m_value * node->m_a->m_value.pow(node->m_b->m_value - float(1)));
    }

    static void matmulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a || !node->m_b) return;
        // Assuming m_a and m_b are the input matrices a and b, respectively,
        // and seed is ∂E/∂c, the gradient of the loss with respect to the output matrix c.
        // Compute gradients with respect to a and b

        // Corrected to use matrix multiplication for backward pass calculations
        node->m_a->backward(seed.matmul(node->m_b->m_value.transpose(0, 1)));      // ∂E/∂a = ∂E/∂c * b^T
        node->m_b->backward(node->m_a->m_value.transpose(0, 1).matmul(seed));      // ∂E/∂b = a^T * ∂E/∂c
    }

    static void transposeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        node->m_a->backward(seed.transpose(node->m_dim0, node->m_dim1));
    }

    static void sumBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // For the sum operation, the gradient is simply the seed
        node->m_a->backward(seed);
    }

    static void meanBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (!node->m_a) return;
        // The gradient of the mean operation is distributed evenly across all elements. grad = 1/N
        node->m_a->backward(seed / float(node->m_a->m_value.size()));
    }

    // Overload the + operator
    Tensor operator+(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), isRequireGrad() || other.isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value + rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = addBackwardFunc;
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), isRequireGrad() || other.isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value - rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = subBackwardFunc;
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(shape(), isRequireGrad() || other.isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value * rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = mulBackwardFunc;
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        Tensor result(bcShape, isRequireGrad() || other.isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value / rhs.m_data->m_value;
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = divBackwardFunc;
        return result;
    }

    Tensor operator-() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = -m_data->m_value;
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = unaryBackwardFunc;
        return result;
    }

    Tensor operator+(const float & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), dataType(), device());
        return *this + tensor;
    }

    Tensor operator-(const float & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), dataType(), device());
        return *this - tensor;
    }

    Tensor operator*(const float & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), dataType(), device());
        return *this * tensor;
    }

    Tensor operator/(const float & scalar) const
    {
        Tensor tensor(scalar, shape(), isRequireGrad(), dataType(), device());
        return *this / tensor;
    }

    friend Tensor operator+(float scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.dataType(), rhsTensor.device());
        return tensor + rhsTensor;
    }

    friend Tensor operator-(float scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.dataType(), rhsTensor.device());
        return tensor - rhsTensor;
    }

    friend Tensor operator*(float scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.dataType(), rhsTensor.device());
        return tensor * rhsTensor;
    }

    friend Tensor operator/(float scalar, const Tensor & rhsTensor)
    {
        Tensor tensor(scalar, rhsTensor.shape(), rhsTensor.isRequireGrad(), rhsTensor.dataType(), rhsTensor.device());
        return tensor / rhsTensor;
    }

    Tensor sqrt() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.sqrt();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sqrtBackwardFunc;
        return result;
    };

    Tensor sin() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.sin();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sinBackwardFunc;
        return result;
    };

    Tensor cos() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.cos();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = cosBackwardFunc;
        return result;
    };

    Tensor tanh() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.tanh();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = tanhBackwardFunc;
        return result;
    };

    Tensor log() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.log();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = logBackwardFunc;
        return result;
    };

    Tensor exp() const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());
        result.m_data->m_value = m_data->m_value.exp();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = expBackwardFunc;
        return result;
    };

    Tensor sum() const
    {
        Tensor result({}, isRequireGrad(), dataType(), device());     // Scalar tensor for the mean result.
        result.m_data->m_value = m_data->m_value.sum();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = sumBackwardFunc;
        return result;
    }

    Tensor mean() const
    {
        Tensor result({}, isRequireGrad(), dataType(), device());     // Scalar tensor for the mean result.
        result.m_data->m_value = m_data->m_value.mean();
        result.m_data->m_a = m_data;
        result.m_data->m_backwardFunc = meanBackwardFunc;
        return result;
    }

    Tensor pow(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);        // Exponent tensor.

        Tensor result(bcShape, isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value.pow(rhs.m_data->m_value);
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = powBackwardFunc;
        return result;
    }

    Tensor matmul(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        auto lhs = to(promotedDType);
        auto rhs = other.to(promotedDType);

        Tensor result({shape()[0], rhs.shape()[1]}, isRequireGrad() || rhs.isRequireGrad(), dataType(), device());
        result.m_data->m_value = lhs.m_data->m_value.matmul(rhs.m_data->m_value);
        result.m_data->m_a = lhs.m_data;
        result.m_data->m_b = rhs.m_data;
        result.m_data->m_backwardFunc = matmulBackwardFunc;
        return result;
    }

    Tensor transpose(const size_t dim0, size_t dim1) const
    {
        Tensor result(shape(), isRequireGrad(), dataType(), device());     // Scalar tensor for the mean result.
        result.m_data->m_value = m_data->m_value.transpose(dim0, dim1);
        result.m_data->m_a = m_data;
        result.m_data->m_dim0 = dim0;
        result.m_data->m_dim1 = dim1;
        result.m_data->m_backwardFunc = transposeBackwardFunc;
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream & operator<<(std::ostream& os, const Tensor& tensor);

protected:
    inline Shape broadcastShape(const Shape& otherShape) const
    {
        return shape() == otherShape ? shape() : TensorValue::broadcastShapes(shape(), otherShape);
    }

    std::shared_ptr<TensorNode>  m_data{nullptr};
};

// Some convenience method definitions.

inline Tensor tensor(float value, bool requireGrad = false)
{
    return Tensor{value, {}, requireGrad};
}

inline Tensor tensor(const std::initializer_list<double> & data, const Shape & shape, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), shape, requireGrad, dtype};
}

inline Tensor tensor(const std::initializer_list<float> & data, const Shape & shape, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), shape, requireGrad, dtype};
}

inline Tensor tensor(const std::initializer_list<double> & data, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), {data.size()}, requireGrad, dtype};
}

inline Tensor tensor(const std::initializer_list<float> & data, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), {data.size()}, requireGrad, dtype};
}

inline Tensor tensor(const std::vector<double> & data, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.data(), data.size(), getDataType<double>(), {data.size()}, requireGrad, dtype};
}

inline Tensor tensor(const std::vector<float> & data, bool requireGrad = false, DataType dtype = DataType::kFloat32)
{
    return Tensor{data.data(), data.size(), getDataType<float>(), {data.size()}, requireGrad, dtype};
}

inline Tensor randn(const Shape & shape, bool requireGrad = false)
{
    std::uniform_real_distribution<float> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<float> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> float { return distr(randGen); });

    return Tensor{rndData.data(), rndData.size(), getDataType<float>(), shape, requireGrad};
}

inline Tensor ones(const Shape & shape, bool requireGrad = false)
{
    return Tensor{1, shape, requireGrad};
}

inline Tensor zeros(const Shape & shape, bool requireGrad = false)
{
    return Tensor{0, shape, requireGrad};
}

inline Tensor onesLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{1, tensor.shape(), requireGrad, tensor.dataType(), tensor.device()};
}

inline Tensor zerosLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{0, tensor.shape(), requireGrad, tensor.dataType(), tensor.device()};
}

inline Tensor sqrt(const Tensor & A)   { return A.sqrt(); }
inline Tensor sin(const Tensor & A)    { return A.sin();  }
inline Tensor cos(const Tensor & A)    { return A.cos();  }
inline Tensor tanh(const Tensor & A)   { return A.tanh(); }
inline Tensor log(const Tensor & A)    { return A.log();  }
inline Tensor exp(const Tensor & A)    { return A.exp();  }
inline Tensor sum(const Tensor & A)    { return A.sum();  }
inline Tensor mean(const Tensor & A)   { return A.mean(); }
inline Tensor pow(const Tensor & A, const Tensor & exp)     { return A.pow(exp); }
inline Tensor matmul(const Tensor & A, const Tensor & B)    { return A.matmul(B); }


// Optimizers Namespace


namespace optim
{

class Optimizer
{
public:
    // Constructor
    explicit Optimizer(const std::vector<Tensor> & parameters) : m_parameters(parameters) { }

    // Destructor
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zeroGrad()
    {
        for (auto & param : m_parameters)
        {
            param.zeroGrad();
        }
    }

protected:
    std::vector<Tensor> m_parameters;
};


class SGD : public Optimizer
{
public:
    explicit SGD(const std::vector<Tensor> & parameters, float lr = 0.01f)
        : Optimizer(parameters), m_lr(lr) { }

    void step() final
    {
        for (auto & param : m_parameters)
        {
            if (param.isRequireGrad())
            {
                param.value() -= param.grad() * m_lr;   // w' = w - lr * w_gradient.
            }
        }
    }

private:
    float m_lr;     // Learning rate
};


class Adam : public Optimizer
{
public:
    explicit Adam(const std::vector<Tensor> & parameters, float lr = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        for (const auto & param : m_parameters)
        {
            m_m.emplace_back(0, param.shape(), param.value().device());
            m_v.emplace_back(0, param.shape(), param.value().device());
        }
    }

    void step() final
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            if (m_parameters[i].isRequireGrad())
            {
                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + float(1.0 - m_beta1) * m_parameters[i].grad();

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + float(1.0 - m_beta2) * m_parameters[i].grad() * m_parameters[i].grad();

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / float(1.0 - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / float(1.0 - std::pow(m_beta2, m_timestep));

                // Update parameter.
                m_parameters[i].value() -= m_lr * mHat / (vHat.sqrt() + m_epsilon);
            }
        }
    }

private:
    float m_lr;                 // Learning rate.
    float m_beta1;              // Exponential decay rate for the first moment estimates.
    float m_beta2;              // Exponential decay rate for the second moment estimates.
    float m_epsilon;            // Small constant for numerical stability.
    size_t m_timestep{0};       // Time step.
    std::vector<TensorValue>    m_m;    // First moment vector.
    std::vector<TensorValue>    m_v;    // Second moment vector.
};

}   // optim namespace


// Neural Network Namespace


namespace nn
{

class Module
{
public:
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) const = 0;

    void registerParameter(Tensor & tensor)
    {
        m_parameters.emplace_back(tensor);
    }

    void registerModule(const Module & module)
    {
        for (const auto & param : module.parameters())
        {
            m_parameters.emplace_back(param);
        }
    }

    std::vector<Tensor> parameters() const
    {
        return m_parameters;
    }

    // Returns the total number of elements (learnable parameters) in each Tensor.
    size_t learnableParameters() const
    {
        size_t totalParams = 0;
        for (const auto & param: m_parameters)
        {
            if (param.isRequireGrad())
            {
                totalParams += param.value().size();
            }
        }
        return totalParams;
    }

    void to(Device & device)
    {
        for (auto & param : parameters())
        {
            param.to(device);
        }
    }

private:
    std::vector<Tensor> m_parameters;
};


class Sequential : public Module
{
public:
    // Override the forward function.
    Tensor forward(Tensor x) const override
    {
        for (const auto & module : m_modules)
        {
            x = module->forward(x);
        }
        return x;
    }

    // Function to add modules dynamically if needed.
    void add(Module* module)
    {
        registerModule(*module);
        m_modules.emplace_back(module);     // Use std::unique_ptr to take ownership of the module pointer.
    }

protected:
    // Use std::unique_ptr for polymorphic containment.
    std::vector<std::unique_ptr<Module>>  m_modules;
};


class Linear : public Module
{
public:
    // Constructor
    Linear(size_t numInputs, size_t numOutputs)
    {
        m_w1 = randn({numInputs, numOutputs}, true);        // A tensor filled with random numbers in [-1, 1].
        m_b1 = randn({1,         numOutputs}, true);

        // Register learnable parameters.
        registerParameter(m_w1);
        registerParameter(m_b1);
    }

    // Forward
    Tensor forward(Tensor x) const override
    {
        return matmul(x, m_w1) + m_b1;
    }

    Tensor  m_w1;
    Tensor  m_b1;
};


class Tanh : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return tanh(x);
    }
};


class Sigmoid : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 1 / (1 + exp(-x));
    }
};


class Softmax : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        auto expX = x.exp();
        return expX / expX.sum();
    }
};


class LogSoftmax : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        // LogSoftmax(x) = log(e^x / sum(e^x)) = log(e^x) - log(sum(e^x)) = x - log(sum(e^x))
        return x - x.exp().sum().log();
    }
};


class GeLU : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 0.5 * x * (1.0 + tanh(std::sqrtf(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
    }
};


class MSELoss
{
public:
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        auto diff = predictions - targets;
        auto loss = mean(diff * diff);
        return loss;
    }
};


class BinaryCrossEntropyLoss
{
public:
    // Prediction values must be in [0..1] range.
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        return -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions));
    }
};

}   // nn namespace


// Auxiliary Features


inline void save(const nn::Module & module, const std::string & filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::ios_base::failure("Failed to open file for writing.");
    }

    const auto params = module.parameters();
    for (auto param : params)
    {
        const auto & value = param.value();
        size_t size = value.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));                       // Save parameter size
        size_t paramDTypeSize = param.device()->dataTypeSize(param.dataType());
        ofs.write(reinterpret_cast<const char*>(value.data()), size * paramDTypeSize);       // Save parameter data
    }

    ofs.close();
}

inline void load(nn::Module & module, const std::string & filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::ios_base::failure("Failed to open model parameter file for reading.");
    }

    const auto params = module.parameters();    // Get model parameters
    for (auto param : params)
    {
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));         // Read size of parameter
        if (size != param.value().size())
        {
            throw std::runtime_error("Invalid parameter size found when loading the model.");
        }
        size_t paramDTypeSize = param.device()->dataTypeSize(param.dataType());
        ifs.read(reinterpret_cast<char*>(param.value().data()), size * paramDTypeSize); // Read the parameter data
    }

    ifs.close();
}

// Overload the << operator to print TensorValue.
std::ostream & operator<<(std::ostream& os, const TensorValue& tensor)
{
    switch (tensor.dataType())
    {
        case DataType::kFloat32:
            tensor.print<float>(os);
            break;

        case DataType::kFloat64:
            tensor.print<double>(os);
            break;

        default:
            throw std::runtime_error("Data type for print is not supported.");
            break;
    }
    return os;
}

// Overload the << operator to print Tensor.
std::ostream & operator<<(std::ostream& os, const Tensor& tensor)
{
    os << tensor.value();
    return os;
}

inline void manualSeed(size_t seed)
{
    randGen.seed(seed);
}

}   // aix namespace
