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
#include "aixDeviceCPU.hpp"
#include "aixFloat16.hpp"
// External includes
// System includes
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <stack>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace aix
{

enum class DataType : size_t
{
    kFloat64  = 0,
    kFloat32  = 1,
    kFloat16  = 2,
    kBFloat16 = 3,
    kInt64    = 4,
    kInt32    = 5,
    kInt16    = 6,
    kInt8     = 7,
    kUInt8    = 8,
};

constexpr size_t DataTypeCount   = 9;

// Primary template (default case)
template <typename T> constexpr DataType getDataType();
template <> constexpr DataType getDataType<double>()     { return DataType::kFloat64;  }
template <> constexpr DataType getDataType<float>()      { return DataType::kFloat32;  }
template <> constexpr DataType getDataType<float16_t>()  { return DataType::kFloat16;  }
template <> constexpr DataType getDataType<bfloat16_t>() { return DataType::kBFloat16; }
template <> constexpr DataType getDataType<int64_t>()    { return DataType::kInt64;    }
template <> constexpr DataType getDataType<int32_t>()    { return DataType::kInt32;    }
template <> constexpr DataType getDataType<int16_t>()    { return DataType::kInt16;    }
template <> constexpr DataType getDataType<int8_t>()     { return DataType::kInt8;     }
template <> constexpr DataType getDataType<uint8_t>()    { return DataType::kUInt8;    }


static DataType promoteDataType(DataType dtype1, DataType dtype2)
{
    assert(static_cast<size_t>(dtype1) < DataTypeCount && static_cast<size_t>(dtype2) < DataTypeCount);
    static_assert(static_cast<size_t>(DataType::kFloat64)  == 0);
    static_assert(static_cast<size_t>(DataType::kFloat32)  == 1);
    static_assert(static_cast<size_t>(DataType::kFloat16)  == 2);
    static_assert(static_cast<size_t>(DataType::kBFloat16) == 3);
    static_assert(static_cast<size_t>(DataType::kInt64)    == 4);
    static_assert(static_cast<size_t>(DataType::kInt32)    == 5);
    static_assert(static_cast<size_t>(DataType::kInt16)    == 6);
    static_assert(static_cast<size_t>(DataType::kInt8)     == 7);
    static_assert(static_cast<size_t>(DataType::kUInt8)    == 8);

    static const size_t promotionTable[DataTypeCount][DataTypeCount] =
    {
        //                  F  F  F  B  I  I  I  I  U
        //                  6  3  1  1  6  3  1  8  8
        //                  4  2  6  6  4  2  6
        /*  kFloat64  */  { 0, 0, 0, 0, 0, 0, 0, 0, 0, },
        /*  kFloat32  */  { 0, 1, 1, 1, 1, 1, 1, 1, 1, },
        /*  kFloat16  */  { 0, 1, 2, 1, 2, 2, 2, 2, 2, },
        /*  kBFloat16 */  { 0, 1, 1, 3, 3, 3, 3, 3, 3, },
        /*  kInt64    */  { 0, 1, 2, 3, 4, 4, 4, 4, 4, },
        /*  kInt32    */  { 0, 1, 2, 3, 4, 5, 5, 5, 5, },
        /*  kInt16    */  { 0, 1, 2, 3, 4, 5, 6, 6, 6, },
        /*  kInt8     */  { 0, 1, 2, 3, 4, 5, 6, 7, 6, },
        /*  kUInt8    */  { 0, 1, 2, 3, 4, 5, 6, 6, 8, },
    };

    return static_cast<DataType>(promotionTable[static_cast<size_t>(dtype1)][static_cast<size_t>(dtype2)]);
}

// Promotes a data type to Float32 if the type is an integer type, otherwise it returns the same float data type.
static DataType promoteDataTypeToFloat(DataType dtype)
{
    assert(static_cast<size_t>(dtype) < DataTypeCount);
    static const size_t formatConversionTable[DataTypeCount] =
    {
    //  F  F  F  B  I  I  I  I  U
    //  6  3  1  1  6  3  1  8  8
    //  4  2  6  6  4  2  6
        0, 1, 2, 3, 1, 1, 1, 1, 1,
    };
    return static_cast<DataType>(formatConversionTable[static_cast<size_t>(dtype)]);
}

// Forward declarations
class Tensor;

// Tensor Index, Shape and Stride Types
using Index  = std::vector<size_t>;
using SIndex = std::vector<ssize_t>;
using Shape  = std::vector<size_t>;
using Stride = std::vector<size_t>;

struct DeviceTensorParams
{
    void*    data{nullptr};
    DataType dtype{aix::DataType::kFloat32};
    bool     isContiguous{true};
    size_t   offset{0};         // Start offset of data on storage.
    Shape    shape{};           // The shape of the tensor.
    size_t   size{0};           // Number of elements in DataType.
    Stride   strides{};         // The strides for indexing the tensor.
};

// TODO: Global parameters needs to move to a global context.
static DeviceCPU defaultDevice;
static std::random_device randomDevice;
static std::mt19937 randGen(randomDevice());


class TensorStorage
{
public:
    TensorStorage() = default;

    explicit TensorStorage(Device* device, size_t size) : m_device{device}, m_size{size}
    {
        m_data = device->allocate(size);
    }

    explicit TensorStorage(Device* device, size_t size, aix::DataType dtype) : m_device{device}
    {
        m_data = device->allocate(size, dtype);
        m_size = size * aix::Device::dataTypeSize(dtype);
    }

    virtual ~TensorStorage()
    {
        if (m_device && m_data)
        {
            m_device->deallocate(m_data);
        }
    }

    inline Device* device()             { return m_device;  }
    inline void*   data()               { return m_data;    }
    inline const void* data() const     { return m_data;    }
    inline size_t  size() const         { return m_size;    }

private:
    Device*   m_device{nullptr};
    void*     m_data{nullptr};
    size_t    m_size{0};
};


class TensorValue
{
public:
    // Constructor
    TensorValue() = default;

    // Constructor
    TensorValue(const void* data, size_t size, DataType srcDType, Shape shape, Device* device,
                DataType dType = DataType::kFloat32) : m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        validateSize(size, m_shape);
        m_storage = std::make_shared<TensorStorage>(device, size, dType);
        device->copy(data, srcDType, m_storage->data(), dType, size);
        m_size = size;
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(const std::shared_ptr<TensorStorage>& storage, size_t size, size_t offset, Shape shape, Device* device,
                DataType dType = DataType::kFloat32) : m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        assert(storage->device() == device);
        m_storage = storage;
        m_size = size;
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
        m_offset = offset;
    }

    TensorValue(const std::shared_ptr<TensorStorage>& storage, size_t size, size_t offset, Shape shape, Stride strides,
                Device* device, DataType dType = DataType::kFloat32) : m_dType(dType), m_shape(std::move(shape)),
                m_strides(std::move(strides)), m_device(device)
    {
        assert(storage->device() == device);
        m_storage = storage;
        m_size = size;
        m_offset = offset;
    }

    // Constructor
    template<typename T>
    TensorValue(const std::initializer_list<T> & data, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        validateSize(data.size(), m_shape);
        m_storage = std::make_shared<TensorStorage>(device, data.size(), dType);
        device->copy(data.begin(), getDataType<T>(), m_storage->data(), dType, data.size());
        m_size = data.size();
        // Compute the strides for indexing multi-dimensional data.
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(float value, Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), static_cast<size_t>(1), std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_storage = std::make_shared<TensorStorage>(device, m_size, dType);
        m_strides = computeStrides();
        // initialize data.
        device->fill(&value, DataType::kFloat32, deviceParams());
    }

    // Constructor
    TensorValue(Shape shape, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape(std::move(shape)), m_device(device)
    {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), static_cast<size_t>(1), std::multiplies<>());
        // Each tensor array must use device specific memory allocator.
        m_storage = std::make_shared<TensorStorage>(device, m_size, dType);
        m_strides = computeStrides();
    }

    // Constructor
    TensorValue(Shape shape, Device * device, size_t size, Stride strides, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_size(size), m_shape(std::move(shape)), m_strides(std::move(strides)), m_device(device)
    {
        validateSize(m_size, m_shape);
        // Each tensor array must use device specific memory allocator.
        m_storage = std::make_shared<TensorStorage>(device, m_size, dType);
        m_isContiguous = computeIsContiguous();
    }

    // Constructor
    TensorValue(float value, Device * device, DataType dType = DataType::kFloat32) :
        m_dType(dType), m_shape{}, m_device(device)
    {
        // Each tensor array must use device specific memory allocator.
        m_size = 1;
        m_storage = std::make_shared<TensorStorage>(device, m_size, dType);
        m_strides = computeStrides();
        device->fill(&value, DataType::kFloat32, deviceParams());
    }

    // Destructor
    ~TensorValue()
    {
        m_device = nullptr;
    }

    // Copy constructor
    TensorValue(const TensorValue& other)
    {
        m_dType   = other.m_dType;
        m_shape   = other.m_shape;
        m_device  = other.m_device;
        m_size    = size();
        m_storage = std::make_shared<TensorStorage>(m_device, m_size, m_dType);

        if (other.isContiguous() && other.m_offset == 0)
        {
            m_strides = other.m_strides;
            m_device->copy(other.data(), other.m_dType, data(), other.m_dType, m_size);
        }
        else
        {
            m_strides = computeStrides();
            other.m_device->contiguous(other.deviceParams(), deviceParams());
        }
    }

    // Copy assignment operator
    TensorValue& operator=(const TensorValue& other)
    {
        if (this != &other)
        {
            m_dType   = other.m_dType;
            m_shape   = other.m_shape;
            m_device  = other.m_device;
            m_size    = size();
            m_storage = std::make_shared<TensorStorage>(m_device, m_size, m_dType);

            if (other.isContiguous() && other.m_offset == 0)
            {
                m_strides = other.m_strides;
                m_device->copy(other.data(), other.m_dType, data(), other.m_dType, m_size);
            }
            else
            {
                m_strides = computeStrides();
                other.m_device->contiguous(other.deviceParams(), deviceParams());
            }
        }

        return *this;
    }

    // Move constructor
    TensorValue(TensorValue&& other) noexcept
    {
        m_dType         = other.m_dType;
        m_storage       = other.m_storage;
        m_size          = other.m_size;
        m_offset        = other.m_offset;
        m_shape         = other.m_shape;
        m_strides       = other.m_strides;
        m_isContiguous  = other.m_isContiguous;
        m_device        = other.m_device;
        other.m_size    = 0;
        other.m_offset  = 0;
        other.m_device  = nullptr;
    }

    // Move assignment operator
    TensorValue& operator=(TensorValue&& other) noexcept
    {
        if (this != &other)
        {
            m_dType         = other.m_dType;
            m_storage       = other.m_storage;
            m_size          = other.m_size;
            m_offset        = other.m_offset;
            m_shape         = other.m_shape;
            m_strides       = other.m_strides;
            m_isContiguous  = other.m_isContiguous;
            m_device        = other.m_device;
            other.m_size    = 0;
            other.m_offset  = 0;
            other.m_device  = nullptr;
        }

        return *this;
    }

    // Select operator.
    TensorValue operator[](ssize_t index) const
    {
        return select(0, index);
    }

    // Access element at a specific index (non-const version).
    template<typename T>
    T & getValueAt(const Index & indices)     { return static_cast<T*>(m_storage->data())[getIndex(indices)]; }

    // Access element at a specific index (const version).
    template<typename T>
    T getValueAt(const Index & indices) const { return static_cast<T*>(m_storage->data())[getIndex(indices)]; }

    // Get the data type of the tensor.
    DataType dataType() const      { return m_dType; }

    // Get the shape of the tensor
    const Shape & shape() const    { return m_shape; }

    // Get the strides of the tensor
    const Stride & strides() const  { return m_strides; }

    // Get the raw data of the tensor.
    const void* data() const    { return m_storage->data(); }
    void* data()                { return m_storage->data(); }

    // Get storage of the tensor.
    inline const std::shared_ptr<TensorStorage>& storage()  { return m_storage; };
    inline size_t storageOffset() const                     { return m_offset; };

    // Get the raw data of the tensor.
    template<typename T>
    const T* data() const       { return static_cast<T*>(m_storage->data()); }
    template<typename T>
    T* data()                   { return static_cast<T*>(m_storage->data()); }

    // Get the size of the data
    size_t size() const
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), static_cast<size_t>(1), std::multiplies<>());
    }

    // Get the device
    Device * device() const     { return m_device; }

    // Get device tensor parameters.
    DeviceTensorParams deviceParams() const
    {
        return { .data=m_storage->data(), .dtype=m_dType, .isContiguous=isContiguous(), .offset=m_offset,
                 .shape=m_shape, .size=size(), .strides=m_strides };
    };

    // Set the device
    TensorValue to(Device * device) const
    {
        if (m_device == device) return shallowCopy();
        auto materialized = isContiguous() && m_offset == 0 ? shallowCopy() : contiguous();
        return {materialized.data(), materialized.size(), materialized.dataType(), materialized.shape(), device, materialized.dataType()};
    }
    inline TensorValue to(std::unique_ptr<Device>& device) const    { return to(device.get()); }
    inline TensorValue to(std::shared_ptr<Device>& device) const    { return to(device.get()); }

    template<typename T>
    T item() const
    {
        if (!m_shape.empty())    // Scalar value must have no dimension.
        {
            throw std::invalid_argument("Tensor is not a scalar.");
        }
        return static_cast<T*>(m_storage->data())[0];
    }

    // Returns a new TensorValue with a new shape.
    TensorValue reshape(const Shape & newShape) const
    {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), static_cast<size_t>(1), std::multiplies<>());
        if (m_size != newSize)
        {
            throw std::invalid_argument("Reshape error: element count mismatch (" +
                                        std::to_string(m_size) + " vs " + std::to_string(newSize) + ").");
        }

        if (newShape == m_shape)
        {
            return shallowCopy();
        }

        if (isContiguous())
        {
            return {m_storage, m_size, m_offset, newShape, m_device, m_dType};
        }

        if (auto newStrides = computeReshapeViewStrides(m_shape, m_strides, newShape, m_size))
        {
            return {m_storage, m_size, m_offset, newShape, *newStrides, m_device, m_dType};
        }

        return contiguous().reshape(newShape);
    }

    // Equalize tensor data types by promoting data type of tensors.
    TensorValue to(DataType newDataType) const
    {
        if (dataType() != newDataType)
        {
            auto materialized = isContiguous() && m_offset == 0 ? shallowCopy() : contiguous();
            return {materialized.data(), materialized.size(), materialized.dataType(), materialized.shape(), device(), newDataType};
        }
        return shallowCopy();
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
        if (shape() == newShape) return shallowCopy();
        if (!checkBroadcastTo(shape(), newShape))
        {
            throw std::invalid_argument("Target TensorValue shape is not broadcastable.");
        }

        // Calculate new strides for broadcasting.
        std::vector<size_t> newStrides(newShape.size(), 0);
        for (int i = m_shape.size() - 1, j = newShape.size() - 1; j >= 0; --i, --j)
        {
            if (i < 0)
            {
                newStrides[j] = 0;      // Broadcast dimension.
            }
            else if (m_shape[i] == newShape[j])
            {
                newStrides[j] = m_strides[i];
            }
            else
            {
                newStrides[j] = 0;      // Broadcast dimension.
            }
        }

        // Create a new TensorValue that shares the same storage.
        TensorValue result(m_storage, m_size, m_offset, newShape, m_device, m_dType);
        result.m_strides = std::move(newStrides);
        return result;
    }

    // Reduces the TensorValue back to the original shape.
    TensorValue reduceTo(const Shape & originalShape) const
    {
        if (shape() == originalShape) return shallowCopy();
        // Ensure tensor values are initialized to zero, as the reduction operation performs a summation.
        TensorValue result(0, originalShape, device(), m_dType);
        device()->reduceTo(deviceParams(), result.deviceParams());
        return result;
    }

    // Returns true if the tensor is contiguous.
    bool isContiguous() const
    {
        return computeIsContiguous();
    }

    TensorValue contiguous() const
    {
        if (isContiguous() && m_offset == 0) return shallowCopy();

        TensorValue result(m_shape, m_device, m_dType);
        m_device->contiguous(deviceParams(), result.deviceParams());
        return result;
    }

    // Operators

    // Overload the + operator
    TensorValue operator+(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::add, other);
    }

    // Overload the - operator
    TensorValue operator-(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::sub, other);
    }

    // Overload the * operator
    TensorValue operator*(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::mul, other);
    }

    // Overload the / operator
    TensorValue operator/(const TensorValue & other) const
    {
        return arithmeticOpFunc(&Device::div, other);
    }

    // Overload the += operator - In-place operation.
    TensorValue & operator+=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::add, other);
    }

    // Overload the -= operator - In-place operation.
    TensorValue & operator-=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::sub, other);
    }

    // Overload the *= operator - In-place operation.
    TensorValue & operator*=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::mul, other);
    }

    // Overload the /= operator - In-place operation.
    TensorValue & operator/=(const TensorValue & other)
    {
        return arithmeticInPlaceOpFunc(&Device::div, other);
    }

    // Overload the unary - operator
    TensorValue operator-() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->unary(deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue operator+(float scalar) const
    {
        return *this + TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator-(float scalar) const
    {
        return *this - TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator*(float scalar) const
    {
        return *this * TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue operator/(float scalar) const
    {
        return *this / TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue& operator+=(float scalar)
    {
        return *this += TensorValue{scalar, m_shape, m_device, promoteDataTypeToFloat(m_dType)};
    }

    TensorValue& operator-=(float scalar)
    {
        return *this -= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    TensorValue& operator*=(float scalar)
    {
        return *this *= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    TensorValue& operator/=(float scalar)
    {
        return *this /= TensorValue{scalar, m_shape, m_device, m_dType};
    }

    friend TensorValue operator+(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} + tensor;
    }

    friend TensorValue operator-(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} - tensor;
    }

    friend TensorValue operator*(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} * tensor;
    }

    friend TensorValue operator/(float scalar, const TensorValue & tensor)
    {
        auto promotedDType = promoteDataTypeToFloat(tensor.dataType());
        return TensorValue{scalar, tensor.shape(), tensor.device(), promotedDType} / tensor;
    }

    void fill(float value) const
    {
        m_device->fill(&value, DataType::kFloat32, deviceParams());
    }

    TensorValue sum() const
    {
        TensorValue result({}, device(), m_dType);
        m_device->sum(deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue sum(ssize_t dim, bool keepDim=false) const
    {
        if (m_shape.empty()) return shallowCopy();      // Return itself if it's a scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::sum() is out of range.");
        }

        Shape resultShape = m_shape;
        resultShape[dim] = 1;
        auto result = reduceTo(resultShape);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue mean() const
    {
        return sum() / size();
    }

    TensorValue mean(ssize_t dim, bool keepDim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        return sum(dim, keepDim) / shape()[dim];
    }

    TensorValue sqrt() const
    {
        return tensorMathFunc(&Device::sqrt);
    }

    TensorValue sin() const
    {
        return tensorMathFunc(&Device::sin);
    }

    TensorValue cos() const
    {
        return tensorMathFunc(&Device::cos);
    }

    TensorValue tanh() const
    {
        return tensorMathFunc(&Device::tanh);
    }

    TensorValue log() const
    {
        return tensorMathFunc(&Device::log);
    }

    TensorValue exp() const
    {
        return tensorMathFunc(&Device::exp);
    }

    TensorValue pow(const TensorValue & exp) const
    {
        if (shape() != exp.shape() || dataType() != exp.dataType())
        {
            TensorValue lhs = shallowCopy();
            TensorValue rhs = exp.shallowCopy();
            auto result = prepareTensors(lhs, rhs);
            result.device()->pow(lhs.deviceParams(), rhs.deviceParams(), result.deviceParams());
            return result;
        }

        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, m_dType);
        m_device->pow(deviceParams(), exp.deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue max() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result({}, m_device, m_dType);
        m_device->max(deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue max(ssize_t dim, bool keepDim=false) const
    {
        if (m_shape.empty()) return shallowCopy();      // Return itself if it's a scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::max() is out of range.");
        }

        Shape newShape = m_shape;
        newShape[dim] = 1;

        TensorValue result(newShape, device(), m_dType);            // Zero initialization is not required.
        auto resDevParams = result.deviceParams();
        device()->fillMin(resDevParams);       // Initialize the tensor with the lowest value.
        device()->maxTo(deviceParams(), resDevParams);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue argmax() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result({}, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmax(deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue argmax(ssize_t dim, bool keepDim=false) const
    {
        if (m_shape.empty()) return {0, m_shape, m_device, aix::DataType::kInt32};  // Scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::argmax() is out of range.");
        }

        Shape newShape = m_shape;
        newShape[dim] = 1;

        TensorValue result(newShape, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmaxTo(deviceParams(), result.deviceParams(), dim);
        return keepDim ? result : result.squeeze(dim);
    }

    TensorValue argmaxIndices() const
    {
        // Create a new TensorValue to store the result. Perform element-wise.
        TensorValue result(m_shape, m_device, aix::DataType::kInt32);   // Index is by default in int32 type.
        m_device->argmaxIndices(deviceParams(), result.deviceParams());
        return result;
    }

    TensorValue argmaxIndices(ssize_t dim) const
    {
        if (m_shape.empty()) return {1, m_shape, m_device, aix::DataType::kInt32};  // Scalar tensor.

        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of TensorValue::argmaxIndices() is out of range.");
        }

        TensorValue result(0, m_shape, m_device, aix::DataType::kInt32);        // Index is by default in int32 type.
        m_device->argmaxIndicesTo(deviceParams(), result.deviceParams(), dim);
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

        // Convert tensors to the promoted data type if necessary.
        if (dataType() != b.dataType())
        {
            TensorValue lhs = shallowCopy();
            TensorValue rhs = b.shallowCopy();
            auto promotedDType = promoteDataType(lhs.dataType(), rhs.dataType());
            lhs = lhs.to(promotedDType);
            rhs = rhs.to(promotedDType);
            TensorValue result(resultShape, lhs.device(), promotedDType);
            result.device()->matmul(lhs.deviceParams(), rhs.deviceParams(), result.deviceParams());
            return result;
        }

        // Result tensor shape.
        TensorValue result(resultShape, m_device, m_dType);
        m_device->matmul(deviceParams(), b.deviceParams(), result.deviceParams());
        return result;
    }

    // Generalized transpose function.
    TensorValue transpose(ssize_t dim0, ssize_t dim1) const
    {
        auto shapeSize = static_cast<ssize_t>(shape().size());
        dim0 = dim0 < 0 ? shapeSize + dim0 : dim0;
        dim1 = dim1 < 0 ? shapeSize + dim1 : dim1;
        if (dim0 < 0 || dim0 >= shapeSize || dim1 < 0 || dim1 >= shapeSize)
        {
            throw std::invalid_argument("Dimension is out of range for transpose.");
        }
        Shape newShape = m_shape;
        Stride newStrides = m_strides;
        std::swap(newShape[dim0], newShape[dim1]);
        std::swap(newStrides[dim0], newStrides[dim1]);

        return {m_storage, m_size, m_offset, newShape, newStrides, m_device, m_dType};
    }

    TensorValue permute(SIndex newDims) const
    {
        if (newDims.size() != shape().size())
        {
            throw std::invalid_argument("Dimension count does not match in permute.");
        }

        if (shape().empty()) return shallowCopy();      // Nothing to do for a scalar tensor.
        if (shape().size() == 1 && (newDims[0] == 0 || newDims[0] == -1)) return shallowCopy();

        auto shapeSize = static_cast<ssize_t>(shape().size());
        std::vector<ssize_t> dimTable(shapeSize, -1);

        // Check if it's an identity permutation and validate dimensions.
        bool isIdentity = true;
        for (ssize_t i = 0; i < shapeSize; ++i)
        {
            auto& dim = newDims[i];
            dim = dim < 0 ? shapeSize + dim : dim;
            if (dim < 0 || dim >= shapeSize)
            {
                throw std::invalid_argument("Dimension is out of range for permute.");
            }
            if (dimTable[dim] != -1)
            {
                throw std::invalid_argument("There is at least one repeated dim in permute.");
            }
            dimTable[dim] = i;
            if (dim != i) isIdentity = false;
        }

        if (isIdentity) return shallowCopy();

        // Create the new shape.
        Shape newShape(shapeSize);
        Stride newStrides(shapeSize);
        for (ssize_t i = 0; i < shapeSize; ++i)
        {
            newShape[i] = m_shape[newDims[i]];
            newStrides[i] = m_strides[newDims[i]];
        }

        return {m_storage, m_size, m_offset, newShape, newStrides, m_device, m_dType};
    }

    TensorValue squeeze(ssize_t dim) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        if (dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Invalid dimension for squeeze.");
        }

        if (m_shape[dim] == 1)
        {
            auto squeezedShape = m_shape;
            auto squeezedStrides = m_strides;
            squeezedShape.erase(squeezedShape.begin() + dim);
            squeezedStrides.erase(squeezedStrides.begin() + dim);
            return {m_storage, m_size, m_offset, squeezedShape, squeezedStrides, m_device, m_dType};
        }
        return shallowCopy();
    }

    TensorValue unsqueeze(ssize_t dim) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        if (dim > static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Invalid dimension for unsqueeze.");
        }

        auto unsqueezedShape = m_shape;
        auto unsqueezedStrides = m_strides;
        unsqueezedShape.insert(unsqueezedShape.begin() + dim, 1);

        size_t insertedStride = 1;
        if (!m_shape.empty())
        {
            if (dim < static_cast<ssize_t>(m_shape.size()))
            {
                insertedStride = m_strides[dim] * m_shape[dim];
            }
            else
            {
                insertedStride = 1;
            }
        }

        unsqueezedStrides.insert(unsqueezedStrides.begin() + dim, insertedStride);
        return {m_storage, m_size, m_offset, unsqueezedShape, unsqueezedStrides, m_device, m_dType};
    }

    TensorValue slice(ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                      std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("slice() cannot be applied to a 0-dim, a scalar tensor.");
        }

        if (step < 1)
        {
            throw std::invalid_argument("Slice step must be greater than zero.");
        }

        // Normalize dimension index.
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of slice() is out of range.");
        }

        // Handle start and end indices.
        ssize_t start = startOpt.value_or(0);
        ssize_t end   = endOpt.value_or(static_cast<ssize_t>(m_shape[dim]));

        // Normalize negative indices.
        start = start < 0 ? static_cast<ssize_t>(m_shape[dim]) + start : start;
        end   = end   < 0 ? static_cast<ssize_t>(m_shape[dim]) + end   : end;

        // Clamp the start and end indices within valid bounds.
        start = std::max<ssize_t>(0, std::min<ssize_t>(start, static_cast<ssize_t>(m_shape[dim])));
        end   = std::max<ssize_t>(0, std::min<ssize_t>(end,   static_cast<ssize_t>(m_shape[dim])));

        if (start >= end)
        {
            throw std::invalid_argument("Start index of slice() must be less than end index.");
        }

        // Compute new size along the sliced dimension.
        size_t newSizeInDim = (end - start + step - 1) / step;

        // Compute new offset.
        size_t newOffset = m_offset + start * m_strides[dim];

        // Compute new strides.
        auto newStrides = m_strides;
        newStrides[dim] *= step;

        // Compute new shape.
        auto newShape = m_shape;
        newShape[dim] = newSizeInDim;

        auto newSize = std::accumulate(newShape.begin(), newShape.end(), static_cast<size_t>(1), std::multiplies<>());

        TensorValue result(m_storage, newSize, newOffset, newShape, device(), dataType());
        result.m_strides = newStrides;
        return result;
    }

    TensorValue sliceSet(const TensorValue& tensor, ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                         std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1, bool inPlace=false) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("slice() cannot be applied to a 0-dim, a scalar tensor.");
        }

        if (step < 1)
        {
            throw std::invalid_argument("Slice step must be greater than zero.");
        }

        // Normalize dimension index.
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(m_shape.size()))
        {
            throw std::invalid_argument("Dimension parameter of slice() is out of range.");
        }

        // Handle start and end indices.
        ssize_t start = startOpt.value_or(0);
        ssize_t end   = endOpt.value_or(static_cast<ssize_t>(m_shape[dim]));

        // Normalize negative indices.
        start = start < 0 ? static_cast<ssize_t>(m_shape[dim]) + start : start;
        end   = end   < 0 ? static_cast<ssize_t>(m_shape[dim]) + end   : end;

        // Clamp the start and end indices within valid bounds.
        start = std::max<ssize_t>(0, std::min<ssize_t>(start, static_cast<ssize_t>(m_shape[dim])));
        end   = std::max<ssize_t>(0, std::min<ssize_t>(end,   static_cast<ssize_t>(m_shape[dim])));

        if (start >= end)
        {
            throw std::invalid_argument("Start index of slice() must be less than end index.");
        }

        // Calculate the new shape for the sliced tensor.
        Shape newShape = m_shape;
        newShape[dim] = (end - start + step - 1) / step; // This computes the size along the slicing dimension.

        if (tensor.shape() != newShape)
        {
            throw std::invalid_argument("The tensor's shape does not match the new shape of sliceSet().");
        }

        if (inPlace)
        {
            // Slice and set tensor's data to the result tensor.
            device()->sliceSet(tensor.deviceParams(), deviceParams(), dim, start, end, step);
            return shallowCopy();
        }

        TensorValue result(0, m_shape, device(), m_dType);  // Zero initialization is required.
        // Slice and set tensor's data to the result tensor.
        device()->sliceSet(tensor.deviceParams(), result.deviceParams(), dim, start, end, step);
        return result;
    }

    TensorValue select(ssize_t dim, ssize_t index) const
    {
        if (m_shape.empty())
        {
            throw std::invalid_argument("select() cannot be applied to a scalar, zero-dimension, tensor.");
        }
        dim = dim < 0 ? static_cast<ssize_t>(m_shape.size()) + dim : dim;
        index = index < 0 ? static_cast<ssize_t>(m_shape[dim]) + index : index;
        return slice(dim, index, index + 1, 1).squeeze(dim);
    }

    TensorValue indexSelect(ssize_t dim, const TensorValue& indices) const
    {
        if (indices.shape().size() > 1)
        {
            throw std::invalid_argument("Indices supposed to be a vector.");
        }
        if (indices.dataType() != aix::DataType::kInt32)
        {
            throw std::invalid_argument("Indices tensor's data type must be int32");
        }

        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;

        if ((!shape().empty() && (dim < 0 || static_cast<size_t>(dim) >= shape().size())) ||
            (shape().empty() && dim != 0))
        {
            throw std::invalid_argument("Dimension is out of range for indexSelect operation.");
        }

        auto newShape = shape();
        if (!newShape.empty())
        {
            newShape[dim] = !indices.shape().empty() ? indices.shape()[0] : 1;
        }

        assert(checkMinMaxValueOverflow(0, (!shape().empty() ? shape()[dim] : 0), indices));

        TensorValue result(newShape, device(), dataType());
        device()->indexSelect(deviceParams(), result.deviceParams(), indices.deviceParams(), dim);
        return result;
    }

    TensorValue indexAdd(ssize_t dim, const TensorValue& indices, const TensorValue& source, bool inPlace=false) const
    {
        if (indices.shape().size() > 1)
        {
            throw std::invalid_argument("Indices supposed to be a vector.");
        }
        if (indices.dataType() != aix::DataType::kInt32)
        {
            throw std::invalid_argument("Indices tensor's data type must be int32");
        }

        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;

        if ((!shape().empty() && (dim < 0 || static_cast<size_t>(dim) >= shape().size())) ||
            (shape().empty() && dim != 0))
        {
            throw std::invalid_argument("Dimension is out of range for indexSelect operation.");
        }

        auto newShape = shape();
        if (!newShape.empty())
        {
            newShape[dim] = !indices.shape().empty() ? indices.shape()[0] : 1;
        }

        if (newShape != source.shape())
        {
            throw std::invalid_argument("Source shape does not match the tensor's shape.");
        }

        if (dataType() != source.dataType())
        {
            throw std::invalid_argument("Source data type does not match the tensor's data type.");
        }

        assert(checkMinMaxValueOverflow(0, (!shape().empty() ? shape()[dim] : 0), indices));

        if (inPlace)
        {
            device()->indexAdd(source.deviceParams(), deviceParams(), indices.deviceParams(), dim);
            return shallowCopy();
        }

        TensorValue result(data(), size(), dataType(), shape(), device(), dataType());
        device()->indexAdd(source.deviceParams(), result.deviceParams(), indices.deviceParams(), dim);
        return result;
    }

    std::vector<TensorValue> split(ssize_t splitSize, ssize_t dim=0) const
    {
        if (splitSize < 0)
        {
            throw std::invalid_argument("Split size must be a positive number.");
        }

        if (m_shape.empty())
        {
            throw std::invalid_argument("Split operation needs at least a 1-dim tensor.");
        }

        const auto shapeSize = static_cast<ssize_t>(m_shape.size());
        dim = dim < 0 ? shapeSize + dim : dim;
        if (dim < 0 || dim >= shapeSize)
        {
            throw std::invalid_argument("Split dimension is out of range.");
        }

        std::vector<TensorValue> tensors;       // Stores splitted tensors.
        for (size_t i=0; i<m_shape[dim]; i+=splitSize)
        {
            tensors.emplace_back(slice(dim, i, i + splitSize, 1));
        }
        return tensors;
    }

    TensorValue tril(ssize_t diagonal=0) const
    {
        if (m_shape.size() < 2)
        {
            throw std::invalid_argument("Tensor must have at least two dimensions for tril operation.");
        }

        TensorValue result(m_shape, m_device, m_dType);
        device()->tril(deviceParams(), result.deviceParams(), diagonal);
        return result;
    }

    TensorValue triu(ssize_t diagonal=0) const
    {
        if (m_shape.size() < 2)
        {
            throw std::invalid_argument("Tensor must have at least two dimensions for triu operation.");
        }

        TensorValue result(m_shape, m_device, m_dType);
        device()->triu(deviceParams(), result.deviceParams(), diagonal);
        return result;
    }

    static TensorValue cat(const std::vector<TensorValue>& tensors, ssize_t dim)
    {
        if (tensors.empty())
        {
            throw std::invalid_argument("cat() operation needs at least one tensor.");
        }

        const auto& tensor = tensors[0];

        if (tensor.shape().empty())
        {
            throw std::invalid_argument("Zero-dimensional tensor cannot be concatenated.");
        }

        if (tensors.size() == 1) return tensor.shallowCopy();

        dim = dim < 0 ? static_cast<ssize_t>(tensor.shape().size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(tensor.shape().size()))
        {
            throw std::invalid_argument("Dimension is out of range for cat() operation.");
        }

        DataType promotedDType = tensor.dataType();
        for (size_t i=0; i<tensors.size()-1; ++i)
        {
            auto shape1 = tensors[i].shape();
            auto shape2 = tensors[i+1].shape();
            shape1[dim] = shape2[dim] = 0;      // Neutralize dimension sizes for comparison.

            if (shape1 != shape2)
            {
                throw std::invalid_argument("Dimension sizes of tensors must match except in dimension " +
                                            std::to_string(dim) + " for the cat() operation.");
            }
            // Tensor devices must be the same.
            if (tensors[i].device() != tensors[i+1].device())
            {
                throw std::invalid_argument("Tensor devices must be the same for the cat() operation.");
            }
            promotedDType = promoteDataType(promotedDType, tensors[i+1].dataType());
        }

        auto newShape = tensor.shape();
        for (size_t i=1; i<tensors.size(); ++i)
            newShape[dim] += tensors[i].shape()[dim];

        size_t dimSize = 0;
        TensorValue result(newShape, tensor.device(), promotedDType);
        for (size_t i=0; i<tensors.size(); ++i)
        {
            result.sliceSet(tensors[i].to(promotedDType), dim, dimSize, dimSize + tensors[i].shape()[dim], 1, true);
            dimSize += tensors[i].shape()[dim];
        }
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream& operator<<(std::ostream & os, const TensorValue & tensor);

private:
    TensorValue shallowCopy() const
    {
        return {m_storage, m_size, m_offset, m_shape, m_strides, m_device, m_dType};
    }

    template<typename T>
    inline TensorValue arithmeticOpFunc(const T & func, const TensorValue & other) const
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue lhs = shallowCopy();
            TensorValue rhs = other.shallowCopy();
            auto result = prepareTensors(lhs, rhs);
            (result.device()->*func)(lhs.deviceParams(), rhs.deviceParams(), result.deviceParams());
            return result;
        }
        TensorValue result(m_shape, m_device, m_dType);
        (m_device->*func)(deviceParams(), other.deviceParams(), result.deviceParams());
        return result;
    }

    template<typename T>
    inline TensorValue & arithmeticInPlaceOpFunc(const T & func, const TensorValue & other)
    {
        if (shape() != other.shape() || dataType() != other.dataType())
        {
            TensorValue rhs = (dataType() != other.dataType()) ? other.to(m_dType) : other.shallowCopy();
            if (shape() != rhs.shape())
            {
                TensorValue lhs = shallowCopy();
                auto bcShape = broadcastShapes(lhs.shape(), rhs.shape());
                lhs = lhs.broadcastTo(bcShape);
                rhs = rhs.broadcastTo(bcShape);
                TensorValue result(lhs.shape(), lhs.device(), lhs.dataType());
                (m_device->*func)(lhs.deviceParams(), rhs.deviceParams(), result.deviceParams());
                *this = std::move(result);
            }
            else
            {
                (m_device->*func)(deviceParams(), rhs.deviceParams(), deviceParams());
            }
            return *this;
        }
        (m_device->*func)(deviceParams(), other.deviceParams(), deviceParams());
        return *this;
    }

    template<typename T>
    inline TensorValue tensorMathFunc(const T & func) const
    {
        auto promotedDType = promoteDataTypeToFloat(m_dType);
        if (dataType() != promotedDType)
        {
            // This constructor requires copy operation.
            TensorValue result(data(), m_size, m_dType, m_shape, m_device, promotedDType);
            (m_device->*func)(result.deviceParams(), result.deviceParams());
            return result;
        }
        // This constructor does not require copy operation.
        TensorValue result(m_shape, m_device, m_dType);
        (m_device->*func)(deviceParams(), result.deviceParams());
        return result;
    }

    bool computeIsContiguous() const
    {
        if (m_shape.empty()) return true;
        Stride expectedStrides = computeStrides();
        return m_strides == expectedStrides;
    }

    static std::optional<Stride> computeReshapeViewStrides(const Shape& oldShape, const Stride& oldStrides,
                                                           const Shape& newShape, size_t size)
    {
        if (oldShape == newShape)
        {
            return oldStrides;
        }

        if (size <= 1)
        {
            return computeContiguousStrides(newShape);
        }

        for (size_t i = 0; i < oldStrides.size(); ++i)
        {
            if (oldStrides[i] == 0)
            {
                return std::nullopt;
            }
        }

        struct Chunk
        {
            size_t numel;
            size_t baseStride;
        };

        std::vector<Chunk> chunks;
        size_t chunkNumel = 1;
        size_t chunkBaseStride = 1;
        bool hasChunk = false;
        for (int64_t i = static_cast<int64_t>(oldShape.size()) - 1; i >= 0; --i)
        {
            if (oldShape[i] == 1)
            {
                continue;
            }

            if (!hasChunk)
            {
                chunkNumel = oldShape[i];
                chunkBaseStride = oldStrides[i];
                hasChunk = true;
                continue;
            }

            if (oldStrides[i] == chunkBaseStride * chunkNumel)
            {
                chunkNumel *= oldShape[i];
                continue;
            }

            chunks.push_back({chunkNumel, chunkBaseStride});
            chunkNumel = oldShape[i];
            chunkBaseStride = oldStrides[i];
        }

        if (!hasChunk)
        {
            return computeContiguousStrides(newShape);
        }
        chunks.push_back({chunkNumel, chunkBaseStride});

        Stride newStrides(newShape.size(), 1);
        int64_t newDim = static_cast<int64_t>(newShape.size()) - 1;
        for (const auto& chunk : chunks)
        {
            size_t viewNumel = 1;
            size_t viewStride = chunk.baseStride;

            do
            {
                while (newDim >= 0 && newShape[newDim] == 1)
                {
                    newStrides[newDim] = viewStride;
                    --newDim;
                }

                if (newDim < 0)
                {
                    return std::nullopt;
                }

                newStrides[newDim] = viewStride;
                viewStride *= newShape[newDim];
                viewNumel *= newShape[newDim];
                --newDim;
            }
            while (viewNumel < chunk.numel);

            if (viewNumel != chunk.numel)
            {
                return std::nullopt;
            }
        }

        while (newDim >= 0 && newShape[newDim] == 1)
        {
            newStrides[newDim] = (newDim + 1 < static_cast<int64_t>(newShape.size()))
                                 ? newStrides[newDim + 1] * newShape[newDim + 1] : 1;
            --newDim;
        }

        if (newDim >= 0)
        {
            return std::nullopt;
        }

        return newStrides;
    }

    // Compute the strides based on the shape of the tensor
    Stride computeStrides() const
    {
        return computeContiguousStrides(m_shape);
    }

    static Stride computeContiguousStrides(const Shape& shape)
    {
        Stride strides(shape.size());
        size_t stride = 1;
        for (int64_t i = strides.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    // Get the flat index from a vector of indices
    size_t getIndex(const Index & indices) const
    {
        assert(indices.size() == m_shape.size());
        return m_offset + std::inner_product(indices.begin(), indices.end(), m_strides.begin(), static_cast<size_t>(0));
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

    static bool checkMinMaxValueOverflow(float minValue, float maxValue, const aix::TensorValue& tensor)
    {
        assert(tensor.dataType() == aix::DataType::kInt32);     // Currently supports only int32 data type.
        tensor.device()->synchronize();
        for (size_t i=0; i < tensor.size(); ++i)
        {
            if (tensor.data<int32_t>()[i] < minValue) return false;
            if (tensor.data<int32_t>()[i] > maxValue) return false;
        }
        return true;
    }

    static void validateSize(const size_t size, const Shape& shape)
    {
        if (size != static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>())))
        {
            throw std::invalid_argument("Data size does not match the tensor shape.");
        }
    }

    // Print Tensor data
    template<typename T>
    void print(std::ostream & os) const
    {
        os << std::fixed << std::setprecision(4);

        // Print scalar value, a tensor with no dimension.
        if (m_shape.empty())
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                os << static_cast<int>(item<T>()) << "\n\n";
            else
                os << item<T>() << "\n\n";
        }
        else if (m_shape.size() == 1)
        {
            // Print tensor that has only one dimension.
            for (size_t i = 0; i < m_shape[0]; ++i)
            {
                if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                    os << "  " << static_cast<int>(getValueAt<T>({i})) << "\n";
                else
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
                            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
                                os << "  " << static_cast<int>(getValueAt<T>(subIndices));
                            else
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

        auto deviceName = device()->name();
        // Print shape
        switch (dataType())
        {
            case DataType::kFloat64:  os << "[ " << deviceName << " Float64 {";  break;
            case DataType::kFloat32:  os << "[ " << deviceName << " Float32 {";  break;
            case DataType::kFloat16:  os << "[ " << deviceName << " Float16 {";  break;
            case DataType::kBFloat16: os << "[ " << deviceName << " BFloat16 {"; break;
            case DataType::kInt64:    os << "[ " << deviceName << " Int64 {";    break;
            case DataType::kInt32:    os << "[ " << deviceName << " Int32 {";    break;
            case DataType::kInt16:    os << "[ " << deviceName << " Int16 {";    break;
            case DataType::kInt8:     os << "[ " << deviceName << " Int8 {";     break;
            case DataType::kUInt8:    os << "[ " << deviceName << " UInt8 {";    break;
            default:                  os << "[ " << deviceName << " Unknown {";  break;
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
    bool      m_isContiguous{true};
    DataType  m_dType{DataType::kFloat32};
    size_t    m_size{0};        // Number of elements in DataType.
    Shape     m_shape;          // The shape of the tensor.
    Stride    m_strides;        // The strides for indexing the tensor.
    size_t    m_offset{0};      // Start offset of data on storage.
    Device *  m_device{nullptr};
    std::shared_ptr<TensorStorage>  m_storage;      // The flat array of tensor elements.
};


class TensorNode
{
public:
    // Constructor
    explicit TensorNode(TensorValue value, bool requireGrad = false) :
        m_value{std::move(value)}, m_requireGrad{requireGrad}
    {
    }

    // Constructor
    explicit TensorNode(const Shape & shape, Device * device, bool requireGrad = false, DataType dType = DataType::kFloat32) :
        m_value{shape, device, dType}, m_requireGrad{requireGrad}
    {
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(const TensorValue & seed)
    {
        if (m_retainGrad)
        {
            grad() += seed;
        }
        if (m_requireGrad || m_retainGrad)
        {
            m_backwardFunc(this, seed);
        }
    }

    TensorValue& grad()
    {
        if (!m_grad.storage())
        {
            m_grad = TensorValue{m_value.shape(), m_value.device(), m_value.size(), m_value.strides(), m_value.dataType()};
            m_grad.fill(0);
        }
        return m_grad;
    }

    Device * device() const  { return m_value.device(); }

    void setBackward(std::vector<std::shared_ptr<TensorNode>> inputs,
                     std::function<void(TensorNode *, const TensorValue &)> func)
    {
        if (m_requireGrad || m_retainGrad)
        {
            m_inputs = std::move(inputs);
            m_backwardFunc = std::move(func);
        }
    }

    std::string  m_name;
    TensorValue  m_value;
    bool  m_requireGrad;
    bool  m_retainGrad{false};
    std::vector<std::shared_ptr<TensorNode>> m_inputs;
    SIndex m_dims;
    size_t m_dim0{0};
    size_t m_dim1{0};
    bool m_keepDim{false};
    TensorValue  m_indices;
    std::optional<ssize_t> m_start;
    std::optional<ssize_t> m_end;
    std::function<void(TensorNode * tensor, const TensorValue & seed)>  m_backwardFunc{nullptr};

private:
    TensorValue  m_grad;
};


struct TensorOptions
{
    inline TensorOptions requireGrad(bool state)    { m_requireGrad = state; return *this; }
    inline TensorOptions dtype(DataType dataType)   { m_dtype = dataType;    return *this; }
    inline TensorOptions device(Device* device)     { m_device = device;     return *this; }
    inline TensorOptions device(std::unique_ptr<aix::Device>& device)  { m_device = device.get(); return *this; }
    inline TensorOptions device(std::shared_ptr<aix::Device>& device)  { m_device = device.get(); return *this; }

    bool m_requireGrad{false};
    aix::DataType m_dtype{aix::DataType::kFloat32};
    aix::Device* m_device{&aix::defaultDevice};
};
inline TensorOptions requireGrad(bool state)    { return { .m_requireGrad=state }; }
inline TensorOptions dtype(DataType dataType)   { return { .m_dtype=dataType    }; }
inline TensorOptions device(Device* device)     { return { .m_device=device     }; }
inline TensorOptions device(std::unique_ptr<aix::Device>& device)     { return { .m_device = device.get() }; }
inline TensorOptions device(std::shared_ptr<aix::Device>& device)     { return { .m_device = device.get() }; }


class Tensor
{
public:
    // Constructor.
    Tensor() = default;

    // Constructor.
    explicit Tensor(const void* data, size_t size, DataType srcDType, const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{data, size, srcDType, shape, opt.m_device, opt.m_dtype},
                                              opt.m_requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const std::shared_ptr<TensorStorage>& storage, size_t size, size_t offset, const Shape & shape,
                    const TensorOptions & opt = {})
    {
        m_data = std::make_shared<TensorNode>(TensorValue{storage, size, offset, shape, opt.m_device, opt.m_dtype},
                                              opt.m_requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(float value, const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(TensorValue{value, shape, opt.m_device, opt.m_dtype}, opt.m_requireGrad);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Constructor.
    explicit Tensor(const Shape & shape, const TensorOptions & opt = {})
    {
        // Create a new Tensor Graph Node.
        m_data = std::make_shared<TensorNode>(shape, opt.m_device, opt.m_requireGrad, opt.m_dtype);
        m_data->m_backwardFunc = defaultBackward;
    }

    // Perform backpropagation to calculate gradients recursively.
    void backward(float value=1)
    {
        if (m_data->m_inputs.empty())
        {
            if (m_data->m_requireGrad) m_data->grad() += TensorValue{value, shape(), device(), dataType()};
            return;
        }
        if (shape().empty())
        {
            m_data->backward(TensorValue{value, device(), dataType()});
            return;
        }
        m_data->backward(TensorValue{value, m_data->m_inputs[0]->m_value.shape(), device(), dataType()});
    }
    void backward(float value, const Shape & gradShape)
    {
        if (m_data->m_inputs.empty())
        {
            if (m_data->m_requireGrad) m_data->grad() += TensorValue{value, gradShape, device(), dataType()};
            return;
        }
        m_data->backward(TensorValue{value, gradShape, device(), dataType()});
    }

    // Getters and setters for the tensor's value.
    inline const TensorValue & value() const    { return m_data->m_value; }
    inline TensorValue & value()                { return m_data->m_value; }
    inline const Shape & shape() const          { return m_data->m_value.shape(); }
    inline DataType dataType() const            { return m_data->m_value.dataType(); }

    // Gradient-related methods.
    inline const TensorValue & grad() const
    {
        validateRetainGradientState();
        return m_data->grad();
    }

    inline TensorValue & grad()
    {
        validateRetainGradientState();
        return m_data->grad();
    }

    inline void zeroGrad()                      { m_data->grad().fill(0); }
    inline bool isRequireGrad() const           { return m_data->m_requireGrad; }
    inline void retainGrad() const              { m_data->m_retainGrad = true; m_data->grad().fill(0); }
    inline const Tensor& requireGrad(bool state) const
    {
        m_data->m_requireGrad = m_data->m_retainGrad = state;
        return *this;
    }

    inline Device * device() const              { return m_data->device(); }

    inline void name(const std::string& name) const  { m_data->m_name = name; }
    inline const std::string& name() const           { return m_data->m_name; }

    // Returns a new Tensor with a new shape.
    Tensor reshape(const Shape & newShape) const
    {
        auto result = makeResult(m_data->m_value.reshape(newShape), isRequireGrad());
        result.m_data->setBackward({ m_data }, reshapeBackwardFunc);
        return result;
    }

    // Returns a new Tensor with a new shape. This method accepts one inferred dimension.
    Tensor reshape(const std::initializer_list<ssize_t>& newShape) const
    {
        return reshape(shapeWithInferredDimToShape(newShape));
    }

    Tensor broadcastTo(const Shape & newShape) const
    {
        if (shape() == newShape) return *this;
        auto result = makeResult(m_data->m_value.broadcastTo(newShape), isRequireGrad());
        result.m_data->setBackward({ m_data }, broadcastBackwardFunc);
        return result;
    }

    // Set operation device for the tensor.
    inline Tensor to(std::unique_ptr<Device>& device) const    { return to(*device); }
    inline Tensor to(std::shared_ptr<Device>& device) const    { return to(*device); }
    inline Tensor to(Device* device) const                     { return to(*device); }
    Tensor to(Device& newDevice) const
    {
        if (&newDevice == m_data->device()) return *this;
        auto result = makeResult(m_data->m_value.to(&newDevice), isRequireGrad());
        result.m_data->setBackward({ m_data }, toDeviceBackwardFunc);
        return result;
    }

    Tensor to(DataType newDataType) const
    {
        if (dataType() == newDataType) return *this;
        auto result = makeResult(m_data->m_value.to(newDataType), isRequireGrad());
        result.m_data->setBackward({ m_data }, toDataTypeBackwardFunc);
        return result;
    }

    static void defaultBackward(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_requireGrad && !node->m_retainGrad)
        {
            assert(node->grad().dataType() == seed.dataType());
            node->grad() += seed;
        }
    }

    static void reshapeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.reshape(node->m_inputs[0]->m_value.shape()));
    }

    static void broadcastBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // Accumulate the gradient to the original node by reducing the gradient from the broadcasted shape to the
        // original shape. Summation is used for gradient accumulation when reducing dimensions because each element
        // of the original tensor contributes to multiple elements of the resulting tensor after broadcasting.
        node->m_inputs[0]->backward(seed.reduceTo(node->m_inputs[0]->m_value.shape()));
    }

    static void toDeviceBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        if (seed.device() != node->m_inputs[0]->m_value.device())
        {
            // Synchronize seed to ensure the seed's data is available before copying it to a different device.
            seed.device()->synchronize();
        }
        node->m_inputs[0]->backward(seed.to(node->m_inputs[0]->m_value.device()));
    }

    static void toDataTypeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // Ensure that the seed gradient is converted back to the data type of the original tensor.
        node->m_inputs[0]->backward(seed.to(node->m_inputs[0]->m_value.dataType()));
    }

    static void addBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0] == node->m_inputs[1])
        {
            node->m_inputs[0]->backward(seed + seed);
        }
        else
        {
            if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
                node->m_inputs[0]->backward(seed);
            if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
                node->m_inputs[1]->backward(seed);
        }
    }

    static void subBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0] == node->m_inputs[1]) return;
        if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
            node->m_inputs[0]->backward(seed);
        if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
            node->m_inputs[1]->backward(-seed);
    }

    static void mulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0] == node->m_inputs[1])
        {
            auto grad = node->m_inputs[0]->m_value * seed;
            node->m_inputs[0]->backward(grad + grad);
        }
        else
        {
            if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
                node->m_inputs[0]->backward(node->m_inputs[1]->m_value * seed);
            if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
                node->m_inputs[1]->backward(node->m_inputs[0]->m_value * seed);
        }
    }

    static void divBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
            node->m_inputs[0]->backward(seed / node->m_inputs[1]->m_value);                                               // ∂f/∂a = 1 / b
        if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
            node->m_inputs[1]->backward(-node->m_inputs[0]->m_value * seed / (node->m_inputs[1]->m_value * node->m_inputs[1]->m_value));  // ∂f/∂b = -a / b^2
    }

    static void unaryBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // Calculate gradients.
        node->m_inputs[0]->backward(-seed);
    }

    static void sqrtBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of sqrt(a) with respect to 'a' is 0.5/sqrt(a).
        // Therefore, the gradient of the input is multiplied by 0.5/sqrt(a).
        node->m_inputs[0]->backward(0.5 / node->m_value * seed);   // ∂f/∂a = 0.5/sqrt(a)
    }

    static void sinBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of sin(a) with respect to 'a' is cos(a).
        // Therefore, the gradient of the input is multiplied by cos(a).
        node->m_inputs[0]->backward(node->m_inputs[0]->m_value.cos() * seed);   // ∂f/∂a = cos(a)
    }

    static void cosBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of cos(a) with respect to 'a' is -sin(a).
        // Therefore, the gradient of the input is multiplied by -sin(a).
        node->m_inputs[0]->backward(-node->m_inputs[0]->m_value.sin() * seed);   // ∂f/∂a = -sin(a)
    }

    static void tanhBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of tanh(a) with respect to 'a' is 1 - tanh^2(a).
        // Therefore, the gradient of the input is multiplied by (1 - tanh^2(a)).
        const auto & tanhValue = node->m_value;
        node->m_inputs[0]->backward((float(1) - tanhValue * tanhValue) * seed);  // ∂f/∂a = (1 - tanh^2(a))
    }

    static void logBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // TODO: Handle division by zero case.
        // The derivative of log(a) with respect to 'a' is 1/a.
        node->m_inputs[0]->backward(seed / node->m_inputs[0]->m_value);  // ∂f/∂a = 1/a
    }

    static void expBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of exp(a) with respect to 'a' is exp(a), itself.
        node->m_inputs[0]->backward(seed * node->m_value);  // ∂f/∂a = exp(a)
    }

    static void maxBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of max(a) with respect to 'a' is a zero tensor with argmax index set to 1.
        node->m_inputs[0]->backward(seed * node->m_inputs[0]->m_value.argmaxIndices());
    }

    static void maxBackwardFunc2(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        // The derivative of max(a) with respect to 'a' is a zero tensor with max indexes set to 1.
        node->m_inputs[0]->backward(seed * node->m_inputs[0]->m_value.argmaxIndices(static_cast<ssize_t>(node->m_dim0)));
    }

    static void powBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
        {
            // ∂f/∂a = b * pow(a, b-1)
            node->m_inputs[0]->backward(seed * node->m_inputs[1]->m_value * node->m_inputs[0]->m_value.pow(node->m_inputs[1]->m_value - float(1)));
        }
        if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
        {
            // ∂f/∂b = pow(a, b) * log(a)
            node->m_inputs[1]->backward(seed * node->m_value * node->m_inputs[0]->m_value.log());
        }
    }

    static void matmulBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.size() < 2) return;
        if (node->m_inputs[0] == node->m_inputs[1])
        {
            auto xT = node->m_inputs[0]->m_value.transpose(0, 1);
            node->m_inputs[0]->backward(seed.matmul(xT) + xT.matmul(seed));
        }
        else
        {
            if (node->m_inputs[0]->m_requireGrad || node->m_inputs[0]->m_retainGrad)
                node->m_inputs[0]->backward(seed.matmul(node->m_inputs[1]->m_value.transpose(0, 1)));
            if (node->m_inputs[1]->m_requireGrad || node->m_inputs[1]->m_retainGrad)
                node->m_inputs[1]->backward(node->m_inputs[0]->m_value.transpose(0, 1).matmul(seed));
        }
    }

    static void transposeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.transpose(node->m_dim0, node->m_dim1));
    }

    static void permuteBackwardFunc(TensorNode* node, const TensorValue& seed)
    {
        if (node->m_inputs.empty()) return;

        // Convert negative reference indices to positive.
        SIndex orgDims = node->m_dims;
        for (size_t i=0; i<orgDims.size(); ++i)
        {
            orgDims[i] = orgDims[i] < 0 ? static_cast<ssize_t>(orgDims.size()) + orgDims[i] : orgDims[i];
        }

        // Calculate permute indexes to put the dimensions back to the original positions.
        SIndex dims(orgDims.size());
        for (size_t i=0; i<orgDims.size(); ++i)
        {
            auto it = std::find(orgDims.begin(), orgDims.end(), i);
            dims[i] = std::distance(orgDims.begin(), it);
        }
        node->m_inputs[0]->backward(seed.permute(dims));
    }

    static void sliceBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(node->m_inputs[0]->m_value.sliceSet(seed, node->m_dim0, node->m_start, node->m_end, node->m_dim1));
    }

    static void sumBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.broadcastTo(node->m_inputs[0]->m_value.shape()));
    }

    static void sumBackwardFunc2(TensorNode* node, const TensorValue& seed)
    {
        if (node->m_inputs.empty()) return;
        const auto& originalShape = node->m_inputs[0]->m_value.shape();

        // For keepDim=False case, 1 dimension was squeezed. That dimension needs to be unsqueezed.
        if (!node->m_keepDim)
            node->m_inputs[0]->backward(seed.unsqueeze(static_cast<ssize_t>(node->m_dim0)).broadcastTo(originalShape));
        else
            node->m_inputs[0]->backward(seed.broadcastTo(originalShape));
    }

    static void squeezeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.unsqueeze(node->m_dim0));
    }

    static void unsqueezeBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.squeeze(node->m_dim0));
    }

    static void trillBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.tril(static_cast<ssize_t>(node->m_dim0)));      // m_dim0 = diagonal
    }

    static void triuBackwardFunc(TensorNode * node, const TensorValue & seed)
    {
        if (node->m_inputs.empty()) return;
        node->m_inputs[0]->backward(seed.triu(static_cast<ssize_t>(node->m_dim0)));      // m_dim0 = diagonal
    }

    static void indexSelectBackwardFunc(TensorNode * node, const TensorValue& seed)
    {
        if (node->m_inputs.empty()) return;
        auto zeros = aix::TensorValue(0.0, node->m_inputs[0]->m_value.shape(), seed.device(), seed.dataType());
        node->m_inputs[0]->backward(zeros.indexAdd(static_cast<ssize_t>(node->m_dim0), node->m_indices, seed, true));
    }

    static void catBackwardFunc(TensorNode* node, const TensorValue& seed)
    {
        size_t numTensors = node->m_inputs.size();
        if (numTensors == 0) return;

        // The dimension along which tensors were concatenated.
        auto dim = static_cast<ssize_t>(node->m_dim0);
        size_t dimOffset = 0;

        // Iterate over each original tensor and propagate the gradient.
        for (size_t i=0; i<numTensors; ++i)
        {
            size_t dimSize = node->m_inputs[i]->m_value.shape()[dim];
            if (node->m_inputs[i]->m_requireGrad || node->m_inputs[i]->m_retainGrad)
                node->m_inputs[i]->backward(seed.slice(dim, dimOffset, dimOffset + dimSize, 1));
            dimOffset += dimSize;
        }
    }

    // Select operator.
    Tensor operator[](ssize_t index) const
    {
        return select(0, index);
    }

    // Overload the + operator
    Tensor operator+(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value + rhs.m_data->m_value, isRequireGrad() || other.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, addBackwardFunc);
        return result;
    }

    // Overload the - operator
    Tensor operator-(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value - rhs.m_data->m_value, isRequireGrad() || other.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, subBackwardFunc);
        return result;
    }

    // Overload the * operator
    Tensor operator*(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value * rhs.m_data->m_value, isRequireGrad() || other.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, mulBackwardFunc);
        return result;
    }

    // Overload the / operator
    Tensor operator/(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value / rhs.m_data->m_value, isRequireGrad() || other.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, divBackwardFunc);
        return result;
    }

    Tensor operator-() const
    {
        auto result = makeResult(-m_data->m_value, isRequireGrad());
        result.m_data->setBackward({ m_data }, unaryBackwardFunc);
        return result;
    }

    Tensor operator+(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        return *this + Tensor(scalar, {}, { .m_dtype=promotedFloatType, .m_device=device() }).broadcastTo(shape());
    }

    Tensor operator-(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        return *this - Tensor(scalar, {}, { .m_dtype=promotedFloatType, .m_device=device() }).broadcastTo(shape());
    }

    Tensor operator*(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        return *this * Tensor(scalar, {}, { .m_dtype=promotedFloatType, .m_device=device() }).broadcastTo(shape());
    }

    Tensor operator/(const float & scalar) const
    {
        auto promotedFloatType = promoteDataTypeToFloat(dataType());
        return *this / Tensor(scalar, {}, { .m_dtype=promotedFloatType, .m_device=device() }).broadcastTo(shape());
    }

    friend Tensor operator+(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, {}, { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor.broadcastTo(rhsTensor.shape()) + rhsTensor;
    }

    friend Tensor operator-(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, {}, { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor.broadcastTo(rhsTensor.shape()) - rhsTensor;
    }

    friend Tensor operator*(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, {}, { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor.broadcastTo(rhsTensor.shape()) * rhsTensor;
    }

    friend Tensor operator/(float scalar, const Tensor & rhsTensor)
    {
        auto promotedFloatType = promoteDataTypeToFloat(rhsTensor.dataType());
        Tensor tensor(scalar, {}, { .m_requireGrad=rhsTensor.isRequireGrad(), .m_dtype=promotedFloatType,
                                                   .m_device=rhsTensor.device() });
        return tensor.broadcastTo(rhsTensor.shape()) / rhsTensor;
    }

    Tensor sqrt() const
    {
        auto result = makeResult(m_data->m_value.sqrt(), isRequireGrad());
        result.m_data->setBackward({ m_data }, sqrtBackwardFunc);
        return result;
    };

    Tensor sin() const
    {
        auto result = makeResult(m_data->m_value.sin(), isRequireGrad());
        result.m_data->setBackward({ m_data }, sinBackwardFunc);
        return result;
    };

    Tensor cos() const
    {
        auto result = makeResult(m_data->m_value.cos(), isRequireGrad());
        result.m_data->setBackward({ m_data }, cosBackwardFunc);
        return result;
    };

    Tensor tanh() const
    {
        auto result = makeResult(m_data->m_value.tanh(), isRequireGrad());
        result.m_data->setBackward({ m_data }, tanhBackwardFunc);
        return result;
    };

    Tensor log() const
    {
        auto result = makeResult(m_data->m_value.log(), isRequireGrad());
        result.m_data->setBackward({ m_data }, logBackwardFunc);
        return result;
    };

    Tensor exp() const
    {
        auto result = makeResult(m_data->m_value.exp(), isRequireGrad());
        result.m_data->setBackward({ m_data }, expBackwardFunc);
        return result;
    };

    Tensor sum() const
    {
        auto result = makeResult(m_data->m_value.sum(), isRequireGrad());
        result.m_data->setBackward({ m_data }, sumBackwardFunc);
        return result;
    }

    Tensor sum(ssize_t dim, bool keepDim=false) const
    {
        auto result = makeResult(m_data->m_value.sum(dim, keepDim), isRequireGrad());
        result.m_data->setBackward({ m_data }, sumBackwardFunc2);
        result.m_data->m_dim0 = dim >= 0 ? dim : dim + m_data->m_value.shape().size();
        result.m_data->m_keepDim = keepDim;
        return result;
    }

    Tensor mean() const
    {
        return sum() / value().size();
    }

    Tensor mean(ssize_t dim, bool keepDim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        return sum(dim, keepDim) / value().shape()[dim];
    }

    Tensor max() const
    {
        auto result = makeResult(m_data->m_value.max(), isRequireGrad());
        result.m_data->setBackward({ m_data }, maxBackwardFunc);
        return result;
    }

    Tensor max(ssize_t dim, bool keepDim=false) const
    {
        auto result = makeResult(m_data->m_value.max(dim, keepDim), isRequireGrad());
        result.m_data->setBackward({ m_data }, maxBackwardFunc2);
        result.m_data->m_dim0 = dim >= 0 ? dim : dim + m_data->m_value.shape().size();
        return result;
    }

    Tensor argmax() const
    {
        auto result = makeResult(m_data->m_value.argmax(), false);
        // argmax does not require gradient.
        return result;
    }

    Tensor argmax(ssize_t dim, bool keepDim=false) const
    {
        auto result = makeResult(m_data->m_value.argmax(dim, keepDim), false);
        // argmax does not require gradient.
        return result;
    }

    Tensor pow(float exp) const
    {
        TensorOptions opt{ .m_dtype=dataType(), .m_device=device() };
        Tensor expTensor = Tensor(exp, Shape{}, opt).broadcastTo(shape());
        auto result = makeResult(m_data->m_value.pow(expTensor.m_data->m_value), isRequireGrad());
        result.m_data->setBackward({ m_data, expTensor.m_data }, powBackwardFunc);
        return result;
    }

    Tensor pow(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        Shape bcShape = broadcastShape(other.shape());
        auto lhs = broadcastTo(bcShape).to(promotedDType);
        auto rhs = other.broadcastTo(bcShape).to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value.pow(rhs.m_data->m_value), isRequireGrad() || other.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, powBackwardFunc);
        return result;
    }

    Tensor matmul(const Tensor & other) const
    {
        auto promotedDType = promoteDataType(dataType(), other.dataType());
        auto lhs = to(promotedDType);
        auto rhs = other.to(promotedDType);

        auto result = makeResult(lhs.m_data->m_value.matmul(rhs.m_data->m_value), isRequireGrad() || rhs.isRequireGrad());
        result.m_data->setBackward({ lhs.m_data, rhs.m_data }, matmulBackwardFunc);
        return result;
    }

    Tensor transpose(ssize_t dim0, ssize_t dim1) const
    {
        auto result = makeResult(m_data->m_value.transpose(dim0, dim1), isRequireGrad());
        result.m_data->setBackward({ m_data }, transposeBackwardFunc);
        result.m_data->m_dim0 = dim0;
        result.m_data->m_dim1 = dim1;
        return result;
    }

    Tensor permute(const SIndex& dims) const
    {
        auto result = makeResult(m_data->m_value.permute(dims), isRequireGrad());
        result.m_data->setBackward({ m_data }, permuteBackwardFunc);
        result.m_data->m_dims = dims;
        return result;
    }

    Tensor slice(ssize_t dim=0, std::optional<ssize_t> startOpt = std::nullopt,
                 std::optional<ssize_t> endOpt = std::nullopt, ssize_t step=1) const
    {
        auto result = makeResult(m_data->m_value.slice(dim, startOpt, endOpt, step), isRequireGrad());
        result.m_data->setBackward({ m_data }, sliceBackwardFunc);
        result.m_data->m_dim0 = dim;
        result.m_data->m_dim1 = step;
        result.m_data->m_start = startOpt;
        result.m_data->m_end = endOpt;
        return result;
    }

    Tensor squeeze(ssize_t dim) const
    {
        auto result = makeResult(m_data->m_value.squeeze(dim), isRequireGrad());
        result.m_data->setBackward({ m_data }, squeezeBackwardFunc);
        result.m_data->m_dim0 = dim;
        return result;
    }

    Tensor unsqueeze(ssize_t dim) const
    {
        auto result = makeResult(m_data->m_value.unsqueeze(dim), isRequireGrad());
        result.m_data->setBackward({ m_data }, unsqueezeBackwardFunc);
        result.m_data->m_dim0 = dim;
        return result;
    }

    Tensor var(bool unbiased=true) const
    {
        auto deviation = *this - mean();
        auto elementCount = unbiased ? deviation.value().size() - 1 : deviation.value().size();
        return (deviation * deviation).sum() / float(elementCount);
    }

    Tensor var(ssize_t dim, bool unbiased=true, bool keepdim=false) const
    {
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        auto deviation = *this - mean(dim, true);
        auto elementCount = unbiased ? shape()[dim] - 1 : shape()[dim];
        auto var = (deviation * deviation).sum(dim, true) / float(elementCount);
        return keepdim ? var : var.squeeze(dim);
    }

    Tensor select(ssize_t dim, ssize_t index) const
    {
        if (shape().empty())
        {
            throw std::invalid_argument("select() cannot be applied to a scalar, zero-dimension, tensor.");
        }
        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;
        index = index < 0 ? static_cast<ssize_t>(shape()[dim]) + index : index;
        return slice(dim, index, index + 1, 1).squeeze(dim);
    }

    std::vector<Tensor> split(ssize_t splitSize, ssize_t dim=0) const
    {
        if (splitSize < 0)
        {
            throw std::invalid_argument("Split size must be a positive number.");
        }

        if (shape().empty())
        {
            throw std::invalid_argument("Split operation needs at least a 1-dim tensor.");
        }

        const auto shapeSize = static_cast<ssize_t>(shape().size());
        dim = dim < 0 ? shapeSize + dim : dim;
        if (dim < 0 || dim >= shapeSize)
        {
            throw std::invalid_argument("Split dimension is out of range.");
        }

        std::vector<Tensor> tensors;        // Stores splitted tensors.
        for (size_t i = 0; i < shape()[dim]; i += splitSize)
        {
            tensors.emplace_back(slice(dim, i, i + splitSize, 1));
        }
        return tensors;
    }

    Tensor tril(ssize_t diagonal=0) const
    {
        auto result = makeResult(m_data->m_value.tril(diagonal), isRequireGrad());
        result.m_data->setBackward({ m_data }, trillBackwardFunc);
        result.m_data->m_dim0 = diagonal;
        return result;
    }

    Tensor triu(ssize_t diagonal=0) const
    {
        auto result = makeResult(m_data->m_value.triu(diagonal), isRequireGrad());
        result.m_data->setBackward({ m_data }, triuBackwardFunc);
        result.m_data->m_dim0 = diagonal;
        return result;
    }

    Tensor indexSelect(ssize_t dim, const Tensor& indices) const
    {
        if (indices.shape().size() > 1)
        {
            throw std::invalid_argument("Indices supposed to be a vector.");
        }

        dim = dim < 0 ? static_cast<ssize_t>(shape().size()) + dim : dim;

        if ((!shape().empty() && (dim < 0 || static_cast<size_t>(dim) >= shape().size())) ||
            (shape().empty() && dim != 0))
        {
            throw std::invalid_argument("Dimension is out of range for indexSelect operation.");
        }

        auto result = makeResult(m_data->m_value.indexSelect(dim, indices.value()), isRequireGrad());
        result.m_data->setBackward({ m_data }, indexSelectBackwardFunc);
        result.m_data->m_dim0 = dim;
        result.m_data->m_indices = indices.value();
        return result;
    }

    static Tensor cat(const std::vector<Tensor>& tensors, ssize_t dim)
    {
        if (tensors.empty())
        {
            throw std::invalid_argument("cat() operation needs at least one tensor.");
        }

        const auto& tensor = tensors[0];

        if (tensor.shape().empty())
        {
            throw std::invalid_argument("Zero-dimensional tensor cannot be concatenated.");
        }

        if (tensors.size() == 1) return tensor;

        dim = dim < 0 ? static_cast<ssize_t>(tensor.shape().size()) + dim : dim;
        if (dim < 0 || dim >= static_cast<ssize_t>(tensor.shape().size()))
        {
            throw std::invalid_argument("Dimension is out of range for cat() operation.");
        }

        bool requireGrad = tensor.isRequireGrad();
        DataType promotedDType = tensor.dataType();
        // Tensor shapes must be the same.
        for (size_t i=0; i<tensors.size()-1; ++i)
        {
            auto shape1 = tensors[i].shape();
            auto shape2 = tensors[i+1].shape();
            shape1[dim] = shape2[dim] = 0;      // Neutralize dimension sizes for comparison.

            if (shape1 != shape2)
            {
                throw std::invalid_argument("Dimension sizes of tensors must match except in dimension " +
                                            std::to_string(dim) + " for the cat() operation.");
            }
            if (tensors[i].device() != tensors[i+1].device())
            {
                throw std::invalid_argument("Tensor devices must be the same for the cat() operation.");
            }
            requireGrad |= tensors[i+1].isRequireGrad();
            promotedDType = promoteDataType(promotedDType, tensors[i+1].dataType());
        }

        auto newShape = tensor.shape();
        for (size_t i=1; i<tensors.size(); ++i)
            newShape[dim] += tensors[i].shape()[dim];

        size_t dimSize = 0;
        Tensor result(newShape, { .m_requireGrad=requireGrad, .m_dtype=promotedDType, .m_device=tensor.device() });
        for (size_t i=0; i<tensors.size(); ++i)
        {
            result.value().sliceSet(tensors[i].to(promotedDType).value(), dim, dimSize, dimSize + tensors[i].shape()[dim], 1, true);
            if (requireGrad)
            {
                result.m_data->m_inputs.emplace_back(tensors[i].m_data);
            }
            dimSize += tensors[i].shape()[dim];
        }
        result.m_data->m_dim0 = dim;
        if (requireGrad)
        {
            result.m_data->m_backwardFunc = catBackwardFunc;
        }
        return result;
    }

    // Friend function to overload operator<<
    inline friend std::ostream & operator<<(std::ostream& os, const Tensor& tensor);

protected:
    static Tensor makeResult(TensorValue value, bool requireGrad)
    {
        Tensor result;
        result.m_data = std::make_shared<TensorNode>(std::move(value), requireGrad);
        result.m_data->m_backwardFunc = defaultBackward;
        return result;
    }

    inline Shape broadcastShape(const Shape& otherShape) const
    {
        return shape() == otherShape ? shape() : TensorValue::broadcastShapes(shape(), otherShape);
    }

    inline void validateRetainGradientState() const
    {
        if (!m_data->m_requireGrad && !m_data->m_retainGrad)
        {
            throw std::runtime_error("Gradients for non-leaf tensors won’t be populated during automatic gradient"
                                     " calculation. Use .retainGrad() on the non-leaf tensor if needed, or access"
                                     " the leaf tensor instead.");
        }
    }

    Shape shapeWithInferredDimToShape(const std::initializer_list<ssize_t>& newShape) const
    {
        Shape currShape = shape();
        Shape resultShape(newShape.size());
        ssize_t inferredDimIndex = -1;
        size_t inferredDimCount  = 0;
        size_t invalidDimCount   = 0;
        ssize_t inferredDimSize  = std::accumulate(currShape.begin(), currShape.end(), static_cast<ssize_t>(1), std::multiplies<>());

        ssize_t i = 0;
        for (const auto dim : newShape)
        {
            if (dim == -1)
            {
                ++inferredDimCount;
                inferredDimIndex = i;
            }
            else
            {
                inferredDimSize /= dim;
                resultShape[i] = dim;
            }
            if (dim == 0 || dim < -1) ++invalidDimCount;
            ++i;
        }

        if (invalidDimCount > 0)
        {
            throw std::invalid_argument("Shape contains invalid dimension.");
        }

        if (inferredDimCount > 1)
        {
            throw std::invalid_argument("Only one dimension can be inferred.");
        }

        if (inferredDimIndex >= 0)
            resultShape[inferredDimIndex] = inferredDimSize;

        return resultShape;
    }

    std::shared_ptr<TensorNode>  m_data{nullptr};
};

// Some convenience method definitions.

inline Tensor tensor(float value, const TensorOptions & opt = {})
{
    return Tensor{value, Shape{}, opt};
}

inline Tensor tensor(const std::initializer_list<double> & data, const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), shape, opt};
}

inline Tensor tensor(const std::initializer_list<float> & data, const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), shape, opt};
}

inline Tensor tensor(const std::initializer_list<double> & data, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<double>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::initializer_list<float> & data, const TensorOptions & opt = {})
{
    return Tensor{data.begin(), data.size(), getDataType<float>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::vector<double> & data, const TensorOptions & opt = {})
{
    return Tensor{data.data(), data.size(), getDataType<double>(), Shape{data.size()}, opt};
}

inline Tensor tensor(const std::vector<float> & data, const TensorOptions & opt = {})
{
    return Tensor{data.data(), data.size(), getDataType<float>(), Shape{data.size()}, opt};
}

inline Tensor ones(const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{1, shape, opt};
}

inline Tensor zeros(const Shape & shape, const TensorOptions & opt = {})
{
    return Tensor{0, shape, opt};
}

inline Tensor onesLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{1, tensor.shape(), { .m_requireGrad=requireGrad, .m_dtype=tensor.dataType(), .m_device=tensor.device() }};
}

inline Tensor zerosLike(const Tensor & tensor, bool requireGrad = false)
{
    return Tensor{0, tensor.shape(), { .m_requireGrad=requireGrad, .m_dtype=tensor.dataType(), .m_device=tensor.device() }};
}

inline Tensor sqrt(const Tensor & A)   { return A.sqrt(); }
inline Tensor sin(const Tensor & A)    { return A.sin();  }
inline Tensor cos(const Tensor & A)    { return A.cos();  }
inline Tensor tanh(const Tensor & A)   { return A.tanh(); }
inline Tensor log(const Tensor & A)    { return A.log();  }
inline Tensor exp(const Tensor & A)    { return A.exp();  }
inline Tensor sum(const Tensor & A)    { return A.sum();  }
inline Tensor mean(const Tensor & A)   { return A.mean(); }
inline Tensor sum(const Tensor & A, ssize_t dim, bool keepDim=false)    { return A.sum(dim, keepDim);  }
inline Tensor mean(const Tensor & A, ssize_t dim, bool keepDim=false)   { return A.mean(dim, keepDim);  }
inline Tensor pow(const Tensor & A, const Tensor & exp)     { return A.pow(exp); }
inline Tensor max(const Tensor & A)         { return A.max();    }
inline Tensor max(const Tensor & A, ssize_t dim, bool keepDim=false)   { return A.max(dim, keepDim); }
inline Tensor argmax(const Tensor & A)      { return A.argmax(); }
inline Tensor matmul(const Tensor & A, const Tensor & B)    { return A.matmul(B); }
inline Tensor squeeze(const Tensor & A, ssize_t dim)    { return A.squeeze(dim);    }
inline Tensor unsqueeze(const Tensor & A, ssize_t dim)  { return A.unsqueeze(dim);  }
inline Tensor cat(const std::vector<Tensor>& tensors, ssize_t dim)     {  return Tensor::cat(tensors, dim);  }
inline Tensor hstack(const std::vector<Tensor>& tensors)    { return Tensor::cat(tensors, 1); }
inline Tensor vstack(const std::vector<Tensor>& tensors)    { return Tensor::cat(tensors, 0); }
inline Tensor var(const Tensor & A, bool unbiased=true)     { return A.var(unbiased); }
inline Tensor var(const Tensor & A, ssize_t dim, bool unbiased=true, bool keepdim=false)
{
    return A.var(dim, unbiased, keepdim);
}

static Tensor randn(const Shape & shape, const TensorOptions & opt = {})
{
    std::uniform_real_distribution<float> distr(-1, 1);

    size_t totalSize = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> rndData(totalSize);

    // Fill rndData with random numbers
    std::generate(rndData.begin(), rndData.end(), [&distr]() -> float { return distr(randGen); });

    return Tensor{rndData.data(), rndData.size(), getDataType<float>(), shape, opt};
}

// Returns evenly spaced values within a given interval. The interval including start but excluding stop. [start, end)
static Tensor arange(float start, float end, float step, const TensorOptions & opt = {})
{
    if (step == 0)
    {
        throw std::invalid_argument("Step must be non-zero.");
    }

    auto range = end - start;
    if ((range > 0 && step < 0) || (range < 0 && step > 0))
    {
        throw std::invalid_argument("Range direction is inconsistent with step sign.");
    }

    auto size = static_cast<size_t>(std::ceil(range / step));
    std::vector<float> data(size);
    std::generate(data.begin(), data.end(), [step,x=start]() mutable -> float { float v=x; x += step; return v; });
    return Tensor{data.data(), data.size(), getDataType<float>(), {size}, opt};
}
inline Tensor arange(float end, const TensorOptions & opt = {})               { return arange(0.0, end, 1.0, opt);   }
inline Tensor arange(float start, float end, const TensorOptions & opt = {})  { return arange(start, end, 1.0, opt); }

static Tensor eye(size_t n, const TensorOptions & opt = {})
{
    std::vector<float> data(n * n, 0);
    for (size_t i=0; i<n; ++i)
    {
        data[i * n + i] = 1;
    }
    return Tensor{data.data(), data.size(), getDataType<float>(), aix::Shape{n, n}, opt};
}


// Overload the << operator to print TensorValue.
std::ostream & operator<<(std::ostream& os, const TensorValue& tensor)
{
    switch (tensor.dataType())
    {
        case DataType::kFloat64:   tensor.print<double    >(os);   break;
        case DataType::kFloat32:   tensor.print<float     >(os);   break;
        case DataType::kFloat16:   tensor.print<float16_t >(os);   break;
        case DataType::kBFloat16:  tensor.print<bfloat16_t>(os);   break;
        case DataType::kInt64:     tensor.print<int64_t   >(os);   break;
        case DataType::kInt32:     tensor.print<int32_t   >(os);   break;
        case DataType::kInt16:     tensor.print<int16_t   >(os);   break;
        case DataType::kInt8:      tensor.print<int8_t    >(os);   break;
        case DataType::kUInt8:     tensor.print<uint8_t   >(os);   break;
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
