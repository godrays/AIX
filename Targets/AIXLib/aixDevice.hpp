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
#include "aixDeviceType.hpp"
// External includes
// System includes
#include <cstddef>
#include <string>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace aix
{

enum class DataType : size_t;
struct DeviceTensorParams;

class Device
{
public:
    virtual ~Device();

    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;

    static size_t dataTypeSize(DataType dtype);

    virtual void* allocate(size_t size) = 0;
    virtual void* allocate(size_t size, DataType dtype) = 0;
    virtual void deallocate(void * memory) = 0;

    virtual void add(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) = 0;
    virtual void sub(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) = 0;
    virtual void mul(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) = 0;
    virtual void div(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) = 0;
    virtual void unary(const DeviceTensorParams& a1, const DeviceTensorParams& result) = 0;

    virtual void fill(const void* scalar, DataType scalarDType, const DeviceTensorParams& result) = 0;
    virtual void fillMin(const DeviceTensorParams& result) = 0;
    virtual void sum(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void sqrt(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void sin(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void cos(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void tanh(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void log(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void exp(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void pow(const DeviceTensorParams& a, const DeviceTensorParams& exp, const DeviceTensorParams& result) = 0;
    virtual void max(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void argmax(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void argmaxIndices(const DeviceTensorParams& a, const DeviceTensorParams& result) = 0;
    virtual void matmul(const DeviceTensorParams& a, const DeviceTensorParams& b, const DeviceTensorParams& result) = 0;
    virtual void transpose(const DeviceTensorParams& a, const DeviceTensorParams& result, size_t dim0, size_t dim1) = 0;

    virtual void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) = 0;
    virtual void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) = 0;
    virtual void contiguous(const DeviceTensorParams& src, const DeviceTensorParams& dst) = 0;
    virtual void reduceTo(const DeviceTensorParams& src, const DeviceTensorParams& dst) = 0;
    virtual void maxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst) = 0;
    virtual void argmaxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim) = 0;
    virtual void argmaxIndicesTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim) = 0;
    virtual void sliceSet(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                          size_t dim, size_t start, size_t end, size_t step) = 0;
    virtual void tril(const DeviceTensorParams& dst, ssize_t diagonal) = 0;
    virtual void triu(const DeviceTensorParams& dst, ssize_t diagonal) = 0;
    virtual void indexSelect(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                             const DeviceTensorParams& indices, size_t dim) = 0;
    virtual void indexAdd(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                          const DeviceTensorParams& indices, size_t dim) = 0;

    virtual void emptyCache() = 0;
    virtual void synchronize() = 0;
};

}   // aix namespace
