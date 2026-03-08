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
#include "aixDevice.hpp"
// External includes
// System includes

namespace aix
{

class DeviceCPU : public Device
{
public:
    explicit DeviceCPU(size_t deviceIndex = 0);
    ~DeviceCPU() override = default;

    DeviceType type() const override;
    std::string name() const override;

    void* allocate(size_t size) override;
    void* allocate(size_t size, DataType dtype) override;
    void deallocate(void * memory) override;

    void add(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) override;
    void sub(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) override;
    void mul(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) override;
    void div(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result) override;
    void unary(const DeviceTensorParams& a1, const DeviceTensorParams& result) override;

    void fill(const void* scalar, DataType scalarDType, const DeviceTensorParams& result) override;
    void fillMin(const DeviceTensorParams& result) override;
    void sum(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void sqrt(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void sin(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void cos(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void tanh(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void log(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void exp(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void pow(const DeviceTensorParams& a, const DeviceTensorParams& exp, const DeviceTensorParams& result) override;
    void max(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void argmax(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void argmaxIndices(const DeviceTensorParams& a, const DeviceTensorParams& result) override;
    void matmul(const DeviceTensorParams& a, const DeviceTensorParams& b, const DeviceTensorParams& result) override;
    void transpose(const DeviceTensorParams& a, const DeviceTensorParams& result, size_t dim0, size_t dim1) override;

    void copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;
    void copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size) override;
    void contiguous(const DeviceTensorParams& src, const DeviceTensorParams& dst) override;
    void reduceTo(const DeviceTensorParams& src, const DeviceTensorParams& dst) override;
    void maxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst) override;
    void argmaxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim) override;
    void argmaxIndicesTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim) override;
    void sliceSet(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                  size_t dim, size_t start, size_t end, size_t step) override;
    void tril(const DeviceTensorParams& dst, ssize_t diagonal) override;
    void triu(const DeviceTensorParams& dst, ssize_t diagonal) override;
    void indexSelect(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                     const DeviceTensorParams& indices, size_t dim) override;
    void indexAdd(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                  const DeviceTensorParams& indices, size_t dim) override;

    void emptyCache() override;
    void synchronize() override;
};

}   // aix namespace
