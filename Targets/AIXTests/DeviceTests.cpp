//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Utils.hpp"
#include <aix.hpp>
#include <aixFuse.hpp>
#if defined(__APPLE__) && defined(__arm64__)
#include <aixDeviceMetal.hpp>
#include <aixDeviceMetalEncoder.hpp>
#include <aixDeviceMetalCache.hpp>
#endif
#include <aixDevices.hpp>
// External includes
#include <doctest/doctest.h>
// System includes
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <random>

using namespace aix;

#define EPSILON             1e-5
#define EPSILON_F16         1e-2
#define EPSILON_MATMUL_F32_METAL 2e-4
//#define DEBUG_LOG

std::uniform_real_distribution<float>  distr(-1, 1);

std::vector<size_t>  testSizes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                                   127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049 };

std::vector<DeviceType>  testDeviceTypes = { aix::DeviceType::kGPU_METAL };


bool isIntegralDataType(aix::DataType dtype)
{
    return dtype == aix::DataType::kInt64 || dtype == aix::DataType::kInt32 ||
           dtype == aix::DataType::kInt16 || dtype == aix::DataType::kInt8 ||
           dtype == aix::DataType::kUInt8;
}


bool verifyResults(const aix::TensorValue & tv1, const aix::TensorValue & tv2, float epsilon = EPSILON)
{
    if (tv1.dataType() != tv2.dataType())
    {
        throw std::invalid_argument("Tensor data types do no match for test result comparison.");
    }

    if (static_cast<size_t>(tv1.dataType()) >= aix::DataTypeCount)
    {
        throw std::invalid_argument("CheckVectorApproxValues does not support the new data type.");
    }

    if (tv1.size() != tv2.size())
    {
        std::cout << "Matrix element sizes does not match!" << std::endl;
    }

    if (tv1.dataType() == aix::DataType::kFloat64)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<double>()[i] - tv2.data<double>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kFloat32)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<float>()[i] - tv2.data<float>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kFloat16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<float16_t>()[i] - tv2.data<float16_t>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kBFloat16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (std::abs(tv1.data<bfloat16_t>()[i] - tv2.data<bfloat16_t>()[i]) > epsilon)
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt64)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int64_t>()[i] != tv2.data<int64_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt32)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int32_t>()[i] != tv2.data<int32_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt16)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int16_t>()[i] != tv2.data<int16_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kInt8)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<int8_t>()[i] != tv2.data<int8_t>()[i])
            {
                return false;
            }
        }
    }
    else if (tv1.dataType() == aix::DataType::kUInt8)
    {
        for (size_t i=0; i<tv1.size(); ++i)
        {
            if (tv1.data<uint8_t>()[i] != tv2.data<uint8_t>()[i])
            {
                return false;
            }
        }
    }

    return true;
}


aix::Shape createRandomShape(ssize_t min, ssize_t max)
{
    std::uniform_int_distribution<size_t> distr_int(min, max);

    Shape shape;
    auto n = distr_int(randGen);
    for (size_t i=0; i<n; i++)
    {
        shape.emplace_back(distr_int(randGen));
    }

    return shape;
}


bool testAllocate(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto cpuBuf    = refDevice.allocate(n, dtype);
        auto deviceBuf = testDevice->allocate(n, dtype);

        // DeviceCPU should be able to allocate memory, and it should be accessible by CPU to read and write.
        for (size_t index=0; index < n * Device::dataTypeSize(dtype); ++index)
        {
            static_cast<uint8_t*>(cpuBuf)[index]    = 5;
            static_cast<uint8_t*>(deviceBuf)[index] = 5;
        }

        refDevice.deallocate(cpuBuf);
        testDevice->deallocate(deviceBuf);
    }

    return true;
}


bool testAdd(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.add(array1.value().deviceParams(), array2.value().deviceParams(), cpuResult.deviceParams());
        testDevice->add(array1.value().deviceParams(), array2.value().deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}

TEST_CASE("FuseEngine requires maxBufferSlots in config")
{
    aix::fuse::FuseCallbacks callbacks;
    callbacks.emitFused = [](const aix::fuse::FusedSubgraphDescriptor&) {};
    callbacks.emitSingle = [](const aix::fuse::OpRecord&) {};
    callbacks.finishFlush = [] {};

    aix::fuse::FuseConfig invalidConfig;
    DOCTEST_CHECK_THROWS_AS(aix::fuse::FuseEngine(invalidConfig, callbacks), std::invalid_argument);

    aix::fuse::FuseConfig validConfig;
    validConfig.maxBufferSlots = 31;
    DOCTEST_CHECK_NOTHROW(aix::fuse::FuseEngine(validConfig, callbacks));
}


TEST_CASE("FuseEngine emits non-contiguous fallback ops once when strided fusion is disabled")
{
    std::vector<aix::fuse::OpType> emittedOps;

    aix::fuse::FuseCallbacks callbacks;
    callbacks.emitFused = [](const aix::fuse::FusedSubgraphDescriptor&) {};
    callbacks.emitSingle = [&](const aix::fuse::OpRecord& op) { emittedOps.push_back(op.type); };
    callbacks.finishFlush = [] {};

    aix::fuse::FuseConfig config;
    config.maxBufferSlots = 31;
    config.supportsStridedFusion = false;

    aix::fuse::FuseEngine engine(config, callbacks);

    std::array<float, 4> inputStorage {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<float, 4> outputStorage{};

    DeviceTensorParams input;
    input.data = inputStorage.data();
    input.dtype = aix::DataType::kFloat32;
    input.isContiguous = false;
    input.shape = {2, 2};
    input.size = 4;
    input.strides = {1, 2};

    DeviceTensorParams output;
    output.data = outputStorage.data();
    output.dtype = aix::DataType::kFloat32;
    output.isContiguous = true;
    output.shape = {2, 2};
    output.size = 4;
    output.strides = {2, 1};

    engine.record(aix::fuse::OpType::Mul, input, input, output);
    engine.flush();

    CHECK(emittedOps == std::vector<aix::fuse::OpType>{aix::fuse::OpType::Mul});
}


bool testSub(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && (dtype == DataType::kFloat64)) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.add(array1.value().deviceParams(), array2.value().deviceParams(), cpuResult.deviceParams());
        testDevice->add(array1.value().deviceParams(), array2.value().deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testNeg(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype).value();
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.neg(array1.deviceParams(), cpuResult.deviceParams());
        testDevice->neg(array1.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1 << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSqrt(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = [&]()
        {
            if (isIntegralDataType(dtype))
            {
                auto noise = aix::randn({1, n});
                return (25.0f * (noise * noise)).to(dtype).value();
            }
            return (50 * aix::randn({1, n})).to(dtype).value();
        }();
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.sqrt(array1.deviceParams(), cpuResult.deviceParams());
        testDevice->sqrt(array1.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1 << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSin(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1       = (50 * aix::randn({1, n})).to(dtype).value();
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.sin(array1.deviceParams(), cpuResult.deviceParams());
        testDevice->sin(array1.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1 << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testCos(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1       = (50 * aix::randn({1, n})).to(dtype).value();
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.cos(array1.deviceParams(), cpuResult.deviceParams());
        testDevice->cos(array1.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1 << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTanh(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array        = (50 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.tanh().value();
        auto deviceResult = array.to(*testDevice).tanh().value();
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testLog(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array        = (11 + 10 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.log().value();
        auto deviceResult = array.to(*testDevice).log().value();
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testExp(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array        = (1 + aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.exp().value();
        auto deviceResult = array.to(*testDevice).exp().value();
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}

bool testMax(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array        = (1 + aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.max().value();
        auto deviceResult = array.to(*testDevice).max().value();
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testArgmax(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto array        = (1 + aix::randn({1, n})).to(dtype);
        auto cpuResult    = array.argmax().value();
        auto deviceResult = array.to(*testDevice).argmax().value();
        testDevice->synchronize();

        if (!verifyResults(cpuResult, deviceResult))
        {
            CAPTURE(n);
            CAPTURE(i);
            CHECK(cpuResult.item<int32_t>() == deviceResult.item<int32_t>());
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testArgmaxIndices(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto array = (1 + aix::randn({1, n})).to(dtype);
        auto cpuResult = array.value().argmaxIndices();
        auto deviceResult = array.to(testDevice).value().argmaxIndices();
        testDevice->synchronize();

        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testArgmaxWithDim(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto shape = createRandomShape(1, 6);      // max element size 6^6 = 46,656
        ssize_t dim = std::rand() % static_cast<ssize_t>(shape.size());
        bool keepdim = static_cast<bool>(std::rand() % 2);

        auto array = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult = array.argmax(dim, keepdim).value();
        auto deviceResult = array.to(*testDevice).argmax(dim, keepdim).value();
        testDevice->synchronize();

        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim: " << dim << ", keepdim: " << keepdim << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testArgmaxIndicesWithDim(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto shape = createRandomShape(1, 6);      // max element size 6^6 = 46,656
        ssize_t dim = std::rand() % static_cast<ssize_t>(shape.size());

        auto array = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult = array.value().argmaxIndices(dim);
        auto deviceResult = array.to(testDevice).value().argmaxIndices(dim);
        testDevice->synchronize();

        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim: " << dim << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


TEST_CASE("DeviceMetal Tests - Argmax keeps first max index across dtypes")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        if (dtype == DataType::kFloat64) continue;

        auto array = aix::tensor({1.0, 5.0, 2.0, 5.0, 4.0}, {1, 5}).to(dtype);
        auto expected = array.argmax().value();
        auto result = array.to(*device).argmax().value();
        device->synchronize();

        CHECK(result.dataType() == DataType::kInt32);
        CHECK(result.item<int32_t>() == expected.item<int32_t>());
        CHECK(result.item<int32_t>() == 1);
    }
}


TEST_CASE("DeviceMetal Tests - ArgmaxIndices keeps first max index across dtypes")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        if (dtype == DataType::kFloat64) continue;

        auto array = aix::tensor({1.0, 5.0, 2.0, 5.0, 4.0}, {1, 5}).to(dtype);
        auto expected = array.value().argmaxIndices();
        auto result = array.to(device.get()).value().argmaxIndices();
        device->synchronize();

        CHECK(result.dataType() == DataType::kInt32);
        CHECK(verifyResults(expected, result));
        CHECK(result.data<int32_t>()[1] == 1);
    }
}


TEST_CASE("DeviceMetal Tests - Argmax with dim keeps first max index across dtypes")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        if (dtype == DataType::kFloat64) continue;

        auto array = aix::tensor({1.0, 5.0, 5.0,
                                  4.0, 4.0, 3.0}, {2, 3}).to(dtype);
        auto expected = array.argmax(1, false).value();
        auto result = array.to(*device).argmax(1, false).value();
        device->synchronize();

        CHECK(result.dataType() == DataType::kInt32);
        CHECK(verifyResults(expected, result));
        CHECK(result.data<int32_t>()[0] == 1);
        CHECK(result.data<int32_t>()[1] == 0);
    }
}


TEST_CASE("DeviceMetal Tests - ArgmaxIndices with dim keeps first max index across dtypes")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        if (dtype == DataType::kFloat64) continue;

        auto array = aix::tensor({1.0, 5.0, 5.0,
                                  4.0, 4.0, 3.0}, {2, 3}).to(dtype);
        auto expected = array.value().argmaxIndices(1);
        auto result = array.to(device.get()).value().argmaxIndices(1);
        device->synchronize();

        CHECK(result.dataType() == DataType::kInt32);
        CHECK(verifyResults(expected, result));
        CHECK(result.data<int32_t>()[1] == 1);
        CHECK(result.data<int32_t>()[3] == 1);
        CHECK(result.data<int32_t>()[2] == 0);
        CHECK(result.data<int32_t>()[4] == 0);
    }
}


bool testMaxWithDim(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto shape  = createRandomShape(1, 6);      // max element size 6^6 = 46,656
        ssize_t dim = std::rand() % static_cast<ssize_t>(shape.size());
        bool keepdim = static_cast<bool>(std::rand() % 2);

        auto array        = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult    = array.max(dim, keepdim).value();
        auto deviceResult = array.to(*testDevice).max(dim, keepdim).value();
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testSlice(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        // TODO: Create a better test config.
        auto shape = createRandomShape(2, 6);
        ssize_t dim   = std::rand() % static_cast<ssize_t>(shape.size());
        ssize_t start = std::rand() % static_cast<ssize_t>(shape.size()-1);
        ssize_t end   = start + (std::rand() % static_cast<ssize_t>(shape.size() - start));
        ssize_t step  = 1 + (std::rand() % static_cast<ssize_t>(shape.size()));

        try
        {
            auto array = (1 + aix::randn(shape)).to(dtype);
            auto cpuResult = array.slice(dim, start, end, step).value();
            auto deviceResult = array.to(*testDevice).slice(dim, start, end, step).value();
            testDevice->synchronize();

            // Compare results with the true/reference results.
            if (!verifyResults(cpuResult, deviceResult))
            {
                #ifdef DEBUG_LOG
                std::cout << "----------------------" << std::endl;
                std::cout << "Array" << std::endl << array.value() << std::endl;
                std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
                std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
                #endif
                return false;
            }
        }
        catch(...)
        {
            // Just skip the test if the test config wss not setup proper.
        }
    }

    return true;
}


bool testSliceSet(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        // TODO: Create a better test config.
        auto shape = createRandomShape(2, 6);
        ssize_t dim   = std::rand() % static_cast<ssize_t>(shape.size());
        ssize_t start = std::rand() % static_cast<ssize_t>(shape.size()-1);
        ssize_t end   = start + (std::rand() % static_cast<ssize_t>(shape.size() - start));
        ssize_t step  = 1 + (std::rand() % static_cast<ssize_t>(shape.size()));

        Shape newShape = shape;
        newShape[dim] = (end - start + step - 1) / step;  // This computes the size along the slicing dimension.

        try
        {
            auto array = (1 + aix::randn(shape)).to(dtype);
            auto tensor = (1 + aix::randn(newShape)).to(dtype);
            auto cpuResult = array.value().sliceSet(tensor.value(), dim, start, end, step);
            auto deviceResult = array.to(*testDevice).value().sliceSet(tensor.value(), dim, start, end, step);
            testDevice->synchronize();

            // Compare results with the true/reference results.
            if (!verifyResults(cpuResult, deviceResult))
            {
                #ifdef DEBUG_LOG
                std::cout << "----------------------" << std::endl;
                std::cout << "Array" << std::endl << array.value() << std::endl;
                std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
                std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
                #endif
                return false;
            }
        }
        catch(...)
        {
            // Just skip the test if the test config wss not setup proper.
        }
    }

    return true;
}


bool testTril(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto shape = createRandomShape(2, 6);
        ssize_t diagonal = (std::rand() % static_cast<ssize_t>(shape.size() * 2)) - static_cast<ssize_t>(shape.size());

        auto array = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult = array.value().tril(diagonal);
        auto deviceResult = array.to(*testDevice).value().tril(diagonal);
        testDevice->synchronize();

        // Compare results with the true/reference results.
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTriu(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto shape = createRandomShape(2, 6);
        ssize_t diagonal = (std::rand() % static_cast<ssize_t>(shape.size() * 2)) - static_cast<ssize_t>(shape.size());

        auto array = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult = array.value().triu(diagonal);
        auto deviceResult = array.to(*testDevice).value().triu(diagonal);
        testDevice->synchronize();

        // Compare results with the true/reference results.
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testIndexSelect(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto shape = createRandomShape(1, 6);
        auto dim = std::rand() % static_cast<ssize_t>(shape.size());
        auto shapeDimSize = shape[dim];
        auto indicesCount = 1 + std::rand() % 10;
        auto indices = (((1.0 + aix::randn({static_cast<size_t>(indicesCount)})) * shapeDimSize / 2.0)
                       .to(aix::DataType::kInt32));

        auto array = (1 + aix::randn(shape)).to(dtype);
        auto cpuResult = array.value().indexSelect(dim, indices.value());
        auto deviceResult = array.value().to(testDevice).indexSelect(dim, indices.value().to(testDevice));
        testDevice->synchronize();

        // Compare results with the true/reference results.
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array.value() << std::endl;
            std::cout << "Indices" << std::endl << indices.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testIndexAdd(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto shape = createRandomShape(1, 6);
        auto dim = std::rand() % static_cast<ssize_t>(shape.size());
        auto shapeDimSize = shape[dim];
        auto indicesCount = 1 + std::rand() % 10;
        auto indices = (((1.0 + aix::randn({static_cast<size_t>(indicesCount)})) * shapeDimSize / 2.0)
                       .to(aix::DataType::kInt32));
        auto newShape = shape;
        if (!newShape.empty())
        {
            newShape[dim] = !indices.shape().empty() ? indices.shape()[0] : 1;
        }
        auto sources = aix::Tensor(1.0, newShape, { .m_dtype=dtype }).value();
        auto array = (1 + aix::randn(shape)).to(dtype).value();
        auto cpuResult = array.indexAdd(dim, indices.value(), sources);
        auto deviceResult = array.to(testDevice).indexAdd(dim, indices.value().to(testDevice), sources.to(testDevice));
        testDevice->synchronize();

        // Compare results with the true/reference results.
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array" << std::endl << array << std::endl;
            std::cout << "Indices" << std::endl << indices.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testPow(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = [&]()
        {
            if (isIntegralDataType(dtype))
            {
                auto noise = aix::randn({1, n});
                return (1.0f + 0.02f * (noise * noise)).to(dtype).value();
            }
            return (2 + 1 * aix::randn({1, n})).to(dtype).value();
        }();
        auto exp = [&]()
        {
            if (isIntegralDataType(dtype))
            {
                auto noise = aix::randn({1, n});
                return (1.0f + 0.02f * (noise * noise)).to(dtype).value();
            }
            return (3 + 2 * aix::randn({1, n})).to(dtype).value();       // Random numbers in [1,5]
        }();
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.pow(array1.deviceParams(), exp.deviceParams(), cpuResult.deviceParams());
        testDevice->pow(array1.deviceParams(), exp.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult,
            !(dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) ? EPSILON * 10 : EPSILON_F16 * 100))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1 << std::endl;
            std::cout << "Exponents" << std::endl << exp << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testMul(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = aix::randn({1, n}).to(dtype);
        auto array2 = aix::randn({1, n}).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.mul(array1.value().deviceParams(), array2.value().deviceParams(), cpuResult.deviceParams());
        testDevice->mul(array1.value().deviceParams(), array2.value().deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testDiv(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto array1 = (21 + 20 * aix::randn({1, n})).to(dtype);
        auto array2 = (21 + 20 * aix::randn({1, n})).to(dtype);
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        refDevice.div(array1.value().deviceParams(), array2.value().deviceParams(), cpuResult.deviceParams());
        testDevice->div(array1.value().deviceParams(), array2.value().deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult, dtype != DataType::kFloat16 ? EPSILON : EPSILON_F16 * 10))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Array1" << std::endl << array1.value() << std::endl;
            std::cout << "Array2" << std::endl << array2.value() << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testMatMul(Device* testDevice, size_t n, size_t inner, size_t m)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL &&
            (dtype == DataType::kFloat64 || dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        auto matA = (11 + 10 * aix::randn({n, inner})).to(dtype).value();
        auto matB = (11 + 10 * aix::randn({inner, m})).to(dtype).value();
        auto cpuResult    = aix::TensorValue({n, m}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({n, m}, testDevice).to(dtype);

        refDevice.matmul(matA.deviceParams(), matB.deviceParams(), cpuResult.deviceParams());
        testDevice->matmul(matA.deviceParams(), matB.deviceParams(), deviceResult.deviceParams());
        testDevice->synchronize();

        auto epsilon = EPSILON;
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat32)
        {
            epsilon = EPSILON_MATMUL_F32_METAL;
        }

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult, epsilon))
        {
            MESSAGE("MatMul mismatch: M=" << n << ", K=" << inner << ", N=" << m
                    << ", data type=" << static_cast<size_t>(dtype));
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "MatA" << std::endl << matA << std::endl;
            std::cout << "MatB" << std::endl << matB << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


TEST_CASE("DeviceCPU Tests - Copy clamps negative BFloat16 to UInt8")
{
    std::array<bfloat16_t, 4> source = { bfloat16_t{-1.0f}, bfloat16_t{0.0f}, bfloat16_t{42.0f}, bfloat16_t{300.0f} };
    std::array<uint8_t, 4> expected = { 0, 0, 42, 255 };

    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        aix::DeviceCPU refDevice;
        auto cpuResult = aix::TensorValue({1, source.size()}, &refDevice).to(DataType::kUInt8);
        auto deviceResult = aix::TensorValue({1, source.size()}, device.get()).to(DataType::kUInt8);

        refDevice.copy(source.data(), DataType::kBFloat16, cpuResult.data(), DataType::kUInt8, source.size());
        device->copy(source.data(), DataType::kBFloat16, deviceResult.data(), DataType::kUInt8, source.size());
        device->synchronize();

        for (size_t i = 0; i < expected.size(); ++i)
        {
            CHECK(cpuResult.data<uint8_t>()[i] == expected[i]);
            CHECK(deviceResult.data<uint8_t>()[i] == expected[i]);
        }
    }
}


TEST_CASE("DeviceCPU Tests - Fill clamps negative BFloat16 to UInt8")
{
    bfloat16_t scalar = -1.0f;

    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        aix::DeviceCPU refDevice;
        auto cpuResult = aix::TensorValue({1, 4}, &refDevice).to(DataType::kUInt8);
        auto deviceResult = aix::TensorValue({1, 4}, device.get()).to(DataType::kUInt8);

        refDevice.fill(&scalar, DataType::kBFloat16, cpuResult.deviceParams());
        device->fill(&scalar, DataType::kBFloat16, deviceResult.deviceParams());
        device->synchronize();

        for (size_t i = 0; i < cpuResult.size(); ++i)
        {
            CHECK(cpuResult.data<uint8_t>()[i] == 0);
            CHECK(deviceResult.data<uint8_t>()[i] == 0);
        }
    }
}


bool testTranspose2D(Device* testDevice, size_t n, size_t m)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.

        size_t dim0 = 0;
        size_t dim1 = 1;
        auto tensor = aix::randn({n, m}).to(dtype).value();

        Shape newShape = tensor.shape();
        std::swap(newShape[dim0], newShape[dim1]);
        auto cpuResult    = aix::TensorValue(newShape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(newShape, testDevice).to(dtype);

        refDevice.transpose(tensor.deviceParams(), cpuResult.deviceParams(), dim0, dim1);
        testDevice->transpose(tensor.deviceParams(), deviceResult.deviceParams(), dim0, dim1);
        testDevice->synchronize();

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim (" << dim0 << "," << dim1 << ")" << std::endl;
            std::cout << "Tensor" << std::endl << tensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testTranspose(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.
        ssize_t maxDim = 5;
        auto tensor = aix::randn(createRandomShape(1, maxDim)).to(dtype).value();

        std::uniform_int_distribution<size_t> distr_int(0, 1000);
        size_t dim0 = distr_int(randGen) % tensor.shape().size();
        size_t dim1 = distr_int(randGen) % tensor.shape().size();

        Shape newShape = tensor.shape();
        std::swap(newShape[dim0], newShape[dim1]);

        auto cpuResult    = aix::TensorValue(newShape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(newShape, testDevice).to(dtype);

        refDevice.transpose(tensor.deviceParams(), cpuResult.deviceParams(), dim0, dim1);
        testDevice->transpose(tensor.deviceParams(), deviceResult.deviceParams(), dim0, dim1);
        testDevice->synchronize();

        // Compare true/cpu result with gpu result
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dim (" << dim0 << "," << dim1 << ")" << std::endl;
            std::cout << "Tensor" << std::endl << tensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testPermute(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.
        ssize_t maxDim = 6;
        auto tensor = aix::randn(createRandomShape(1, maxDim)).to(dtype);
        SIndex dims(tensor.shape().size());
        std::iota(dims.begin(), dims.end(), 0);   // Initialize to [0, 1, 2, ...]

        for (size_t j=0; j<dims.size(); ++j)
        {
            std::uniform_int_distribution<size_t> distr_int(0, 1000);
            size_t dim0 = distr_int(randGen) % tensor.shape().size();
            size_t dim1 = distr_int(randGen) % tensor.shape().size();
            std::swap(dims[dim0], dims[dim1]);
        }

        auto cpuResult    = tensor.to(refDevice).permute(dims);
        auto deviceResult = tensor.to(testDevice).permute(dims);
        testDevice->synchronize();

        // Compare true/cpu result with gpu result.
        if (!verifyResults(cpuResult.value(), deviceResult.value()))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Dims: ";  for (auto val: dims) std::cout << val << ","; std::cout << "\n";
            std::cout << "Shape: "; for (auto val: deviceResult.shape()) std::cout << val << ","; std::cout << "\n";
            std::cout << "Tensor" << std::endl << tensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


bool testCopy(Device* testDevice, size_t n)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            auto srcDType = static_cast<DataType>(i);
            auto dstDType = static_cast<DataType>(j);
            auto hasFloat64 = srcDType == DataType::kFloat64 || dstDType == DataType::kFloat64;
            auto hasFloat16 = srcDType == DataType::kFloat16 || dstDType == DataType::kFloat16;

            // Apple Metal Framework does not support kFloat64 data type.
            if (testDevice->type() == DeviceType::kGPU_METAL && hasFloat64) continue;

            aix::DeviceCPU  refDevice;     // Reference/CPU device.
            auto src = aix::randn({1, n}).to(srcDType);
            auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dstDType);
            auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dstDType);

            refDevice.copy(src.value().data(), srcDType, cpuResult.data(), dstDType, n);
            testDevice->copy(src.value().data(), srcDType, deviceResult.data(), dstDType, n);
            testDevice->synchronize();

            // Compare results with the true/reference results
            if (!verifyResults(cpuResult, deviceResult, hasFloat16 ? EPSILON_F16 : EPSILON))
            {
                #ifdef DEBUG_LOG
                std::cout << "----------------------" << std::endl;
                std::cout << "Source" << std::endl << src.value() << std::endl;
                std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
                std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
                #endif
                return false;
            }
        }
    }
    return true;
}


bool testFill(Device* testDevice, size_t n)
{
    unsigned char unifiedScalarValue[sizeof(double)];

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            auto scalarDType = static_cast<DataType>(i);
            auto dstDType = static_cast<DataType>(j);
            auto hasFloat64 = scalarDType == DataType::kFloat64 || dstDType == DataType::kFloat64;

            // Apple Metal Framework does not support kFloat64 data type.
            if (testDevice->type() == DeviceType::kGPU_METAL && hasFloat64) continue;

            // Convert the scalar value to unifiedScalarValue. We need a float data without a fraction to eliminate
            // F16 and BF16 conversion issues.
            auto scalar = static_cast<float>(static_cast<int>(5 + 5 * distr(randGen)));
            memset(unifiedScalarValue, 0, sizeof(unifiedScalarValue));
            switch (scalarDType)
            {
                case DataType::kFloat64:   *reinterpret_cast<double*    >(unifiedScalarValue) = static_cast<double    >(scalar); break;
                case DataType::kFloat32:   *reinterpret_cast<float*     >(unifiedScalarValue) = static_cast<float     >(scalar); break;
                case DataType::kFloat16:   *reinterpret_cast<float16_t* >(unifiedScalarValue) = static_cast<float16_t >(scalar); break;
                case DataType::kBFloat16:  *reinterpret_cast<bfloat16_t*>(unifiedScalarValue) = static_cast<bfloat16_t>(scalar); break;
                case DataType::kInt64:     *reinterpret_cast<int64_t*   >(unifiedScalarValue) = static_cast<int64_t   >(scalar); break;
                case DataType::kInt32:     *reinterpret_cast<int32_t*   >(unifiedScalarValue) = static_cast<int32_t   >(scalar); break;
                case DataType::kInt16:     *reinterpret_cast<int16_t*   >(unifiedScalarValue) = static_cast<int16_t   >(scalar); break;
                case DataType::kInt8:      *reinterpret_cast<int8_t*    >(unifiedScalarValue) = static_cast<int8_t    >(scalar); break;
                case DataType::kUInt8:     *reinterpret_cast<uint8_t*   >(unifiedScalarValue) = static_cast<uint8_t   >(scalar); break;
                default:
                    throw std::runtime_error("Data type is not supported in the fill test.");
                    break;
            }

            aix::DeviceCPU  refDevice;     // Reference/CPU device.
            auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dstDType);
            auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dstDType);

            // We used unifiedScalarValue to pass a pointer to data to the fill method with its data type.
            refDevice.fill(&unifiedScalarValue, scalarDType, cpuResult.deviceParams());
            testDevice->fill(&unifiedScalarValue, scalarDType, deviceResult.deviceParams());
            testDevice->synchronize();

            // Compare results with the true/reference results
            if (!verifyResults(cpuResult, deviceResult))
            {
                #ifdef DEBUG_LOG
                std::cout << "----------------------" << std::endl;
                std::cout << "Scalar: " << scalar << std::endl;
                std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
                std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
                #endif
                return false;
            }
        }
    }

    return true;
}


bool testFillMin(Device* testDevice, size_t n)
{
    for (size_t j=0; j<aix::DataTypeCount; ++j)
    {
        auto dtype = static_cast<DataType>(j);

        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        aix::DeviceCPU  refDevice;     // Reference/CPU device.
        auto cpuResult    = aix::TensorValue({1, n}, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue({1, n}, testDevice).to(dtype);

        // We used unifiedScalarValue to pass a pointer to data to the fill method with its data type.
        refDevice.fillMin(cpuResult.deviceParams());
        testDevice->fillMin(deviceResult.deviceParams());
        testDevice->synchronize();

        // Compare results with the true/reference results
        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


static double reduceToErrorBound(DataType dtype, size_t count, double sumAbs, double expected)
{
    if (isIntegralDataType(dtype)) return 0.0;

    // These accelerated-device tests allow Float32 accumulation for Float16/BFloat16,
    // not bitwise parity with another backend. gamma_n * sum(abs(x)) bounds
    // sequential or tree summation of these bounded binary-fraction inputs,
    // including cancellation and the initial destination. Their FP64 sums are exact.
    const double u = (dtype == DataType::kFloat64 ? std::numeric_limits<double>::epsilon() :
                                                  std::numeric_limits<float>::epsilon()) / 2.0;
    const double nu = static_cast<double>(count) * u;
    const double accumulationError = nu / (1.0 - nu) * sumAbs;
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
    {
        const double outputU = dtype == DataType::kFloat16 ? 0x1p-11 : 0x1p-8;
        const double underflowError = dtype == DataType::kFloat16 ? 0x1p-25 : 0x1p-134;
        return accumulationError + outputU * (std::abs(expected) + accumulationError) + underflowError;
    }
    return accumulationError;
}


static void checkReduceTo(Device* testDevice, const TensorValue& src, const Shape& shape, float initialValue = 0.0f)
{
    REQUIRE(TensorValue::checkBroadcastTo(shape, src.shape()));
    Index axes;
    const size_t leading = src.shape().size() - shape.size();
    for (size_t i = 0; i < src.shape().size(); ++i)
    {
        if (i < leading || shape[i - leading] != src.shape()[i]) axes.push_back(i);
    }
    aix::DeviceCPU refDevice;
    const auto dtype = src.dataType();

    // Widen the already quantized inputs, not the original Float32 values.
    // A same-dtype CPU sum rounds on every addition and is not an accuracy
    // oracle for a backend using Float32 accumulation for Float16/BFloat16.
    auto referenceSrc = src.to(&refDevice).to(DataType::kFloat64);
    auto reference = TensorValue(initialValue, shape, &refDevice, DataType::kFloat64);
    auto magnitudes = referenceSrc;
    for (size_t i = 0; i < magnitudes.size(); ++i)
    {
        magnitudes.data<double>()[i] = std::abs(magnitudes.data<double>()[i]);
    }
    auto sumAbs = TensorValue(std::abs(initialValue), shape, &refDevice, DataType::kFloat64);
    reduceAxes(refDevice, referenceSrc.deviceParams(), reference.deviceParams(), axes, ReduceOp::kSum);
    reduceAxes(refDevice, magnitudes.deviceParams(), sumAbs.deviceParams(), axes, ReduceOp::kSum);

    auto actual = TensorValue(initialValue, shape, testDevice, dtype);
    auto deviceSrc = src.to(testDevice);
    reduceAxes(*testDevice, deviceSrc.deviceParams(), actual.deviceParams(), axes, ReduceOp::kSum);
    testDevice->synchronize();
    auto actual64 = actual.to(&refDevice).to(DataType::kFloat64);
    REQUIRE(actual64.shape() == reference.shape());

    const size_t count = src.size() / reference.size();
    CAPTURE(static_cast<size_t>(testDevice->type()));
    CAPTURE(static_cast<size_t>(dtype));
    CAPTURE(count);
    INFO("Source: " << src);
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const double expected = reference.data<double>()[i];
        const double observed = actual64.data<double>()[i];
        const double tolerance = reduceToErrorBound(dtype, count, sumAbs.data<double>()[i], expected);
        CAPTURE(i);
        CAPTURE(expected);
        CAPTURE(observed);
        CAPTURE(tolerance);
        CHECK(std::isfinite(tolerance));
        CHECK(std::isfinite(observed));
        CHECK(std::abs(observed - expected) <= tolerance);
    }
}


void testReduceTo(Device* testDevice, uint32_t seed)
{
    // Local state makes each trial reproducible and independent of test order.
    aix::DeviceCPU refDevice;
    std::mt19937 generator(seed);
    CAPTURE(seed);
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        Shape newShape(1 + generator() % 5);
        for (auto& dim : newShape) dim = 1 + generator() % 5;
        Shape shape = newShape;
        for (auto& dim : shape)
        {
            if (generator() % 2) dim = 1;
        }
        // Removing leading axes also produces a valid reduction, never a skipped trial.
        const size_t leading = generator() % shape.size();
        shape.erase(shape.begin(), shape.begin() + leading);

        const size_t size = std::accumulate(newShape.begin(), newShape.end(), size_t{1}, std::multiplies<>());
        std::vector<float> values(size);
        for (size_t j = 0; j < size; ++j)
        {
            // Use meaningful integer inputs without overflowing even Int8 sums.
            if (isIntegralDataType(dtype))
            {
                const size_t spacing = (size + 31) / 32;
                values[j] = j % spacing == 0 ? static_cast<float>(generator() % 4) : 0.0f;
            }
            else
            {
                values[j] = static_cast<float>(static_cast<int>(generator() >> 8) - 8388608) / 8388608.0f;
            }
        }
        auto srcTensor = TensorValue(values.data(), size, DataType::kFloat32, newShape, &refDevice, dtype);
        checkReduceTo(testDevice, srcTensor, shape);
    }
}


bool testMaxTo(Device* testDevice)
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        // Apple Metal Framework does not support kFloat64 data type.
        if (testDevice->type() == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;

        auto shape    = createRandomShape(1, 5);
        auto newShape = createRandomShape(1, 5);
        if (!TensorValue::checkBroadcastTo(shape, newShape)) return true;

        aix::DeviceCPU  refDevice;

        auto srcTensor    = aix::randn(newShape).to(dtype).value();
        auto cpuResult    = aix::TensorValue(shape, &refDevice).to(dtype);
        auto deviceResult = aix::TensorValue(shape, testDevice).to(dtype);

        refDevice.fillMin(cpuResult.deviceParams());
        testDevice->fillMin(deviceResult.deviceParams());
        Index axes;
        const size_t leading = newShape.size() - shape.size();
        for (size_t axis = 0; axis < newShape.size(); ++axis)
        {
            if (axis < leading || shape[axis - leading] != newShape[axis]) axes.push_back(axis);
        }
        reduceAxes(refDevice, srcTensor.deviceParams(), cpuResult.deviceParams(), axes, ReduceOp::kMax);
        reduceAxes(*testDevice, srcTensor.deviceParams(), deviceResult.deviceParams(), axes, ReduceOp::kMax);
        testDevice->synchronize();

        if (!verifyResults(cpuResult, deviceResult))
        {
            #ifdef DEBUG_LOG
            std::cout << "----------------------" << std::endl;
            std::cout << "Source" << std::endl << srcTensor << std::endl;
            std::cout << "Expected Result" << std::endl << cpuResult << std::endl;
            std::cout << "DeviceCPU Result" << std::endl << deviceResult << std::endl;
            #endif
            return false;
        }
    }

    return true;
}


TEST_CASE("DeviceMetal single-row matmul preserves short reduction accuracy")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    // Exact inputs captured from the randomized MatMul failure. SIMD reassociation
    // changed this seven-term dot product by two float32 ULPs (above the absolute tolerance).
    const std::array<float, 7> a = {
        0x1.32f94p+4f, 0x1.08d0ap+0f, 0x1.0e0d3p+4f, 0x1.3eea96p+4f,
        0x1.e99304p+3f, 0x1.8826cap+3f, 0x1.048a8cp+2f
    };
    const std::array<float, 7> column = {
        0x1.79e4a2p+3f, 0x1.0b7d5ap+3f, 0x1.ffab52p+3f, 0x1.349754p+3f,
        0x1.0d287p+3f, 0x1.21d56cp+4f, 0x1.845fbp+3f
    };
    for (size_t k : {1, 3, 7, 15, 31, 32})
    {
        CAPTURE(k);
        std::vector<float> input(k);
        for (size_t i = 0; i < k; ++i)
        {
            input[i] = a[i % a.size()];
        }
        for (size_t n : {1, 4, 7, 8})
        {
            CAPTURE(n);
            std::vector<float> b(k * n);
            for (size_t i = 0; i < k; ++i)
            {
                std::fill_n(b.begin() + i * n, n, column[i % column.size()]);
            }
            TensorValue cpuA(input.data(), input.size(), DataType::kFloat32, {1, k}, &refDevice);
            TensorValue cpuB(b.data(), b.size(), DataType::kFloat32, {k, n}, &refDevice);
            auto metalA = cpuA.to(device.get());
            auto metalB = cpuB.to(device.get());
            auto expected = cpuA.matmul(cpuB);
            auto actual = metalA.matmul(metalB);
            device->synchronize();
            for (size_t j = 0; j < n; ++j)
            {
                CHECK(std::abs(actual.data<float>()[j] - expected.data<float>()[j]) <= EPSILON_MATMUL_F32_METAL);
            }
        }
    }
}


TEST_CASE("DeviceMetal single-row matmul float32 layout and reduction boundaries")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    const std::pair<size_t, size_t> dimensions[] = {
        {1, 1}, {3, 3}, {7, 4}, {31, 5}, {32, 15}, {33, 16}, {63, 17},
        {64, 31}, {65, 32}, {127, 33}, {128, 128}, {129, 129}, {768, 768}, {3072, 768}
    };
    for (const auto& [k, n] : dimensions)
    {
        for (size_t layout = 0; layout < 5; ++layout)
        {
            CAPTURE(k);
            CAPTURE(n);
            CAPTURE(layout);
            const size_t columns = layout == 0 ? n : (layout == 1 ? k :
                                   (layout == 2 ? n + 4 : (layout == 3 ? n + 1 : 2 * n + 2)));
            const size_t rows = layout == 1 ? n : k;
            std::vector<float> a(2 * k + 1);
            std::vector<float> b(rows * columns);
            for (size_t i = 0; i < a.size(); ++i)
            {
                a[i] = static_cast<float>(static_cast<int>((i * 7) % 29) - 14) / 17.0f;
            }
            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] = static_cast<float>(static_cast<int>((i * 13) % 47) - 23) / 31.0f;
            }

            auto makeInputs = [&](aix::Device* target)
            {
                TensorValue input(a.data(), a.size(), DataType::kFloat32, {1, a.size()}, target);
                TensorValue weights(b.data(), b.size(), DataType::kFloat32, {rows, columns}, target);
                auto inputView = layout < 2 ? input.slice(1, 0, k) : input.slice(1, 1, 2 * k + 1, 2);
                auto weightView = [&]()
                {
                    if (layout == 1) return weights.transpose(0, 1);
                    if (layout == 2) return weights.slice(1, 4, n + 4);
                    if (layout == 3) return weights.slice(1, 1, n + 1);
                    if (layout == 4) return weights.slice(1, 1, 2 * n + 1, 2);
                    return weights.slice(1, 0, n);
                }();
                return std::pair{std::move(inputView), std::move(weightView)};
            };

            auto [cpuInput, cpuWeights] = makeInputs(&refDevice);
            auto [metalInput, metalWeights] = makeInputs(device.get());
            auto expected = cpuInput.matmul(cpuWeights);
            auto actual = metalInput.matmul(metalWeights);
            device->synchronize();
            for (size_t i = 0; i < n; ++i)
            {
                CHECK(actual.data<float>()[i] == doctest::Approx(expected.data<float>()[i])
                      .epsilon(EPSILON_MATMUL_F32_METAL).scale(1.0));
            }
        }
    }
}


TEST_CASE("DeviceMetal single-row matmul projection and attention dispatch boundaries")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    const std::pair<size_t, size_t> dimensions[] = {
        {31, 64}, {32, 64}, {33, 32}, {36, 5}, {36, 33}, {63, 64}, {64, 3}, {64, 4},
        {64, 7}, {64, 31}, {64, 33}, {64, 107}, {64, 108}, {65, 64}, {68, 511}, {68, 512},
        {96, 64}, {108, 64}, {127, 64}, {128, 64}, {129, 64}, {63, 512}, {64, 512},
        {768, 511}, {768, 512}, {768, 513}, {768, 768}, {768, 2048}, {768, 2112},
        {768, 2304}, {768, 3072}, {3072, 768}, {768, 50257}
    };
    for (const auto& [k, n] : dimensions)
    {
        for (size_t layout = 0; layout < 4; ++layout)
        {
            // Offset unit-stride vectors exercise the fast paths; stepped vectors
            // and doubly strided matrices must remain on the general M1 kernel.
            const bool transposed = layout == 1;
            const size_t step = layout == 2 ? 2 : 1;
            const size_t rows = transposed ? n : k;
            const size_t columns = (transposed ? k : (layout == 3 ? 2 * n : n)) + 3;
            CAPTURE(k);
            CAPTURE(n);
            CAPTURE(layout);
            std::vector<float> a(k * step + 1);
            std::vector<float> b(rows * columns);
            for (size_t i = 0; i < a.size(); ++i)
            {
                a[i] = static_cast<float>(static_cast<int>((i * 17) % 67) - 33) / 131.0f;
            }
            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] = static_cast<float>(static_cast<int>((i * 29) % 97) - 48) / 193.0f;
            }
            auto makeInputs = [&](aix::Device* target)
            {
                TensorValue input(a.data(), a.size(), DataType::kFloat32, {1, a.size()}, target);
                TensorValue weights(b.data(), b.size(), DataType::kFloat32, {rows, columns}, target);
                auto inputView = input.slice(1, 1, k * step + 1, step);
                auto weightView = transposed ? weights.slice(1, 1, k + 1).transpose(0, 1)
                    : weights.slice(1, 1, (layout == 3 ? 2 * n : n) + 1, layout == 3 ? 2 : 1);
                return std::pair{std::move(inputView), std::move(weightView)};
            };
            auto [cpuInput, cpuWeights] = makeInputs(&refDevice);
            auto [metalInput, metalWeights] = makeInputs(device.get());
            auto expected = cpuInput.matmul(cpuWeights);
            auto actual = metalInput.matmul(metalWeights);
            device->synchronize();
            for (size_t i = 0; i < n; ++i)
            {
                CHECK(std::abs(actual.data<float>()[i] - expected.data<float>()[i]) <= EPSILON_MATMUL_F32_METAL);
            }
        }
    }
}


TEST_CASE("DeviceMetal queued matmul dependencies survive batch boundaries and temporary retirement")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    std::vector<float> identity(64 * 64, 0.0f);
    for (size_t i = 0; i < 64; ++i)
    {
        identity[i * 64 + i] = 1.0f;
    }
    TensorValue weights(identity.data(), identity.size(), DataType::kFloat32, {64, 64}, device.get());
    TensorValue value(1.0f, {1, 64}, device.get());
    TensorValue increment(0.25f, {1, 64}, device.get());
    std::vector<TensorValue> saved;
    std::vector<float> expected;
    for (size_t i = 0; i < 256; ++i)
    {
        value = value.matmul(weights) + increment;
        if (i % 17 == 0)
        {
            saved.push_back(value.matmul(weights));
            expected.push_back(1.0f + (i + 1) * 0.25f);
        }
    }
    device->synchronize();
    // Repeat across an explicit completion boundary using the same device/fence.
    value = value.matmul(weights) + increment;
    device->synchronize();
    for (size_t i = 0; i < 64; ++i)
    {
        CHECK(value.data<float>()[i] == 65.25f);
        for (size_t j = 0; j < saved.size(); ++j)
        {
            CHECK(saved[j].data<float>()[i] == expected[j]);
        }
    }
}


#if defined(__APPLE__) && defined(__arm64__)
TEST_CASE("DeviceMetal concurrent encoder tracks RAW WAR WAW and distinct overlapping heap buffers")
{
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto queue = NS::TransferPtr(device->newCommandQueue());
    auto fence = NS::TransferPtr(device->newFence());
    auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setFastMathEnabled(false);
    NS::Error* error = nullptr;
    const char* source = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void fillTest(device float* dst [[buffer(0)]], constant float& value [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) { dst[i] = value; }
        kernel void copyTest(const device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }
        kernel void addTest(device float* dst [[buffer(0)]], constant float& value [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) { dst[i] += value; }
    )";
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(source, NS::UTF8StringEncoding),
                                                     options.get(), &error));
    REQUIRE(library);
    auto fillFunction = NS::TransferPtr(library->newFunction(NS::String::string("fillTest", NS::UTF8StringEncoding)));
    auto copyFunction = NS::TransferPtr(library->newFunction(NS::String::string("copyTest", NS::UTF8StringEncoding)));
    auto addFunction = NS::TransferPtr(library->newFunction(NS::String::string("addTest", NS::UTF8StringEncoding)));
    aix::metal::MetalComputeEncoder encoder;
    auto fill = NS::TransferPtr(encoder.createPipeline(device.get(), fillFunction.get(), &error));
    auto copy = NS::TransferPtr(encoder.createPipeline(device.get(), copyFunction.get(), &error));
    auto add = NS::TransferPtr(encoder.createPipeline(device.get(), addFunction.get(), &error));
    // A PSO without registered reflection exercises the serialized fallback.
    auto unknown = NS::TransferPtr(device->newComputePipelineState(copyFunction.get(), &error));
    REQUIRE(fill);
    REQUIRE(copy);
    REQUIRE(add);
    REQUIRE(unknown);

    auto desc = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
    desc->setType(MTL::HeapTypePlacement);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    desc->setSize(65536);
    auto heap = NS::TransferPtr(device->newHeap(desc.get()));
    REQUIRE(heap);
    auto a = NS::TransferPtr(heap->newBuffer(16384, MTL::ResourceStorageModeShared, 0));
    auto alias = NS::TransferPtr(heap->newBuffer(16384, MTL::ResourceStorageModeShared, 0));
    auto disjoint = NS::TransferPtr(heap->newBuffer(16384, MTL::ResourceStorageModeShared, 16384));
    auto saved = NS::TransferPtr(device->newBuffer(3 * 16384, MTL::ResourceStorageModeShared));
    auto privateBuffer = NS::TransferPtr(device->newBuffer(16384, MTL::ResourceStorageModePrivate));
    REQUIRE(a);
    REQUIRE(alias);
    REQUIRE(disjoint);
    REQUIRE(saved);
    REQUIRE(privateBuffer);
    REQUIRE(a.get() != alias.get());

    auto encodeFill = [&](MTL::Buffer* buffer, float value)
    {
        encoder.setComputePipelineState(fill.get());
        encoder.setBuffer(buffer, 0, 0);
        encoder.setBytes(&value, sizeof(value), 1);
        encoder.dispatchThreads({4096, 1, 1}, {128, 1, 1});
    };
    auto encodeCopy = [&](MTL::Buffer* src, MTL::Buffer* dst, size_t offset, bool fallback = false)
    {
        encoder.setComputePipelineState(fallback ? unknown.get() : copy.get());
        encoder.setBuffer(src, 0, 0);
        encoder.setBuffer(dst, offset, 1);
        encoder.dispatchThreads({4096, 1, 1}, {128, 1, 1});
    };
    for (size_t round = 0; round < 8; ++round)
    {
        // Repeated command boundaries retain the same fence and backing heap.
        auto command = queue->commandBuffer();
        encoder.reset(command->computeCommandEncoder(MTL::DispatchTypeConcurrent));
        if (round) encoder.waitForFence(fence.get());
        for (size_t i = 0; i < 64; ++i)
        {
            encodeFill(a.get(), 1.0f);
            encodeFill(disjoint.get(), 2.0f);             // Independent heap allocation.
            encodeCopy(alias.get(), saved.get(), 0);    // RAW through a distinct alias.
            encodeFill(alias.get(), 7.0f);              // WAR: preserve the old read.
            encodeCopy(a.get(), saved.get(), 16384);
            encodeFill(a.get(), 9.0f);                  // WAW and WAR.
            encoder.setComputePipelineState(add.get());
            encoder.setBuffer(alias.get(), 0, 0);
            const float increment = 1.0f;
            encoder.setBytes(&increment, sizeof(increment), 1);
            encoder.dispatchThreads({4096, 1, 1}, {128, 1, 1});
            encodeCopy(a.get(), privateBuffer.get(), 0); // Unsupported memory fallback.
            encodeCopy(privateBuffer.get(), saved.get(), 32768, true);
        }
        encoder.updateFence(fence.get());
        encoder.endEncoding();
        command->commit();
        command->waitUntilCompleted();
        REQUIRE(command->status() == MTL::CommandBufferStatusCompleted);
        auto values = static_cast<float*>(saved->contents());
        for (size_t i = 0; i < 4096; ++i)
        {
            CHECK(values[i] == 1.0f);
            CHECK(values[4096 + i] == 7.0f);
            CHECK(values[8192 + i] == 10.0f);
            CHECK(static_cast<float*>(disjoint->contents())[i] == 2.0f);
        }
    }
}
#endif


TEST_CASE("DeviceMetal concurrent fused views reductions copies and pending destruction")
{
    for (size_t pass = 0; pass < 4; ++pass)
    {
        auto device = createDevice(DeviceType::kGPU_METAL);
        if (!device) return;
        TensorValue storage(1.0f, {16, 64}, device.get());
        TensorValue increment(0.25f, {16, 64}, device.get());
        std::vector<TensorValue> branches;
        for (size_t i = 0; i < 96; ++i)
        {
            auto left = storage.slice(1, 0, 32);
            auto right = storage.slice(1, 32, 64);
            auto sum = (left + right).sum();
            branches.push_back(sum);
            // An in-place write follows reads from both aliases and a reduction.
            storage += increment;
            auto copied = storage.contiguous();
            auto row = copied.slice(0, 0, 1);
            storage.sliceSet(row, 0, 15, 16, 1, true);
            if (i % 23 == 0) device->synchronize();
        }
        device->synchronize();
        for (size_t i = 0; i < branches.size(); ++i)
        {
            CHECK(branches[i].item<float>() == 1024.0f * (1.0f + i * 0.25f));
        }
        // Destruction must finish pending fused work and submitted work, including
        // temporary buffers retired while later batches reuse the allocator.
        branches.clear();
        for (size_t i = 0; i < 96; ++i)
        {
            storage = storage + increment;
        }
        std::array<float, 64 * 64> identity{};
        for (size_t i = 0; i < 64; ++i) identity[i * 64 + i] = 1.0f;
        TensorValue weights(identity.data(), identity.size(), DataType::kFloat32, {64, 64}, device.get());
        storage = storage.matmul(weights); // Leave an actual dispatch pending at destruction.
    }
}


TEST_CASE("DeviceMetal concurrent slice ranges preserve disjoint and overlapping view writes")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    for (size_t round = 0; round < 16; ++round)
    {
        TensorValue storage(-1.0f, {1, 80}, device.get());
        auto view = storage.slice(1, 8, 72);
        for (size_t head = 0; head < 4; ++head)
        {
            TensorValue values(static_cast<float>(head + 1), {1, 16}, device.get());
            view.sliceSet(values, 1, head * 16, (head + 1) * 16, 1, true);
        }
        auto before = view.sum();
        TensorValue replacement(9.0f, {1, 16}, device.get());
        view.sliceSet(replacement, 1, 12, 28, 1, true);
        auto after = view.sum();
        device->synchronize();
        CHECK(before.item<float>() == 160.0f);
        CHECK(after.item<float>() == 276.0f);
        for (size_t i = 0; i < 80; ++i)
        {
            float expected = -1.0f;
            if (i >= 8 && i < 72) expected = static_cast<float>((i - 8) / 16 + 1);
            if (i >= 20 && i < 36) expected = 9.0f;
            CHECK(storage.data<float>()[i] == expected);
        }
        auto stepped = storage.slice(1, 0, 80, 2);
        TensorValue sevens(7.0f, {1, 20}, device.get());
        stepped.sliceSet(sevens, 1, 0, 20, 1, true);
        device->synchronize();
        for (size_t i = 0; i < 40; i += 2) CHECK(storage.data<float>()[i] == 7.0f);
    }
}


TEST_CASE("DeviceMetal strided indexing preserves leading zero coordinates and broadcast offsets")
{
    aix::DeviceCPU reference;
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    for (size_t layout = 0; layout < 3; ++layout)
    {
        const Shape shape = layout == 0 ? Shape{257} : (layout == 1 ? Shape{1, 257} : Shape{2, 3, 5, 7});
        const size_t count = layout < 2 ? 257 : 210;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) values[i] = static_cast<float>(static_cast<int>(i % 97) - 48) / 4.0f;
        auto evaluate = [&](Device* target)
        {
            TensorValue storage(values.data(), values.size(), DataType::kFloat32, shape, target);
            auto view = layout == 0 ? storage.slice(0, 1, 256, 3)
                : (layout == 1 ? storage.slice(1, 1, 256, 3) : storage.slice(2, 1, 5, 2).transpose(1, 3));
            TensorValue scalar(2.0f, Shape{}, target);
            auto broadcast = scalar.broadcastTo(view.shape());
            auto divided = view / broadcast;
            target->synchronize(); // Exercise the ordinary strided kernel separately.
            auto fused = (divided + view) * broadcast - view;
            target->synchronize(); // Also exercise generated strided address calculation.
            return std::pair{std::move(divided), std::move(fused)};
        };
        auto expected = evaluate(&reference);
        auto actual = evaluate(device.get());
        for (size_t i = 0; i < actual.first.size(); ++i)
        {
            CHECK(actual.first.data<float>()[i] == expected.first.data<float>()[i]);
            CHECK(actual.second.data<float>()[i] == expected.second.data<float>()[i]);
        }
    }
}


#if defined(__APPLE__) && defined(__arm64__)
TEST_CASE("DeviceMetal concurrent no-work synchronization drains retired allocations")
{
    class InspectableDevice : public aix::metal::DeviceMetal
    {
    public:
        size_t pendingRetirements() const { return m_tempBuffers.size(); }
        size_t mappedAllocations() const { return m_allocMap.size(); }
        size_t cachedBytes() const { return m_bufferCache->size(); }
    };

    auto device = std::make_unique<InspectableDevice>();
    for (size_t i = 0; i < 8; ++i)
    {
        {
            TensorValue value(2.0f, {4096}, device.get());
            device->synchronize();
            CHECK(value.data<float>()[0] == 2.0f);
        }
        // The tensor is retired after synchronize, with no new GPU dispatch.
        CHECK(device->pendingRetirements() > 0);
        device->synchronize();
        CHECK(device->pendingRetirements() == 0);
        CHECK(device->mappedAllocations() == 0);
        device->emptyCache();
        CHECK(device->cachedBytes() == 0);
    }
    // Also cover allocations that have never been bound to a command encoder.
    auto memory = device->allocate(4096);
    device->deallocate(memory);
    device->synchronize();
    CHECK(device->pendingRetirements() == 0);
    CHECK(device->mappedAllocations() == 0);
    device->emptyCache();
    CHECK(device->cachedBytes() == 0);
    // Driver/shader-validation caches are not owned allocations and can remain
    // in currentAllocatedSize after device teardown. Check our exact owners above.
    device.reset();
}
#endif


TEST_CASE("DeviceMetal small buffers reuse allocator-sized cache entries after completion")
{
    for (size_t size : {1, 7, 8, 9, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097})
    {
        CAPTURE(size);
        auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
        if (!device) return;
        void* retired = nullptr;
        auto result = [&]()
        {
            TensorValue input(4.0f, {size}, device.get());
            retired = input.data();
            return input.sqrt();
        }();

        // Pending work must retain its storage until command-buffer completion.
        TensorValue other(9.0f, {size}, device.get());
        CHECK(other.data() != retired);
        device->synchronize();

        // The completed input is the only freed buffer in this device's cache.
        TensorValue reused(16.0f, {size}, device.get());
        CHECK(reused.data() == retired);
        auto reusedResult = reused.sqrt();
        device->synchronize();
        for (size_t i = 0; i < size; ++i)
        {
            CHECK(result.data<float>()[i] == doctest::Approx(2.0f));
            CHECK(other.data<float>()[i] == doctest::Approx(9.0f));
            CHECK(reusedResult.data<float>()[i] == doctest::Approx(4.0f));
        }
    }
}


TEST_CASE("DeviceMetal stride add parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto cpuOther = TensorValue({10.0f, 20.0f,
                                 30.0f, 40.0f,
                                 50.0f, 60.0f}, {3, 2}, &refDevice);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);
    auto metalOther = TensorValue({10.0f, 20.0f,
                                   30.0f, 40.0f,
                                   50.0f, 60.0f}, {3, 2}, device.get());

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc + cpuOther;
    auto actual = metalSrc + metalOther;
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride sqrt parity with sliced source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &refDevice).slice(1, 0, 4, 2);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, device.get()).slice(1, 0, 4, 2);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.sqrt();
    auto actual = metalSrc.sqrt();
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride reduceTo parity with broadcast source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f}, {1, 3}, &refDevice).broadcastTo({2, 3});
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f}, {1, 3}, device.get()).broadcastTo({2, 3});

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.sum({0}, true);
    auto actual = metalSrc.sum({0}, true);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride maxTo parity with broadcast source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 4.0f, 2.0f}, {1, 3}, &refDevice).broadcastTo({2, 3});
    auto metalSrc = TensorValue({1.0f, 4.0f, 2.0f}, {1, 3}, device.get()).broadcastTo({2, 3});

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.max(0, true);
    auto actual = metalSrc.max(0, true);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


#if defined(__APPLE__) && defined(__arm64__)
TEST_CASE("DeviceMetal fused kernels reuse runtime scalar bindings")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto* metal = dynamic_cast<aix::metal::DeviceMetal*>(device.get());
    REQUIRE(metal != nullptr);

    auto runExpression = [&](float mulScalar, float addScalar) {
        auto input = aix::tensor({1.0f, 2.0f,
                                  3.0f, 4.0f}, aix::Shape{2, 2}, { .m_device=device.get() });
        auto output = input * mulScalar + addScalar;
        device->synchronize();
        return output;
    };

    auto [hits0, misses0] = metal->fuseKernelCacheStats();

    auto first = runExpression(2.0f, 1.0f);
    auto [hits1, misses1] = metal->fuseKernelCacheStats();

    auto second = runExpression(3.0f, 5.0f);
    auto [hits2, misses2] = metal->fuseKernelCacheStats();

    CheckVectorApproxValues(first, aix::tensor({3.0f, 5.0f,
                                                7.0f, 9.0f}, aix::Shape{2, 2}));
    CheckVectorApproxValues(second, aix::tensor({8.0f, 11.0f,
                                                 14.0f, 17.0f}, aix::Shape{2, 2}));

    CHECK(misses1 > misses0);
    CHECK(misses2 == misses1);
    CHECK(hits2 > hits1);
    CHECK(hits1 >= hits0);
}
#endif


TEST_CASE("Device fusion preserves pending producer view indexing")
{
    DeviceCPU reference;
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(type));
        for (auto dtype : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
        {
            CAPTURE(static_cast<size_t>(dtype));
            for (bool transpose : {false, true})
            {
                CAPTURE(transpose);
                for (bool synchronizeProducer : {false, true})
                {
                    CAPTURE(synchronizeProducer);
                    for (bool viewFirst : {false, true})
                    {
                        CAPTURE(viewFirst);
                        auto input = TensorValue({0.0f, 2.0f, 7.0f, 0.0f, 0.0f, 11.0f}, {3, 2}, device.get(), dtype);
                        auto pending = input * TensorValue(1.0f, {}, device.get(), dtype);
                        if (synchronizeProducer) device->synchronize();
                        auto view = transpose ? pending.transpose(0, 1) : pending.reshape({2, 3});
                        CHECK(view.shape() == Shape{2, 3});
                        CHECK(view.strides() == (transpose ? Stride{1, 2} : Stride{3, 1}));
                        CHECK(view.storage() == pending.storage());
                        auto result = viewFirst ? view + TensorValue(0.0f, {2, 3}, device.get(), dtype) :
                                                  TensorValue(0.0f, {2, 3}, device.get(), dtype) + view;
                        // Do not observe or materialize the pending producer before recording its consumer.
                        device->synchronize();
                        CHECK(result.shape() == Shape{2, 3});
                        CHECK(result.dataType() == dtype);
                        auto expected = transpose ?
                            TensorValue({0.0f, 7.0f, 0.0f, 2.0f, 0.0f, 11.0f}, {2, 3}, &reference, dtype) :
                            TensorValue({0.0f, 2.0f, 7.0f, 0.0f, 0.0f, 11.0f}, {2, 3}, &reference, dtype);
                        CheckVectorApproxValues(result, expected);
                        CheckVectorApproxValues(pending, input);
                    }
                }
            }
        }
    }
}

TEST_CASE("Device fusion orders reindexed temporary chains")
{
    DeviceCPU reference;
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(type));
        for (auto dtype : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
        {
            CAPTURE(static_cast<size_t>(dtype));
            for (bool square : {false, true})
            {
                CAPTURE(square);
                auto input = square ? TensorValue({0.0f, 2.0f, 7.0f, 11.0f}, {2, 2}, device.get(), dtype) :
                                     TensorValue({0.0f, 2.0f, 7.0f, 0.0f, 0.0f, 11.0f}, {3, 2}, device.get(), dtype);
                // Both the producer and the reindexed result are temporary: neither is retained for observation.
                auto result = -(TensorValue(0.0f, square ? Shape{2, 2} : Shape{2, 3}, device.get(), dtype) +
                                (input * TensorValue(1.0f, {}, device.get(), dtype)).transpose(0, 1));
                device->synchronize();
                auto expected = square ? TensorValue({0.0f, -7.0f, -2.0f, -11.0f}, {2, 2}, &reference, dtype) :
                                         TensorValue({0.0f, -7.0f, 0.0f, -2.0f, 0.0f, -11.0f}, {2, 3}, &reference, dtype);
                CheckVectorApproxValues(result, expected);
            }
        }
    }
}

TEST_CASE("FuseEngine materializes changed mappings and preserves same-lane fusion")
{
    for (bool multiOutput : {false, true})
    {
        CAPTURE(multiOutput);
        for (bool viewFirst : {false, true})
        {
            CAPTURE(viewFirst);
            for (size_t layout = 0; layout < 4; ++layout)
            {
                CAPTURE(layout);
                std::vector<aix::fuse::FusedSubgraphDescriptor> fused;
                std::vector<aix::fuse::OpType> singles;
                std::vector<std::string> emitted;
                aix::fuse::FuseCallbacks callbacks;
                callbacks.emitFused = [&](const aix::fuse::FusedSubgraphDescriptor& desc)
                {
                    fused.push_back(desc);
                    emitted.push_back("fused");
                };
                callbacks.emitSingle = [&](const aix::fuse::OpRecord& op)
                {
                    singles.push_back(op.type);
                    emitted.push_back("single");
                };
                callbacks.finishFlush = [] {};
                aix::fuse::FuseConfig config;
                config.maxBufferSlots = 31;
                config.multiOutputKernels = multiOutput;
                aix::fuse::FuseEngine engine(config, callbacks);

                DeviceCPU device;
                auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, &device);
                auto stage = TensorValue(Shape{2, 2}, &device);
                auto pending = TensorValue(Shape{2, 2}, &device);
                auto other = TensorValue(Shape{2, 2}, &device);
                auto consumed = TensorValue(Shape{2, 2}, &device);
                auto output = TensorValue(Shape{2, 2}, &device);
                auto producer = pending.deviceParams();
                auto view = producer;
                if (layout == 0)
                {
                    // A contiguous reshape changes rank/shape but not the lane mapping.
                    view.shape = {4};
                    view.strides = {1};
                }
                else if (layout == 1)
                {
                    // A square transpose changes only strides, not shape or element count.
                    view.strides = {1, 2};
                    view.isContiguous = false;
                }
                else if (layout == 2)
                {
                    // Strided producer/consumer mappings that agree remain fusible.
                    producer.strides = view.strides = {1, 2};
                    producer.isContiguous = view.isContiguous = false;
                }
                else
                {
                    view.strides = {0, 1};
                    view.isContiguous = false;
                }
                engine.record(aix::fuse::OpType::Negate, input.deviceParams(), stage.deviceParams());
                engine.record(aix::fuse::OpType::Negate, stage.deviceParams(), producer);
                engine.record(aix::fuse::OpType::Add, viewFirst ? view : other.deviceParams(),
                              viewFirst ? other.deviceParams() : view, consumed.deviceParams());
                engine.record(aix::fuse::OpType::Negate, consumed.deviceParams(), output.deviceParams());
                engine.flush();

                REQUIRE(fused.size() == 1);
                const bool reindexed = layout == 1 || layout == 3;
                CHECK(fused[0].ops.size() == (reindexed ? 2 : 4));
                if (reindexed)
                {
                    CHECK(singles == std::vector<aix::fuse::OpType>{aix::fuse::OpType::Add, aix::fuse::OpType::Negate});
                    CHECK(emitted == std::vector<std::string>{"fused", "single", "single"});
                    // The producer escapes to fallback even when no live-buffer retention requests a store.
                    REQUIRE(fused[0].outputBuffers.size() == 1);
                    CHECK(fused[0].outputBuffers[0].data == producer.data);
                    CHECK(fused[0].outputBuffers[0].strides == producer.strides);
                }
                else
                {
                    CHECK(singles.empty());
                    CHECK(emitted == std::vector<std::string>{"fused"});
                    CHECK(fused[0].allContiguous == (layout == 0));
                }
            }
        }
    }
}

TEST_CASE("FuseEngine does not fuse ops that depend on rejected fallback producers")
{
    std::vector<std::string> emitted;

    aix::fuse::FuseCallbacks callbacks;
    callbacks.emitFused = [&](const aix::fuse::FusedSubgraphDescriptor& subgraph)
    {
        emitted.push_back("fused:" + std::to_string(subgraph.ops.size()));
    };
    callbacks.emitSingle = [&](const aix::fuse::OpRecord& op)
    {
        switch (op.type)
        {
            case aix::fuse::OpType::Add: emitted.push_back("Add"); break;
            case aix::fuse::OpType::Exp: emitted.push_back("Exp"); break;
            case aix::fuse::OpType::Tanh: emitted.push_back("Tanh"); break;
            default: emitted.push_back("Other"); break;
        }
    };
    callbacks.finishFlush = [] {};

    aix::fuse::FuseConfig config;
    config.maxBufferSlots = 31;
    config.supportsStridedFusion = false;

    aix::fuse::FuseEngine engine(config, callbacks);

    std::array<float, 4> storageA {1.0f, 2.0f, 3.0f, 4.0f};
    std::array<float, 4> storageB {5.0f, 6.0f, 7.0f, 8.0f};
    std::array<float, 4> storageC {};
    std::array<float, 4> storageD {};
    std::array<float, 4> storageE {};

    DeviceTensorParams input0;
    input0.data = storageA.data();
    input0.dtype = aix::DataType::kFloat32;
    input0.isContiguous = true;
    input0.shape = {2, 2};
    input0.size = 4;
    input0.strides = {2, 1};

    DeviceTensorParams input1;
    input1.data = storageB.data();
    input1.dtype = aix::DataType::kFloat32;
    input1.isContiguous = false;
    input1.shape = {2, 2};
    input1.size = 4;
    input1.strides = {0, 1};

    DeviceTensorParams intermediate0;
    intermediate0.data = storageC.data();
    intermediate0.dtype = aix::DataType::kFloat32;
    intermediate0.isContiguous = true;
    intermediate0.shape = {2, 2};
    intermediate0.size = 4;
    intermediate0.strides = {2, 1};

    DeviceTensorParams intermediate1 = intermediate0;
    intermediate1.data = storageD.data();

    DeviceTensorParams output = intermediate0;
    output.data = storageE.data();

    engine.record(aix::fuse::OpType::Add, input0, input1, intermediate0);
    engine.record(aix::fuse::OpType::Exp, intermediate0, intermediate1);
    engine.record(aix::fuse::OpType::Tanh, intermediate1, output);
    engine.flush();

    CHECK(emitted == std::vector<std::string>{"Add", "Exp", "Tanh"});
}


TEST_CASE("DeviceMetal in place fusion respects prior writer order")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto refMoment = TensorValue({1.0f, -2.0f,
                                  3.0f, -4.0f}, {2, 2}, &refDevice);
    auto refGrad = TensorValue({0.5f, 1.0f,
                                -1.5f, 2.0f}, {2, 2}, &refDevice);
    refMoment *= 0.9f;
    refMoment += 0.1f * refGrad;
    auto refOutput = refMoment / 0.5f;

    auto metalMoment = TensorValue({1.0f, -2.0f,
                                    3.0f, -4.0f}, {2, 2}, device.get());
    auto metalGrad = TensorValue({0.5f, 1.0f,
                                  -1.5f, 2.0f}, {2, 2}, device.get());

    device->synchronize();

    metalMoment *= 0.9f;
    metalMoment += 0.1f * metalGrad;
    auto metalOutput = metalMoment / 0.5f;
    device->synchronize();

    CheckVectorApproxValues(metalMoment, refMoment);
    CheckVectorApproxValues(metalOutput, refOutput);
}


TEST_CASE("DeviceMetal stride argmaxTo parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 8.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 8.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.argmax(1, true);
    auto actual = metalSrc.argmax(1, true);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride sum parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.sum();
    auto actual = metalSrc.sum();
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride max parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 8.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 8.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.max();
    auto actual = metalSrc.max();
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride argmax parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 8.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 8.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.argmax();
    auto actual = metalSrc.argmax();
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride transpose parity with sliced source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &refDevice).slice(1, 0, 4, 2);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, device.get()).slice(1, 0, 4, 2);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.transpose(0, 1).contiguous();
    auto actual = metalSrc.transpose(0, 1).contiguous();
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride sliceSet parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuDst = TensorValue(0.0f, {3, 4}, &refDevice);
    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalDst = TensorValue(0.0f, {3, 4}, device.get());
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuDst.sliceSet(cpuSrc, 1, 1, 3, 1);
    auto actual = metalDst.sliceSet(metalSrc, 1, 1, 3, 1);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride sliceSet parity with destination view")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto metalBase = TensorValue(0.0f, {2, 3}, device.get());
    auto metalDst = metalBase.transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f}, {3, 1}, device.get());

    CHECK_FALSE(metalDst.isContiguous());

    metalDst.sliceSet(metalSrc, 1, 1, 2, 1, true);
    device->synchronize();

    auto expectedBase = TensorValue({0.0f, 0.0f, 0.0f,
                                     1.0f, 2.0f, 3.0f}, {2, 3}, &refDevice);
    auto expectedDst = expectedBase.transpose(0, 1);

    CheckVectorApproxValues(metalBase, expectedBase);
    CheckVectorApproxValues(metalDst, expectedDst);
}


TEST_CASE("DeviceMetal stride indexSelect parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f, 16.0f,
                               17.0f, 18.0f, 19.0f, 20.0f,
                               21.0f, 22.0f, 23.0f, 24.0f}, {2, 3, 4}, &refDevice).transpose(0, 1);
    auto cpuIndices = TensorValue({1, 0}, {2}, &refDevice, DataType::kInt32);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f,
                                 21.0f, 22.0f, 23.0f, 24.0f}, {2, 3, 4}, device.get()).transpose(0, 1);
    auto metalIndices = TensorValue({1, 0}, {2}, device.get(), DataType::kInt32);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.indexSelect(1, cpuIndices);
    auto actual = metalSrc.indexSelect(1, metalIndices);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride indexSelect parity with strided indices")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &refDevice);
    auto cpuIndices = TensorValue({1, 9, 0, 7}, {4}, &refDevice, DataType::kInt32).slice(0, 0, 4, 2);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, device.get());
    auto metalIndices = TensorValue({1, 9, 0, 7}, {4}, device.get(), DataType::kInt32).slice(0, 0, 4, 2);

    CHECK_FALSE(cpuIndices.isContiguous());
    CHECK_FALSE(metalIndices.isContiguous());

    auto expected = TensorValue({5.0f, 6.0f, 7.0f, 8.0f,
                                 1.0f, 2.0f, 3.0f, 4.0f}, {2, 4}, &refDevice);
    auto actual = metalSrc.indexSelect(0, metalIndices);
    device->synchronize();

    CheckVectorApproxValues(cpuIndices.contiguous(), TensorValue({1, 0}, {2}, &refDevice, DataType::kInt32));
    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride indexAdd parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuDst = TensorValue(0.0f, {3, 4}, &refDevice);
    auto cpuIndices = TensorValue({2, 0}, {2}, &refDevice, DataType::kInt32);
    auto cpuSrc = TensorValue({1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f,
                               7.0f, 8.0f}, {4, 2}, &refDevice).transpose(0, 1);
    auto metalDst = TensorValue(0.0f, {3, 4}, device.get());
    auto metalIndices = TensorValue({2, 0}, {2}, device.get(), DataType::kInt32);
    auto metalSrc = TensorValue({1.0f, 2.0f,
                                 3.0f, 4.0f,
                                 5.0f, 6.0f,
                                 7.0f, 8.0f}, {4, 2}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuDst.indexAdd(0, cpuIndices, cpuSrc);
    auto actual = metalDst.indexAdd(0, metalIndices, metalSrc);
    device->synchronize();

    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride indexAdd parity with strided indices")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuDst = TensorValue(0.0f, {3, 2}, &refDevice);
    auto cpuIndices = TensorValue({2, 9, 0, 7}, {4}, &refDevice, DataType::kInt32).slice(0, 0, 4, 2);
    auto cpuSrc = TensorValue({1.0f, 2.0f,
                               3.0f, 4.0f}, {2, 2}, &refDevice);
    auto metalDst = TensorValue(0.0f, {3, 2}, device.get());
    auto metalIndices = TensorValue({2, 9, 0, 7}, {4}, device.get(), DataType::kInt32).slice(0, 0, 4, 2);
    auto metalSrc = TensorValue({1.0f, 2.0f,
                                 3.0f, 4.0f}, {2, 2}, device.get());

    CHECK_FALSE(cpuIndices.isContiguous());
    CHECK_FALSE(metalIndices.isContiguous());

    auto expected = TensorValue({3.0f, 4.0f,
                                 0.0f, 0.0f,
                                 1.0f, 2.0f}, {3, 2}, &refDevice);
    auto actual = metalDst.indexAdd(0, metalIndices, metalSrc);
    device->synchronize();

    CheckVectorApproxValues(cpuIndices.contiguous(), TensorValue({2, 0}, {2}, &refDevice, DataType::kInt32));
    CheckVectorApproxValues(actual, expected);
}


TEST_CASE("DeviceMetal stride tril parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.tril(-1);
    auto expectedSource = cpuSrc.contiguous();
    auto actual = metalSrc.tril(-1);
    device->synchronize();

    CHECK(actual.storage() != metalSrc.storage());
    CheckVectorApproxValues(actual, expected);
    CheckVectorApproxValues(metalSrc, expectedSource);
}


TEST_CASE("DeviceMetal stride triu parity with transposed source")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuSrc = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice).transpose(0, 1);
    auto metalSrc = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuSrc.isContiguous());
    CHECK_FALSE(metalSrc.isContiguous());

    auto expected = cpuSrc.triu(1);
    auto expectedSource = cpuSrc.contiguous();
    auto actual = metalSrc.triu(1);
    device->synchronize();

    CHECK(actual.storage() != metalSrc.storage());
    CheckVectorApproxValues(actual, expected);
    CheckVectorApproxValues(metalSrc, expectedSource);
}


TEST_CASE("DeviceMetal stride matmul parity with transposed lhs operand")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuLhs = TensorValue({1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f}, {3, 2}, &refDevice).transpose(0, 1);
    auto cpuRhs = TensorValue({7.0f, 8.0f,
                               9.0f, 10.0f,
                               11.0f, 12.0f}, {3, 2}, &refDevice);
    auto metalLhs = TensorValue({1.0f, 2.0f,
                                 3.0f, 4.0f,
                                 5.0f, 6.0f}, {3, 2}, device.get()).transpose(0, 1);
    auto metalRhs = TensorValue({7.0f, 8.0f,
                                 9.0f, 10.0f,
                                 11.0f, 12.0f}, {3, 2}, device.get());

    CHECK_FALSE(cpuLhs.isContiguous());
    CHECK_FALSE(metalLhs.isContiguous());

    auto expected = cpuLhs.matmul(cpuRhs);
    auto actual = metalLhs.matmul(metalRhs);
    device->synchronize();

    CHECK(verifyResults(expected, actual, EPSILON_MATMUL_F32_METAL));
}


TEST_CASE("DeviceMetal stride matmul parity with transposed rhs operand")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuLhs = TensorValue({1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f}, {2, 3}, &refDevice);
    auto cpuRhs = TensorValue({7.0f, 8.0f, 9.0f,
                               10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f}, {4, 3}, &refDevice).transpose(0, 1);
    auto metalLhs = TensorValue({1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f}, {2, 3}, device.get());
    auto metalRhs = TensorValue({7.0f, 8.0f, 9.0f,
                                 10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f,
                                 16.0f, 17.0f, 18.0f}, {4, 3}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuRhs.isContiguous());
    CHECK_FALSE(metalRhs.isContiguous());

    auto expected = cpuLhs.matmul(cpuRhs);
    auto actual = metalLhs.matmul(metalRhs);
    device->synchronize();

    CHECK(verifyResults(expected, actual, EPSILON_MATMUL_F32_METAL));
}


TEST_CASE("DeviceMetal stride matmul parity with M=1 transposed rhs operand")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuLhs = TensorValue({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}, &refDevice);
    auto cpuRhs = TensorValue({1.0f, 5.0f, 9.0f,
                               2.0f, 6.0f, 10.0f,
                               3.0f, 7.0f, 11.0f,
                               4.0f, 8.0f, 12.0f}, {3, 4}, &refDevice).transpose(0, 1);
    auto metalLhs = TensorValue({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}, device.get());
    auto metalRhs = TensorValue({1.0f, 5.0f, 9.0f,
                                 2.0f, 6.0f, 10.0f,
                                 3.0f, 7.0f, 11.0f,
                                 4.0f, 8.0f, 12.0f}, {3, 4}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuRhs.isContiguous());
    CHECK_FALSE(metalRhs.isContiguous());

    auto expected = cpuLhs.matmul(cpuRhs);
    auto actual = metalLhs.matmul(metalRhs);
    device->synchronize();

    CHECK(verifyResults(expected, actual, EPSILON_MATMUL_F32_METAL));
}


TEST_CASE("DeviceMetal stride matmul parity with transposed lhs and rhs operands")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuLhs = TensorValue({1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f,
                               7.0f, 8.0f}, {4, 2}, &refDevice).transpose(0, 1);
    auto cpuRhs = TensorValue({9.0f, 10.0f, 11.0f, 12.0f,
                               13.0f, 14.0f, 15.0f, 16.0f,
                               17.0f, 18.0f, 19.0f, 20.0f}, {3, 4}, &refDevice).transpose(0, 1);
    auto metalLhs = TensorValue({1.0f, 2.0f,
                                 3.0f, 4.0f,
                                 5.0f, 6.0f,
                                 7.0f, 8.0f}, {4, 2}, device.get()).transpose(0, 1);
    auto metalRhs = TensorValue({9.0f, 10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f}, {3, 4}, device.get()).transpose(0, 1);

    CHECK_FALSE(cpuLhs.isContiguous());
    CHECK_FALSE(cpuRhs.isContiguous());
    CHECK_FALSE(metalLhs.isContiguous());
    CHECK_FALSE(metalRhs.isContiguous());

    auto expected = cpuLhs.matmul(cpuRhs);
    auto actual = metalLhs.matmul(metalRhs);
    device->synchronize();

    CHECK(verifyResults(expected, actual, EPSILON_MATMUL_F32_METAL));
}


TEST_CASE("DeviceCPU Tests - createDevice")
{
    std::vector<aix::DeviceType> deviceTypes
    {
        aix::DeviceType::kCPU,
        aix::DeviceType::kGPU_METAL
    };

    for (const auto type : deviceTypes)
    {
        auto device = aix::createDevice(type);
        if (!device) continue;
        CHECK(device->type() == type);
    }
}


TEST_CASE("DeviceCPU Tests - Allocate")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testAllocate(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Add")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testAdd(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Sub")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSub(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Neg")
{
    // For each available device, test negation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testNeg(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Sqrt")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSqrt(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Sin")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSin(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Cos")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testCos(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Tanh")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTanh(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Log")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testLog(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Exp")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testExp(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Max")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testMax(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Argmax")
{
    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        for (auto size : testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testArgmax(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - ArgmaxIndices")
{
    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        for (auto size : testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testArgmaxIndices(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Argmax with dim")
{
    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testArgmaxWithDim(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - ArgmaxIndices with dim")
{
    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testArgmaxIndicesWithDim(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Max with dim")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testMaxWithDim(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Slice")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSlice(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - SliceSet")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testSliceSet(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Tril")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTril(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Triu")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTriu(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Pow")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testPow(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Mul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testMul(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Div")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testDiv(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - MatMul")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t i = 1; i < 8; i+=2)
            {
                for (size_t m = 1; m < 8; m+=2)
                {
                    auto device2 = aix::createDevice(deviceType);
                    CHECK(testMatMul(&*device2, n, i, m));
                }
            }
        }

        for (size_t m=1; m<=1024; m*=2)
        {
            CHECK(testMatMul(&*device, m, m, m));
        }

        for (size_t m=0; m<100; ++m)
        {
            CHECK(testMatMul(&*device, 32 * (1 + std::rand() % 5),
                                       32 * (1 + std::rand() % 5),
                                       32 * (1 + std::rand() % 5)));
        }

        CHECK(testMatMul(&*device, 257, 129, 513));
        CHECK(testMatMul(&*device, 258, 130, 514));
        CHECK(testMatMul(&*device, 256, 128, 512));
        CHECK(testMatMul(&*device, 255, 127, 511));
        CHECK(testMatMul(&*device, 254, 126, 510));

        CHECK(testMatMul(&*device, 129, 257, 513));
        CHECK(testMatMul(&*device, 130, 258, 514));
        CHECK(testMatMul(&*device, 128, 256, 512));
        CHECK(testMatMul(&*device, 127, 255, 511));
        CHECK(testMatMul(&*device, 126, 254, 510));

        CHECK(testMatMul(&*device, 257, 513, 129));
        CHECK(testMatMul(&*device, 258, 514, 130));
        CHECK(testMatMul(&*device, 256, 512, 128));
        CHECK(testMatMul(&*device, 255, 511, 127));
        CHECK(testMatMul(&*device, 254, 510, 126));

        CHECK(testMatMul(&*device, 129, 513, 257));
        CHECK(testMatMul(&*device, 130, 514, 258));
        CHECK(testMatMul(&*device, 128, 512, 256));
        CHECK(testMatMul(&*device, 127, 511, 255));
        CHECK(testMatMul(&*device, 126, 510, 254));

        CHECK(testMatMul(&*device, 513, 257, 129));
        CHECK(testMatMul(&*device, 514, 258, 130));
        CHECK(testMatMul(&*device, 512, 256, 128));
        CHECK(testMatMul(&*device, 511, 255, 127));
        CHECK(testMatMul(&*device, 510, 254, 126));

        CHECK(testMatMul(&*device, 513, 129, 257));
        CHECK(testMatMul(&*device, 514, 130, 258));
        CHECK(testMatMul(&*device, 512, 128, 256));
        CHECK(testMatMul(&*device, 511, 127, 255));
        CHECK(testMatMul(&*device, 510, 126, 254));
    }
}


TEST_CASE("DeviceCPU Tests - Transpose2D")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 1; n < 8; n+=2)
        {
            for (size_t m = 1; m < 8; m+=2)
            {
                auto device2 = aix::createDevice(deviceType);
                CHECK(testTranspose2D(&*device2, n, m));
            }
        }

        CHECK(testTranspose2D(&*device, 129, 513));
        CHECK(testTranspose2D(&*device, 130, 514));
        CHECK(testTranspose2D(&*device, 128, 512));
        CHECK(testTranspose2D(&*device, 127, 511));
        CHECK(testTranspose2D(&*device, 126, 510));
    }
}


TEST_CASE("DeviceCPU Tests - Transpose")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 0; n < 100; ++n)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testTranspose(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Permute")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        for (size_t n = 0; n < 100; ++n)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testPermute(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Copy")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testCopy(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - Fill")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testFill(&*device2, size));
        }
    }
}


TEST_CASE("DeviceCPU Tests - FillMin")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (auto size: testSizes)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testFillMin(&*device2, size));
        }
    }
}


TEST_CASE("Device Tests - reduceTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            testReduceTo(&*device2, static_cast<uint32_t>(i));
        }
    }
}


TEST_CASE("Device reduction API rejects unsupported operations modes and dtypes before mutation")
{
    DeviceCPU cpu;
    auto metal = createDevice(DeviceType::kGPU_METAL);
    for (Device* device : {static_cast<Device*>(&cpu), metal.get()})
    {
        if (!device) continue;
        auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device);
        auto output = TensorValue(99.0f, {2, 3}, device);
        device->synchronize();
        auto src = input.deviceParams();
        auto dst = output.deviceParams();
        dst.shape = {1, 3};
        dst.size = 3;
        const size_t axis[] = {0};
        auto rejected = [&](DeviceTensorParams source, DeviceTensorParams destination, ReduceOp op)
        {
            CHECK_THROWS_AS(device->reduce(source, destination, op, axis), std::invalid_argument);
            for (size_t i = 0; i < output.size(); ++i) CHECK(output.data<float>()[i] == 99);
        };

        rejected(src, dst, static_cast<ReduceOp>(99));
        const size_t multiple[] = {0, 1};
        auto allDst = dst;
        allDst.shape = {1, 1};
        allDst.strides = {1, 1};
        allDst.size = 1;
        CHECK_THROWS_AS(device->argmax(src, allDst, multiple), std::invalid_argument);

        auto bad = src;
        bad.dtype = static_cast<DataType>(DataTypeCount);
        rejected(bad, dst, ReduceOp::kSum);
        bad = dst;
        bad.dtype = DataType::kInt32;
        rejected(src, bad, ReduceOp::kSum);
        rejected(src, bad, ReduceOp::kMax);
        CHECK_THROWS_AS(device->argmax(src, dst, axis), std::invalid_argument);

        uint8_t values[] = {1, 3, 3, 2};
        int32_t winner = -1;
        DeviceTensorParams valid{.data=values, .dtype=DataType::kUInt8, .shape={4}, .size=4, .strides={1}};
        DeviceTensorParams indexDst{.data=&winner, .dtype=DataType::kInt32, .shape={1}, .size=1, .strides={1}};
        device->argmax(valid, indexDst);
        CHECK(winner == 1);
    }
}

TEST_CASE("Device reduction maps offset multi-axis transpose and broadcast coordinates")
{
    DeviceCPU device;
    float input[64];
    for (size_t i = 0; i < 64; ++i) input[i] = static_cast<float>(i);
    float output[24]{};
    DeviceTensorParams src{.data=input, .offset=5, .shape={2, 3, 4}, .size=24, .strides={12, 4, 1}};
    DeviceTensorParams dst{.data=output, .shape={2, 1, 4}, .size=8, .strides={4, 4, 1}};
    size_t axis[] = {1};
    device.reduce(src, dst, ReduceOp::kSum, axis);
    for (size_t i = 0; i < 8; ++i) CHECK(output[i] == 27 + 36 * (i / 4) + 3 * (i % 4));

    const size_t axes[] = {2, 0};
    DeviceTensorParams destination{.data=output, .shape={1, 3, 1}, .size=3, .strides={3, 1, 1}};
    std::fill_n(output, 3, 100);
    device.reduce(src, destination, ReduceOp::kSum, axes, true);
    for (size_t i = 0; i < 3; ++i) CHECK(output[i] == 200 + 32 * i);

    src = {.data=input, .isContiguous=false, .shape={3, 2}, .size=6, .strides={1, 3}};
    dst = {.data=output, .shape={3, 1}, .size=3, .strides={1, 1}};
    axis[0] = 1;
    device.reduce(src, dst, ReduceOp::kSum, axis);
    for (size_t i = 0; i < 3; ++i) CHECK(output[i] == 3 + 2 * i);
    src = {.data=input, .isContiguous=false, .shape={2, 3}, .size=6, .strides={0, 1}};
    dst = {.data=output, .shape={1, 3}, .size=3, .strides={3, 1}};
    axis[0] = 0;
    device.reduce(src, dst, ReduceOp::kSum, axis);
    for (size_t i = 0; i < 3; ++i) CHECK(output[i] == 2 * i);

    src = {.data=input, .shape={2, 3}, .size=6, .strides={3, 1}};
    device.reduce(src, dst, ReduceOp::kSum, axis);
    for (size_t i = 0; i < 3; ++i) CHECK(output[i] == 3 + 2 * i);
    dst = {.data=output, .shape={2, 1}, .size=2, .strides={1, 1}};
    axis[0] = 1;
    device.reduce(src, dst, ReduceOp::kSum, axis);
    CHECK(output[0] == 3);
    CHECK(output[1] == 12);
}

TEST_CASE("Device reduction handles no axes singleton scalar and high rank")
{
    DeviceCPU device;
    float input[] = {1, 2, 3, 4};
    float output[4]{};
    DeviceTensorParams src{.data=input, .shape={2, 1, 2}, .size=4, .strides={2, 2, 1}};
    DeviceTensorParams dst{.data=output, .shape={2, 1, 2}, .size=4, .strides={2, 2, 1}};
    device.contiguous(src, dst);
    for (size_t i = 0; i < 4; ++i) CHECK(output[i] == input[i]);
    const size_t singleton[] = {1};
    std::fill_n(output, 4, 99);
    device.reduce(src, dst, ReduceOp::kSum, singleton);
    for (size_t i = 0; i < 4; ++i) CHECK(output[i] == input[i]);

    src = {.data=input, .size=1};
    dst = {.data=output, .size=1};
    output[0] = 99;
    device.reduce(src, dst, ReduceOp::kSum);
    CHECK(output[0] == 1);
    output[0] = 99;
    device.contiguous(src, dst);
    CHECK(output[0] == 1);

    src.shape.assign(17, 1);
    src.strides.assign(17, 1);
    dst.shape = src.shape;
    dst.strides = src.strides;
    output[0] = 99;
    device.reduce(src, dst, ReduceOp::kSum);
    CHECK(output[0] == 1);
}

TEST_CASE("DeviceMetal native reduction explicit route and dtype matrix")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    DeviceCPU reference;
    struct Fixture
    {
        Shape shape;
        Shape output;
        Index axes;
        std::vector<float> values;
        std::vector<float> sums;
        std::vector<float> maxima;
    };
    std::vector<Fixture> fixtures = {
        {{3, 8}, {3, 1}, {1}, {1, 2, 3, 4, 1, 2, 3, 4,
                               2, 3, 4, 5, 2, 3, 4, 5,
                               3, 4, 5, 6, 3, 4, 5, 6}, {20, 28, 36}, {4, 5, 6}},
        {{3, 7}, {3, 1}, {1}, {1, 2, 3, 1, 2, 3, 1,
                               2, 3, 1, 2, 3, 1, 2,
                               3, 1, 2, 3, 1, 2, 3}, {13, 14, 15}, {3, 3, 3}},
        {{5, 3}, {1, 3}, {0}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, {5, 10, 15}, {1, 2, 3}},
        {{2, 2}, {1, 2}, {0}, {1, 2, 3, 4}, {4, 6}, {3, 4}},
        {{2, 2}, {2, 1}, {1}, {1, 2, 3, 4}, {3, 7}, {2, 4}},
        {{2, 2, 2}, {1, 2, 1}, {2, 0}, {1, 2, 3, 4, 5, 6, 7, 8}, {14, 22}, {6, 8}},
        {{3, 4, 5}, {3, 1, 5}, {1}, {}, std::vector<float>(15, 10), std::vector<float>(15, 4)}
    };
    for (size_t i = 0; i < 60; ++i) fixtures.back().values.push_back(static_cast<float>((i / 5) % 4 + 1));
    for (size_t dtypeIndex = 1; dtypeIndex < DataTypeCount; ++dtypeIndex)
    {
        const auto dtype = static_cast<DataType>(dtypeIndex);
        for (const auto& fixture : fixtures)
        {
            CAPTURE(dtypeIndex);
            CAPTURE(fixture.shape);
            CAPTURE(fixture.axes);
            auto src = TensorValue(fixture.values.data(), fixture.values.size(), DataType::kFloat32,
                                   fixture.shape, device.get(), dtype);
            auto dst = TensorValue(99.0f, fixture.output, device.get(), dtype);
            for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
            {
                const auto& values = op == ReduceOp::kSum ? fixture.sums : fixture.maxima;
                dst.fill(isIntegralDataType(dtype) ? 99.0f : std::numeric_limits<float>::quiet_NaN());
                device->reduce(src.deviceParams(), dst.deviceParams(), op, fixture.axes);
                device->synchronize();
                auto overwritten = dst.to(&reference).to(DataType::kFloat32);
                for (size_t i = 0; i < values.size(); ++i) CHECK(overwritten.data<float>()[i] == values[i]);
                std::vector<float> seeds(values.size());
                for (size_t i = 0; i < seeds.size(); ++i) seeds[i] = static_cast<float>(i + 1);
                dst = TensorValue(seeds.data(), seeds.size(), DataType::kFloat32, fixture.output, device.get(), dtype);
                for (size_t call = 1; call <= 2; ++call)
                {
                    device->reduce(src.deviceParams(), dst.deviceParams(), op, fixture.axes, true);
                    device->synchronize();
                    auto actual = dst.to(&reference).to(DataType::kFloat32);
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        CHECK(actual.data<float>()[i] == (op == ReduceOp::kSum ? seeds[i] + call * values[i] :
                                                                                  std::max(seeds[i], values[i])));
                    }
                }
            }
        }
    }
}

TEST_CASE("DeviceMetal native reduction max preserves integer minimum identities")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    DeviceCPU reference;
    struct LimitCase
    {
        DataType dtype;
        int64_t minimum;
    };
    const LimitCase cases[] =
    {
        {DataType::kInt64, std::numeric_limits<int64_t>::lowest()},
        {DataType::kInt32, std::numeric_limits<int32_t>::lowest()},
        {DataType::kInt16, std::numeric_limits<int16_t>::lowest()},
        {DataType::kInt8, std::numeric_limits<int8_t>::lowest()},
        {DataType::kUInt8, 0}
    };
    for (const auto& [dtype, minimum] : cases)
    {
        CAPTURE(static_cast<size_t>(dtype));
        auto broadcast = TensorValue(&minimum, 1, DataType::kInt64, {}, device.get(), dtype).broadcastTo({3, 4});
        auto dense = broadcast.contiguous();
        auto scalarInner = dense.reshape({4, 3});
        for (const auto* src : {&broadcast, &dense, &scalarInner})
        {
            for (size_t axis : {size_t{0}, size_t{1}})
            {
                auto output = src->shape();
                output[axis] = 1;
                auto dst = TensorValue(99, output, device.get(), dtype);
                const size_t axes[] = {axis};
                device->reduce(src->deviceParams(), dst.deviceParams(), ReduceOp::kMax, axes);
                device->synchronize();
                auto actual = dst.to(&reference).to(DataType::kInt64);
                for (size_t i = 0; i < actual.size(); ++i) CHECK(actual.data<int64_t>()[i] == minimum);
            }
        }
    }
}

TEST_CASE("DeviceMetal native reduction global noaxis singleton and layout routes")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    DeviceCPU reference;
    for (size_t dtypeIndex = 1; dtypeIndex < DataTypeCount; ++dtypeIndex)
    {
        const auto dtype = static_cast<DataType>(dtypeIndex);
        CAPTURE(dtypeIndex);
        auto base = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, device.get(), dtype);
        const size_t axis[] = {1};
        for (auto src : {base.slice(0, 1, 3), base.slice(0, 1, 3).slice(1, 0, 4, 2), base.transpose(0, 1)})
        {
            Shape output = src.shape();
            output[1] = 1;
            auto dst = TensorValue(99.0f, output, device.get(), dtype);
            device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis);
            device->synchronize();
            const std::vector<float> values = src.shape()[0] == 4 ? std::vector<float>{15, 18, 21, 24} :
                (src.shape()[1] == 2 ? std::vector<float>{12, 20} : std::vector<float>{26, 42});
            CheckVectorApproxValues(dst, TensorValue(values.data(), values.size(), DataType::kFloat32,
                                                     output, &reference, dtype));
            for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
            {
                auto copied = TensorValue(99.0f, src.shape(), device.get(), dtype);
                device->contiguous(src.deviceParams(), copied.deviceParams());
                device->synchronize();
                auto hostSource = src.contiguous();
                device->synchronize();
                CheckVectorApproxValues(copied, hostSource);
                if (op == ReduceOp::kSum) device->add(src.deviceParams(), copied.deviceParams(), copied.deviceParams());
                device->synchronize();
                auto expected = hostSource.to(&reference);
                if (op == ReduceOp::kSum) expected += expected;
                CheckVectorApproxValues(copied, expected);
            }
        }
        auto global = TensorValue(99.0f, {1, 1}, device.get(), dtype);
        for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
        {
            device->reduce(base.deviceParams(), global.deviceParams(), op);
            device->synchronize();
            CHECK(global.to(&reference).to(DataType::kFloat32).data<float>()[0] == (op == ReduceOp::kSum ? 78 : 12));
            global = TensorValue(1.0f, {1, 1}, device.get(), dtype);
            device->reduce(base.deviceParams(), global.deviceParams(), op, {}, true);
            device->synchronize();
            CHECK(global.to(&reference).to(DataType::kFloat32).data<float>()[0] == (op == ReduceOp::kSum ? 79 : 12));
        }
        auto singleton = TensorValue({2.0f, 5.0f}, {2, 1}, device.get(), dtype);
        auto dst = TensorValue(99.0f, {2, 1}, device.get(), dtype);
        device->reduce(singleton.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis);
        device->synchronize();
        CheckVectorApproxValues(dst, singleton);
    }
}

TEST_CASE("DeviceMetal native reduction global stages use separate input and output buffers")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;

    // More than 1024 squared elements forces a multi-threadgroup intermediate sum/max stage.
    // Debug builds reject overlapping stage bindings even when GPU scheduling hides corruption.
    std::vector<int32_t> values(2 * 1024 * 1024, 0);
    values[1] = 7;
    auto src = TensorValue(values.data(), values.size(), DataType::kInt32,
                           {values.size()}, device.get(), DataType::kInt32);
    auto dst = TensorValue(99, {1}, device.get(), DataType::kInt32);

    SUBCASE("sum")
    {
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum);
        device->synchronize();
        CHECK(dst.data<int32_t>()[0] == 7);
    }
    SUBCASE("max")
    {
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax);
        device->synchronize();
        CHECK(dst.data<int32_t>()[0] == 7);
    }
    SUBCASE("argmax")
    {
        device->argmax(src.deviceParams(), dst.deviceParams());
        device->synchronize();
        CHECK(dst.data<int32_t>()[0] == 1);
    }
}

TEST_CASE("DeviceMetal native reduction global stage buffers preserve singleton and staged results across dtypes")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    DeviceCPU reference;
    for (size_t dtypeIndex = 1; dtypeIndex < DataTypeCount; ++dtypeIndex)
    {
        const auto dtype = static_cast<DataType>(dtypeIndex);
        for (size_t count : {size_t{1}, size_t{1024}, size_t{2048}})
        {
            CAPTURE(dtypeIndex);
            CAPTURE(count);
            std::vector<float> values(count, 0.0f);
            values[count / 2] = 7.0f;
            auto src = TensorValue(values.data(), values.size(), DataType::kFloat32, {count}, device.get(), dtype);
            auto sum = TensorValue(99.0f, {1}, device.get(), dtype);
            auto max = TensorValue(99.0f, {1}, device.get(), dtype);
            auto index = TensorValue(99, {1}, device.get(), DataType::kInt32);
            device->reduce(src.deviceParams(), sum.deviceParams(), ReduceOp::kSum);
            device->reduce(src.deviceParams(), max.deviceParams(), ReduceOp::kMax);
            device->argmax(src.deviceParams(), index.deviceParams());
            device->synchronize();
            CHECK(sum.to(&reference).to(DataType::kFloat32).data<float>()[0] == 7.0f);
            CHECK(max.to(&reference).to(DataType::kFloat32).data<float>()[0] == 7.0f);
            CHECK(index.data<int32_t>()[0] == static_cast<int32_t>(count / 2));
        }
    }
}

TEST_CASE("DeviceMetal native reduction global stage buffers survive queued consumers and temporary retirement")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    constexpr size_t count = 2048;
    std::vector<float> values(count, 0.0f);
    values[17] = 3.0f;
    values.back() = 2.0f;
    auto left = TensorValue(values.data(), values.size(), DataType::kFloat32, {count}, device.get());
    values[17] = 2.0f;
    values.back() = 3.0f;
    auto right = TensorValue(values.data(), values.size(), DataType::kFloat32, {count}, device.get());
    auto sum = TensorValue(99.0f, {1}, device.get());
    auto max = TensorValue(99.0f, {1}, device.get());
    auto index = TensorValue(99, {1}, device.get(), DataType::kInt32);
    std::vector<TensorValue> sums;
    std::vector<TensorValue> maxima;
    std::vector<TensorValue> indices;
    for (size_t i = 0; i < 96; ++i)
    {
        auto produced = (i % 2 == 0 ? left : right) + TensorValue(static_cast<float>(i), {}, device.get());
        device->reduce(produced.deviceParams(), sum.deviceParams(), ReduceOp::kSum);
        sums.push_back(sum + sum);
        device->reduce(produced.deviceParams(), max.deviceParams(), ReduceOp::kMax);
        maxima.push_back(max + max);
        device->argmax(produced.deviceParams(), index.deviceParams());
        indices.push_back(index + index);
    }
    device->synchronize();
    for (size_t i = 0; i < sums.size(); ++i)
    {
        CAPTURE(i);
        CHECK(sums[i].data<float>()[0] == 2 * (5 + count * i));
        CHECK(maxima[i].data<float>()[0] == 2 * (3 + i));
        CHECK(indices[i].data<int32_t>()[0] == 2 * (i % 2 == 0 ? 17 : count - 1));
    }
}

TEST_CASE("DeviceMetal native reduction logical argmax coordinates")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    const size_t axis[] = {1};
    for (size_t dtypeIndex = 1; dtypeIndex < DataTypeCount; ++dtypeIndex)
    {
        CAPTURE(dtypeIndex);
        const auto dtype = static_cast<DataType>(dtypeIndex);
        auto src = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get(), dtype).transpose(0, 1);
        auto dst = TensorValue(99, {3, 1}, device.get(), DataType::kInt32);
        device->argmax(src.deviceParams(), dst.deviceParams(), axis);
        device->synchronize();
        CHECK(dst.data<int32_t>()[0] == 1);
        CHECK(dst.data<int32_t>()[1] == 0);
        CHECK(dst.data<int32_t>()[2] == 1);
        argmaxAxis(*device, src.deviceParams(), dst.deviceParams(), 1);
        device->synchronize();
        CHECK(dst.data<int32_t>()[0] == 1);
        CHECK(dst.data<int32_t>()[1] == 0);
        CHECK(dst.data<int32_t>()[2] == 1);
        auto global = TensorValue(99, {1, 1}, device.get(), DataType::kInt32);
        device->argmax(src.deviceParams(), global.deviceParams());
        device->synchronize();
        CHECK(global.data<int32_t>()[0] == 2);
        auto broadcast = TensorValue({1.0f, 8.0f, 3.0f}, {3, 1}, device.get(), dtype).broadcastTo({3, 2});
        device->argmax(broadcast.deviceParams(), dst.deviceParams(), axis);
        device->synchronize();
        for (size_t i = 0; i < 3; ++i) CHECK(dst.data<int32_t>()[i] == 0);
        auto sums = TensorValue(99.0f, {3, 1}, device.get(), dtype);
        device->reduce(broadcast.deviceParams(), sums.deviceParams(), ReduceOp::kSum, axis);
        device->synchronize();
        CHECK(sums.dataType() == dtype);
        DeviceCPU reference;
        CheckVectorApproxValues(sums, TensorValue({2.0f, 16.0f, 6.0f}, {3, 1}, &reference, dtype));
    }
}

#if defined(__APPLE__) && defined(__arm64__)
TEST_CASE("DeviceMetal native reduction stays deferred and falls back with explicit metadata")
{
    class InspectableDevice : public aix::metal::DeviceMetal
    {
    public:
        void synchronize() override { ++synchronizations; DeviceMetal::synchronize(); }
        size_t synchronizations{0};
    } device;
    DeviceCPU reference;
    const size_t axis[] = {1};
    auto src = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &device).transpose(0, 1);
    auto dst = TensorValue(99, {3, 1}, &device, DataType::kInt32);
    device.argmax(src.deviceParams(), dst.deviceParams(), axis);
    auto broadcast = TensorValue(5.0f, {3, 1}, &device).broadcastTo({3, 2});
    device.argmax(broadcast.deviceParams(), dst.deviceParams(), axis);
    const size_t multiple[] = {0, 1};
    auto allDst = dst.deviceParams();
    allDst.shape = {1, 1};
    allDst.strides = {1, 1};
    allDst.size = 1;
    CHECK_THROWS_AS(device.argmax(src.deviceParams(), allDst, multiple), std::invalid_argument);
    CHECK(device.synchronizations == 0);
    device.synchronize();
    for (size_t i = 0; i < 3; ++i) CHECK(dst.data<int32_t>()[i] == 0);

    // Non-device destinations must complete pending GPU producers before native CPU fallback.
    auto hostIndices = TensorValue(99, {3, 1}, &reference, DataType::kInt32);
    auto produced = src + TensorValue(1.0f, {}, &device);
    device.argmax(produced.deviceParams(), hostIndices.deviceParams(), axis);
    CHECK(device.synchronizations == 2);
    CHECK(hostIndices.data<int32_t>()[0] == 1);
    CHECK(hostIndices.data<int32_t>()[1] == 0);
    CHECK(hostIndices.data<int32_t>()[2] == 1);

    const double values[] = {1, 8, 3, 4, 5, 6};
    auto doubles = TensorValue(values, 6, DataType::kFloat64, {2, 3}, &reference).transpose(0, 1);
    device.argmax(doubles.deviceParams(), hostIndices.deviceParams(), axis);
    CHECK(device.synchronizations == 3);
    CHECK(hostIndices.data<int32_t>()[0] == 1);
    CHECK(hostIndices.data<int32_t>()[1] == 0);
    CHECK(hostIndices.data<int32_t>()[2] == 1);

    Shape shape(17, 1);
    shape.front() = shape.back() = 2;
    auto highRank = TensorValue({1.0f, 2.0f, 3.0f, 4.0f}, shape, &device);
    shape.front() = shape.back() = 1;
    auto total = TensorValue(99.0f, shape, &device);
    const size_t mixed[] = {16, 0};
    device.reduce(highRank.deviceParams(), total.deviceParams(), ReduceOp::kSum, mixed);
    CHECK(device.synchronizations == 4);
    CHECK(total.data<float>()[0] == 10);
    // Axis argmax has no fixed-rank coordinate array and need not take the shape-kernel fallback.
    shape.front() = 2;
    auto highIndices = TensorValue(99, shape, &device, DataType::kInt32);
    auto transposed = highRank.transpose(0, 16);
    const size_t last[] = {16};
    device.argmax(transposed.deviceParams(), highIndices.deviceParams(), last);
    CHECK(device.synchronizations == 4);
    device.synchronize();
    CHECK(highIndices.data<int32_t>()[0] == 1);
    CHECK(highIndices.data<int32_t>()[1] == 1);
}

TEST_CASE("DeviceMetal Tensor shape lowering uses canonical native accumulation")
{
    class InspectableDevice : public aix::metal::DeviceMetal
    {
    public:
        void synchronize() override { ++synchronizations; DeviceMetal::synchronize(); }
        size_t synchronizations{0};
    } device;
    auto input = TensorValue({2048.0f, 2048.0f, 2048.0f, 2048.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              -2048.0f, -2048.0f, -2048.0f, -2048.0f}, {3, 4}, &device, DataType::kFloat16);
    auto produced = input + TensorValue(0.0f, {}, &device, DataType::kFloat16);
    auto lowerRank = produced.sum({0}, true).reshape({4});
    auto canonical = TensorValue(0.0f, {1, 4}, &device, DataType::kFloat16);
    const size_t axis[] = {0};
    device.reduce(produced.deviceParams(), canonical.deviceParams(), ReduceOp::kSum, axis, true);
    CHECK(device.synchronizations == 0);
    device.synchronize();
    for (size_t i = 0; i < canonical.size(); ++i)
    {
        CHECK(static_cast<float>(lowerRank.data<float16_t>()[i]) == 1.0f);
        CHECK(lowerRank.data<float16_t>()[i] == canonical.data<float16_t>()[i]);
    }

    auto highRankSource = input.reshape(Shape{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4});
    SIndex axes(16);
    std::iota(axes.begin(), axes.end(), 0);
    auto highRankResult = highRankSource.sum(axes, true).reshape({4});
    CHECK(device.synchronizations == 2);
    for (size_t i = 0; i < highRankResult.size(); ++i) CHECK(static_cast<float>(highRankResult.data<float16_t>()[i]) == 0.0f);
}
#endif

TEST_CASE("DeviceMetal native reduction queued combine consumers and temporary retirement")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    auto dst = TensorValue(3.0f, {2, 1}, device.get());
    std::vector<TensorValue> consumers;
    const size_t axis[] = {1};
    for (size_t i = 0; i < 96; ++i)
    {
        auto src = TensorValue(1.0f, {2, 8}, device.get());
        auto produced = (src + src) * src;
        device->reduce(produced.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis, true);
        consumers.push_back(dst + dst);
        dst += TensorValue(1.0f, {2, 1}, device.get());
    }
    device->synchronize();
    for (size_t i = 0; i < consumers.size(); ++i)
    {
        for (size_t j = 0; j < 2; ++j) CHECK(consumers[i].data<float>()[j] == 2 * (19 + 17 * i));
    }
    for (size_t i = 0; i < 2; ++i) CHECK(dst.data<float>()[i] == 1635);
}

TEST_CASE("DeviceMetal native reduction queues overwrite and accumulation with pending consumers")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    const size_t axis[] = {1};
    for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
    {
        CAPTURE(static_cast<size_t>(op));
        auto dst = TensorValue(3.0f, {2, 1}, device.get());
        std::vector<TensorValue> before;
        std::vector<TensorValue> after;
        for (size_t i = 0; i < 96; ++i)
        {
            auto src = TensorValue(1.0f, {2, 8}, device.get());
            auto produced = (src + src) * src;
            before.push_back(dst + dst);
            device->reduce(produced.deviceParams(), dst.deviceParams(), op, axis);
            after.push_back(dst + dst);
            device->reduce(produced.deviceParams(), dst.deviceParams(), op, axis, true);
            dst += TensorValue(1.0f, {2, 1}, device.get());
        }
        device->synchronize();
        const float finalValue = op == ReduceOp::kSum ? 33 : 3;
        for (size_t i = 0; i < before.size(); ++i)
        {
            for (size_t j = 0; j < dst.size(); ++j)
            {
                CHECK(before[i].data<float>()[j] == (i == 0 ? 6 : 2 * finalValue));
                CHECK(after[i].data<float>()[j] == (op == ReduceOp::kSum ? 32 : 4));
            }
        }
        for (size_t i = 0; i < dst.size(); ++i) CHECK(dst.data<float>()[i] == finalValue);
    }
}

TEST_CASE("DeviceMetal native reduction preserves accumulator and destination combination order")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    DeviceCPU reference;
    const size_t axis[] = {1};
    for (auto dtype : {DataType::kFloat16, DataType::kBFloat16})
    {
        const float large = dtype == DataType::kFloat16 ? 2048.0f : 256.0f;
        auto src = TensorValue({large, -large, 1.0f, 1.0f}, {1, 4}, device.get(), dtype);
        auto global = TensorValue(99.0f, {1, 1}, device.get(), dtype);
        auto shaped = TensorValue(99.0f, {1, 1}, device.get(), dtype);
        device->reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kSum);
        device->reduce(src.deviceParams(), shaped.deviceParams(), ReduceOp::kSum, axis);
        device->synchronize();
        CHECK(global.to(&reference).to(DataType::kFloat32).data<float>()[0] == 1);
        CHECK(shaped.to(&reference).to(DataType::kFloat32).data<float>()[0] == 2);
    }

    const float large = 0x1p24f;
    auto inner = TensorValue({large, 1.0f, -large}, {1, 3}, device.get());
    auto outer = TensorValue({large, 1.0f, -large}, {3, 1}, device.get());
    auto strided = outer.broadcastTo({3, 2});
    auto vector = TensorValue({large, 1.0f, -large, 1.0f}, {1, 4}, device.get());
    auto innerDst = TensorValue(1.0f, {1, 1}, device.get());
    auto outerDst = TensorValue(1.0f, {1, 1}, device.get());
    auto stridedDst = TensorValue(1.0f, {1, 2}, device.get());
    auto vectorDst = TensorValue(1.0f, {1, 1}, device.get());
    const size_t first[] = {0};
    device->reduce(inner.deviceParams(), innerDst.deviceParams(), ReduceOp::kSum, axis, true);
    device->reduce(outer.deviceParams(), outerDst.deviceParams(), ReduceOp::kSum, first, true);
    device->reduce(strided.deviceParams(), stridedDst.deviceParams(), ReduceOp::kSum, first, true);
    device->reduce(vector.deviceParams(), vectorDst.deviceParams(), ReduceOp::kSum, axis, true);
    device->synchronize();
    CHECK(innerDst.data<float>()[0] == 2);
    CHECK(outerDst.data<float>()[0] == 0);
    CHECK(stridedDst.data<float>()[0] == 0);
    CHECK(stridedDst.data<float>()[1] == 0);
    CHECK(vectorDst.data<float>()[0] == 2);

    std::vector<float> values(25, 99.0f);
    for (size_t i = 1; i < values.size(); ++i) values[i] = static_cast<float>((i - 1) % 4 + 1);
    auto storage = TensorValue(values.data(), values.size(), DataType::kFloat32, {25}, device.get());
    auto offset = storage.deviceParams();
    offset.offset = 1;
    offset.shape = {3, 8};
    offset.strides = {8, 1};
    offset.size = 24;
    auto offsetDst = TensorValue(99.0f, {3, 1}, device.get());
    for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
    {
        device->reduce(offset, offsetDst.deviceParams(), op, axis);
        device->synchronize();
        for (size_t i = 0; i < 3; ++i) CHECK(offsetDst.data<float>()[i] == (op == ReduceOp::kSum ? 20 : 4));
    }
}

TEST_CASE("DeviceMetal native reduction propagates max special values and retains argmax staged ties")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    const size_t axis[] = {1};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    auto src = TensorValue({nan, 3.0f, 2.0f, -0.0f, 0.0f, -0.0f}, {2, 3}, device.get());
    auto dst = TensorValue(-10.0f, {2, 1}, device.get());
    device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, axis, true);
    auto infinities = TensorValue({-infinity, -infinity}, {1, 2}, device.get());
    auto global = TensorValue(99.0f, {1, 1}, device.get());
    auto shaped = TensorValue(99.0f, {1, 1}, device.get());
    device->reduce(infinities.deviceParams(), global.deviceParams(), ReduceOp::kMax);
    device->reduce(infinities.deviceParams(), shaped.deviceParams(), ReduceOp::kMax, axis);
    device->synchronize();
    CHECK(std::isnan(dst.data<float>()[0]));
    CHECK(dst.data<float>()[1] == 0);
    CHECK_FALSE(std::signbit(dst.data<float>()[1]));
    CHECK(global.data<float>()[0] == -infinity);
    CHECK(shaped.data<float>()[0] == -infinity);

    {
        auto copied = TensorValue(99.0f, src.shape(), device.get());
        device->contiguous(src.deviceParams(), copied.deviceParams());
        device->synchronize();
        for (size_t i = 0; i < src.size(); ++i)
        {
            CHECK(std::bit_cast<uint32_t>(copied.data<float>()[i]) == std::bit_cast<uint32_t>(src.data<float>()[i]));
        }
    }

    std::vector<float> values(2051, 1.0f);
    values[1023] = values[2049] = 8.0f;
    auto tied = TensorValue(values.data(), values.size(), DataType::kFloat32, {1, values.size()}, device.get());
    auto index = TensorValue(99, {1, 1}, device.get(), DataType::kInt32);
    device->argmax(tied.deviceParams(), index.deviceParams());
    device->synchronize();
    CHECK(index.data<int32_t>()[0] == 1023);
    device->argmax(tied.deviceParams(), index.deviceParams(), axis);
    device->synchronize();
    CHECK(index.data<int32_t>()[0] == 1023);
}

TEST_CASE("DeviceCPU native reduction explicit mixed equal and singleton axes")
{
    DeviceCPU device;
    auto src = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2}, &device);
    const size_t mixed[] = {2, 0};
    auto dst = TensorValue(100.0f, {1, 2, 1}, &device);
    device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, mixed);
    CheckVectorApproxValues(dst, TensorValue({14.0f, 22.0f}, {1, 2, 1}, &device));
    device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, mixed, true);
    CheckVectorApproxValues(dst, TensorValue({28.0f, 44.0f}, {1, 2, 1}, &device));
    device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, mixed);
    CheckVectorApproxValues(dst, TensorValue({6.0f, 8.0f}, {1, 2, 1}, &device));

    const size_t first[] = {0};
    const size_t middle[] = {1};
    auto outer = TensorValue(100.0f, {1, 2, 2}, &device);
    auto inner = TensorValue(100.0f, {2, 1, 2}, &device);
    device.reduce(src.deviceParams(), outer.deviceParams(), ReduceOp::kSum, first);
    device.reduce(src.deviceParams(), inner.deviceParams(), ReduceOp::kSum, middle);
    CheckVectorApproxValues(outer, TensorValue({6.0f, 8.0f, 10.0f, 12.0f}, {1, 2, 2}, &device));
    CheckVectorApproxValues(inner, TensorValue({4.0f, 6.0f, 12.0f, 14.0f}, {2, 1, 2}, &device));

    auto singleton = TensorValue({1.0f, 2.0f, 3.0f, 4.0f}, {2, 1, 2}, &device);
    for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
    {
        device.reduce(singleton.deviceParams(), inner.deviceParams(), op, middle);
        CheckVectorApproxValues(inner, singleton);
    }
}

TEST_CASE("DeviceCPU native reduction all dtypes overwrite combine and no axes")
{
    DeviceCPU device;
    const size_t axis[] = {1};
    for (size_t i = 0; i < DataTypeCount; ++i)
    {
        CAPTURE(i);
        const auto dtype = static_cast<DataType>(i);
        auto src = TensorValue({1.0f, 7.0f, 7.0f, 4.0f, 7.0f, 2.0f}, {2, 3}, &device, dtype);
        auto dst = TensorValue(99.0f, {2, 1}, &device, dtype);
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis);
        CheckVectorApproxValues(dst, TensorValue({15.0f, 13.0f}, {2, 1}, &device, dtype));
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis, true);
        CheckVectorApproxValues(dst, TensorValue({30.0f, 26.0f}, {2, 1}, &device, dtype));

        dst = TensorValue({3.0f, 9.0f}, {2, 1}, &device, dtype);
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, axis, true);
        CheckVectorApproxValues(dst, TensorValue({7.0f, 9.0f}, {2, 1}, &device, dtype));
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, axis);
        CheckVectorApproxValues(dst, TensorValue({7.0f, 7.0f}, {2, 1}, &device, dtype));

        auto global = TensorValue(99.0f, {1, 1}, &device, dtype);
        device.reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kSum);
        CHECK(global.to(DataType::kFloat64).data<double>()[0] == 28);
        device.reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kSum, {}, true);
        CHECK(global.to(DataType::kFloat64).data<double>()[0] == 56);
        device.reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kMax);
        CHECK(global.to(DataType::kFloat64).data<double>()[0] == 7);
        device.reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kMax, {}, true);
        CHECK(global.to(DataType::kFloat64).data<double>()[0] == 7);

        auto indices = TensorValue(99, {2, 1}, &device, DataType::kInt32);
        device.argmax(src.deviceParams(), indices.deviceParams(), axis);
        CHECK(indices.data<int32_t>()[0] == 1);
        CHECK(indices.data<int32_t>()[1] == 1);
        auto index = TensorValue(99, {1, 1}, &device, DataType::kInt32);
        device.argmax(src.deviceParams(), index.deviceParams());
        CHECK(index.data<int32_t>()[0] == 1);

        auto copied = TensorValue(99.0f, src.shape(), &device, dtype);
        for (auto op : {ReduceOp::kSum, ReduceOp::kMax})
        {
            device.contiguous(src.deviceParams(), copied.deviceParams());
            CheckVectorApproxValues(copied, src);
            if (op == ReduceOp::kSum) device.add(src.deviceParams(), copied.deviceParams(), copied.deviceParams());
            CheckVectorApproxValues(copied, op == ReduceOp::kSum ?
                TensorValue({2.0f, 14.0f, 14.0f, 8.0f, 14.0f, 4.0f}, src.shape(), &device, dtype) : src);
        }
    }
}

TEST_CASE("DeviceCPU native reduction retains input precision and destination first order")
{
    DeviceCPU device;
    const size_t axis[] = {0};
    const std::pair<DataType, double> cases[] = {
        {DataType::kFloat16, 0x1p11}, {DataType::kBFloat16, 0x1p8},
        {DataType::kFloat32, 0x1p24}, {DataType::kFloat64, 0x1p53}
    };
    for (const auto& [dtype, large] : cases)
    {
        CAPTURE(static_cast<size_t>(dtype));
        const double values[] = {large, large, 1, 1, -large, -large};
        auto src = TensorValue(values, 6, DataType::kFloat64, {3, 2}, &device, dtype);
        for (bool mode : {false, true})
        {
            auto dst = TensorValue(1.0f, {1, 2}, &device, dtype);
            device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis, mode);
            auto actual = dst.to(DataType::kFloat64);
            CHECK(actual.data<double>()[0] == 0);
            CHECK(actual.data<double>()[1] == 0);
            auto global = TensorValue(1.0f, {1, 1}, &device, dtype);
            device.reduce(src.deviceParams(), global.deviceParams(), ReduceOp::kSum, {}, mode);
            CHECK(global.to(DataType::kFloat64).data<double>()[0] == 0);
        }
    }
}

TEST_CASE("DeviceCPU native reduction layout and logical argmax coordinates")
{
    DeviceCPU device;
    const size_t axis[] = {1};
    auto src = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &device).transpose(0, 1);
    auto indices = TensorValue(99, {3, 1}, &device, DataType::kInt32);
    device.argmax(src.deviceParams(), indices.deviceParams(), axis);
    CHECK(indices.data<int32_t>()[0] == 1);
    CHECK(indices.data<int32_t>()[1] == 0);
    CHECK(indices.data<int32_t>()[2] == 1);
    argmaxAxis(device, src.deviceParams(), indices.deviceParams(), 1);
    CHECK(indices.data<int32_t>()[0] == 1);
    CHECK(indices.data<int32_t>()[1] == 0);
    CHECK(indices.data<int32_t>()[2] == 1);
    auto global = TensorValue(99, {1, 1}, &device, DataType::kInt32);
    device.argmax(src.deviceParams(), global.deviceParams());
    CHECK(global.data<int32_t>()[0] == 2);

    auto broadcast = TensorValue({1.0f, 8.0f, 3.0f}, {3, 1}, &device).broadcastTo({3, 2});
    device.argmax(broadcast.deviceParams(), indices.deviceParams(), axis);
    for (size_t i = 0; i < 3; ++i) CHECK(indices.data<int32_t>()[i] == 0);
    device.argmax(broadcast.deviceParams(), global.deviceParams());
    CHECK(global.data<int32_t>()[0] == 2);
    auto sums = TensorValue(99.0f, {3, 1}, &device);
    device.reduce(broadcast.deviceParams(), sums.deviceParams(), ReduceOp::kSum, axis);
    CheckVectorApproxValues(sums, TensorValue({2.0f, 16.0f, 6.0f}, {3, 1}, &device));

    auto base = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                             9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &device);
    for (auto view : {base.slice(0, 1, 3), base.slice(0, 1, 3).slice(1, 0, 4, 2)})
    {
        auto dst = TensorValue(99.0f, {2, 1}, &device);
        device.reduce(view.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis);
        const bool stepped = view.shape()[1] == 2;
        CheckVectorApproxValues(dst, stepped ? TensorValue({12.0f, 20.0f}, {2, 1}, &device) :
                                               TensorValue({26.0f, 42.0f}, {2, 1}, &device));
        auto copied = TensorValue(99.0f, view.shape(), &device);
        device.contiguous(view.deviceParams(), copied.deviceParams());
        CheckVectorApproxValues(copied, view.contiguous());
    }
}

TEST_CASE("DeviceCPU native reduction global axis and noaxis special value seeds")
{
    DeviceCPU device;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const size_t axis[] = {0};
    for (auto values : {std::array{nan, 3.0f}, std::array{-infinity, -infinity}, std::array{-0.0f, -0.0f}})
    {
        auto src = TensorValue(values.data(), values.size(), DataType::kFloat32, {2}, &device);
        auto dst = TensorValue(99.0f, {1}, &device);
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax);
        CHECK(std::bit_cast<uint32_t>(dst.data<float>()[0]) == std::bit_cast<uint32_t>(values[0]));
        device.reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, axis);
        const float axisExpected = values[0];
        CHECK(std::bit_cast<uint32_t>(dst.data<float>()[0]) == std::bit_cast<uint32_t>(axisExpected));
        auto index = TensorValue(99, {1}, &device, DataType::kInt32);
        for (bool global : {true, false})
        {
            device.argmax(src.deviceParams(), index.deviceParams(), global ? std::span<const size_t>{} : axis);
            CHECK(index.data<int32_t>()[0] == 0);
        }
        {
            auto copied = TensorValue(99.0f, src.shape(), &device);
            device.contiguous(src.deviceParams(), copied.deviceParams());
            for (size_t i = 0; i < src.size(); ++i)
            {
                CHECK(std::bit_cast<uint32_t>(copied.data<float>()[i]) == std::bit_cast<uint32_t>(values[i]));
            }
        }
    }
}

TEST_CASE("Device reduction max special values agree across backends and routes")
{
    DeviceCPU reference;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    struct MaxCase
    {
        std::array<float, 3> values;
        float expected;
    };
    const MaxCase cases[] =
    {
        {{nan, 3, 2}, nan}, {{3, nan, 2}, nan}, {{3, 2, nan}, nan}, {{nan, nan, nan}, nan},
        {{-infinity, -infinity, -infinity}, -infinity}, {{-infinity, infinity, 3}, infinity},
        {{-0.0f, -0.0f, -0.0f}, -0.0f}, {{-0.0f, 0.0f, -0.0f}, 0.0f}, {{0.0f, -0.0f, -0.0f}, 0.0f}
    };
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(type));
        for (auto dtype : {DataType::kFloat64, DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
        {
            if (type == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(static_cast<size_t>(dtype));
            auto check = [&](const TensorValue& result, float expected)
            {
                device->synchronize();
                auto actual = result.to(&reference).to(DataType::kFloat64);
                for (size_t i = 0; i < actual.size(); ++i)
                {
                    const double value = actual.data<double>()[i];
                    if (std::isnan(expected)) CHECK(std::isnan(value));
                    else
                    {
                        CHECK(value == expected);
                        if (expected == 0) CHECK(std::signbit(value) == std::signbit(expected));
                    }
                }
            };
            for (const auto& [values, expected] : cases)
            {
                for (size_t count : {size_t{3}, size_t{8}, size_t{2051}})
                {
                    CAPTURE(count);
                    std::vector<float> repeated(count);
                    for (size_t i = 0; i < count; ++i) repeated[i] = values[i % values.size()];
                    auto input = TensorValue(repeated.data(), repeated.size(), DataType::kFloat32, {1, count}, device.get()).to(dtype);
                    check(input.max(), expected);              // Global, including multiple stages.
                    check(input.max(1, true), expected);       // Scalar/vector inner routes.
                    auto transposed = input.transpose(0, 1).broadcastTo({count, 2});
                    check(transposed.max(0, true), expected);  // Strided/broadcast route.
                    auto outer = transposed.contiguous();
                    check(outer.max(0, true), expected);       // Outer route.
                    auto dst = TensorValue(99, {1, 1}, device.get(), dtype);
                    const Index axes{1, 0};
                    device->reduce(input.deviceParams(), dst.deviceParams(), ReduceOp::kMax, axes);
                    check(dst, expected);                     // Explicit multi-axis overwrite.
                    for (bool global : {true, false})
                    {
                        for (float seed : {-infinity, infinity, nan, -0.0f, 0.0f})
                        {
                            dst.fill(seed);
                            device->reduce(input.deviceParams(), dst.deviceParams(),
                                           ReduceOp::kMax,
                                           global ? std::span<const size_t>{} : std::span<const size_t>{axes}, true);
                            float combined = expected;
                            if (std::isnan(seed) || std::isnan(expected)) combined = nan;
                            else if (seed == infinity || (seed == 0 && expected < 0)) combined = seed;
                            else if (seed == 0 && expected == 0 && !std::signbit(seed)) combined = seed;
                            check(dst, combined);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Device reduction API executes backend layout and scalar routes")
{
    DeviceCPU reference;
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        auto src = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);
        auto dst = TensorValue(100.0f, {3, 1}, device.get());
        const size_t axis[] = {1};
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis);
        device->synchronize();
        CheckVectorApproxValues(dst, TensorValue({5.0f, 7.0f, 9.0f}, {3, 1}, &reference));
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, axis, true);
        device->synchronize();
        CheckVectorApproxValues(dst, TensorValue({10.0f, 14.0f, 18.0f}, {3, 1}, &reference));

        auto scalar = TensorValue(-0.0f, {}, device.get());
        auto scalarDst = TensorValue(100.0f, {}, device.get());
        device->reduce(scalar.deviceParams(), scalarDst.deviceParams(), ReduceOp::kMax);
        device->synchronize();
        CHECK(std::signbit(scalarDst.to(&reference).item<float>()));
        auto index = TensorValue(99, {}, device.get(), DataType::kInt32);
        device->argmax(scalar.deviceParams(), index.deviceParams());
        device->synchronize();
        CHECK(index.to(&reference).item<int32_t>() == 0);
    }
}

TEST_CASE("Device reductions overwrite scalars and combine destinations")
{
    aix::DeviceCPU refDevice;
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));

        for (size_t i = 0; i < aix::DataTypeCount; ++i)
        {
            const auto dtype = static_cast<DataType>(i);
            if (deviceType == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(i);
            auto src = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get(), dtype);
            auto sum = TensorValue(100.0f, {}, device.get(), dtype);
            auto max = TensorValue(100.0f, {}, device.get(), dtype);
            auto added = TensorValue({10.0f, 20.0f, 30.0f}, {3}, device.get(), dtype);
            auto maxima = TensorValue({10.0f, 0.0f, 5.0f}, {3}, device.get(), dtype);

            for (size_t call = 1; call <= 2; ++call)
            {
                CAPTURE(call);
                reduceAll(*device, src.deviceParams(), sum.deviceParams(), ReduceOp::kSum);
                reduceAll(*device, src.deviceParams(), max.deviceParams(), ReduceOp::kMax);
                reduceAxes(*device, src.deviceParams(), added.deviceParams(), {0}, ReduceOp::kSum);
                reduceAxes(*device, src.deviceParams(), maxima.deviceParams(), {0}, ReduceOp::kMax);
                device->synchronize();

                CHECK(sum.to(&refDevice).to(DataType::kFloat64).item<double>() == 21.0);
                CHECK(max.to(&refDevice).to(DataType::kFloat64).item<double>() == 6.0);
                auto actualAdded = added.to(&refDevice).to(DataType::kFloat64);
                auto actualMaxima = maxima.to(&refDevice).to(DataType::kFloat64);
                const double contributions[] = {5.0, 7.0, 9.0};
                const double expectedMaxima[] = {10.0, 5.0, 6.0};
                for (size_t j = 0; j < 3; ++j)
                {
                    CHECK(actualAdded.data<double>()[j] == 10.0 * (j + 1) + call * contributions[j]);
                    CHECK(actualMaxima.data<double>()[j] == expectedMaxima[j]);
                }
            }
        }
    }
}


TEST_CASE("Device reduction shape lowering preserves source layouts")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));
        auto check = [&](const TensorValue& src, const Index& axes,
                          std::initializer_list<float> sums, std::initializer_list<float> maxima)
        {
            auto shape = src.shape();
            for (auto axis : axes) shape[axis] = 1;
            auto added = TensorValue(1.0f, shape, device.get());
            auto max = TensorValue(0.0f, shape, device.get());
            reduceAxes(*device, src.deviceParams(), added.deviceParams(), axes, ReduceOp::kSum);
            reduceAxes(*device, src.deviceParams(), max.deviceParams(), axes, ReduceOp::kMax);
            device->synchronize();
            REQUIRE(added.size() == sums.size());
            REQUIRE(max.size() == maxima.size());
            for (size_t i = 0; i < added.size(); ++i)
            {
                CHECK(added.data<float>()[i] == sums.begin()[i] + 1.0f);
                CHECK(max.data<float>()[i] == maxima.begin()[i]);
            }
        };

        auto rows = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}, {3, 8}, device.get());
        check(rows, {1}, {36, 100, 164}, {8, 16, 24});       // Vector inner.
        auto offsetRows = rows.slice(0, 1, 3);
        CHECK(offsetRows.isContiguous());
        CHECK(offsetRows.storageOffset() == 8);
        check(offsetRows, {1}, {100, 164}, {16, 24});
        auto unaligned = rows.reshape({24}).slice(0, 1, 8);
        CHECK(unaligned.isContiguous());
        CHECK(unaligned.storageOffset() == 1);
        check(unaligned, {0}, {35}, {8});
        auto columns = rows.slice(1, 1, 8);
        CHECK_FALSE(columns.isContiguous());
        CHECK(columns.storageOffset() == 1);
        check(columns, {1}, {35, 91, 147}, {8, 16, 24});
        check(rows.slice(1, 0, 8, 2), {1}, {16, 48, 80}, {7, 15, 23});
        check(rows.reshape({2, 3, 4}), {1}, {15, 18, 21, 24, 51, 54, 57, 60}, {9, 10, 11, 12, 21, 22, 23, 24});

        auto small = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get());
        check(small, {0}, {5, 7, 9}, {4, 5, 6});                // Outer.
        check(small, {}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6});
        check(small.transpose(0, 1), {1}, {5, 7, 9}, {4, 5, 6});
        auto broadcast = small.slice(0, 0, 1).broadcastTo({2, 3});
        CHECK(broadcast.strides()[0] == 0);
        check(broadcast, {0}, {2, 4, 6}, {1, 2, 3});

        // Leave the producer pending until the reductions consume it.
        auto produced = small + 1.0f;
        check(produced, {1}, {9, 18}, {4, 7});               // Scalar inner, odd extent.
    }
}


TEST_CASE("DeviceMetal empty storage retains zero logical size with valid allocation lifetime")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    for (size_t i = 1; i < DataTypeCount; ++i)
    {
        const auto dtype = static_cast<DataType>(i);
        CAPTURE(i);
        auto empty = TensorValue(Shape{0, 3}, device.get(), dtype);
        CHECK(empty.size() == 0);
        CHECK(empty.shape() == Shape{0, 3});
        CHECK(empty.storage()->size() == 0);
        CHECK(empty.data() != nullptr);
        auto other = TensorValue(Shape{2, 0}, device.get(), dtype);
        CHECK(other.storage()->size() == 0);
        CHECK(other.data() != empty.data());
        auto sum = empty.sum();
        auto actual = sum.to(DataType::kFloat32);
        device->synchronize();
        CHECK(actual.item<float>() == 0);
    }
    auto bytes = device->allocate(0);
    REQUIRE(bytes != nullptr);
    device->deallocate(bytes);
    device->synchronize();
    device->emptyCache();
}

TEST_CASE("Device reduction empty axes are global and accumulation is opt in")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(deviceType);
        if (!device) continue;
        auto src = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get());
        auto dst = TensorValue(100.0f, {1, 1}, device.get());
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum);
        device->synchronize();
        CHECK(dst.data<float>()[0] == 21);
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, {}, true);
        device->synchronize();
        CHECK(dst.data<float>()[0] == 42);
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kSum, {});
        device->synchronize();
        CHECK(dst.data<float>()[0] == 21);
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax, {}, true);
        device->synchronize();
        CHECK(dst.data<float>()[0] == 21);
        device->reduce(src.deviceParams(), dst.deviceParams(), ReduceOp::kMax);
        device->synchronize();
        CHECK(dst.data<float>()[0] == 6);
        auto index = TensorValue(99, {1, 1}, device.get(), DataType::kInt32);
        device->argmax(src.deviceParams(), index.deviceParams(), {});
        device->synchronize();
        CHECK(index.data<int32_t>()[0] == 5);
    }
}

TEST_CASE("Device argmax and argmaxMask overwrite indices and complete masks")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));
        for (size_t i = 0; i < aix::DataTypeCount; ++i)
        {
            const auto dtype = static_cast<DataType>(i);
            if (deviceType == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(i);
            auto src = TensorValue({1.0f, 5.0f, 5.0f, 4.0f, 4.0f, 3.0f}, {2, 3}, device.get(), dtype);
            auto index = TensorValue(100.0f, {}, device.get(), DataType::kInt32);
            auto coordinates = TensorValue(100.0f, {2, 1}, device.get(), DataType::kInt32);
            auto mask = TensorValue(7.0f, {2, 3}, device.get(), DataType::kInt32);
            auto axisMask = TensorValue(7.0f, {2, 3}, device.get(), DataType::kInt32);
            for (size_t call = 0; call < 2; ++call)
            {
                argmaxAll(*device, src.deviceParams(), index.deviceParams());
                argmaxAxis(*device, src.deviceParams(), coordinates.deviceParams(), 1);
                device->argmaxMask(src.deviceParams(), mask.deviceParams());
                device->argmaxMask(src.deviceParams(), axisMask.deviceParams(), std::array<size_t, 1>{1});
                device->synchronize();
                CHECK(index.dataType() == DataType::kInt32);
                CHECK(index.item<int32_t>() == 1);
                CHECK(coordinates.data<int32_t>()[0] == 1);
                CHECK(coordinates.data<int32_t>()[1] == 0);
                for (size_t j = 0; j < src.size(); ++j)
                {
                    CHECK(mask.data<int32_t>()[j] == (j == 1 ? 1 : 0));
                    CHECK(axisMask.data<int32_t>()[j] == (j == 1 || j == 3 ? 1 : 0));
                }
            }
        }

        // Reusing masks with different winners must clear the previous winners too.
        auto first = TensorValue({9.0f, 1.0f, 2.0f, 8.0f, 3.0f, 4.0f}, {2, 3}, device.get());
        auto second = TensorValue({1.0f, 2.0f, 9.0f, 3.0f, 4.0f, 8.0f}, {2, 3}, device.get());
        auto reused = TensorValue(7, {2, 3}, device.get(), DataType::kInt32);
        const size_t axis[] = {1};
        device->argmaxMask(first.deviceParams(), reused.deviceParams(), axis);
        device->argmaxMask(second.deviceParams(), reused.deviceParams(), axis);
        device->synchronize();
        for (size_t i = 0; i < reused.size(); ++i)
            CHECK(reused.data<int32_t>()[i] == (i == 2 || i == 5 ? 1 : 0));
        device->argmaxMask(first.deviceParams(), reused.deviceParams());
        device->argmaxMask(second.deviceParams(), reused.deviceParams());
        device->synchronize();
        for (size_t i = 0; i < reused.size(); ++i)
            CHECK(reused.data<int32_t>()[i] == (i == 2 ? 1 : 0));
        const size_t multiple[] = {0, 1};
        CHECK_THROWS_AS(device->argmaxMask(second.deviceParams(), reused.deviceParams(), multiple), std::invalid_argument);

        // Global indices and masks use logical order, not physical storage order.
        auto transposed = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);
        auto invalidResult = TensorValue(0.0f, {3, 2}, device.get());
        CHECK_THROWS_AS(argmaxAll(*device, transposed.deviceParams(), invalidResult.deviceParams()), std::invalid_argument);
        CHECK_THROWS_AS(argmaxAxis(*device, transposed.deviceParams(), invalidResult.deviceParams(), 1), std::invalid_argument);
        CHECK_THROWS_AS(device->argmaxMask(transposed.deviceParams(), invalidResult.deviceParams()), std::invalid_argument);
        CHECK_THROWS_AS(device->argmaxMask(transposed.deviceParams(), invalidResult.deviceParams(), std::array<size_t, 1>{1}), std::invalid_argument);
        auto index = transposed.argmax();
        auto mask = transposed.argmaxIndices();
        device->synchronize();
        CHECK(index.item<int32_t>() == 2);
        for (size_t i = 0; i < mask.size(); ++i)
        {
            CHECK(mask.data<int32_t>()[i] == (i == 2 ? 1 : 0));
        }

        // Equal maxima straddle the maximum 1024-thread Metal threadgroup extent.
        std::vector<float> values(1025, 1.0f);
        values[31] = values[1024] = 5.0f;
        auto tied = TensorValue(values.data(), values.size(), DataType::kFloat32, {1025}, device.get());
        auto winner = tied.argmax();
        auto winners = tied.argmaxIndices();
        device->synchronize();
        CHECK(winner.item<int32_t>() == 31);
        for (size_t i = 0; i < winners.size(); ++i)
        {
            CHECK(winners.data<int32_t>()[i] == (i == 31 ? 1 : 0));
        }
    }
}


TEST_CASE("Device Float64 shape reductions and argmax use CPU fallback")
{
    aix::DeviceCPU refDevice;
    auto device = aix::createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    auto src = TensorValue({1.0, 5.0, 5.0, 4.0, 4.0, 3.0}, {2, 3}, &refDevice, DataType::kFloat64);
    auto sum = TensorValue(1.0f, {2, 1}, &refDevice, DataType::kFloat64);
    auto max = TensorValue(2.0f, {2, 1}, &refDevice, DataType::kFloat64);
    auto index = TensorValue(100.0f, {}, &refDevice, DataType::kInt32);
    auto coordinates = TensorValue(100.0f, {2, 1}, &refDevice, DataType::kInt32);
    auto mask = TensorValue(7.0f, {2, 3}, &refDevice, DataType::kInt32);
    auto axisMask = TensorValue(7.0f, {2, 3}, &refDevice, DataType::kInt32);
    reduceAxes(*device, src.deviceParams(), sum.deviceParams(), {1}, ReduceOp::kSum);
    reduceAxes(*device, src.deviceParams(), max.deviceParams(), {1}, ReduceOp::kMax);
    argmaxAll(*device, src.deviceParams(), index.deviceParams());
    argmaxAxis(*device, src.deviceParams(), coordinates.deviceParams(), 1);
    device->argmaxMask(src.deviceParams(), mask.deviceParams());
    device->argmaxMask(src.deviceParams(), axisMask.deviceParams(), std::array<size_t, 1>{1});
    device->synchronize();
    CHECK(sum.data<double>()[0] == 12.0);
    CHECK(sum.data<double>()[1] == 12.0);
    CHECK(max.data<double>()[0] == 5.0);
    CHECK(max.data<double>()[1] == 4.0);
    CHECK(index.item<int32_t>() == 1);
    CHECK(coordinates.data<int32_t>()[0] == 1);
    CHECK(coordinates.data<int32_t>()[1] == 0);
    for (size_t i = 0; i < mask.size(); ++i)
    {
        CHECK(mask.data<int32_t>()[i] == (i == 1 ? 1 : 0));
        CHECK(axisMask.data<int32_t>()[i] == (i == 1 || i == 3 ? 1 : 0));
    }
}


TEST_CASE("Device reduceTo exact sums across backends")
{
    aix::DeviceCPU refDevice;
    auto deviceTypes = testDeviceTypes;
    deviceTypes.insert(deviceTypes.begin(), DeviceType::kCPU);
    const std::array<float, 6> values = {1, 2, 3, 4, 5, 6};
    // Include the initial destination value of one in each independently known sum.
    const std::pair<Shape, std::vector<float>> cases[] = {
        {{3}, {6, 8, 10}},
        {{2, 1}, {7, 16}},
        {{1}, {22}}
    };
    for (auto deviceType : deviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));

        for (size_t i = 0; i < aix::DataTypeCount; ++i)
        {
            const auto dtype = static_cast<DataType>(i);
            if (deviceType == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(i);
            auto src = TensorValue(values.data(), values.size(), DataType::kFloat32, {2, 3}, &refDevice, dtype);
            auto deviceSrc = src.to(device.get());
            for (const auto& [shape, expected] : cases)
            {
                auto actual = TensorValue(1.0f, shape, device.get(), dtype);
                const Index axes = shape == Shape{3} ? Index{0} : shape == Shape{2, 1} ? Index{1} : Index{0, 1};
                reduceAxes(*device, deviceSrc.deviceParams(), actual.deviceParams(), axes, ReduceOp::kSum);
                device->synchronize();
                auto hostResult = actual.to(&refDevice).to(DataType::kFloat64);
                REQUIRE(hostResult.size() == expected.size());
                for (size_t j = 0; j < expected.size(); ++j)
                {
                    CAPTURE(j);
                    // Small integer sums are exact even in Float16/BFloat16,
                    // regardless of accumulation precision or reduction order.
                    CHECK(hostResult.data<double>()[j] == expected[j]);
                }
            }
        }
    }
}


TEST_CASE("Device reduceTo low-precision accumulation regression")
{
    aix::DeviceCPU refDevice;

    // Minimized from seed 0, trial 92 of the old randomized test. For column 0,
    // sequential Float16 accumulation gave -0.66015625; Metal gave -0.65966796875.
    // The latter is the correctly rounded sum, but their difference exceeds 1e-5.
    const std::array<float, 18> values = {
        -0x1.bb4p-3f,  0x1.b9cp-1f, -0x1.08p-1f,
         0x1.64p-2f,  -0x1.ff4p-2f, -0x1.1a4p-1f,
        -0x1.1p-5f,   -0x1.e6cp-1f, -0x1.d7p-1f,
        -0x1.f84p-1f,  0x1.1ep-2f,   0x1.b5cp-1f,
        -0x1.778p-3f,  0x1.8b8p-2f, -0x1.f64p-3f,
         0x1.a44p-2f,  0x1.3ccp-1f,  0x1.898p-3f
    };
    for (auto deviceType : testDeviceTypes)
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;

        for (auto dtype : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
        {
            auto src = TensorValue(values.data(), values.size(), DataType::kFloat32, {6, 3}, &refDevice, dtype);
            for (float initialValue : {0.0f, 0.5f})
            {
                CAPTURE(initialValue);
                checkReduceTo(device.get(), src, {3}, initialValue);
                checkReduceTo(device.get(), src, {1}, initialValue);
            }
        }
    }
}


TEST_CASE("Device reduceTo error bound remains sensitive to incorrect sums")
{
    // A near-zero result still needs a cancellation-aware accumulation allowance.
    const double cancellationBound = reduceToErrorBound(DataType::kFloat32, 128, 64.0, 0.0);
    CHECK(cancellationBound > 1e-5);
    CHECK(cancellationBound < 0.001);
    for (auto dtype : {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
    {
        const double bound = reduceToErrorBound(dtype, 6, 2.0, -0.65966796875);
        CHECK(bound > 0.0);
        // A 1% error in this short reduction must not be accepted.
        CHECK(bound < 0.01 * 0.65966796875);
        CHECK_FALSE(std::abs(std::numeric_limits<double>::quiet_NaN()) <= bound);
        CHECK_FALSE(std::abs(std::numeric_limits<double>::infinity()) <= bound);
    }
    CHECK(reduceToErrorBound(DataType::kInt8, 32, 96.0, 96.0) == 0.0);
    CHECK(reduceToErrorBound(DataType::kFloat64, 128, 64.0, 0.0) < 1e-10);
}


TEST_CASE("DeviceCPU Tests - maxTo")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<100; ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testMaxTo(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - indexSelect")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testIndexSelect(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - indexAdd")
{
    // For each available devices, tests add operation.
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        // Create a new device per test.
        for (size_t i=0; i<testSizes.size(); ++i)
        {
            auto device2 = aix::createDevice(deviceType);
            CHECK(testIndexAdd(&*device2));
        }
    }
}


TEST_CASE("DeviceCPU Tests - batch compute")
{
    // If a device uses an advanced command queuing method, subsequent commands should be executed properly once the
    // synchronize method is called.

    Shape shape{2,3};
    std::initializer_list<float> data1{1.0, 2.0, 3.0,  4.0,  5.0,  6.0};
    std::initializer_list<float> data2{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t queueSize = 200;

    SUBCASE("Add")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x + y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x + y;
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({1608.0,2010.0,2412.0,2814.0,3216.0,3618.0},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
        }
    }

    SUBCASE("Sub")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x - y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z - x - y;
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({-1606.0,-2006.0,-2406.0,-2806.0,-3206.0,-3606.0},  shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-199.0,-199.0,-199.0,-199.0,-199.0,-199.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-201.0,-201.0,-201.0,-201.0,-201.0,-201.0}, shape).value());
        }
    }

    SUBCASE("Mul")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x * y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x * y;
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({1407.0,3216.0,5427.0,8040.0,11055.0,14472.0}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({1407.0,1608.0,1809.0,2010.0,2211.0,2412.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,402.0,603.0,804.0,1005.0,1206.0}, shape).value());
        }
    }

    SUBCASE("Div")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x / y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x / y;
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({28.7143,50.25,66.9999,80.4002,91.3635,100.5}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({28.7143,25.125,22.3333,20.1,18.2727,16.75}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-4.1020,-6.2812,-7.4444,-8.0400,-8.3058,-8.3750}, shape).value());
        }
    }

    SUBCASE("Sum")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.sum() + y.sum();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sum() + y.sum();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor(15678));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({201.0,201.0,201.0,201.0,201.0,201.0}, shape).value());
        }
    }

    SUBCASE("Mean")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.mean() + y.mean();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.mean() + y.mean();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor(2613));
            CheckVectorApproxValues(x.grad(), aix::tensor({33.5,33.5,33.5,33.5,33.5,33.5}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({33.5,33.5,33.5,33.5,33.5,33.5}, shape).value());
        }
    }

    SUBCASE("sqrt")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.sqrt() + y.sqrt();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sqrt() + y.sqrt();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({732.7961,852.7692,951.1430,1037.6198,1116.0948,1188.6305}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({100.5000,71.0643,58.0236,50.2500,44.9449,41.0290}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({37.9855,35.5321,33.5000,31.7808,30.3018,29.0118}, shape).value());
        }
    }

    SUBCASE("sin")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.sin() + y.sin();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.sin() + y.sin();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({301.1897,381.6301,111.2009,-261.4660,-393.7420,-164.0143}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({108.6004,-83.6453,-198.9882,-131.3821,57.0160,192.9943}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({151.5342,-29.2455,-183.1375,-168.6533,0.889566,169.6149}, shape).value());
        }
    }

    SUBCASE("cos")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.cos() + y.cos();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.cos() + y.cos();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({260.1352,-112.8908,-382.1257,-300.0356,57.9055,362.6089}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-169.1358,-182.7688,-28.3652,152.1176,192.7436,56.1625}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({-132.0546,-198.8614,-82.8357,109.3483,200.9977,107.8513}, shape).value());
        }
    }

    SUBCASE("log")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.log() + y.log();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.log() + y.log();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({391.1286,557.2905,662.4633,741.4656,805.4725,859.6092}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({201.0000,100.5000,66.9999,50.2500,40.2001,33.5000}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({28.7143,25.1250,22.3333,20.1000,18.2727,16.7500}, shape).value());
        }
    }

    SUBCASE("exp")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.exp() + y.exp();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.exp() + y.exp();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({220970.0,600657.0,1.63276e+06,4.43829e+06,1.20645e+07,3.27948e+07}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({546.375,1485.2,4037.19,10974.3,29831.1,81089.3}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({220424.0,599173.0,1.62872e+06,4.42732e+06,1.20347e+07,3.27137e+07}, shape).value());
        }
    }

    SUBCASE("pow")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });
            auto exp = aix::tensor(2.0);

            auto z = x.pow(exp) + y.pow(exp);
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.pow(exp) + y.pow(exp);
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({10050.0,13668.0,18090.0,23316.0,29346.0,36180.0}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({402.0,804.0,1206.0,1608.0,2010.0,2412.0}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({2814.0,3216.0,3618.0,4020.0,4422.0,4824.0}, shape).value());
        }
    }

    SUBCASE("tanh")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x.tanh() + y.tanh();
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.tanh() + y.tanh();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({354.0808,394.7695,401.0063,401.8651,401.9816,401.9982}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({84.4149,14.2008,1.98307,0.269514,0.0365,0.00493598}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({0.00067091,9.58443e-05,2.39611e-05,0.0,0.0,0.0}, shape).value());
        }
    }

    SUBCASE("matmul")
    {
        Shape matShape{2, 2};
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor({1.0, 2.0, 3.0, 4.0}, matShape, { .m_requireGrad=true });
            auto y = aix::tensor({5.0, 6.0, 7.0, 8.0}, matShape, { .m_requireGrad=true });

            auto z = x.matmul(y) + y.matmul(x);
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x.matmul(y) + y.matmul(x);
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({8442.0,11256.0,14874.0,19296.0}, matShape));
            CheckVectorApproxValues(x.grad(), aix::tensor({4623.0,5427.0,5025.0,5829.0}, matShape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({1407.0,2211.0,1809.0,2613.0}, matShape).value());
        }
    }

    SUBCASE("complex equation")
    {
        // For each available devices, tests add operation.
        for (auto deviceType : testDeviceTypes)
        {
            // Check if the devices is available.
            auto device = aix::createDevice(deviceType);
            if (!device) continue;      // Skip if the device is not available.

            auto x = aix::tensor(data1, shape, { .m_requireGrad=true });
            auto y = aix::tensor(data2, shape, { .m_requireGrad=true });

            auto z = x + y - (x * y).log() + y/x.exp() + (x-y) * x * x.sin() / y;
            for (size_t i=0; i<queueSize; ++i)
            {
                z = z + x + y*y / x.sum() - (x * y).sin()- y / y.exp() + (x-y) * x * x.sin() / y.tanh() + (y * y) / (x*x).mean();
            }
            z.backward(aix::onesLike(z));
            device->synchronize();

            CheckVectorApproxValues(z, aix::tensor({178.2879,-264.8438,1748.9033,6566.8809,9716.0850,6445.9224}, shape));
            CheckVectorApproxValues(x.grad(), aix::tensor({-2764.4619,1425.6165,3467.7439,4074.8318,-2422.4048,-5619.4829}, shape).value());
            CheckVectorApproxValues(y.grad(), aix::tensor({1.1793,384.1357,500.5189,1594.3270,1437.5804,2042.0659}, shape).value());
        }
    }
}


TEST_CASE("DeviceCPU Tests - long command batch queue")
{
    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        ssize_t size = 1024 * 1024;
        std::vector<float> data(size, 1);
        auto x = aix::tensor(data, { .m_dtype=aix::DataType::kFloat32, .m_device=device.get() }).reshape({1, size});
        auto y = aix::tensor(data, { .m_dtype=aix::DataType::kFloat32, .m_device=device.get() }).reshape({1, size});
        auto z = x + y;

        for (size_t i=1; i<1024; ++i)
        {
            z = z + x + y;
        }
        device->synchronize();

        CHECK(z.value().data<float>()[0] == 2048);
    }
}


TEST_CASE("DeviceCPU Tests - loop without sync")
{
    constexpr int kNumSamples  = 4;
    constexpr int kNumInputs   = 2;
    constexpr int kNumTargets  = 1;
    constexpr int kNumEpochs   = 1000;
    constexpr float kLearningRate  = 0.01f;
    constexpr float kLossThreshold = 1e-3f;

    for (auto deviceType : testDeviceTypes)
    {
        // Check if the devices is available.
        auto device = aix::createDevice(deviceType);
        if (!device) continue;      // Skip if the device is not available.

        aix::nn::Sequential model;
        model.add(new aix::nn::Linear(kNumInputs, 10));
        model.add(new aix::nn::Tanh());
        model.add(new aix::nn::Linear(10, kNumTargets));
        model.to(device);

        // Example inputs and targets for demonstration purposes.
        auto inputs = aix::tensor({0.0, 0.0,
                                   0.0, 1.0,
                                   1.0, 0.0,
                                   1.0, 1.0}, {kNumSamples, kNumInputs}).to(device);

        auto targets = aix::tensor({0.0,
                                    1.0,
                                    1.0,
                                    0.0}, {kNumSamples, kNumTargets}).to(device);

        aix::optim::Adam optimizer(model.parameters(), kLearningRate);
        auto lossFunc = aix::nn::MSELoss();

        for (size_t epoch = 0; epoch < kNumEpochs; ++epoch)
        {
            auto predictions = model.forward(inputs);
            auto loss = lossFunc(predictions, targets);
            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();
            // IMPORTANT NOTE: We keep optimizing without synchronizing.
        }
        auto loss = lossFunc(model.forward(inputs), targets);
        device->synchronize();

        CHECK(loss.value().item<float>() <= kLossThreshold);
    }
}
