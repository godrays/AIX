//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// Project includes
#include "aixDeviceMetal.hpp"
#include "aixDeviceMetalCache.hpp"
#include "aixDeviceMetalShaders.hpp"
// External includes
#include <Metal/Metal.hpp>
// System includes


namespace aix::metal
{

DeviceMetal::DeviceMetal(size_t deviceIndex)
{
    // Create autorelease pool.
    m_pool = NS::AutoreleasePool::alloc()->init();
    m_mtlDevice = createMTLDevice(deviceIndex);
    m_maxWorkingSetSize = static_cast<size_t>(static_cast<double>(m_mtlDevice->recommendedMaxWorkingSetSize()) * 0.7);
    m_allocator = std::make_unique<MetalAllocator>(m_mtlDevice, ALLOCATOR_ALIGNMENT_SIZE);
    m_bufferCache = std::make_unique<MTLBufferCache>();
    auto defaultLibrary = createLibrary(shaders::aixDeviceMetalShaders);
    auto nullKernelName = "nullKernel";

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto iDType = static_cast<DataType>(i);
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            auto jDType = static_cast<DataType>(j);
            // Metal Framework does not support kFloat64 format.
            bool isNull = iDType == DataType::kFloat64 || jDType == DataType::kFloat64;
            std::string kernelName = "copy_" + toString(i) + "_" + toString(j);
            m_compFuncPSOCopyAA[i][j] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : kernelName);
            kernelName = "fill_" + toString(i) + "_" + toString(j);
            m_compFuncPSOFill[i][j]   = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : kernelName);
        }

        // Metal Framework does not support kFloat64 format.
        bool isNull = iDType == DataType::kFloat64;
        std::string dtypeStr = toString(i);
        m_compFuncPSOAdd[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "add_" + dtypeStr);
        m_compFuncPSOAddStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "add_strided_" + dtypeStr);
        m_compFuncPSOSub[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sub_" + dtypeStr);
        m_compFuncPSOSubStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sub_strided_" + dtypeStr);
        m_compFuncPSOMul[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "mul_" + dtypeStr);
        m_compFuncPSOMulStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "mul_strided_" + dtypeStr);
        m_compFuncPSODiv[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "div_" + dtypeStr);
        m_compFuncPSODivStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "div_strided_" + dtypeStr);
        m_compFuncPSOUnary[i]       = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "unary_" + dtypeStr);
        m_compFuncPSOUnaryStrided[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "unary_strided_" + dtypeStr);
        m_compFuncPSOFillMin[i]     = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "fillMin_" + dtypeStr);
        m_compFuncPSOSqrt[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sqrt_" + dtypeStr);
        m_compFuncPSOSqrtStrided[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sqrt_strided_" + dtypeStr);
        m_compFuncPSOSin[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sin_" + dtypeStr);
        m_compFuncPSOSinStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sin_strided_" + dtypeStr);
        m_compFuncPSOCos[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "cos_" + dtypeStr);
        m_compFuncPSOCosStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "cos_strided_" + dtypeStr);
        m_compFuncPSOTanh[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "tanh_" + dtypeStr);
        m_compFuncPSOTanhStrided[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "tanh_strided_" + dtypeStr);
        m_compFuncPSOLog[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "log_" + dtypeStr);
        m_compFuncPSOLogStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "log_strided_" + dtypeStr);
        m_compFuncPSOExp[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "exp_" + dtypeStr);
        m_compFuncPSOExpStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "exp_strided_" + dtypeStr);
        m_compFuncPSOPow[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "pow_" + dtypeStr);
        m_compFuncPSOPowStrided[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "pow_strided_" + dtypeStr);
        m_compFuncPSOSum[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sum_" + dtypeStr);
        m_compFuncPSOMax[i]         = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "max_" + dtypeStr);
        m_compFuncPSOArgmaxInit[i]   = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "argmaxInit_" + dtypeStr);
        m_compFuncPSOArgmaxReduce[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "argmaxReduce_" + dtypeStr);
        m_compFuncPSOArgmaxTo[i]     = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "argmaxTo_" + dtypeStr);
        m_compFuncPSOMatMulStrided[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMulStrided_" + dtypeStr);
        m_compFuncPSOMatMulTiledBC6464888[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMulTiledBC_64_64_8_8_8_" + dtypeStr);
        m_compFuncPSOMatMulTiled32x32[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMulTiled_32_32_" + dtypeStr);
        m_compFuncPSOMatMulTiled32x64[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMulTiled_32_64_" + dtypeStr);
        m_compFuncPSOMatMulTiled32x128[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "matrixMulTiled_32_128_" + dtypeStr);
        m_compFuncPSOTranspose2D[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose2D_" + dtypeStr);
        m_compFuncPSOTranspose2DTiled16x16x8[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose2DTiled_16_16_8_" + dtypeStr);
        m_compFuncPSOTranspose2DTiled32x32x8[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose2DTiled_32_32_8_" + dtypeStr);
        m_compFuncPSOTranspose[i]   = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "transpose_" + dtypeStr);
        m_compFuncPSOContiguous[i]  = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "contiguous_" + dtypeStr);
        m_compFuncPSOReduceTo[i]    = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "reduceTo_" + dtypeStr);
        m_compFuncPSOMaxTo[i]       = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "maxTo_" + dtypeStr);
        m_compFuncPSOSliceSet[i]    = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "sliceSet_" + dtypeStr);
        m_compFuncPSOTril[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "tril_" + dtypeStr);
        m_compFuncPSOTriu[i]        = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "triu_" + dtypeStr);
        m_compFuncPSOIndexSelect[i] = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "indexSelect_" + dtypeStr);
        m_compFuncPSOIndexAdd[i]    = createComputeFuncPSO(defaultLibrary, isNull ? nullKernelName : "indexAdd_" + dtypeStr);
    }

    m_compFuncPSOArgmaxIndicesSet = createComputeFuncPSO(defaultLibrary, "argmaxIndicesSet");
    m_compFuncPSOArgmaxIndicesToSet = createComputeFuncPSO(defaultLibrary, "argmaxIndicesToSet");

    m_cmdQueue = createCommandQueue();
    m_cmdBuffer = m_cmdQueue->commandBufferWithUnretainedReferences();
    m_compEncoder = m_cmdBuffer->computeCommandEncoder();
}

// Destructor
DeviceMetal::~DeviceMetal()
{
    if (m_currentBatchSize > 0)
    {
        std::cerr << "WARNING: Queued tensor operations detected. Did you forget to call synchronize()?" << std::endl;
    }

    m_bufferCache->clear();

    // Note: No need to release MTL Buffer objects in m_allocMap.
    m_compEncoder->endEncoding();

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        for (size_t j=0; j<aix::DataTypeCount; ++j)
        {
            m_compFuncPSOCopyAA[i][j]->release();
            m_compFuncPSOFill[i][j]->release();
        }
        m_compFuncPSOAdd[i]->release();
        m_compFuncPSOAddStrided[i]->release();
        m_compFuncPSOSub[i]->release();
        m_compFuncPSOSubStrided[i]->release();
        m_compFuncPSOMul[i]->release();
        m_compFuncPSOMulStrided[i]->release();
        m_compFuncPSODiv[i]->release();
        m_compFuncPSODivStrided[i]->release();
        m_compFuncPSOUnary[i]->release();
        m_compFuncPSOUnaryStrided[i]->release();
        m_compFuncPSOSqrt[i]->release();
        m_compFuncPSOSqrtStrided[i]->release();
        m_compFuncPSOSin[i]->release();
        m_compFuncPSOSinStrided[i]->release();
        m_compFuncPSOCos[i]->release();
        m_compFuncPSOCosStrided[i]->release();
        m_compFuncPSOTanh[i]->release();
        m_compFuncPSOTanhStrided[i]->release();
        m_compFuncPSOLog[i]->release();
        m_compFuncPSOLogStrided[i]->release();
        m_compFuncPSOExp[i]->release();
        m_compFuncPSOExpStrided[i]->release();
        m_compFuncPSOPow[i]->release();
        m_compFuncPSOPowStrided[i]->release();
        m_compFuncPSOSum[i]->release();
        m_compFuncPSOMax[i]->release();
        m_compFuncPSOArgmaxInit[i]->release();
        m_compFuncPSOArgmaxReduce[i]->release();
        m_compFuncPSOArgmaxTo[i]->release();
        m_compFuncPSOMatMulTiledBC6464888[i]->release();
        m_compFuncPSOMatMulTiled32x32[i]->release();
        m_compFuncPSOMatMulTiled32x64[i]->release();
        m_compFuncPSOMatMulTiled32x128[i]->release();
        m_compFuncPSOTranspose2D[i]->release();
        m_compFuncPSOTranspose2DTiled16x16x8[i]->release();
        m_compFuncPSOTranspose2DTiled32x32x8[i]->release();
        m_compFuncPSOTranspose[i]->release();
        m_compFuncPSOContiguous[i]->release();
        m_compFuncPSOReduceTo[i]->release();
        m_compFuncPSOMaxTo[i]->release();
        m_compFuncPSOSliceSet[i]->release();
        m_compFuncPSOTril[i]->release();
        m_compFuncPSOTriu[i]->release();
        m_compFuncPSOIndexSelect[i]->release();
        m_compFuncPSOIndexAdd[i]->release();
    }
    m_compFuncPSOArgmaxIndicesSet->release();
    m_compFuncPSOArgmaxIndicesToSet->release();

    m_cmdQueue->release();
    m_mtlDevice->release();
    m_pool->release();
}

// Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
void* DeviceMetal::allocate(size_t size)
{
    auto mtlBuf = newBuffer(size);
    auto contentPtr = mtlBuf->contents();
    m_allocMap[contentPtr] = mtlBuf;
    return contentPtr;
}

void* DeviceMetal::allocate(size_t size, DataType dtype)
{
    return allocate(align(size, TOTAL_COMPONENT_COUNT) * dataTypeSize(dtype));
}

// Deallocate GPU memory if it's allocated by current device.
void DeviceMetal::deallocate(void * memory)
{
    if (!isDeviceBuffer(memory))
        throw std::invalid_argument("DeviceMetal::deallocate() - Found different type of memory to free.");
    auto mtlBuf = m_allocMap[memory];
    // IMPORTANT: Delay all deallocations of device buffers until all commands in the batch queue are executed.
    m_tempBuffers.emplace_back(mtlBuf, mtlBuf->contents());
}

void DeviceMetal::add(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeTripleArrayCmd(a1, a2, result, m_compFuncPSOAdd[iDType], m_compFuncPSOAddStrided[iDType],
                          "add_" + toString(result.dtype));
}

void DeviceMetal::sub(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeTripleArrayCmd(a1, a2, result, m_compFuncPSOSub[iDType], m_compFuncPSOSubStrided[iDType],
                          "sub_" + toString(result.dtype));
}

void DeviceMetal::mul(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeTripleArrayCmd(a1, a2, result, m_compFuncPSOMul[iDType], m_compFuncPSOMulStrided[iDType],
                          "mul_" + toString(result.dtype));
}

void DeviceMetal::div(const DeviceTensorParams& a1, const DeviceTensorParams& a2, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeTripleArrayCmd(a1, a2, result, m_compFuncPSODiv[iDType], m_compFuncPSODivStrided[iDType],
                          "div_" + toString(result.dtype));
}

void DeviceMetal::unary(const DeviceTensorParams& a1, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a1, result, m_compFuncPSOUnary[iDType], m_compFuncPSOUnaryStrided[iDType],
                          "unary_" + toString(result.dtype));
}

void DeviceMetal::fill(const void* scalar, DataType scalarDType, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    validateDataType(scalarDType);
    validateDataType(result.dtype);
    auto iSrcDType = static_cast<size_t>(scalarDType);
    auto iDstDType = static_cast<size_t>(result.dtype);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::fill() result must have GPU memory.");

    if (isDeviceBuffer(scalar))
        throw std::invalid_argument("DeviceMetal::fill() scalar address cannot be a device-allocated address.");

    // bufScalar is a temporary size aligned buffer to be used as vector of 4.
    auto bufScalar = getReadOnlyMTLBuffer(scalar, 1, dataTypeSize(scalarDType), 1);
    auto bufResult = m_allocMap[result.data];
    auto compFuncPSO = m_compFuncPSOFill[iSrcDType][iDstDType];

    // Calculate maximum thread group dimensions
    auto asize = align(result.size, TOTAL_COMPONENT_COUNT) / TOTAL_COMPONENT_COUNT;
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup());

    // Serialize resource and states to be called by GPU.
    encodeComputeCommandDoubleBuffer(bufScalar, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});
    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufScalar);
    commitBatchQueue();
}

void DeviceMetal::fillMin(const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    validateDataType(result.dtype);
    auto iDType = static_cast<size_t>(result.dtype);
    auto compFuncPSO = m_compFuncPSOFillMin[iDType];

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::fillMin() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto bufResult = m_allocMap[result.data];

    // Calculate maximum thread group dimensions
    auto asize = align(result.size, TOTAL_COMPONENT_COUNT) / TOTAL_COMPONENT_COUNT;
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup());

    // Encode the pipeline state object and its parameters.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufResult, 0, 0);
    m_compEncoder->dispatchThreads({asize, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    commitBatchQueue();
}

void DeviceMetal::sum(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    auto iDType = static_cast<size_t>(result.dtype);
    auto compFuncPSO = m_compFuncPSOSum[iDType];

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::sum() result must have GPU memory.");

    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP, compFuncPSO->maxTotalThreadsPerThreadgroup());

    auto bufSrc = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto bufTemp = m_allocMap[allocate(a.size, a.dtype)];
    auto boundLayout = bindTensorLayout(a, {.bufferIndex=4});

    auto encodeReduction = [&](const MTL::Buffer* srcBuffer, MTL::Buffer* dstBuffer, size_t elementCount, size_t useLayout)
    {
        m_compEncoder->setComputePipelineState(compFuncPSO);
        m_compEncoder->setBuffer(srcBuffer, 0, 0);
        m_compEncoder->setBuffer(dstBuffer, 0, 1);
        m_compEncoder->setBytes(&elementCount, sizeof(elementCount), 2);
        m_compEncoder->setBytes(&useLayout, sizeof(useLayout), 3);
        m_compEncoder->setBytes(boundLayout.data.data(), boundLayout.data.size() * sizeof(size_t), 4);

        NS::UInteger w = std::min(elementCount, maxThreadsPerTG);
        m_compEncoder->dispatchThreads({elementCount, 1, 1}, {w, 1, 1});
        commitBatchQueue();
    };

    size_t elementCount = a.size;
    size_t useLayout = (a.isContiguous && a.offset == 0) ? 0 : 1;
    encodeReduction(bufSrc, bufTemp, elementCount, useLayout);
    while (elementCount > 1)
    {
        elementCount = (elementCount + maxThreadsPerTG - 1) / maxThreadsPerTG;
        if (elementCount == 1) break;
        encodeReduction(bufTemp, bufTemp, elementCount, 0);
    }

    // Copy result from temp buf to result buffer.
    copy(bufTemp->contents(), result.dtype, result.data, result.dtype, 1);

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundLayout);
    deallocate(bufTemp->contents());
}

void DeviceMetal::sqrt(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOSqrt[iDType], m_compFuncPSOSqrtStrided[iDType],
                          "sqrt_" + toString(result.dtype));
}

void DeviceMetal::sin(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOSin[iDType], m_compFuncPSOSinStrided[iDType],
                          "sin_" + toString(result.dtype));
}

void DeviceMetal::cos(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOCos[iDType], m_compFuncPSOCosStrided[iDType],
                          "cos_" + toString(result.dtype));
}

void DeviceMetal::tanh(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOTanh[iDType], m_compFuncPSOTanhStrided[iDType],
                          "tanh_" + toString(result.dtype));
}

void DeviceMetal::log(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOLog[iDType], m_compFuncPSOLogStrided[iDType],
                          "log_" + toString(result.dtype));
}

void DeviceMetal::exp(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeDoubleArrayCmd(a, result, m_compFuncPSOExp[iDType], m_compFuncPSOExpStrided[iDType],
                          "exp_" + toString(result.dtype));
}

void DeviceMetal::pow(const DeviceTensorParams& a, const DeviceTensorParams& exp, const DeviceTensorParams& result)
{
    auto iDType = static_cast<size_t>(result.dtype);
    executeTripleArrayCmd(a, exp, result, m_compFuncPSOPow[iDType], m_compFuncPSOPowStrided[iDType],
                          "pow_" + toString(result.dtype));
}

void DeviceMetal::max(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    auto iDType = static_cast<size_t>(result.dtype);
    auto compFuncPSO = m_compFuncPSOMax[iDType];

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::max() result must have GPU memory.");

    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP, compFuncPSO->maxTotalThreadsPerThreadgroup());

    auto bufSrc = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto bufTemp = m_allocMap[allocate(a.size, a.dtype)];
    auto boundLayout = bindTensorLayout(a, {.bufferIndex=4});

    auto encodeReduction = [&](const MTL::Buffer* srcBuffer, MTL::Buffer* dstBuffer, size_t elementCount, size_t useLayout)
    {
        m_compEncoder->setComputePipelineState(compFuncPSO);
        m_compEncoder->setBuffer(srcBuffer, 0, 0);
        m_compEncoder->setBuffer(dstBuffer, 0, 1);
        m_compEncoder->setBytes(&elementCount, sizeof(elementCount), 2);
        m_compEncoder->setBytes(&useLayout, sizeof(useLayout), 3);
        m_compEncoder->setBytes(boundLayout.data.data(), boundLayout.data.size() * sizeof(size_t), 4);

        NS::UInteger w = std::min(elementCount, maxThreadsPerTG);
        m_compEncoder->dispatchThreads({elementCount, 1, 1}, {w, 1, 1});
        commitBatchQueue();
    };

    size_t elementCount = a.size;
    size_t useLayout = (a.isContiguous && a.offset == 0) ? 0 : 1;
    encodeReduction(bufSrc, bufTemp, elementCount, useLayout);
    while (elementCount > 1)
    {
        elementCount = (elementCount + maxThreadsPerTG - 1) / maxThreadsPerTG;
        if (elementCount == 1) break;
        encodeReduction(bufTemp, bufTemp, elementCount, 0);
    }

    // Copy result from temp buf to result buffer.
    copy(bufTemp->contents(), a.dtype, result.data, result.dtype, 1);

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundLayout);
    deallocate(bufTemp->contents());
}

void DeviceMetal::argmax(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    if (result.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceMetal::argmax supports only int32 data type for its result.");
    }

    if (a.dtype == DataType::kFloat64 || !isDeviceBuffer(result.data))
    {
        synchronize();
        defaultDevice.argmax(a, result);
        return;
    }

    assert(a.size > 0);
    auto iDType = static_cast<size_t>(a.dtype);
    auto argmaxInitPSO = m_compFuncPSOArgmaxInit[iDType];
    auto argmaxReducePSO = m_compFuncPSOArgmaxReduce[iDType];
    size_t maxThreadsPerTG = std::min<size_t>(MAX_THREADS_PER_THREADGROUP,
                                              argmaxReducePSO->maxTotalThreadsPerThreadgroup());

    auto bufSrc = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto bufTempValues = m_allocMap[allocate(a.size, a.dtype)];
    auto bufTempIndices = m_allocMap[allocate(a.size, DataType::kInt32)];
    auto boundLayout = bindTensorLayout(a, {.bufferIndex=4});

    auto encodeArgmaxInit = [&]()
    {
        size_t useLayout = (a.isContiguous && a.offset == 0) ? 0 : 1;
        m_compEncoder->setComputePipelineState(argmaxInitPSO);
        m_compEncoder->setBuffer(bufSrc, 0, 0);
        m_compEncoder->setBuffer(bufTempValues, 0, 1);
        m_compEncoder->setBuffer(bufTempIndices, 0, 2);
        m_compEncoder->setBytes(&useLayout, sizeof(useLayout), 3);
        m_compEncoder->setBytes(boundLayout.data.data(), boundLayout.data.size() * sizeof(size_t), 4);

        NS::UInteger w = std::min(a.size, static_cast<size_t>(argmaxInitPSO->maxTotalThreadsPerThreadgroup()));
        m_compEncoder->dispatchThreads({a.size, 1, 1}, {w, 1, 1});
        commitBatchQueue();
    };

    auto encodeArgmaxReduce = [&](size_t elementCount)
    {
        m_compEncoder->setComputePipelineState(argmaxReducePSO);
        m_compEncoder->setBuffer(bufTempValues,  0, 0);
        m_compEncoder->setBuffer(bufTempIndices, 0, 1);
        m_compEncoder->setBuffer(bufTempValues,  0, 2);
        m_compEncoder->setBuffer(bufTempIndices, 0, 3);

        NS::UInteger w = std::min(elementCount, maxThreadsPerTG);
        m_compEncoder->dispatchThreads({elementCount, 1, 1}, {w, 1, 1});
        commitBatchQueue();
    };

    encodeArgmaxInit();

    size_t elementCount = a.size;
    while (elementCount > 1)
    {
        encodeArgmaxReduce(elementCount);
        elementCount = (elementCount + maxThreadsPerTG - 1) / maxThreadsPerTG;
    }

    copy(bufTempIndices->contents(), DataType::kInt32, result.data, result.dtype, 1);

    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundLayout);
    deallocate(bufTempValues->contents());
    deallocate(bufTempIndices->contents());
}

void DeviceMetal::argmaxIndices(const DeviceTensorParams& a, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    if (result.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceMetal::argmaxIndices supports only int32 data type for its result.");
    }

    if (a.dtype == DataType::kFloat64 || !isDeviceBuffer(result.data))
    {
        synchronize();
        defaultDevice.argmaxIndices(a, result);
        return;
    }

    auto winningIndex = DeviceTensorParams
    {
        .data=allocate(1, DataType::kInt32),
        .dtype=DataType::kInt32,
        .isContiguous=true,
        .offset=0,
        .shape={},
        .size=1,
        .strides={}
    };

    argmax(a, winningIndex);

    int32_t zero = 0;
    fill(&zero, DataType::kInt32, result);

    auto bufWinningIndex = m_allocMap[winningIndex.data];
    auto bufResult = m_allocMap[result.data];

    m_compEncoder->setComputePipelineState(m_compFuncPSOArgmaxIndicesSet);
    m_compEncoder->setBuffer(bufWinningIndex, 0, 0);
    m_compEncoder->setBuffer(bufResult, 0, 1);
    m_compEncoder->setBytes(&result.size, sizeof(result.size), 2);
    m_compEncoder->dispatchThreads({1, 1, 1}, {1, 1, 1});
    commitBatchQueue();

    deallocate(winningIndex.data);
}

void DeviceMetal::matmul(const DeviceTensorParams& a, const DeviceTensorParams& b, const DeviceTensorParams& result)
{
    assert(result.isContiguous == true);
    validateDataType(result.dtype);
    auto iDType = static_cast<size_t>(result.dtype);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::matmul() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto buf2 = getReadOnlyMTLBuffer(b.data, b.size, dataTypeSize(b.dtype));
    auto bufResult = m_allocMap[result.data];
    auto buf1Size = MatrixSize{a.shape[0], a.shape[1]};
    auto buf2Size = MatrixSize{b.shape[0], b.shape[1]};

    size_t M = buf1Size.rows;
    size_t K = buf1Size.cols;
    size_t N = buf2Size.cols;

    auto encodeParams = [&](const MTL::ComputePipelineState* compFuncPSO)
    {
        m_compEncoder->setComputePipelineState(compFuncPSO);
        m_compEncoder->setBuffer(buf1, 0, 0);
        m_compEncoder->setBuffer(buf2, 0, 1);
        m_compEncoder->setBuffer(bufResult, 0, 2);
        m_compEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 3);
        m_compEncoder->setBytes(&buf2Size, sizeof(MatrixSize), 4);
    };

    auto dispatchTiled = [&](const MTL::ComputePipelineState* compFuncPSO, const size_t tileSizeX, const size_t tileSizeY)
    {
        // Encode the pipeline state object and its parameters.
        uint numThreadgroupsX = (N + tileSizeX - 1) / tileSizeX;
        uint numThreadgroupsY = (M + tileSizeY - 1) / tileSizeY;
        assert(tileSizeX * tileSizeY / tileSizeX <= compFuncPSO->maxTotalThreadsPerThreadgroup());
        encodeParams(compFuncPSO);
        m_compEncoder->dispatchThreadgroups({numThreadgroupsX, numThreadgroupsY, 1}, {tileSizeX, tileSizeY/tileSizeX, 1});
    };

    bool packedOperands = a.isContiguous && a.offset == 0 && b.isContiguous && b.offset == 0;
    bool commonCondition = packedOperands && K % 32 == 0 && N % 32 == 0 &&
                           (result.dtype == aix::DataType::kFloat32 ||
                            result.dtype == aix::DataType::kFloat16 ||
                            result.dtype == aix::DataType::kBFloat16);
    if (!packedOperands)
    {
        auto compFuncPSO = m_compFuncPSOMatMulStrided[iDType];
        m_compEncoder->setComputePipelineState(compFuncPSO);
        m_compEncoder->setBuffer(buf1, 0, 0);
        m_compEncoder->setBuffer(buf2, 0, 1);
        m_compEncoder->setBuffer(bufResult, 0, 2);
        m_compEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 3);
        m_compEncoder->setBytes(&buf2Size, sizeof(MatrixSize), 4);
        auto boundLayoutA = bindTensorLayout(a, {.bufferIndex=5});
        auto boundLayoutB = bindTensorLayout(b, {.bufferIndex=6});
        constexpr NS::UInteger tileWidth = 8;
        constexpr NS::UInteger tileHeight = 8;
        assert(tileWidth * tileHeight <= compFuncPSO->maxTotalThreadsPerThreadgroup());
        m_compEncoder->dispatchThreads({N, M, 1}, {tileWidth, tileHeight, 1});
        releaseTensorLayout(boundLayoutA);
        releaseTensorLayout(boundLayoutB);
    }
    // Use fast matmul where dimensions are multiple of TSY.
    // TODO: Make SIMD comparison.
    else if (M % 128 == 0 && commonCondition)
    {
        dispatchTiled(m_compFuncPSOMatMulTiled32x128[iDType], 32, 128);
    }
    else if (M % 64 == 0 && commonCondition)
    {
        dispatchTiled(m_compFuncPSOMatMulTiled32x64[iDType], 32, 64);
    }
    else if (M % 32 == 0 && commonCondition)
    {
        dispatchTiled(m_compFuncPSOMatMulTiled32x32[iDType], 32, 32);
    }
    else
    {
        constexpr size_t tileSize = 64;
        constexpr size_t numThreads = 64;
        uint numThreadgroupsX = (N + tileSize - 1) / tileSize;
        uint numThreadgroupsY = (M + tileSize - 1) / tileSize;
        auto compFuncPSO = m_compFuncPSOMatMulTiledBC6464888[iDType];
        assert(numThreads <= compFuncPSO->maxTotalThreadsPerThreadgroup());
        encodeParams(compFuncPSO);
        m_compEncoder->dispatchThreadgroups({numThreadgroupsX, numThreadgroupsY, 1}, {numThreads, 1, 1});
    }

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(buf1);
    freeTemporaryBuffer(buf2);
    commitBatchQueue();
}

void DeviceMetal::transpose(const DeviceTensorParams& a, const DeviceTensorParams& result, size_t dim0, size_t dim1)
{
    assert(result.isContiguous == true);
    auto iDType = static_cast<size_t>(result.dtype);
    // Use fast and simplified version of the general transpose for matrix transpose operations.
    if (a.isContiguous && a.offset == 0 && a.shape.size() == 2 && dim0 == 0 && dim1 == 1)
    {
        transpose2D(a, result);
        return;
    }

    if (a.strides.size() > 16)
        throw std::invalid_argument("Metal device does not support tensors with more than 16 dimensions for acceleration.");

    validateDataType(result.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::transpose() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto bufData   = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto bufResult = m_allocMap[result.data];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(m_compFuncPSOTranspose[iDType]);
    m_compEncoder->setBuffer(bufData,   0,                0);
    m_compEncoder->setBuffer(bufResult, 0,                1);
    m_compEncoder->setBytes(&dim0,      sizeof(dim0),     2);
    m_compEncoder->setBytes(&dim1,      sizeof(dim1),     3);
    auto boundSrcLayout = bindTensorLayout(a, {.bufferIndex=4});
    m_compEncoder->setBytes(&a.size,    sizeof(a.size),   5);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(a.size, m_compFuncPSOTranspose[iDType]->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({a.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufData);
    releaseTensorLayout(boundSrcLayout);
    commitBatchQueue();
}

void DeviceMetal::copy(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    validateDataType(srcDType);
    validateDataType(dstDType);
    auto iSrcDType = static_cast<size_t>(srcDType);
    auto iDstDType = static_cast<size_t>(dstDType);

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst))
        throw std::invalid_argument("DeviceMetal::copy() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(src, size, dataTypeSize(srcDType));
    auto bufResult = m_allocMap[dst];
    auto compFuncPSO = m_compFuncPSOCopyAA[iSrcDType][iDstDType];

    // Calculate maximum thread group dimensions
    auto asize = align(size, TOTAL_COMPONENT_COUNT) / TOTAL_COMPONENT_COUNT;
    NS::UInteger w = std::min(asize, compFuncPSO->maxTotalThreadsPerThreadgroup());

    encodeComputeCommandDoubleBuffer(buf1, bufResult, compFuncPSO, {asize, 1, 1}, {w, 1, 1});
    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(buf1);
    commitBatchQueue();
}

void DeviceMetal::copyImmediate(const void* src, DataType srcDType, void* dst, DataType dstDType, size_t size)
{
    copy(src, srcDType, dst, dstDType, size);
    synchronize();
}

void DeviceMetal::contiguous(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    assert(src.isContiguous == false && dst.isContiguous == true);
    validateDataType(src.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::contiguous() result must have GPU memory.");

    auto iDType = static_cast<size_t>(src.dtype);
    assert(src.shape.size() == src.strides.size());

    auto bufSrc     = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufDst     = m_allocMap[dst.data];
    auto computePSO = m_compFuncPSOContiguous[iDType];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(computePSO);
    m_compEncoder->setBuffer(bufSrc,     0, 0);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    auto boundLayout = bindTensorLayout(src, {.bufferIndex=2});

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(dst.size, computePSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({dst.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundLayout);
    commitBatchQueue();
}

void DeviceMetal::reduceTo(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    assert(dst.isContiguous == true);
    validateDataType(src.dtype);
    // NOTE: Metal Framework supports add and sub operations for only atomic_float, atomic_uint and atomic_int.
    //       Since reduceTo uses atomic<T>, we can only allow certain formats for acceleration for now.
    if (!(src.dtype == DataType::kFloat32 || src.dtype == DataType::kInt32))
    {
        synchronize();
        defaultDevice.reduceTo(src, dst);
        return;
    }

    auto iDType = static_cast<size_t>(src.dtype);
    translation(src, dst, m_compFuncPSOReduceTo[iDType], "reduceTo_" + toString(src.dtype));
    // NOTE: The ReduceTo function performs a sum operation. The order of these operations by GPU threads is not
    // guaranteed, which might result in minor differences in the final results due to floating-point precision limits.
}

void DeviceMetal::maxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst)
{
    assert(dst.isContiguous == true);
    validateDataType(src.dtype);
    // NOTE: Only certain data types are supported due to limitation of Metal Framework atomics.
    if (!(src.dtype == DataType::kFloat32 || src.dtype == DataType::kInt32))
    {
        synchronize();
        defaultDevice.maxTo(src, dst);
        return;
    }

    auto iDType = static_cast<size_t>(src.dtype);
    translation(src, dst, m_compFuncPSOMaxTo[iDType], "maxTo_" + toString(src.dtype));
}

void DeviceMetal::argmaxTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    assert(dst.isContiguous == true);
    if (dst.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceMetal::argmaxTo supports only int32 data type for its result.");
    }

    if (src.dtype == DataType::kFloat64 || !isDeviceBuffer(dst.data))
    {
        synchronize();
        defaultDevice.argmaxTo(src, dst, dim);
        return;
    }

    assert(dim < src.shape.size());
    assert(src.shape.size() == src.strides.size());
    assert(dst.size > 0);

    auto iDType = static_cast<size_t>(src.dtype);
    auto compFuncPSO = m_compFuncPSOArgmaxTo[iDType];
    auto bufSrc = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufDst = m_allocMap[dst.data];

    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufSrc, 0, 0);
    m_compEncoder->setBuffer(bufDst, 0, 1);
    auto boundLayout = bindTensorLayout(src, {.bufferIndex=2});
    m_compEncoder->setBytes(&dim, sizeof(dim), 3);

    NS::UInteger w = std::min(dst.size, static_cast<size_t>(compFuncPSO->maxTotalThreadsPerThreadgroup()));
    m_compEncoder->dispatchThreads({dst.size, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundLayout);
    commitBatchQueue();
}

void DeviceMetal::argmaxIndicesTo(const DeviceTensorParams& src, const DeviceTensorParams& dst, size_t dim)
{
    assert(dst.isContiguous == true);
    if (dst.dtype != DataType::kInt32)
    {
        throw std::invalid_argument("DeviceMetal::argmaxIndicesTo supports only int32 data type for its result.");
    }

    if (src.dtype == DataType::kFloat64 || !isDeviceBuffer(dst.data))
    {
        synchronize();
        defaultDevice.argmaxIndicesTo(src, dst, dim);
        return;
    }

    assert(dim < src.shape.size());
    assert(src.shape.size() == src.strides.size());
    assert(src.shape.size() == dst.strides.size());
    assert(dst.size > 0);

    auto reducedShape = src.shape;
    reducedShape[dim] = 1;

    Stride reducedStrides(reducedShape.size());
    size_t reducedSize = 1;
    for (int64_t i = static_cast<int64_t>(reducedShape.size()) - 1; i >= 0; --i)
    {
        reducedStrides[i] = reducedSize;
        reducedSize *= reducedShape[i];
    }

    auto winningCoords = DeviceTensorParams
    {
        .data=allocate(reducedSize, DataType::kInt32),
        .dtype=DataType::kInt32,
        .isContiguous=true,
        .offset=0,
        .shape=reducedShape,
        .size=reducedSize,
        .strides=reducedStrides
    };

    argmaxTo(src, winningCoords, dim);

    int32_t zero = 0;
    fill(&zero, DataType::kInt32, dst);

    auto shapeSize = src.shape.size();
    auto bufWinningCoords = m_allocMap[winningCoords.data];
    auto bufDst = m_allocMap[dst.data];
    auto bufShape = getReadOnlyMTLBuffer(src.shape.data(), shapeSize, sizeof(size_t));
    auto bufStrides = getReadOnlyMTLBuffer(dst.strides.data(), shapeSize, sizeof(size_t));

    m_compEncoder->setComputePipelineState(m_compFuncPSOArgmaxIndicesToSet);
    m_compEncoder->setBuffer(bufWinningCoords, 0, 0);
    m_compEncoder->setBuffer(bufDst, 0, 1);
    m_compEncoder->setBuffer(bufShape, 0, 2);
    m_compEncoder->setBuffer(bufStrides, 0, 3);
    m_compEncoder->setBytes(&shapeSize, sizeof(shapeSize), 4);
    m_compEncoder->setBytes(&dim, sizeof(dim), 5);

    NS::UInteger w = std::min(reducedSize,
                              static_cast<size_t>(m_compFuncPSOArgmaxIndicesToSet->maxTotalThreadsPerThreadgroup()));
    m_compEncoder->dispatchThreads({reducedSize, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(bufShape);
    freeTemporaryBuffer(bufStrides);
    commitBatchQueue();
    deallocate(winningCoords.data);
}

void DeviceMetal::sliceSet(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                           size_t dim, size_t start, size_t end, size_t step)
{
    validateDataType(src.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::sliceSet() result must have GPU memory.");

    auto newShape = dst.shape;
    newShape[dim] = (end - start + step - 1) / step; // This computes the size along the slicing dimension.

    // For a special case, two scalar tensors, use just a copy operation.
    if (dst.shape.empty() && newShape.empty())
    {
        copy(src.data, src.dtype, dst.data, dst.dtype, src.size);
        return;
    }

    assert(src.size > 0);

    auto bufSrc     = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufDst     = m_allocMap[dst.data];
    auto compFuncPSO = m_compFuncPSOSliceSet[static_cast<size_t>(src.dtype)];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufSrc,     0, 0);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    auto boundSrcLayout = bindTensorLayout(src, {.bufferIndex=2});
    auto boundDstLayout = bindTensorLayout(dst, {.bufferIndex=3});
    m_compEncoder->setBytes(&dim,   sizeof(dim),   4);
    m_compEncoder->setBytes(&start, sizeof(start), 5);
    m_compEncoder->setBytes(&step,  sizeof(step),  6);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(src.size, compFuncPSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({src.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufSrc);
    releaseTensorLayout(boundSrcLayout);
    releaseTensorLayout(boundDstLayout);
    commitBatchQueue();
}

void DeviceMetal::tril(const DeviceTensorParams& dst, ssize_t diagonal)
{
    assert(dst.isContiguous == true);
    validateDataType(dst.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::tril() result must have GPU memory.");

    size_t shapeSize   = dst.shape.size();
    size_t stridesSize = dst.strides.size();
    assert(dst.size > 0);

    // NOTE: For a scalar tensor shape size could be zero.
    auto bufShape   = shapeSize    != 0 ? getReadOnlyMTLBuffer(dst.shape.data(),    shapeSize,    sizeof(size_t)) : nullptr;
    auto bufStrides = stridesSize  != 0 ? getReadOnlyMTLBuffer(dst.strides.data(),  stridesSize,  sizeof(size_t)) : nullptr;
    auto bufDst     = m_allocMap[dst.data];
    auto compFuncPSO = m_compFuncPSOTril[static_cast<size_t>(dst.dtype)];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    m_compEncoder->setBuffer(bufShape,   0, 2);
    m_compEncoder->setBuffer(bufStrides, 0, 3);
    m_compEncoder->setBytes(&shapeSize,    sizeof(shapeSize),    4);
    m_compEncoder->setBytes(&stridesSize,  sizeof(stridesSize),  5);
    m_compEncoder->setBytes(&diagonal,     sizeof(diagonal),     6);
    m_compEncoder->setBytes(&dst.size,     sizeof(dst.size),     7);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(dst.size, compFuncPSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({dst.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufShape);
    freeTemporaryBuffer(bufStrides);
    commitBatchQueue();
}

void DeviceMetal::triu(const DeviceTensorParams& dst, ssize_t diagonal)
{
    assert(dst.isContiguous == true);
    validateDataType(dst.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::triu() result must have GPU memory.");

    size_t shapeSize   = dst.shape.size();
    size_t stridesSize = dst.strides.size();
    assert(dst.size > 0);

    // NOTE: For a scalar tensor shape size could be zero.
    auto bufShape   = shapeSize    != 0 ? getReadOnlyMTLBuffer(dst.shape.data(),    shapeSize,    sizeof(size_t)) : nullptr;
    auto bufStrides = stridesSize  != 0 ? getReadOnlyMTLBuffer(dst.strides.data(),  stridesSize,  sizeof(size_t)) : nullptr;
    auto bufDst     = m_allocMap[dst.data];
    auto compFuncPSO = m_compFuncPSOTriu[static_cast<size_t>(dst.dtype)];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    m_compEncoder->setBuffer(bufShape,   0, 2);
    m_compEncoder->setBuffer(bufStrides, 0, 3);
    m_compEncoder->setBytes(&shapeSize,    sizeof(shapeSize),    4);
    m_compEncoder->setBytes(&stridesSize,  sizeof(stridesSize),  5);
    m_compEncoder->setBytes(&diagonal,     sizeof(diagonal),     6);
    m_compEncoder->setBytes(&dst.size,     sizeof(dst.size),     7);

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(dst.size, compFuncPSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({dst.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufShape);
    freeTemporaryBuffer(bufStrides);
    commitBatchQueue();
}

void DeviceMetal::indexSelect(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                              const DeviceTensorParams& indices, size_t dim)
{
    assert(dst.isContiguous == true);
    validateDataType(src.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::indexSelect() result must have GPU memory.");

    // Calculate the number of elements in one slice after the specified dimension.
    size_t sliceSize = 1;
    for (size_t i = dim + 1; i < src.shape.size(); ++i)
    {
        sliceSize *= src.shape[i];
    }
    size_t dimSize = !src.shape.empty() ? src.shape[dim] * sliceSize : 0;   // Size of one entire slice for the dimension.

    auto bufSrc      = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufIndices  = getReadOnlyMTLBuffer(indices.data, indices.size, dataTypeSize(aix::DataType::kInt32));
    auto bufDst      = m_allocMap[dst.data];
    auto compFuncPSO = m_compFuncPSOIndexSelect[static_cast<size_t>(src.dtype)];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufSrc,     0, 0);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    m_compEncoder->setBuffer(bufIndices, 0, 2);
    m_compEncoder->setBytes(&indices.size, sizeof(size_t), 3);
    m_compEncoder->setBytes(&dimSize,      sizeof(size_t), 4);
    m_compEncoder->setBytes(&sliceSize,    sizeof(size_t), 5);
    auto boundSrcLayout = bindTensorLayout(src, {.bufferIndex=6});
    auto boundIndicesLayout = bindTensorLayout(indices, {.bufferIndex=7});

    NS::UInteger w = std::min(dst.size, compFuncPSO->maxTotalThreadsPerThreadgroup());

    m_compEncoder->dispatchThreads({dst.size, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(bufSrc);
    freeTemporaryBuffer(bufIndices);
    releaseTensorLayout(boundSrcLayout);
    releaseTensorLayout(boundIndicesLayout);
    commitBatchQueue();
}

void DeviceMetal::indexAdd(const DeviceTensorParams& src, const DeviceTensorParams& dst, const DeviceTensorParams& indices,
                           size_t dim)
{
    validateDataType(src.dtype);
    // NOTE: Only certain data types are supported due to limitation of Metal Framework atomics.
    if (!(src.dtype == DataType::kFloat32 || src.dtype == DataType::kInt32))
    {
        synchronize();
        defaultDevice.indexAdd(src, dst, indices, dim);
        return;
    }

    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::indexAdd() result must have GPU memory.");

    // Calculate the number of elements in one slice after the specified dimension.
    size_t sliceSize = 1;
    for (size_t i = dim + 1; i < dst.shape.size(); ++i)
    {
        sliceSize *= dst.shape[i];
    }
    size_t dimSize = !dst.shape.empty() ? dst.shape[dim] * sliceSize : 0;   // Size of one entire slice for the dimension.
    auto bufSrc      = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufIndices  = getReadOnlyMTLBuffer(indices.data, indices.size, dataTypeSize(aix::DataType::kInt32));
    auto bufDst      = m_allocMap[dst.data];
    auto compFuncPSO = m_compFuncPSOIndexAdd[static_cast<size_t>(src.dtype)];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(bufSrc,     0, 0);
    m_compEncoder->setBuffer(bufDst,     0, 1);
    m_compEncoder->setBuffer(bufIndices, 0, 2);
    m_compEncoder->setBytes(&indices.size, sizeof(size_t), 3);
    m_compEncoder->setBytes(&dimSize,      sizeof(size_t), 4);
    m_compEncoder->setBytes(&sliceSize,    sizeof(size_t), 5);
    auto boundSrcLayout = bindTensorLayout(src, {.bufferIndex=6});
    auto boundDstLayout = bindTensorLayout(dst, {.bufferIndex=7});
    auto boundIndicesLayout = bindTensorLayout(indices, {.bufferIndex=8});

    NS::UInteger w = std::min(src.size, compFuncPSO->maxTotalThreadsPerThreadgroup());

    m_compEncoder->dispatchThreads({src.size, 1, 1}, {w, 1, 1});

    freeTemporaryBuffer(bufSrc);
    freeTemporaryBuffer(bufIndices);
    releaseTensorLayout(boundSrcLayout);
    releaseTensorLayout(boundDstLayout);
    releaseTensorLayout(boundIndicesLayout);
    commitBatchQueue();
}

void DeviceMetal::emptyCache()
{
    m_bufferCache->clear();
    m_allocator->clearEmptyHeaps();
}

void DeviceMetal::commit()
{
    if (m_currentBatchSize == 0) return;

    if (m_committedCmdBuffer)
    {
        m_committedCmdBuffer->waitUntilCompleted();
    }
    m_compEncoder->endEncoding();
    m_cmdBuffer->addCompletedHandler([&,tempBuffers=m_tempBuffers](MTL::CommandBuffer* commandBuffer)
    {
        CheckCommandBufferStatus(commandBuffer);
        // We must recycle the buffers only after the current command buffer execution is completed since the buffers
        // could be in use.
        for (const auto& [buf, bufPtr] : tempBuffers)
        {
            m_bufferCache->recycle(buf);
        }
    });
    m_cmdBuffer->commit();                // Execute the command

    // No need to track the allocations anymore for the buffer used in the last commit.
    for (const auto& [buf, bufPtr] : m_tempBuffers)
    {
        m_allocMap.erase(bufPtr);
    }

    // Reduce the size of the MTL buffer cache if the cache size is bigger than the max allowed working set size.
    if (m_bufferCache->size() > m_maxWorkingSetSize)
    {
        m_bufferCache->reduceSize(m_bufferCache->size() - m_maxWorkingSetSize);
    }

    m_tempBuffers.clear();
    m_tempBuffers.reserve(MAX_CMD_BATCH_SIZE);

    m_committedCmdBuffer = m_cmdBuffer;
    // Create a new command buffer for the next batch.
    m_cmdBuffer = m_cmdQueue->commandBufferWithUnretainedReferences();
    m_compEncoder = m_cmdBuffer->computeCommandEncoder();

    // Update batch size metrics.
    m_maxBatchSize = std::max(m_currentBatchSize, m_maxBatchSize);
    m_currentBatchSize = 0;
    m_currentWorkingSetSize = 0;
}

void DeviceMetal::synchronize()
{
    commit();
    m_committedCmdBuffer->waitUntilCompleted();
}

void DeviceMetal::commitBatchQueue()
{
    if (++m_currentBatchSize >= MAX_CMD_BATCH_SIZE)
    {
        commit();
    }
}

MTL::Buffer* DeviceMetal::newBuffer(size_t size)
{
    assert(size > 0);
    size_t asize = size < vm_page_size ? align(size, ALLOCATION_BYTE_ALIGNMENT_SIZE) : align(size, vm_page_size);

    m_currentWorkingSetSize += asize;
    // Reduce memory footprint if the current working set size exceeds the limit.
    if (m_currentWorkingSetSize * 2 >= m_maxWorkingSetSize)
    {
        commit();
    }

    // Try to reuse a buffer from the MTL buffer cache if possible.
    auto buffer = m_bufferCache->reuse(asize);
    if (buffer)
    {
        return buffer;
    }

    // Allocate MTL buffer.
    buffer = m_allocator->alloc(asize);
    if (!buffer)
    {
        m_bufferCache->clear();
        std::cout << "Buffer's cache was cleared to create memory. "
                     "Consider increasing memory size to improve performance." << std::endl;
        buffer = m_allocator->alloc(asize);
        if (!buffer)
        {
            // Release empty heaps to satisfy the allocation request if possible.
            m_allocator->clearEmptyHeaps();
            std::cout << "Allocator's cache was cleared to create memory. "
                         "Consider increasing memory size to improve performance." << std::endl;
            if (!buffer)
            {
                throw std::runtime_error("GPU memory allocation has failed for size: " + std::to_string(size) + " bytes.");
            }
        }
    }
    assert(buffer);
    return buffer;
}


MTL::Buffer* DeviceMetal::getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType, size_t alignSize)
{
    // Memory could be from other devices. Create a temporary buffer for read only case.
    if (!isDeviceBuffer(address))
    {
        auto asize = align(size, alignSize);
        auto buff = newBuffer(asize * sizeofType);
        std::memcpy(buff->contents(), address, size * sizeofType);
        return buff;
    }

    return m_allocMap[address];    // Return MTL Buffer if the memory is from the current device.
}


void DeviceMetal::freeTemporaryBuffer(MTL::Buffer * buffer)
{
    // Release only temporary buffer.
    if (buffer && !isDeviceBuffer(buffer->contents()))
    {
        // Add the buffer to the list to be released when commit() is executed.
        // Until then, the buffer could be in use, especially when a batch command is used.
        m_tempBuffers.emplace_back(buffer, buffer->contents());
    }
}


void DeviceMetal::clearContiguousTemps()
{
    for (void* ptr : m_contiguousTempAllocations)
    {
        deallocate(ptr);
    }
    m_contiguousTempAllocations.clear();
}


MTL::Device* DeviceMetal::createMTLDevice(size_t deviceIndex) const
{
    try
    {
        return reinterpret_cast<MTL::Device*>(MTL::CopyAllDevices()->object(deviceIndex));
    }
    catch (...)
    {
        throw std::invalid_argument("Device index is not supported.");
    }

    return nullptr;
}


MTL::Library* DeviceMetal::createLibrary(const char* shaders)
{
    // Create a compile options.
    MTL::CompileOptions* compileOptions = MTL::CompileOptions::alloc()->init();
    compileOptions->setOptimizationLevel(MTL::LibraryOptimizationLevelDefault);
    compileOptions->setFastMathEnabled(false);

    NS::Error* error = nullptr;
    MTL::Library* defaultLibrary = m_mtlDevice->newLibrary(NS::String::string(shaders, NS::UTF8StringEncoding),
                                                           compileOptions, &error);
    compileOptions->release();

    if (!defaultLibrary)
    {
        std::cerr << "Failed to load default library. Details: " << error->localizedDescription()->utf8String() << "\n";
        exit(-1);
    }

    return defaultLibrary;
}

MTL::CommandQueue* DeviceMetal::createCommandQueue()
{
    auto cmdQueue = m_mtlDevice->newCommandQueue();
    if (!cmdQueue)
    {
        std::cerr << "Failed to create command queue.\n";
        exit(-1);
    }

    return cmdQueue;
}

MTL::ComputePipelineState* DeviceMetal::createComputeFuncPSO(MTL::Library* library, const std::string & kernelName)
{
    auto funcName = NS::String::string(kernelName.c_str(), NS::ASCIIStringEncoding);
    auto compFunc = library->newFunction(funcName);
    if (!compFunc)
    {
        std::cerr << "Failed to find the compute function.\n";
        // No need to halt the application here.
    }

    NS::Error* error = nullptr;
    auto compFuncPSO = m_mtlDevice->newComputePipelineState(compFunc, &error);
    if (!compFuncPSO)
    {
        std::cerr << "Failed to create the pipeline state object.\n";
        exit(-1);
    }

    return compFuncPSO;
}

void DeviceMetal::encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf, MTL::Buffer* bufResult,
                                                   const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                                   const MTL::Size& threadsPerTG) const
{
    // Encode the pipeline state object and its parameters.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(buf, 0, 0);
    m_compEncoder->setBuffer(bufResult, 0, 1);
    m_compEncoder->dispatchThreads(gridSize, threadsPerTG);
}

void DeviceMetal::encodeComputeCommandTripleBuffer(const MTL::Buffer* buf1, const MTL::Buffer* buf2, MTL::Buffer* bufResult,
                                                   const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                                   const MTL::Size& threadsPerTG) const
{
    // Encode the pipeline state object and its parameters.
    m_compEncoder->setComputePipelineState(compFuncPSO);
    m_compEncoder->setBuffer(buf1, 0, 0);
    m_compEncoder->setBuffer(buf2, 0, 1);
    m_compEncoder->setBuffer(bufResult, 0, 2);
    m_compEncoder->dispatchThreads(gridSize, threadsPerTG);
}

DeviceMetal::MetalBoundLayout DeviceMetal::bindTensorLayout(const DeviceTensorParams& params,
                                                            const MetalLayoutBindingSlots& slots)
{
    assert(params.shape.size() == params.strides.size());

    MetalBoundLayout layout;
    const size_t rank = params.shape.size();

    auto entryCount = 2 + rank + rank;
    layout.data.resize(entryCount);
    auto layoutData = layout.data.data();
    layoutData[0] = params.offset;
    layoutData[1] = rank;
    for (size_t i = 0; i < rank; ++i)
    {
        layoutData[2 + i] = params.shape[i];
        layoutData[2 + rank + i] = params.strides[i];
    }

    m_compEncoder->setBytes(layoutData, entryCount * sizeof(size_t), slots.bufferIndex);

    return layout;
}

void DeviceMetal::releaseTensorLayout(const MetalBoundLayout& layout)
{
    (void) layout;
}

void DeviceMetal::executeDoubleArrayCmd(const DeviceTensorParams& a, const DeviceTensorParams& result,
                                        const MTL::ComputePipelineState* contiguousPSO,
                                        const MTL::ComputePipelineState* stridedPSO, const std::string & cmdName)
{
    assert(result.isContiguous == true);
    validateDataType(result.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf = getReadOnlyMTLBuffer(a.data, a.size, dataTypeSize(a.dtype));
    auto bufResult = m_allocMap[result.data];

    if (a.isContiguous && a.offset == 0)
    {
        auto asize = align(a.size, TOTAL_COMPONENT_COUNT) / TOTAL_COMPONENT_COUNT;
        NS::UInteger w = std::min(asize, contiguousPSO->maxTotalThreadsPerThreadgroup());
        encodeComputeCommandDoubleBuffer(buf, bufResult, contiguousPSO, {asize, 1, 1}, {w, 1, 1});
    }
    else
    {
        m_compEncoder->setComputePipelineState(stridedPSO);
        m_compEncoder->setBuffer(buf, 0, 0);
        m_compEncoder->setBuffer(bufResult, 0, 1);
        auto boundLayout = bindTensorLayout(a, {.bufferIndex=2});
        NS::UInteger w = std::min(result.size, stridedPSO->maxTotalThreadsPerThreadgroup());
        m_compEncoder->dispatchThreads({result.size, 1, 1}, {w, 1, 1});
        releaseTensorLayout(boundLayout);
    }

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(buf);
    commitBatchQueue();
}

void DeviceMetal::executeTripleArrayCmd(const DeviceTensorParams& a1, const DeviceTensorParams& a2,
                                        const DeviceTensorParams& result, const MTL::ComputePipelineState* contiguousPSO,
                                        const MTL::ComputePipelineState* stridedPSO, const std::string & cmdName)
{
    assert(result.isContiguous == true);
    validateDataType(result.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::" + cmdName + "() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(a1.data, a1.size, dataTypeSize(a1.dtype));
    auto buf2 = getReadOnlyMTLBuffer(a2.data, a2.size, dataTypeSize(a2.dtype));
    auto bufResult = m_allocMap[result.data];

    if (a1.isContiguous && a1.offset == 0 && a2.isContiguous && a2.offset == 0)
    {
        auto asize = align(a1.size, TOTAL_COMPONENT_COUNT) / TOTAL_COMPONENT_COUNT;
        NS::UInteger w = std::min(asize, contiguousPSO->maxTotalThreadsPerThreadgroup());
        encodeComputeCommandTripleBuffer(buf1, buf2, bufResult, contiguousPSO, {asize, 1, 1}, {w, 1, 1});
    }
    else
    {
        m_compEncoder->setComputePipelineState(stridedPSO);
        m_compEncoder->setBuffer(buf1, 0, 0);
        m_compEncoder->setBuffer(buf2, 0, 1);
        m_compEncoder->setBuffer(bufResult, 0, 2);
        auto boundLayout1 = bindTensorLayout(a1, {.bufferIndex=3});
        auto boundLayout2 = bindTensorLayout(a2, {.bufferIndex=4});
        NS::UInteger w = std::min(result.size, stridedPSO->maxTotalThreadsPerThreadgroup());
        m_compEncoder->dispatchThreads({result.size, 1, 1}, {w, 1, 1});
        releaseTensorLayout(boundLayout1);
        releaseTensorLayout(boundLayout2);
    }

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(buf1);
    freeTemporaryBuffer(buf2);
    commitBatchQueue();
}

void DeviceMetal::translation(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                              const MTL::ComputePipelineState* computePSO, const std::string & name)
{
    validateDataType(src.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(dst.data))
        throw std::invalid_argument("DeviceMetal::" + name + "() result must have GPU memory.");

    // For a special case, two scalar tensors, use just a copy operation.
    if (src.shape.empty() && dst.shape.empty())
    {
        copy(src.data, src.dtype, dst.data, dst.dtype, src.size);
        return;
    }

    size_t newShapeSize = dst.shape.size();
    assert(src.size > 0);

    // NOTE: For a scalar tensor shape size could be zero.
    auto bufSrc = getReadOnlyMTLBuffer(src.data, src.size, dataTypeSize(src.dtype));
    auto bufShape = newShapeSize != 0 ? getReadOnlyMTLBuffer(dst.shape.data(), newShapeSize, sizeof(size_t)) : nullptr;
    auto bufDst = m_allocMap[dst.data];

    // Serialize resources and states to be used by the GPU.
    m_compEncoder->setComputePipelineState(computePSO);
    m_compEncoder->setBuffer(bufSrc, 0, 0);
    m_compEncoder->setBuffer(bufDst, 0, 1);
    m_compEncoder->setBuffer(bufShape, 0, 2);
    m_compEncoder->setBytes(&newShapeSize, sizeof(newShapeSize), 3);
    auto boundLayout = bindTensorLayout(src, {.bufferIndex=4});

    // Calculate maximum thread group dimensions
    NS::UInteger w = std::min(src.size, computePSO->maxTotalThreadsPerThreadgroup());

    // Use dispatch threads which is the most efficient but requires non-uniform grid size feature support in HW.
    m_compEncoder->dispatchThreads({src.size, 1, 1}, {w, 1, 1});

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(bufSrc);
    freeTemporaryBuffer(bufShape);
    releaseTensorLayout(boundLayout);
    commitBatchQueue();
}

void DeviceMetal::transpose2D(const DeviceTensorParams& mat, const DeviceTensorParams& result)
{
    assert(mat.isContiguous == result.isContiguous == true);
    validateDataType(result.dtype);
    auto iDType = static_cast<size_t>(result.dtype);
    // Result buffer has to be allocated in advance and has to be a GPU memory.
    if (!isDeviceBuffer(result.data))
        throw std::invalid_argument("DeviceMetal::transpose2D() result must have GPU memory.");

    // Memory could be a GPU allocated memory or system memory.
    auto buf1 = getReadOnlyMTLBuffer(mat.data, mat.shape[0] * mat.shape[1], dataTypeSize(mat.dtype));
    auto bufResult = m_allocMap[result.data];
    auto buf1Size = MatrixSize{mat.shape[0], mat.shape[1]};
    auto compFuncPSO = m_compFuncPSOTranspose2D[iDType];

    size_t M = buf1Size.rows;
    size_t N = buf1Size.cols;

    auto encodeParams = [&](const MTL::ComputePipelineState* compFuncPSO)
    {
        m_compEncoder->setComputePipelineState(compFuncPSO);
        m_compEncoder->setBuffer(buf1, 0, 0);
        m_compEncoder->setBuffer(bufResult, 0, 1);
        m_compEncoder->setBytes(&buf1Size, sizeof(MatrixSize), 2);
    };

    auto dispatchTiled = [&](const MTL::ComputePipelineState* compFuncPSO, const size_t tileSize, const size_t batchSize)
    {
        // Encode the pipeline state object and its parameters.
        uint numThreadgroupsX = (N + tileSize - 1) / tileSize;
        uint numThreadgroupsY = (M + tileSize - 1) / tileSize;
        assert(tileSize * batchSize <= compFuncPSO->maxTotalThreadsPerThreadgroup());
        encodeParams(compFuncPSO);
        m_compEncoder->dispatchThreadgroups({numThreadgroupsX, numThreadgroupsY, 1}, {tileSize, batchSize, 1});
    };

    if (M % 32 == 0 && N % 32 == 0)
    {
        dispatchTiled(m_compFuncPSOTranspose2DTiled32x32x8[iDType], 32, 8);
    }
    else if (M % 16 == 0 && N % 16 == 0)
    {
        dispatchTiled(m_compFuncPSOTranspose2DTiled16x16x8[iDType], 16, 8);
    }
    else
    {
        NS::UInteger w = compFuncPSO->threadExecutionWidth();
        NS::UInteger h = compFuncPSO->maxTotalThreadsPerThreadgroup() / w;
        encodeParams(compFuncPSO);
        m_compEncoder->dispatchThreads({mat.shape[0], mat.shape[1], 1}, {w, h, 1});
    }

    // Free operation is delayed until the commit is done.
    freeTemporaryBuffer(buf1);
    commitBatchQueue();
}

const std::string& DeviceMetal::toString(size_t dtypeIndex)
{
    assert(dtypeIndex < aix::DataTypeCount);
    static std::string formatStrTable[aix::DataTypeCount] =
    {
        "f64",
        "f32",
        "f16",
        "bf16",
        "i64",
        "i32",
        "i16",
        "i8",
        "ui8",
    };
    return formatStrTable[dtypeIndex];
}

const std::string& DeviceMetal::toString(DataType dtype)
{
    return toString(static_cast<size_t>(dtype));
}

void DeviceMetal::validateDataType(DataType dtype)
{
    if (dtype == aix::DataType::kFloat64)
    {
        throw std::invalid_argument("Apple Metal Framework does not support Float64 data type.");
    }
}

void DeviceMetal::CheckCommandBufferStatus(const MTL::CommandBuffer *commandBuffer)
{
    if (commandBuffer->status() == MTL::CommandBufferStatusError)
    {
        auto error = commandBuffer->error();
        if (error)
        {
            std::string errorMsg = "Command buffer execution failed due to ";
            switch (error->code())
            {
                case MTL::CommandBufferError::CommandBufferErrorOutOfMemory:
                    errorMsg += "insufficient memory.";
                    break;
                case MTL::CommandBufferError::CommandBufferErrorTimeout:
                    errorMsg += "timeout.";
                    break;
                case MTL::CommandBufferError::CommandBufferErrorStackOverflow:
                    errorMsg += "stack overflow.";
                    break;
                case MTL::CommandBufferError::CommandBufferErrorInvalidResource:
                    errorMsg += "invalid resource.";
                    break;
                case MTL::CommandBufferError::CommandBufferErrorPageFault:
                    errorMsg += "page fault.";
                    break;
                default:
                    errorMsg = "Command buffer execution failed with error code: " + std::to_string(error->code());
                    break;
            }
            std::cerr << errorMsg << std::endl;
        }
    }
}

}   // namespace
