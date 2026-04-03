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
#include "aixCore.hpp"
#include "aixDeviceType.hpp"
#include "aixFuse.hpp"
#include "aixDeviceMetalKernelGen.hpp"
// External includes
// System includes
#include <cstdint>
#include <cstddef>
#include <mach/vm_page_size.h>


// Forward declarations
namespace NS
{
    class AutoreleasePool;
}

// Forward declarations
namespace MTL
{
    class Buffer;
    class ComputePipelineState;
    class CommandQueue;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class Device;
    class Event;
    class Library;
    struct Size;
}

namespace aix::metal
{

#define ALLOCATOR_ALIGNMENT_SIZE            256
#define MAX_CMD_BATCH_SIZE                  50
#define MAX_THREADS_PER_THREADGROUP         1024
#define ALLOCATION_BYTE_ALIGNMENT_SIZE      32      // Should be power of two and min 32 bytes.
#define VECTOR_TYPE_COMPONENT_COUNT         4       // i.e. float4 has 4 components.
#define BATCH_PROCESS_SIZE_PER_THREAD       1       // i.e. each GPU thread will access/process 16 of float4 per dispatch.
#define TOTAL_COMPONENT_COUNT               (BATCH_PROCESS_SIZE_PER_THREAD * VECTOR_TYPE_COMPONENT_COUNT)

class MetalAllocator;
class MTLBufferCache;
class MetalFuseEmitter;
class aixDeviceMetalKernelGen;

class DeviceMetal : public aix::Device
{
public:
    friend class MetalFuseEmitter;

    // Constructor
    explicit DeviceMetal(size_t deviceIndex = 0);

    // Destructor
    ~DeviceMetal() override;

    DeviceType type() const override { return DeviceType::kGPU_METAL; }
    std::string name() const override { return "MCS"; }             // MCS = Metal Custom/Compute Shaders

    // Allocate GPU memory and return MTL Buffer contents and keeps MTL Buffer pointers in a hashmap.
    void* allocate(size_t size) override;

    void* allocate(size_t size, DataType dtype) override;

    // Deallocate GPU memory if it's allocated by current device.
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

    void indexSelect(const DeviceTensorParams& src, const DeviceTensorParams& dst, const DeviceTensorParams& indices,
                     size_t dim) override;

    void indexAdd(const DeviceTensorParams& src, const DeviceTensorParams& dst, const DeviceTensorParams& indices,
                  size_t dim) override;

    void emptyCache() override;

    void synchronize() override;

    std::pair<size_t, size_t> fuseKernelCacheStats() const;

protected:
    struct MetalLayoutBindingSlots
    {
        uint32_t bufferIndex;
    };

    struct MetalBoundLayout
    {
        std::vector<size_t> data{};
    };

    void commit();
    void commitBatchQueue();
    void flushPendingFusedWork();
    void recordExternalRead(const void* buffer);
    void recordExternalRead(const DeviceTensorParams& params);

    inline static void validateDataType(DataType dtype);

    struct MatrixSize
    {
        size_t rows;
        size_t cols;
    };

    static size_t align(size_t size, size_t alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);       // Padding for alignment.
    }

    inline bool isDeviceBuffer(const void* bufPtr)
    {
        return m_allocMap.find(bufPtr) != m_allocMap.end();
    }

    MTL::Buffer* newBuffer(size_t size);

    MTL::Buffer* getReadOnlyMTLBuffer(const void * address, size_t size, size_t sizeofType,
                                      size_t alignSize = TOTAL_COMPONENT_COUNT);

    void freeTemporaryBuffer(MTL::Buffer * buffer);

    MTL::Device* createMTLDevice(size_t deviceIndex) const;

    MTL::Library* createLibrary(const char* shaders);

    MTL::CommandQueue* createCommandQueue();

    MTL::ComputePipelineState* createComputeFuncPSO(MTL::Library* library, const std::string & kernelName);

    void encodeComputeCommandDoubleBuffer(const MTL::Buffer* buf, MTL::Buffer* bufResult,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    void encodeComputeCommandTripleBuffer(const MTL::Buffer* buf1, const MTL::Buffer* buf2, MTL::Buffer* bufResult,
                                          const MTL::ComputePipelineState* compFuncPSO, const MTL::Size& gridSize,
                                          const MTL::Size& threadsPerTG) const;

    MetalBoundLayout bindTensorLayout(const DeviceTensorParams& params, const MetalLayoutBindingSlots& slots);

    void releaseTensorLayout(const MetalBoundLayout& layout);

    void clearContiguousTemps();

    void executeDoubleArrayCmd(const DeviceTensorParams& a, const DeviceTensorParams& result,
                               const MTL::ComputePipelineState* contiguousPSO,
                               const MTL::ComputePipelineState* stridedPSO, const std::string & cmdName);

    void executeTripleArrayCmd(const DeviceTensorParams& a1, const DeviceTensorParams& a2,
                               const DeviceTensorParams& result, const MTL::ComputePipelineState* contiguousPSO,
                               const MTL::ComputePipelineState* stridedPSO, const std::string & cmdName);

    void translation(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                     const MTL::ComputePipelineState *computePSO, const std::string & name);

    void transpose2D(const DeviceTensorParams& mat, const DeviceTensorParams& result);

    static const std::string& toString(size_t dtype);
    inline static const std::string& toString(DataType dtype);

    static void CheckCommandBufferStatus(const MTL::CommandBuffer* commandBuffer);

    NS::AutoreleasePool*   m_pool{nullptr};
    MTL::Device*           m_mtlDevice{nullptr};
    MTL::CommandQueue*     m_cmdQueue{nullptr};
    MTL::CommandBuffer*    m_cmdBuffer{nullptr};
    MTL::CommandBuffer*    m_committedCmdBuffer{nullptr};
    MTL::ComputeCommandEncoder*  m_compEncoder{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAdd[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOAddStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSub[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSubStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMul[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMulStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODiv[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSODivStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOUnary[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOUnaryStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSqrt[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSqrtStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSin[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSinStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCos[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCosStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTanh[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTanhStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOLog[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOLogStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOExp[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOExpStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOPow[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOPowStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSum[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMax[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOArgmaxInit[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOArgmaxReduce[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOArgmaxTo[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOArgmaxIndicesSet{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOArgmaxIndicesToSet{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulStrided[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulTiledBC6464888[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulTiled32x32[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulTiled32x64[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMatMulTiled32x128[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTranspose2D[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTranspose2DTiled16x16x8[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTranspose2DTiled32x32x8[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTranspose[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOCopyAA[aix::DataTypeCount][aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOFill[aix::DataTypeCount][aix::DataTypeCount];
    MTL::ComputePipelineState*   m_compFuncPSOFillMin[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOContiguous[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOReduceTo[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOMaxTo[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOSliceSet[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTril[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOTriu[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOIndexSelect[aix::DataTypeCount]{nullptr};
    MTL::ComputePipelineState*   m_compFuncPSOIndexAdd[aix::DataTypeCount]{nullptr};
    std::vector<std::pair<MTL::Buffer*, void*>>    m_tempBuffers;
    std::vector<void*>                              m_contiguousTempAllocations;
    std::unordered_map<const void*, MTL::Buffer*>  m_allocMap;
    std::unique_ptr<MetalAllocator>  m_allocator;
    std::unique_ptr<MTLBufferCache>  m_bufferCache;
    std::unique_ptr<MetalFuseEmitter>  m_fuseEmitter;
    std::unique_ptr<aix::fuse::FuseEngine>  m_fuseEngine;
    std::unique_ptr<aixDeviceMetalKernelGen>  m_kernelGen;
    size_t   m_currentBatchSize{0};
    size_t   m_maxBatchSize{0};
    size_t   m_maxWorkingSetSize{0};
    size_t   m_currentWorkingSetSize{0};
    MTL::Event*  m_event{nullptr};
    uint64_t     m_eventValue{0};
    MTL::Buffer* m_sentinelBuffer{nullptr};
};

class MetalFuseEmitter : public aix::fuse::FuseEmitter
{
public:
    explicit MetalFuseEmitter(DeviceMetal* device) : m_device(device) {}

    void emitFused(const aix::fuse::FusedSubgraphDescriptor& subgraph) override;
    void emitSingle(const aix::fuse::OpRecord& op) override;
    void finishFlush() override;
    std::pair<size_t, size_t> getKernelCacheStats() const override;

private:
    DeviceMetal* m_device;
};

}   // namespace
