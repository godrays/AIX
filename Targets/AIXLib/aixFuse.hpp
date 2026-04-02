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
// External includes
// System includes
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>


namespace aix::fuse
{

enum class OpType
{
    Add,
    Sub,
    Mul,
    Div,
    Negate,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Log,
    Exp,
    Pow,
    Fill,
    FillMin,
    Cast,
    Sum,
    Max,
    Argmax,
    ArgmaxIndices,
    Matmul,
    Transpose,
    Copy,
    Contiguous,
    ReduceTo,
    MaxTo,
    ArgmaxTo,
    ArgmaxIndicesTo,
    SliceSet,
    Tril,
    Triu,
    IndexSelect,
    IndexAdd,
};

inline bool isElementwiseOp(OpType t)
{
    switch (t)
    {
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Negate:
        case OpType::Sqrt:
        case OpType::Sin:
        case OpType::Cos:
        case OpType::Tanh:
        case OpType::Log:
        case OpType::Exp:
        case OpType::Pow:
        case OpType::Fill:
        case OpType::FillMin:
        case OpType::Cast:
            return true;
        default:
            return false;
    }
}

struct FuseConfig
{
    bool   elementwiseFusion      = true;
    bool   multiOutputKernels     = true;
    bool   absorbFills            = true;
    bool   deadResultElimination  = true;
    bool   diagnostics            = false;
    size_t flushThreshold         = 200;
};

struct OpRecord
{
    OpType               type;
    DeviceTensorParams   input0;
    DeviceTensorParams   input1;
    DeviceTensorParams   output;
    std::vector<uint8_t> scalarData;
    DataType             scalarDType = DataType::kFloat32;
    bool                 hasInput1   = false;

    OpRecord() = default;

    OpRecord(OpType t, const DeviceTensorParams& in0, const DeviceTensorParams& out)
        : type(t), input0(in0), output(out) {}

    OpRecord(OpType t, const DeviceTensorParams& in0, const DeviceTensorParams& in1, const DeviceTensorParams& out)
        : type(t), input0(in0), input1(in1), output(out), hasInput1(true) {}

    OpRecord(OpType t, const std::vector<uint8_t>& scalar, DataType sDType, const DeviceTensorParams& out)
        : type(t), output(out), scalarData(scalar), scalarDType(sDType) {}
};

struct FusedSubgraphDescriptor
{
    struct Op
    {
        OpType type;
        size_t inputIndex0  = SIZE_MAX;
        size_t inputIndex1  = SIZE_MAX;
        size_t outputIndex  = SIZE_MAX;
        size_t scalarIndex0 = SIZE_MAX;
        size_t scalarIndex1 = SIZE_MAX;
    };

    std::vector<Op> ops;
    std::vector<DeviceTensorParams> inputBuffers;
    std::vector<DeviceTensorParams> outputBuffers;
    std::vector<uint8_t> scalarData;
    DataType dtype;
    bool     allContiguous;
    size_t   elementCount;
};

struct FlushDiagnostics
{
    size_t opsRecorded = 0;
    size_t opsAfterDeadElim = 0;
    size_t fusibleSubgraphs = 0;
    size_t fusedOps = 0;
    size_t fallbackOps = 0;
    size_t dispatchesSaved = 0;
    size_t fillsAbsorbed = 0;
    size_t deadResultsEliminated = 0;
    size_t kernelCacheHits = 0;
    size_t kernelCacheMisses = 0;
    std::string subgraphSummary;
    std::string fallbackSummary;
};

class FuseEmitter
{
public:
    virtual ~FuseEmitter() = default;

    virtual void emitFused(const FusedSubgraphDescriptor&) = 0;
    virtual void emitSingle(const OpRecord&) = 0;
    virtual void commitCommandBuffer() = 0;
    virtual std::pair<size_t, size_t> getKernelCacheStats() const { return {0, 0}; }
};

class FuseEngine
{
public:
    FuseEngine(const FuseConfig& config, FuseEmitter& emitter)
        : m_config(config), m_emitter(emitter) {}

    void record(OpType type, const DeviceTensorParams& input0, const DeviceTensorParams& output);
    void record(OpType type, const DeviceTensorParams& input0, const DeviceTensorParams& input1,
                const DeviceTensorParams& output);
    void recordFill(const void* scalar, DataType scalarDType, const DeviceTensorParams& output);
    void recordFillMin(const DeviceTensorParams& output);
    void flush();

    const FuseConfig& config() const { return m_config; }

    const FlushDiagnostics& lastFlushDiagnostics() const
    {
        static const FlushDiagnostics empty{};
        if (m_diagnostics.has_value()) return *m_diagnostics;
        return empty;
    }

    void retainBuffer(void* buffer);
    void releaseBuffer(void* buffer);
    void invalidateBuffer(void* buffer);

private:
    FuseConfig                                m_config;
    FuseEmitter&                              m_emitter;
    std::vector<OpRecord>                     m_pendingOps;
    std::unordered_set<void*>                 m_liveBuffers;
    std::unordered_set<void*>                 m_externalLiveBuffers;
    std::unordered_set<void*>                 m_absorbedFillOutputs;
    std::unordered_map<void*, OpRecord>       m_deferredFills;
    std::unordered_set<void*>                 m_emittedFillBuffers;
    std::optional<FlushDiagnostics>           m_diagnostics;
};

}  // namespace aix::fuse
