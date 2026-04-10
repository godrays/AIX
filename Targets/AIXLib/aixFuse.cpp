//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "aixFuse.hpp"
// External includes
// System includes
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>


namespace aix::fuse
{

namespace
{

size_t requiredFusedBufferSlots(size_t materializedBufferCount, bool allContiguous, bool hasRuntimeScalars)
{
    size_t requiredBufferSlots = materializedBufferCount;
    if (!allContiguous) requiredBufferSlots += materializedBufferCount;
    if (hasRuntimeScalars) requiredBufferSlots += 1;
    return requiredBufferSlots;
}

struct BufferBindingKey
{
    const void*         data{nullptr};
    aix::DataType       dtype{aix::DataType::kFloat32};
    size_t              offset{0};

    bool operator==(const BufferBindingKey& other) const
    {
        return data == other.data && dtype == other.dtype && offset == other.offset;
    }
};

struct BufferBindingKeyHash
{
    size_t operator()(const BufferBindingKey& key) const
    {
        size_t hash = std::hash<const void*>{}(key.data);
        auto combine = [&](size_t value) {
            hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };

        combine(static_cast<size_t>(key.dtype));
        combine(key.offset);
        return hash;
    }
};

BufferBindingKey makeBufferBindingKey(const DeviceTensorParams& params)
{
    return {
        .data = params.data,
        .dtype = params.dtype,
        .offset = params.offset,
    };
}

struct DagNode
{
    size_t              opIndex{0};
    size_t              input0Producer{SIZE_MAX};
    size_t              input1Producer{SIZE_MAX};
    std::vector<size_t> deps;
    std::vector<size_t> consumers;
};

struct Dag
{
    std::vector<DagNode> nodes;
};

Dag buildDag(const std::vector<OpRecord>& ops)
{
    Dag dag;
    dag.nodes.resize(ops.size());
    std::unordered_map<BufferBindingKey, size_t, BufferBindingKeyHash> latestWriter;

    for (size_t i = 0; i < ops.size(); ++i)
    {
        const auto& op = ops[i];
        auto& node = dag.nodes[i];
        node.opIndex = i;

        auto addDependency = [&](size_t dep) {
            if (dep == SIZE_MAX) return;
            if (std::find(node.deps.begin(), node.deps.end(), dep) != node.deps.end()) return;
            node.deps.push_back(dep);
            dag.nodes[dep].consumers.push_back(i);
        };

        if (op.input0.data != nullptr)
        {
            auto input0Key = makeBufferBindingKey(op.input0);
            auto it = latestWriter.find(input0Key);
            if (it != latestWriter.end())
            {
                node.input0Producer = it->second;
                addDependency(it->second);
            }
        }

        if (op.hasInput1 && op.input1.data != nullptr)
        {
            auto input1Key = makeBufferBindingKey(op.input1);
            auto it = latestWriter.find(input1Key);
            if (it != latestWriter.end())
            {
                node.input1Producer = it->second;
                addDependency(it->second);
            }
        }

        if (op.output.data != nullptr)
        {
            latestWriter[makeBufferBindingKey(op.output)] = i;
        }
    }

    return dag;
}

struct FusionResult
{
    std::vector<FusedSubgraphDescriptor> subgraphs;
    std::vector<size_t>                  fallbackIndices;
    std::vector<OpRecord>                nonInPlaceAbsorbedFills;
};

enum class ScalarKind
{
    kFloat,
    kSigned,
    kUnsigned,
};

struct DecodedScalar
{
    ScalarKind  kind{ScalarKind::kFloat};
    long double floatValue{0.0L};
    int64_t     signedValue{0};
    uint64_t    unsignedValue{0};
};

DecodedScalar decodeScalar(const std::vector<uint8_t>& scalarData, aix::DataType dtype)
{
    DecodedScalar decoded;
    if (scalarData.empty()) return decoded;

    switch (dtype)
    {
        case aix::DataType::kFloat64:
        {
            double value = 0.0;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(double), scalarData.size()));
            decoded.floatValue = value;
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kFloat32:
        {
            float value = 0.0f;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(float), scalarData.size()));
            decoded.floatValue = value;
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kFloat16:
        {
            aix::float16_t value = 0.0f;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(aix::float16_t), scalarData.size()));
            decoded.floatValue = static_cast<float>(value);
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kBFloat16:
        {
            aix::bfloat16_t value = 0.0f;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(aix::bfloat16_t), scalarData.size()));
            decoded.floatValue = static_cast<float>(value);
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kInt64:
            std::memcpy(&decoded.signedValue, scalarData.data(), std::min(sizeof(int64_t), scalarData.size()));
            decoded.kind = ScalarKind::kSigned;
            break;
        case aix::DataType::kInt32:
        {
            int32_t value = 0;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(int32_t), scalarData.size()));
            decoded.signedValue = value;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kInt16:
        {
            int16_t value = 0;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(int16_t), scalarData.size()));
            decoded.signedValue = value;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kInt8:
        {
            int8_t value = 0;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(int8_t), scalarData.size()));
            decoded.signedValue = value;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kUInt8:
        {
            uint8_t value = 0;
            std::memcpy(&value, scalarData.data(), std::min(sizeof(uint8_t), scalarData.size()));
            decoded.unsignedValue = value;
            decoded.kind = ScalarKind::kUnsigned;
            break;
        }
        default:
            break;
    }

    return decoded;
}

template <typename T>
static T clampFloatToIntegral(long double value)
{
    if (std::isnan(value)) return 0;

    value = std::clamp(value, static_cast<long double>(std::numeric_limits<T>::lowest()),
                              static_cast<long double>(std::numeric_limits<T>::max()));
    return static_cast<T>(value);
}

template <typename DstType>
DstType convertDecodedScalar(const DecodedScalar& decoded)
{
    constexpr bool isFloatLikeDst = std::is_floating_point_v<DstType> ||
                                    std::is_same_v<DstType, aix::float16_t> ||
                                    std::is_same_v<DstType, aix::bfloat16_t>;

    if constexpr (isFloatLikeDst)
    {
        switch (decoded.kind)
        {
            case ScalarKind::kFloat:    return static_cast<DstType>(decoded.floatValue);
            case ScalarKind::kSigned:   return static_cast<DstType>(decoded.signedValue);
            case ScalarKind::kUnsigned: return static_cast<DstType>(decoded.unsignedValue);
        }
    }
    else if constexpr (std::is_signed_v<DstType>)
    {
        switch (decoded.kind)
        {
            case ScalarKind::kFloat:
                return clampFloatToIntegral<DstType>(decoded.floatValue);
            case ScalarKind::kSigned:
                return static_cast<DstType>(std::clamp(decoded.signedValue,
                                                       static_cast<int64_t>(std::numeric_limits<DstType>::lowest()),
                                                       static_cast<int64_t>(std::numeric_limits<DstType>::max())));
            case ScalarKind::kUnsigned:
                return decoded.unsignedValue > static_cast<uint64_t>(std::numeric_limits<DstType>::max())
                       ? std::numeric_limits<DstType>::max()
                       : static_cast<DstType>(decoded.unsignedValue);
        }
    }
    else
    {
        switch (decoded.kind)
        {
            case ScalarKind::kFloat:
                return clampFloatToIntegral<DstType>(decoded.floatValue);
            case ScalarKind::kSigned:
                return decoded.signedValue < 0
                       ? 0
                       : static_cast<DstType>(std::min<uint64_t>(static_cast<uint64_t>(decoded.signedValue),
                                                                 std::numeric_limits<DstType>::max()));
            case ScalarKind::kUnsigned:
                return static_cast<DstType>(std::min<uint64_t>(decoded.unsignedValue,
                                                               std::numeric_limits<DstType>::max()));
        }
    }

    return DstType{};
}

std::vector<uint8_t> convertScalarData(const std::vector<uint8_t>& scalarData,
                                       aix::DataType scalarDType,
                                       aix::DataType targetDType)
{
    if (scalarData.empty()) return {};

    const auto decoded = decodeScalar(scalarData, scalarDType);
    std::vector<uint8_t> converted(aix::Device::dataTypeSize(targetDType));

    auto store = [&](const auto& value)
    {
        std::memcpy(converted.data(), &value, sizeof(value));
    };

    switch (targetDType)
    {
        case aix::DataType::kFloat64:  store(convertDecodedScalar<double>(decoded)); break;
        case aix::DataType::kFloat32:  store(convertDecodedScalar<float>(decoded)); break;
        case aix::DataType::kFloat16:  store(convertDecodedScalar<aix::float16_t>(decoded)); break;
        case aix::DataType::kBFloat16: store(convertDecodedScalar<aix::bfloat16_t>(decoded)); break;
        case aix::DataType::kInt64:    store(convertDecodedScalar<int64_t>(decoded)); break;
        case aix::DataType::kInt32:    store(convertDecodedScalar<int32_t>(decoded)); break;
        case aix::DataType::kInt16:    store(convertDecodedScalar<int16_t>(decoded)); break;
        case aix::DataType::kInt8:     store(convertDecodedScalar<int8_t>(decoded)); break;
        case aix::DataType::kUInt8:    store(convertDecodedScalar<uint8_t>(decoded)); break;
        default:                       break;
    }

    return converted;
}

FusionResult analyzeFusion(const std::vector<OpRecord>& ops,
                           const Dag& dag,
                           const FuseConfig& config,
                           const std::unordered_set<void*>& liveBuffers)
{
    FusionResult result;
    std::vector<bool> visited(ops.size(), false);

    for (size_t i = 0; i < ops.size(); ++i)
    {
        if (visited[i]) continue;
        if (!isElementwiseOp(ops[i].type))
        {
            result.fallbackIndices.push_back(i);
            visited[i] = true;
            continue;
        }

        std::vector<size_t> subgraphOps;
        std::queue<size_t> worklist;
        worklist.push(i);
        bool valid = true;
        size_t elementCount = ops[i].output.size;
        DataType subgraphDtype = ops[i].output.dtype;
        bool allContiguous = (ops[i].output.isContiguous && ops[i].output.offset == 0);

        if (ops[i].input0.data && !ops[i].input0.isContiguous) allContiguous = false;
        if (ops[i].hasInput1 && ops[i].input1.data && !ops[i].input1.isContiguous) allContiguous = false;

        while (!worklist.empty() && valid)
        {
            size_t opIdx = worklist.front();
            worklist.pop();

            if (visited[opIdx]) continue;

            const auto& op = ops[opIdx];

            if (!isElementwiseOp(op.type)) { valid = false; break; }
            if (op.output.size != elementCount && op.type != OpType::Fill && op.type != OpType::FillMin)
            {
                if (elementCount == 1) elementCount = op.output.size;
                else { valid = false; break; }
            }

            if (op.type != OpType::Cast && op.output.dtype != subgraphDtype) { valid = false; break; }

            if (!op.output.isContiguous || op.output.offset != 0) allContiguous = false;
            if (op.input0.data && (!op.input0.isContiguous || op.input0.offset != 0)) allContiguous = false;
            if (op.hasInput1 && op.input1.data && (!op.input1.isContiguous || op.input1.offset != 0)) allContiguous = false;
            if (!config.supportsStridedFusion && !allContiguous) { valid = false; break; }

            subgraphOps.push_back(opIdx);
            visited[opIdx] = true;

            for (size_t dep : dag.nodes[opIdx].deps)
            {
                if (!visited[dep]) worklist.push(dep);
            }

            if (config.multiOutputKernels)
            {
                for (size_t consumer : dag.nodes[opIdx].consumers)
                {
                    if (!visited[consumer]) worklist.push(consumer);
                }
            }
            else
            {
                if (dag.nodes[opIdx].consumers.size() == 1)
                {
                    size_t consumer = dag.nodes[opIdx].consumers[0];
                    if (!visited[consumer]) worklist.push(consumer);
                }
            }
        }

        if (!valid || subgraphOps.empty())
        {
            for (size_t idx : subgraphOps)
            {
                if (idx != i) result.fallbackIndices.push_back(idx);
            }
            result.fallbackIndices.push_back(i);
            continue;
        }

        std::vector<size_t> sortedOps;
        std::set<size_t> subgraphSet(subgraphOps.begin(), subgraphOps.end());
        {
            std::queue<size_t> q;
            std::unordered_map<size_t, int> inDegree;
            for (size_t idx : subgraphOps)
            {
                inDegree[idx] = 0;
                for (size_t dep : dag.nodes[idx].deps)
                {
                    if (subgraphSet.contains(dep)) inDegree[idx]++;
                }
                if (inDegree[idx] == 0) q.push(idx);
            }
            while (!q.empty())
            {
                size_t idx = q.front(); q.pop();
                sortedOps.push_back(idx);
                for (size_t consumer : dag.nodes[idx].consumers)
                {
                    if (subgraphSet.contains(consumer))
                    {
                        inDegree[consumer]--;
                        if (inDegree[consumer] == 0) q.push(consumer);
                    }
                }
            }
        }

        if (sortedOps.size() != subgraphOps.size())
        {
            for (size_t idx : subgraphOps) result.fallbackIndices.push_back(idx);
            continue;
        }

        bool dependsOnOutsidePendingProducer = false;
        for (size_t idx : sortedOps)
        {
            size_t input0Producer = dag.nodes[idx].input0Producer;
            if (input0Producer != SIZE_MAX && !subgraphSet.contains(input0Producer))
            {
                dependsOnOutsidePendingProducer = true;
                break;
            }

            size_t input1Producer = dag.nodes[idx].input1Producer;
            if (input1Producer != SIZE_MAX && !subgraphSet.contains(input1Producer))
            {
                dependsOnOutsidePendingProducer = true;
                break;
            }
        }

        if (dependsOnOutsidePendingProducer)
        {
            for (size_t idx : sortedOps) result.fallbackIndices.push_back(idx);
            continue;
        }

        FusedSubgraphDescriptor desc;
        desc.elementCount = elementCount;
        desc.dtype = subgraphDtype;
        desc.allContiguous = allContiguous;

        std::unordered_map<BufferBindingKey, size_t, BufferBindingKeyHash> inputBufferIndex;
        std::unordered_map<BufferBindingKey, size_t, BufferBindingKeyHash> outputBufferIndex;
        std::unordered_map<size_t, size_t> opToDescriptorIndex;

        auto appendScalar = [&](const std::vector<uint8_t>& scalarData, DataType scalarDType) -> size_t {
            assert(!scalarData.empty());
            auto converted = convertScalarData(scalarData, scalarDType, desc.dtype);
            auto scalarSize = aix::Device::dataTypeSize(desc.dtype);
            assert(converted.size() == scalarSize);
            size_t scalarIndex = desc.scalarData.size() / scalarSize;
            desc.scalarData.insert(desc.scalarData.end(), converted.begin(), converted.end());
            return scalarIndex;
        };

        auto getOrCreateInputBuffer = [&](const DeviceTensorParams& params) -> size_t {
            if (params.data == nullptr) return SIZE_MAX;
            auto key = makeBufferBindingKey(params);
            auto it = inputBufferIndex.find(key);
            if (it != inputBufferIndex.end()) return it->second;
            size_t idx = desc.inputBuffers.size();
            desc.inputBuffers.push_back(params);
            inputBufferIndex[key] = idx;
            return idx;
        };

        auto getOrCreateOutputBuffer = [&](const DeviceTensorParams& params) -> size_t {
            auto key = makeBufferBindingKey(params);
            auto it = outputBufferIndex.find(key);
            if (it != outputBufferIndex.end()) return it->second;
            size_t idx = desc.outputBuffers.size();
            desc.outputBuffers.push_back(params);
            outputBufferIndex[key] = idx;
            return idx;
        };

        auto needsExternalInput = [&](size_t consumerIdx, bool useInput1) -> bool {
            const auto& input = useInput1 ? ops[consumerIdx].input1 : ops[consumerIdx].input0;
            if (input.data == nullptr) return false;
            size_t producerIdx = useInput1 ? dag.nodes[consumerIdx].input1Producer
                                           : dag.nodes[consumerIdx].input0Producer;
            return producerIdx == SIZE_MAX || !subgraphSet.contains(producerIdx);
        };

        auto getExistingInputBuffer = [&](const DeviceTensorParams& params) -> size_t {
            assert(params.data != nullptr);
            auto key = makeBufferBindingKey(params);
            auto it = inputBufferIndex.find(key);
            assert(it != inputBufferIndex.end());
            return it->second;
        };

        auto outputEscapesSubgraph = [&](size_t opIdx) -> bool {
            if (ops[opIdx].output.data && liveBuffers.contains(ops[opIdx].output.data)) return true;

            const auto& consumers = dag.nodes[opIdx].consumers;
            if (consumers.empty()) return true;

            for (size_t consumerIdx : consumers)
            {
                if (!subgraphSet.contains(consumerIdx)) return true;
            }

            return false;
        };

        std::unordered_set<size_t> absorbedFills;
        std::unordered_map<size_t, size_t> fillConsumer;
        if (config.absorbFills)
        {
            for (size_t idx : sortedOps)
            {
                if (ops[idx].type == OpType::Fill || ops[idx].type == OpType::FillMin)
                {
                    bool outputStillLive = ops[idx].output.data && liveBuffers.contains(ops[idx].output.data);
                    if (!outputStillLive && dag.nodes[idx].consumers.size() == 1)
                    {
                        size_t consumer = dag.nodes[idx].consumers[0];
                        if (subgraphSet.contains(consumer))
                        {
                            absorbedFills.insert(idx);
                            fillConsumer[idx] = consumer;
                            if (ops[idx].output.data != ops[consumer].output.data)
                            {
                                result.nonInPlaceAbsorbedFills.push_back(ops[idx]);
                            }
                        }
                    }
                }
            }
        }

        std::unordered_set<BufferBindingKey, BufferBindingKeyHash> estimatedInputBuffers;
        std::unordered_set<BufferBindingKey, BufferBindingKeyHash> estimatedOutputBuffers;
        bool hasRuntimeScalars = false;
        for (size_t idx : sortedOps)
        {
            const auto& op = ops[idx];
            if (needsExternalInput(idx, false))
            {
                estimatedInputBuffers.insert(makeBufferBindingKey(op.input0));
            }
            if (op.hasInput1 && needsExternalInput(idx, true))
            {
                estimatedInputBuffers.insert(makeBufferBindingKey(op.input1));
            }
            if (!absorbedFills.contains(idx) && op.output.data != nullptr && outputEscapesSubgraph(idx))
            {
                estimatedOutputBuffers.insert(makeBufferBindingKey(op.output));
            }
            if (op.type == OpType::Fill)
            {
                hasRuntimeScalars = true;
            }
        }

        size_t estimatedMaterializedBufferCount = estimatedInputBuffers.size() + estimatedOutputBuffers.size();
        size_t estimatedRequiredBufferSlots = requiredFusedBufferSlots(estimatedMaterializedBufferCount,
                                                                       allContiguous,
                                                                       hasRuntimeScalars);
        if (estimatedRequiredBufferSlots > config.maxBufferSlots)
        {
            for (size_t idx : sortedOps) result.fallbackIndices.push_back(idx);
            continue;
        }

        for (size_t idx : sortedOps)
        {
            const auto& op = ops[idx];
            if (needsExternalInput(idx, false))
            {
                getOrCreateInputBuffer(op.input0);
            }
            if (op.hasInput1 && needsExternalInput(idx, true))
            {
                getOrCreateInputBuffer(op.input1);
            }
        }

        size_t inputBufCount = desc.inputBuffers.size();

        for (size_t idx : sortedOps)
        {
            FusedSubgraphDescriptor::Op descOp;
            descOp.type = ops[idx].type;
            opToDescriptorIndex[idx] = desc.ops.size();

            const auto& op = ops[idx];

            if (op.type == OpType::Fill || op.type == OpType::FillMin)
            {
                descOp.inputIndex0 = SIZE_MAX;
                if (op.type == OpType::Fill)
                {
                    descOp.scalarIndex0 = appendScalar(op.scalarData, op.scalarDType);
                }
            }
            else
            {
                size_t producerIdx = dag.nodes[idx].input0Producer;
                if (producerIdx != SIZE_MAX && subgraphSet.contains(producerIdx))
                {
                    descOp.inputIndex0 = inputBufCount + opToDescriptorIndex[producerIdx];
                }
                else if (op.input0.data)
                {
                    descOp.inputIndex0 = getExistingInputBuffer(op.input0);
                }
                else
                {
                    descOp.inputIndex0 = SIZE_MAX;
                }
            }

            if (op.hasInput1 && op.input1.data)
            {
                size_t producerIdx = dag.nodes[idx].input1Producer;
                if (producerIdx != SIZE_MAX && subgraphSet.contains(producerIdx))
                {
                    descOp.inputIndex1 = inputBufCount + opToDescriptorIndex[producerIdx];
                }
                else
                {
                    descOp.inputIndex1 = getExistingInputBuffer(op.input1);
                }
            }
            else if (op.hasInput1)
            {
                descOp.inputIndex1 = SIZE_MAX;
            }

            if (absorbedFills.contains(idx))
            {
                descOp.outputIndex = SIZE_MAX;
            }
            else if (outputEscapesSubgraph(idx))
            {
                descOp.outputIndex = getOrCreateOutputBuffer(op.output);
            }
            else
            {
                descOp.outputIndex = SIZE_MAX;
            }

            desc.ops.push_back(descOp);
        }

        for (auto& [fillIdx, consumerIdx] : fillConsumer)
        {
            auto descIdx = opToDescriptorIndex[consumerIdx];
            auto& consumerOp = desc.ops[descIdx];
            const auto& fillOp = ops[fillIdx];
            auto fillKey = makeBufferBindingKey(fillOp.output);

            if (consumerOp.inputIndex0 != SIZE_MAX && makeBufferBindingKey(ops[consumerIdx].input0) == fillKey)
            {
                consumerOp.inputIndex0 = SIZE_MAX;
                consumerOp.scalarIndex0 = appendScalar(fillOp.scalarData, fillOp.scalarDType);
            }
            if (consumerOp.inputIndex1 != SIZE_MAX && makeBufferBindingKey(ops[consumerIdx].input1) == fillKey)
            {
                consumerOp.inputIndex1 = SIZE_MAX;
                consumerOp.scalarIndex1 = appendScalar(fillOp.scalarData, fillOp.scalarDType);
            }
        }

        if (!absorbedFills.empty())
        {
            std::vector<FusedSubgraphDescriptor::Op> finalOps;
            std::unordered_map<size_t, size_t> oldToNew;
            size_t newIdx = 0;
            for (size_t j = 0; j < sortedOps.size(); ++j)
            {
                if (!absorbedFills.contains(sortedOps[j]))
                {
                    oldToNew[j] = newIdx++;
                    finalOps.push_back(desc.ops[j]);
                }
            }

            size_t inputBufCount = desc.inputBuffers.size();
            for (auto& op : finalOps)
            {
                if (op.inputIndex0 != SIZE_MAX && op.inputIndex0 >= inputBufCount)
                {
                    size_t oldOpPos = op.inputIndex0 - inputBufCount;
                    auto remapIt = oldToNew.find(oldOpPos);
                    if (remapIt != oldToNew.end())
                    {
                        op.inputIndex0 = inputBufCount + remapIt->second;
                    }
                    else
                    {
                        op.inputIndex0 = SIZE_MAX;
                    }
                }
                if (op.inputIndex1 != SIZE_MAX && op.inputIndex1 >= inputBufCount)
                {
                    size_t oldOpPos = op.inputIndex1 - inputBufCount;
                    auto remapIt = oldToNew.find(oldOpPos);
                    if (remapIt != oldToNew.end())
                    {
                        op.inputIndex1 = inputBufCount + remapIt->second;
                    }
                    else
                    {
                        op.inputIndex1 = SIZE_MAX;
                    }
                }
            }

            desc.ops = std::move(finalOps);
        }

        if (!desc.scalarData.empty())
        {
            auto scalarSize = aix::Device::dataTypeSize(desc.dtype);
            std::vector<uint8_t> compactScalarData;
            std::unordered_map<size_t, size_t> scalarRemap;

            auto remapScalarIndex = [&](size_t& scalarIndex) {
                if (scalarIndex == SIZE_MAX) return;

                auto remapIt = scalarRemap.find(scalarIndex);
                if (remapIt != scalarRemap.end())
                {
                    scalarIndex = remapIt->second;
                    return;
                }

                size_t byteOffset = scalarIndex * scalarSize;
                assert(byteOffset + scalarSize <= desc.scalarData.size());
                size_t newIndex = compactScalarData.size() / scalarSize;
                compactScalarData.insert(compactScalarData.end(),
                                         desc.scalarData.begin() + static_cast<std::ptrdiff_t>(byteOffset),
                                         desc.scalarData.begin() + static_cast<std::ptrdiff_t>(byteOffset + scalarSize));
                scalarRemap[scalarIndex] = newIndex;
                scalarIndex = newIndex;
            };

            for (auto& op : desc.ops)
            {
                remapScalarIndex(op.scalarIndex0);
                remapScalarIndex(op.scalarIndex1);
            }

            desc.scalarData = std::move(compactScalarData);
        }

        if (!desc.ops.empty())
        {
            size_t totalBufferCount = desc.inputBuffers.size() + desc.outputBuffers.size();
            size_t requiredBufferSlots = requiredFusedBufferSlots(totalBufferCount,
                                                                  desc.allContiguous,
                                                                  !desc.scalarData.empty());
            if (requiredBufferSlots > config.maxBufferSlots)
            {
                for (size_t idx : sortedOps) result.fallbackIndices.push_back(idx);
                continue;
            }

            result.subgraphs.push_back(std::move(desc));
        }
    }

    for (size_t i = 0; i < ops.size(); ++i)
    {
        if (!visited[i]) result.fallbackIndices.push_back(i);
    }

    return result;
}

}  // anonymous namespace

void FuseEngine::record(OpType type, const DeviceTensorParams& input0, const DeviceTensorParams& output)
{
    m_pendingOps.emplace_back(type, input0, output);
    if (m_pendingOps.size() >= m_config.flushThreshold)
    {
        flush();
    }
}

void FuseEngine::record(OpType type, const DeviceTensorParams& input0, const DeviceTensorParams& input1,
                        const DeviceTensorParams& output)
{
    m_pendingOps.emplace_back(type, input0, input1, output);
    if (m_pendingOps.size() >= m_config.flushThreshold)
    {
        flush();
    }
}

void FuseEngine::recordFill(const void* scalar, DataType scalarDType, const DeviceTensorParams& output)
{
    auto scalarSize = aix::Device::dataTypeSize(scalarDType);
    std::vector<uint8_t> scalarBytes(scalarSize);
    std::memcpy(scalarBytes.data(), scalar, scalarSize);

    DeviceTensorParams dummyInput;
    m_pendingOps.emplace_back(OpType::Fill, dummyInput, output);
    m_pendingOps.back().scalarData = std::move(scalarBytes);
    m_pendingOps.back().scalarDType = scalarDType;
    if (output.data) m_absorbedFillOutputs.insert(output.data);
    if (m_pendingOps.size() >= m_config.flushThreshold)
    {
        flush();
    }
}

void FuseEngine::recordFillMin(const DeviceTensorParams& output)
{
    DeviceTensorParams dummyInput;
    m_pendingOps.emplace_back(OpType::FillMin, dummyInput, output);
    if (output.data) m_absorbedFillOutputs.insert(output.data);
    if (m_pendingOps.size() >= m_config.flushThreshold)
    {
        flush();
    }
}

void FuseEngine::flush()
{
    thread_local bool flushing = false;
    if (flushing) return;
    flushing = true;

    if (m_config.diagnostics)
    {
        m_diagnostics.emplace();
        m_diagnostics->opsRecorded = m_pendingOps.size();
    }

    if (m_config.elementwiseFusion && !m_pendingOps.empty())
    {
        if (m_pendingOps.size() < 3)
        {
            for (const auto& op : m_pendingOps)
            {
                m_callbacks.emitSingle(op);
            }
        }
        else
        {
            if (!m_deferredFills.empty())
            {
                for (const auto& op : m_pendingOps)
                {
                    if (op.input0.data)
                    {
                        auto it = m_deferredFills.find(op.input0.data);
                        if (it != m_deferredFills.end())
                        {
                            m_callbacks.emitSingle(it->second);
                            m_emittedFillBuffers.insert(it->first);
                            m_deferredFills.erase(it);
                        }
                    }
                    if (op.hasInput1 && op.input1.data)
                    {
                        auto it = m_deferredFills.find(op.input1.data);
                        if (it != m_deferredFills.end())
                        {
                            m_callbacks.emitSingle(it->second);
                            m_emittedFillBuffers.insert(it->first);
                            m_deferredFills.erase(it);
                        }
                    }
                }
            }

            auto liveDag = buildDag(m_pendingOps);
            auto result = analyzeFusion(m_pendingOps, liveDag, m_config, m_liveBuffers);

                size_t totalFusedOps = 0;
                size_t fillsAbsorbed = 0;
                for (auto& subgraph : result.subgraphs)
                {
                    m_callbacks.emitFused(subgraph);
                    totalFusedOps += subgraph.ops.size();
                    for (const auto& op : subgraph.ops)
                    {
                        if ((op.type != OpType::Fill && op.scalarIndex0 != SIZE_MAX) || op.scalarIndex1 != SIZE_MAX)
                        {
                            ++fillsAbsorbed;
                        }
                    }
                }

                std::sort(result.fallbackIndices.begin(), result.fallbackIndices.end());
                result.fallbackIndices.erase(std::unique(result.fallbackIndices.begin(), result.fallbackIndices.end()),
                                             result.fallbackIndices.end());
                for (size_t idx : result.fallbackIndices)
                {
                    m_callbacks.emitSingle(m_pendingOps[idx]);
                }

                for (const auto& fill : result.nonInPlaceAbsorbedFills)
                {
                    if (fill.output.data && !m_emittedFillBuffers.contains(fill.output.data))
                    {
                        m_deferredFills[fill.output.data] = fill;
                    }
                }

                if (m_config.diagnostics)
                {
                    m_diagnostics->fusibleSubgraphs = result.subgraphs.size();
                    m_diagnostics->fusedOps = totalFusedOps;
                    m_diagnostics->fallbackOps = result.fallbackIndices.size();
                    m_diagnostics->dispatchesSaved = (totalFusedOps + result.fallbackIndices.size())
                                                      - (result.subgraphs.size() + result.fallbackIndices.size());
                    m_diagnostics->fillsAbsorbed = fillsAbsorbed;

                    std::ostringstream subgraphSummary;
                    subgraphSummary << result.subgraphs.size() << " subgraphs";
                    if (!result.subgraphs.empty())
                    {
                        subgraphSummary << " (";
                        for (size_t i = 0; i < result.subgraphs.size(); ++i)
                        {
                            if (i > 0) subgraphSummary << ", ";
                            subgraphSummary << result.subgraphs[i].ops.size() << " ops";
                        }
                        subgraphSummary << ")";
                    }
                    m_diagnostics->subgraphSummary = subgraphSummary.str();

                    std::ostringstream fallbackSummary;
                    fallbackSummary << result.fallbackIndices.size() << " individual ops";
                    m_diagnostics->fallbackSummary = fallbackSummary.str();
                }
        }
    }
    else
    {
        for (const auto& op : m_pendingOps)
        {
            m_callbacks.emitSingle(op);
        }

        if (m_config.diagnostics)
        {
            m_diagnostics->fusibleSubgraphs = 0;
            m_diagnostics->fusedOps = 0;
            m_diagnostics->fallbackOps = m_pendingOps.size();
            m_diagnostics->dispatchesSaved = 0;
            m_diagnostics->fillsAbsorbed = 0;
            m_diagnostics->subgraphSummary = "0 subgraphs (fusion disabled)";
            m_diagnostics->fallbackSummary = std::to_string(m_pendingOps.size()) + " individual ops";
        }
    }

    m_callbacks.finishFlush();

    if (m_config.diagnostics)
    {
        auto [hits, misses] = m_callbacks.getKernelCacheStats();
        m_diagnostics->kernelCacheHits = hits;
        m_diagnostics->kernelCacheMisses = misses;
    }

    m_pendingOps.clear();
    m_externalLiveBuffers.clear();
    m_absorbedFillOutputs.clear();

    flushing = false;
}

void FuseEngine::recordExternalRead(const void* buffer)
{
    if (buffer) m_externalLiveBuffers.insert(const_cast<void*>(buffer));
}

void FuseEngine::retainBuffer(void* buffer)
{
    if (buffer) m_liveBuffers.insert(buffer);
}

void FuseEngine::releaseBuffer(void* buffer)
{
    if (!buffer) return;
    m_liveBuffers.erase(buffer);
    m_externalLiveBuffers.erase(buffer);
    invalidateBuffer(buffer);
}

void FuseEngine::invalidateBuffer(void* buffer)
{
    m_deferredFills.erase(buffer);
    m_emittedFillBuffers.erase(buffer);
}

}  // namespace aix::fuse
