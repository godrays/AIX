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
#include <Metal/Metal.hpp>
// System includes
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace aix::metal
{

// All backend dispatches, including generated kernels, pass through this encoder.
// Reflection supplies access modes; whole-buffer ranges conservatively cover views,
// shader offsets and vectorized tails. Existing inter-command-buffer fences remain.
class MetalComputeEncoder
{
public:
    MTL::ComputePipelineState* createPipeline(MTL::Device* device, MTL::Function* function, NS::Error** error)
    {
        MTL::ComputePipelineReflection* reflection = nullptr;
        auto pso = device->newComputePipelineState(function, MTL::PipelineOptionArgumentInfo, &reflection, error);
        if (pso && reflection)
        {
            Pipeline pipeline;
            pipeline.known = true;
            auto arguments = reflection->arguments();
            for (NS::UInteger i = 0; i < arguments->count(); ++i)
            {
                auto argument = arguments->object<MTL::Argument>(i);
                if (!argument->active()) continue;
                if (argument->type() == MTL::ArgumentTypeThreadgroupMemory) continue;
                if (argument->type() != MTL::ArgumentTypeBuffer || argument->index() >= m_bindings.size())
                {
                    pipeline.known = false;
                    break;
                }
                pipeline.buffers.push_back({argument->index(), argument->access() != MTL::BindingAccessReadOnly});
            }
            m_pipelines.emplace(pso, std::move(pipeline));
        }
        return pso;
    }

    void reset(MTL::ComputeCommandEncoder* encoder)
    {
        m_encoder = encoder;
        m_pipeline = nullptr;
        m_bindings = {};
        m_nodeCount = 0;
        m_bytes.clear();
    }

    void setComputePipelineState(const MTL::ComputePipelineState* pso)
    {
        auto it = m_pipelines.find(pso);
        m_pipeline = it == m_pipelines.end() ? nullptr : &it->second;
        m_pso = pso;
    }

    void setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
    {
        auto& binding = m_bindings.at(index);
        binding = {};
        binding.valid = true;
        binding.buffer = buffer;
        binding.offset = offset;
        binding.index = index;
        if (buffer)
        {
            auto& range = binding.range;
            range.bound = true;
            // metal-cpp exposes these resource queries only as non-const methods.
            auto queryBuffer = const_cast<MTL::Buffer*>(buffer);
            range.known = buffer->storageMode() == MTL::StorageModeShared && !queryBuffer->isAliasable();
            if (range.known)
            {
                range.begin = reinterpret_cast<uintptr_t>(queryBuffer->contents());
                range.end = range.begin + buffer->length();
                range.heap = buffer->heap();
                // heapOffset is always zero for automatic heaps. Their live,
                // non-aliasable shared allocations are tracked by contents instead.
                // The allocator only recycles storage after command completion.
                if (range.heap && range.heap->type() == MTL::HeapTypePlacement)
                {
                    range.heapBegin = buffer->heapOffset();
                    range.heapEnd = range.heapBegin + buffer->length();
                }
                else
                {
                    range.heap = nullptr;
                }
            }
        }
    }

    void setBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
    {
        // Metal copies inline constants. They do not alias a previously bound buffer.
        auto& binding = m_bindings.at(index);
        binding = {};
        binding.valid = true;
        binding.inlineBytes = true;
        binding.index = index;
        binding.offset = m_bytes.size();
        binding.length = length;
        m_bytes.resize(m_bytes.size() + length);
        std::memcpy(m_bytes.data() + binding.offset, bytes, length);
    }

    void dispatchThreads(MTL::Size grid, MTL::Size group)
    {
        record(grid, group, false);
    }

    void dispatchThreadgroups(MTL::Size grid, MTL::Size group)
    {
        record(grid, group, true);
    }

    void waitForFence(const MTL::Fence* fence) { m_encoder->waitForFence(fence); }
    void updateFence(const MTL::Fence* fence) { flush(); m_encoder->updateFence(fence); }
    void endEncoding() { flush(); m_encoder->endEncoding(); }

private:
    struct BufferAccess
    {
        NS::UInteger index;
        bool write;
    };

    struct Pipeline
    {
        bool known{false};
        std::vector<BufferAccess> buffers;
    };

    struct Range
    {
        uintptr_t begin{0}, end{0};
        const MTL::Heap* heap{nullptr};
        NS::UInteger heapBegin{0}, heapEnd{0};
        bool bound{false}, known{false}, write{false};
    };

    struct Binding
    {
        const MTL::Buffer* buffer{nullptr};
        NS::UInteger offset{0}, length{0}, index{0};
        Range range;
        bool valid{false}, inlineBytes{false};
    };

    struct Dispatch
    {
        const MTL::ComputePipelineState* pso{nullptr};
        std::vector<Binding> bindings;
        MTL::Size grid, group;
        size_t level{0};
        bool threadgroups{false}, serial{false};
    };

    static bool overlaps(const Range& a, const Range& b)
    {
        // CPU address ranges also catch distinct shared-buffer wrappers. Heap ranges
        // cover different resource objects aliasing the same heap allocation.
        return (a.begin < b.end && b.begin < a.end)
            || (a.heap && a.heap == b.heap && a.heapBegin < b.heapEnd && b.heapBegin < a.heapEnd);
    }

    static bool dependsOn(const Dispatch& current, const Dispatch& previous)
    {
        if (current.serial || previous.serial) return true;
        for (const auto& a : current.bindings)
        {
            if (!a.range.bound) continue;
            for (const auto& b : previous.bindings)
            {
                if (b.range.bound && (a.range.write || b.range.write) && overlaps(a.range, b.range)) return true;
            }
        }
        return false;
    }

    void record(MTL::Size grid, MTL::Size group, bool threadgroups)
    {
        if (m_nodeCount == m_nodes.size()) m_nodes.emplace_back();
        auto& node = m_nodes[m_nodeCount];
        node.bindings.clear();
        node.pso = m_pso;
        node.grid = grid;
        node.group = group;
        node.threadgroups = threadgroups;
        node.serial = !m_pipeline || !m_pipeline->known;
        node.level = 0;
        if (!node.serial)
        {
            for (const auto& access : m_pipeline->buffers)
            {
                auto binding = m_bindings[access.index];
                if (binding.range.bound && !binding.range.known) node.serial = true;
                binding.range.write = access.write;
                if (binding.valid) node.bindings.push_back(binding);
            }
        }
        else
        {
            for (const auto& binding : m_bindings)
            {
                if (binding.valid) node.bindings.push_back(binding);
            }
        }
        // Reorder only within the existing command buffer, never across a commit
        // boundary. RAW/WAR/WAW edges preserve the original dispatch semantics.
        for (size_t i = 0; i < m_nodeCount; ++i)
        {
            if (dependsOn(node, m_nodes[i])) node.level = std::max(node.level, m_nodes[i].level + 1);
        }
        ++m_nodeCount;
    }

    void flush()
    {
        if (!m_nodeCount) return;
        size_t remaining = m_nodeCount;
        for (size_t level = 0; remaining; ++level)
        {
            if (level) m_encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            for (size_t i = 0; i < m_nodeCount; ++i)
            {
                const auto& node = m_nodes[i];
                if (node.level != level) continue;
                m_encoder->setComputePipelineState(node.pso);
                for (const auto& binding : node.bindings)
                {
                    if (binding.inlineBytes)
                        m_encoder->setBytes(m_bytes.data() + binding.offset, binding.length, binding.index);
                    else
                        m_encoder->setBuffer(binding.buffer, binding.offset, binding.index);
                }
                if (node.threadgroups) m_encoder->dispatchThreadgroups(node.grid, node.group);
                else m_encoder->dispatchThreads(node.grid, node.group);
                --remaining;
            }
        }
        m_nodeCount = 0;
    }

    MTL::ComputeCommandEncoder* m_encoder{nullptr};
    std::unordered_map<const MTL::ComputePipelineState*, Pipeline> m_pipelines;
    const Pipeline* m_pipeline{nullptr};
    const MTL::ComputePipelineState* m_pso{nullptr};
    std::array<Binding, 31> m_bindings{};
    std::vector<Dispatch> m_nodes;
    size_t m_nodeCount{0};
    std::vector<std::byte> m_bytes;
};

}  // namespace aix::metal
