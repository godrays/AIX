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
#include "aixFuse.hpp"
// External includes
// System includes
#include <cstddef>
#include <string>
#include <unordered_map>


// Forward declarations
namespace MTL
{
    class Buffer;
    class ComputePipelineState;
    class ComputeCommandEncoder;
    class Device;
}

namespace aix::metal
{

class aixDeviceMetalKernelGen
{
public:
    explicit aixDeviceMetalKernelGen(MTL::Device* device);

    ~aixDeviceMetalKernelGen();

    aixDeviceMetalKernelGen(const aixDeviceMetalKernelGen&) = delete;
    aixDeviceMetalKernelGen& operator=(const aixDeviceMetalKernelGen&) = delete;

    MTL::ComputePipelineState* getOrCreatePSO(const aix::fuse::FusedSubgraphDescriptor& subgraph);

    void encodeFusedDispatch(
        MTL::ComputeCommandEncoder* encoder,
        MTL::ComputePipelineState* pso,
        const aix::fuse::FusedSubgraphDescriptor& subgraph,
        const std::unordered_map<const void*, MTL::Buffer*>& allocMap);

    size_t cacheHits() const { return m_cacheHits; }

    size_t cacheMisses() const { return m_cacheMisses; }

private:
    std::string generateShaderSource(const aix::fuse::FusedSubgraphDescriptor& subgraph);

    std::string computeSignature(const aix::fuse::FusedSubgraphDescriptor& subgraph);

    MTL::Device*  m_device{nullptr};
    std::unordered_map<std::string, MTL::ComputePipelineState*>  m_kernelCache;
    size_t  m_cacheHits{0};
    size_t  m_cacheMisses{0};
};

}  // namespace aix::metal
