//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "aixDeviceMetalKernelGen.hpp"
// External includes
#include <Metal/Metal.hpp>
// System includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>


namespace aix::metal
{

static const char* opTypeName(aix::fuse::OpType type)
{
    switch (type)
    {
        case aix::fuse::OpType::Add:     return "Add";
        case aix::fuse::OpType::Sub:     return "Sub";
        case aix::fuse::OpType::Mul:     return "Mul";
        case aix::fuse::OpType::Div:     return "Div";
        case aix::fuse::OpType::Negate:  return "Negate";
        case aix::fuse::OpType::Sqrt:    return "Sqrt";
        case aix::fuse::OpType::Sin:     return "Sin";
        case aix::fuse::OpType::Cos:     return "Cos";
        case aix::fuse::OpType::Tanh:    return "Tanh";
        case aix::fuse::OpType::Log:     return "Log";
        case aix::fuse::OpType::Exp:     return "Exp";
        case aix::fuse::OpType::Pow:     return "Pow";
        case aix::fuse::OpType::Fill:    return "Fill";
        case aix::fuse::OpType::FillMin: return "FillMin";
        case aix::fuse::OpType::Cast:    return "Cast";
        default:                         return "Unknown";
    }
}

static const char* metalScalarTypeName(aix::DataType dtype)
{
    switch (dtype)
    {
        case aix::DataType::kFloat32:  return "float";
        case aix::DataType::kFloat16:  return "half";
        case aix::DataType::kBFloat16: return "bfloat";
        case aix::DataType::kInt64:    return "long";
        case aix::DataType::kInt32:    return "int";
        case aix::DataType::kInt16:    return "short";
        case aix::DataType::kInt8:     return "char";
        case aix::DataType::kUInt8:    return "uchar";
        default:                       return "float";
    }
}

static const char* metalVectorTypeName(aix::DataType dtype)
{
    switch (dtype)
    {
        case aix::DataType::kFloat32:  return "float4";
        case aix::DataType::kFloat16:  return "half4";
        case aix::DataType::kBFloat16: return "bfloat4";
        case aix::DataType::kInt64:    return "long4";
        case aix::DataType::kInt32:    return "int4";
        case aix::DataType::kInt16:    return "short4";
        case aix::DataType::kInt8:     return "char4";
        case aix::DataType::kUInt8:    return "uchar4";
        default:                       return "float4";
    }
}

static std::string dtypeSuffix(aix::DataType dtype)
{
    switch (dtype)
    {
        case aix::DataType::kFloat64:  return "f64";
        case aix::DataType::kFloat32:  return "f32";
        case aix::DataType::kFloat16:  return "f16";
        case aix::DataType::kBFloat16: return "bf16";
        case aix::DataType::kInt64:    return "i64";
        case aix::DataType::kInt32:    return "i32";
        case aix::DataType::kInt16:    return "i16";
        case aix::DataType::kInt8:     return "i8";
        case aix::DataType::kUInt8:    return "u8";
        default:                       return "f32";
    }
}

static std::string physicalIndexHelpers()
{
    return R"(
METAL_FUNC size_t layoutOffset(const constant size_t* layout) { return layout[0]; }
METAL_FUNC size_t layoutRank(const constant size_t* layout) { return layout[1]; }
METAL_FUNC const constant size_t* layoutShape(const constant size_t* layout) { return layout + 2; }
METAL_FUNC const constant size_t* layoutStrides(const constant size_t* layout) { return layout + 2 + layoutRank(layout); }
METAL_FUNC size_t physicalIndex(size_t flatIndex, const constant size_t* layout) {
    size_t idx = flatIndex;
    size_t ofs = layoutOffset(layout);
    const constant size_t* shape = layoutShape(layout);
    const constant size_t* strides = layoutStrides(layout);
    size_t rank = layoutRank(layout);
    for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
        size_t dimIndex = idx % shape[dim];
        idx /= shape[dim];
        ofs += dimIndex * strides[dim];
    }
    return ofs;
}
)";
}

static bool isTranscendentalOp(aix::fuse::OpType type)
{
    switch (type)
    {
        case aix::fuse::OpType::Sqrt:
        case aix::fuse::OpType::Sin:
        case aix::fuse::OpType::Cos:
        case aix::fuse::OpType::Tanh:
        case aix::fuse::OpType::Log:
        case aix::fuse::OpType::Exp:
        case aix::fuse::OpType::Pow:
            return true;
        default:
            return false;
    }
}

static std::string wrapScalarForDtype(const std::string& scalarLit, aix::DataType dtype, bool isVector)
{
    std::string scalarType = metalScalarTypeName(dtype);
    std::string vectorType = metalVectorTypeName(dtype);
    std::string typeName = isVector ? vectorType : scalarType;

    if (dtype == aix::DataType::kBFloat16 && isVector)
    {
        return vectorType + "(" + scalarType + "(" + scalarLit + "))";
    }
    return typeName + "(" + scalarLit + ")";
}

static std::string fillMinLiteral(aix::DataType dtype)
{
    switch (dtype)
    {
        case aix::DataType::kFloat64:  return "-1.7976931348623157e+308";
        case aix::DataType::kFloat32:  return "-3.402823466e+38f";
        case aix::DataType::kFloat16:  return "-65504.0f";
        case aix::DataType::kBFloat16: return "-3.389531389251535e+38f";
        case aix::DataType::kInt64:    return "(-9223372036854775807LL - 1)";
        case aix::DataType::kInt32:    return "(-2147483647 - 1)";
        case aix::DataType::kInt16:    return "-32768";
        case aix::DataType::kInt8:     return "-128";
        case aix::DataType::kUInt8:    return "0";
        default:                       return "-3.402823466e+38f";
    }
}

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

static DecodedScalar decodeScalar(const std::vector<uint8_t>& scalarData, aix::DataType dtype)
{
    DecodedScalar decoded;
    if (scalarData.empty()) return decoded;

    switch (dtype)
    {
        case aix::DataType::kFloat64:
        {
            double val = 0.0;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(double), scalarData.size()));
            decoded.floatValue = val;
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kFloat32:
        {
            float val = 0.0f;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(float), scalarData.size()));
            decoded.floatValue = val;
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kFloat16:
        {
            float16_t val = 0.0f;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(float16_t), scalarData.size()));
            decoded.floatValue = static_cast<float>(val);
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kBFloat16:
        {
            bfloat16_t val = 0.0f;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(bfloat16_t), scalarData.size()));
            decoded.floatValue = static_cast<float>(val);
            decoded.kind = ScalarKind::kFloat;
            break;
        }
        case aix::DataType::kInt64:
            std::memcpy(&decoded.signedValue, scalarData.data(), std::min(sizeof(int64_t), scalarData.size()));
            decoded.kind = ScalarKind::kSigned;
            break;
        case aix::DataType::kInt32:
        {
            int32_t val = 0;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(int32_t), scalarData.size()));
            decoded.signedValue = val;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kInt16:
        {
            int16_t val = 0;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(int16_t), scalarData.size()));
            decoded.signedValue = val;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kInt8:
        {
            int8_t val = 0;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(int8_t), scalarData.size()));
            decoded.signedValue = val;
            decoded.kind = ScalarKind::kSigned;
            break;
        }
        case aix::DataType::kUInt8:
        {
            uint8_t val = 0;
            std::memcpy(&val, scalarData.data(), std::min(sizeof(uint8_t), scalarData.size()));
            decoded.unsignedValue = val;
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

    value = std::clamp(value,
                       static_cast<long double>(std::numeric_limits<T>::lowest()),
                       static_cast<long double>(std::numeric_limits<T>::max()));
    return static_cast<T>(value);
}

static std::string formatScalarLiteral(const std::vector<uint8_t>& scalarData,
                                       aix::DataType scalarDType,
                                       aix::DataType targetDType)
{
    if (scalarData.empty()) return "0";

    auto decoded = decodeScalar(scalarData, scalarDType);
    auto asFloat = [&]() -> long double {
        switch (decoded.kind)
        {
            case ScalarKind::kFloat:    return decoded.floatValue;
            case ScalarKind::kSigned:   return static_cast<long double>(decoded.signedValue);
            case ScalarKind::kUnsigned: return static_cast<long double>(decoded.unsignedValue);
        }
        return 0.0L;
    };

    switch (targetDType)
    {
        case aix::DataType::kFloat64:
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", static_cast<double>(asFloat()));
            std::string s = buf;
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos)
                s += ".0";
            return s;
        }
        case aix::DataType::kFloat32:
        case aix::DataType::kFloat16:
        case aix::DataType::kBFloat16:
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.9g", static_cast<float>(asFloat()));
            std::string s = buf;
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos)
                s += ".0";
            s += "f";
            return s;
        }
        case aix::DataType::kInt64:
        {
            int64_t val = 0;
            switch (decoded.kind)
            {
                case ScalarKind::kFloat:
                    val = clampFloatToIntegral<int64_t>(decoded.floatValue);
                    break;
                case ScalarKind::kSigned:
                    val = decoded.signedValue;
                    break;
                case ScalarKind::kUnsigned:
                    val = decoded.unsignedValue > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                        ? std::numeric_limits<int64_t>::max()
                        : static_cast<int64_t>(decoded.unsignedValue);
                    break;
            }
            return std::to_string(val) + "LL";
        }
        case aix::DataType::kInt32:
        {
            int32_t val = 0;
            switch (decoded.kind)
            {
                case ScalarKind::kFloat:
                    val = clampFloatToIntegral<int32_t>(decoded.floatValue);
                    break;
                case ScalarKind::kSigned:
                    val = static_cast<int32_t>(std::clamp(decoded.signedValue,
                                                          static_cast<int64_t>(std::numeric_limits<int32_t>::lowest()),
                                                          static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
                    break;
                case ScalarKind::kUnsigned:
                    val = static_cast<int32_t>(std::min(decoded.unsignedValue,
                                                        static_cast<uint64_t>(std::numeric_limits<int32_t>::max())));
                    break;
            }
            return std::to_string(val);
        }
        case aix::DataType::kInt16:
        {
            int16_t val = 0;
            switch (decoded.kind)
            {
                case ScalarKind::kFloat:
                    val = clampFloatToIntegral<int16_t>(decoded.floatValue);
                    break;
                case ScalarKind::kSigned:
                    val = static_cast<int16_t>(std::clamp(decoded.signedValue,
                                                          static_cast<int64_t>(std::numeric_limits<int16_t>::lowest()),
                                                          static_cast<int64_t>(std::numeric_limits<int16_t>::max())));
                    break;
                case ScalarKind::kUnsigned:
                    val = static_cast<int16_t>(std::min(decoded.unsignedValue,
                                                        static_cast<uint64_t>(std::numeric_limits<int16_t>::max())));
                    break;
            }
            return std::to_string(val);
        }
        case aix::DataType::kInt8:
        {
            int8_t val = 0;
            switch (decoded.kind)
            {
                case ScalarKind::kFloat:
                    val = clampFloatToIntegral<int8_t>(decoded.floatValue);
                    break;
                case ScalarKind::kSigned:
                    val = static_cast<int8_t>(std::clamp(decoded.signedValue,
                                                         static_cast<int64_t>(std::numeric_limits<int8_t>::lowest()),
                                                         static_cast<int64_t>(std::numeric_limits<int8_t>::max())));
                    break;
                case ScalarKind::kUnsigned:
                    val = static_cast<int8_t>(std::min(decoded.unsignedValue,
                                                       static_cast<uint64_t>(std::numeric_limits<int8_t>::max())));
                    break;
            }
            return std::to_string(val);
        }
        case aix::DataType::kUInt8:
        {
            uint8_t val = 0;
            switch (decoded.kind)
            {
                case ScalarKind::kFloat:
                    val = clampFloatToIntegral<uint8_t>(decoded.floatValue);
                    break;
                case ScalarKind::kSigned:
                    val = static_cast<uint8_t>(std::clamp(decoded.signedValue,
                                                          static_cast<int64_t>(0),
                                                          static_cast<int64_t>(std::numeric_limits<uint8_t>::max())));
                    break;
                case ScalarKind::kUnsigned:
                    val = static_cast<uint8_t>(std::min(decoded.unsignedValue,
                                                        static_cast<uint64_t>(std::numeric_limits<uint8_t>::max())));
                    break;
            }
            return std::to_string(static_cast<unsigned>(val));
        }
        default:
            return "0";
    }
}

static void hashCombine(uint64_t& hash, uint64_t value)
{
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

static std::string subgraphSignatureHash(const aix::fuse::FusedSubgraphDescriptor& subgraph)
{
    uint64_t hash = 1469598103934665603ULL;
    hashCombine(hash, static_cast<uint64_t>(subgraph.dtype));
    hashCombine(hash, subgraph.allContiguous ? 1ULL : 0ULL);
    hashCombine(hash, subgraph.inputBuffers.size());
    hashCombine(hash, subgraph.outputBuffers.size());
    hashCombine(hash, subgraph.scalarData.size());

    for (const auto& op : subgraph.ops)
    {
        hashCombine(hash, static_cast<uint64_t>(op.type));
        hashCombine(hash, op.inputIndex0 == SIZE_MAX ? std::numeric_limits<uint64_t>::max() : op.inputIndex0);
        hashCombine(hash, op.inputIndex1 == SIZE_MAX ? std::numeric_limits<uint64_t>::max() : op.inputIndex1);
        hashCombine(hash, op.outputIndex == SIZE_MAX ? std::numeric_limits<uint64_t>::max() : op.outputIndex);
        hashCombine(hash, op.scalarIndex0 == SIZE_MAX ? std::numeric_limits<uint64_t>::max() : op.scalarIndex0);
        hashCombine(hash, op.scalarIndex1 == SIZE_MAX ? std::numeric_limits<uint64_t>::max() : op.scalarIndex1);
    }

    std::ostringstream oss;
    oss << std::hex << hash;
    return oss.str();
}

aixDeviceMetalKernelGen::aixDeviceMetalKernelGen(MTL::Device* device)
    : m_device(device)
{
}

aixDeviceMetalKernelGen::~aixDeviceMetalKernelGen()
{
    for (auto& [sig, pso] : m_kernelCache)
    {
        if (pso) pso->release();
    }
    m_kernelCache.clear();
}

std::string aixDeviceMetalKernelGen::computeSignature(const aix::fuse::FusedSubgraphDescriptor& subgraph)
{
    std::string sig;
    for (size_t i = 0; i < subgraph.ops.size(); ++i)
    {
        if (i > 0) sig += "_";
        sig += opTypeName(subgraph.ops[i].type);
    }
    sig += "_" + dtypeSuffix(subgraph.dtype);
    sig += subgraph.allContiguous ? "_contig" : "_strided";
    sig += "_i" + std::to_string(subgraph.inputBuffers.size());
    sig += "_o" + std::to_string(subgraph.outputBuffers.size());
    sig += "_h" + subgraphSignatureHash(subgraph);
    return sig;
}

std::string aixDeviceMetalKernelGen::generateShaderSource(const aix::fuse::FusedSubgraphDescriptor& subgraph)
{
    const bool isContiguous = subgraph.allContiguous;
    const char* typeStr = isContiguous ? metalVectorTypeName(subgraph.dtype) : metalScalarTypeName(subgraph.dtype);
    const char* scalarType = metalScalarTypeName(subgraph.dtype);
    const char* floatType = isContiguous ? "float4" : "float";
    const size_t totalBufferCount = subgraph.inputBuffers.size() + subgraph.outputBuffers.size();
    const bool hasRuntimeScalars = !subgraph.scalarData.empty();
    const size_t scalarBufferSlot = totalBufferCount + (isContiguous ? 0 : totalBufferCount);

    auto inputSlot = [](size_t inputIndex) -> size_t {
        return inputIndex;
    };
    auto outputSlot = [&](size_t outputIndex) -> size_t {
        return subgraph.inputBuffers.size() + outputIndex;
    };

    auto resolveInput = [&](size_t inputIndex, size_t scalarIndex) -> std::string {
        if (scalarIndex != SIZE_MAX)
        {
            std::string scalarRef = "scalars[" + std::to_string(scalarIndex) + "]";
            return wrapScalarForDtype(scalarRef, subgraph.dtype, isContiguous);
        }
        if (inputIndex < subgraph.inputBuffers.size())
        {
            size_t slot = inputSlot(inputIndex);
            if (isContiguous)
            {
                return "buf" + std::to_string(slot) + "[gid]";
            }
            return "buf" + std::to_string(slot) + "[physicalIndex(gid, layout" + std::to_string(slot) + ")]";
        }
        if (inputIndex == SIZE_MAX)
        {
            return wrapScalarForDtype("0", subgraph.dtype, isContiguous);
        }
        size_t regIdx = inputIndex - subgraph.inputBuffers.size();
        return "v" + std::to_string(regIdx);
    };

    auto validateSubgraph = [&]() -> bool {
        size_t regCount = 0;
        for (const auto& op : subgraph.ops)
        {
            if (op.inputIndex0 != SIZE_MAX)
            {
                if (op.inputIndex0 >= subgraph.inputBuffers.size() + regCount)
                    return false;
            }
            if (op.inputIndex1 != SIZE_MAX)
            {
                if (op.inputIndex1 >= subgraph.inputBuffers.size() + regCount)
                    return false;
            }
            ++regCount;
        }
        return true;
    };

    if (!validateSubgraph()) return "";

    auto castInput = [&](const std::string& input, aix::fuse::OpType type) -> std::string {
        if (subgraph.dtype != aix::DataType::kFloat32 && isTranscendentalOp(type))
        {
            return std::string(floatType) + "(" + input + ")";
        }
        return input;
    };

    auto wrapResult = [&](const std::string& expr, aix::fuse::OpType type) -> std::string {
        if (subgraph.dtype != aix::DataType::kFloat32 && isTranscendentalOp(type))
        {
            return std::string(typeStr) + "(" + expr + ")";
        }
        return expr;
    };

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n\n";

    if (!isContiguous)
    {
        ss << physicalIndexHelpers() << "\n";
    }

    std::string sig = computeSignature(subgraph);
    ss << "kernel void fused_" << sig << "(\n";

    for (size_t i = 0; i < subgraph.inputBuffers.size(); ++i)
    {
        auto slot = inputSlot(i);
        ss << "    device " << typeStr << "* buf" << slot
           << " [[buffer(" << slot << ")]],\n";
    }
    for (size_t i = 0; i < subgraph.outputBuffers.size(); ++i)
    {
        auto slot = outputSlot(i);
        ss << "    device " << typeStr << "* buf" << slot
           << " [[buffer(" << slot << ")]],\n";
    }

    if (!isContiguous)
    {
        for (size_t i = 0; i < totalBufferCount; ++i)
        {
            ss << "    const constant size_t* layout" << i
               << " [[buffer(" << (i + totalBufferCount) << ")]],\n";
        }
    }

    if (hasRuntimeScalars)
    {
        ss << "    const constant " << scalarType << "* scalars"
           << " [[buffer(" << scalarBufferSlot << ")]],\n";
    }

    ss << "    uint gid [[thread_position_in_grid]]) {\n";

    int regCount = 0;
    for (const auto& op : subgraph.ops)
    {
        std::string resultVar = "v" + std::to_string(regCount);

        switch (op.type)
        {
            case aix::fuse::OpType::Fill:
                ss << "    " << typeStr << " " << resultVar << " = "
                   << resolveInput(SIZE_MAX, op.scalarIndex0) << ";\n";
                break;
            case aix::fuse::OpType::FillMin:
            {
                std::string minLit = fillMinLiteral(subgraph.dtype);
                std::string wrappedLit = wrapScalarForDtype(minLit, subgraph.dtype, isContiguous);
                ss << "    " << typeStr << " " << resultVar << " = " << wrappedLit << ";\n";
                break;
            }
            case aix::fuse::OpType::Cast:
                ss << "    " << typeStr << " " << resultVar << " = " << resolveInput(op.inputIndex0, op.scalarIndex0) << ";\n";
                break;
            case aix::fuse::OpType::Negate:
                ss << "    " << typeStr << " " << resultVar << " = -" << resolveInput(op.inputIndex0, op.scalarIndex0) << ";\n";
                break;
            case aix::fuse::OpType::Sqrt:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::sqrt(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Sin:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::sin(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Cos:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::cos(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Tanh:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::tanh(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Log:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::log(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Exp:
            {
                std::string in = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                   << wrapResult("metal::exp(" + in + ")", op.type) << ";\n";
                break;
            }
            case aix::fuse::OpType::Add:
                ss << "    " << typeStr << " " << resultVar << " = " << resolveInput(op.inputIndex0, op.scalarIndex0)
                   << " + " << resolveInput(op.inputIndex1, op.scalarIndex1) << ";\n";
                break;
            case aix::fuse::OpType::Sub:
                ss << "    " << typeStr << " " << resultVar << " = " << resolveInput(op.inputIndex0, op.scalarIndex0)
                   << " - " << resolveInput(op.inputIndex1, op.scalarIndex1) << ";\n";
                break;
            case aix::fuse::OpType::Mul:
                ss << "    " << typeStr << " " << resultVar << " = " << resolveInput(op.inputIndex0, op.scalarIndex0)
                   << " * " << resolveInput(op.inputIndex1, op.scalarIndex1) << ";\n";
                break;
            case aix::fuse::OpType::Div:
                ss << "    " << typeStr << " " << resultVar << " = " << resolveInput(op.inputIndex0, op.scalarIndex0)
                   << " / " << resolveInput(op.inputIndex1, op.scalarIndex1) << ";\n";
                break;
            case aix::fuse::OpType::Pow:
            {
                std::string in0 = castInput(resolveInput(op.inputIndex0, op.scalarIndex0), op.type);
                std::string in1 = castInput(resolveInput(op.inputIndex1, op.scalarIndex1), op.type);
                ss << "    " << typeStr << " " << resultVar << " = "
                    << wrapResult("metal::pow(" + in0 + ", " + in1 + ")", op.type) << ";\n";
                break;
            }
            default:
                break;
        }

        if (op.outputIndex != SIZE_MAX && op.outputIndex < subgraph.outputBuffers.size())
        {
            size_t slot = outputSlot(op.outputIndex);
            if (isContiguous)
            {
                ss << "    buf" << slot << "[gid] = " << resultVar << ";\n";
            }
            else
            {
                ss << "    buf" << slot << "[physicalIndex(gid, layout" << slot << ")] = " << resultVar << ";\n";
            }
        }

        ++regCount;
    }

    ss << "}\n";
    return ss.str();
}

MTL::ComputePipelineState* aixDeviceMetalKernelGen::getOrCreatePSO(const aix::fuse::FusedSubgraphDescriptor& subgraph)
{
    std::string sig = computeSignature(subgraph);

    auto it = m_kernelCache.find(sig);
    if (it != m_kernelCache.end())
    {
        ++m_cacheHits;
        return it->second;
    }

    ++m_cacheMisses;

    std::string source = generateShaderSource(subgraph);
    if (source.empty()) return nullptr;

    auto compileOptions = MTL::CompileOptions::alloc()->init();
    compileOptions->setOptimizationLevel(MTL::LibraryOptimizationLevelDefault);
    compileOptions->setFastMathEnabled(false);

    NS::Error* error = nullptr;
    auto nsSource = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    auto library = m_device->newLibrary(nsSource, compileOptions, &error);
    compileOptions->release();

    if (!library)
    {
        std::cerr << "DeviceMetalKernelGen: failed to compile fused kernel. ";
        if (error) std::cerr << error->localizedDescription()->utf8String();
        std::cerr << "\n";
        return nullptr;
    }

    std::string kernelName = "fused_" + sig;
    auto funcName = NS::String::string(kernelName.c_str(), NS::ASCIIStringEncoding);
    auto function = library->newFunction(funcName);
    library->release();

    if (!function)
    {
        std::cerr << "DeviceMetalKernelGen: failed to find kernel function '" << kernelName << "'\n";
        return nullptr;
    }

    error = nullptr;
    auto pso = m_device->newComputePipelineState(function, &error);
    function->release();

    if (!pso)
    {
        std::cerr << "DeviceMetalKernelGen: failed to create PSO for '" << kernelName << "'. ";
        if (error) std::cerr << error->localizedDescription()->utf8String();
        std::cerr << "\n";
        return nullptr;
    }

    m_kernelCache[sig] = pso;
    return pso;
}

void aixDeviceMetalKernelGen::encodeFusedDispatch(
    MTL::ComputeCommandEncoder* encoder,
    MTL::ComputePipelineState* pso,
    const aix::fuse::FusedSubgraphDescriptor& subgraph,
    const std::unordered_map<const void*, MTL::Buffer*>& allocMap)
{
    encoder->setComputePipelineState(pso);

    const size_t totalBufferCount = subgraph.inputBuffers.size() + subgraph.outputBuffers.size();

    for (size_t i = 0; i < subgraph.inputBuffers.size(); ++i)
    {
        auto allocIt = allocMap.find(subgraph.inputBuffers[i].data);
        if (allocIt != allocMap.end())
        {
            encoder->setBuffer(allocIt->second, 0, i);
        }
    }
    for (size_t i = 0; i < subgraph.outputBuffers.size(); ++i)
    {
        auto allocIt = allocMap.find(subgraph.outputBuffers[i].data);
        if (allocIt != allocMap.end())
        {
            encoder->setBuffer(allocIt->second, 0, subgraph.inputBuffers.size() + i);
        }
    }

    if (!subgraph.allContiguous)
    {
        std::vector<std::vector<size_t>> layoutStorage;
        layoutStorage.reserve(totalBufferCount);

        auto bindLayout = [&](const aix::DeviceTensorParams& params, size_t slot) {
            std::vector<size_t> layout;
            layout.push_back(params.offset);
            layout.push_back(params.shape.size());
            for (size_t s : params.shape)
            {
                layout.push_back(s);
            }

            std::vector<size_t> strides = params.strides;
            if (strides.empty() && !params.shape.empty())
            {
                strides.resize(params.shape.size());
                strides.back() = 1;
                for (int64_t d = static_cast<int64_t>(strides.size()) - 2; d >= 0; --d)
                {
                    strides[d] = strides[d + 1] * params.shape[d + 1];
                }
            }
            for (size_t s : strides)
            {
                layout.push_back(s);
            }

            layoutStorage.push_back(std::move(layout));
            encoder->setBytes(layoutStorage.back().data(),
                              static_cast<NS::UInteger>(layoutStorage.back().size() * sizeof(size_t)),
                              totalBufferCount + slot);
        };

        for (size_t i = 0; i < subgraph.inputBuffers.size(); ++i)
        {
            bindLayout(subgraph.inputBuffers[i], i);
        }
        for (size_t i = 0; i < subgraph.outputBuffers.size(); ++i)
        {
            bindLayout(subgraph.outputBuffers[i], subgraph.inputBuffers.size() + i);
        }
    }

    if (!subgraph.scalarData.empty())
    {
        encoder->setBytes(subgraph.scalarData.data(),
                          static_cast<NS::UInteger>(subgraph.scalarData.size()),
                          totalBufferCount + (subgraph.allContiguous ? 0 : totalBufferCount));
    }

    NS::UInteger gridSize = subgraph.allContiguous
        ? (subgraph.elementCount + 3) / 4
        : subgraph.elementCount;
    NS::UInteger tgSize = std::min(gridSize, pso->maxTotalThreadsPerThreadgroup());

    encoder->dispatchThreads({gridSize, 1, 1}, {tgSize, 1, 1});
}

}  // namespace aix::metal
