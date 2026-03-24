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
#include <aix.hpp>
// External includes
#include <doctest/doctest.h>
// System includes

inline auto Approx(auto value, double epsilon = 1e-4)
{
    return doctest::Approx(value).epsilon(epsilon);
}

inline void CheckVectorApproxValues(const aix::TensorValue& results, const aix::TensorValue& expected, double epsilon = 1e-4)
{
    auto actual = results.contiguous();
    auto expectedValue = expected.contiguous();

    if (actual.device())
    {
        actual.device()->synchronize();
    }
    if (expectedValue.device() && expectedValue.device() != actual.device())
    {
        expectedValue.device()->synchronize();
    }

    if (actual.size() != expectedValue.size())
    {
        throw std::invalid_argument("Tensor data sizes do no match for test result comparison.");
    }

    if (actual.dataType() != expectedValue.dataType())
    {
        throw std::invalid_argument("Tensor data types do no match for test result comparison.");
    }

    if (static_cast<size_t>(actual.dataType()) >= aix::DataTypeCount)
    {
        throw std::invalid_argument("CheckVectorApproxValues does not support the new data type.");
    }

    if (actual.dataType() == aix::DataType::kFloat64)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<double>()[i] == Approx(expectedValue.data<double>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kFloat32)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<float>()[i] == Approx(expectedValue.data<float>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kFloat16)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<aix::float16_t>()[i] == Approx(expectedValue.data<aix::float16_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kBFloat16)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<aix::bfloat16_t>()[i] == Approx(expectedValue.data<aix::bfloat16_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kInt64)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<int64_t>()[i] == Approx(expectedValue.data<int64_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kInt32)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<int32_t>()[i] == Approx(expectedValue.data<int32_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kInt16)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<int16_t>()[i] == Approx(expectedValue.data<int16_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kInt8)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<int8_t>()[i] == Approx(expectedValue.data<int8_t>()[i], epsilon));
        }
    }
    else if (actual.dataType() == aix::DataType::kUInt8)
    {
        for (size_t i=0; i<expectedValue.size(); ++i)
        {
            CHECK(actual.data<uint8_t>()[i] == Approx(expectedValue.data<uint8_t>()[i], epsilon));
        }
    }
}

inline void CheckVectorApproxValues(const aix::Tensor & results, const aix::Tensor & expected, double epsilon = 1e-4)
{
    CheckVectorApproxValues(results.value().contiguous(), expected.value().contiguous(), epsilon);
}
