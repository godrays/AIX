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
// External includes
#include <doctest/doctest.h>
// System includes

using namespace aix;


TEST_CASE("Loss Func - BinaryCrossEntropy - Scalar")
{
    auto pred    = aix::tensor(0.1);
    auto target  = aix::tensor(0.2);

    auto bceLoss = aix::nn::BinaryCrossEntropyLoss();
    auto loss    = bceLoss(pred, target);

    loss.backward();   // grad [1] shape 1x1

    CHECK(loss.dataType() == DataType::kFloat32);
    CHECK(loss.value().item<float>() == doctest::Approx(0.544805));
    // Note: Results are consistent with those from PyTorch.
}


TEST_CASE("Loss Func - BinaryCrossEntropy - 2x2")
{
    Shape  shape{2, 2};

    auto pred    = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape);
    auto target  = aix::tensor({0.2, 0.3, 0.4, 0.5}, shape);

    auto bceLoss = aix::nn::BinaryCrossEntropyLoss();
    auto loss    = bceLoss(pred, target);

    loss.backward();

    CHECK(loss.dataType() == DataType::kFloat32);
    CHECK(loss.value().item<float>() == doctest::Approx(0.648247));
    // Note: Results are consistent with those from PyTorch.
}
