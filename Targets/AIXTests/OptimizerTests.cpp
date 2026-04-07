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


TEST_CASE("Simple optimizer test")
{
    auto x = tensor(2, { .m_requireGrad=true });
    auto y = tensor(3, { .m_requireGrad=true });
    auto t = tensor(4, { .m_requireGrad=true });
    auto u = tensor(5, { .m_requireGrad=true });

    // Create an instance of the SGD optimizer with a specified learning rate.
    optim::SGD optimizer({x, y, t, u}, 0.001f);

    for (size_t i=0; i<100; ++i)
    {
        auto z = x * (x + y) / t - y * y;
        auto m = x * z + sin(u) * u;

        optimizer.zeroGrad();
        m.backward();      // ∂m/∂m = 1.
        optimizer.step();
    }

    CHECK(x.dataType() == DataType::kFloat32);
    CHECK(x.value().item<float>() == Approx(2.59247));
    CHECK(y.value().item<float>() == Approx(4.53346));
    CHECK(t.value().item<float>() == Approx(4.18035));
    CHECK(u.value().item<float>() == Approx(4.96434));
    CHECK(x.grad().item<float>()  == Approx(-9.98867));
    CHECK(y.grad().item<float>()  == Approx(-21.7066));
    CHECK(t.grad().item<float>()  == Approx(-2.71093));
    CHECK(u.grad().item<float>()  == Approx(0.27058));
    // Note: Results are consistent with those from PyTorch.
}


TEST_CASE("Adam optimizer test")
{
    auto x = tensor(2, { .m_requireGrad=true });
    auto y = tensor(3, { .m_requireGrad=true });
    auto t = tensor(4, { .m_requireGrad=true });
    auto u = tensor(5, { .m_requireGrad=true });

    // Create an instance of the Adam optimizer with a specified learning rate.
    optim::Adam optimizer({x, y, t, u}, 0.01f);

    for (size_t i=0; i<100; ++i)
    {
        auto z = x * (x + y) / t - y * y;
        auto m = x * z + sin(u) * u;

        optimizer.zeroGrad();
        m.backward();      // ∂m/∂m = 1.
        optimizer.step();
    }

    CHECK(x.dataType() == DataType::kFloat32);
    CHECK(x.value().item<float>() == Approx(3.12134));
    CHECK(y.value().item<float>() == Approx(4.12712));
    CHECK(t.value().item<float>() == Approx(5.12626));
    CHECK(u.value().item<float>() == Approx(4.91345));
    CHECK(x.grad().item<float>()  == Approx(-6.25593));
    CHECK(y.grad().item<float>()  == Approx(-23.6874));
    CHECK(t.grad().item<float>()  == Approx(-2.66895));
    CHECK(u.grad().item<float>()  == Approx(0.00182182));
    // Note: Results are consistent with those from PyTorch.
}


TEST_CASE("Optimizer - NoGradGuard does not change trainable filter")
{
    auto trainable = tensor(2.0f, { .m_requireGrad=true });
    auto frozen = tensor(3.0f, { .m_requireGrad=false });
    optim::SGD optimizer({trainable, frozen}, 0.1f);

    {
        NoGradGuard guard;
        auto noGradOutput = trainable + frozen;

        CHECK(noGradOutput.isRequireGrad() == false);
        CHECK(noGradOutput.value().item<float>() == Approx(5.0f));
    }

    CHECK(trainable.isRequireGrad() == true);
    CHECK(frozen.isRequireGrad() == false);

    const float trainableBeforeStep = trainable.value().item<float>();
    const float frozenBeforeStep = frozen.value().item<float>();

    optimizer.zeroGrad();

    auto loss = trainable * trainable + frozen;
    CHECK(loss.isRequireGrad() == true);

    loss.backward();
    optimizer.step();

    CHECK(trainable.value().item<float>() != Approx(trainableBeforeStep));
    CHECK(trainable.value().item<float>() == Approx(1.6f));
    CHECK(frozen.value().item<float>() == Approx(frozenBeforeStep));
}
