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
#include <aixDevices.hpp>
// External includes
#include <doctest/doctest.h>
// System includes

using namespace aix;


struct TestModel : public aix::nn::Module
{
    TestModel(const std::initializer_list<float> & xData,
              const std::initializer_list<float> & yData,
              const std::initializer_list<float> & tData,
              const std::initializer_list<float> & uData,
              const Shape & shape)
    {
        m_x = tensor(xData, shape, { .m_requireGrad=true });
        m_y = tensor(yData, shape, { .m_requireGrad=true });
        m_t = tensor(tData, shape, { .m_requireGrad=true });
        m_u = tensor(uData, shape, { .m_requireGrad=true });

        registerParameter("m_x", m_x);
        registerParameter("m_y", m_y);
        registerParameter("m_t", m_t);
        registerParameter("m_u", m_u);
    }

    Tensor forward([[maybe_unused]] Tensor x) const final
    {
        auto z = m_x * (m_x + m_y) / m_t - tanh(m_y * m_y);
        auto m = m_x * z + sin(m_u) * m_u;
        return m;
    }

    Tensor  m_x;
    Tensor  m_y;
    Tensor  m_t;
    Tensor  m_u;
};


TEST_CASE("Auto Grad - Module Test - 1x1 Tensor")
{
    auto shape = std::vector<size_t>{};     // Scalar has no dimensions.

    auto tm = TestModel({2},   // x
                        {3},   // y
                        {4},   // t
                        {5},   // u
                        shape);

    auto m = tm.forward({});

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CHECK(tm.m_x.grad().item<float>() == Approx(5));
    CHECK(tm.m_y.grad().item<float>() == Approx(0.999999));
    CHECK(tm.m_t.grad().item<float>() == Approx(-1.25));
    CHECK(tm.m_u.grad().item<float>() == Approx(0.459387));
    CHECK(m.value().item<float>()     == Approx(-1.79462));
}


TEST_CASE("Auto Grad - Module Test - 1x2 Tensor")
{
    auto shape = std::vector<size_t>{1, 2};

    auto tm = TestModel({1, 2},   // x
                        {3, 4},   // y
                        {5, 6},   // t
                        {7, 8},   // u
                        shape);

    auto m = tm.forward({});

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();   // ∂m/∂m = [1, 1]  1x2 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad(), tensor({0.8, 3.66667},       shape).value());
    CheckVectorApproxValues(tm.m_y.grad(), tensor({0.199999, 0.666667}, shape).value());
    CheckVectorApproxValues(tm.m_t.grad(), tensor({-0.16, -0.666667},   shape).value());
    CheckVectorApproxValues(tm.m_u.grad(), tensor({5.9343, -0.174642},  shape).value());
    CheckVectorApproxValues(m,             tensor({4.39891, 9.91487},   shape));
}


TEST_CASE("Auto Grad - Module Test - 2x3 Tensor")
{
    auto shape = std::vector<size_t>{2, 3};

    auto tm = TestModel({ 1,  2,  3,  4,  5,  6},   // x
                        { 7,  8,  9, 10, 11, 12},   // y
                        {13, 14, 15, 16, 17, 18},   // t
                        {19, 20, 21, 22, 23, 24},   // u
                        shape);

    auto m = tm.forward({});

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();   // ∂m/∂m = [1,1,1,1,1,1]  2x3 tensor

    // Check shapes
    CHECK(tm.m_x.grad().shape()  == shape);
    CHECK(tm.m_y.grad().shape()  == shape);
    CHECK(tm.m_t.grad().shape()  == shape);
    CHECK(tm.m_u.grad().shape()  == shape);
    CHECK(tm.m_x.value().shape() == shape);
    CHECK(tm.m_y.value().shape() == shape);
    CHECK(tm.m_t.value().shape() == shape);
    CHECK(tm.m_u.value().shape() == shape);
    CHECK(m.value().shape()      == shape);

    CheckVectorApproxValues(tm.m_x.grad(), tensor({0.307692, 2.14286, 4.4, 7.0, 9.88235, 13.0}, shape).value());
    CheckVectorApproxValues(tm.m_y.grad(), tensor({0.0769231, 0.285714, 0.6, 1.0, 1.47059, 2.0}, shape).value());
    CheckVectorApproxValues(tm.m_t.grad(), tensor({-0.0473373, -0.204082, -0.48, -0.875, -1.38408, -2.0}, shape).value());
    CheckVectorApproxValues(tm.m_u.grad(), tensor({18.9353, 9.07459, -10.6657, -22.008, -13.1014, 9.27472}, shape).value());
    CheckVectorApproxValues(m,             tensor({2.46305, 19.116, 21.7698, 9.80527, -0.933655, 8.26612}, shape));
}


TEST_CASE("Auto Grad with broadcasting")
{
    auto shape1 = Shape{1, 3};
    auto shape2 = Shape{2, 3};

    auto m_x = tensor({ 1.0,  2.0,  3.0},                   shape1, { .m_requireGrad=true });
    auto m_y = tensor({ 7.0,  8.0,  9.0, 10.0, 11.0, 12.0}, shape2, { .m_requireGrad=true });
    auto m_t = tensor({13.0, 14.0, 15.0},                   shape1, { .m_requireGrad=true });
    auto m_u = tensor({19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, shape2, { .m_requireGrad=true });

    auto z = m_x * (m_x + m_y) / m_t - tanh(m_y * m_y);
    auto m = m_x * z + sin(m_u) * m_u;

    // Traverse the graph (starting from the end) to calculate all tensor gradients.
    m.backward();   // ∂m/∂m = [1,1,1,1,1,1]  2x3 tensor

    // Check shapes
    CHECK(m_x.grad().shape()  == shape1);
    CHECK(m_y.grad().shape()  == shape2);
    CHECK(m_t.grad().shape()  == shape1);
    CHECK(m_u.grad().shape()  == shape2);
    CHECK(m_x.value().shape() == shape1);
    CHECK(m_y.value().shape() == shape2);
    CHECK(m_t.value().shape() == shape1);
    CHECK(m_u.value().shape() == shape2);
    CHECK(m.value().shape()   == shape2);

    CheckVectorApproxValues(m_x.grad(), tensor({1.07692, 5.14286, 10.0}, shape1).value());
    CheckVectorApproxValues(m_y.grad(), tensor({0.0769231, 0.285714, 0.6, 0.0769231, 0.285714, 0.6}, shape2).value());
    CheckVectorApproxValues(m_t.grad(), tensor({-0.112426, -0.469388, -1.08}, shape1).value());
    CheckVectorApproxValues(m_u.grad(), tensor({18.9353, 9.07459, -10.6657, -22.008, -13.1014, 9.27472}, shape2).value());
    CheckVectorApproxValues(m,          tensor({2.46305, 19.116, 21.7698, -0.348575, -17.7488, -15.7339}, shape2));
}


TEST_CASE("Auto Grad - log Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, { .m_requireGrad=true });
    auto z = log(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({10.0, 5.0, 3.33333, 2.5}, shape).value());
}


TEST_CASE("Auto Grad - exp Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, { .m_requireGrad=true });
    auto z = exp(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({1.10517, 1.2214, 1.34986, 1.49182}, shape).value());
}


TEST_CASE("Auto Grad - pow Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({1.0, 2.0, 3.0, 4.0}, shape, { .m_requireGrad=true });
    auto exp = aix::tensor({1.0, 2.0, 3.0, 4.0}, shape);
    auto z = pow(x, exp);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({1.0, 4.0, 27.0, 256.0}, shape).value());
}


TEST_CASE("Auto Grad - sum Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, { .m_requireGrad=true });
    auto z = x.sum();
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({1.0, 1.0, 1.0, 1.0}, shape).value());
}


TEST_CASE("Auto Grad - sigmoid Test - 2x2")
{
    aix::Shape shape{2,2};

    auto x = aix::tensor({0.1, 0.2, 0.3, 0.4}, shape, { .m_requireGrad=true });
    auto z = aix::nn::Sigmoid().forward(x);
    z.backward();

    // Check shapes
    CHECK(x.grad().shape() == shape);
    CheckVectorApproxValues(x.grad(), tensor({0.249376, 0.247517, 0.244458, 0.240261}, shape).value());
}


TEST_CASE("Auto Grad - transpose")
{
    SUBCASE("3x2")
    {
        aix::Shape shape{3,2};

        auto x = aix::tensor({1.0,2.0,3.0,4.0,5.0,6.0}, shape, { .m_requireGrad=true });
        auto z = x.transpose(0, 1);
        z.backward(1, {2,3});       // Starting with the transposed shape

        // Check shapes
        CHECK(x.grad().shape() == shape);
        CheckVectorApproxValues(x.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape).value());
    }

    SUBCASE("back propagation initial gradient shape must be transposed")
    {
        aix::Shape shape{3,2};
        auto x = aix::tensor({1.0,2.0,3.0,4.0,5.0,6.0}, shape, { .m_requireGrad=true });
        auto z = x.transpose(0, 1);
        DOCTEST_CHECK_THROWS_AS(z.backward(), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(z.backward(1, {3,2}), std::invalid_argument);
    }
}


TEST_CASE("Auto Grad - Broadcast from [1x3] to [2x3]")
{
    auto shape1 = Shape{1, 3};
    auto shape2 = Shape{2, 3};
    auto data1 = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto data2 = std::initializer_list<float>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    SUBCASE("Add - x+y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x + y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({2.0,2.0,2.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Add - y+x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y + x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({2.0,2.0,2.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Sub - x-y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x - y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({2.0,2.0,2.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({-1.0,-1.0,-1.0,-1.0,-1.0,-1.0}, shape2).value());
    }

    SUBCASE("Sub - y-x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y - x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({-2.0,-2.0,-2.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Mul - x*y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x * y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({17.0,19.0,21.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,2.0,3.0,1.0,2.0,3.0}, shape2).value());
    }

    SUBCASE("Mul - y*x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y * x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({17.0,19.0,21.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,2.0,3.0,1.0,2.0,3.0}, shape2).value());
    }

    SUBCASE("Div - x/y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x / y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({0.242857, 0.215909, 0.194444}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({-0.0204082, -0.03125, -0.037037,
                                                  -0.01, -0.0165289, -0.0208333}, shape2).value());
    }

    SUBCASE("Div - y/x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y / x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({-17.0, -4.75, -2.33333}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0, 0.5, 0.333333, 1.0, 0.5, 0.333333}, shape2).value());
    }
}


TEST_CASE("Auto Grad - Broadcast from Scalar to [2x3]")
{
    auto shape1 = Shape{};      // Scalar has no shape/dimension
    auto shape2 = Shape{2, 3};
    auto data1 = std::initializer_list<float>{5};
    auto data2 = std::initializer_list<float>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    SUBCASE("Add - x+y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x + y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({6.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Add - y+x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y + x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({6.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Sub - x-y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x - y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({6.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({-1.0,-1.0,-1.0,-1.0,-1.0,-1.0}, shape2).value());
    }

    SUBCASE("Sub - y-x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y - x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({-6.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, shape2).value());
    }

    SUBCASE("Mul - x*y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x * y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({57.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({5.0,5.0,5.0,5.0,5.0,5.0}, shape2).value());
    }

    SUBCASE("Mul - y*x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y * x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({57.0}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({5.0,5.0,5.0,5.0,5.0,5.0}, shape2).value());
    }

    SUBCASE("Div - x/y")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = x / y;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({0.653211}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({-0.102041, -0.078125, -0.0617284,
                                                  -0.05, -0.0413223, -0.0347222}, shape2).value());
    }

    SUBCASE("Div - y/x")
    {
        auto x = aix::tensor(data1, shape1, { .m_requireGrad=true });
        auto y = aix::tensor(data2, shape2, { .m_requireGrad=true });
        auto z = y / x;
        z.backward();
        CHECK(x.grad().shape() == shape1);
        CHECK(y.grad().shape() == shape2);
        CheckVectorApproxValues(x.grad(), tensor({-2.28}, shape1).value());
        CheckVectorApproxValues(y.grad(), tensor({0.2,0.2,0.2,0.2,0.2,0.2}, shape2).value());
    }
}


TEST_CASE("Auto Grad - sum with dimension")
{
    auto t  = aix::tensor({1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0,
                           10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0,
                           16.0, 17.0, 18.0,
                           19.0, 20.0, 21.0,
                           22.0, 23.0, 24.0}, aix::Shape{3, 4, 2}, { .m_requireGrad=true });

    SUBCASE("Shape{3,4,2} - dim=0 keepDim=false")
    {
        auto sum = t.sum(0, false);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{4, 2});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }

    SUBCASE("Shape{3,4,2} - dim=0 keepDim=true")
    {
        auto sum = t.sum(0, true);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{1, 4, 2});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }

    SUBCASE("Shape{3,4,2} - dim=1 keepDim=false")
    {
        auto sum = t.sum(1, false);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{3, 2});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }

    SUBCASE("Shape{3,4,2} - dim=1 keepDim=true")
    {
        auto sum = t.sum(1, true);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{3, 1, 2});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }

    SUBCASE("Shape{3,4,2} - dim=2 keepDim=false")
    {
        auto sum = t.sum(2, false);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{3, 4});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }

    SUBCASE("Shape{3,4,2} - dim=2 keepDim=true")
    {
        auto sum = t.sum(2, true);
        sum.retainGrad();
        sum.backward(1, sum.shape());
        CHECK(t.grad().shape() == t.shape());
        CHECK(sum.grad().shape() == Shape{3, 4, 1});
        CheckVectorApproxValues(t.grad(), aix::onesLike(t).value());
        CheckVectorApproxValues(sum.grad(), aix::onesLike(sum).value());
    }
}


TEST_CASE("Auto Grad - sum with dimension - complex")
{
    auto a  = aix::tensor({ 1.0,  2.0,  3.0,
                            4.0,  5.0,  6.0,
                            7.0,  8.0,  9.0,
                           10.0, 11.0, 12.0,
                           13.0, 14.0, 15.0,
                           16.0, 17.0, 18.0,
                           19.0, 20.0, 21.0,
                           22.0, 23.0, 24.0}, aix::Shape{3, 4, 2}, { .m_requireGrad=true });

    SUBCASE("Complex 1")
    {
        auto b = aix::tensor({1.0,2.0,3.0}, aix::Shape{3}, { .m_requireGrad=true });
        auto z = a.sum(1, false).sum(1, true);
        z.retainGrad();
        auto sum = z * b;
        sum.backward();

        CHECK(z.shape() == Shape{3,1});
        CHECK(z.grad().shape() == Shape{3,1});
        CHECK(a.grad().shape() == Shape{3,4,2});

        CheckVectorApproxValues(z, aix::tensor({36.0, 100.0, 164.0}, z.shape()));
        CheckVectorApproxValues(z.grad(), aix::Tensor(6.0, z.shape()).value());
        CheckVectorApproxValues(a.grad(), aix::Tensor(6.0, a.shape()).value());
    }

    SUBCASE("Complex 2")
    {
        auto a2 = aix::Tensor(5.0, aix::Shape{3, 4, 2}, { .m_requireGrad=true });
        auto b = aix::Tensor(5.0, aix::Shape{3, 2}, { .m_requireGrad=true });
        auto b2 = aix::tensor({1.0,2.0,3.0}, aix::Shape{3}, { .m_requireGrad=true });

        auto sum = ((a * a2).sum(1, false) / b).sum(1, true);
        sum.retainGrad();
        sum.backward(1, sum.shape());

        CHECK(a.grad().shape() == Shape{3,4,2});
        CHECK(a2.grad().shape() == Shape{3,4,2});
        CHECK(sum.grad().shape() == Shape{3,1});
        CHECK(sum.shape() == Shape{3,1});

        CheckVectorApproxValues(a.grad(), aix::onesLike(a).value());
        CheckVectorApproxValues(a2.grad(), aix::tensor({0.2, 0.4,
                                                        0.6, 0.8,
                                                        1.0, 1.2,
                                                        1.4, 1.6,
                                                        1.8, 2.0,
                                                        2.2, 2.4,
                                                        2.6, 2.8,
                                                        3.0, 3.2,
                                                        3.4, 3.6,
                                                        3.8, 4.0,
                                                        4.2, 4.4,
                                                        4.6, 4.8}, aix::Shape{3,4,2}).value());
        CheckVectorApproxValues(sum.grad(), aix::tensor({1.0,1.0,1.0}, aix::Shape{3,1}).value());
        CheckVectorApproxValues(sum.value(), aix::tensor({36.0,100.0,164.0}, aix::Shape{3,1}).value());
    }

    SUBCASE("Complex 3")
    {
        auto a2 = aix::Tensor(5.0, aix::Shape{3, 4, 2}, { .m_requireGrad=true });
        auto b = aix::Tensor(5.0, aix::Shape{3, 2}, { .m_requireGrad=true });
        auto b2 = aix::tensor({1.0,2.0,3.0}, aix::Shape{3}, { .m_requireGrad=true });

        auto sum = b2 * ((a * a2).sum(1, false) / b).sum(1, true);
        sum.retainGrad();
        sum.backward();

        CHECK(a.grad().shape() == Shape{3,4,2});
        CHECK(a2.grad().shape() == Shape{3,4,2});
        CHECK(sum.grad().shape() == Shape{3,3});
        CHECK(sum.shape() == Shape{3,3});

        CheckVectorApproxValues(a.grad(), aix::Tensor(6.0, a.shape()).value());
        CheckVectorApproxValues(a2.grad(), aix::tensor({   1.2,  2.4,
                                                           3.6,  4.8,
                                                           6.0,  7.2,
                                                           8.4,  9.6,
                                                           10.8, 12.0,
                                                           13.2, 14.4,
                                                           15.6, 16.8,
                                                           18.0, 19.2,
                                                           20.4, 21.6,
                                                           22.8, 24.0,
                                                           25.2, 26.4,
                                                           27.6, 28.8}, aix::Shape{3,4,2}).value());
        CheckVectorApproxValues(sum.grad(), aix::Tensor(1.0, aix::Shape{3,3}).value());
        CheckVectorApproxValues(sum.value(), aix::tensor({ 36.0,   72.0, 108.0,
                                                           100.0,  200.0, 300.0,
                                                           164.0,  328.0, 492.0}, aix::Shape{3,3}).value());
    }

    SUBCASE("Complex 4")
    {
        auto a2 = aix::tensor({4.0f, 5.0f, 6.0f}, aix::Shape{3, 1}, { .m_requireGrad=true });
        auto b = aix::tensor({1.0f, 2.0f, 3.0f}, aix::Shape{3, 1}, { .m_requireGrad=true });

        auto z = a2 * b;
        z.retainGrad();
        auto sum = z;
        sum.backward(1, sum.shape());

        CHECK(z.shape() == Shape{3,1});
        CHECK(z.grad().shape() == Shape{3,1});
        CHECK(a2.grad().shape() == Shape{3,1});

        CheckVectorApproxValues(z, aix::tensor({4.0, 10.0, 18.0}, z.shape()));
        CheckVectorApproxValues(z.grad(), aix::Tensor(1.0, z.shape()).value());
        CheckVectorApproxValues(a2.grad(), aix::tensor({1.0,2.0,3.0}, a2.shape()).value());
    }

    SUBCASE("Complex 5")
    {
        auto a2 = aix::tensor({4.0f, 5.0f, 6.0f}, aix::Shape{3, 1}, { .m_requireGrad=true });
        auto b = aix::tensor({1.0f, 2.0f, 3.0f}, aix::Shape{3}, { .m_requireGrad=true });

        auto z = a2 * b;
        z.retainGrad();
        auto sum = z;
        sum.backward(1, sum.shape());

        CHECK(z.shape() == Shape{3,3});
        CHECK(z.grad().shape() == Shape{3,3});
        CHECK(a2.grad().shape() == Shape{3,1});

        CheckVectorApproxValues(z, aix::tensor({4.0, 8.0, 12.0,
                                                5.0, 10.0, 15.0,
                                                6.0, 12.0, 18.0}, z.shape()));
        CheckVectorApproxValues(z.grad(), aix::Tensor(1.0, z.shape()).value());
        CheckVectorApproxValues(a2.grad(), aix::Tensor(6.0, a2.shape()).value());
    }
}


TEST_CASE("Auto Grad - Squeeze")
{
    std::initializer_list<float> data = { 1.0, 2.0, 3.0, 4.0 };
    Shape shape{2,1,2};

    SUBCASE("dim 1")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto s = a.squeeze(1);
        s.backward(1, s.shape());
        CheckVectorApproxValues(a.grad(), aix::tensor({1.0, 1.0, 1.0, 1.0}, a.shape()).value());
    }
}


TEST_CASE("Auto Grad - Unsqueeze")
{
    std::initializer_list<float> data = { 1.0, 2.0, 3.0, 4.0 };
    Shape shape{2,2};

    SUBCASE("dim 1")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto s = a.unsqueeze(1);
        s.backward(1, s.shape());
        CheckVectorApproxValues(a.grad(), aix::tensor({1.0, 1.0, 1.0, 1.0}, a.shape()).value());
    }
}


TEST_CASE("Auto Grad - variance")
{
    std::initializer_list<float> data = { 1.0, 2.0,
                                          3.0, 4.0 };
    Shape shape{2,2};

    SUBCASE("default")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var();
        var.backward(1, var.shape());
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0000, -0.3333,
                                                        0.3333,  1.0000}, a.shape()).value());
    }

    SUBCASE("unbiased = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var  = a.var(true);
        var.backward(1, var.shape());
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0000, -0.3333,
                                                        0.3333,  1.0000}, a.shape()).value());
    }

    SUBCASE("unbiased = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var  = a.var(false);
        var.backward(1, var.shape());
        CheckVectorApproxValues(a.grad(), aix::tensor({-0.7500, -0.2500,
                                                        0.2500,  0.7500}, a.shape()).value());
    }

    SUBCASE("dim = 0 unbiased = default, keepdim = default")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(ssize_t(0));
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-2.0, -2.0,
                                                        2.0,  2.0}, shape).value());
    }

    SUBCASE("dim = 0 unbiased = true, keepdim = default")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(ssize_t(0), true);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-2.0, -2.0,
                                                        2.0,  2.0}, shape).value());
    }

    // ---

    SUBCASE("dim = 0 unbiased = true, keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(0, true, false);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-2.0, -2.0,
                                                        2.0,  2.0}, shape).value());
    }

    SUBCASE("dim = 0 unbiased = true, keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(0, true, true);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-2.0, -2.0,
                                                        2.0,  2.0}, shape).value());
    }

    SUBCASE("dim = 0 unbiased = false, keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(0, false, false);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0, -1.0,
                                                        1.0,  1.0}, shape).value());
    }

    SUBCASE("dim = 0 unbiased = false, keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(0, false, true);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{1, 2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0, -1.0,
                                                        1.0,  1.0}, shape).value());
    }

    // ---

    SUBCASE("dim = 1 unbiased = true, keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(1, true, false);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0, 1.0,
                                                       -1.0, 1.0}, shape).value());
    }

    SUBCASE("dim = 1 unbiased = true, keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(1, true, true);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), aix::tensor({-1.0, 1.0,
                                                       -1.0, 1.0}, shape).value());
    }

    SUBCASE("dim = 1 unbiased = false, keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(1, false, false);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2});
        CheckVectorApproxValues(a.grad(), aix::tensor({-0.5, 0.5,
                                                       -0.5, 0.5}, shape).value());
    }

    SUBCASE("dim = 1 unbiased = false, keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto var = a.var(1, false, true);
        var.backward(1, var.shape());
        CHECK(var.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), aix::tensor({-0.5, 0.5,
                                                       -0.5, 0.5}, shape).value());
    }
}


TEST_CASE("Auto Grad - Metal view chain permute slice mul backward parity")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuInput = aix::tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f,
                                 21.0f, 22.0f, 23.0f, 24.0f}, aix::Shape{2, 3, 4}, { .m_requireGrad=true });
    auto metalInput = aix::tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f,
                                   17.0f, 18.0f, 19.0f, 20.0f,
                                   21.0f, 22.0f, 23.0f, 24.0f}, aix::Shape{2, 3, 4},
                                  { .m_requireGrad=true, .m_device=device.get() });

    auto cpuView = cpuInput.permute({1, 2, 0}).slice(1, 1, 4, 2);
    auto metalView = metalInput.permute({1, 2, 0}).slice(1, 1, 4, 2);

    CHECK_FALSE(cpuView.value().isContiguous());
    CHECK_FALSE(metalView.value().isContiguous());

    auto cpuLoss = (cpuView * cpuView).sum();
    auto metalLoss = (metalView * metalView).sum();

    cpuLoss.backward();
    metalLoss.backward();
    device->synchronize();

    CheckVectorApproxValues(metalLoss, cpuLoss);
    CheckVectorApproxValues(metalInput.grad(), cpuInput.grad());
}


TEST_CASE("Auto Grad - Metal Broadcast view reduceTo backward parity")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuInput = aix::tensor({1.0f, 2.0f, 3.0f}, aix::Shape{1, 3}, { .m_requireGrad=true });
    auto cpuOther = aix::tensor({4.0f, 5.0f, 6.0f,
                                 7.0f, 8.0f, 9.0f}, aix::Shape{2, 3}, { .m_requireGrad=true });
    auto metalInput = aix::tensor({1.0f, 2.0f, 3.0f}, aix::Shape{1, 3},
                                  { .m_requireGrad=true, .m_device=device.get() });
    auto metalOther = aix::tensor({4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f}, aix::Shape{2, 3},
                                  { .m_requireGrad=true, .m_device=device.get() });

    auto cpuView = cpuInput.broadcastTo({2, 3});
    auto metalView = metalInput.broadcastTo({2, 3});

    CHECK_FALSE(cpuView.value().isContiguous());
    CHECK_FALSE(metalView.value().isContiguous());

    auto cpuLoss = (cpuView * cpuOther).sum();
    auto metalLoss = (metalView * metalOther).sum();

    cpuLoss.backward();
    metalLoss.backward();
    device->synchronize();

    CheckVectorApproxValues(metalLoss, cpuLoss);
    CheckVectorApproxValues(metalInput.grad(), cpuInput.grad());
    CheckVectorApproxValues(metalOther.grad(), cpuOther.grad());
}


TEST_CASE("Auto Grad - transpose matmul backward parity")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuLhs = aix::tensor({1.0f, 2.0f,
                               3.0f, 4.0f,
                               5.0f, 6.0f}, aix::Shape{3, 2}, { .m_requireGrad=true }).transpose(0, 1);
    auto cpuRhs = aix::tensor({7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f}, aix::Shape{3, 4}, { .m_requireGrad=true });
    auto metalLhs = aix::tensor({1.0f, 2.0f,
                                 3.0f, 4.0f,
                                 5.0f, 6.0f}, aix::Shape{3, 2},
                                { .m_requireGrad=true, .m_device=device.get() }).transpose(0, 1);
    auto metalRhs = aix::tensor({7.0f, 8.0f, 9.0f, 10.0f,
                                 11.0f, 12.0f, 13.0f, 14.0f,
                                 15.0f, 16.0f, 17.0f, 18.0f}, aix::Shape{3, 4},
                                { .m_requireGrad=true, .m_device=device.get() });

    CHECK_FALSE(cpuLhs.value().isContiguous());
    CHECK_FALSE(metalLhs.value().isContiguous());

    auto cpuLoss = cpuLhs.matmul(cpuRhs).sum();
    auto metalLoss = metalLhs.matmul(metalRhs).sum();

    cpuLoss.backward();
    metalLoss.backward();
    device->synchronize();

    CheckVectorApproxValues(metalLoss, cpuLoss);
    CheckVectorApproxValues(metalLhs.grad(), cpuLhs.grad());
    CheckVectorApproxValues(metalRhs.grad(), cpuRhs.grad());
}


TEST_CASE("Auto Grad - max")
{
    SUBCASE("scalar")
    {
        auto a = aix::tensor(5.0, {}).requireGrad(true);
        auto max = a.max();
        max.backward();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 5.0);
        CHECK(a.grad().shape() == Shape{});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("1x1 tensor")
    {
        auto a = aix::tensor({5.0}, {1, 1}).requireGrad(true);
        auto max = a.max();
        max.backward();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 5.0);
        CHECK(a.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("2x2 tensor - first is max")
    {
        auto a = aix::tensor({10.0, 9.0, 8.0, -10.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        max.backward();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 10.0);
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0, 0.0, 0.0, 0.0 }, a.shape()).value());
    }

    SUBCASE("2x2 tensor - last is max")
    {
        auto a = aix::tensor({-10.0, 9.0, 8.0, 10.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        max.backward();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 10.0);
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 0.0, 1.0 }, a.shape()).value());
    }

    SUBCASE("2x2 tensor - first found max")
    {
        auto a = aix::tensor({7.0, 8.0, 9.0, 9.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        max.backward();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 9.0);
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 1.0, 0.0 }, a.shape()).value());
    }

    SUBCASE("2x2 tensor - complex")
    {
        auto a = aix::tensor({1.0, 4.0, 3.0, 2.0}, {2,2}).requireGrad(true);
        auto max = a.max() * a;
        max.backward(1, max.shape());
        CHECK(max.shape() == Shape{2,2});
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 4.0, 14.0, 4.0, 4.0 }, a.shape()).value());
    }
}


TEST_CASE("Auto Grad - argmax")
{
    SUBCASE("scalar")
    {
        auto a = aix::tensor(5.0, {}).requireGrad(true);
        auto amax = a.argmax();
        amax.backward(1, amax.shape());
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
        CHECK(a.grad().shape() == Shape{});
        CheckVectorApproxValues(a.grad(), aix::Tensor(0.0, a.shape()).value());
    }

    SUBCASE("1 tensor")
    {
        auto a = aix::Tensor(5.0, Shape{1}).requireGrad(true);
        auto amax = a.argmax();
        amax.backward(1, amax.shape());
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
        CHECK(a.grad().shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), aix::Tensor(0.0, a.shape()).value());
    }

    SUBCASE("1x1 tensor")
    {
        auto a = aix::Tensor(5.0, Shape{1,1}).requireGrad(true);
        auto amax = a.argmax();
        amax.backward(1, amax.shape());
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
        CHECK(a.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), aix::Tensor(0.0, a.shape()).value());
    }

    SUBCASE("2x2 tensor")
    {
        auto a = aix::tensor({1.0,2.0,3.0,4.0}, Shape{2,2}).requireGrad(true);
        auto amax = a.argmax();
        amax.backward(1, amax.shape());
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 3);
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::Tensor(0.0, a.shape()).value());
    }

    SUBCASE("2x2 tensor - complex")
    {
        auto a = aix::tensor({1.0,4.0,3.0,2.0}, Shape{2,2}).requireGrad(true);
        auto amax = a.argmax() * a;
        amax.backward(1, amax.shape());
        CHECK(amax.shape() == Shape{2,2});
        CHECK(a.grad().shape() == Shape{2,2});
        CheckVectorApproxValues(a.grad(), aix::Tensor(1.0, a.shape()).value());
    }
}


TEST_CASE("Auto Grad - max with dimension")
{
    std::initializer_list<float> data = { 1.0, 2.0, 3.0,
                                          4.0, 6.0, 5.0,
                                          9.0, 8.0, 7.0 };
    Shape shape{3,3};

    SUBCASE("{} dim = 0 keepdim = false")
    {
        auto a = aix::Tensor(5, {}).requireGrad(true);
        auto max = a.max(0, false);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("{} dim = 0 keepdim = true")
    {
        auto a = aix::Tensor(5, {}).requireGrad(true);
        auto max = a.max(0, true);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("{1} dim = 0 keepdim = false")
    {
        auto a = aix::Tensor(5, {1}).requireGrad(true);
        auto max = a.max(0, false);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("{1} dim = 0 keepdim = true")
    {
        auto a = aix::Tensor(5, {1}).requireGrad(true);
        auto max = a.max(0, true);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("{1,1} dim = 0 keepdim = false")
    {
        auto a = aix::Tensor(5, {1,1}).requireGrad(true);
        auto max = a.max(0, false);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }

    SUBCASE("{1,1} dim = 0 keepdim = true")
    {
        auto a = aix::Tensor(5, {1,1}).requireGrad(true);
        auto max = a.max(0, true);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 1.0 }, a.shape()).value());
    }


    SUBCASE("{3,3} dim = 0 keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto max = a.max(0, false);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{3,3});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0,
                                                        1.0, 1.0, 1.0 }, a.shape()).value());
    }

    SUBCASE("{3,3} dim = 0 keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto max = a.max(0, true);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{3,3});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0,
                                                        1.0, 1.0, 1.0 }, a.shape()).value());
    }

    // ---

    SUBCASE("{3,3} dim = 1 keepdim = false")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto max = a.max(1, false);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{3,3});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 1.0,
                                                        0.0, 1.0, 0.0,
                                                        1.0, 0.0, 0.0 }, a.shape()).value());
    }

    SUBCASE("{3,3} dim = 1 keepdim = true")
    {
        auto a = aix::tensor(data, shape).requireGrad(true);
        auto max = a.max(1, true);
        max.backward(1, max.shape());
        CHECK(a.shape() == Shape{3,3});
        CheckVectorApproxValues(a.grad(), aix::tensor({ 0.0, 0.0, 1.0,
                                                        0.0, 1.0, 0.0,
                                                        1.0, 0.0, 0.0 }, a.shape()).value());
    }
}


TEST_CASE("Auto Grad - slice")
{
    auto t33  = aix::tensor({1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0}, aix::Shape{3,3}).requireGrad(true);

    auto t222  = aix::tensor({1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0}, aix::Shape{2,2,2}).requireGrad(true);

    // Default parameters

    SUBCASE("Shape{1} - default parameters")
    {
        auto t3  = aix::tensor({5.0}, aix::Shape{1}).requireGrad(true);
        auto t = t3.slice();
        t.backward(1, t.shape());
        CHECK(t3.grad().shape() == Shape{1});
        CheckVectorApproxValues(t3.grad(), aix::Tensor(1.0, t3.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - default parameters")
    {
        auto t3  = aix::tensor({5.0}, aix::Shape{1,1}).requireGrad(true);
        auto t = t3.slice();
        t.backward(1, t.shape());
        CHECK(t3.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t3.grad(), aix::Tensor(1.0, t3.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - default parameters")
    {
        auto t = t33.slice();
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::Tensor(1.0, t33.grad().shape()).value());
    }

    // Dim 0

    SUBCASE("Shape{1,1} - dim=0 start=0 end=1 step=1")
    {
        auto t11  = aix::tensor({5.0}, aix::Shape{1,1}).requireGrad(true);
        auto t = t11.slice(0,0,1,1);
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(1.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=1")
    {
        auto t = t33.slice(0,0,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::Tensor(1.0, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=2 step=1")
    {
        auto t = t33.slice(0,0,2,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=-2 start=-3 end=-1 step=1")
    {
        auto t = t33.slice(-2,-3,-1,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=1 step=1")
    {
        auto t = t33.slice(0,0,1,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=1")
    {
        auto t = t33.slice(0,1,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0,
                                                          1.0, 1.0, 1.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=1")
    {
        auto t = t33.slice(0,2,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=2")
    {
        auto t = t33.slice(0,0,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=3")
    {
        auto t = t33.slice(0,0,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=4")
    {
        auto t = t33.slice(0,0,3,4);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=2")
    {
        auto t = t33.slice(0,1,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=3")
    {
        auto t = t33.slice(0,1,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=2")
    {
        auto t = t33.slice(0,2,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=3")
    {
        auto t = t33.slice(0,2,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          1.0, 1.0, 1.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=0 end=2 step=1")
    {
        auto t = t33.slice(0,0,2,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 1.0, 1.0, 1.0,
                                                          1.0, 1.0, 1.0,
                                                          0.0, 0.0, 0.0, }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=0 end=1 step=1")
    {
        auto t = t222.slice(0,0,1,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 1.0, 1.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           0.0, 0.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=1 end=2 step=1")
    {
        auto t = t222.slice(0,1,2,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 0.0, 0.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=1 end=2 step=2")
    {
        auto t = t222.slice(0,1,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 0.0, 0.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-3 start=-2 end=2 step=2")
    {
        auto t = t222.slice(-3,-2,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 1.0, 1.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           0.0, 0.0, }, t222.grad().shape()).value());
    }

    // Dim 1

    SUBCASE("Shape{1,1} - dim=1 start=0 end=1 step=1")
    {
        auto t3  = aix::tensor({5.0}, aix::Shape{1,1}).requireGrad(true);
        auto t = t3.slice(1,0,1,1);
        t.backward(1, t.shape());
        CHECK(t3.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t3.grad(), aix::Tensor(1.0, t3.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=1")
    {
        auto t = t33.slice(1,0,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::Tensor(1.0, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=2 step=1")
    {
        auto t = t33.slice(1,0,2,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          1.0, 1.0, 0.0,
                                                          1.0, 1.0, 0.0,
                                                          1.0, 1.0, 0.0,
                                                         }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=1 step=1")
    {
        auto t = t33.slice(1,0,1,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=1")
    {
        auto t = t33.slice(1,1,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 1.0, 1.0,
                                                          0.0, 1.0, 1.0,
                                                          0.0, 1.0, 1.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=2 end=3 step=1")
    {
        auto t = t33.slice(1,2,3,1);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=2")
    {
        auto t = t33.slice(1,0,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          1.0, 0.0, 1.0,
                                                          1.0, 0.0, 1.0,
                                                          1.0, 0.0, 1.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=3")
    {
        auto t = t33.slice(1,0,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=4")
    {
        auto t = t33.slice(1,0,3,4);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                          1.0, 0.0, 0.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=2")
    {
        auto t = t33.slice(1,1,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 1.0, 0.0,
                                                          0.0, 1.0, 0.0,
                                                          0.0, 1.0, 0.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=3")
    {
        auto t = t33.slice(1,1,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 1.0, 0.0,
                                                          0.0, 1.0, 0.0,
                                                          0.0, 1.0, 0.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=2 end=3 step=2")
    {
        auto t = t33.slice(1,2,3,2);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 start=2 end=3 step=3")
    {
        auto t = t33.slice(1,2,3,3);
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                          0.0, 0.0, 1.0,
                                                        }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=0 end=2 step=1")
    {
        auto t = t222.slice(1,0,2,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=0 end=1 step=1")
    {
        auto t = t222.slice(1,0,1,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=1 end=2 step=1")
    {
        auto t = t222.slice(1,1,2,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=1 end=2 step=2")
    {
        auto t = t222.slice(1,1,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-2 start=1 end=2 step=2")
    {
        auto t = t222.slice(-2,1,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                           0.0, 0.0,
                                                           1.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    // Dim 2

    SUBCASE("Shape{2,2,2} - dim=2 start=0 end=2 step=1")
    {
        auto t = t222.slice(2,0,2,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                           1.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=0 end=1 step=1")
    {
        auto t = t222.slice(2,0,1,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           1.0, 0.0,
                                                           1.0, 0.0,
                                                           1.0, 0.0,
                                                           1.0, 0.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=1 end=2 step=1")
    {
        auto t = t222.slice(2,1,2,1);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=1 end=2 step=2")
    {
        auto t = t222.slice(2,1,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-1 start=1 end=2 step=2")
    {
        auto t = t222.slice(-1,1,2,2);
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                           0.0, 1.0,
                                                         }, t222.grad().shape()).value());
    }

    SUBCASE("invalid parameters")
    {
        CHECK_THROWS_AS({ aix::Tensor(5.0, aix::Shape{}).slice(); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.slice(0, 0, 1, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.slice(0, 0, 0, 1); }, std::invalid_argument);
    }
}


TEST_CASE("Auto Grad - tril")
{
    auto ts  = aix::tensor(5.0).requireGrad(true);
    auto t1  = aix::tensor({5.0}, aix::Shape{1}).requireGrad(true);
    auto t11  = aix::tensor({5.0}, aix::Shape{1,1}).requireGrad(true);
    auto t33  = aix::tensor({1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0}, aix::Shape{3,3}).requireGrad(true);
    auto t222  = aix::tensor({1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0}, aix::Shape{2,2,2}).requireGrad(true);

    // Default parameters

    SUBCASE("Shape{1,1} - diagonal=default")
    {
        auto t = t11.tril() * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(10.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=default")
    {
        auto t = t33.tril() * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                           2.0,  0.0,  0.0,
                                                           8.0, 10.0,  0.0,
                                                          14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=1")
    {
        auto t = t11.tril(1) * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(10.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=-1")
    {
        auto t = t11.tril(-1) * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(0.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=0")
    {
        auto t = t33.tril(0) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  2.0,  0.0,  0.0,
                                                           8.0, 10.0,  0.0,
                                                          14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=1")
    {
        auto t = t33.tril(1) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  2.0,  4.0,  0.0,
                                                           8.0, 10.0, 12.0,
                                                          14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=2")
    {
        auto t = t33.tril(2) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  2.0,  4.0,  6.0,
                                                           8.0, 10.0, 12.0,
                                                          14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-1")
    {
        auto t = t33.tril(-1) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  0.0,  0.0, 0.0,
                                                           8.0,  0.0, 0.0,
                                                          14.0, 16.0, 0.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-2")
    {
        auto t = t33.tril(-2) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  0.0, 0.0, 0.0,
                                                           0.0, 0.0, 0.0,
                                                          14.0, 0.0, 0.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-3")
    {
        auto t = t33.tril(-3) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  0.0, 0.0, 0.0,
                                                           0.0, 0.0, 0.0,
                                                           0.0, 0.0, 0.0 }, t33.grad().shape()).value());
    }

    // ---

    SUBCASE("Shape{2,2,2} - diagonal=0")
    {
        auto t = t222.tril(0) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  2.0,  0.0,
                                                            6.0,  8.0,
                                                           10.0,  0.0,
                                                           14.0, 16.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=1")
    {
        auto t = t222.tril(1) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  2.0,  4.0,
                                                            6.0,  8.0,
                                                           10.0, 12.0,
                                                           14.0, 16.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-1")
    {
        auto t = t222.tril(-1) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  0.0, 0.0,
                                                            6.0, 0.0,
                                                            0.0, 0.0,
                                                           14.0, 0.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-2")
    {
        auto t = t222.tril(-2) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  0.0, 0.0,
                                                            0.0, 0.0,
                                                            0.0, 0.0,
                                                            0.0, 0.0, }, t222.grad().shape()).value());
    }

}


TEST_CASE("Auto Grad - triu")
{
    auto ts  = aix::tensor(5.0).requireGrad(true);
    auto t1  = aix::tensor({5.0}, aix::Shape{1}).requireGrad(true);
    auto t11  = aix::tensor({5.0}, aix::Shape{1,1}).requireGrad(true);
    auto t33  = aix::tensor({1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0}, aix::Shape{3,3}).requireGrad(true);
    auto t222  = aix::tensor({1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0}, aix::Shape{2,2,2}).requireGrad(true);

    // Default parameters

    SUBCASE("Shape{1,1} - diagonal=default")
    {
        auto t = t11.triu() * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(10.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=default")
    {
        auto t = t33.triu() * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({
                                                           2.0,  4.0,  6.0,
                                                           0.0, 10.0, 12.0,
                                                           0.0,  0.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=1")
    {
        auto t = t11.triu(1) * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(0.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=-1")
    {
        auto t = t11.triu(-1) * t11;
        t.backward(1, t.shape());
        CHECK(t11.grad().shape() == Shape{1,1});
        CheckVectorApproxValues(t11.grad(), aix::Tensor(10.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=0")
    {
        auto t = t33.triu(0) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({  2.0,  4.0,  6.0,
                                                           0.0, 10.0, 12.0,
                                                           0.0,  0.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=1")
    {
        auto t = t33.triu(1) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0,  4.0,  6.0,
                                                          0.0,  0.0, 12.0,
                                                          0.0,  0.0,  0.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=2")
    {
        auto t = t33.triu(2) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 0.0,  0.0,  6.0,
                                                          0.0,  0.0,  0.0,
                                                          0.0,  0.0,  0.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-1")
    {
        auto t = t33.triu(-1) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 2.0,  4.0,  6.0,
                                                          8.0, 10.0, 12.0,
                                                          0.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-2")
    {
        auto t = t33.triu(-2) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 2.0,  4.0,  6.0,
                                                          8.0, 10.0, 12.0,
                                                         14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-3")
    {
        auto t = t33.triu(-3) * t33;
        t.backward(1, t.shape());
        CHECK(t33.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({ 2.0,  4.0,  6.0,
                                                          8.0, 10.0, 12.0,
                                                         14.0, 16.0, 18.0 }, t33.grad().shape()).value());
    }

    // ---

    SUBCASE("Shape{2,2,2} - diagonal=0")
    {
        auto t = t222.triu(0) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  2.0,  4.0,
                                                            0.0,  8.0,
                                                           10.0, 12.0,
                                                            0.0, 16.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=1")
    {
        auto t = t222.triu(1) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  0.0,  4.0,
                                                            0.0,  0.0,
                                                            0.0, 12.0,
                                                            0.0,  0.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-1")
    {
        auto t = t222.triu(-1) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  2.0,  4.0,
                                                            6.0,  8.0,
                                                           10.0, 12.0,
                                                           14.0, 16.0, }, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-2")
    {
        auto t = t222.triu(-2) * t222;
        t.backward(1, t.shape());
        CHECK(t222.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({  2.0,  4.0,
                                                            6.0,  8.0,
                                                           10.0, 12.0,
                                                           14.0, 16.0, }, t222.grad().shape()).value());
    }

}


TEST_CASE("Auto Grad - Cat")
{
    auto ts  = aix::tensor(5.0).requireGrad(true);
    auto t1  = aix::tensor({5.0}, Shape{1}).requireGrad(true);
    auto t11 = aix::tensor({5.0}, Shape{1,1}).requireGrad(true);
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3}).requireGrad(true);
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2}).requireGrad(true);

    SUBCASE("Shape{1} - dim=0")
    {
        auto t = aix::cat({t1}, 0);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1});
        CheckVectorApproxValues(t.grad(), aix::tensor({1.0}, t.grad().shape()).value());
    }

    SUBCASE("Shape{1} - dim=0 - 2x")
    {
        auto t = aix::cat({t1,t1}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t1.grad(), aix::Tensor(2.0, t1.grad().shape()).value());
    }

    SUBCASE("Shape{1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t1,t1}, -1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t1.grad(), aix::Tensor(2.0, t1.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=0")
    {
        auto t = aix::cat({t11}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(1.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=0 - 2x")
    {
        auto t = aix::cat({t11,t11}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(2.0, t11.shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=-2 - 2x")
    {
        auto t = aix::cat({t11,t11}, -2);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(2.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=1 - 2x")
    {
        auto t = aix::cat({t11, t11}, 1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(2.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t11, t11}, -1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(2.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t11, t11}, -1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(2.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0")
    {
        auto t = aix::cat({t33}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t33.grad(), aix::Tensor(1.0, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - 2x")
    {
        auto t = aix::cat({t33, t33}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t33.grad(), aix::Tensor(2.0, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - 2x")
    {
        auto t = aix::cat({t33, t33}, 1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t33.grad(), aix::Tensor(2.0, t33.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 - 2x")
    {
        auto t = aix::cat({t222,t222}, 0);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::Tensor(2.0, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-3 - 3x")
    {
        auto t = aix::cat({t222,t222,t222}, -3);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::Tensor(3.0, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 - 2x")
    {
        auto t = aix::cat({t222,t222}, 1);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::Tensor(2.0, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 - 2x")
    {
        auto t = aix::cat({t222,t222}, 2);
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::Tensor(2.0, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-1 - 2x - complex")
    {
        auto t224 = aix::tensor({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                                 9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0}, {2,2,4}).requireGrad(true);
        auto t = aix::cat({t222,t222}, -1) * t224;
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::tensor({  4.0,  6.0,
                                                           12.0, 14.0,
                                                           20.0, 22.0,
                                                           28.0, 30.0, }, t222.grad().shape()).value());
        CheckVectorApproxValues(t224.grad(), aix::tensor({ 1.0, 2.0, 1.0, 2.0,
                                                           3.0, 4.0, 3.0, 4.0,
                                                           5.0, 6.0, 5.0, 6.0,
                                                           7.0, 8.0, 7.0, 8.0, }, t224.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-2 - 2x - complex")
    {
        auto t242 = aix::tensor({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                                 9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0}, {2,4,2}).requireGrad(true);
        auto t = aix::cat({t222,t222}, -2) * t242;
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 6.0,  8.0,
                                                          10.0, 12.0,
                                                          22.0, 24.0,
                                                          26.0, 28.0 }, t222.grad().shape()).value());
        CheckVectorApproxValues(t242.grad(), aix::tensor({ 1.0, 2.0,
                                                           3.0, 4.0,
                                                           1.0, 2.0,
                                                           3.0, 4.0,
                                                           5.0, 6.0,
                                                           7.0, 8.0,
                                                           5.0, 6.0,
                                                           7.0, 8.0 }, t242.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=-3 - 2x - complex")
    {
        auto t422 = aix::tensor({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                                 9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0}, {4,2,2}).requireGrad(true);
        auto t = aix::cat({t222,t222}, -3) * t422;
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::tensor({ 10.0, 12.0,
                                                           14.0, 16.0,
                                                           18.0, 20.0,
                                                           22.0, 24.0 }, t222.grad().shape()).value());
        CheckVectorApproxValues(t422.grad(), aix::tensor({ 1.0, 2.0,
                                                           3.0, 4.0,
                                                           5.0, 6.0,
                                                           7.0, 8.0,
                                                           1.0, 2.0,
                                                           3.0, 4.0,
                                                           5.0, 6.0,
                                                           7.0, 8.0 }, t422.grad().shape()).value());
    }

}


TEST_CASE("Auto Grad - Reshape")
{
    auto ts  = aix::tensor(5.0).requireGrad(true);
    auto t1  = aix::tensor({5.0}, Shape{1}).requireGrad(true);
    auto t11 = aix::tensor({5.0}, Shape{1,1}).requireGrad(true);
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2}).requireGrad(true);
    auto t412n = aix::tensor({ -1.0, -2.0,
                               -3.0, -4.0,
                               -5.0, -6.0,
                               -7.0, -8.0 }, Shape{4,1,2}).requireGrad(true);

    SUBCASE("Shape{}")
    {
        auto t = ts.reshape({});
        t.backward(1, t.shape());
        CheckVectorApproxValues(ts.grad(), aix::tensor({1.0}, ts.grad().shape()).value());
    }

    SUBCASE("Shape{1}")
    {
        auto t = t1.reshape({1});
        t.backward(1, t.shape());
        CheckVectorApproxValues(t1.grad(), aix::Tensor(1.0, t1.grad().shape()).value());
    }

    SUBCASE("Shape{1,1}")
    {
        auto t = t11.reshape({1,1});
        t.backward(1, t.shape());
        CheckVectorApproxValues(t11.grad(), aix::Tensor(1.0, t11.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} -> Shape{4,1,2}")
    {
        auto t = t222.reshape({4,1,2});
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::Tensor(1.0, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} -> Shape{4,1,2} complex")
    {
        auto t = t222.reshape({4,1,2}) * t412n;
        t.backward(1, t.shape());
        CheckVectorApproxValues(t222.grad(), aix::tensor({-1.0, -2.0,
                                                          -3.0, -4.0,
                                                          -5.0, -6.0,
                                                          -7.0, -8.0,}, t222.grad().shape()).value());
    }

}


TEST_CASE("Auto Grad - Arange")
{
    auto t1  = aix::tensor({5.0}, Shape{1}).requireGrad(true);
    auto t4  = aix::tensor({ 1.0, 2.0, 3.0, 4.0 }).requireGrad(true);

    SUBCASE("Shape{1}")
    {
        auto t = aix::arange(5, 6, 1, { .m_requireGrad=true });
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1});
        CheckVectorApproxValues(t.grad(), aix::tensor({1.0}, t.grad().shape()).value());
    }

    SUBCASE("Shape{4}")
    {
        auto t = aix::arange(1, 5, 1, { .m_requireGrad=true });
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{4});
        CheckVectorApproxValues(t.grad(), aix::Tensor(1.0, t.grad().shape()).value());
    }

    SUBCASE("Shape{1} - complex")
    {
        auto t = t1 * aix::arange(5, 6, 1, { .m_requireGrad=true });
        t.backward(1, t.shape());
        CHECK(t1.grad().shape() == Shape{1});
        CheckVectorApproxValues(t1.grad(), aix::tensor({5.0}, t1.grad().shape()).value());
    }

    SUBCASE("Shape{4} - complex")
    {
        auto t = t4 * aix::arange(1, 5, 1, { .m_requireGrad=true });
        t.backward(1, t.shape());
        CHECK(t4.grad().shape() == Shape{4});
        CheckVectorApproxValues(t4.grad(), aix::tensor({1.0, 2.0, 3.0, 4.0}, t4.grad().shape()).value());
    }
}


TEST_CASE("Auto Grad - indexSelect")
{
    auto ts  = aix::tensor(5.0).requireGrad(true);
    auto t1  = aix::tensor({5.0}, Shape{1}).requireGrad(true);
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3}).requireGrad(true);
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2}).requireGrad(true);

    auto is  = aix::tensor(0.0, aix::dtype(aix::DataType::kInt32));
    auto i1  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));

    SUBCASE("Shape{} - dim=0 - index{}")
    {
        auto t = ts.indexSelect(0, is);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{});
        CheckVectorApproxValues(ts.grad(), aix::Tensor(1.0, ts.grad().shape()).value());
    }

    SUBCASE("Shape{1} - dim=0 - index{}")
    {
        auto t = t1.indexSelect(0, is);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1});
        CheckVectorApproxValues(t1.grad(), aix::Tensor(1.0, t1.grad().shape()).value());
    }

    SUBCASE("Shape{} - dim=0 - index{1}")
    {
        auto t = ts.indexSelect(0, i1);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{});
        CheckVectorApproxValues(ts.grad(), aix::Tensor(1.0, ts.grad().shape()).value());
    }

    SUBCASE("Shape{1} - dim=0 - index{1}")
    {
        auto t = t1.indexSelect(0, i1);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1});
        CheckVectorApproxValues(t1.grad(), aix::Tensor(1.0, t1.grad().shape()).value());
    }

    // dim 0

    SUBCASE("Shape{3,3} - dim=0 - index = {0}")
    {
        auto indices  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 0.0, 0.0,
                                                         1.0, 1.0, 1.0,
                                                         0.0, 0.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2}")
    {
        auto indices  = aix::tensor({2.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,
                                                         0.0, 0.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1,2}")
    {
        auto indices  = aix::tensor({1.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 0.0, 0.0,
                                                         1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {0,2}")
    {
        auto indices  = aix::tensor({0.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         0.0, 0.0, 0.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,0}")
    {
        auto indices  = aix::tensor({2.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         0.0, 0.0, 0.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 0.0, 0.0,
                                                         2.0, 2.0, 2.0,
                                                         0.0, 0.0, 0.0,}, t33.grad().shape()).value());
    }


    SUBCASE("Shape{3,3} - dim=0 - index = {0,1,2}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 2.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,1,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    // dim 1

    SUBCASE("Shape{3,3} - dim=1 - index = {0}")
    {
        auto indices  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,1});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 0.0, 0.0,
                                                         1.0, 0.0, 0.0,
                                                         1.0, 0.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,1});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 1.0, 0.0,
                                                         0.0, 1.0, 0.0,
                                                         0.0, 1.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {2}")
    {
        auto indices  = aix::tensor({2.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,1});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 0.0, 1.0,
                                                         0.0, 0.0, 1.0,
                                                         0.0, 0.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,2});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 0.0,
                                                         1.0, 1.0, 0.0,
                                                         1.0, 1.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1,2}")
    {
        auto indices  = aix::tensor({1.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,2});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 1.0, 1.0,
                                                         0.0, 1.0, 1.0,
                                                         0.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,2}")
    {
        auto indices  = aix::tensor({0.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,2});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 0.0, 1.0,
                                                         1.0, 0.0, 1.0,
                                                         1.0, 0.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,0}")
    {
        auto indices  = aix::tensor({2.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,2});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 0.0, 1.0,
                                                         1.0, 0.0, 1.0,
                                                         1.0, 0.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,2});
        CheckVectorApproxValues(t33.grad(), aix::tensor({0.0, 2.0, 0.0,
                                                         0.0, 2.0, 0.0,
                                                         0.0, 2.0, 0.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,1,2}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 2.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {2,1,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,
                                                         1.0, 1.0, 1.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,1,0,2,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0, 2.0, 0.0}, Shape{5}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{5,3});
        CheckVectorApproxValues(t33.grad(), aix::tensor({2.0, 2.0, 2.0,
                                                         1.0, 1.0, 1.0,
                                                         2.0, 2.0, 2.0,}, t33.grad().shape()).value());
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {2,1,0,2,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0, 2.0, 0.0}, Shape{5}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{3,5});
        CheckVectorApproxValues(t33.grad(), aix::tensor({2.0, 1.0, 2.0,
                                                         2.0, 1.0, 2.0,
                                                         2.0, 1.0, 2.0,}, t33.grad().shape()).value());
    }

    //

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{1,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 0.0,
                                                          0.0, 0.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,1,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 0.0,
                                                          1.0, 1.0,
                                                          0.0, 0.0,
                                                          1.0, 1.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,1});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 1.0,
                                                          0.0, 1.0,
                                                          0.0, 1.0,
                                                          0.0, 1.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,
                                                          1.0, 1.0,}, t222.grad().shape()).value());
    }

    //

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 0.0,
                                                          0.0, 0.0,
                                                          2.0, 2.0,
                                                          2.0, 2.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 0.0,
                                                          2.0, 2.0,
                                                          0.0, 0.0,
                                                          2.0, 2.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({0.0, 2.0,
                                                          0.0, 2.0,
                                                          0.0, 2.0,
                                                          0.0, 2.0,}, t222.grad().shape()).value());
    }

    //

    SUBCASE("Shape{2,2,2} - dim=0 - index = {0,1,1,0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 1.0, 0.0, 1.0}, Shape{5}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{5,2,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({2.0, 2.0,
                                                          2.0, 2.0,
                                                          3.0, 3.0,
                                                          3.0, 3.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {0,1,1,0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 1.0, 0.0, 1.0}, Shape{5}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,5,2});
        CheckVectorApproxValues(t222.grad(), aix::tensor({2.0, 2.0,
                                                          3.0, 3.0,
                                                          2.0, 2.0,
                                                          3.0, 3.0,}, t222.grad().shape()).value());
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {0,1,1,0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 1.0, 0.0, 1.0}, Shape{5}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        t.backward(1, t.shape());
        CHECK(t.grad().shape() == Shape{2,2,5});
        CheckVectorApproxValues(t222.grad(), aix::tensor({2.0, 3.0,
                                                          2.0, 3.0,
                                                          2.0, 3.0,
                                                          2.0, 3.0,}, t222.grad().shape()).value());
    }

}


TEST_CASE("Auto Grad - permute()")
{
    auto t12 = tensor({1.0,2.0}, Shape{1,2}, requireGrad(true));
    auto t32 = tensor({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, requireGrad(true));
    auto t324 = arange(1.0, 25.0, 1.0).reshape({3,2,4}).requireGrad(true);
    auto a2  = arange(1.0, 3.0);
    auto a6  = arange(1.0, 7.0, 1.0);
    auto a24 = arange(1.0, 25.0, 1.0);

    SUBCASE("s{} p{}")
    {
        auto a = tensor({5.0}, Shape{}, requireGrad(true));
        auto t = a.permute({});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1} p{0}")
    {
        auto a = tensor({5.0}, Shape{1}, requireGrad(true));
        auto t = a.permute({0});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1} p{-1}")
    {
        auto a = tensor({5.0}, Shape{1}, requireGrad(true));
        auto t = a.permute({-1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{0,1}")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({0,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{-2,-1}")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({-2,-1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{-1,-2}")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({-1,-2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{0,1}")
    {
        auto a = t12;
        auto t = a.permute({0,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{-2,-1}")
    {
        auto a = t12;
        auto t = a.permute({-2,-1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{-1,-2}")
    {
        auto a = t12;
        auto t = a.permute({-1,-2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{0,1}")
    {
        auto a = t12.reshape({2,1}).requireGrad(true);
        auto t = a.permute({0,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{-2,-1}")
    {
        auto a = t12.reshape({2,1}).requireGrad(true);
        auto t = a.permute({-2,-1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{-1,-2}")
    {
        auto a = t12.reshape({2,1}).requireGrad(true);
        auto t = a.permute({-1,-2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{0,1}")
    {
        auto a = t32;
        auto t = a.permute({0,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{1,0}")
    {
        auto a = t32;
        auto t = a.permute({1,0});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{-2,-1}")
    {
        auto a = t32;
        auto t = a.permute({-2,-1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{-1,-2}")
    {
        auto a = t32;
        auto t = a.permute({-1,-2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{0,1,2}")
    {
        auto a = t324;
        auto t = a.permute({0,1,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2,4});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{0,2,1}")
    {
        auto a = t324;
        auto t = a.permute({0,2,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,4,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{1,0,2}")
    {
        auto a = t324;
        auto t = a.permute({1,0,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3,4});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{1,2,0}")
    {
        auto a = t324;
        auto t = a.permute({1,2,0});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,4,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{2,0,1}")
    {
        auto a = t324;
        auto t = a.permute({2,0,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{2,1,0}")
    {
        auto a = t324;
        auto t = a.permute({2,1,0});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{-1,-2,-3}")
    {
        auto a = t324;
        auto t = a.permute({-1,-2,-3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0,
                                                  1.0,1.0,1.0,1.0,1.0,1.0}, a.shape()).value());
    }

    // ------------------------------
    // Complex
    // ------------------------------

    SUBCASE("s{} p{} complex")
    {
        auto a = tensor({5.0}, Shape{}, requireGrad(true));
        auto t = a.permute({}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1} p{0} complex")
    {
        auto a = tensor({5.0}, Shape{1}, requireGrad(true));
        auto t = a.permute({0}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1} p{-1} complex")
    {
        auto a = tensor({5.0}, Shape{1}, requireGrad(true));
        auto t = a.permute({-1}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{0,1} complex")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({0,1}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{-2,-1} complex")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({-2,-1}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1,1} p{-1,-2} complex")
    {
        auto a = tensor({5.0}, Shape{1,1}, requireGrad(true));
        auto t = a.permute({-1,-2}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(a.grad(), tensor({10.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{0,1} complex")
    {
        auto a = a2.reshape({1,2}).requireGrad(true);
        auto t = a.permute({0,1}) * a;
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({2.0,4.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{-2,-1} complex")
    {
        auto a = a2.reshape({1,2}).requireGrad(true);
        auto t = a.permute({-2,-1}) * a2.reshape({1,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0}, a.shape()).value());
    }

    SUBCASE("s{1,2} p{-1,-2} complex")
    {
        auto a = a2.reshape({1,2}).requireGrad(true);
        auto t = a.permute({-1,-2}) * a2.reshape({2,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{0,1} complex")
    {
        auto a = a2.reshape({2,1}).requireGrad(true);
        auto t = a.permute({0,1}) * a2.reshape({2,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{-2,-1} complex")
    {
        auto a = a2.reshape({2,1}).requireGrad(true);
        auto t = a.permute({-2,-1}) * a2.reshape({2,1});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0}, a.shape()).value());
    }

    SUBCASE("s{2,1} p{-1,-2} complex")
    {
        auto a = a2.reshape({2,1}).requireGrad(true);
        auto t = a.permute({-1,-2}) * a2.reshape({1,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{0,1} complex")
    {
        auto a = a6.reshape({3,2}).requireGrad(true);
        auto t = a.permute({0,1}) * a6.reshape({3,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0,3.0,4.0,5.0,6.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{1,0} complex")
    {
        auto a = a6.reshape({3,2}).requireGrad(true);
        auto t = a.permute({1,0}) * a6.reshape({2,3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,4.0,2.0,5.0,3.0,6.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{-2,-1} complex")
    {
        auto a = a6.reshape({3,2}).requireGrad(true);
        auto t = a.permute({-2,-1}) * a6.reshape({3,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,2.0,3.0,4.0,5.0,6.0}, a.shape()).value());
    }

    SUBCASE("s{3,2} p{-1,-2} complex")
    {
        auto a = a6.reshape({3,2}).requireGrad(true);
        auto t = a.permute({-1,-2}) * a6.reshape({2,3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,4.0,2.0,5.0,3.0,6.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{0,1,2} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({0,1,2}) * a24.reshape({3,2,4});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,2,4});
        CheckVectorApproxValues(a.grad(), tensor({ 1.0,  2.0,  3.0,  4.0,
                                                   5.0,  6.0,  7.0,  8.0,
                                                   9.0, 10.0, 11.0, 12.0,
                                                  13.0, 14.0, 15.0, 16.0,
                                                  17.0, 18.0, 19.0, 20.0,
                                                  21.0, 22.0, 23.0, 24.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{0,2,1} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({0,2,1}) * a24.reshape({3,4,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{3,4,2});
        CheckVectorApproxValues(a.grad(), tensor({ 1.0,  3.0,  5.0,  7.0,
                                                   2.0,  4.0,  6.0,  8.0,
                                                   9.0, 11.0, 13.0, 15.0,
                                                  10.0, 12.0, 14.0, 16.0,
                                                  17.0, 19.0, 21.0, 23.0,
                                                  18.0, 20.0, 22.0, 24.0 }, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{1,0,2} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({1,0,2}) * a24.reshape({2,3,4});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,3,4});
        CheckVectorApproxValues(a.grad(), tensor({ 1.0,  2.0,  3.0,  4.0,
                                                  13.0, 14.0, 15.0, 16.0,
                                                   5.0,  6.0,  7.0,  8.0,
                                                  17.0, 18.0, 19.0, 20.0,
                                                   9.0, 10.0, 11.0, 12.0,
                                                  21.0, 22.0, 23.0, 24.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{1,2,0} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({1,2,0}) * a24.reshape({2,4,3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{2,4,3});
        CheckVectorApproxValues(a.grad(), tensor({ 1.0,  4.0,  7.0, 10.0,
                                                  13.0, 16.0, 19.0, 22.0,
                                                   2.0,  5.0,  8.0, 11.0,
                                                  14.0, 17.0, 20.0, 23.0,
                                                   3.0,  6.0,  9.0, 12.0,
                                                  15.0, 18.0, 21.0, 24.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{2,0,1} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({2,0,1}) * a24.reshape({4,3,2});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,3,2});
        CheckVectorApproxValues(a.grad(), tensor({1.0,  7.0, 13.0, 19.0,
                                                  2.0,  8.0, 14.0, 20.0,
                                                  3.0,  9.0, 15.0, 21.0,
                                                  4.0, 10.0, 16.0, 22.0,
                                                  5.0, 11.0, 17.0, 23.0,
                                                  6.0, 12.0, 18.0, 24.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{2,1,0} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({2,1,0}) * a24.reshape({4,2,3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,  7.0, 13.0, 19.0,
                                                  4.0, 10.0, 16.0, 22.0,
                                                  2.0,  8.0, 14.0, 20.0,
                                                  5.0, 11.0, 17.0, 23.0,
                                                  3.0,  9.0, 15.0, 21.0,
                                                  6.0, 12.0, 18.0, 24.0}, a.shape()).value());
    }

    SUBCASE("s{3,2,4} p{-1,-2,-3} complex")
    {
        auto a = a24.reshape({3,2,4}).requireGrad(true);
        auto t = a.permute({-1,-2,-3}) * a24.reshape({4,2,3});
        t.backward(1, t.shape());
        CHECK(t.shape() == Shape{4,2,3});
        CheckVectorApproxValues(a.grad(), tensor({1.0,  7.0, 13.0, 19.0,
                                                  4.0, 10.0, 16.0, 22.0,
                                                  2.0,  8.0, 14.0, 20.0,
                                                  5.0, 11.0, 17.0, 23.0,
                                                  3.0,  9.0, 15.0, 21.0,
                                                  6.0, 12.0, 18.0, 24.0}, a.shape()).value());
    }
}
