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
#include <array>
#include <cstring>
#include <limits>
#include <sstream>


using namespace aix;

DeviceCPU  testDevice;     // Default CPU device.


TEST_CASE("Simple TensorValue 1 dim - Add")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    auto y = TensorValue({4.0, 5.0, 6.0}, {1, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{1, 3});
    CheckVectorApproxValues(z, TensorValue({5.0, 7.0, 9.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Add")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},    {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x + y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({8.0, 10.0, 12.0, 14.0, 16.0, 18.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Sub")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},    {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x - y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-6.0, -6.0, -6.0, -6.0, -6.0, -6.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Mul")
{
    auto x = TensorValue({1.0, 2.0, 3.0,  4.0,  5.0,  6.0}, {2, 3}, &testDevice);
    auto y = TensorValue({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2, 3}, &testDevice);

    auto z = x * y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({7.0, 16.0, 27.0, 40.0, 55.0, 72.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Div")
{
    auto x = TensorValue({-5.0, 0.0, 5.0, 10.0, 15.0, 20.0}, {2, 3}, &testDevice);
    auto y = TensorValue({5.0, 5.0, 5.0, 5.0, -5.0, -20.0},  {2, 3}, &testDevice);

    auto z = x / y;

    CHECK(z.shape() == Shape{2, 3});
    CheckVectorApproxValues(z, TensorValue({-1.0, 0.0, 1.0, 2.0, -3.0, -1.0}, z.shape(), &testDevice));
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice);
    auto z = x;
    z.data<float>()[0] = 1;

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("Simple TensorValue 2 dim - Copy Const")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice);
    auto copyTensor = [](TensorValue value) { return value; };
    auto z = copyTensor(x);

    CHECK(z.shape() == x.shape());
    CheckVectorApproxValues(z, x);
}


TEST_CASE("TensorValue - item()")
{
    SUBCASE("Must fail due to a non-scalar tensor")
    {
        TensorValue input = TensorValue({1.0, 2.0}, {2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input.item<float>(), std::invalid_argument);

        TensorValue input2 = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.item<float>(), std::invalid_argument);
    }

    SUBCASE("Should return a scalar value")
    {
        TensorValue input = TensorValue(1.0, Shape{}, &testDevice);
        CHECK(input.item<float>() == 1.0);
    }
}


TEST_CASE("TensorValue - matmul()")
{
    SUBCASE("Must fail due to an inner dimension mismatch")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
        TensorValue B = TensorValue({1.0, 2.0, 3.0, 4.0}, {1, 4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("Must fail due to dimension size mismatch")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0}, {4}, &testDevice);
        TensorValue B = TensorValue({1.0, 2.0, 3.0, 4.0}, {1, 4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("A[1x1] B[1x1] multiplication")
    {
        TensorValue A = TensorValue(2.0, Shape{1, 1}, &testDevice);
        TensorValue B = TensorValue(3.0, Shape{1, 1}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(6.0, Shape{1, 1}, &testDevice));
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        TensorValue A = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3}, &testDevice);
        TensorValue B = TensorValue({12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}, Shape{3, 4}, &testDevice);
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.size() == 8);
        CheckVectorApproxValues(result, TensorValue({40.0, 34.0, 28.0, 22.0, 112.0, 97.0, 82.0, 67.0}, Shape{2, 4}, &testDevice));
    }
}


TEST_CASE("TensorValue - transpose()")
{
    auto t322 = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {3,2,2}, &testDevice);
    auto t32 = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);

    SUBCASE("Must fail if dimension size is higher")
    {
        TensorValue input1 = TensorValue({1.0,2.0,3.0,4.0}, {2,1,2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input1.transpose(1, 3), std::invalid_argument);
        TensorValue input2 = TensorValue({1.0,2.0,3.0,4.0}, {4}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(input2.transpose(0, 1), std::invalid_argument);
    }

    SUBCASE("{} transpose")
    {
        TensorValue A = TensorValue(1.0, &testDevice);
        DOCTEST_CHECK_THROWS_AS(A.transpose(0, 0), std::invalid_argument);
    }

    SUBCASE("1x1 transpose")
    {
        TensorValue A = TensorValue(1.0, {1,1}, &testDevice);
        auto result = A.transpose(0, 1);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, TensorValue(1.0, {1,1}, &testDevice));
    }

    SUBCASE("3x2 transpose")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(0, 1);
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(0,0)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(0, 0);
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(1,1)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(1, 1);
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("3x2 transpose - dims(1,0)")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
        auto result = A.transpose(1, 0);
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("3x2x2 transpose(0,1) -> 2x3x2")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {3,2,2}, &testDevice);
        auto result1 = A.transpose(0, 1);
        CHECK(result1.shape() == Shape{2,3,2});
        CheckVectorApproxValues(result1, TensorValue({1.0,2.0,5.0,6.0,9.0,10.0,
                                                      3.0,4.0,7.0,8.0,11.0,12.0}, {2,3,2}, &testDevice));

        auto result2 = A.transpose(1, 0);
        CHECK(result2.shape() == Shape{2,3,2});
        CheckVectorApproxValues(result2, TensorValue({1.0,2.0,5.0,6.0,9.0,10.0,
                                                      3.0,4.0,7.0,8.0,11.0,12.0}, {2,3,2}, &testDevice));
    }

    SUBCASE("3x2x2 transpose(0,2) -> 2x2x3")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {3,2,2}, &testDevice);
        auto result1 = A.transpose(0, 2);
        CHECK(result1.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result1, TensorValue({1.0,5.0,9.0,3.0,7.0,11.0,
                                                      2.0,6.0,10.0,4.0,8.0,12.0}, {2,2,3}, &testDevice));

        auto result2 = A.transpose(2, 0);
        CHECK(result2.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result2, TensorValue({1.0,5.0,9.0,3.0,7.0,11.0,
                                                      2.0,6.0,10.0,4.0,8.0,12.0}, {2,2,3}, &testDevice));
    }

    SUBCASE("2x3x2 transpose(1,2) -> 2x2x3")
    {
        TensorValue A = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}, {2,3,2}, &testDevice);
        auto result1 = A.transpose(1, 2);
        CHECK(result1.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result1, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));

        auto result2 = A.transpose(2, 1);
        CHECK(result2.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result2, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));
    }

    SUBCASE("2x3x2 transpose(-2,-1) -> 2x2x3")
    {
        auto A = t322.reshape({2,3,2});
        auto result1 = A.transpose(-2, -1);
        CHECK(result1.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result1, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));

        auto result2 = A.transpose(-1, -2);
        CHECK(result2.shape() == Shape{2,2,3});
        CheckVectorApproxValues(result2, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0,
                                                      7.0,9.0,11.0,8.0,10.0,12.0}, {2,2,3}, &testDevice));
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ t32.transpose(0, 3); }, std::invalid_argument);
        CHECK_THROWS_AS({ t32.transpose(4, 2); }, std::invalid_argument);
        CHECK_THROWS_AS({ t32.transpose(0, -4); }, std::invalid_argument);
        CHECK_THROWS_AS({ t32.transpose(-5, 2); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue::tanh 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.tanh(), TensorValue({0.099668, 0.197375, 0.291313, 0.379949}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::log 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, 0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.log(), TensorValue({-2.30259, -1.60944, -1.20397, -0.916291}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::exp 2x2")
{
    auto x = TensorValue({0.1, 0.2, 0.3, -0.4}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.exp(), TensorValue({1.10517, 1.2214, 1.34986, 0.67032}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue::pow 2x2")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
    auto exp = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);
    CheckVectorApproxValues(x.pow(exp), TensorValue({1.0, 4.0, 27.0, 256.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with Scalar")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x += scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({6.0, 7.0, 8.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with Scalar")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x -= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({1.0, 2.0, 3.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with Scalar")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    float scalar = 5;
    x *= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5.0, 10.0, 15.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with Scalar")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    float scalar = 2;
    x /= scalar;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3.0, 3.5, 4.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Add with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({1.0, 2.0, -1.0}, {1, 3}, &testDevice);
    x += y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7.0, 9.0, 7.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Sub with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({1.0, 2.0, -1.0}, {1, 3}, &testDevice);
    x -= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({5.0, 5.0, 9.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Mul with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({2.0, 3.0, -2.0}, {1, 3}, &testDevice);
    x *= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({12.0, 21.0, -16.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - In-place Div with TensorValue")
{
    auto x = TensorValue({6.0, 7.0, 8.0}, {1, 3}, &testDevice);
    auto y = TensorValue({2.0, 2.0, -2.0}, {1, 3}, &testDevice);
    x /= y;

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({3.0, 3.5, -4.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Unary Minus")
{
    auto x = TensorValue({1.0, -2.0, 3.0}, {1, 3}, &testDevice);
    auto y = -x;

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({-1.0, 2.0, -3.0}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Fill")
{
    auto x = TensorValue({1.0, 1.0, 1.0}, {1, 3}, &testDevice);
    float fillValue = 7;
    x.fill(fillValue);

    CHECK(x.shape() == Shape{1, 3});
    CheckVectorApproxValues(x, TensorValue({7.0, 7.0, 7.0}, x.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sum")
{
    auto ts  = TensorValue(5.0, {}, &testDevice);
    auto t22 = TensorValue({-1.0, 2.2, 0.0, 4.8}, {2, 2}, &testDevice);

    SUBCASE("Shape{2,2}")
    {
        CheckVectorApproxValues(t22.sum(), TensorValue(6.0, &testDevice));
    }

    SUBCASE("Shape{} - dim=0")
    {
        CheckVectorApproxValues(ts.sum(0), TensorValue(5.0, &testDevice));
    }

    SUBCASE("Shape{} - dim=1")
    {
        CheckVectorApproxValues(ts.sum(1), TensorValue(5.0, &testDevice));
    }
}


TEST_CASE("TensorValue - Sum with dim")
{
    auto t1  = aix::TensorValue({1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0,
                                 10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0,
                                 16.0, 17.0, 18.0,
                                 19.0, 20.0, 21.0,
                                 22.0, 23.0, 24.0}, aix::Shape{3, 4, 2}, &testDevice);

    SUBCASE("Shape{3,4,2} - dim=0 keepDim=false")
    {
        auto t = t1.sum(0, false);
        auto expectedShape = Shape{4, 2};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({27.0, 30.0,
                                                33.0, 36.0,
                                                39.0, 42.0,
                                                45.0, 48.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=0 keepDim=true")
    {
        auto t = t1.sum(0, true);
        auto expectedShape = Shape{1, 4, 2};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({27.0, 30.0,
                                                33.0, 36.0,
                                                39.0, 42.0,
                                                45.0, 48.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=1 keepDim=false")
    {
        auto t = t1.sum(1, false);
        auto expectedShape = Shape{3, 2};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({16.0, 20.0,
                                                48.0, 52.0,
                                                80.0, 84.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=1 keepDim=true")
    {
        auto t = t1.sum(1, true);
        auto expectedShape = Shape{3, 1, 2};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({16.0, 20.0,
                                                48.0, 52.0,
                                                80.0, 84.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=2 keepDim=false")
    {
        auto t = t1.sum(2, false);
        auto expectedShape = Shape{3, 4};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({ 3.0,  7.0, 11.0, 15.0,
                                                19.0, 23.0, 27.0, 31.0,
                                                35.0, 39.0, 43.0, 47.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=2 keepDim=true")
    {
        auto t = t1.sum(2, true);
        auto expectedShape = Shape{3, 4, 1};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({ 3.0,  7.0, 11.0, 15.0,
                                                19.0, 23.0, 27.0, 31.0,
                                                35.0, 39.0, 43.0, 47.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=-1 keepDim=false")
    {
        auto t = t1.sum(-1, false);
        auto expectedShape = Shape{3, 4};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({ 3.0,  7.0, 11.0, 15.0,
                                                19.0, 23.0, 27.0, 31.0,
                                                35.0, 39.0, 43.0, 47.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dim=-1 keepDim=true")
    {
        auto t = t1.sum(-1, true);
        auto expectedShape = Shape{3, 4, 1};
        CHECK(t.shape() == expectedShape);
        CheckVectorApproxValues(t, TensorValue({ 3.0,  7.0, 11.0, 15.0,
                                                19.0, 23.0, 27.0, 31.0,
                                                35.0, 39.0, 43.0, 47.0}, expectedShape, &testDevice));
    }

    SUBCASE("Shape{3,4,2} - dimension out of range")
    {
        CHECK_THROWS_AS({ t1.sum(3, false);  }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.sum(3, true);   }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.sum(-4, false); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.sum(-4, true);  }, std::invalid_argument);
    }

}


TEST_CASE("TensorValue - keepDim reductions avoid redundant allocations")
{
    class CountingDevice : public DeviceCPU
    {
    public:
        void* allocate(size_t size, DataType dtype) override
        {
            ++m_allocations;
            return DeviceCPU::allocate(size, dtype);
        }

        size_t m_allocations{0};
    } device;

    auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &device);

    device.m_allocations = 0;
    auto sum = input.sum(-1, true);
    CHECK(device.m_allocations == 1);
    CHECK(sum.shape() == Shape{2, 1});
    CHECK(sum.storage() != input.storage());
    CheckVectorApproxValues(sum, TensorValue({6.0f, 15.0f}, {2, 1}, &testDevice));

    device.m_allocations = 0;
    auto max = input.max(-1, true);
    CHECK(device.m_allocations == 1);
    CHECK(max.shape() == Shape{2, 1});
    CHECK(max.storage() != input.storage());
    CheckVectorApproxValues(max, TensorValue({3.0f, 6.0f}, {2, 1}, &testDevice));
}


TEST_CASE("TensorValue - multi-axis sum shapes values and layouts")
{
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(type));
        for (size_t dtypeIndex = 0; dtypeIndex < DataTypeCount; ++dtypeIndex)
        {
            const auto dtype = static_cast<DataType>(dtypeIndex);
            if (type == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(dtypeIndex);
            const auto sumType = dtype > DataType::kInt64 ? DataType::kInt64 : dtype;
            auto input = TensorValue({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2}, device.get(), dtype);
            for (bool keepDim : {false, true})
            {
                auto sum = input.sum({2, 0}, keepDim);
                const Shape expectedShape = keepDim ? Shape{1, 3, 1} : Shape{3};
                CHECK(sum.shape() == expectedShape);
                CHECK(sum.dataType() == sumType);
                CHECK(sum.device() == device.get());
                CHECK(sum.storage() != input.storage());
                CHECK(sum.storageOffset() == 0);
                CHECK(sum.isContiguous());
                CheckVectorApproxValues(sum, TensorValue({18, 26, 34}, expectedShape, device.get(), sumType));
                CheckVectorApproxValues(sum, input.sum({0, 2}, keepDim));

                auto all = input.sum({2, 0, 1}, keepDim);
                CHECK(all.shape() == (keepDim ? Shape{1, 1, 1} : Shape{}));
                CheckVectorApproxValues(all, TensorValue(78, all.shape(), device.get(), sumType));
                auto implicitAll = input.sum({}, keepDim);
                CHECK(implicitAll.shape() == all.shape());
                CHECK(implicitAll.storage() != input.storage());
                CheckVectorApproxValues(implicitAll, all);
                CheckVectorApproxValues(input.sum({1}, keepDim), input.sum(-2, keepDim));
                CheckVectorApproxValues(input.sum({-1, -3}, keepDim), sum);
                CheckVectorApproxValues(input.sum({0, -1}, keepDim), sum);
            }

            auto transposed = input.transpose(0, 1);
            CheckVectorApproxValues(transposed.sum({2, 0}), TensorValue({21, 57}, {2}, device.get(), sumType));
            auto offset = input.slice(2, 1, 2);
            CheckVectorApproxValues(offset.sum({2, 0}), TensorValue({10, 14, 18}, {3}, device.get(), sumType));
            auto broadcast = TensorValue({1, 2, 3}, {1, 3, 1}, device.get(), dtype).broadcastTo({2, 3, 2});
            CheckVectorApproxValues(broadcast.sum({2, 0}), TensorValue({4, 8, 12}, {3}, device.get(), sumType));
            auto singleton = offset.sum({2}, true);
            CHECK(singleton.storage() != input.storage());
            CheckVectorApproxValues(singleton, offset.to(sumType));
            singleton.fill(0);
            CheckVectorApproxValues(offset, TensorValue({2, 4, 6, 8, 10, 12}, {2, 3, 1}, device.get(), dtype));

            CHECK_THROWS_AS(input.sum({0, 0}), std::invalid_argument);
            CHECK_THROWS_AS(input.sum({2, 0, 2}, true), std::invalid_argument);
            CHECK_THROWS_AS(input.sum({3}), std::invalid_argument);
            CHECK_THROWS_AS(input.sum({-4}), std::invalid_argument);
            CHECK_THROWS_AS(input.sum({2, -1}), std::invalid_argument);
            CHECK_THROWS_AS(input.sum(SIndex{std::numeric_limits<ssize_t>::min()}), std::invalid_argument);
            CHECK_THROWS_AS(input.sum(SIndex{std::numeric_limits<ssize_t>::max()}), std::invalid_argument);
            auto scalar = TensorValue(7, {}, device.get(), dtype);
            auto scalarCopy = scalar.sum({});
            CHECK(scalarCopy.shape() == Shape{});
            CHECK(scalarCopy.storage() != scalar.storage());
            CheckVectorApproxValues(scalarCopy, scalar.to(sumType));
            CheckVectorApproxValues(scalar.sum({0}), scalar.to(sumType));
            CheckVectorApproxValues(scalar.sum({-1}, true), scalar.to(sumType));
            CHECK_THROWS_AS(scalar.sum({-1, 0}), std::invalid_argument);
            CHECK_THROWS_AS(scalar.sum({1}), std::invalid_argument);
        }
    }
}


TEST_CASE("TensorValue - integer sum promotes before accumulation")
{
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        for (auto dtype : {DataType::kUInt8, DataType::kInt8, DataType::kInt16, DataType::kInt32, DataType::kInt64})
        {
            CAPTURE(static_cast<size_t>(type));
            CAPTURE(static_cast<size_t>(dtype));
            const int64_t value = dtype == DataType::kInt32 ? 2147483647 :
                                  dtype == DataType::kInt64 ? 9007199254740993LL :
                                  dtype == DataType::kInt16 ? 32767 : 100;
            const int64_t values[] = {value, value, value, value};
            auto input = TensorValue(values, 4, DataType::kInt64, {2, 2}, device.get(), dtype);
            auto check = [&](TensorValue actual, int64_t expected)
            {
                REQUIRE(actual.dataType() == DataType::kInt64);
                CHECK(actual.device() == device.get());
                CHECK(actual.storage() != input.storage());
                device->synchronize();
                for (size_t i = 0; i < actual.size(); ++i) CHECK(actual.data<int64_t>()[i] == expected);
            };
            check(input.sum(), value * 4);
            check(input.sum({}), value * 4);
            check(input.sum({-1, -2}, true), value * 4);
            check(input.sum(0), value * 2);
            check(input.sum(-1, true), value * 2);
            check(input.transpose(0, 1).sum({0}), value * 2);
            check(input.slice(1, 1, 2).sum(), value * 2);
            check(input.slice(0, 0, 1).broadcastTo({3, 2}).sum({0}), value * 3);
            if (dtype != DataType::kUInt8)
            {
                auto negative = TensorValue({-100, -100, -100, -100}, {2, 2}, device.get(), dtype);
                check(negative.sum(), -400);
                check(negative.sum({-1}), -200);
            }
            auto scalar = TensorValue(values, 1, DataType::kInt64, {}, device.get(), dtype);
            check(scalar.sum(), value);
            check(scalar.sum(0), value);
            check(scalar.sum({-1}, true), value);
            auto empty = TensorValue({2, 0, 3}, device.get(), dtype);
            check(empty.sum(), 0);
            check(empty.sum({1}, true), 0);
            auto keptEmpty = empty.sum({0, 2});
            CHECK(keptEmpty.shape() == Shape{0});
            CHECK(keptEmpty.dataType() == DataType::kInt64);
            CHECK(input.dataType() == dtype);
        }
    }
}

TEST_CASE("TensorValue - multi-axis sum empty inputs")
{
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        for (size_t dtypeIndex = 0; dtypeIndex < DataTypeCount; ++dtypeIndex)
        {
            const auto dtype = static_cast<DataType>(dtypeIndex);
            if (type == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            auto base = TensorValue({2, 0, 3}, device.get(), dtype);
            for (size_t layout = 0; layout < 2; ++layout)
            {
                auto input = layout == 0 ? base.contiguous() : base.transpose(0, 2);
                for (const auto& axes : {SIndex{}, SIndex{0}, SIndex{1}, SIndex{2, 0}, SIndex{2, 1, 0}, SIndex{-1, -3}})
                {
                    for (bool keepDim : {false, true})
                    {
                        auto expectedShape = input.shape();
                        auto normalized = axes;
                        if (normalized.empty()) normalized = {0, 1, 2};
                        for (auto& axis : normalized)
                        {
                            if (axis < 0) axis += static_cast<ssize_t>(expectedShape.size());
                            expectedShape[axis] = 1;
                        }
                        if (!keepDim)
                        {
                            for (size_t i = expectedShape.size(); i-- > 0;)
                            {
                                if (std::find(normalized.begin(), normalized.end(), i) != normalized.end())
                                    expectedShape.erase(expectedShape.begin() + i);
                            }
                        }
                        auto sum = input.sum(axes, keepDim);
                        CHECK(sum.shape() == expectedShape);
                        CHECK(sum.dataType() == (dtype > DataType::kInt64 ? DataType::kInt64 : dtype));
                        CHECK(sum.device() == device.get());
                        CHECK(sum.storage() != input.storage());
                        if (sum.size() != 0)
                            CheckVectorApproxValues(sum, TensorValue(0, expectedShape, device.get(), sum.dataType()));
                    }
                }
                CHECK_THROWS_AS(input.sum({1, 1}), std::invalid_argument);
                CHECK_THROWS_AS(input.sum({3}), std::invalid_argument);
            }
        }
    }
}


TEST_CASE("TensorValue - keepDim singleton reductions preserve independent storage")
{
    auto check = [](Device* device)
    {
        auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device);
        auto column = input.slice(1, 1, 2);
        auto sum = column.sum(-1, true);
        auto max = column.max(-1, true);
        CHECK(sum.shape() == Shape{2, 1});
        CHECK(max.shape() == Shape{2, 1});
        CHECK(sum.storage() != input.storage());
        CHECK(max.storage() != input.storage());
        CHECK(sum.storageOffset() == 0);
        CHECK(sum.isContiguous());
        CheckVectorApproxValues(sum, TensorValue({2.0f, 5.0f}, {2, 1}, &testDevice));
        CheckVectorApproxValues(max, TensorValue({2.0f, 5.0f}, {2, 1}, &testDevice));

        sum.fill(0);
        max.fill(0);
        CheckVectorApproxValues(input, TensorValue({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &testDevice));
    };

    SUBCASE("CPU")
    {
        check(&testDevice);
    }

    SUBCASE("Metal")
    {
        auto device = createDevice(DeviceType::kGPU_METAL);
        if (!device) return;
        check(device.get());
    }
}


TEST_CASE("TensorValue - reduction shapes dtypes and storage")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));
        auto input = TensorValue({1.0f, 5.0f, 5.0f, 4.0f, 4.0f, 3.0f}, {2, 3}, device.get(), DataType::kInt16);
        for (bool keepDim : {false, true})
        {
            CAPTURE(keepDim);
            auto sum = input.sum(-1, keepDim);
            auto max = input.max(-1, keepDim);
            auto index = input.argmax(-1, keepDim);
            const Shape expectedShape = keepDim ? Shape{2, 1} : Shape{2};
            CHECK(sum.shape() == expectedShape);
            CHECK(max.shape() == expectedShape);
            CHECK(index.shape() == expectedShape);
            CHECK(sum.dataType() == DataType::kInt64);
            CHECK(max.dataType() == DataType::kInt16);
            CHECK(index.dataType() == DataType::kInt32);
            device->synchronize();
            CHECK(sum.data<int64_t>()[0] == 11);
            CHECK(sum.data<int64_t>()[1] == 11);
            CHECK(max.data<int16_t>()[0] == 5);
            CHECK(max.data<int16_t>()[1] == 4);
            CHECK(index.data<int32_t>()[0] == 1);
            CHECK(index.data<int32_t>()[1] == 0);
        }
        auto mask = input.argmaxIndices(-1);
        CHECK(mask.shape() == input.shape());
        CHECK(mask.dataType() == DataType::kInt32);
        device->synchronize();
        for (size_t i = 0; i < mask.size(); ++i)
        {
            CHECK(mask.data<int32_t>()[i] == (i == 1 || i == 3 ? 1 : 0));
        }
        CHECK_THROWS_AS(input.sum(-3), std::invalid_argument);
        CHECK_THROWS_AS(input.max(2), std::invalid_argument);
        CHECK_THROWS_AS(input.argmax(-3), std::invalid_argument);
        CHECK_THROWS_AS(input.argmaxIndices(2), std::invalid_argument);

        auto scalar = TensorValue(5.0f, {}, device.get(), DataType::kInt16);
        auto scalarSum = scalar.sum(0, true);
        auto scalarMax = scalar.max(0, false);
        auto scalarIndex = scalar.argmax(0, true);
        auto scalarMask = scalar.argmaxIndices(0);
        CHECK(scalarSum.storage() != scalar.storage());
        CHECK(scalarSum.dataType() == DataType::kInt64);
        CHECK(scalarMax.storage() == scalar.storage());
        CHECK(scalarSum.shape() == Shape{});
        CHECK(scalarMax.shape() == Shape{});
        CHECK(scalarIndex.shape() == Shape{});
        CHECK(scalarMask.shape() == Shape{});
        CHECK(scalarIndex.dataType() == DataType::kInt32);
        CHECK(scalarMask.dataType() == DataType::kInt32);
        device->synchronize();
        CHECK(scalarSum.item<int64_t>() == 5);
        CHECK(scalarMax.item<int16_t>() == 5);
        CHECK(scalarIndex.item<int32_t>() == 0);
        CHECK(scalarMask.item<int32_t>() == 1);
    }
}


TEST_CASE("TensorValue - generalized reduction integration dispatch")
{
    class TrackingDevice : public DeviceCPU
    {
    public:
        void* allocate(size_t size, DataType dtype) override
        {
            ++m_allocations;
            return DeviceCPU::allocate(size, dtype);
        }

        void fill(const void* scalar, DataType dtype, const DeviceTensorParams& dst) override
        {
            CHECK(dst.size != 0);
            ++m_fills;
            DeviceCPU::fill(scalar, dtype, dst);
        }

        void reduce(const DeviceTensorParams& src, const DeviceTensorParams& dst, ReduceOp op,
                    std::span<const size_t> axes = {}, bool accumulate = false) override
        {
            ++m_calls;
            m_op = op;
            m_accumulate = accumulate;
            m_axes.assign(axes.begin(), axes.end());
            CHECK(dst.shape.size() == src.shape.size());
            DeviceCPU::reduce(src, dst, op, axes, accumulate);
        }

        void argmax(const DeviceTensorParams& src, const DeviceTensorParams& dst,
                    std::span<const size_t> axes = {}) override
        {
            ++m_calls;
            m_accumulate = false;
            m_axes.assign(axes.begin(), axes.end());
            DeviceCPU::argmax(src, dst, axes);
        }

        size_t m_calls{0};
        size_t m_allocations{0};
        size_t m_fills{0};
        ReduceOp m_op{};
        bool m_accumulate{};
        Index m_axes;
    } device;

    auto input = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &device);
    auto sum = input.sum();
    CHECK(sum.shape() == Shape{});
    CHECK(sum.item<float>() == 27.0f);
    CHECK(device.m_calls == 1);
    CHECK(device.m_op == ReduceOp::kSum);
    CHECK_FALSE(device.m_accumulate);
    CHECK(device.m_axes.empty());
    auto max = input.max();
    CHECK(max.item<float>() == 8.0f);
    CHECK(device.m_op == ReduceOp::kMax);
    auto index = input.argmax();
    CHECK(index.item<int32_t>() == 1);
    CHECK(device.m_axes.empty());
    CHECK(device.m_calls == 3);

    auto axisSum = input.sum(-1, true);
    CHECK(device.m_calls == 4);
    CHECK(device.m_op == ReduceOp::kSum);
    CHECK_FALSE(device.m_accumulate);
    CHECK(device.m_axes == Index{1});
    CheckVectorApproxValues(axisSum, TensorValue({12.0f, 15.0f}, {2, 1}, &testDevice));
    auto axisMax = input.max(-1, true);
    CHECK(device.m_calls == 5);
    CHECK(device.m_op == ReduceOp::kMax);
    CHECK_FALSE(device.m_accumulate);
    CheckVectorApproxValues(axisMax, TensorValue({8.0f, 6.0f}, {2, 1}, &testDevice));
    auto axisIndex = input.argmax(-1, true);
    CHECK(device.m_calls == 6);
    CHECK_FALSE(device.m_accumulate);
    CHECK(axisIndex.data<int32_t>()[0] == 1);
    CHECK(axisIndex.data<int32_t>()[1] == 2);

    auto transposed = input.transpose(0, 1);
    auto transposedIndex = transposed.argmax(-1, true);
    CHECK(device.m_calls == 7);
    CHECK(transposedIndex.data<int32_t>()[0] == 1);
    CHECK(transposedIndex.data<int32_t>()[1] == 0);
    CHECK(transposedIndex.data<int32_t>()[2] == 1);
    auto mask = transposed.argmaxIndices(-1);
    CHECK(device.m_calls == 7);
    CHECK(mask.data<int32_t>()[1] == 1);
    CHECK(mask.data<int32_t>()[2] == 1);
    CHECK(mask.data<int32_t>()[5] == 1);

    auto lowerRank = input.sum({0}, true).reshape({3});
    CHECK(device.m_calls == 8);
    CheckVectorApproxValues(lowerRank, TensorValue({5.0f, 13.0f, 9.0f}, {3}, &testDevice));

    auto cube = input.reshape({2, 1, 3});
    for (bool keepDim : {false, true})
    {
        for (const auto& axes : {SIndex{2, 0}, SIndex{1}, SIndex{2, 1, 0}, SIndex{}, SIndex{-1, -3}})
        {
            const auto calls = device.m_calls;
            const auto allocations = device.m_allocations;
            auto sum = cube.sum(axes, keepDim);
            CHECK(device.m_calls == calls + 1);
            CHECK(device.m_allocations == allocations + 1);
            CHECK(device.m_op == ReduceOp::kSum);
            Index normalized;
            for (auto axis : axes) normalized.push_back(axis < 0 ? static_cast<ssize_t>(cube.shape().size()) + axis : axis);
            CHECK(device.m_axes == normalized);
            CHECK_FALSE(device.m_accumulate);
            CHECK(sum.storage() != cube.storage());
        }
    }

    auto tensorInput = Tensor(1, {2, 3, 4}, {.m_requireGrad=true, .m_device=&device});
    const auto tensorCalls = device.m_calls;
    const auto tensorAllocations = device.m_allocations;
    auto tensorSum = tensorInput.sum({2, 0});
    CHECK(tensorSum.shape() == Shape{3});
    CHECK(tensorSum.isRequireGrad());
    CHECK(device.m_calls == tensorCalls + 1);
    CHECK(device.m_allocations == tensorAllocations + 1);
    CHECK(device.m_axes == Index{2, 0});
    const auto allocations = device.m_allocations;
    const auto calls = device.m_calls;
    auto reduced = cube.sum({0, 1, 2}, true).reshape({1});
    CHECK(device.m_allocations == allocations + 1);
    CHECK(device.m_calls == calls + 1);
    CHECK(device.m_axes == Index{0, 1, 2});
    CHECK(reduced.shape() == Shape{1});
    CHECK(reduced.data<float>()[0] == 27);
    const auto fills = device.m_fills;
    const auto validationAllocations = device.m_allocations;
    const auto validationCalls = device.m_calls;
    CHECK_THROWS_AS(cube.sum({0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(cube.sum({3}), std::invalid_argument);
    CHECK_THROWS_AS(cube.sum({2, -1}), std::invalid_argument);
    CHECK_THROWS_AS(cube.sum({-4}), std::invalid_argument);
    CHECK(device.m_allocations == validationAllocations);
    CHECK(device.m_calls == validationCalls);
    CHECK(device.m_fills == fills);

    auto empty = TensorValue({2, 0, 3}, &device);
    const auto maxAllocations = device.m_allocations;
    const auto maxFills = device.m_fills;
    const auto maxCalls = device.m_calls;
    CHECK_THROWS_AS(empty.max(1), std::invalid_argument);
    CHECK_THROWS_AS(empty.max(-2, true), std::invalid_argument);
    CHECK(device.m_allocations == maxAllocations);
    CHECK(device.m_fills == maxFills);
    CHECK(device.m_calls == maxCalls);
    for (const auto& axes : {SIndex{}, SIndex{0, 2}, SIndex{0, 1, 2}})
    {
        const auto calls = device.m_calls;
        const auto fills = device.m_fills;
        const auto allocations = device.m_allocations;
        auto sum = empty.sum(axes);
        CHECK(device.m_calls == calls);
        CHECK(device.m_allocations == allocations + 1);
        CHECK(device.m_fills == fills + (sum.size() != 0 ? 1 : 0));
    }

    auto original = std::make_shared<TensorNode>(TensorValue(0, cube.shape(), &device), true);
    TensorNode broadcast(TensorValue{}, true);
    broadcast.m_inputs = {original};
    size_t backwardCalls = 0;
    original->m_backwardFunc = [&](TensorNode*, const TensorValue& seed)
    {
        ++backwardCalls;
        CHECK(seed.data() == cube.data());
        CHECK(seed.shape() == cube.shape());
        CHECK(seed.strides() == cube.strides());
        CHECK(seed.storageOffset() == cube.storageOffset());
    };
    auto backwardAllocations = device.m_allocations;
    auto backwardReductions = device.m_calls;
    auto backwardFills = device.m_fills;
    Tensor::broadcastBackwardFunc(&broadcast, cube);
    CHECK(backwardCalls == 1);
    CHECK(device.m_allocations == backwardAllocations);
    CHECK(device.m_calls == backwardReductions);
    CHECK(device.m_fills == backwardFills);

    original->m_value = TensorValue(0, {1}, &device);
    original->m_backwardFunc = [&](TensorNode*, const TensorValue& seed)
    {
        ++backwardCalls;
        CHECK(seed.shape() == Shape{1});
        CHECK(seed.dataType() == cube.dataType());
        CHECK(seed.device() == &device);
        CHECK(seed.data<float>()[0] == 27);
    };
    backwardAllocations = device.m_allocations;
    backwardReductions = device.m_calls;
    backwardFills = device.m_fills;
    Tensor::broadcastBackwardFunc(&broadcast, cube);
    CHECK(backwardCalls == 2);
    CHECK(device.m_allocations == backwardAllocations + 1);
    CHECK(device.m_calls == backwardReductions + 1);
    CHECK(device.m_fills == backwardFills);
    CHECK(device.m_axes == Index{0, 1, 2});
    CHECK_FALSE(device.m_accumulate);

    original->m_value = TensorValue(0, {1, 1, 3}, &device);
    original->m_backwardFunc = [&](TensorNode*, const TensorValue& seed)
    {
        ++backwardCalls;
        CHECK(seed.shape() == Shape{1, 1, 3});
        for (size_t i = 0; i < seed.size(); ++i) CHECK(seed.data<float>()[i] == 0);
    };
    backwardAllocations = device.m_allocations;
    backwardReductions = device.m_calls;
    backwardFills = device.m_fills;
    Tensor::broadcastBackwardFunc(&broadcast, empty);
    CHECK(backwardCalls == 3);
    CHECK(device.m_allocations == backwardAllocations + 1);
    CHECK(device.m_calls == backwardReductions);
    CHECK(device.m_fills == backwardFills + 1);
}


TEST_CASE("TensorValue - logical argmax and masks for transpose and zero strides")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));
        auto transposed = TensorValue({1.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device.get()).transpose(0, 1);
        auto indices = transposed.argmax(-1, true);
        auto mask = transposed.argmaxIndices(-1);
        device->synchronize();
        CHECK(indices.shape() == Shape{3, 1});
        CHECK(indices.data<int32_t>()[0] == 1);
        CHECK(indices.data<int32_t>()[1] == 0);
        CHECK(indices.data<int32_t>()[2] == 1);
        const int32_t transposeMask[] = {0, 1, 1, 0, 0, 1};
        for (size_t i = 0; i < 6; ++i) CHECK(mask.data<int32_t>()[i] == transposeMask[i]);

        auto broadcast = TensorValue({1.0f, 8.0f, 3.0f}, {3, 1}, device.get()).broadcastTo({3, 2});
        auto broadcastIndices = broadcast.argmax(-1, false);
        auto broadcastMask = broadcast.argmaxIndices(-1);
        auto globalIndex = broadcast.argmax();
        auto globalMask = broadcast.argmaxIndices();
        device->synchronize();
        CHECK(broadcastIndices.shape() == Shape{3});
        for (size_t i = 0; i < 3; ++i) CHECK(broadcastIndices.data<int32_t>()[i] == 0);
        CHECK(globalIndex.item<int32_t>() == 2);
        for (size_t i = 0; i < 6; ++i)
        {
            CHECK(broadcastMask.data<int32_t>()[i] == (i % 2 == 0 ? 1 : 0));
            CHECK(globalMask.data<int32_t>()[i] == (i == 2 ? 1 : 0));
        }
    }
}


TEST_CASE("TensorValue - multi-axis sum reshape preserves broadcast gradient values")
{
    DeviceCPU reference;
    for (auto type : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(type);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(type));
        auto check = [&](TensorValue& input, const Shape& target, const SIndex& axes,
                         std::initializer_list<float> values)
        {
            auto actual = input.sum(axes, true).reshape(target);
            auto canonical = input.shape();
            for (auto axis : axes) canonical[axis] = 1;
            auto promoted = input.to(actual.dataType());
            auto direct = TensorValue(0, canonical, device.get(), actual.dataType());
            const Index deviceAxes(axes.begin(), axes.end());
            device->reduce(promoted.deviceParams(), direct.deviceParams(), ReduceOp::kSum, deviceAxes, true);
            device->synchronize();
            CHECK(actual.shape() == target);
            CHECK(actual.dataType() == (input.dataType() > DataType::kInt64 ? DataType::kInt64 : input.dataType()));
            CHECK(actual.device() == device.get());
            CHECK(actual.storage() != input.storage());
            CHECK(actual.storageOffset() == 0);
            REQUIRE(actual.size() == direct.size());
            CHECK(std::memcmp(actual.data(), direct.data(), actual.size() * Device::dataTypeSize(actual.dataType())) == 0);
            CheckVectorApproxValues(actual.to(&reference).to(DataType::kFloat32),
                                    TensorValue(values, target, &reference));
        };
        for (auto dtype : {DataType::kFloat64, DataType::kFloat32, DataType::kFloat16,
                           DataType::kBFloat16, DataType::kInt32})
        {
            if (type == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(static_cast<size_t>(dtype));
            auto input = TensorValue({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3}, device.get(), dtype);
            check(input, {2, 3}, {0}, {8, 10, 12, 14, 16, 18});
            check(input, {3}, {0, 1}, {22, 26, 30});
            check(input, {1, 3}, {0, 1}, {22, 26, 30});
            check(input, {2, 1}, {0, 2}, {30, 48});
            check(input, {}, {0, 1, 2}, {78});
            check(input, {1}, {0, 1, 2}, {78});

            auto singleton = TensorValue({1, 2, 3, 4, 5, 6}, {1, 2, 3}, device.get(), dtype);
            check(singleton, {2, 3}, {0}, {1, 2, 3, 4, 5, 6});
            auto one = TensorValue(7, {1, 1}, device.get(), dtype);
            check(one, {}, {0, 1}, {7});
            check(one, {1}, {0}, {7});

            if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16)
            {
                const float large = dtype == DataType::kFloat16 ? 2048.0f : 256.0f;
                auto rounded = TensorValue({large, large, 1.0f, 1.0f, -large, -large}, {3, 2}, device.get(), dtype);
                const float expected = type == DeviceType::kCPU ? 0.0f : 1.0f;
                check(rounded, {2}, {0}, {expected, expected});
            }
        }

        for (const auto& shape : {Shape{0}, Shape{2, 0}, Shape{0, 3}, Shape{2, 0, 3}, Shape{0, 0}})
        {
            auto base = TensorValue(shape, device.get());
            for (size_t layout = 0; layout < (shape.size() == 1 ? 1 : 2); ++layout)
            {
                auto input = layout == 0 ? base.contiguous() : base.transpose(0, static_cast<ssize_t>(shape.size() - 1));
                SIndex allAxes(input.shape().size());
                std::iota(allAxes.begin(), allAxes.end(), 0);
                auto leadingAxes = allAxes;
                leadingAxes.pop_back();
                for (const auto& [target, axes] : {std::pair{Shape{}, allAxes},
                                                  std::pair{Shape(input.shape().size(), 1), allAxes},
                                                  std::pair{Shape{input.shape().back()}, leadingAxes}})
                {
                    auto sum = axes.empty() ? input.reshape(target) : input.sum(axes, true).reshape(target);
                    device->synchronize();
                    CHECK(sum.shape() == target);
                    CHECK(sum.dataType() == input.dataType());
                    CHECK(sum.device() == device.get());
                    for (size_t i = 0; i < sum.size(); ++i) CHECK(sum.data<float>()[i] == 0.0f);
                }
            }
        }
    }
}


TEST_CASE("TensorValue - reduction integration preserves numerics and shape mapping")
{
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));
        for (auto dtype : {DataType::kFloat64, DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16})
        {
            if (deviceType == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(static_cast<size_t>(dtype));
            auto base = TensorValue({1.0f, -0.125f, 3.0f, 0.0625f, -2.0f, 6.0f,
                                     0.5f, 1.0f, -3.0f, 4.0f, 0.25f, 2.0f}, {3, 4}, device.get(), dtype);
            for (size_t layout = 0; layout < 4; ++layout)
            {
                CAPTURE(layout);
                auto input = layout == 0 ? base.contiguous() :
                             layout == 1 ? base.transpose(0, 1) :
                             layout == 2 ? base.slice(1, 1, 4) : base.slice(0, 0, 1).broadcastTo({3, 4});
                auto checkExact = [&](const TensorValue& actual, const TensorValue& expected)
                {
                    device->synchronize();
                    REQUIRE(actual.shape() == expected.shape());
                    REQUIRE(actual.dataType() == expected.dataType());
                    CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * Device::dataTypeSize(actual.dataType())) == 0);
                };
                auto directSum = TensorValue({}, device.get(), dtype);
                reduceAll(*device, input.deviceParams(), directSum.deviceParams(), ReduceOp::kSum);
                checkExact(input.sum(), directSum);
                Index axes(input.shape().size());
                std::iota(axes.begin(), axes.end(), 0);
                auto explicitSum = TensorValue(0, Shape(input.shape().size(), 1), device.get(), dtype);
                reduceAxes(*device, input.deviceParams(), explicitSum.deviceParams(), axes, ReduceOp::kSum);
                checkExact(input.sum(SIndex(axes.begin(), axes.end()), true), explicitSum);
                auto directMax = TensorValue({}, device.get(), dtype);
                reduceAll(*device, input.deviceParams(), directMax.deviceParams(), ReduceOp::kMax);
                checkExact(input.max(), directMax);
                auto directIndex = TensorValue({}, device.get(), DataType::kInt32);
                argmaxAll(*device, input.deviceParams(), directIndex.deviceParams());
                checkExact(input.argmax(), directIndex);
                for (size_t axis = 0; axis < input.shape().size(); ++axis)
                {
                    auto shape = input.shape();
                    shape[axis] = 1;
                    auto sum = TensorValue(0.0f, shape, device.get(), dtype);
                    reduceAxes(*device, input.deviceParams(), sum.deviceParams(), {axis}, ReduceOp::kSum);
                    checkExact(input.sum(static_cast<ssize_t>(axis), true), sum);
                    auto max = TensorValue(shape, device.get(), dtype);
                    device->fillMin(max.deviceParams());
                    reduceAxes(*device, input.deviceParams(), max.deviceParams(), {axis}, ReduceOp::kMax);
                    checkExact(input.max(static_cast<ssize_t>(axis), true), max);
                    auto index = TensorValue(shape, device.get(), DataType::kInt32);
                    argmaxAxis(*device, input.deviceParams(), index.deviceParams(), axis);
                    checkExact(input.argmax(static_cast<ssize_t>(axis), true), index);
                    auto mask = TensorValue(0.0f, input.shape(), device.get(), DataType::kInt32);
                    const size_t maskAxes[] = {axis};
                    device->argmaxMask(input.deviceParams(), mask.deviceParams(), maskAxes);
                    checkExact(input.argmaxIndices(static_cast<ssize_t>(axis)), mask);
                }
            }
            auto special = TensorValue({-0.0f, 0.0f, std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()},
                                       {1, 5, 1}, device.get(), dtype);
            auto all = special.sum({});
            auto global = special.sum();
            auto sum = special.sum({2, 0}, true);
            auto direct = TensorValue(0, special.shape(), device.get(), dtype);
            reduceAxes(*device, special.deviceParams(), direct.deviceParams(), {2, 0}, ReduceOp::kSum);
            device->synchronize();
            CHECK(all.shape() == Shape{});
            CHECK(std::memcmp(all.data(), global.data(), Device::dataTypeSize(dtype)) == 0);
            CHECK(std::memcmp(sum.data(), direct.data(), sum.size() * Device::dataTypeSize(dtype)) == 0);
        }

        auto cube = TensorValue({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {2, 3, 4}, device.get());
        auto expected = TensorValue({15, 18, 21, 24, 51, 54, 57, 60}, {2, 4}, &testDevice);
        CheckVectorApproxValues(cube.sum({1}, true).reshape({2, 4}), expected);
        auto negativeInfinity = TensorValue(-std::numeric_limits<float>::infinity(), {2, 1}, device.get());
        auto globalMax = negativeInfinity.max();
        auto axisMax = negativeInfinity.max(0, true);
        device->synchronize();
        CHECK(globalMax.item<float>() == -std::numeric_limits<float>::infinity());
        CHECK(axisMax.data<float>()[0] == -std::numeric_limits<float>::infinity());
    }
}


TEST_CASE("TensorValue - empty CPU reduction integration returns initialized results")
{
    DeviceCPU device;
    for (auto dtype : {DataType::kFloat32, DataType::kFloat64, DataType::kInt32})
    {
        CAPTURE(static_cast<size_t>(dtype));
        for (const auto& shape : {Shape{0}, Shape{2, 0}, Shape{0, 3}, Shape{2, 0, 3}, Shape{0, 0}})
        {
            auto base = TensorValue(shape, &device, dtype);
            for (size_t layout = 0; layout < (shape.size() == 1 ? 1 : 2); ++layout)
            {
                CAPTURE(layout);
                auto input = layout == 0 ? base.contiguous() : base.transpose(0, static_cast<ssize_t>(shape.size() - 1));
                auto zeroSum = TensorValue(0, {}, &device, dtype);
                auto sum = input.sum();
                CHECK(sum.shape() == Shape{});
                CHECK(sum.dataType() == (dtype > DataType::kInt64 ? DataType::kInt64 : dtype));
                CHECK(sum.device() == &device);
                CHECK(sum.storage() != input.storage());
                CHECK(sum.to(DataType::kFloat64).item<double>() == 0.0);
                CHECK(sum.to(DataType::kFloat64).item<double>() == zeroSum.to(DataType::kFloat64).item<double>());
                CHECK_THROWS_AS(input.max(), std::invalid_argument);
                CHECK_THROWS_AS(input.argmax(), std::invalid_argument);

                for (size_t axis = 0; axis < input.shape().size(); ++axis)
                {
                    CAPTURE(axis);
                    auto resultShape = input.shape();
                    resultShape[axis] = 1;
                    CHECK_THROWS_AS(input.argmax(static_cast<ssize_t>(axis)), std::invalid_argument);
                    for (bool keepDim : {false, true})
                    {
                        if (input.shape()[axis] == 0)
                        {
                            CHECK_THROWS_AS(input.max(static_cast<ssize_t>(axis), keepDim), std::invalid_argument);
                            CHECK_THROWS_AS(input.max(static_cast<ssize_t>(axis) - static_cast<ssize_t>(input.shape().size()), keepDim),
                                            std::invalid_argument);
                            continue;
                        }
                        auto max = input.max(static_cast<ssize_t>(axis) - static_cast<ssize_t>(input.shape().size()), keepDim);
                        auto expectedShape = resultShape;
                        if (!keepDim) expectedShape.erase(expectedShape.begin() + axis);
                        CHECK(max.shape() == expectedShape);
                        CHECK(max.dataType() == dtype);
                        CHECK(max.device() == &device);
                        CHECK(max.storage() != input.storage());
                        CHECK(max.storageOffset() == 0);
                        CHECK(max.size() == 0);
                    }
                }
            }
        }
    }
}


TEST_CASE("TensorValue - reduction argmax bounds on valid broadcast storage")
{
    DeviceCPU cpu;
    auto metal = createDevice(DeviceType::kGPU_METAL);
    for (Device* device : {static_cast<Device*>(&cpu), metal.get()})
    {
        if (!device) continue;
        auto input = TensorValue(1, {1}, device).broadcastTo({static_cast<size_t>(INT32_MAX) + 2});
        CHECK_THROWS_AS(input.argmax(), std::invalid_argument);
        CHECK_THROWS_AS(input.argmax(0), std::invalid_argument);
    }
}

TEST_CASE("TensorValue - Metal empty reduction integration stays safe")
{
    auto device = createDevice(DeviceType::kGPU_METAL);
    if (!device) return;
    for (size_t dtypeIndex = 1; dtypeIndex < DataTypeCount; ++dtypeIndex)
    {
        const auto dtype = static_cast<DataType>(dtypeIndex);
        for (const auto& shape : {Shape{0}, Shape{2, 0}, Shape{0, 3}, Shape{2, 0, 3}, Shape{0, 0}})
        {
            auto base = TensorValue(shape, device.get(), dtype);
            auto input = shape.size() == 1 ? base.contiguous() : base.transpose(0, shape.size() - 1);
            auto sum = input.sum().to(DataType::kFloat32);
            CHECK_THROWS_AS(input.max(), std::invalid_argument);
            CHECK_THROWS_AS(input.argmax(), std::invalid_argument);
            device->synchronize();
            CHECK(sum.item<float>() == 0);
            for (size_t axis = 0; axis < shape.size(); ++axis)
            {
                CHECK_THROWS_AS(input.argmax(axis), std::invalid_argument);
                if (input.shape()[axis] == 0)
                {
                    CHECK_THROWS_AS(input.max(axis, true), std::invalid_argument);
                    CHECK_THROWS_AS(input.max(axis, false), std::invalid_argument);
                    continue;
                }
                auto result = input.max(axis, true);
                auto expectedShape = input.shape();
                expectedShape[axis] = 1;
                CHECK(result.shape() == expectedShape);
                CHECK(result.size() == 0);
                device->synchronize();
            }
        }
    }
}

TEST_CASE("TensorValue - Mean")
{
    auto x = TensorValue({1.0, 2.0, 3.0, 4.0}, {2, 2}, &testDevice);

    CheckVectorApproxValues(x.mean(), TensorValue(2.5, &testDevice));
}


TEST_CASE("TensorValue - Sqrt")
{
    auto x = TensorValue({4.0, 9.0, 16.0}, {1, 3}, &testDevice);
    auto y = x.sqrt();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({2.0, 3.0, 4.0}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Sin")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.sin();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({float(std::sin(0.5)),
                                            float(std::sin(0)),
                                            float(std::sin(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("TensorValue - Cos")
{
    auto x = TensorValue({0.5, 0.0, -0.5}, {1, 3}, &testDevice);
    auto y = x.cos();

    CHECK(y.shape() == Shape{1, 3});
    CheckVectorApproxValues(y, TensorValue({float(std::cos(0.5)),
                                            float(std::cos(0)),
                                            float(std::cos(-0.5))}, y.shape(), &testDevice));
}


TEST_CASE("Default Device - Sqrt maps negative Int32 to zero")
{
    std::array<int32_t, 3> source = {-1, 0, 4};
    TensorValue src(source.data(), source.size(), DataType::kInt32, {1, source.size()}, &testDevice, DataType::kInt32);
    TensorValue result({1, source.size()}, &testDevice, DataType::kInt32);

    testDevice.sqrt(src.deviceParams(), result.deviceParams());

    std::array<int32_t, 3> expected = {0, 0, 2};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        CHECK(result.data<int32_t>()[i] == expected[i]);
    }
}


TEST_CASE("Default Device - Log clamps integral nonfinite results")
{
    std::array<int32_t, 4> source = {-1, 0, 1, 10};
    TensorValue src(source.data(), source.size(), DataType::kInt32, {1, source.size()}, &testDevice, DataType::kInt32);
    TensorValue result({1, source.size()}, &testDevice, DataType::kInt32);

    testDevice.log(src.deviceParams(), result.deviceParams());

    std::array<int32_t, 4> expected = {0, std::numeric_limits<int32_t>::lowest(), 0, 2};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        CHECK(result.data<int32_t>()[i] == expected[i]);
    }
}


TEST_CASE("Default Device - Exp clamps integral overflow")
{
    std::array<int32_t, 4> source = {0, 1, 10, 100};
    TensorValue src(source.data(), source.size(), DataType::kInt32, {1, source.size()}, &testDevice, DataType::kInt32);
    TensorValue result({1, source.size()}, &testDevice, DataType::kInt32);

    testDevice.exp(src.deviceParams(), result.deviceParams());

    std::array<int32_t, 4> expected = {1, 2, 22026, std::numeric_limits<int32_t>::max()};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        CHECK(result.data<int32_t>()[i] == expected[i]);
    }
}


TEST_CASE("Default Device - Pow clamps integral overflow")
{
    std::array<int32_t, 3> base = {2, 100, -2};
    std::array<int32_t, 3> exponent = {8, 100, 31};
    TensorValue baseTensor(base.data(), base.size(), DataType::kInt32, {1, base.size()}, &testDevice, DataType::kInt32);
    TensorValue exponentTensor(exponent.data(), exponent.size(), DataType::kInt32,
                               {1, exponent.size()}, &testDevice, DataType::kInt32);
    TensorValue result({1, base.size()}, &testDevice, DataType::kInt32);

    testDevice.pow(baseTensor.deviceParams(), exponentTensor.deviceParams(), result.deviceParams());

    std::array<int32_t, 3> expected = {256, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::lowest()};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        CHECK(result.data<int32_t>()[i] == expected[i]);
    }
}


TEST_CASE("TensorValue - Device Switch")
{
    auto x = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);

    DeviceCPU  newDevice;
    auto newX = x.to(&newDevice);

    CHECK(newX.device() == &newDevice);
    CheckVectorApproxValues(x, TensorValue({1.0, 2.0, 3.0}, x.shape(), &newDevice));
}


TEST_CASE("TensorValue - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = TensorValue(5.0, {}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = TensorValue(5.0, {}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = TensorValue(5.0, {1}, &testDevice);
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = TensorValue(5.0, {1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = TensorValue(5.0, {1,1}, &testDevice);
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = TensorValue(5.0, {1,1}, &testDevice);
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(5.0, newShape, &testDevice));
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0};
        auto input = TensorValue(data, {1,3}, &testDevice);
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {6}, &testDevice);
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = TensorValue(data, {2,3}, &testDevice);
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.size() == input.size());
        CheckVectorApproxValues(x, TensorValue(data, newShape, &testDevice));
    }

    SUBCASE("Stride-compatible non-contiguous reshape stays a view")
    {
        auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f,
                                  9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &testDevice);
        auto view = input.slice(1, 0, 4, 2);
        auto reshaped = view.reshape({6});

        CHECK_FALSE(view.isContiguous());
        CHECK(reshaped.storage() == view.storage());
        CHECK(reshaped.storageOffset() == view.storageOffset());
        CHECK(reshaped.strides() == Stride{2});
        CheckVectorApproxValues(reshaped, TensorValue({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {6}, &testDevice));
    }

    SUBCASE("Singleton-dimension reshape on non-contiguous view stays metadata-only")
    {
        auto input = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f,
                                  9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, &testDevice);
        auto view = input.slice(1, 0, 4, 2);
        auto reshaped = view.reshape({3, 1, 2});

        CHECK(reshaped.storage() == view.storage());
        CHECK(reshaped.storageOffset() == view.storageOffset());
        CHECK(reshaped.strides() == Stride{4, 4, 2});
        CheckVectorApproxValues(reshaped, TensorValue({1.0f, 3.0f,
                                                       5.0f, 7.0f,
                                                       9.0f, 11.0f}, {3, 1, 2}, &testDevice));
    }

    SUBCASE("Broadcast stride-0 reshape remains rejected")
    {
        auto view = TensorValue({1.0f, 2.0f, 3.0f}, {1, 3}, &testDevice).broadcastTo({2, 3});

        CHECK_FALSE(view.isContiguous());
        DOCTEST_CHECK_THROWS_AS(view.reshape({6}), std::invalid_argument);
    }

    SUBCASE("Invalid reshape throws invalid_argument")
    {
        TensorValue inputs = TensorValue({1.0,2.0,3.0,4.0,5.0}, {1,5}, &testDevice);
        // Check that reshape throws an invalid_argument exception
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({2,2}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({1,6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({4}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({}), std::invalid_argument);
    }
}


TEST_CASE("TensorValue - broadcastTo")
{
    SUBCASE("Scalar to Scalar")
    {
        Shape newShape{};
        auto bct = TensorValue(1.0, {}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 1);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0}, newShape, &testDevice));
    }

    SUBCASE("Scalar to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue(1.0, {}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0,1.0,1.0,1.0,1.0,1.0}, newShape, &testDevice));
    }

    SUBCASE("[1] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue(1.0, {1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0,1.0,1.0,1.0,1.0,1.0}, newShape, &testDevice));
    }

    SUBCASE("[2x1] to [2x2x3]")
    {
        Shape newShape{2,2,3};
        auto bct = TensorValue({1.0,2.0}, {2,1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 12);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0,1.0,1.0,
                                                  2.0,2.0,2.0,
                                                  1.0,1.0,1.0,
                                                  2.0,2.0,2.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [1x3]")
    {
        Shape newShape{1, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 3);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3] to [3x3]")
    {
        Shape newShape{3, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 9);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0,
                                                  1.0, 2.0, 3.0,
                                                  1.0, 2.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[2x1] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0}, {2, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0, 1.0, 2.0, 2.0, 2.0}, newShape, &testDevice));
    }

    SUBCASE("[2x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice));
    }

    SUBCASE("[3x1] to [3x3]")
    {
        Shape newShape{3, 3};
        auto bct = TensorValue({1.0, 2.0, 3.0}, {3, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 9);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0, 1.0,
                                                  2.0, 2.0, 2.0,
                                                  3.0, 3.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[1x3x1] to [1x2x3x2]")
    {
        Shape newShape{1,2,3,2};
        auto bct = TensorValue({1.0,2.0,3.0}, {1, 3, 1}, &testDevice).broadcastTo(newShape);
        CHECK(bct.size() == 12);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, TensorValue({1.0, 1.0,
                                                  2.0, 2.0,
                                                  3.0, 3.0,
                                                  1.0, 1.0,
                                                  2.0, 2.0,
                                                  3.0, 3.0}, newShape, &testDevice));
    }

    SUBCASE("[4] to [2]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0, 4.0}, {4}, &testDevice).broadcastTo({2}), std::invalid_argument);
    }

    SUBCASE("[2] to [4]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0}, {2}, &testDevice).broadcastTo({4}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [6]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({6}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [1x3]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({1, 3}), std::invalid_argument);
    }

    SUBCASE("[2x3] to [3x2]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 3}, &testDevice).broadcastTo({3, 2}), std::invalid_argument);
    }

    SUBCASE("[2x1x3] to [2x3]")
    {
        CHECK_THROWS_AS(TensorValue({1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0}, {2, 1, 3}, &testDevice).broadcastTo({2, 3}), std::invalid_argument);
    }
}


TEST_CASE("TensorValue - broadcastTo view metadata")
{
    auto base = TensorValue({1.0, 2.0, 3.0}, {1, 3}, &testDevice);
    auto view = base.broadcastTo({2, 3});

    CHECK(view.shape() == Shape{2, 3});
    CHECK(view.strides() == Stride{0, 1});
    CHECK(view.storageOffset() == 0);
    CHECK_FALSE(view.isContiguous());
}


TEST_CASE("TensorValue - slice view metadata")
{
    auto base = TensorValue({1.0, 2.0, 3.0, 4.0,
                             5.0, 6.0, 7.0, 8.0,
                             9.0, 10.0, 11.0, 12.0}, {3, 4}, &testDevice);
    auto view = base.slice(1, 1, 4, 2);

    CHECK(view.shape() == Shape{3, 2});
    CHECK(view.strides() == Stride{4, 2});
    CHECK(view.storageOffset() == 1);
    CHECK_FALSE(view.isContiguous());
}


TEST_CASE("TensorValue - index validation accepts logical views")
{
    // Observe dispatch without depending on any backend's indexing kernels.
    // The shared TensorValue validator must accept the logical indices even
    // when their backing storage contains out-of-range values outside the view.
    struct IndexValidationDevice : DeviceCPU
    {
        bool m_indexSelectCalled{false};
        bool m_indexAddCalled{false};

        void indexSelect(const DeviceTensorParams&, const DeviceTensorParams&,
                         const DeviceTensorParams&, size_t) override
        {
            m_indexSelectCalled = true;
        }

        void indexAdd(const DeviceTensorParams&, const DeviceTensorParams&,
                      const DeviceTensorParams&, size_t) override
        {
            m_indexAddCalled = true;
        }
    };

    for (size_t layout = 0; layout < 6; ++layout)
    {
        CAPTURE(layout);
        IndexValidationDevice device;
        auto values = TensorValue({10.0f, 20.0f, 30.0f}, {3}, &device);
        auto indices = [&]()
        {
            if (layout == 0) // Strided vector.
                return TensorValue({1, 9, 0, 7}, {4}, &device, DataType::kInt32).slice(0, 0, 4, 2);
            if (layout == 1) // Contiguous vector with a storage offset.
                return TensorValue({9, 1, 0, 9}, {4}, &device, DataType::kInt32).slice(0, 1, 3);
            if (layout == 2) // Stride and offset together.
                return TensorValue({9, 1, 9, 0, 9}, {5}, &device, DataType::kInt32).slice(0, 1, 5, 2);
            if (layout == 3) // Zero-stride broadcast view with an offset.
                return TensorValue({9, 1}, {2}, &device, DataType::kInt32).slice(0, 1, 2).broadcastTo({2});
            if (layout == 4) // Scalar view with an offset.
                return TensorValue({9, 1}, {2}, &device, DataType::kInt32).select(0, 1);
            return TensorValue({1, 0}, {2}, &device, DataType::kInt32);
        }();

        auto selected = values.indexSelect(0, indices);
        CHECK(device.m_indexSelectCalled);
        CHECK(selected.shape() == Shape{indices.size()});

        auto updates = TensorValue(1.0f, selected.shape(), &device);
        auto added = values.indexAdd(0, indices, updates);
        CHECK(device.m_indexAddCalled);
        CHECK(added.shape() == values.shape());
    }
}


TEST_CASE("TensorValue - Metal index validation accepts offset views")
{
    auto device = aix::createDevice(DeviceType::kGPU_METAL);
    if (!device) return;

    for (size_t step : {1, 2})
    {
        CAPTURE(step);
        auto values = TensorValue({10.0f, 20.0f, 30.0f}, {3}, device.get());
        auto indices = [&]()
        {
            if (step == 1)
                return TensorValue({9, 1, 0}, {3}, device.get(), DataType::kInt32).slice(0, 1, 3);
            return TensorValue({9, 1, 9, 0}, {4}, device.get(), DataType::kInt32).slice(0, 1, 4, 2);
        }();
        REQUIRE(indices.storageOffset() == 1);
        REQUIRE(indices.strides() == Stride{step});

        auto selected = values.indexSelect(0, indices);
        CheckVectorApproxValues(selected, TensorValue({20.0f, 10.0f}, {2}, &testDevice));

        auto updates = TensorValue({4.0f, 5.0f}, {2}, device.get());
        auto added = values.indexAdd(0, indices, updates);
        CheckVectorApproxValues(added, TensorValue({15.0f, 24.0f, 30.0f}, {3}, &testDevice));
    }
}


TEST_CASE("TensorValue - contiguous normalizes views across devices")
{
    const std::pair<Shape, std::vector<double>> cases[] = {
        {{3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
        {{2, 4}, {5, 6, 7, 8, 9, 10, 11, 12}},
        {{3}, {6, 7, 8}},
        {{}, {7}},
        {{3, 2}, {2, 4, 6, 8, 10, 12}},
        {{2, 3}, {7, 7, 7, 7, 7, 7}},
        {{4, 3}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}}
    };
    for (auto deviceType : {DeviceType::kCPU, DeviceType::kGPU_METAL})
    {
        auto device = aix::createDevice(deviceType);
        if (!device) continue;
        CAPTURE(static_cast<size_t>(deviceType));

        for (size_t i = 0; i < aix::DataTypeCount; ++i)
        {
            const auto dtype = static_cast<DataType>(i);
            if (deviceType == DeviceType::kGPU_METAL && dtype == DataType::kFloat64) continue;
            CAPTURE(i);
            auto base = TensorValue({1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, device.get(), dtype);
            for (size_t layout = 0; layout < std::size(cases); ++layout)
            {
                CAPTURE(layout);
                auto view = [&]()
                {
                    if (layout == 0) return base.reshape({3, 4});
                    if (layout == 1) return base.slice(0, 1, 3);
                    if (layout == 2) return base.select(0, 1).slice(0, 1, 4);
                    if (layout == 3) return base.select(0, 1).select(0, 2);
                    if (layout == 4) return base.slice(1, 1, 4, 2);
                    if (layout == 5) return base.select(0, 1).select(0, 2).broadcastTo({2, 3});
                    return base.transpose(0, 1);
                }();
                if (layout >= 1 && layout <= 3)
                {
                    REQUIRE(view.isContiguous());
                    REQUIRE(view.storageOffset() > 0);
                }

                auto result = view.contiguous();
                device->synchronize();
                CHECK(result.shape() == cases[layout].first);
                CHECK(result.dataType() == dtype);
                CHECK(result.isContiguous());
                CHECK(result.storageOffset() == 0);
                CHECK((result.storage() == view.storage()) == (layout == 0));

                auto hostResult = result.to(&testDevice).to(DataType::kFloat64);
                const auto& expected = cases[layout].second;
                REQUIRE(hostResult.size() == expected.size());
                for (size_t j = 0; j < expected.size(); ++j)
                {
                    CAPTURE(j);
                    CHECK(hostResult.data<double>()[j] == expected[j]);
                }
            }
        }
    }
}


TEST_CASE("TensorValue - transpose view metadata")
{
    auto base = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                             13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                             19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, {2, 3, 4}, &testDevice);
    auto view = base.transpose(0, 2);

    CHECK(view.shape() == Shape{4, 3, 2});
    CHECK(view.strides() == Stride{1, 4, 12});
    CHECK(view.storageOffset() == 0);
    CHECK_FALSE(view.isContiguous());
}


TEST_CASE("TensorValue - permute view metadata")
{
    auto base = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                             13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                             19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, {2, 3, 4}, &testDevice);
    auto view = base.permute({1, 2, 0});

    CHECK(view.shape() == Shape{3, 4, 2});
    CHECK(view.strides() == Stride{4, 1, 12});
    CHECK(view.storageOffset() == 0);
    CHECK_FALSE(view.isContiguous());
}


TEST_CASE("TensorValue - view aliasing through shared storage mirror")
{
    auto base = TensorValue({1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0}, {2, 3}, &testDevice);
    auto view = base.slice(0, 0, 2, 1);
    auto mirrored = TensorValue(view.storage(), view.size(), view.storageOffset(), view.shape(), view.strides(),
                                view.device(), view.dataType());

    mirrored.getValueAt<float>({1, 2}) = 42.0f;

    CHECK(base.getValueAt<float>({1, 2}) == Approx(42.0f));
    CHECK(view.getValueAt<float>({1, 2}) == Approx(42.0f));
    CHECK(mirrored.getValueAt<float>({1, 2}) == Approx(42.0f));
}


TEST_CASE("TensorValue - no operation value returns share storage")
{
    auto base = TensorValue({1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f}, {2, 3}, &testDevice);

    SUBCASE("reshape")
    {
        auto result = base.reshape({2, 3});

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
    }

    SUBCASE("broadcastTo")
    {
        auto result = base.broadcastTo({2, 3});

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
    }

    SUBCASE("contiguous")
    {
        auto result = base.contiguous();

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
    }

    SUBCASE("sum on scalar")
    {
        auto scalar = TensorValue(5.0f, {}, &testDevice);
        auto result = scalar.sum(0);

        CHECK(result.storage() == scalar.storage());
        CHECK(result.storageOffset() == scalar.storageOffset());
        CHECK(result.strides() == scalar.strides());
    }

    SUBCASE("max on scalar")
    {
        auto scalar = TensorValue(5.0f, {}, &testDevice);
        auto result = scalar.max(0);

        CHECK(result.storage() == scalar.storage());
        CHECK(result.storageOffset() == scalar.storageOffset());
        CHECK(result.strides() == scalar.strides());
    }

    SUBCASE("permute identity")
    {
        auto result = base.permute({0, 1});

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
    }

    SUBCASE("squeeze no-op")
    {
        auto result = base.squeeze(0);

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
    }

    SUBCASE("sliceSet in-place")
    {
        auto replacement = TensorValue({9.0f, 8.0f, 7.0f}, {1, 3}, &testDevice);
        auto result = base.sliceSet(replacement, 0, 1, 2, 1, true);

        CHECK(result.storage() == base.storage());
        CHECK(result.storageOffset() == base.storageOffset());
        CHECK(result.strides() == base.strides());
        CheckVectorApproxValues(base, TensorValue({1.0f, 2.0f, 3.0f,
                                                   9.0f, 8.0f, 7.0f}, {2, 3}, &testDevice));
    }

    SUBCASE("indexAdd in-place")
    {
        auto vector = TensorValue({1.0f, 2.0f, 3.0f}, {3}, &testDevice);
        auto indices = TensorValue({1}, {1}, &testDevice, DataType::kInt32);
        auto source = TensorValue({10.0f}, {1}, &testDevice);
        auto result = vector.indexAdd(0, indices, source, true);

        CHECK(result.storage() == vector.storage());
        CHECK(result.storageOffset() == vector.storageOffset());
        CHECK(result.strides() == vector.strides());
        CheckVectorApproxValues(vector, TensorValue({1.0f, 12.0f, 3.0f}, {3}, &testDevice));
    }

    SUBCASE("cat single tensor")
    {
        std::vector<TensorValue> tensors{base};
        auto result = TensorValue::cat(tensors, 1);

        CHECK(result.storage() == tensors[0].storage());
        CHECK(result.storageOffset() == tensors[0].storageOffset());
        CHECK(result.strides() == tensors[0].strides());
    }
}


TEST_CASE("TensorValue - broadcast")
{
    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3]")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = TensorValue(2.0, shape, &testDevice);
            auto y = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice);

            auto a1 = x + y;
            CHECK(a1.size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            // Try reverse order
            auto a2 = y + x;
            CHECK(a2.size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            auto s1 = x - y;
            CHECK(s1.size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, TensorValue({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape, &testDevice));

            // Try reverse order
            auto s2 = y - x;
            CHECK(s2.size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, TensorValue({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape, &testDevice));

            auto m1 = x * y;
            CHECK(m1.size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            // Try reverse order
            auto m2 = y * x;
            CHECK(m2.size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            auto d1 = x / y;
            CHECK(d1.size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, TensorValue({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape, &testDevice));

            // Try reverse order
            auto d2 = y / x;
            CHECK(d2.size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, TensorValue({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape, &testDevice));
        }
    }

    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3] - In-place operations")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = TensorValue(2.0, shape, &testDevice);
            auto y = TensorValue({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape, &testDevice);

            auto a1 = x;
            a1 += y;
            CHECK(a1.size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            // Try reverse order
            auto a2 = y;
            a2 += x;
            CHECK(a2.size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, TensorValue({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape, &testDevice));

            auto s1 = x;
            s1 -= y;
            CHECK(s1.size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, TensorValue({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape, &testDevice));

            // Try reverse order
            auto s2 = y;
            s2 -= x;
            CHECK(s2.size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, TensorValue({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape, &testDevice));

            auto m1 = x;
            m1 *= y;
            CHECK(m1.size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            // Try reverse order
            auto m2 = y;
            m2 *= x;
            CHECK(m2.size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, TensorValue({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape, &testDevice));

            auto d1 = x;
            d1 /= y;
            CHECK(d1.size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, TensorValue({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape, &testDevice));

            // Try reverse order
            auto d2 = y;
            d2 /= x;
            CHECK(d2.size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, TensorValue({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape, &testDevice));
        }
    }

    SUBCASE("[2x3] [3x2]")
    {
        std::initializer_list<float> data{1.0, 2.0, 3.0,4.0, 5.0, 6.0};
        Shape shape1{2,3};
        Shape shape2{3,2};
        auto tensorVal1 = TensorValue(data, shape1, &testDevice);
        auto tensorVal2 = TensorValue(data, shape2, &testDevice);

        // Add
        CHECK_THROWS_AS({ auto t = tensorVal1; t += tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t += tensorVal1; }, std::invalid_argument);

        // Sub
        CHECK_THROWS_AS({ auto t = tensorVal1; t -= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t -= tensorVal1; }, std::invalid_argument);

        // Mul
        CHECK_THROWS_AS({ auto t = tensorVal1; t *= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t *= tensorVal1; }, std::invalid_argument);

        // Div
        CHECK_THROWS_AS({ auto t = tensorVal1; t /= tensorVal2; }, std::invalid_argument);
        CHECK_THROWS_AS({ auto t = tensorVal2; t /= tensorVal1; }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Data Type Conversion")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f64Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }
}


TEST_CASE("TensorValue - Data Type Conversion wit .to()")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = TensorValue(f64Data, shape, &testDevice).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f64Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<double>()[0] == 1.0);
        CHECK(tv1.data<double>()[1] == 2.0);
        CHECK(tv1.data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.size() == f32Data.size());
        CHECK(tv1.data<float>()[0] == 1.0f);
        CHECK(tv1.data<float>()[1] == 2.0f);
        CHECK(tv1.data<float>()[2] == 3.0f);
    }
}


TEST_CASE("TensorValue - Data Type Promotion")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add

    SUBCASE("Add - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0f);
        CHECK(tv.data<float>()[1] == 6.0f);
        CHECK(tv.data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    // Sub

    SUBCASE("Sub - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    // Mul

    SUBCASE("Mul - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0f);
        CHECK(tv.data<float>()[1] ==  8.0f);
        CHECK(tv.data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    // Div

    SUBCASE("Div - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  2.0);
        CHECK(tv.data<double>()[2] ==  2.0);
    }

    // Pow

    SUBCASE("Pow - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==   2.0f);
        CHECK(tv.data<float>()[1] ==  16.0f);
        CHECK(tv.data<float>()[2] == 216.0f);
    }

    SUBCASE("Pow - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==   2.0);
        CHECK(tv.data<double>()[1] ==  16.0);
        CHECK(tv.data<double>()[2] == 216.0);
    }

    // Matmul

    SUBCASE("Matmul - F64 and F64")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F32 and F32")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == 1);
        CHECK(tv.data<float>()[0] == 28.0f);
    }

    SUBCASE("Matmul - F32 and F64")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat64);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F64 and F32")
    {
        auto tv1 = TensorValue(f32Data, {1,3}, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, {3,1}, &testDevice, DataType::kFloat32);
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == 1);
        CHECK(tv.data<double>()[0] == 28.0);
    }
}


TEST_CASE("TensorValue - Data Type Promotion (In-Place")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add - In-place

    SUBCASE("Add - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0f);
        CHECK(tv.data<float>()[1] == 6.0f);
        CHECK(tv.data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 3.0);
        CHECK(tv.data<float>()[1] == 6.0);
        CHECK(tv.data<float>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv += tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 3.0);
        CHECK(tv.data<double>()[1] == 6.0);
        CHECK(tv.data<double>()[2] == 9.0);
    }

    // Sub - In-place

    SUBCASE("Sub - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 1.0);
        CHECK(tv.data<float>()[1] == 2.0);
        CHECK(tv.data<float>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv -= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 1.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 3.0);
    }

    // Mul - In-place

    SUBCASE("Mul - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0f);
        CHECK(tv.data<float>()[1] ==  8.0f);
        CHECK(tv.data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] ==  2.0);
        CHECK(tv.data<float>()[1] ==  8.0);
        CHECK(tv.data<float>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv *= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  8.0);
        CHECK(tv.data<double>()[2] == 18.0);
    }

    // Div - In-place

    SUBCASE("Div - F64 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] == 2.0);
        CHECK(tv.data<double>()[1] == 2.0);
        CHECK(tv.data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0f);
        CHECK(tv.data<float>()[1] == 2.0f);
        CHECK(tv.data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat32);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat64);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<float>()[0] == 2.0);
        CHECK(tv.data<float>()[1] == 2.0);
        CHECK(tv.data<float>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32 (In-place)")
    {
        auto tv1 = TensorValue(f32Data, shape, &testDevice, DataType::kFloat64);
        auto tv2 = TensorValue(f64Data, shape, &testDevice, DataType::kFloat32);
        auto tv = tv1;
        tv /= tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.size() == f32Data.size());
        CHECK(tv.data<double>()[0] ==  2.0);
        CHECK(tv.data<double>()[1] ==  2.0);
        CHECK(tv.data<double>()[2] ==  2.0);
    }
}


TEST_CASE("TensorValue - promoteToMinFloat")
{
    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto dtype = static_cast<DataType>(i);
        auto promotedDType = promoteDataTypeToFloat(dtype);
        // TensorValue must promote its data type to Float32 if the type is an integer type, otherwise it returns the
        // same float data type.
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).sqrt().dataType());
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).sin().dataType());
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).cos().dataType());
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).tanh().dataType());
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).log().dataType());
        CHECK(promotedDType == TensorValue({1.0f, 2.0f}, Shape{2}, &testDevice).to(dtype).exp().dataType());
        // Note: Results are consistent with those from PyTorch.
    }
}


TEST_CASE("TensorValue - TensorValue OP Scalar - Data Type")
{
    float scalar = 2.0f;
    Shape shape{4};
    std::initializer_list<float> testData{1, 2, 3, 4};

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto testDType = static_cast<DataType>(i);
        auto x = TensorValue(testData, shape, &testDevice, testDType);
        auto expectedType = promoteDataTypeToFloat(testDType);

        SUBCASE("Add")
        {
            auto y = x + scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({3.0, 4.0, 5.0, 6.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Sub")
        {
            auto y = x - scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({-1.0, 0.0, 1.0, 2.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Mul")
        {
            auto y = x * scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({2.0, 4.0, 6.0, 8.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Div")
        {
            auto y = x / scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({0.5, 1.0, 1.5, 2.0}, shape, &testDevice, expectedType));
        }
    }
}


TEST_CASE("TensorValue - Scalar OP TensorValue - Data Type")
{
    float scalar = 2.0f;
    Shape shape{4};
    std::initializer_list<float> testData{1, 2, 3, 4};

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto testDType = static_cast<DataType>(i);
        auto x = TensorValue(testData, shape, &testDevice, testDType);
        auto expectedType = promoteDataTypeToFloat(testDType);

        SUBCASE("Add")
        {
            auto y = scalar + x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({3.0, 4.0, 5.0, 6.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Sub")
        {
            auto y = scalar - x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({1.0, 0.0, -1.0, -2.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Mul")
        {
            auto y = scalar * x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({2.0, 4.0, 6.0, 8.0}, shape, &testDevice, expectedType));
        }

        SUBCASE("Div")
        {
            auto y = scalar / x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, TensorValue({2.0, 1.0, 0.666667, 0.5}, shape, &testDevice, expectedType));
        }
    }
}


TEST_CASE("TensorValue - Compound Assignment OP Scalar - Data Type")
{
    Shape shape{4};
    std::initializer_list<float> testData{1, 2, 3, 4};

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto testDType = static_cast<DataType>(i);

        SUBCASE("AddAssign")
        {
            auto x = TensorValue(testData, shape, &testDevice, testDType);
            x += 2.0f;
            CHECK(x.dataType() == testDType);
        }

        SUBCASE("SubAssign")
        {
            auto x = TensorValue(testData, shape, &testDevice, testDType);
            x -= 2.0f;
            CHECK(x.dataType() == testDType);
        }

        SUBCASE("MulAssign")
        {
            auto x = TensorValue(testData, shape, &testDevice, testDType);
            x *= 2.0f;
            CHECK(x.dataType() == testDType);
        }

        SUBCASE("DivAssign")
        {
            auto x = TensorValue(testData, shape, &testDevice, testDType);
            x /= 2.0f;
            CHECK(x.dataType() == testDType);
        }
    }
}


TEST_CASE("TensorValue - Squeeze")
{
    std::initializer_list<float> data = {1.0, 2.0, 3.0, 4.0};
    auto t = TensorValue(data, {1, 2, 1, 2, 1}, &testDevice);

    SUBCASE("dim 0")
    {
        auto s = t.squeeze(0);
        CHECK(s.shape() == Shape{2,1,2,1});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 1")
    {
        auto s = t.squeeze(1);
        CHECK(s.shape() == Shape{1,2,1,2,1});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 2")
    {
        auto s = t.squeeze(2);
        CHECK(s.shape() == Shape{1,2,2,1});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 3")
    {
        auto s = t.squeeze(3);
        CHECK(s.shape() == Shape{1,2,1,2,1});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 4")
    {
        auto s = t.squeeze(4);
        CHECK(s.shape() == Shape{1,2,1,2});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 4")
    {
        auto s = t.squeeze(0);
        s = s.squeeze(0);
        s = s.squeeze(1);
        s = s.squeeze(1);
        s = s.squeeze(2);
        CHECK(s.shape() == Shape{2,2});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 5")
    {
        CHECK_THROWS_AS(t.squeeze(5), std::invalid_argument);
        CheckVectorApproxValues(t, tensor(data, t.shape()).value());
    }
}


TEST_CASE("TensorValue - Unsqueeze")
{
    std::initializer_list<float> data = {1.0, 2.0, 3.0, 4.0};
    auto t = TensorValue(data, {2,2}, &testDevice);

    SUBCASE("dim 0")
    {
        auto s = t.unsqueeze(0);
        CHECK(s.shape() == Shape{1,2,2});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 1")
    {
        auto s = t.unsqueeze(1);
        CHECK(s.shape() == Shape{2,1,2});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 2")
    {
        auto s = t.unsqueeze(2);
        CHECK(s.shape() == Shape{2,2,1});
        CHECK(s.size() == 4);
        CheckVectorApproxValues(s, tensor(data, s.shape()).value());
    }

    SUBCASE("dim 3")
    {
        CHECK_THROWS_AS(t.unsqueeze(3), std::invalid_argument);
        CheckVectorApproxValues(t, tensor(data, t.shape()).value());
    }
}


TEST_CASE("TensorValue - argmaxIndices")
{
    SUBCASE("Scalar")
    {
        auto a = TensorValue(1.0, {}, &testDevice);
        auto amax = a.argmaxIndices();
        CHECK(amax.shape() == Shape{});
        CheckVectorApproxValues(amax, Tensor(1.0, amax.shape(), { .m_dtype=DataType::kInt32 }).value());
    }

    SUBCASE("1 Tensor")
    {
        auto a = TensorValue({1.0}, {1}, &testDevice);
        auto amax = a.argmaxIndices();
        CHECK(amax.shape() == Shape{1});
        CheckVectorApproxValues(amax, tensor({1.0}, amax.shape(), { .m_dtype=DataType::kInt32 }).value());
    }

    SUBCASE("2x2 Tensor - first max")
    {
        auto a = TensorValue({4.0, 2.0, 3.0, 1.0}, {2,2}, &testDevice);
        auto amax = a.argmaxIndices();
        CHECK(amax.shape() == Shape{2,2});
        CheckVectorApproxValues(amax, tensor({1.0, 0.0, 0.0, 0.0}, amax.shape(), { .m_dtype=DataType::kInt32 }).value());
    }

    SUBCASE("2x2 Tensor - last max")
    {
        auto a = TensorValue({1.0, 2.0, 3.0, 4.0}, {2,2}, &testDevice);
        auto amax = a.argmaxIndices();
        CHECK(amax.shape() == Shape{2,2});
        CheckVectorApproxValues(amax, tensor({0.0, 0.0, 0.0, 1.0}, amax.shape(), { .m_dtype=DataType::kInt32 }).value());
    }

    SUBCASE("2x2 Tensor - first found max")
    {
        auto a = TensorValue({1.0, 4.0, 4.0, 4.0}, {2,2}, &testDevice);
        auto amax = a.argmaxIndices();
        CHECK(amax.shape() == Shape{2,2});
        CheckVectorApproxValues(amax, tensor({0.0, 1.0, 0.0, 0.0}, amax.shape(), { .m_dtype=DataType::kInt32 }).value());
    }
}


TEST_CASE("TensorValue - Max with dim")
{
    auto t1  = aix::TensorValue({1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0}, aix::Shape{3, 3}, &testDevice);

    SUBCASE("Shape{} - dim=0 keepDim=false")
    {
        auto t  = aix::TensorValue(5.0, aix::Shape{}, &testDevice);     // Scalar Tensor
        t = t.max(0, false);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, TensorValue(5.0, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=false")
    {
        auto t  = aix::TensorValue({5.0}, aix::Shape{1}, &testDevice);     // Scalar Tensor
        t = t.max(0, false);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, TensorValue(5.0, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=true")
    {
        auto t  = aix::TensorValue({5.0}, aix::Shape{1}, &testDevice);     // Scalar Tensor
        t = t.max(0, true);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, TensorValue(5.0, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=false")
    {
        auto t  = aix::TensorValue({5.0}, aix::Shape{1,1}, &testDevice);
        t = t.max(0, false);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, TensorValue({5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=true")
    {
        auto t  = aix::TensorValue({5.0}, aix::Shape{1,1}, &testDevice);
        t = t.max(0, true);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, TensorValue({5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=false")
    {
        auto t = t1.max(0, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, TensorValue({7,8,9}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=true")
    {
        auto t = t1.max(0, true);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, TensorValue({7,8,9}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=false")
    {
        auto t = t1.max(1, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, TensorValue({3,6,9}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=true")
    {
        auto t = t1.max(1, true);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, TensorValue({3,6,9}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dimension out of range")
    {
        CHECK_THROWS_AS({ t1.max(2, false);  }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.max(2, true);   }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.max(-3, false); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.max(-3, true);  }, std::invalid_argument);
    }

}


TEST_CASE("TensorValue - slice")
{
    auto t1  = aix::TensorValue({1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0}, aix::Shape{3, 3}, &testDevice);

    auto t2  = aix::TensorValue({1.0, 2.0,
                                 3.0, 4.0,
                                 5.0, 6.0,
                                 7.0, 8.0}, aix::Shape{2,2,2}, &testDevice);

    // Default parameters

    SUBCASE("Shape{1} - default parameters")
    {
        auto t2  = aix::TensorValue({5.0}, aix::Shape{1}, &testDevice);
        auto t = t2.slice();
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{1,1} - default parameters")
    {
        auto t2  = aix::TensorValue({5.0}, aix::Shape{1,1}, &testDevice);
        auto t = t2.slice();
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{3,3} - default parameters")
    {
        auto t = t1.slice();
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, t1);
    }

    // Dim 0

    SUBCASE("Shape{1,1} - dim=0 start=0 end=1 step=1")
    {
        auto t2  = aix::TensorValue({5.0}, aix::Shape{1,1}, &testDevice);
        auto t = t2.slice(0, 0, 1, 1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=1")
    {
        auto t = t1.slice(0,0,3,1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, t1);
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=2 step=1")
    {
        auto t = t1.slice(0,0,2,1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::TensorValue({1.0, 2.0, 3.0,
                                                     4.0, 5.0, 6.0}, aix::Shape{2, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=-2 start=-3 end=-1 step=1")
    {
        auto t = t1.slice(-2,-3,-1,1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::TensorValue({1.0, 2.0, 3.0,
                                                     4.0, 5.0, 6.0}, aix::Shape{2, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=1 step=1")
    {
        auto t = t1.slice(0,0,1,1);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({1.0, 2.0, 3.0}, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=1")
    {
        auto t = t1.slice(0,1,3,1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 4.0, 5.0, 6.0,
                                                      7.0, 8.0, 9.0 }, aix::Shape{2, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=1")
    {
        auto t = t1.slice(0,2,3,1);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 7.0, 8.0, 9.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=2")
    {
        auto t = t1.slice(0,0,3,2);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 2.0, 3.0,
                                                      7.0, 8.0, 9.0 }, aix::Shape{2, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=3")
    {
        auto t = t1.slice(0,0,3,3);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 2.0, 3.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=0 end=3 step=4")
    {
        auto t = t1.slice(0,0,3,4);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 2.0, 3.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=2")
    {
        auto t = t1.slice(0,1,3,2);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 4.0, 5.0, 6.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=1 end=3 step=3")
    {
        auto t = t1.slice(0,1,3,2);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 4.0, 5.0, 6.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=2")
    {
        auto t = t1.slice(0,2,3,2);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 7.0, 8.0, 9.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=3")
    {
        auto t = t1.slice(0,2,3,3);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 7.0, 8.0, 9.0 }, aix::Shape{1, 3}, &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=0 end=2 step=1")
    {
        auto t = t2.slice(0,0,2,1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=0 end=1 step=1")
    {
        auto t = t2.slice(0,0,1,1);
        CHECK(t.shape() == Shape{1,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 2.0,
                                                      3.0, 4.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=1 end=2 step=1")
    {
        auto t = t2.slice(0,1,2,1);
        CHECK(t.shape() == Shape{1,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 5.0, 6.0,
                                                      7.0, 8.0 }, aix::Shape{1, 2, 2}, &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=0 start=1 end=2 step=2")
    {
        auto t = t2.slice(0,1,2,2);
        CHECK(t.shape() == Shape{1,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 5.0, 6.0,
                                                      7.0, 8.0 }, aix::Shape{1, 2, 2}, &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=-3 start=-2 end=2 step=2")
    {
        auto t = t2.slice(-3,-1,2,2);
        CHECK(t.shape() == Shape{1,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 5.0, 6.0,
                                                      7.0, 8.0 }, aix::Shape{1, 2, 2}, &testDevice));
    }

    // Dim 1

    SUBCASE("Shape{1,1} - dim=1 start=0 end=1 step=1")
    {
        auto t2  = aix::TensorValue({5.0}, aix::Shape{1,1}, &testDevice);
        auto t = t2.slice(1, 0, 1, 1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=1")
    {
        auto t = t1.slice(1,0,3,1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, t1);
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=2 step=1")
    {
        auto t = t1.slice(1,0,2,1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::TensorValue({1.0, 2.0,
                                                     4.0, 5.0,
                                                     7.0, 8.0,}, aix::Shape{3,2}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=1 step=1")
    {
        auto t = t1.slice(1,0,1,1);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({1.0,
                                                     4.0,
                                                     7.0}, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=1")
    {
        auto t = t1.slice(1,1,3,1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0, 3.0,
                                                      5.0, 6.0,
                                                      8.0, 9.0 }, aix::Shape{3, 2}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=2 end=3 step=1")
    {
        auto t = t1.slice(1,2,3,1);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0,
                                                      6.0,
                                                      9.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=2")
    {
        auto t = t1.slice(1,0,3,2);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 3.0,
                                                      4.0, 6.0,
                                                      7.0, 9.0 }, aix::Shape{3, 2}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=3")
    {
        auto t = t1.slice(1,0,3,3);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,
                                                      4.0,
                                                      7.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=0 end=3 step=4")
    {
        auto t = t1.slice(1,0,3,4);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,
                                                      4.0,
                                                      7.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=2")
    {
        auto t = t1.slice(1,1,3,2);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0,
                                                      5.0,
                                                      8.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=1 end=3 step=3")
    {
        auto t = t1.slice(1,1,3,3);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0,
                                                      5.0,
                                                      8.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 start=2 end=3 step=2")
    {
        auto t = t1.slice(1,2,3,2);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0,
                                                      6.0,
                                                      9.0 }, aix::Shape{3, 1}, &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0 start=2 end=3 step=3")
    {
        auto t = t1.slice(1,2,3,3);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0,
                                                      6.0,
                                                      9.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=0 end=2 step=1")
    {
        auto t = t2.slice(1,0,2,1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=0 end=1 step=1")
    {
        auto t = t2.slice(1,0,1,1);
        CHECK(t.shape() == Shape{2,1,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0, 2.0,
                                                      5.0, 6.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=1 end=2 step=1")
    {
        auto t = t2.slice(1,1,2,1);
        CHECK(t.shape() == Shape{2,1,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0, 4.0,
                                                      7.0, 8.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=1 start=1 end=2 step=2")
    {
        auto t = t2.slice(1,1,2,2);
        CHECK(t.shape() == Shape{2,1,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0, 4.0,
                                                      7.0, 8.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=-2 start=1 end=2 step=2")
    {
        auto t = t2.slice(-2,1,2,2);
        CHECK(t.shape() == Shape{2,1,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 3.0, 4.0,
                                                      7.0, 8.0 }, t.shape(), &testDevice));
    }

    // Dim 2

    SUBCASE("Shape{2,2,2} - dim=2 start=0 end=2 step=1")
    {
        auto t = t2.slice(2,0,2,1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t2);
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=0 end=1 step=1")
    {
        auto t = t2.slice(2,0,1,1);
        CHECK(t.shape() == Shape{2,2,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,
                                                      3.0,
                                                      5.0,
                                                      7.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=1 end=2 step=1")
    {
        auto t = t2.slice(2,1,2,1);
        CHECK(t.shape() == Shape{2,2,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0,
                                                      4.0,
                                                      6.0,
                                                      8.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=2 start=1 end=2 step=2")
    {
        auto t = t2.slice(2,1,2,2);
        CHECK(t.shape() == Shape{2,2,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0,
                                                      4.0,
                                                      6.0,
                                                      8.0 }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=-1 start=1 end=2 step=2")
    {
        auto t = t2.slice(-1,1,2,2);
        CHECK(t.shape() == Shape{2,2,1});
        CheckVectorApproxValues(t, aix::TensorValue({ 2.0,
                                                      4.0,
                                                      6.0,
                                                      8.0 }, t.shape(), &testDevice));
    }

    SUBCASE("invalid parameters")
    {
        CHECK_THROWS_AS({ aix::TensorValue(5.0, aix::Shape{}, &testDevice).slice(); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.slice(0, 0, 1, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.slice(0, 0, 0, 1); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Split")
{
    auto ts  = aix::TensorValue(5.0, &testDevice);
    auto t1  = aix::TensorValue({5.0}, Shape{1}, &testDevice);
    auto t11 = aix::TensorValue({5.0}, Shape{1,1}, &testDevice);
    auto t33 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0,
                                  7.0, 8.0, 9.0 }, Shape{3,3}, &testDevice);

    SUBCASE("Shape{3,3} - splitSize=1, dim=0")
    {
        auto tlist = t33.split(1, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{1,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CHECK(tlist[2].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 4.0, 5.0, 6.0 }, tlist[1].shape()).value());
        CheckVectorApproxValues(tlist[2], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[2].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=2, dim=0")
    {
        auto tlist = t33.split(2, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 2);
        CHECK(tlist[0].shape() == Shape{2,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[1].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=3, dim=0")
    {
        auto tlist = t33.split(3, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 1);
        CHECK(tlist[0].shape() == Shape{3,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0,
                                                        7.0, 8.0, 9.0 }, tlist[0].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=-2")
    {
        auto tlist = t33.split(1, -2);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{1,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CHECK(tlist[2].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 4.0, 5.0, 6.0 }, tlist[1].shape()).value());
        CheckVectorApproxValues(tlist[2], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[2].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=1")
    {
        auto tlist = t33.split(1, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{3,1});
        CHECK(tlist[1].shape() == Shape{3,1});
        CHECK(tlist[2].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 4.0, 7.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 2.0, 5.0, 8.0 }, tlist[1].shape()).value());
        CheckVectorApproxValues(tlist[2], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[2].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=2, dim=1")
    {
        auto tlist = t33.split(2, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 2);
        CHECK(tlist[0].shape() == Shape{3,2});
        CHECK(tlist[1].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0,
                                                        4.0, 5.0,
                                                        7.0, 8.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[1].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=3, dim=1")
    {
        auto tlist = t33.split(3, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 1);
        CHECK(tlist[0].shape() == Shape{3,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0,
                                                        7.0, 8.0, 9.0 }, tlist[0].shape()).value());
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=-1")
    {
        auto tlist = t33.split(1, -1);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{3,1});
        CHECK(tlist[1].shape() == Shape{3,1});
        CHECK(tlist[2].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 4.0, 7.0 }, tlist[0].shape()).value());
        CheckVectorApproxValues(tlist[1], aix::tensor({ 2.0, 5.0, 8.0 }, tlist[1].shape()).value());
        CheckVectorApproxValues(tlist[2], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[2].shape()).value());
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts.split(0,0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.split(0,0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.split(-1,0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.split(1,2); }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.split(1,-3); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Tril")
{
    auto ts  = aix::TensorValue(5.0, &testDevice);
    auto t1  = aix::TensorValue({5.0}, Shape{1}, &testDevice);
    auto t11 = aix::TensorValue({5.0}, Shape{1,1}, &testDevice);
    auto t23 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0 }, Shape{2,3}, &testDevice);
    auto t32 = aix::TensorValue({ 1.0, 2.0,
                                  3.0, 4.0,
                                  5.0, 6.0 }, Shape{3,2}, &testDevice);
    auto t33 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0,
                                  7.0, 8.0, 9.0 }, Shape{3,3}, &testDevice);
    auto t222 = aix::TensorValue({ 1.0, 2.0,
                                   3.0, 4.0,
                                   5.0, 6.0,
                                   7.0, 8.0 }, Shape{2,2,2}, &testDevice);

    SUBCASE("Shape{1,1} - diagonal=default")
    {
        auto t = t11.tril();
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=0")
    {
        auto t = t11.tril(0);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=1")
    {
        auto t = t11.tril(1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=-1")
    {
        auto t = t11.tril(-1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 0.0 }, t.shape()).value());
    }

    // ----

    SUBCASE("Shape{2,3} - diagonal=0")
    {
        auto t = t23.tril(0);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 0.0, 0.0,
                                                 4.0, 5.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=1")
    {
        auto t = t23.tril(1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 0.0,
                                                 4.0, 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=2")
    {
        auto t = t23.tril(2);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-1")
    {
        auto t = t23.tril(-1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 4.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-2")
    {
        auto t = t23.tril(-2);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-3")
    {
        auto t = t23.tril(-3);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    // ----

    SUBCASE("Shape{3,2} - diagonal=0")
    {
        auto t = t32.tril(0);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 0.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=1")
    {
        auto t = t32.tril(1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=2")
    {
        auto t = t32.tril(2);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-1")
    {
        auto t = t32.tril(-1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 3.0, 0.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-2")
    {
        auto t = t32.tril(-2);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 5.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-3")
    {
        auto t = t32.tril(-3);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    // ---

    SUBCASE("Shape{3,3} - diagonal=0")
    {
        auto t = t33.tril(0);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 0.0, 0.0,
                                                 4.0, 5.0, 0.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=1")
    {
        auto t = t33.tril(1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 0.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=2")
    {
        auto t = t33.tril(2);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=3")
    {
        auto t = t33.tril(3);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-1")
    {
        auto t = t33.tril(-1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 4.0, 0.0, 0.0,
                                                 7.0, 8.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-2")
    {
        auto t = t33.tril(-2);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 7.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-3")
    {
        auto t = t33.tril(-3);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-4")
    {
        auto t = t33.tril(-4);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    // ---

    SUBCASE("Shape{2,2,2} - diagonal=0")
    {
        auto t = t222.tril(0);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 0.0,
                                                 3.0, 4.0,
                                                 5.0, 0.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=1")
    {
        auto t = t222.tril(1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=2")
    {
        auto t = t222.tril(1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-1")
    {
        auto t = t222.tril(-1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 3.0, 0.0,
                                                 0.0, 0.0,
                                                 7.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-2")
    {
        auto t = t222.tril(-2);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-3")
    {
        auto t = t222.tril(-3);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts.tril(); }, std::invalid_argument);
        CHECK_THROWS_AS({ ts.tril(0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.tril(); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.tril(0); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Triu")
{
    auto ts  = aix::TensorValue(5.0, &testDevice);
    auto t1  = aix::TensorValue({5.0}, Shape{1}, &testDevice);
    auto t11 = aix::TensorValue({5.0}, Shape{1,1}, &testDevice);
    auto t23 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0 }, Shape{2,3}, &testDevice);
    auto t32 = aix::TensorValue({ 1.0, 2.0,
                                  3.0, 4.0,
                                  5.0, 6.0 }, Shape{3,2}, &testDevice);
    auto t33 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0,
                                  7.0, 8.0, 9.0 }, Shape{3,3}, &testDevice);
    auto t222 = aix::TensorValue({ 1.0, 2.0,
                                   3.0, 4.0,
                                   5.0, 6.0,
                                   7.0, 8.0 }, Shape{2,2,2}, &testDevice);

    SUBCASE("Shape{1,1} - diagonal=default")
    {
        auto t = t11.triu();
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=0")
    {
        auto t = t11.triu(0);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=1")
    {
        auto t = t11.triu(1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{1,1} - diagonal=-1")
    {
        auto t = t11.triu(-1);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({ 5.0 }, t.shape()).value());
    }

    // ----

    SUBCASE("Shape{2,3} - diagonal=0")
    {
        auto t = t23.triu(0);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 0.0, 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=1")
    {
        auto t = t23.triu(1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 2.0, 3.0,
                                                 0.0, 0.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=2")
    {
        auto t = t23.triu(2);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 3.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-1")
    {
        auto t = t23.triu(-1);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-2")
    {
        auto t = t23.triu(-2);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,3} - diagonal=-3")
    {
        auto t = t23.triu(-3);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0 }, t.shape()).value());
    }

    // ----

    SUBCASE("Shape{3,2} - diagonal=0")
    {
        auto t = t32.triu(0);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 0.0, 4.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=1")
    {
        auto t = t32.triu(1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 2.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=2")
    {
        auto t = t32.triu(2);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-1")
    {
        auto t = t32.triu(-1);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 0.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-2")
    {
        auto t = t32.triu(-2);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,2} - diagonal=-3")
    {
        auto t = t32.triu(-3);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0 }, t.shape()).value());
    }

    // ---

    SUBCASE("Shape{3,3} - diagonal=0")
    {
        auto t = t33.triu(0);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 0.0, 5.0, 6.0,
                                                 0.0, 0.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=1")
    {
        auto t = t33.triu(1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 2.0, 3.0,
                                                 0.0, 0.0, 6.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=2")
    {
        auto t = t33.triu(2);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 3.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=3")
    {
        auto t = t33.triu(3);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-1")
    {
        auto t = t33.triu(-1);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 0.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-2")
    {
        auto t = t33.triu(-2);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-3")
    {
        auto t = t33.triu(-3);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    SUBCASE("Shape{3,3} - diagonal=-4")
    {
        auto t = t33.triu(-4);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0 }, t.shape()).value());
    }

    // ---

    SUBCASE("Shape{2,2,2} - diagonal=0")
    {
        auto t = t222.triu(0);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 0.0, 4.0,
                                                 5.0, 6.0,
                                                 0.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=1")
    {
        auto t = t222.triu(1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 2.0,
                                                 0.0, 0.0,
                                                 0.0, 6.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=2")
    {
        auto t = t222.triu(2);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0,
                                                 0.0, 0.0 }, t.shape()).value());
    }


    SUBCASE("Shape{2,2,2} - diagonal=-1")
    {
        auto t = t222.triu(-1);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-2")
    {
        auto t = t222.triu(-2);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Shape{2,2,2} - diagonal=-3")
    {
        auto t = t222.triu(-3);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 3.0, 4.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()).value());
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts.triu();  }, std::invalid_argument);
        CHECK_THROWS_AS({ ts.triu(0); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.triu();  }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.triu(0); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - Cat")
{
    auto ts  = aix::TensorValue(5.0, &testDevice);
    auto t1  = aix::TensorValue({5.0}, Shape{1}, &testDevice);
    auto t11 = aix::TensorValue({5.0}, Shape{1,1}, &testDevice);
    auto t33 = aix::TensorValue({ 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0,
                                  7.0, 8.0, 9.0 }, Shape{3,3}, &testDevice);
    auto t222 = aix::TensorValue({ 1.0, 2.0,
                                   3.0, 4.0,
                                   5.0, 6.0,
                                   7.0, 8.0 }, Shape{2,2,2}, &testDevice);

    SUBCASE("Shape{1} - dim=0")
    {
        auto t = aix::TensorValue::cat({t1}, 0);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::TensorValue({5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1} - dim=0 - 2x")
    {
        auto t = aix::TensorValue::cat({t1,t1}, 0);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::TensorValue({5.0, 5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1} - dim=-1 - 2x")
    {
        auto t = aix::TensorValue::cat({t1,t1}, -1);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::TensorValue({5.0, 5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=0")
    {
        auto t = aix::TensorValue::cat({t11}, 0);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::TensorValue({5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=0 - 2x")
    {
        auto t = aix::TensorValue::cat({t11, t11}, 0);
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(t, aix::TensorValue({5.0,5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=-2 - 2x")
    {
        auto t = aix::TensorValue::cat({t11, t11}, -2);
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(t, aix::TensorValue({5.0,5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=1 - 2x")
    {
        auto t = aix::TensorValue::cat({t11, t11}, 1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::TensorValue({5.0,5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::TensorValue::cat({t11, t11}, -1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::TensorValue({5.0,5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::TensorValue::cat({t11, t11}, -1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::TensorValue({5.0,5.0}, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=0")
    {
        auto t = aix::TensorValue::cat({t33}, 0);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,3.0,
                                                      4.0,5.0,6.0,
                                                      7.0,8.0,9.0,}, t.shape(), &testDevice));
        }

    SUBCASE("Shape{3,3} - dim=0 - 2x")
    {
        auto t = aix::TensorValue::cat({t33, t33}, 0);
        CHECK(t.shape() == Shape{6,3});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,3.0,
                                                      4.0,5.0,6.0,
                                                      7.0,8.0,9.0,
                                                      1.0,2.0,3.0,
                                                      4.0,5.0,6.0,
                                                      7.0,8.0,9.0, }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{3,3} - dim=1 - 2x")
    {
        auto t = aix::TensorValue::cat({t33, t33}, 1);
        CHECK(t.shape() == Shape{3,6});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,3.0, 1.0,2.0,3.0,
                                                      4.0,5.0,6.0, 4.0,5.0,6.0,
                                                      7.0,8.0,9.0, 7.0,8.0,9.0, }, t.shape(), &testDevice));
        }

    SUBCASE("Shape{2,2,2} - dim=0 - 2x")
    {
        auto t = aix::TensorValue::cat({t222,t222}, 0);
        CHECK(t.shape() == Shape{4,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,
                                                      3.0,4.0,
                                                      5.0,6.0,
                                                      7.0,8.0,
                                                      1.0,2.0,
                                                      3.0,4.0,
                                                      5.0,6.0,
                                                      7.0,8.0, }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=-3 - 2x")
    {
        auto t = aix::TensorValue::cat({t222,t222}, -3);
        CHECK(t.shape() == Shape{4,2,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,
                                                      3.0,4.0,
                                                      5.0,6.0,
                                                      7.0,8.0,
                                                      1.0,2.0,
                                                      3.0,4.0,
                                                      5.0,6.0,
                                                      7.0,8.0, }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=1 - 2x")
    {
        auto t = aix::TensorValue::cat({t222,t222}, 1);
        CHECK(t.shape() == Shape{2,4,2});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,
                                                      3.0,4.0,
                                                      1.0,2.0,
                                                      3.0,4.0,
                                                      5.0,6.0,
                                                      7.0,8.0,
                                                      5.0,6.0,
                                                      7.0,8.0, }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=2 - 2x")
    {
        auto t = aix::TensorValue::cat({t222,t222}, 2);
        CHECK(t.shape() == Shape{2,2,4});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,1.0,2.0,
                                                      3.0,4.0,3.0,4.0,
                                                      5.0,6.0,5.0,6.0,
                                                      7.0,8.0,7.0,8.0, }, t.shape(), &testDevice));
    }

    SUBCASE("Shape{2,2,2} - dim=-1 - 2x")
    {
        auto t = aix::TensorValue::cat({t222,t222}, -1);
        CHECK(t.shape() == Shape{2,2,4});
        CheckVectorApproxValues(t, aix::TensorValue({ 1.0,2.0,1.0,2.0,
                                                      3.0,4.0,3.0,4.0,
                                                      5.0,6.0,5.0,6.0,
                                                      7.0,8.0,7.0,8.0, }, t.shape(), &testDevice));
    }

    // Type promotion.

    SUBCASE("{Int32, Int32} -> Int32")
    {
        auto t = aix::TensorValue::cat({t33.to(aix::DataType::kInt32), t33.to(aix::DataType::kInt32)}, 0);
        CHECK(t.dataType() == aix::DataType::kInt32);
    }

    SUBCASE("{Int32, Float16} -> Float16")
    {
        auto t = aix::TensorValue::cat({t33.to(aix::DataType::kInt32), t33.to(aix::DataType::kInt32)}, 0);
        CHECK(t.dataType() == aix::DataType::kInt32);
    }

    SUBCASE("{Float16, Float32} -> Float32")
    {
        auto t = aix::TensorValue::cat({t33.to(aix::DataType::kFloat16), t33.to(aix::DataType::kFloat32)}, 0);
        CHECK(t.dataType() == aix::DataType::kFloat32);
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ aix::TensorValue::cat({ts}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({ts,ts}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({t1,t1}, 1); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({t1,t1}, -2); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({t11,t11}, -3); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({t1,t33}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::TensorValue::cat({t33,t11}, 1); }, std::invalid_argument);
    }
}


TEST_CASE("TensorValue - permute()")
{
    auto ts = TensorValue(5.0,{}, &testDevice);
    auto t12 = TensorValue({1.0,2.0}, {1,2}, &testDevice);
    auto t32 = TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice);
    auto t324 = aix::arange(1.0, 25.0, 1.0, aix::device(&testDevice)).reshape({3,2,4}).value();

    SUBCASE("Fail tests")
    {
        TensorValue s = TensorValue(1.0, {}, &testDevice);     // Scalar tensor.
        TensorValue t = TensorValue({1.0,2.0,3.0,4.0}, {2,1,2}, &testDevice);
        DOCTEST_CHECK_THROWS_AS(s.permute({0}),       std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,1,2,3}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,1}),     std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,1,3}),   std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,1,1}),   std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,0,1}),   std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({0,1,-4}),  std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({-1,0,-1}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t.permute({-2,0,-2}), std::invalid_argument);
    }

    SUBCASE("s{} p{}")
    {
        TensorValue t = TensorValue(5.0, {}, &testDevice);
        auto result = t.permute({});
        CHECK(result.shape() == Shape{});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1} p{0}")
    {
        TensorValue t = TensorValue(5.0, {1}, &testDevice);
        auto result = t.permute({0});
        CHECK(result.shape() == Shape{1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1} p{-1}")
    {
        TensorValue t = TensorValue(5.0, {1}, &testDevice);
        auto result = t.permute({-1});
        CHECK(result.shape() == Shape{1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{0,1}")
    {
        TensorValue t = TensorValue(5.0, {1,1}, &testDevice);
        auto result = t.permute({0,1});
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{-2,-1}")
    {
        TensorValue t = TensorValue(5.0, {1,1}, &testDevice);
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{-1,-2}")
    {
        TensorValue t = TensorValue(5.0, {1,1}, &testDevice);
        auto result = t.permute({-1,-2});
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,2} p{0,1}")
    {
        auto t = t12;
        auto result = t.permute({0,1});
        CHECK(result.shape() == Shape{1,2});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,2} p{-2,-1}")
    {
        auto t = t12;
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{1,2});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,2} p{-1,-2}")
    {
        auto t = t12;
        auto result = t.permute({-1,-2});
        CHECK(result.shape() == Shape{2,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{2,1} p{0,1}")
    {
        auto t = t12.reshape({2,1});
        auto result = t.permute({0,1});
        CHECK(result.shape() == Shape{2,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{2,1} p{-2,-1}")
    {
        auto t = t12.reshape({2,1});
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{2,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{2,1} p{-1,-2}")
    {
        auto t = t12.reshape({2,1});
        auto result = t.permute({-1,-2});
        CHECK(result.shape() == Shape{1,2});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{3,2} p{0,1}")
    {
        auto t = t32;
        auto result = t.permute({0,1});
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("s{3,2} p{1,0}")
    {
        auto t = t32;
        auto result = t.permute({1,0});
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("s{3,2} p{-2,-1}")
    {
        auto t = t32;
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, TensorValue({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}, &testDevice));
    }

    SUBCASE("s{3,2} p{-1,-2}")
    {
        auto t = t32;
        auto result = t.permute({-1,-2});
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, TensorValue({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{0,1,2}")
    {
        auto t = t324;
        auto result = t.permute({0,1,2});
        CHECK(result.shape() == Shape{3,2,4});
        CheckVectorApproxValues(result, t324);
    }

    SUBCASE("s{3,2,4} p{0,2,1}")
    {
        auto t = t324;
        auto result = t.permute({0,2,1});
        CHECK(result.shape() == Shape{3,4,2});
        CheckVectorApproxValues(result, TensorValue({ 1.0 , 5.0,
                                                      2.0 , 6.0,
                                                      3.0 , 7.0,
                                                      4.0 , 8.0,
                                                      9.0 ,13.0,
                                                     10.0, 14.0,
                                                     11.0, 15.0,
                                                     12.0, 16.0,
                                                     17.0, 21.0,
                                                     18.0, 22.0,
                                                     19.0, 23.0,
                                                     20.0, 24.0}, {3,4,2}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{1,0,2}")
    {
        auto t = t324;
        auto result = t.permute({1,0,2});
        CHECK(result.shape() == Shape{2,3,4});
        CheckVectorApproxValues(result, TensorValue({  1.0,  2.0,  3.0,  4.0,
                                                       9.0, 10.0, 11.0, 12.0,
                                                      17.0, 18.0, 19.0, 20.0,
                                                       5.0,  6.0,  7.0,  8.0,
                                                      13.0, 14.0, 15.0, 16.0,
                                                      21.0, 22.0, 23.0, 24.0}, {2,3,4}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{1,2,0}")
    {
        auto t = t324;
        auto result = t.permute({1,2,0});
        CHECK(result.shape() == Shape{2,4,3});
        CheckVectorApproxValues(result, TensorValue({ 1.0,  9.0, 17.0,
                                                      2.0, 10.0, 18.0,
                                                      3.0, 11.0, 19.0,
                                                      4.0, 12.0, 20.0,
                                                      5.0, 13.0, 21.0,
                                                      6.0, 14.0, 22.0,
                                                      7.0, 15.0, 23.0,
                                                      8.0, 16.0, 24.0}, {2,4,3}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{2,0,1}")
    {
        auto t = t324;
        auto result = t.permute({2,0,1});
        CHECK(result.shape() == Shape{4,3,2});
        CheckVectorApproxValues(result, TensorValue({  1.0,  5.0,
                                                       9.0, 13.0,
                                                      17.0, 21.0,
                                                       2.0,  6.0,
                                                      10.0, 14.0,
                                                      18.0, 22.0,
                                                       3.0,  7.0,
                                                      11.0, 15.0,
                                                      19.0, 23.0,
                                                       4.0,  8.0,
                                                      12.0, 16.0,
                                                      20.0, 24.0}, {4,3,2}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{2,1,0}")
    {
        auto t = t324;
        auto result = t.permute({2,1,0});
        CHECK(result.shape() == Shape{4,2,3});
        CheckVectorApproxValues(result, TensorValue({ 1.0,  9.0, 17.0,
                                                      5.0, 13.0, 21.0,
                                                      2.0, 10.0, 18.0,
                                                      6.0, 14.0, 22.0,
                                                      3.0, 11.0, 19.0,
                                                      7.0, 15.0, 23.0,
                                                      4.0, 12.0, 20.0,
                                                      8.0, 16.0, 24.0}, {4,2,3}, &testDevice));
    }

    SUBCASE("s{3,2,4} p{-1,-2,-3}")
    {
        auto t = t324;
        auto result = t.permute({-1,-2,-3});
        CHECK(result.shape() == Shape{4,2,3});
        CheckVectorApproxValues(result, TensorValue({ 1.0,  9.0, 17.0,
                                                      5.0, 13.0, 21.0,
                                                      2.0, 10.0, 18.0,
                                                      6.0, 14.0, 22.0,
                                                      3.0, 11.0, 19.0,
                                                      7.0, 15.0, 23.0,
                                                      4.0, 12.0, 20.0,
                                                      8.0, 16.0, 24.0}, {4,2,3}, &testDevice));
    }

}
