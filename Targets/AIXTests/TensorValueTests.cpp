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
#include <array>
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
