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
#include <sstream>


using namespace aix;


TEST_CASE("Tensor - name")
{
    auto tensor = aix::tensor({1.0, 2.0}, {2}, { .requireGrad=false });
    CHECK(tensor.name().empty());
    tensor.name("tensor");
    CHECK(tensor.name() == "tensor");
}


TEST_CASE("Tensor - ones")
{
    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3});
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == false);
        CheckVectorApproxValues(t, Tensor(1.0, {2, 3}));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::ones({2, 3}, { .requireGrad=true });
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);
        CheckVectorApproxValues(t, Tensor(1.0, {2, 3}));
    }
}


TEST_CASE("Tensor - zeros")
{
    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::zeros({2, 3});
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == false);
        CheckVectorApproxValues(t, Tensor(0.0, {2, 3}));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zeros({2, 3}, { .requireGrad=true });
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);
        CheckVectorApproxValues(t, Tensor(0.0, {2, 3}));
    }
}


TEST_CASE("Tensor - onesLike")
{
    aix::Tensor src(1.5, {2, 3});

    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::onesLike(src);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == false);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(1.0, src.shape()));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::onesLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(1.0, src.shape()));
    }
}


TEST_CASE("Tensor - zerosLike")
{
    aix::Tensor src(1.5, {2, 3});

    SUBCASE("without requiring gradient")
    {
        aix::Tensor t = aix::zerosLike(src);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == false);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(0.0, src.shape()));
    }

    SUBCASE("requiring gradient")
    {
        aix::Tensor t = aix::zerosLike(src, true);
        CHECK(t.shape() == src.shape());
        CHECK(t.value().size() == src.value().size());
        CHECK(t.isRequireGrad() == true);
        CHECK(t.value().device() == src.value().device());
        CheckVectorApproxValues(t, Tensor(0.0, src.shape()));
    }
}


TEST_CASE("Tensor - matmul()")
{
    SUBCASE("Must fail due to an inner dimension mismatch")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0}, {2,2});
        Tensor B = tensor({1.0,2.0,3.0,4.0}, {1,4});
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("Must fail due to dimension size mismatch")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0}, Shape{4});
        Tensor B = tensor({1.0,2.0,3.0,4.0}, Shape{1,4});
        DOCTEST_CHECK_THROWS_AS(A.matmul(B), std::invalid_argument);
    }

    SUBCASE("A[1x1] B[1x1] multiplication")
    {
        Tensor A = tensor({2.0}, Shape{1,1});
        Tensor B = tensor({3.0}, Shape{1,1});
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, tensor({6.0}, Shape{1,1}));
    }

    SUBCASE("A[2x3] B[3x4] multiplication")
    {
        Tensor A = tensor({1.0,2.0,3.0,4.0,5.0,6.0}, Shape{2,3});
        Tensor B = tensor({12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0}, Shape{3,4});
        auto result = A.matmul(B);
        CHECK(result.shape() == Shape{2,4});
        CHECK(result.value().size() == 8);
        CheckVectorApproxValues(result, tensor({40.0,34.0,28.0,22.0,112.0,97.0,82.0,67.0}, Shape{2,4}));
    }
}


TEST_CASE("Tensor - print")
{
    SUBCASE("scalar tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor(1.0);

        ss << input;
        std::string expected = R"(1.0000

[ CPU Float32 {} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1 dimension tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, Shape{1});

        ss << input;
        std::string expected = R"(  1.0000

[ CPU Float32 {1} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1 dimension tensor - three elements")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0, 2.0, 3.0}, Shape{3});

        ss << input;
        std::string expected = R"(  1.0000
  2.0000
  3.0000

[ CPU Float32 {3} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1x1 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, {1,1});

        ss << input;
        std::string expected = R"(  1.0000

[ CPU Float32 {1,1} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 }, {2, 2});

        ss << input;
        std::string expected = R"(  1.0000  2.0000
  2.0000  3.0000

[ CPU Float32 {2,2} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("1x2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 }, {1, 2, 2});

        ss << input;
        std::string expected = R"((0,.,.) =
  1.0000  2.0000
  2.0000  3.0000

[ CPU Float32 {1,2,2} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("2x2x2 tensor")
    {
        std::stringstream ss;
        auto input = aix::tensor({
                                 1.0, 2.0,
                                 2.0, 3.0,
                                 3.0, 4.0,
                                 4.0, 5.0,
                                 }, {2, 2, 2});

        ss << input;
        std::string expected = R"((0,.,.) =
  1.0000  2.0000
  2.0000  3.0000

(1,.,.) =
  3.0000  4.0000
  4.0000  5.0000

[ CPU Float32 {2,2,2} ]
)";
        CHECK(ss.str() == expected);
    }

    SUBCASE("DataType Float64")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kFloat64 });

        ss << input;
        std::string expected = R"(  1.0000

[ CPU Float64 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int64")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kInt64 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int64 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int32")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kInt32 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int32 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int16")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kInt16 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int16 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int8")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kInt8 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int8 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  UInt8")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .requireGrad=false, .dtype=DataType::kUInt8 });

        ss << input;
        std::string expected = R"(  1

[ CPU UInt8 {1} ]
)";
        CHECK(ss.str() == expected);
    }

}


TEST_CASE("Tensor - Tensor OP Scalar")
{
    float scalar = 5;
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.0, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input + scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x += scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input - scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x -= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input * scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x *= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input / scalar;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x /= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Scalar OP Tensor")
{
    float scalar = 5;
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.5, 3.0};

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar + input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar + x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar - input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar - x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar * input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar * x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = scalar / input;

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x = scalar / x; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Tensor OP Scalar Tensor")
{
    float scalar = 5;
    Tensor scalarTensor = tensor(scalar);
    Shape shape{2, 2};
    std::vector<float> testData = {1.0, -2.0, 0.0, 3.0};

    // Scalar tensor has no dimension.
    CHECK(scalarTensor.shape().size() == 0);

    SUBCASE("Add")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input + scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x += scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Sub")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input - scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x -= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Mul")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input * scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x *= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }

    SUBCASE("Div")
    {
        auto data = testData;
        auto input = aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape);
        auto result = input / scalarTensor;
        CHECK(result.shape() == input.shape());

        std::for_each(data.begin(), data.end(), [scalar](float & x) { x /= scalar; });
        CheckVectorApproxValues(result, aix::Tensor(data.data(), data.size(), DataType::kFloat32, shape));
    }
}


TEST_CASE("Tensor - Reshape")
{
    SUBCASE("scalar -> 1 dimension")
    {
        auto input = Tensor(5.0, {});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("scalar -> 1x1 dimension")
    {
        auto input = Tensor(5.0, {});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1 dimension -> 1x1 dimension")
    {
        auto input = Tensor(5.0, {1});
        auto newShape = Shape{1,1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1 dimension -> scalar")
    {
        auto input = Tensor(5.0, {1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x1 dimension -> scalar")
    {
        auto input = Tensor(5.0, {1,1});
        auto newShape = Shape{};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x1 dimension -> 1 dimension")
    {
        auto input = Tensor(5.0, {1,1});
        auto newShape = Shape{1};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(5.0, newShape));
    }

    SUBCASE("1x3 dimension -> 3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {1,3});
        auto newShape = Shape{3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 6 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{6};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("6 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {6});
        auto newShape = Shape{2, 3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 3x1x2 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{3, 1, 2};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("2x3 dimension -> 2x3 dimension")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("Size mismatch")
    {
        auto data = std::initializer_list<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto input = Tensor(data.begin(), data.size(), DataType::kFloat32, {2,3});
        auto newShape = Shape{2,3};
        auto x = input.reshape(newShape);
        CHECK(x.shape() == newShape);
        CHECK(x.value().size() == input.value().size());
        CheckVectorApproxValues(x, Tensor(data.begin(), data.size(), DataType::kFloat32, newShape));
    }

    SUBCASE("Invalid reshape throws invalid_argument")
    {
        Tensor inputs = tensor({1.0,2.0,3.0,4.0,5.0}, {1,5});
        // Check that reshape throws an invalid_argument exception
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({2,2}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({1,6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({4}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({6}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(inputs.reshape({}), std::invalid_argument);
    }
}


TEST_CASE("Tensor - broadcastTo")
{
    // NOTE: Since TensorValue tests cover the broadcast tests, Tensor does not need exhaustive broadcastTo tests.

    SUBCASE("Scalar to Scalar")
    {
        Shape newShape{};
        auto bct = tensor(1.0).broadcastTo(newShape);
        CHECK(bct.value().size() == 1);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, tensor({1.0}, newShape));
    }

    SUBCASE("Scalar to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = tensor({1.0}, Shape{}).broadcastTo(newShape);
        CHECK(bct.value().size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, tensor({1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, newShape));
    }

    SUBCASE("[1x3] to [2x3]")
    {
        Shape newShape{2, 3};
        auto bct = tensor({1.0, 2.0, 3.0}, {1, 3}).broadcastTo(newShape);
        CHECK(bct.value().size() == 6);
        CHECK(bct.shape() == newShape);
        CheckVectorApproxValues(bct, tensor({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, newShape));
    }

    SUBCASE("[2x3] to [3x2]")
    {
        CHECK_THROWS_AS(tensor({1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0}, {2, 3}).broadcastTo({3, 2}), std::invalid_argument);
    }
}


TEST_CASE("Tensor - Data Type Conversion")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f64Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }
}


TEST_CASE("Tensor - Data Type Conversion wit .to()")
{
    auto f32Data = std::initializer_list<float>{1.0, 2.0, 3.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    SUBCASE("Constructors - F64 to F64")
    {
        auto tv1 = aix::tensor(f64Data, shape, { .requireGrad=false }).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = aix::tensor(f64Data, shape).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f64Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false }).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false }).to(DataType::kFloat32);
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }
}


TEST_CASE("Tensor - Data Type Promotion")
{
    auto f32Data = std::initializer_list<float>{2.0, 4.0, 6.0};
    auto f64Data = std::initializer_list<double>{1.0, 2.0, 3.0};
    Shape shape{f32Data.size()};

    // Add

    SUBCASE("Add - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 3.0f);
        CHECK(tv.value().data<float>()[1] == 6.0f);
        CHECK(tv.value().data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    // Sub

    SUBCASE("Sub - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 1.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    // Mul

    SUBCASE("Mul - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==  2.0f);
        CHECK(tv.value().data<float>()[1] ==  8.0f);
        CHECK(tv.value().data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    // Div

    SUBCASE("Div - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 2.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  2.0);
        CHECK(tv.value().data<double>()[2] ==  2.0);
    }

    // Pow

    SUBCASE("Pow - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==   2.0f);
        CHECK(tv.value().data<float>()[1] ==  16.0f);
        CHECK(tv.value().data<float>()[2] == 216.0f);
    }

    SUBCASE("Pow - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    // Matmul

    SUBCASE("Matmul - F64 and F64")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<float>()[0] == 28.0f);
    }

    SUBCASE("Matmul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .requireGrad=false, .dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .requireGrad=false, .dtype=DataType::kFloat32 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }
}


TEST_CASE("Tensor - broadcast")
{
    SUBCASE("([],[1],[1,1],[1,3],[2,3]) op [2x3]")
    {
        std::vector<Shape> shapes{{}, {1}, {1,1}, {1,3}, {2,3}};
        for (const auto& shape : shapes)
        {
            Shape newShape{2,3};
            size_t newSize = 6;
            auto x = Tensor(2.0, shape);
            auto y = tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, newShape);

            auto a1 = x + y;
            CHECK(a1.value().size() == newSize);
            CHECK(a1.shape() == newShape);
            CheckVectorApproxValues(a1, tensor({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape));

            // Try reverse order
            auto a2 = y + x;
            CHECK(a2.value().size() == newSize);
            CHECK(a2.shape() == newShape);
            CheckVectorApproxValues(a2, tensor({3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, newShape));

            auto s1 = x - y;
            CHECK(s1.value().size() == newSize);
            CHECK(s1.shape() == newShape);
            CheckVectorApproxValues(s1, tensor({1.0, 0.0, -1.0, -2.0, -3.0, -4.0}, newShape));

            // Try reverse order
            auto s2 = y - x;
            CHECK(s2.value().size() == newSize);
            CHECK(s2.shape() == newShape);
            CheckVectorApproxValues(s2, tensor({-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, newShape));

            auto m1 = x * y;
            CHECK(m1.value().size() == newSize);
            CHECK(m1.shape() == newShape);
            CheckVectorApproxValues(m1, tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape));

            // Try reverse order
            auto m2 = y * x;
            CHECK(m2.value().size() == newSize);
            CHECK(m2.shape() == newShape);
            CheckVectorApproxValues(m2, tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0}, newShape));

            auto d1 = x / y;
            CHECK(d1.value().size() == newSize);
            CHECK(d1.shape() == newShape);
            CheckVectorApproxValues(d1, tensor({2.0, 1.0, 0.666667, 0.5, 0.4, 0.333334}, newShape));

            // Try reverse order
            auto d2 = y / x;
            CHECK(d2.value().size() == newSize);
            CHECK(d2.shape() == newShape);
            CheckVectorApproxValues(d2, tensor({0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, newShape));
        }
    }

    SUBCASE("[2x3] [3x2]")
    {
        std::initializer_list<float> data{1.0, 2.0, 3.0,4.0, 5.0, 6.0};
        Shape shape1{2,3};
        Shape shape2{3,2};
        auto tensor1 = tensor(data, shape1);
        auto tensor2 = tensor(data, shape2);

        // Add
        CHECK_THROWS_AS(tensor1 + tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 + tensor1, std::invalid_argument);

        // Sub
        CHECK_THROWS_AS(tensor1 - tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 - tensor1, std::invalid_argument);

        // Mul
        CHECK_THROWS_AS(tensor1 * tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 * tensor1, std::invalid_argument);

        // Div
        CHECK_THROWS_AS(tensor1 / tensor2, std::invalid_argument);
        CHECK_THROWS_AS(tensor2 / tensor1, std::invalid_argument);
    }
}


TEST_CASE("Tensor - Tensor OP Scalar - Data Type")
{
    float scalar = 2.0f;
    Shape shape{4};
    std::initializer_list<float> testData{1, 2, 3, 4};

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto testDType = static_cast<DataType>(i);
        auto x = aix::tensor(testData).to(testDType);
        auto expectedType = promoteDataTypeToFloat(testDType);

        SUBCASE("Add")
        {
            auto y = x + scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({3.0, 4.0, 5.0, 6.0}).to(expectedType));
        }

        SUBCASE("Sub")
        {
            auto y = x - scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({-1.0, 0.0, 1.0, 2.0}).to(expectedType));
        }

        SUBCASE("Mul")
        {
            auto y = x * scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({2.0, 4.0, 6.0, 8.0}).to(expectedType));
        }

        SUBCASE("Div")
        {
            auto y = x / scalar;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({0.5, 1.0, 1.5, 2.0}).to(expectedType));
        }
    }
}


TEST_CASE("Tensor - Scalar OP Tensor - Data Type")
{
    float scalar = 2.0f;
    Shape shape{4};
    std::initializer_list<float> testData{1, 2, 3, 4};

    for (size_t i=0; i<aix::DataTypeCount; ++i)
    {
        auto testDType = static_cast<DataType>(i);
        auto x = aix::tensor(testData).to(testDType);
        auto expectedType = promoteDataTypeToFloat(testDType);

        SUBCASE("Add")
        {
            auto y = scalar + x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({3.0, 4.0, 5.0, 6.0}).to(expectedType));
        }

        SUBCASE("Sub")
        {
            auto y = scalar - x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({1.0, 0.0, -1.0, -2.0}).to(expectedType));
        }

        SUBCASE("Mul")
        {
            auto y = scalar * x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({2.0, 4.0, 6.0, 8.0}).to(expectedType));
        }

        SUBCASE("Div")
        {
            auto y = scalar / x;
            CHECK(y.dataType() == promoteDataTypeToFloat(expectedType));
            CheckVectorApproxValues(y, aix::tensor({2.0, 1.0, 0.666667, 0.5}).to(expectedType));
        }
    }
}


TEST_CASE("Tensor - Variance")
{
    std::initializer_list<float> data = { 1.0,  2.0,  3.0,
                                          4.0,  5.0,  6.0,
                                          7.0,  8.0,  9.0,
                                          10.0, 11.0, 12.0,
                                          13.0, 14.0, 15.0,
                                          16.0, 17.0, 18.0,
                                          19.0, 20.0, 21.0,
                                          22.0, 23.0, 24.0};
    Shape shape{3, 4, 2};

    SUBCASE("default")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var();
        CHECK(var.value().item<float>() == doctest::Approx(50));
    }

    SUBCASE("unbiased = true")
    {
        auto a = aix::tensor(data, shape);
        auto var  = a.var(true);
        CHECK(var.value().item<float>() == doctest::Approx(50));
    }

    SUBCASE("unbiased = false")
    {
        auto a = aix::tensor(data, shape);
        auto var  = a.var(false);
        CHECK(var.value().item<float>() == doctest::Approx(47.9167));
    }

    SUBCASE("dim = 0 unbiased = default, keepdim = default")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0));
        CHECK(var.shape() == Shape{4,2});
        CheckVectorApproxValues(var, aix::Tensor(64.0, {4,2}));
    }

    SUBCASE("dim = 0 unbiased = true, keepdim = default")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0), true);
        CHECK(var.shape() == Shape{4, 2});
        CheckVectorApproxValues(var, aix::Tensor(64.0, {4,2}));
    }

    // ---

    SUBCASE("dim = 0 unbiased = true, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0), true, false);
        CHECK(var.shape() == Shape{4, 2});
        CheckVectorApproxValues(var, aix::Tensor(64.0, {4,2}));
    }

    SUBCASE("dim = 0 unbiased = true, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0), true, true);
        CHECK(var.shape() == Shape{1,4,2});
        CheckVectorApproxValues(var, aix::Tensor(64.0, {1,4,2}));
    }

    SUBCASE("dim = 0 unbiased = false, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0), false, false);
        CHECK(var.shape() == Shape{4, 2});
        CheckVectorApproxValues(var, aix::Tensor(42.6667, {4,2}));
    }

    SUBCASE("dim = 0 unbiased = false, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(0), false, true);
        CHECK(var.shape() == Shape{1,4,2});
        CheckVectorApproxValues(var, aix::Tensor(42.6667, {1,4,2}));
    }

    // ---

    SUBCASE("dim = 1 unbiased = true, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(1), true, false);
        CHECK(var.shape() == Shape{3,2});
        CheckVectorApproxValues(var, aix::Tensor(6.6667, {3,2}));
    }

    SUBCASE("dim = 1 unbiased = true, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(1), true, true);
        CHECK(var.shape() == Shape{3,1,2});
        CheckVectorApproxValues(var, aix::Tensor(6.6667, {3,1,2}));
    }

    SUBCASE("dim = 1 unbiased = false, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(1), false, false);
        CHECK(var.shape() == Shape{3, 2});
        CheckVectorApproxValues(var, aix::Tensor(5.0, {3,2}));
    }

    SUBCASE("dim = 1 unbiased = false, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(1), false, true);
        CHECK(var.shape() == Shape{3,1,2});
        CheckVectorApproxValues(var, aix::Tensor(5.0, {3,1,2}));
    }

    // ---

    SUBCASE("dim = 2 unbiased = true, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(2), true, false);
        CHECK(var.shape() == Shape{3,4});
        CheckVectorApproxValues(var, aix::Tensor(0.5, {3,4}));
    }

    SUBCASE("dim = 2 unbiased = true, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(2), true, true);
        CHECK(var.shape() == Shape{3,4,1});
        CheckVectorApproxValues(var, aix::Tensor(0.5, {3,4,1}));
    }

    SUBCASE("dim = 2 unbiased = false, keepdim = false")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(2), false, false);
        CHECK(var.shape() == Shape{3, 4});
        CheckVectorApproxValues(var, aix::Tensor(0.25, {3,4}));
    }

    SUBCASE("dim = 2 unbiased = false, keepdim = true")
    {
        auto a = aix::tensor(data, shape);
        auto var = a.var(ssize_t(2), false, true);
        CHECK(var.shape() == Shape{3,4,1});
        CheckVectorApproxValues(var, aix::Tensor(0.25, {3,4,1}));
    }

}


TEST_CASE("Tensor - max")
{
    SUBCASE("scalar")
    {
        auto a = aix::tensor(5.0, {}).requireGrad(true);
        auto max = a.max();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 5.0);
    }

    SUBCASE("1x1 tensor")
    {
        auto a = aix::tensor({5.0}, {1, 1}).requireGrad(true);
        auto max = a.max();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 5.0);
    }

    SUBCASE("2x2 tensor - first is max")
    {
        auto a = aix::tensor({10.0, 9.0, 8.0, -10.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 10.0);
    }

    SUBCASE("2x2 tensor - last is max")
    {
        auto a = aix::tensor({-10.0, 9.0, 8.0, 10.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 10.0);
    }

    SUBCASE("2x2 tensor - first found max")
    {
        auto a = aix::tensor({7.0, 8.0, 9.0, 9.0 }, {2,2}).requireGrad(true);
        auto max = a.max();
        CHECK(max.shape() == Shape{});
        CHECK(max.value().item<float>() == 9.0);
    }

    SUBCASE("2x2 tensor - complex")
    {
        auto a = aix::tensor({1.0, 4.0, 3.0, 2.0}, {2,2}).requireGrad(true);
        auto max = a.max() * a;
        CHECK(max.shape() == Shape{2,2});
        CheckVectorApproxValues(max, aix::tensor({ 4.0, 16.0, 12.0, 8.0 }, max.shape()));
    }
}


TEST_CASE("Tensor - argmax")
{
    SUBCASE("scalar")
    {
        auto a = aix::tensor(5.0, {}).requireGrad(true);
        auto amax = a.argmax();
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
    }

    SUBCASE("1 tensor")
    {
        auto a = aix::Tensor(5.0, Shape{1}).requireGrad(true);
        auto amax = a.argmax();
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
    }

    SUBCASE("1x1 tensor")
    {
        auto a = aix::Tensor(5.0, Shape{1,1}).requireGrad(true);
        auto amax = a.argmax();
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 0);
    }

    SUBCASE("2x2 tensor")
    {
        auto a = aix::tensor({1.0,2.0,3.0,4.0}, Shape{2,2}).requireGrad(true);
        auto amax = a.argmax();
        CHECK(amax.shape() == Shape{});
        CHECK(amax.value().item<int32_t>() == 3);
    }

    SUBCASE("2x2 tensor - complex")
    {
        auto a = aix::tensor({1.0,4.0,3.0,2.0}, Shape{2,2}).requireGrad(true);
        auto amax = a.argmax() * a;
        CHECK(amax.shape() == Shape{2,2});
        CheckVectorApproxValues(amax, aix::tensor({ 1.0,4.0,3.0,2.0 }, amax.shape()));
    }
}


TEST_CASE("Tensor - Argmax with dim")
{
    auto t1  = tensor({1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 9.0, 8.0, 7.0}, Shape{3,3});

    SUBCASE("Shape{} - dim=0 keepDim=false")
    {
        auto t = Tensor(5.0, Shape{});     // Scalar tensor.
        t = t.argmax(0, false);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=false")
    {
        auto t  = tensor({5.0}, Shape{1});
        t = t.argmax(0, false);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=true")
    {
        auto t  = tensor({5.0}, Shape{1});
        t = t.argmax(0, true);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=false")
    {
        auto t  = tensor({5.0}, Shape{1,1});
        t = t.argmax(0, false);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=true")
    {
        auto t  = tensor({5.0}, Shape{1,1});
        t = t.argmax(0, true);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=false")
    {
        auto t = t1.argmax(0, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, tensor({2.0, 2.0, 2.0}, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=true")
    {
        auto t = t1.argmax(0, true);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, tensor({2.0, 2.0, 2.0}, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=false")
    {
        auto t = t1.argmax(1, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, tensor({2.0, 1.0, 0.0}, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=true")
    {
        auto t = t1.argmax(1, true);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, tensor({2.0, 1.0, 0.0}, t.shape(), { .dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dimension out of range")
    {
        CHECK_THROWS_AS({ t1.argmax(2, false);  }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.argmax(2, true);   }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.argmax(-3, false); }, std::invalid_argument);
        CHECK_THROWS_AS({ t1.argmax(-3, true);  }, std::invalid_argument);
    }

}


TEST_CASE("Tensor - Select operator")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});

    SUBCASE("Shape{1} - [0]")
    {
        auto t = t1[0];
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, aix::tensor({5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - [0]")
    {
        auto t = t11[0];
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({5.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - [0]")
    {
        auto t = t33[0];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - [1]")
    {
        auto t = t33[1];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 4.0, 5.0, 6.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - [2]")
    {
        auto t = t33[2];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 7.0, 8.0, 9.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - [-3]")
    {
        auto t = t33[0];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - [-2]")
    {
        auto t = t33[1];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 4.0, 5.0, 6.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - [-1]")
    {
        auto t = t33[2];
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 7.0, 8.0, 9.0 }, t.shape()));
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts[0]; }, std::invalid_argument);
        CHECK_THROWS_AS({ t11[1]; }, std::invalid_argument);
    }
}


TEST_CASE("Tensor - Select")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});

    // Skipping the dim=0 cases since they are already tested through the select operator tests.

    SUBCASE("Shape{3,3} - dim=1 index=0")
    {
        auto t = t33.select(1,0);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 4.0, 7.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 index=1")
    {
        auto t = t33.select(1,1);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 5.0, 8.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 index=2")
    {
        auto t = t33.select(1,2);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 6.0, 9.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 index=-3")
    {
        auto t = t33.select(1,-3);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 4.0, 7.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 index=-2")
    {
        auto t = t33.select(1,-2);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 5.0, 8.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 index=-1")
    {
        auto t = t33.select(1,-1);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 6.0, 9.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=-1 index=-3")
    {
        auto t = t33.select(-1,-3);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 4.0, 7.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=-1 index=-2")
    {
        auto t = t33.select(-1,-2);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 5.0, 8.0 }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=-1 index=-1")
    {
        auto t = t33.select(-1,-1);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 6.0, 9.0 }, t.shape()));
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts[0]; }, std::invalid_argument);
        CHECK_THROWS_AS({ t11[1]; }, std::invalid_argument);
    }
}


TEST_CASE("Tensor - Split")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});

    SUBCASE("Shape{3,3} - splitSize=1, dim=0")
    {
        auto tlist = t33.split(1, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{1,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CHECK(tlist[2].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 4.0, 5.0, 6.0 }, tlist[1].shape()));
        CheckVectorApproxValues(tlist[2], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[2].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=2, dim=0")
    {
        auto tlist = t33.split(2, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 2);
        CHECK(tlist[0].shape() == Shape{2,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[1].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=3, dim=0")
    {
        auto tlist = t33.split(3, 0);   // Returns vector or tensors.
        CHECK(tlist.size() == 1);
        CHECK(tlist[0].shape() == Shape{3,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0,
                                                        7.0, 8.0, 9.0 }, tlist[0].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=-2")
    {
        auto tlist = t33.split(1, -2);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{1,3});
        CHECK(tlist[1].shape() == Shape{1,3});
        CHECK(tlist[2].shape() == Shape{1,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 4.0, 5.0, 6.0 }, tlist[1].shape()));
        CheckVectorApproxValues(tlist[2], aix::tensor({ 7.0, 8.0, 9.0 }, tlist[2].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=1")
    {
        auto tlist = t33.split(1, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{3,1});
        CHECK(tlist[1].shape() == Shape{3,1});
        CHECK(tlist[2].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 4.0, 7.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 2.0, 5.0, 8.0 }, tlist[1].shape()));
        CheckVectorApproxValues(tlist[2], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[2].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=2, dim=1")
    {
        auto tlist = t33.split(2, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 2);
        CHECK(tlist[0].shape() == Shape{3,2});
        CHECK(tlist[1].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0,
                                                        4.0, 5.0,
                                                        7.0, 8.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[1].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=3, dim=1")
    {
        auto tlist = t33.split(3, 1);   // Returns vector or tensors.
        CHECK(tlist.size() == 1);
        CHECK(tlist[0].shape() == Shape{3,3});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 2.0, 3.0,
                                                        4.0, 5.0, 6.0,
                                                        7.0, 8.0, 9.0 }, tlist[0].shape()));
    }

    SUBCASE("Shape{3,3} - splitSize=1, dim=-1")
    {
        auto tlist = t33.split(1, -1);   // Returns vector or tensors.
        CHECK(tlist.size() == 3);
        CHECK(tlist[0].shape() == Shape{3,1});
        CHECK(tlist[1].shape() == Shape{3,1});
        CHECK(tlist[2].shape() == Shape{3,1});
        CheckVectorApproxValues(tlist[0], aix::tensor({ 1.0, 4.0, 7.0 }, tlist[0].shape()));
        CheckVectorApproxValues(tlist[1], aix::tensor({ 2.0, 5.0, 8.0 }, tlist[1].shape()));
        CheckVectorApproxValues(tlist[2], aix::tensor({ 3.0, 6.0, 9.0 }, tlist[2].shape()));
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
