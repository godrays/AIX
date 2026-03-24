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
#include <sstream>


using namespace aix;


TEST_CASE("Tensor - name")
{
    auto tensor = aix::tensor({1.0, 2.0}, {2}, { .m_requireGrad=false });
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
        aix::Tensor t = aix::ones({2, 3}, { .m_requireGrad=true });
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
        aix::Tensor t = aix::zeros({2, 3}, { .m_requireGrad=true });
        CHECK(t.shape() == aix::Shape{2, 3});
        CHECK(t.value().size() == 6);
        CHECK(t.isRequireGrad() == true);
        CheckVectorApproxValues(t, Tensor(0.0, {2, 3}));
    }
}


TEST_CASE("Tensor - Device Switch")
{
    auto x = aix::tensor({1.0, 2.0, 3.0}, {1, 3});

    DeviceCPU  newDevice;
    auto newX = x.to(&newDevice);

    CHECK(newX.device() == &newDevice);
    CheckVectorApproxValues(x, aix::tensor({1.0, 2.0, 3.0}, x.shape(), { .m_device=&newDevice }));
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


TEST_CASE("Tensor - matmul equivalent broadcast reduction expression")
{
    auto check = [](aix::Device* device)
    {
        auto a = tensor({1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f,
                         7.0f, 8.0f, 9.0f}, {3, 3}, { .m_device=device });
        auto b = tensor({9.0f, 8.0f, 7.0f,
                         6.0f, 5.0f, 4.0f,
                         3.0f, 2.0f, 1.0f}, {3, 3}, { .m_device=device });

        auto lhsView = a.reshape({3, 1, 3});
        auto rhsTranspose = b.transpose(0, 1);
        auto rhsReshaped = rhsTranspose.reshape({1, 3, 3});
        auto expr = (lhsView * rhsReshaped).sum(2);
        auto mm = a.matmul(b);

        CheckVectorApproxValues(expr, mm);

        CHECK(lhsView.value().storage() == a.value().storage());
        CHECK(lhsView.value().storageOffset() == a.value().storageOffset());
        CHECK(lhsView.value().data() == a.value().data());

        CHECK(rhsTranspose.value().storage() == b.value().storage());
        CHECK(rhsTranspose.value().storageOffset() == b.value().storageOffset());
        CHECK(rhsTranspose.value().data() == b.value().data());
        CHECK_FALSE(rhsTranspose.value().isContiguous());

        CHECK(rhsReshaped.value().storage() == rhsTranspose.value().storage());
        CHECK(rhsReshaped.value().storageOffset() == rhsTranspose.value().storageOffset());
        CHECK(rhsReshaped.value().data() == rhsTranspose.value().data());
        CHECK_FALSE(rhsReshaped.value().isContiguous());
        CHECK(rhsReshaped.value().strides() == Stride{3, 1, 3});

        auto lhsBroadcast = lhsView.broadcastTo({3, 3, 3});
        auto rhsBroadcast = rhsReshaped.broadcastTo({3, 3, 3});
        CHECK(lhsBroadcast.value().storage() == lhsView.value().storage());
        CHECK(rhsBroadcast.value().storage() == rhsReshaped.value().storage());
    };

    SUBCASE("CPU")
    {
        aix::DeviceCPU device;
        check(&device);
    }

    SUBCASE("Metal")
    {
        auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
        if (!device) return;
        check(device.get());
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
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });

        ss << input;
        std::string expected = R"(  1.0000

[ CPU Float64 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int64")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kInt64 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int64 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int32")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kInt32 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int32 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int16")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kInt16 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int16 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  Int8")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kInt8 });

        ss << input;
        std::string expected = R"(  1

[ CPU Int8 {1} ]
)";
        CHECK(ss.str() == expected);
    }


    SUBCASE("DataType  UInt8")
    {
        std::stringstream ss;
        auto input = aix::tensor({1.0}, aix::Shape{1}, { .m_requireGrad=false, .m_dtype=DataType::kUInt8 });

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

    SUBCASE("Contiguous reshape remains zero-copy")
    {
        auto input = tensor({1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f}, {2, 3});
        auto reshaped = input.reshape({3, 2});

        CHECK(reshaped.value().storage() == input.value().storage());
        CHECK(reshaped.value().storageOffset() == input.value().storageOffset());
        CHECK(reshaped.value().data() == input.value().data());
        CheckVectorApproxValues(reshaped, tensor({1.0f, 2.0f, 3.0f,
                                                  4.0f, 5.0f, 6.0f}, {3, 2}));
    }

    SUBCASE("Non-contiguous view reshape preserves logical order on CPU and Metal")
    {
        DeviceCPU cpuDevice;
        auto expected = tensor({1.0f, 4.0f, 2.0f,
                                5.0f, 3.0f, 6.0f}, {2, 3}, { .m_device=&cpuDevice });

        Tensor cpuInput = tensor({1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f}, {2, 3}, { .m_device=&cpuDevice });
        auto cpuView = cpuInput.transpose(0, 1);
        auto cpuReshaped = cpuView.reshape({2, 3});

        CHECK_FALSE(cpuView.value().isContiguous());
        CHECK(cpuReshaped.shape() == Shape{2, 3});
        CheckVectorApproxValues(cpuReshaped, expected);

        auto metalDevice = aix::createDevice(aix::DeviceType::kGPU_METAL);
        if (metalDevice)
        {
            Tensor metalInput = tensor({1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f}, {2, 3}, { .m_device=metalDevice.get() });
            auto metalView = metalInput.transpose(0, 1);
            auto metalReshaped = metalView.reshape({2, 3});

            CHECK_FALSE(metalView.value().isContiguous());
            CHECK(metalReshaped.shape() == Shape{2, 3});

            auto actual = metalReshaped.value().contiguous();
            metalDevice->synchronize();
            auto expectedValue = expected.value().contiguous();
            for (size_t i = 0; i < expectedValue.size(); ++i)
            {
                CHECK(actual.data<float>()[i] == Approx(expectedValue.data<float>()[i]));
            }
        }
    }

    SUBCASE("Non-contiguous reshape does not reinterpret underlying storage order")
    {
        auto input = tensor({1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f}, {2, 3});
        auto view = input.transpose(0, 1);
        auto reshaped = view.reshape({2, 3});

        CHECK_FALSE(view.value().isContiguous());
        CHECK(reshaped.value().storage() != view.value().storage());
        CheckVectorApproxValues(reshaped, tensor({1.0f, 4.0f, 2.0f,
                                                  5.0f, 3.0f, 6.0f}, {2, 3}));
        CHECK(reshaped.value().contiguous().data<float>()[1] == Approx(4.0f));
    }

    SUBCASE("Stride-compatible non-contiguous reshape stays zero-copy on CPU and Metal")
    {
        DeviceCPU cpuDevice;
        auto expected = tensor({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {6}, { .m_device=&cpuDevice });

        Tensor cpuInput = tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f,
                                  9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, { .m_device=&cpuDevice });
        auto cpuView = cpuInput.slice(1, 0, 4, 2);
        auto cpuReshaped = cpuView.reshape({6});

        CHECK_FALSE(cpuView.value().isContiguous());
        CHECK(cpuReshaped.value().storage() == cpuView.value().storage());
        CHECK(cpuReshaped.value().storageOffset() == cpuView.value().storageOffset());
        CHECK(cpuReshaped.value().strides() == Stride{2});
        CheckVectorApproxValues(cpuReshaped, expected);

        auto metalDevice = aix::createDevice(aix::DeviceType::kGPU_METAL);
        if (metalDevice)
        {
            Tensor metalInput = tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f,
                                        9.0f, 10.0f, 11.0f, 12.0f}, {3, 4}, { .m_device=metalDevice.get() });
            auto metalView = metalInput.slice(1, 0, 4, 2);
            auto metalReshaped = metalView.reshape({6});

            CHECK_FALSE(metalView.value().isContiguous());
            CHECK(metalReshaped.value().storage() == metalView.value().storage());
            CHECK(metalReshaped.value().storageOffset() == metalView.value().storageOffset());
            CHECK(metalReshaped.value().strides() == Stride{2});
            CheckVectorApproxValues(metalReshaped, tensor({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {6},
                                                          { .m_device=metalDevice.get() }));
        }
    }

    SUBCASE("Singleton reshape on non-contiguous view stays zero-copy")
    {
        auto input = tensor({1.0f, 2.0f, 3.0f, 4.0f,
                             5.0f, 6.0f, 7.0f, 8.0f,
                             9.0f, 10.0f, 11.0f, 12.0f}, {3, 4});
        auto view = input.slice(1, 0, 4, 2);
        auto reshaped = view.reshape({3, 1, 2});

        CHECK(reshaped.value().storage() == view.value().storage());
        CHECK(reshaped.value().storageOffset() == view.value().storageOffset());
        CHECK(reshaped.value().strides() == Stride{4, 4, 2});
        CheckVectorApproxValues(reshaped, tensor({1.0f, 3.0f,
                                                  5.0f, 7.0f,
                                                  9.0f, 11.0f}, {3, 1, 2}));
    }

    SUBCASE("Broadcast stride-0 reshape remains rejected")
    {
        auto input = tensor({1.0f, 2.0f, 3.0f}, {1, 3});
        auto view = input.broadcastTo({2, 3});

        CHECK_FALSE(view.value().isContiguous());
        DOCTEST_CHECK_THROWS_AS(view.reshape({6}), std::invalid_argument);
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


TEST_CASE("Tensor - Reshape with an inferred dimension")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2});

    SUBCASE("{} - {}")
    {
        auto t = ts.reshape({});
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{} -> {-1}")
    {
        auto t = ts.reshape({-1});
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{} -> {1,-1}")
    {
        auto t = ts.reshape({1,-1});
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{1} -> {-1}")
    {
        auto t = t1.reshape({-1});
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{1,1} -> {1,-1}")
    {
        auto t = t11.reshape({1,-1});
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{1,1} -> {-1,1}")
    {
        auto t = t11.reshape({-1,1});
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, Tensor(5.0, t.shape()));
    }

    SUBCASE("{3,3} -> {-1,1}")
    {
        auto t = t33.reshape({-1,9});
        CHECK(t.shape() == Shape{1,9});
        CheckVectorApproxValues(t, t33);
    }

    SUBCASE("{3,3} -> {1,-1}")
    {
        auto t = t33.reshape({9,-1});
        CHECK(t.shape() == Shape{9,1});
        CheckVectorApproxValues(t, t33);
    }

    SUBCASE("{3,3} -> {3,-1}")
    {
        auto t = t33.reshape({3,-1});
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, t33);
    }

    SUBCASE("{3,3} -> {-1,3}")
    {
        auto t = t33.reshape({-1,3});
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, t33);
    }

    SUBCASE("{2,2,2} -> {-1,8}")
    {
        auto t = t222.reshape({-1,8});
        CHECK(t.shape() == Shape{1,8});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {-1,1,8}")
    {
        auto t = t222.reshape({-1,1,8});
        CHECK(t.shape() == Shape{1,1,8});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {8,-1}")
    {
        auto t = t222.reshape({8,-1});
        CHECK(t.shape() == Shape{8,1});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {8,1,-1}")
    {
        auto t = t222.reshape({8,1,-1});
        CHECK(t.shape() == Shape{8,1,1});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {1,-1,1,1}")
    {
        auto t = t222.reshape({1,-1,1,1});
        CHECK(t.shape() == Shape{1,8,1,1});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {-1,2,2}")
    {
        auto t = t222.reshape({-1,2,2});
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {2,-1,2}")
    {
        auto t = t222.reshape({2,-1,2});
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {2,2,-1}")
    {
        auto t = t222.reshape({2,2,-1});
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {4,2,-1}")
    {
        auto t = t222.reshape({4,2,-1});
        CHECK(t.shape() == Shape{4,2,1});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {2,4,-1}")
    {
        auto t = t222.reshape({2,4,-1});
        CHECK(t.shape() == Shape{2,4,1});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {2,2,2}")
    {
        auto t = t222.reshape({2,2,2});
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {1,2,4}")
    {
        auto t = t222.reshape({1,2,4});
        CHECK(t.shape() == Shape{1,2,4});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("{2,2,2} -> {8}")
    {
        auto t = t222.reshape({8});
        CHECK(t.shape() == Shape{8});
        CheckVectorApproxValues(t, t222);
    }

    SUBCASE("Invalid use cases")
    {
        DOCTEST_CHECK_THROWS_AS(ts.reshape({-2}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t33.reshape({-1,-1}), std::invalid_argument);
        DOCTEST_CHECK_THROWS_AS(t33.reshape({-2,9}), std::invalid_argument);
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
        auto tv1 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F64 to F32")
    {
        auto tv1 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        CHECK(tv1.dataType() == DataType::kFloat32);
        CHECK(tv1.value().size() == f64Data.size());
        CHECK(tv1.value().data<float>()[0] == 1.0f);
        CHECK(tv1.value().data<float>()[1] == 2.0f);
        CHECK(tv1.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Constructors - F32 to F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f64Data, shape, { .m_requireGrad=false }).to(DataType::kFloat64);
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false }).to(DataType::kFloat64);
        CHECK(tv1.dataType() == DataType::kFloat64);
        CHECK(tv1.value().size() == f32Data.size());
        CHECK(tv1.value().data<double>()[0] == 1.0);
        CHECK(tv1.value().data<double>()[1] == 2.0);
        CHECK(tv1.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Constructors - F32 to F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false }).to(DataType::kFloat32);
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 3.0f);
        CHECK(tv.value().data<float>()[1] == 6.0f);
        CHECK(tv.value().data<float>()[2] == 9.0f);
    }

    SUBCASE("Add - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 + tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 3.0);
        CHECK(tv.value().data<double>()[1] == 6.0);
        CHECK(tv.value().data<double>()[2] == 9.0);
    }

    SUBCASE("Add - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 1.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 3.0f);
    }

    SUBCASE("Sub - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 - tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 1.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 3.0);
    }

    SUBCASE("Sub - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==  2.0f);
        CHECK(tv.value().data<float>()[1] ==  8.0f);
        CHECK(tv.value().data<float>()[2] == 18.0f);
    }

    SUBCASE("Mul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 * tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==  2.0);
        CHECK(tv.value().data<double>()[1] ==  8.0);
        CHECK(tv.value().data<double>()[2] == 18.0);
    }

    SUBCASE("Mul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] == 2.0f);
        CHECK(tv.value().data<float>()[1] == 2.0f);
        CHECK(tv.value().data<float>()[2] == 2.0f);
    }

    SUBCASE("Div - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1 / tv2;
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] == 2.0);
        CHECK(tv.value().data<double>()[1] == 2.0);
        CHECK(tv.value().data<double>()[2] == 2.0);
    }

    SUBCASE("Div - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<float>()[0] ==   2.0f);
        CHECK(tv.value().data<float>()[1] ==  16.0f);
        CHECK(tv.value().data<float>()[2] == 216.0f);
    }

    SUBCASE("Pow - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1.pow(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == f32Data.size());
        CHECK(tv.value().data<double>()[0] ==   2.0);
        CHECK(tv.value().data<double>()[1] ==  16.0);
        CHECK(tv.value().data<double>()[2] == 216.0);
    }

    SUBCASE("Pow - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, shape, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        auto tv1 = aix::tensor(f32Data, {1,3}, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F32 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat32);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<float>()[0] == 28.0f);
    }

    SUBCASE("Matmul - F32 and F64")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv = tv1.matmul(tv2);
        CHECK(tv.dataType() == DataType::kFloat64);
        CHECK(tv.value().size() == 1);
        CHECK(tv.value().data<double>()[0] == 28.0);
    }

    SUBCASE("Matmul - F64 and F32")
    {
        auto tv1 = aix::tensor(f32Data, {1,3}, { .m_requireGrad=false, .m_dtype=DataType::kFloat64 });
        auto tv2 = aix::tensor(f64Data, {3,1}, { .m_requireGrad=false, .m_dtype=DataType::kFloat32 });
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
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=false")
    {
        auto t  = tensor({5.0}, Shape{1});
        t = t.argmax(0, false);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1} - dim=0 keepDim=true")
    {
        auto t  = tensor({5.0}, Shape{1});
        t = t.argmax(0, true);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=false")
    {
        auto t  = tensor({5.0}, Shape{1,1});
        t = t.argmax(0, false);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{1,1} - dim=0 keepDim=true")
    {
        auto t  = tensor({5.0}, Shape{1,1});
        t = t.argmax(0, true);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, Tensor(0, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=false")
    {
        auto t = t1.argmax(0, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, tensor({2.0, 2.0, 2.0}, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=0 keepDim=true")
    {
        auto t = t1.argmax(0, true);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, tensor({2.0, 2.0, 2.0}, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=false")
    {
        auto t = t1.argmax(1, false);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, tensor({2.0, 1.0, 0.0}, t.shape(), { .m_dtype=DataType::kInt32 }));
    }

    SUBCASE("Shape{3,3} - dim=1 keepDim=true")
    {
        auto t = t1.argmax(1, true);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, tensor({2.0, 1.0, 0.0}, t.shape(), { .m_dtype=DataType::kInt32 }));
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


TEST_CASE("Tensor - Cat")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2});

    SUBCASE("Shape{1} - dim=0")
    {
        auto t = aix::cat({t1}, 0);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({5.0}, t.shape()));
    }

    SUBCASE("Shape{1} - dim=0 - 2x")
    {
        auto t = aix::cat({t1,t1}, 0);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({5.0, 5.0}, t.shape()));
    }

    SUBCASE("Shape{1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t1,t1}, -1);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({5.0, 5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=0")
    {
        auto t = aix::cat({t11}, 0);
        CHECK(t.shape() == Shape{1,1});
        CheckVectorApproxValues(t, aix::tensor({5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=0 - 2x")
    {
        auto t = aix::cat({t11, t11}, 0);
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(t, aix::tensor({5.0,5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=-2 - 2x")
    {
        auto t = aix::cat({t11, t11}, -2);
        CHECK(t.shape() == Shape{2,1});
        CheckVectorApproxValues(t, aix::tensor({5.0,5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=1 - 2x")
    {
        auto t = aix::cat({t11, t11}, 1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::tensor({5.0,5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t11, t11}, -1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::tensor({5.0,5.0}, t.shape()));
    }

    SUBCASE("Shape{1,1} - dim=-1 - 2x")
    {
        auto t = aix::cat({t11, t11}, -1);
        CHECK(t.shape() == Shape{1,2});
        CheckVectorApproxValues(t, aix::tensor({5.0,5.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0")
    {
        auto t = aix::cat({t33}, 0);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,3.0,
                                                 4.0,5.0,6.0,
                                                 7.0,8.0,9.0,}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - 2x")
    {
        auto t = aix::cat({t33, t33}, 0);
        CHECK(t.shape() == Shape{6,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,3.0,
                                                 4.0,5.0,6.0,
                                                 7.0,8.0,9.0,
                                                 1.0,2.0,3.0,
                                                 4.0,5.0,6.0,
                                                 7.0,8.0,9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - 2x")
    {
        auto t = aix::cat({t33, t33}, 1);
        CHECK(t.shape() == Shape{3,6});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,3.0, 1.0,2.0,3.0,
                                                 4.0,5.0,6.0, 4.0,5.0,6.0,
                                                 7.0,8.0,9.0, 7.0,8.0,9.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=0 - 2x")
    {
        auto t = aix::cat({t222,t222}, 0);
        CHECK(t.shape() == Shape{4,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,
                                                 3.0,4.0,
                                                 5.0,6.0,
                                                 7.0,8.0,
                                                 1.0,2.0,
                                                 3.0,4.0,
                                                 5.0,6.0,
                                                 7.0,8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=-3 - 2x")
    {
        auto t = aix::cat({t222,t222}, -3);
        CHECK(t.shape() == Shape{4,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,
                                                 3.0,4.0,
                                                 5.0,6.0,
                                                 7.0,8.0,
                                                 1.0,2.0,
                                                 3.0,4.0,
                                                 5.0,6.0,
                                                 7.0,8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=1 - 2x")
    {
        auto t = aix::cat({t222,t222}, 1);
        CHECK(t.shape() == Shape{2,4,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,
                                                 3.0,4.0,
                                                 1.0,2.0,
                                                 3.0,4.0,
                                                 5.0,6.0,
                                                 7.0,8.0,
                                                 5.0,6.0,
                                                 7.0,8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=2 - 2x")
    {
        auto t = aix::cat({t222,t222}, 2);
        CHECK(t.shape() == Shape{2,2,4});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,1.0,2.0,
                                                 3.0,4.0,3.0,4.0,
                                                 5.0,6.0,5.0,6.0,
                                                 7.0,8.0,7.0,8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=-1 - 2x")
    {
        auto t = aix::cat({t222,t222}, -1);
        CHECK(t.shape() == Shape{2,2,4});
        CheckVectorApproxValues(t, aix::tensor({ 1.0,2.0,1.0,2.0,
                                                 3.0,4.0,3.0,4.0,
                                                 5.0,6.0,5.0,6.0,
                                                 7.0,8.0,7.0,8.0, }, t.shape()));
    }

    SUBCASE("Irregular sizes - {{9}, {1}, {4}}")
    {
        auto r1 = aix::arange(0.0, 9.0, 1.0);
        auto r2 = aix::tensor({9.0}, Shape{1});
        auto r3 = aix::arange(10.0, 14.0, 1.0);
        auto t = aix::cat({r1, r2, r3}, 0);
        CHECK(t.shape() == Shape{14});
        CheckVectorApproxValues(t, aix::arange(0.0, 14.0, 1.0).reshape(t.shape()));
    }

    // Type promotion.

    SUBCASE("{Int32, Int32} -> Int32")
    {
        auto t = aix::cat({t33.to(aix::DataType::kInt32), t33.to(aix::DataType::kInt32)}, 0);
        CHECK(t.dataType() == aix::DataType::kInt32);
    }

    SUBCASE("{Int32, Float16} -> Float16")
    {
        auto t = aix::cat({t33.to(aix::DataType::kInt32), t33.to(aix::DataType::kInt32)}, 0);
        CHECK(t.dataType() == aix::DataType::kInt32);
    }

    SUBCASE("{Float16, Float32} -> Float32")
    {
        auto t = aix::cat({t33.to(aix::DataType::kFloat16), t33.to(aix::DataType::kFloat32)}, 0);
        CHECK(t.dataType() == aix::DataType::kFloat32);
    }

    // Require Gradient.

    SUBCASE("{NotReq, NotReq} -> NotReq")
    {
        auto t = aix::cat({t33.requireGrad(false), t33.requireGrad(false)}, 0);
        CHECK(t.isRequireGrad() == false);
    }

    SUBCASE("{NotReq, Req} -> NotReq")
    {
        auto t = aix::cat({t33.requireGrad(false), t33.requireGrad(true)}, 0);
        CHECK(t.isRequireGrad() == true);
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ aix::cat({ts}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({ts,ts}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t1,t1}, 1); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t1,t1}, -2); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t11,t11}, -3); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t1,t33}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t33,t11}, 0); }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::cat({t33,t11}, 1); }, std::invalid_argument);
    }
}


TEST_CASE("Tensor - Arange")
{
    // Positive step.

    SUBCASE("start=0, end=0, step=1")
    {
        auto t = aix::arange(0, 0, 1);
        CHECK(t.shape() == Shape{0});
    }

    SUBCASE("start=0, end=0, step=0.5")
    {
        auto t = aix::arange(0, 0, 0.5);
        CHECK(t.shape() == Shape{0});
    }

    SUBCASE("start=0, end=0.1, step=1")
    {
        auto t = aix::arange(0, 0.1, 1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=0.1, step=0.5")
    {
        auto t = aix::arange(0, 0.1, 0.5);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=1, step=1")
    {
        auto t = aix::arange(0, 1, 1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=1, step=0.5")
    {
        auto t = aix::arange(0, 1, 0.5);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({0.0, 0.5}, t.shape()));
    }

    SUBCASE("start=1, end=2, step=1")
    {
        auto t = aix::arange(1, 2, 1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({1.0}, t.shape()));
    }

    SUBCASE("start=1, end=2, step=0.5")
    {
        auto t = aix::arange(1, 2, 0.5);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({1.0, 1.5}, t.shape()));
    }

    SUBCASE("start=1, end=1.1, step=1")
    {
        auto t = aix::arange(1, 1.1, 1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({1.0}, t.shape()));
    }

    SUBCASE("start=1, end=1.1, step=0.5")
    {
        auto t = aix::arange(1, 1.1, 1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({1.0}, t.shape()));
    }

    SUBCASE("start=0, end=2, step=1")
    {
        auto t = aix::arange(0, 2, 1);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({0.0, 1.0}, t.shape()));
    }

    SUBCASE("start=0, end=2, step=0.5")
    {
        auto t = aix::arange(0, 2, 0.5);
        CHECK(t.shape() == Shape{4});
        CheckVectorApproxValues(t, aix::tensor({0.0, 0.5, 1.0, 1.5}, t.shape()));
    }

    SUBCASE("start=-2, end=0, step=1")
    {
        auto t = aix::arange(-2, 0, 0.5);
        CHECK(t.shape() == Shape{4});
        CheckVectorApproxValues(t, aix::tensor({-2.0, -1.5, -1.0, -0.5}, t.shape()));
    }

    // Negative step.

    SUBCASE("start=0, end=0, step=-1")
    {
        auto t = aix::arange(0, 0, -1);
        CHECK(t.shape() == Shape{0});
    }

    SUBCASE("start=0, end=0, step=-0.5")
    {
        auto t = aix::arange(0, 0, -0.5);
        CHECK(t.shape() == Shape{0});
    }

    SUBCASE("start=0, end=-0.1, step=-1")
    {
        auto t = aix::arange(0, -0.1, -1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=-0.1, step=-0.5")
    {
        auto t = aix::arange(0, -0.1, -0.5);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=-1, step=-1")
    {
        auto t = aix::arange(0, -1, -1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({0.0}, t.shape()));
    }

    SUBCASE("start=0, end=-1, step=-0.5")
    {
        auto t = aix::arange(0, -1, -0.5);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({0.0, -0.5}, t.shape()));
    }

    SUBCASE("start=1, end=-2, step=-1")
    {
        auto t = aix::arange(1, -2, -1);
        CHECK(t.shape() == Shape{3});
        CheckVectorApproxValues(t, aix::tensor({1.0, 0.0, -1.0}, t.shape()));
    }

    SUBCASE("start=1, end=-2, step=-0.5")
    {
        auto t = aix::arange(1, -2, -0.5);
        CHECK(t.shape() == Shape{6});
        CheckVectorApproxValues(t, aix::tensor({1.0, 0.5, 0.0, -0.5, -1.0, -1.5}, t.shape()));
    }

    SUBCASE("start=-1, end=-1.1, step=-1")
    {
        auto t = aix::arange(-1, -1.1, -1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({-1.0}, t.shape()));
    }

    SUBCASE("start=-1, end=-1.1, step=-0.5")
    {
        auto t = aix::arange(-1, -1.1, -0.5);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::tensor({-1.0}, t.shape()));
    }

    SUBCASE("start=0, end=-2, step=-1")
    {
        auto t = aix::arange(0, -2, -1);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({0.0, -1.0}, t.shape()));
    }

    SUBCASE("start=0, end=-2, step=-0.5")
    {
        auto t = aix::arange(0, -2, -0.5);
        CHECK(t.shape() == Shape{4});
        CheckVectorApproxValues(t, aix::tensor({0.0, -0.5, -1.0, -1.5}, t.shape()));
    }

    SUBCASE("start=-1, end=-2, step=-0.5")
    {
        auto t = aix::arange(-1, -2, -0.5);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({-1.0, -1.5}, t.shape()));
    }

    SUBCASE("start=-2, end=-1, step=0.5")
    {
        auto t = aix::arange(-2, -1, 0.5);
        CHECK(t.shape() == Shape{2});
        CheckVectorApproxValues(t, aix::tensor({-2.0, -1.5}, t.shape()));
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ aix::arange(0, 0, 0);  }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::arange(-1, 1,-1);  }, std::invalid_argument);
        CHECK_THROWS_AS({ aix::arange(1, -1, 1);  }, std::invalid_argument);
    }
}


TEST_CASE("Tensor - indexSelect")
{
    auto ts  = aix::tensor(5.0);
    auto t1  = aix::tensor({5.0}, Shape{1});
    auto t11 = aix::tensor({5.0}, Shape{1,1});
    auto t33 = aix::tensor({ 1.0, 2.0, 3.0,
                             4.0, 5.0, 6.0,
                             7.0, 8.0, 9.0 }, Shape{3,3});
    auto t222 = aix::tensor({ 1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0 }, Shape{2,2,2});

    auto is  = aix::tensor(0.0, aix::dtype(aix::DataType::kInt32));
    auto i1  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
    auto i2  = aix::tensor({0.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
    auto i11 = aix::tensor({0.0}, Shape{1,1}, aix::dtype(aix::DataType::kInt32));

    SUBCASE("Shape{} - dim=0 - index{}")
    {
        auto t = ts.indexSelect(0, is);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, aix::Tensor(5.0, t.shape()));
    }

    SUBCASE("Shape{1} - dim=0 - index{}")
    {
        auto t = t1.indexSelect(0, is);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::Tensor(5.0, t.shape()));
    }

    SUBCASE("Shape{} - dim=0 - index{1}")
    {
        auto t = ts.indexSelect(0, i1);
        CHECK(t.shape() == Shape{});
        CheckVectorApproxValues(t, aix::Tensor(5.0, t.shape()));
    }

    SUBCASE("Shape{1} - dim=0 - index{1}")
    {
        auto t = t1.indexSelect(0, i1);
        CHECK(t.shape() == Shape{1});
        CheckVectorApproxValues(t, aix::Tensor(5.0, t.shape()));
    }

    // dim 0

    SUBCASE("Shape{3,3} - dim=0 - index = {0}")
    {
        auto indices  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::tensor({1.0, 2.0, 3.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::tensor({4.0, 5.0, 6.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2}")
    {
        auto indices  = aix::tensor({2.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{1,3});
        CheckVectorApproxValues(t, aix::tensor({7.0, 8.0, 9.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1,2}")
    {
        auto indices  = aix::tensor({1.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {0,2}")
    {
        auto indices  = aix::tensor({0.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 7.0, 8.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,0}")
    {
        auto indices  = aix::tensor({2.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 7.0, 8.0, 9.0,
                                                 1.0, 2.0, 3.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,3});
        CheckVectorApproxValues(t, aix::tensor({ 4.0, 5.0, 6.0,
                                                 4.0, 5.0, 6.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {0,1,2}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 2.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,1,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(0, indices);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 7.0, 8.0, 9.0,
                                                 4.0, 5.0, 6.0,
                                                 1.0, 2.0, 3.0, }, t.shape()));
    }

    // dim 1

    SUBCASE("Shape{3,3} - dim=1 - index = {0}")
    {
        auto indices  = aix::tensor({0.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::tensor({1.0, 4.0, 7.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::tensor({2.0, 5.0, 8.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {2}")
    {
        auto indices  = aix::tensor({2.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,1});
        CheckVectorApproxValues(t, aix::tensor({3.0, 6.0, 9.0}, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,1}")
    {
        auto indices  = aix::tensor({0.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0,
                                                 4.0, 5.0,
                                                 7.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1,2}")
    {
        auto indices  = aix::tensor({1.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 3.0,
                                                 5.0, 6.0,
                                                 8.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,2}")
    {
        auto indices  = aix::tensor({0.0, 2.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 3.0,
                                                 4.0, 6.0,
                                                 7.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=0 - index = {2,0}")
    {
        auto indices  = aix::tensor({2.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 1.0,
                                                 6.0, 4.0,
                                                 9.0, 7.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,2});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 2.0,
                                                 5.0, 5.0,
                                                 8.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {0,1,2}")
    {
        auto indices  = aix::tensor({0.0, 1.0, 2.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 1.0, 2.0, 3.0,
                                                 4.0, 5.0, 6.0,
                                                 7.0, 8.0, 9.0, }, t.shape()));
    }

    SUBCASE("Shape{3,3} - dim=1 - index = {2,1,0}")
    {
        auto indices  = aix::tensor({2.0, 1.0, 0.0}, Shape{3}, aix::dtype(aix::DataType::kInt32));
        auto t = t33.indexSelect(1, indices);
        CHECK(t.shape() == Shape{3,3});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 2.0, 1.0,
                                                 6.0, 5.0, 4.0,
                                                 9.0, 8.0, 7.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        CHECK(t.shape() == Shape{1,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 5.0, 6.0,
                                                 7.0, 8.0 }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        CHECK(t.shape() == Shape{2,1,2});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 4.0,
                                                 7.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1}")
    {
        auto indices  = aix::tensor({1.0}, Shape{1}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        CHECK(t.shape() == Shape{2,2,1});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 4.0,
                                                 6.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 5.0, 6.0,
                                                 7.0, 8.0,
                                                 1.0, 2.0,
                                                 3.0, 4.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 4.0,
                                                 1.0, 2.0,
                                                 7.0, 8.0,
                                                 5.0, 6.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1,0}")
    {
        auto indices  = aix::tensor({1.0, 0.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 1.0,
                                                 4.0, 3.0,
                                                 6.0, 5.0,
                                                 8.0, 7.0, }, t.shape()));
    }

    //

    SUBCASE("Shape{2,2,2} - dim=0 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(0, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 5.0, 6.0,
                                                 7.0, 8.0,
                                                 5.0, 6.0,
                                                 7.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=1 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(1, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 3.0, 4.0,
                                                 3.0, 4.0,
                                                 7.0, 8.0,
                                                 7.0, 8.0, }, t.shape()));
    }

    SUBCASE("Shape{2,2,2} - dim=2 - index = {1,1}")
    {
        auto indices  = aix::tensor({1.0, 1.0}, Shape{2}, aix::dtype(aix::DataType::kInt32));
        auto t = t222.indexSelect(2, indices);
        CHECK(t.shape() == Shape{2,2,2});
        CheckVectorApproxValues(t, aix::tensor({ 2.0, 2.0,
                                                 4.0, 4.0,
                                                 6.0, 6.0,
                                                 8.0, 8.0, }, t.shape()));
    }

    SUBCASE("Invalid use cases.")
    {
        CHECK_THROWS_AS({ ts.indexSelect(0, i11); }, std::invalid_argument);
        CHECK_THROWS_AS({ ts.indexSelect(1, i1);  }, std::invalid_argument);
        // Only kInt32 type should be allowed as an index tensor.
        for (size_t i=0; i<aix::DataTypeCount; ++i)
        {
            auto dtype = static_cast<aix::DataType>(i);
            if (dtype != aix::DataType::kInt32)
            {
                CHECK_THROWS_AS({ t33.indexSelect(0, i1.to(dtype)); }, std::invalid_argument);
            }
        }
        CHECK_THROWS_AS({ t33.indexSelect(3, i1);  }, std::invalid_argument);
        CHECK_THROWS_AS({ t33.indexSelect(-3, i1); }, std::invalid_argument);
    }
}

TEST_CASE("Tensor - eye")
{
    SUBCASE("eye(1)")
    {
        auto a = aix::eye(1);
        CheckVectorApproxValues(a, aix::tensor({ 1.0 }, aix::Shape{1, 1}));
    }

    SUBCASE("eye(2)")
    {
        auto a = aix::eye(2);
        CheckVectorApproxValues(a, aix::tensor({ 1.0, 0.0,
                                                 0.0, 1.0, }, aix::Shape{2, 2}));
    }

    SUBCASE("eye(3)")
    {
        auto a = aix::eye(3);
        CheckVectorApproxValues(a, aix::tensor({ 1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0, }, aix::Shape{3, 3}));
    }
}


TEST_CASE("Tensor - permute()")
{
    auto ts = tensor(5.0,{});
    auto t12 = tensor({1.0,2.0}, {1,2});
    auto t32 = tensor({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2});
    auto t324 = aix::arange(1.0, 25.0, 1.0).reshape({3,2,4});

    SUBCASE("Fail tests")
    {
        Tensor s = tensor(1.0, {});                         // Scalar tensor.
        Tensor t = tensor({1.0,2.0,3.0,4.0}, {2,1,2});
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
        Tensor t = tensor(5.0, {});
        auto result = t.permute({});
        CHECK(result.shape() == Shape{});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1} p{0}")
    {
        Tensor t = Tensor(5.0, {1});
        auto result = t.permute({0});
        CHECK(result.shape() == Shape{1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1} p{-1}")
    {
        Tensor t = Tensor(5.0, {1});
        auto result = t.permute({-1});
        CHECK(result.shape() == Shape{1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{0,1}")
    {
        Tensor t = Tensor(5.0, {1,1});
        auto result = t.permute({0,1});
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{-2,-1}")
    {
        Tensor t = Tensor(5.0, {1,1});
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{1,1});
        CheckVectorApproxValues(result, t);
    }

    SUBCASE("s{1,1} p{-1,-2}")
    {
        Tensor t = Tensor(5.0, {1,1});
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
        CheckVectorApproxValues(result, tensor({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}));
    }

    SUBCASE("s{3,2} p{1,0}")
    {
        auto t = t32;
        auto result = t.permute({1,0});
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, tensor({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}));
    }

    SUBCASE("s{3,2} p{-2,-1}")
    {
        auto t = t32;
        auto result = t.permute({-2,-1});
        CHECK(result.shape() == Shape{3,2});
        CheckVectorApproxValues(result, tensor({1.0,2.0,3.0,4.0,5.0,6.0}, {3,2}));
    }

    SUBCASE("s{3,2} p{-1,-2}")
    {
        auto t = t32;
        auto result = t.permute({-1,-2});
        CHECK(result.shape() == Shape{2,3});
        CheckVectorApproxValues(result, tensor({1.0,3.0,5.0,2.0,4.0,6.0}, {2,3}));
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
        CheckVectorApproxValues(result, tensor({ 1.0 , 5.0,
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
                                                 20.0, 24.0}, {3,4,2}));
    }

    SUBCASE("s{3,2,4} p{1,0,2}")
    {
        auto t = t324;
        auto result = t.permute({1,0,2});
        CHECK(result.shape() == Shape{2,3,4});
        CheckVectorApproxValues(result, tensor({  1.0,  2.0,  3.0,  4.0,
                                                  9.0, 10.0, 11.0, 12.0,
                                                  17.0, 18.0, 19.0, 20.0,
                                                  5.0,  6.0,  7.0,  8.0,
                                                  13.0, 14.0, 15.0, 16.0,
                                                  21.0, 22.0, 23.0, 24.0}, {2,3,4}));
    }

    SUBCASE("s{3,2,4} p{1,2,0}")
    {
        auto t = t324;
        auto result = t.permute({1,2,0});
        CHECK(result.shape() == Shape{2,4,3});
        CheckVectorApproxValues(result, tensor({ 1.0,  9.0, 17.0,
                                                 2.0, 10.0, 18.0,
                                                 3.0, 11.0, 19.0,
                                                 4.0, 12.0, 20.0,
                                                 5.0, 13.0, 21.0,
                                                 6.0, 14.0, 22.0,
                                                 7.0, 15.0, 23.0,
                                                 8.0, 16.0, 24.0}, {2,4,3}));
    }

    SUBCASE("s{3,2,4} p{2,0,1}")
    {
        auto t = t324;
        auto result = t.permute({2,0,1});
        CHECK(result.shape() == Shape{4,3,2});
        CheckVectorApproxValues(result, tensor({ 1.0,  5.0,
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
                                                 20.0, 24.0}, {4,3,2}));
    }

    SUBCASE("s{3,2,4} p{2,1,0}")
    {
        auto t = t324;
        auto result = t.permute({2,1,0});
        CHECK(result.shape() == Shape{4,2,3});
        CheckVectorApproxValues(result, tensor({ 1.0,  9.0, 17.0,
                                                 5.0, 13.0, 21.0,
                                                 2.0, 10.0, 18.0,
                                                 6.0, 14.0, 22.0,
                                                 3.0, 11.0, 19.0,
                                                 7.0, 15.0, 23.0,
                                                 4.0, 12.0, 20.0,
                                                 8.0, 16.0, 24.0}, {4,2,3}));
    }

    SUBCASE("s{3,2,4} p{-1,-2,-3}")
    {
        auto t = t324;
        auto result = t.permute({-1,-2,-3});
        CHECK(result.shape() == Shape{4,2,3});
        CheckVectorApproxValues(result, tensor({ 1.0,  9.0, 17.0,
                                                 5.0, 13.0, 21.0,
                                                 2.0, 10.0, 18.0,
                                                 6.0, 14.0, 22.0,
                                                 3.0, 11.0, 19.0,
                                                 7.0, 15.0, 23.0,
                                                 4.0, 12.0, 20.0,
                                                 8.0, 16.0, 24.0}, {4,2,3}));
    }
}


TEST_CASE("Tensor - Metal chained views stay metadata-only until elementwise execution")
{
    auto device = aix::createDevice(aix::DeviceType::kGPU_METAL);
    if (!device) return;

    auto cpuInput = aix::tensor({1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f,
                                 21.0f, 22.0f, 23.0f, 24.0f}, aix::Shape{2, 3, 4});
    auto input = aix::tensor({1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f,
                              13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f,
                              21.0f, 22.0f, 23.0f, 24.0f}, aix::Shape{2, 3, 4},
                             { .m_device=device.get() });
    auto cpuView = cpuInput.permute({1, 2, 0}).slice(1, 1, 4, 2);
    auto view = input.permute({1, 2, 0}).slice(1, 1, 4, 2);

    CHECK(cpuView.shape() == Shape{3, 2, 2});
    CHECK(view.shape() == Shape{3, 2, 2});
    CHECK(view.value().storage() == input.value().storage());
    CHECK(view.value().storageOffset() == cpuView.value().storageOffset());
    CHECK_FALSE(view.value().isContiguous());

    auto expected = cpuView * cpuView;
    auto result = view * view;
    device->synchronize();

    CHECK(result.shape() == Shape{3, 2, 2});
    CHECK(result.value().storage() != view.value().storage());
    CheckVectorApproxValues(result, expected);
}
