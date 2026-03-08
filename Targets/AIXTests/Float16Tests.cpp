//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include <aix.hpp>
// External includes
#include <doctest/doctest.h>
// System includes
#include <sstream>


using namespace aix;


TEST_CASE("float16_t Tests")
{
    SUBCASE("Parameterized Constructor")
    {
        // Python reference values obtained from struct.pack('>e', value).
        float16_t f1(3.14f);
        CHECK(f1.raw() == 0x4248);
        CHECK(f1.toFloat32() == doctest::Approx(3.140625f));

        float16_t f2(-3.14f);
        CHECK(f2.raw() == 0xC248);
        CHECK(f2.toFloat32() == doctest::Approx(-3.140625f));

        float16_t f3(0.0f);
        CHECK(f3.raw() == 0x0000);
        CHECK(f3.toFloat32() == doctest::Approx(0.0f));

        float16_t f4;
        CHECK(f4.raw() == 0x0000);
        CHECK(f4.toFloat32() == doctest::Approx(0.0f));
    }

    SUBCASE("Copy Constructor")
    {
        float16_t f1(3.14f);
        float16_t f2(f1);
        CHECK(f2.raw() == 0x4248);
        CHECK(f2.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Assignment Operator")
    {
        float16_t f1(3.14f);
        float16_t f2;
        f2 = f1;
        CHECK(f2.raw() == 0x4248);
        CHECK(f2.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Comparison Operators")
    {
        float16_t f1(3.14f);
        float16_t f2(3.14f);
        float16_t f3(1.0f);
        float16_t pz(0.0f);
        float16_t nz(-0.0f);
        float16_t nan(std::numeric_limits<float>::quiet_NaN());

        CHECK(f1 == f2);
        CHECK(f1 != f3);
        CHECK(f1 > f3);
        CHECK(f3 < f1);
        CHECK(f1 >= f2);
        CHECK(f3 <= f1);
        CHECK(pz == nz);
        CHECK_FALSE(pz != nz);
        CHECK(nz.raw() == 0x8000);
        CHECK_FALSE(nan == nan);
        CHECK(nan != nan);
    }

    SUBCASE("Arithmetic Operators")
    {
        float16_t f1(3.14f);
        float16_t f2(1.0f);

        CHECK((f1 + f2).toFloat32() == doctest::Approx(4.140625f));
        CHECK((f1 - f2).toFloat32() == doctest::Approx(2.140625f));
        CHECK((f1 * f2).toFloat32() == doctest::Approx(3.140625f));
        CHECK((f1 / f2).toFloat32() == doctest::Approx(3.140625f));

        f1 += f2;
        CHECK(f1.toFloat32() == doctest::Approx(4.140625f));

        f1 -= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.140625f));

        f1 *= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.140625f));

        f1 /= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Unary Operators")
    {
        float16_t f1(3.14f);
        float16_t f2 = -f1;
        CHECK(f2.raw() == 0xC248);
        CHECK(f2.toFloat32() == doctest::Approx(-3.140625f));
    }

    SUBCASE("Increment and Decrement Operators")
    {
        float16_t f1(1.0f);
        
        f1++;
        CHECK(f1.toFloat32() == doctest::Approx(2.0f));
        
        ++f1;
        CHECK(f1.toFloat32() == doctest::Approx(3.0f));

        f1--;
        CHECK(f1.toFloat32() == doctest::Approx(2.0f));

        --f1;
        CHECK(f1.toFloat32() == doctest::Approx(1.0f));

        f1 = 1.0f;
        float16_t f2(2.0f);
        auto f = f1 + f2++;
        CHECK(f.toFloat32() == doctest::Approx(3.0f));

        f = f1 + ++f2;
        CHECK(f.toFloat32() == doctest::Approx(5.0f));

        f1 = 1.0f;
        f2 = 2.0f;
        f = f1 + f2--;
        CHECK(f.toFloat32() == doctest::Approx(3.0f));

        f = f1 + --f2;
        CHECK(f.toFloat32() == doctest::Approx(1.0f));
    }

    SUBCASE("Special Values")
    {
        float16_t inf(std::numeric_limits<float>::infinity());
        float16_t ninf(-std::numeric_limits<float>::infinity());
        float16_t nan(std::numeric_limits<float>::quiet_NaN());
        auto posSignalingNaN = std::bit_cast<float>(uint32_t{0x7F800001u});
        auto negSignalingNaN = std::bit_cast<float>(uint32_t{0xFF800001u});
        float16_t posNan(posSignalingNaN);
        float16_t negNan(negSignalingNaN);

        CHECK(inf.raw() == 0x7C00);
        CHECK(ninf.raw() == 0xFC00);
        CHECK(inf.toFloat32()  == std::numeric_limits<float>::infinity());
        CHECK(ninf.toFloat32() == -std::numeric_limits<float>::infinity());
        CHECK(std::isnan(nan.toFloat32()));
        CHECK(std::isnan(posNan.toFloat32()));
        CHECK(std::isnan(negNan.toFloat32()));
        CHECK_FALSE(std::isinf(posNan.toFloat32()));
        CHECK_FALSE(std::isinf(negNan.toFloat32()));
    }

    SUBCASE("Conversion to and from float")
    {
        float16_t f1(3.14f);
        float f2 = f1.toFloat32();
        CHECK(f2 == doctest::Approx(3.140625f));

        float16_t f3 = float16_t{3.14f};
        CHECK(f3.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Edge Cases")
    {
        float16_t tie(1.00146484375f);
        float16_t maxFinite(65519.0f);
        float16_t minSubnormal(5.960464477539063e-08f);
        float16_t maxSubnormal(6.097555160522461e-05f);

        CHECK(tie.raw() == 0x3C02);
        CHECK(tie.toFloat32() == doctest::Approx(1.001953125f));
        CHECK(maxFinite.raw() == 0x7BFF);
        CHECK(maxFinite.toFloat32() == doctest::Approx(65504.0f));
        CHECK(minSubnormal.raw() == 0x0001);
        CHECK(minSubnormal.toFloat32() == doctest::Approx(5.960464477539063e-08f));
        CHECK(maxSubnormal.raw() == 0x03FF);
        CHECK(maxSubnormal.toFloat32() == doctest::Approx(6.097555160522461e-05f));
        CHECK(float16_t::lowest().raw() == 0xFBFF);
        CHECK(std::numeric_limits<float16_t>::lowest().raw() == 0xFBFF);
    }
}


TEST_CASE("bfloat16_t Tests")
{
    SUBCASE("Parameterized Constructor")
    {
        bfloat16_t f1(3.14f);
        CHECK(f1.toFloat32() == doctest::Approx(3.140625f));

        bfloat16_t f2(-3.14f);
        CHECK(f2.toFloat32() == doctest::Approx(-3.140625f));

        bfloat16_t f3(0.0f);
        CHECK(f3.toFloat32() == doctest::Approx(0.0f));

        bfloat16_t f4;
        CHECK(f4.toFloat32() == doctest::Approx(0.0f));
    }

    SUBCASE("Copy Constructor")
    {
        bfloat16_t f1(3.14f);
        bfloat16_t f2(f1);
        CHECK(f2.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Assignment Operator")
    {
        bfloat16_t f1(3.14f);
        bfloat16_t f2;
        f2 = f1;
        CHECK(f2.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Comparison Operators")
    {
        bfloat16_t f1(3.140625f);
        bfloat16_t f2(3.140625f);
        bfloat16_t f3(1.0f);
        bfloat16_t pz(0.0f);
        bfloat16_t nz(-0.0f);
        bfloat16_t nan(std::numeric_limits<float>::quiet_NaN());

        CHECK(f1 == f2);
        CHECK(f1 != f3);
        CHECK(f1 > f3);
        CHECK(f3 < f1);
        CHECK(f1 >= f2);
        CHECK(f3 <= f1);
        CHECK(pz == nz);
        CHECK_FALSE(pz != nz);
        CHECK_FALSE(nan == nan);
        CHECK(nan != nan);
    }

    SUBCASE("Arithmetic Operators")
    {
        bfloat16_t f1(3.14f);
        bfloat16_t f2(1.0f);

        CHECK((f1 + f2).toFloat32() == doctest::Approx(4.125f));
        CHECK((f1 - f2).toFloat32() == doctest::Approx(2.140625f));
        CHECK((f1 * f2).toFloat32() == doctest::Approx(3.140625f));
        CHECK((f1 / f2).toFloat32() == doctest::Approx(3.140625f));

        f1 += f2;
        CHECK(f1.toFloat32() == doctest::Approx(4.125f));

        f1 -= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.125f));

        f1 *= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.125f));

        f1 /= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.125f));
    }

    SUBCASE("Unary Operators")
    {
        bfloat16_t f1(3.14f);
        bfloat16_t f2 = -f1;
        CHECK(f2.toFloat32() == doctest::Approx(-3.140625f));
    }

    SUBCASE("Increment and Decrement Operators")
    {
        bfloat16_t f1(1.0f);

        f1++;
        CHECK(f1.toFloat32() == doctest::Approx(2.0f));

        ++f1;
        CHECK(f1.toFloat32() == doctest::Approx(3.0f));

        f1--;
        CHECK(f1.toFloat32() == doctest::Approx(2.0f));

        --f1;
        CHECK(f1.toFloat32() == doctest::Approx(1.0f));

        f1 = 1.0f;
        bfloat16_t f2(2.0f);
        auto f = f1 + f2++;
        CHECK(f.toFloat32() == doctest::Approx(3.0f));

        f = f1 + ++f2;
        CHECK(f.toFloat32() == doctest::Approx(5.0f));

        f1 = 1.0f;
        f2 = 2.0f;
        f = f1 + f2--;
        CHECK(f.toFloat32() == doctest::Approx(3.0f));

        f = f1 + --f2;
        CHECK(f.toFloat32() == doctest::Approx(1.0f));
    }

    SUBCASE("Special Values")
    {
        bfloat16_t inf(std::numeric_limits<float>::infinity());
        bfloat16_t ninf(-std::numeric_limits<float>::infinity());
        bfloat16_t nan(std::numeric_limits<float>::quiet_NaN());
        auto trickyNaN = std::bit_cast<float>(uint32_t{0x7FFF8001u});
        bfloat16_t roundedNaN(trickyNaN);

        CHECK(inf.toFloat32()  == std::numeric_limits<float>::infinity());
        CHECK(ninf.toFloat32() == -std::numeric_limits<float>::infinity());
        CHECK(std::isnan(nan.toFloat32()));
        CHECK(std::isnan(roundedNaN.toFloat32()));
        CHECK_FALSE(std::isinf(roundedNaN.toFloat32()));
    }

    SUBCASE("Conversion to and from float")
    {
        bfloat16_t f1(3.14f);
        float f2 = f1.toFloat32();
        CHECK(f2 == doctest::Approx(3.140625f));

        bfloat16_t f3 = bfloat16_t{3.14f};
        CHECK(f3.toFloat32() == doctest::Approx(3.140625f));
    }

    SUBCASE("Edge Cases")
    {
        bfloat16_t f1(-1);
        float f2 = f1.toFloat32();
        CHECK(f2 == doctest::Approx(-1));

        CHECK(bfloat16_t::lowest().raw() == 0xFF7F);
        CHECK(std::numeric_limits<bfloat16_t>::lowest().raw() == 0xFF7F);
    }
}
