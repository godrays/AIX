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
        float16_t f1(3.14f);
        CHECK(f1.toFloat32() == doctest::Approx(3.13867f));
        
        float16_t f2(-3.14f);
        CHECK(f2.toFloat32() == doctest::Approx(-3.13867f));
        
        float16_t f3(0.0f);
        CHECK(f3.toFloat32() == doctest::Approx(0.0f));
    }
    
    SUBCASE("Copy Constructor")
    {
        float16_t f1(3.14f);
        float16_t f2(f1);
        CHECK(f2.toFloat32() == doctest::Approx(3.13867f));
    }
    
    SUBCASE("Assignment Operator")
    {
        float16_t f1(3.14f);
        float16_t f2;
        f2 = f1;
        CHECK(f2.toFloat32() == doctest::Approx(3.13867f));
    }
    
    SUBCASE("Comparison Operators")
    {
        float16_t f1(3.14f);
        float16_t f2(3.14f);
        float16_t f3(1.0f);
        
        CHECK(f1 == f2);
        CHECK(f1 != f3);
        CHECK(f1 > f3);
        CHECK(f3 < f1);
        CHECK(f1 >= f2);
        CHECK(f3 <= f1);
    }
    
    SUBCASE("Arithmetic Operators")
    {
        float16_t f1(3.14f);
        float16_t f2(1.0f);
        
        CHECK((f1 + f2).toFloat32() == doctest::Approx(4.13672f));
        CHECK((f1 - f2).toFloat32() == doctest::Approx(2.13867f));
        CHECK((f1 * f2).toFloat32() == doctest::Approx(3.13867f));
        CHECK((f1 / f2).toFloat32() == doctest::Approx(3.13867f));
        
        f1 += f2;
        CHECK(f1.toFloat32() == doctest::Approx(4.13672f));
        
        f1 -= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.13672f));
        
        f1 *= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.13672f));
        
        f1 /= f2;
        CHECK(f1.toFloat32() == doctest::Approx(3.13672f));
    }

    SUBCASE("Unary Operators")
    {
        float16_t f1(3.14f);
        float16_t f2 = -f1;
        CHECK(f2.toFloat32() == doctest::Approx(-3.13867f));
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

        CHECK(inf.toFloat32()  == std::numeric_limits<float>::infinity());
        CHECK(ninf.toFloat32() == -std::numeric_limits<float>::infinity());
        CHECK(std::isnan(nan.toFloat32()));
    }
    
    SUBCASE("Conversion to and from float")
    {
        float16_t f1(3.14f);
        float f2 = f1.toFloat32();
        CHECK(f2 == doctest::Approx(3.13867f));
        
        float16_t f3 = float16_t{3.14f};
        CHECK(f3.toFloat32() == doctest::Approx(3.13867f));
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
