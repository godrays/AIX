//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

// Project includes
#include "Common.hpp"
// External includes
// System includes


// --------------------------------------------------------------------------------
// ADD
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceAdd : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->add(m_t1.value().deviceParams(), m_t2.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceAddF3210M = BenchmarkDeviceAdd<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceAddF3210M, "device_add_f32_10m")

// --------------------------------------------------------------------------------
// SUB
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSub : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->sub(m_t1.value().deviceParams(), m_t2.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSubF3210M = BenchmarkDeviceSub<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSubF3210M, "device_sub_f32_10m")

// --------------------------------------------------------------------------------
// MUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceMul : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->mul(m_t1.value().deviceParams(), m_t2.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMulF3210M = BenchmarkDeviceMul<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceMulF3210M, "device_mul_f32_10m")

// --------------------------------------------------------------------------------
// DIV
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceDiv : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->div(m_t1.value().deviceParams(), m_t2.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceDivF3210M = BenchmarkDeviceDiv<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceDivF3210M, "device_div_f32_10m")

// --------------------------------------------------------------------------------
// UNARY
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceUnary : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->unary(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceUnaryF3210M = BenchmarkDeviceUnary<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceUnaryF3210M, "device_unary_f32_10m")

// --------------------------------------------------------------------------------
// FILL
// --------------------------------------------------------------------------------

template<aix::DataType scalarDataType, aix::DataType dstDataType, size_t elementCount>
class BenchmarkDeviceFill : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dstDataType, .m_device=m_device.get() };
        m_result = aix::Tensor(aix::Shape{elementCount}, opt);  // No fill
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        double scalar = 5;  // Type and accuracy do not matter here.
        m_device->fill(&scalar, scalarDataType, m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceFillF32F3210M = BenchmarkDeviceFill<aix::DataType::kFloat32, aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceFillF32F3210M, "device_fill_f32_f32_10m")

// --------------------------------------------------------------------------------
// SUM
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSum : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->sum(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSumF3210M = BenchmarkDeviceSum<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSumF3210M, "device_sum_f32_10m")

// --------------------------------------------------------------------------------
// MAX
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceMax : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->max(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMaxF3210M = BenchmarkDeviceMax<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceMaxF3210M, "device_max_f32_10m")

// --------------------------------------------------------------------------------
// SQRT
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSqrt : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->sqrt(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSqrtF3210M = BenchmarkDeviceSqrt<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSqrtF3210M, "device_sqrt_f32_10m")

// --------------------------------------------------------------------------------
// SIN
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceSin : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->sin(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceSinF3210M = BenchmarkDeviceSin<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceSinF3210M, "device_sin_f32_10m")

// --------------------------------------------------------------------------------
// COS
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceCos : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->cos(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceCosF3210M = BenchmarkDeviceCos<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceCosF3210M, "device_cos_f32_10m")

// --------------------------------------------------------------------------------
// TANH
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceTanh : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->tanh(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceTanhF3210M = BenchmarkDeviceTanh<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceTanhF3210M, "device_tanh_f32_10m")

// --------------------------------------------------------------------------------
// LOG
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceLog : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->log(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceLogF3210M = BenchmarkDeviceLog<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceLogF3210M, "device_log_f32_10m")

// --------------------------------------------------------------------------------
// EXP
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDeviceExp : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->exp(m_t.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceExpF3210M = BenchmarkDeviceExp<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceExpF3210M, "device_exp_f32_10m")

// --------------------------------------------------------------------------------
// POW
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkDevicePow : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({1, elementCount}, opt);
        m_exp = 2 + aix::randn({1, elementCount}, opt);
        m_result = aix::Tensor(aix::Shape{1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->pow(m_t.value().deviceParams(), m_exp.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_exp, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDevicePowF3210M = BenchmarkDevicePow<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDevicePowF3210M, "device_pow_f32_10m")

// --------------------------------------------------------------------------------
// MATMUL
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkDeviceMatMul : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({M, M}, opt);
        m_t2 = aix::randn({M, M}, opt);
        m_result = aix::Tensor(aix::Shape{M, M}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->matmul(m_t1.value().deviceParams(), m_t2.value().deviceParams(), m_result.value().deviceParams());
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceMatMulF3210   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,10>;
using BenchmarkDeviceMatMulF3225   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,25>;
using BenchmarkDeviceMatMulF3250   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,50>;
using BenchmarkDeviceMatMulF32100  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,100>;
using BenchmarkDeviceMatMulF32200  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,200>;
using BenchmarkDeviceMatMulF32500  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,500>;
using BenchmarkDeviceMatMulF32800  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,800>;
using BenchmarkDeviceMatMulF321000 = BenchmarkDeviceMatMul<aix::DataType::kFloat32,1000>;
using BenchmarkDeviceMatMulF322000 = BenchmarkDeviceMatMul<aix::DataType::kFloat32,2000>;

using BenchmarkDeviceMatMulF3216   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,16>;
using BenchmarkDeviceMatMulF3232   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,32>;
using BenchmarkDeviceMatMulF3264   = BenchmarkDeviceMatMul<aix::DataType::kFloat32,64>;
using BenchmarkDeviceMatMulF32128  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,128>;
using BenchmarkDeviceMatMulF32256  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,256>;
using BenchmarkDeviceMatMulF32512  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,512>;
using BenchmarkDeviceMatMulF32768  = BenchmarkDeviceMatMul<aix::DataType::kFloat32,768>;
using BenchmarkDeviceMatMulF321024 = BenchmarkDeviceMatMul<aix::DataType::kFloat32,1024>;
using BenchmarkDeviceMatMulF322048 = BenchmarkDeviceMatMul<aix::DataType::kFloat32,2048>;

BENCHMARK(BenchmarkDeviceMatMulF3210  , "device_matmul_f32_10")
BENCHMARK(BenchmarkDeviceMatMulF3225  , "device_matmul_f32_25")
BENCHMARK(BenchmarkDeviceMatMulF3250  , "device_matmul_f32_50")
BENCHMARK(BenchmarkDeviceMatMulF32100 , "device_matmul_f32_100")
BENCHMARK(BenchmarkDeviceMatMulF32200 , "device_matmul_f32_200")
BENCHMARK(BenchmarkDeviceMatMulF32500 , "device_matmul_f32_500")
BENCHMARK(BenchmarkDeviceMatMulF32800 , "device_matmul_f32_800")
BENCHMARK(BenchmarkDeviceMatMulF321000, "device_matmul_f32_1000")
BENCHMARK(BenchmarkDeviceMatMulF322000, "device_matmul_f32_2000")

BENCHMARK(BenchmarkDeviceMatMulF3216  , "device_matmul_f32_16")
BENCHMARK(BenchmarkDeviceMatMulF3232  , "device_matmul_f32_32")
BENCHMARK(BenchmarkDeviceMatMulF3264  , "device_matmul_f32_64")
BENCHMARK(BenchmarkDeviceMatMulF32128 , "device_matmul_f32_128")
BENCHMARK(BenchmarkDeviceMatMulF32256 , "device_matmul_f32_256")
BENCHMARK(BenchmarkDeviceMatMulF32512 , "device_matmul_f32_512")
BENCHMARK(BenchmarkDeviceMatMulF32768 , "device_matmul_f32_768")
BENCHMARK(BenchmarkDeviceMatMulF321024, "device_matmul_f32_1024")
BENCHMARK(BenchmarkDeviceMatMulF322048, "device_matmul_f32_2048")

// --------------------------------------------------------------------------------
// TRANSPOSE
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t M>
class BenchmarkDeviceTranspose : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t = aix::randn({M, M}, opt);
        m_result = aix::Tensor(aix::Shape{M, M}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->transpose(m_t.value().deviceParams(), m_result.value().deviceParams(), 0, 1);
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t, m_result;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceTransposeF325K = BenchmarkDeviceTranspose<aix::DataType::kFloat32,5000>;

BENCHMARK(BenchmarkDeviceTransposeF325K, "device_transpose_f32_5k")

using BenchmarkDeviceTransposeF3210   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,10>;
using BenchmarkDeviceTransposeF3225   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,25>;
using BenchmarkDeviceTransposeF3250   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,50>;
using BenchmarkDeviceTransposeF32100  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,100>;
using BenchmarkDeviceTransposeF32200  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,200>;
using BenchmarkDeviceTransposeF32500  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,500>;
using BenchmarkDeviceTransposeF32800  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,800>;
using BenchmarkDeviceTransposeF321000 = BenchmarkDeviceTranspose<aix::DataType::kFloat32,1000>;
using BenchmarkDeviceTransposeF322000 = BenchmarkDeviceTranspose<aix::DataType::kFloat32,2000>;

using BenchmarkDeviceTransposeF3216   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,16>;
using BenchmarkDeviceTransposeF3232   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,32>;
using BenchmarkDeviceTransposeF3264   = BenchmarkDeviceTranspose<aix::DataType::kFloat32,64>;
using BenchmarkDeviceTransposeF32128  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,128>;
using BenchmarkDeviceTransposeF32256  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,256>;
using BenchmarkDeviceTransposeF32512  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,512>;
using BenchmarkDeviceTransposeF32768  = BenchmarkDeviceTranspose<aix::DataType::kFloat32,768>;
using BenchmarkDeviceTransposeF321024 = BenchmarkDeviceTranspose<aix::DataType::kFloat32,1024>;
using BenchmarkDeviceTransposeF322048 = BenchmarkDeviceTranspose<aix::DataType::kFloat32,2048>;

BENCHMARK(BenchmarkDeviceTransposeF3210  , "device_transpose_f32_10")
BENCHMARK(BenchmarkDeviceTransposeF3225  , "device_transpose_f32_25")
BENCHMARK(BenchmarkDeviceTransposeF3250  , "device_transpose_f32_50")
BENCHMARK(BenchmarkDeviceTransposeF32100 , "device_transpose_f32_100")
BENCHMARK(BenchmarkDeviceTransposeF32200 , "device_transpose_f32_200")
BENCHMARK(BenchmarkDeviceTransposeF32500 , "device_transpose_f32_500")
BENCHMARK(BenchmarkDeviceTransposeF32800 , "device_transpose_f32_800")
BENCHMARK(BenchmarkDeviceTransposeF321000, "device_transpose_f32_1000")
BENCHMARK(BenchmarkDeviceTransposeF322000, "device_transpose_f32_2000")

BENCHMARK(BenchmarkDeviceTransposeF3216  , "device_transpose_f32_16")
BENCHMARK(BenchmarkDeviceTransposeF3232  , "device_transpose_f32_32")
BENCHMARK(BenchmarkDeviceTransposeF3264  , "device_transpose_f32_64")
BENCHMARK(BenchmarkDeviceTransposeF32128 , "device_transpose_f32_128")
BENCHMARK(BenchmarkDeviceTransposeF32256 , "device_transpose_f32_256")
BENCHMARK(BenchmarkDeviceTransposeF32512 , "device_transpose_f32_512")
BENCHMARK(BenchmarkDeviceTransposeF32768 , "device_transpose_f32_768")
BENCHMARK(BenchmarkDeviceTransposeF321024, "device_transpose_f32_1024")
BENCHMARK(BenchmarkDeviceTransposeF322048, "device_transpose_f32_2048")


// --------------------------------------------------------------------------------
// COPY
// --------------------------------------------------------------------------------

template<aix::DataType srcDataType, aix::DataType dstDataType, size_t elementCount>
class BenchmarkDeviceCopy : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        m_src = aix::Tensor(aix::Shape{elementCount}, { .m_dtype=srcDataType, .m_device=m_device.get() });  // No fill
        m_dst = aix::Tensor(aix::Shape{elementCount}, { .m_dtype=dstDataType, .m_device=m_device.get() });  // No fill
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        m_device->copy(m_src.value().data(), srcDataType, m_dst.value().data(), dstDataType, elementCount);
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_src, m_dst;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkDeviceCopyF32F3210M = BenchmarkDeviceCopy<aix::DataType::kFloat32, aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkDeviceCopyF32F3210M, "device_copy_f32_f32_10m")
