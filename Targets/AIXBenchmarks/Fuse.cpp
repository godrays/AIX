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
// FUSE CHAIN CONTIGUOUS
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorFuseChainContig : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_t1 = aix::randn({1, elementCount}, opt);
        m_t2 = aix::randn({1, elementCount}, opt);
        m_t3 = aix::randn({1, elementCount}, opt);
        m_t4 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        auto t = (((((m_t1 + m_t2) - m_t3) * ((m_t1 + m_t2) - m_t3)) + 1.0f) / (m_t4 + 10.0f)).log().exp().sin().tanh();
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_t1, m_t2, m_t3, m_t4;
    std::unique_ptr<aix::Device>  m_device;
};

// --------------------------------------------------------------------------------
// FUSE DUAL CHAIN CONTIGUOUS
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorFuseDualChainContig : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_a1 = aix::randn({1, elementCount}, opt);
        m_a2 = aix::randn({1, elementCount}, opt);
        m_a3 = aix::randn({1, elementCount}, opt);
        m_a4 = aix::randn({1, elementCount}, opt);
        m_b1 = aix::randn({1, elementCount}, opt);
        m_b2 = aix::randn({1, elementCount}, opt);
        m_b3 = aix::randn({1, elementCount}, opt);
        m_b4 = aix::randn({1, elementCount}, opt);
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        auto t1 = (((((m_a1 + m_a2) - m_a3) * ((m_a1 + m_a2) - m_a3)) + 1.0f) / (m_a4 + 10.0f)).log().exp().sin().tanh();
        auto t2 = (((((m_b1 + m_b2) - m_b3) * ((m_b1 + m_b2) - m_b3)) + 1.0f) / (m_b4 + 10.0f)).log().exp().sin().tanh();
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_a1, m_a2, m_a3, m_a4;
    aix::Tensor  m_b1, m_b2, m_b3, m_b4;
    std::unique_ptr<aix::Device>  m_device;
};

// --------------------------------------------------------------------------------
// FUSE CHAIN STRIDED
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t rows, size_t cols>
class BenchmarkTensorFuseChainStrided : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };
        m_transposed = aix::randn({cols, rows}, opt).transpose(0, 1);
        m_sliced = aix::randn({rows, cols * 2}, opt).slice(1, 0, static_cast<ssize_t>(cols * 2), 2);
        m_broadcasted = aix::randn({rows, 1}, opt).broadcastTo({rows, cols});
        m_divisor = aix::randn({rows, 1}, opt).broadcastTo({rows, cols});
        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        auto t = (((((m_transposed + m_sliced) - m_broadcasted) * (((m_transposed + m_sliced) - m_broadcasted) + 1.0f)) + 1.0f)
                  / (m_divisor + 10.0f)).log().exp().sin().tanh();
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::Tensor  m_transposed, m_sliced, m_broadcasted, m_divisor;
    std::unique_ptr<aix::Device>  m_device;
};

// --------------------------------------------------------------------------------
// FUSE ADAM OPTIMIZER STEP (4 PARAMETERS)
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t elementCount>
class BenchmarkTensorFuseAdamF324P : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        m_device = aix::createDevice(configs.deviceType);
        aix::TensorOptions opt = { .m_dtype=dataType, .m_device=m_device.get() };

        constexpr size_t kNumParams = 4;
        m_params.resize(kNumParams);
        m_m.resize(kNumParams);
        m_v.resize(kNumParams);
        m_grads.resize(kNumParams);

        for (size_t i = 0; i < kNumParams; ++i)
        {
            m_params[i] = aix::randn({1, elementCount}, opt);
            m_grads[i] = aix::randn({1, elementCount}, opt);
            m_m[i] = aix::TensorValue(0.0f, {1, elementCount}, m_device.get(), dataType);
            m_v[i] = aix::TensorValue(0.0f, {1, elementCount}, m_device.get(), dataType);
        }

        m_beta1 = 0.9f;
        m_beta2 = 0.999f;
        m_lr = 0.001f;
        m_epsilon = 1e-8f;
        m_timestep = 1;

        m_device->synchronize();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        for (size_t i = 0; i < m_params.size(); ++i)
        {
            aix::TensorValue & grad = m_grads[i].value();
            aix::TensorValue & m = m_m[i];
            aix::TensorValue & v = m_v[i];

            m = m_beta1 * m + float(1.0 - m_beta1) * grad;
            v = m_beta2 * v + float(1.0 - m_beta2) * grad * grad;

            aix::TensorValue mHat = m / (1.0f - std::pow(m_beta1, m_timestep));
            aix::TensorValue vHat = v / (1.0f - std::pow(m_beta2, m_timestep));

            m_params[i].value() -= (m_lr * mHat / (vHat.sqrt() + m_epsilon));
        }

        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    std::vector<aix::Tensor>             m_params;
    std::vector<aix::Tensor>             m_grads;
    std::vector<aix::TensorValue>        m_m;
    std::vector<aix::TensorValue>        m_v;
    float                           m_beta1{};
    float                           m_beta2{};
    float                           m_lr{};
    float                           m_epsilon{};
    size_t                          m_timestep{};
    std::unique_ptr<aix::Device>    m_device;
};

using BenchmarkTensorFuseChainContigF3210M = BenchmarkTensorFuseChainContig<aix::DataType::kFloat32, 10000000>;
using BenchmarkTensorFuseDualChainContigF3210M = BenchmarkTensorFuseDualChainContig<aix::DataType::kFloat32, 10000000>;
using BenchmarkTensorFuseChainStridedF324M = BenchmarkTensorFuseChainStrided<aix::DataType::kFloat32, 2048, 2048>;
using BenchmarkTensorFuseAdamF3210M4P = BenchmarkTensorFuseAdamF324P<aix::DataType::kFloat32, 10000000>;

BENCHMARK(BenchmarkTensorFuseChainContigF3210M, "tensor_fuse_chain_contig_f32_10m")
BENCHMARK(BenchmarkTensorFuseDualChainContigF3210M, "tensor_fuse_dual_chain_contig_f32_10m")
BENCHMARK(BenchmarkTensorFuseChainStridedF324M, "tensor_fuse_chain_strided_f32_4m")
BENCHMARK(BenchmarkTensorFuseAdamF3210M4P, "tensor_fuse_adam_f32_10m_4p")
