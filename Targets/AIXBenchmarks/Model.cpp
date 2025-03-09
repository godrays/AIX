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
// MODEL XOR
// --------------------------------------------------------------------------------

template<aix::DataType dataType, size_t layerSize>
class BenchmarkModelXOR : public BenchmarkBase
{
public:
    void setup(const AIXBenchmarkConfigs& configs) final
    {
        constexpr int kNumSamples  = 4;
        constexpr int kNumInputs   = 2;
        constexpr int kNumTargets  = 1;
        constexpr float kLearningRate  = 0.01f;

        // Create a device that uses Apple Metal for GPU computations.
        m_device = aix::createDevice(configs.deviceType);

        m_model = aix::nn::Sequential();
        m_model.add(new aix::nn::Linear(kNumInputs, layerSize));
        m_model.add(new aix::nn::Tanh());
        m_model.add(new aix::nn::Linear(layerSize, layerSize));
        m_model.add(new aix::nn::Tanh());
        m_model.add(new aix::nn::Linear(layerSize, kNumTargets));

        m_model.to(m_device);
        m_model.to(dataType);

        // Example inputs and targets for demonstration purposes.
        m_inputs = aix::tensor({0.0, 0.0,
                                0.0, 1.0,
                                1.0, 0.0,
                                1.0, 1.0}, {kNumSamples, kNumInputs}).to(m_device).to(dataType);

        m_targets = aix::tensor({0.0,
                                 1.0,
                                 1.0,
                                 0.0}, {kNumSamples, kNumTargets}).to(m_device).to(dataType);

        // Define an optimizer and a loss function.
        m_optimizer = aix::optim::Adam(m_model.parameters(), kLearningRate);
        m_lossFunc = aix::nn::MSELoss();
    }

    void run(const AIXBenchmarkConfigs&) final
    {
        auto predictions = m_model.forward(m_inputs);
        auto loss = m_lossFunc(predictions, m_targets);
        m_optimizer.zeroGrad();
        loss.backward();
        m_optimizer.step();
        m_device->synchronize();
    }

    void cleanUp() final
    {
        m_device.release();
        m_device = nullptr;
    }

private:
    aix::nn::Sequential  m_model;
    aix::optim::Adam  m_optimizer;
    aix::nn::MSELoss  m_lossFunc;
    aix::Tensor  m_inputs;
    aix::Tensor  m_targets;
    std::unique_ptr<aix::Device>  m_device;
};

using BenchmarkModelXORF321K = BenchmarkModelXOR<aix::DataType::kFloat32, 1000>;
using BenchmarkModelXORF3210 = BenchmarkModelXOR<aix::DataType::kFloat32, 10>;

BENCHMARK(BenchmarkModelXORF321K,  "model_xor_f32_1k")
BENCHMARK(BenchmarkModelXORF3210, "model_xor_f32_10")
