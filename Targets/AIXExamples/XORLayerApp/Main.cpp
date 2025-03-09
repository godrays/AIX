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
// System includes
#include <iostream>

// This application shows how to use core aix features to build a simple fully connected layer = Linear.
// Exactly the same model of XORApp is now using a smaller Linear module to build a neural net.

class Linear : public aix::nn::Module
{
public:
    Linear() = default;

    // Constructor
    Linear(size_t numInputs, size_t numOutputs)
    {
        m_w = aix::randn({numInputs, numOutputs}, aix::requireGrad(true));   // A tensor filled with random numbers in [-1, 1].
        m_b = aix::randn({1,         numOutputs}, aix::requireGrad(true));

        // Register learnable parameters.
        registerParameter("w", m_w);
        registerParameter("b", m_b);
    }

    // Forward
    aix::Tensor forward(aix::Tensor x) const override
    {
        return aix::matmul(x, m_w) + m_b;
    }

    aix::Tensor  m_w;
    aix::Tensor  m_b;
};


class NeuralNet : public aix::nn::Module
{
public:
    // Constructor
    NeuralNet(size_t numInputs, size_t numOutputs)
    {
        m_fc1 = Linear(numInputs, 4);
        m_fc2 = Linear(4, numOutputs);

        registerModule(m_fc1);
        registerModule(m_fc2);
    }

    // Forward
    aix::Tensor forward(aix::Tensor x) const override
    {
        x = aix::tanh(m_fc1.forward(x));    // Layer 1.
        x = m_fc2.forward(x);               // Layer 2.
        return x;
    }

    Linear  m_fc1;     // Fully connected layer.
    Linear  m_fc2;     // Fully connected layer.
};


int main()
{
    constexpr int kNumSamples  = 4;
    constexpr int kNumInputs   = 2;
    constexpr int kNumTargets  = 1;
    constexpr int kNumEpochs   = 1000;
    constexpr int kLogInterval = 100;
    constexpr float kLearningRate  = 0.02f;
    constexpr float kLossThreshold = 1e-5f;

    // Example inputs and targets for demonstration purposes.
    auto inputs  = aix::tensor({0.0, 0.0,
                                0.0, 1.0,
                                1.0, 0.0,
                                1.0, 1.0}, {kNumSamples, kNumInputs});

    auto targets = aix::tensor({0.0,
                                1.0,
                                1.0,
                                0.0}, {kNumSamples, kNumTargets});

    // Create a model with a single hidden layer.
    NeuralNet model(kNumInputs, kNumTargets);

    // Define a loss function and an optimizer.
    aix::optim::Adam optimizer(model.parameters(), kLearningRate);

    auto lossFunc = aix::nn::MSELoss();
    auto timeStart = std::chrono::steady_clock::now();

    // Training loop.
    size_t epoch;
    for (epoch = 0; epoch < kNumEpochs; ++epoch)
    {
        optimizer.zeroGrad();                               // Zero the gradients before backward pass.

        // Forward step.
        auto predictions = model.forward(inputs);
        auto loss = lossFunc(predictions, targets);         // Loss calculations are still part of computation graph.

        // Backward step.
        loss.backward();                                    // Compute all gradients in the graph.

        // Optimization step.
        optimizer.step();                                   // Update neural net's learnable parameters.

        // Log loss value.
        if (epoch % kLogInterval == 0 || loss.value().item<float>() <= kLossThreshold)
            std::cout << "Epoch: " << epoch << " Loss = " << loss.value().item<float>() << std::endl << std::flush;

        // Stop training process when loss is lower than the threshold.
        if (loss.value().item<float>() <= kLossThreshold)
            break;
    }
    std::cout << std::endl;

    auto timeEnd = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
    std::cout << "Training: " << duration << " ms - Avg Iteration: " << duration/double(epoch) << " ms\n";

    // Final predictions after training the neural network model.
    auto finalPredictions = model.forward(inputs);

    std::cout << "Final Predictions: " << std::endl;
    std::cout << finalPredictions << std::endl;

    return 0;
}
