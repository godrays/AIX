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


class NeuralNet : public aix::nn::Module
{
public:
    // Constructor
    NeuralNet(size_t numInputs, size_t numOutputs)
    {
        constexpr size_t hlSize = 4;      // Hidden layer size.
        m_w1 = aix::randn({numInputs,  hlSize},     aix::requireGrad(true));
        m_b1 = aix::randn({1,          hlSize},     aix::requireGrad(true));
        m_w2 = aix::randn({hlSize,     numOutputs}, aix::requireGrad(true));
        m_b2 = aix::randn({1,          numOutputs}, aix::requireGrad(true));

        // Register learnable parameters.
        registerParameter("w1", m_w1);
        registerParameter("b1", m_b1);
        registerParameter("w2", m_w2);
        registerParameter("b2", m_b2);
    }

    // Forward
    aix::Tensor forward(aix::Tensor x) const override
    {
        x = aix::tanh(aix::matmul(x, m_w1) + m_b1);
        x = aix::matmul(x, m_w2) + m_b2;
        return x;
    }

private:
    aix::Tensor m_w1, m_b1;
    aix::Tensor m_w2, m_b2;
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
