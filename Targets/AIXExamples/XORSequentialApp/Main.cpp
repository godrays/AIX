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

// This application shows how to use sequential module to build a neural network with other modules easily.

int main()
{
    constexpr int kNumSamples  = 4;
    constexpr int kNumInputs   = 2;
    constexpr int kNumTargets  = 1;
    constexpr int kNumEpochs   = 1000;
    constexpr int kLogInterval = 100;
    constexpr float kLearningRate  = 0.02f;
    constexpr float kLossThreshold = 1e-5f;

    aix::Device  cpuDevice;    // aix framework can still work without device creation.

    aix::nn::Sequential  model;
    model.add(new aix::nn::Linear(kNumInputs, 8));
    model.add(new aix::nn::Tanh());
    model.add(new aix::nn::Linear(8, 4));
    model.add(new aix::nn::Tanh());
    model.add(new aix::nn::Linear(4, kNumTargets));

    model.to(cpuDevice);

    std::cout << "Total parameters: " << model.learnableParameters() << std::endl;

    // Example inputs and targets for demonstration purposes.
    auto inputs  = aix::tensor({0.0, 0.0,
                                0.0, 1.0,
                                1.0, 0.0,
                                1.0, 1.0}, {kNumSamples, kNumInputs}).to(cpuDevice);

    auto targets = aix::tensor({0.0,
                                1.0,
                                1.0,
                                0.0}, {kNumSamples, kNumTargets}).to(cpuDevice);

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
            std::cout << "Epoch: " << epoch << " Loss = " << loss.value().item<float>()<< std::endl << std::flush;

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
