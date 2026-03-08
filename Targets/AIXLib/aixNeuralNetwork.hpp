//
//  Copyright © 2024-Present, Arkin Terli. All rights reserved.
//
//  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
//  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
//  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
//  trade secret or copyright law. Dissemination of this information or reproduction of this
//  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

#pragma once

// Project includes
#include "aixCore.hpp"
// External includes
// System includes


// Neural Network Namespace
namespace aix::nn
{

class Module
{
public:
    virtual ~Module() = default;

    virtual Tensor forward(Tensor x) const = 0;

    void registerParameter(const std::string & paramName, Tensor & tensor)
    {
        m_parameters.emplace_back(paramName, tensor);
    }

    void registerModule(const Module & module)
    {
        for (const auto & [paramName, param] : module.parameters())
        {
            m_parameters.emplace_back(paramName, param);
        }
    }

    std::vector<std::pair<std::string, Tensor>> parameters() const
    {
        return m_parameters;
    }

    // Returns the total number of elements (learnable parameters) in each Tensor.
    size_t learnableParameters() const
    {
        size_t totalParams = 0;
        for (const auto & [paramName, param] : m_parameters)
        {
            if (param.isRequireGrad())
            {
                totalParams += param.value().size();
            }
        }
        return totalParams;
    }

    void to(std::unique_ptr<Device> & device) const   { to(*device); }
    void to(Device * device) const                    { to(*device); }
    void to(Device & device) const
    {
        for (auto & [paramName, param] : parameters())
        {
            param.value() = param.value().to(&device);
            param.grad() = param.grad().to(&device);
        }
    }

    void to(DataType newDtype) const
    {
        for (auto & [paramName, param] : parameters())
        {
            if (param.isRequireGrad())
            {
                param.value() = param.value().to(newDtype);
                param.grad() = param.grad().to(newDtype);
            }
        }
    }

private:
    std::vector<std::pair<std::string, Tensor>> m_parameters;
};


class Sequential : public Module
{
public:
    // Override the forward function.
    Tensor forward(Tensor x) const override
    {
        for (const auto & module : m_modules)
        {
            x = module->forward(x);
        }
        return x;
    }

    // Function to add modules dynamically if needed.
    void add(Module * module)
    {
        registerModule(*module);
        m_modules.emplace_back(module);     // Use std::unique_ptr to take ownership of the module pointer.
    }

protected:
    // Use std::unique_ptr for polymorphic containment.
    std::vector<std::unique_ptr<Module>> m_modules;
};


class Linear : public Module
{
public:
    // Constructor
    Linear() = default;

    // Constructor
    Linear(size_t numInputs, size_t numOutputs)
    {
        m_w = randn({numInputs, numOutputs}, { .m_requireGrad=true });
        m_b = randn({1, numOutputs}, { .m_requireGrad=true });

        // Register learnable parameters.
        registerParameter("w", m_w);
        registerParameter("b", m_b);
    }

    // Forward
    Tensor forward(Tensor x) const override
    {
        return matmul(x, m_w) + m_b;
    }

    Tensor m_w;
    Tensor m_b;
};


class Tanh : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return tanh(x);
    }
};


class Sigmoid : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 1 / (1 + exp(-x));
    }
};


class Softmax : public Module
{
public:
    // Constructor.
    explicit Softmax(ssize_t dim=0, bool keepDim=false) : m_dim{dim}, m_keepDim{keepDim} { }

    Tensor forward(Tensor x) const override
    {
        x = (x - x.max(m_dim, m_keepDim)).exp();
        return x / x.sum(m_dim, m_keepDim);
    }

private:
    ssize_t m_dim{0};
    bool m_keepDim{false};
};


class LogSoftmax : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        // LogSoftmax(x) = log(e^x / sum(e^x)) = log(e^x) - log(sum(e^x)) = x - log(sum(e^x))
        return x - x.exp().sum().log();
    }
};


class GeLU : public Module
{
public:
    // Forward
    Tensor forward(Tensor x) const override
    {
        return 0.5 * x * (1.0 + tanh(std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x.pow(3))));
    }
};


class MSELoss
{
public:
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        auto diff = predictions - targets;
        auto loss = mean(diff * diff);
        return loss;
    }
};


class BinaryCrossEntropyLoss
{
public:
    // Prediction values must be in [0..1] range.
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        return -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions));
    }
};


class CrossEntropyLoss
{
public:
    // Prediction values must be in [0..1] range. Targets must be (one-shot).
    Tensor operator()(const Tensor & predictions, const Tensor & targets)
    {
        return -mean(targets * log(predictions));
    }
};

// Auxiliary Features

inline void save(const nn::Module & module, const std::string & filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::ios_base::failure("Failed to open file for writing.");
    }

    const auto params = module.parameters();
    for (const auto& [paramName, param] : params)
    {
        const auto & value = param.value();
        size_t size = value.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));                       // Save parameter size
        size_t paramDTypeSize = Device::dataTypeSize(param.dataType());
        ofs.write(reinterpret_cast<const char*>(value.data()), size * paramDTypeSize);       // Save parameter data
    }

    ofs.close();
}

inline void load(nn::Module & module, const std::string & filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::ios_base::failure("Failed to open model parameter file for reading.");
    }

    auto params = module.parameters();    // Get model parameters.
    for (auto& [paramName, param] : params)
    {
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size));         // Read size of parameter
        if (size != param.value().size())
        {
            throw std::runtime_error("Invalid parameter size found when loading the model.");
        }
        size_t paramDTypeSize = Device::dataTypeSize(param.dataType());
        ifs.read(reinterpret_cast<char*>(param.value().data()), size * paramDTypeSize); // Read the parameter data
    }

    ifs.close();
}

}   // aix::nn namespace
