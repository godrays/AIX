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

namespace aix::optim
{

class Optimizer
{
public:
    // Constructor
    Optimizer() = default;

    // Constructor
    explicit Optimizer(const std::vector<std::pair<std::string, Tensor>> & parameters) : m_parameters(parameters) { }

    // Constructor
    explicit Optimizer(const std::vector<Tensor> & parameters)
    {
        for (auto & param : parameters)
        {
            m_parameters.emplace_back("", param);
        }
    }

    // Destructor
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zeroGrad()
    {
        for (auto & [name, param] : m_parameters)
        {
            param.zeroGrad();
        }
    }

    inline void setDataType(DataType dtype)
    {
        if (dtype != DataType::kFloat64 && dtype != DataType::kFloat32 &&
            dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
        {
            throw std::invalid_argument("Optimization has to perform in Float data type to be effective.");
        }
        m_calculationDType = dtype;
    }

protected:
    std::vector<std::pair<std::string, Tensor>> m_parameters;
    DataType m_calculationDType{DataType::kFloat32};
};


class SGD : public Optimizer
{
public:
    SGD() = default;

    explicit SGD(const std::vector<std::pair<std::string, Tensor>> & parameters, float lr = 0.01f)
        : Optimizer(parameters), m_lr(lr) { }

    explicit SGD(const std::vector<Tensor> & parameters, float lr = 0.01f) : Optimizer(parameters), m_lr(lr) { }

    void step() final
    {
        for (auto & [name, param] : m_parameters)
        {
            if (param.isRequireGrad())
            {
                param.value() -= param.grad() * m_lr;   // w' = w - lr * w_gradient.
            }
        }
    }

private:
    float m_lr{0.01f};      // Learning rate
};


class Adam : public Optimizer
{
public:
    Adam() = default;

    explicit Adam(const std::vector<std::pair<std::string, Tensor>> & parameters, float lr = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        initializeParameters();
    }

    explicit Adam(const std::vector<Tensor> & parameters, float lr = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(parameters), m_lr(lr), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
    {
        initializeParameters();
    }

    void step() final
    {
        ++m_timestep;
        for (size_t i = 0; i < m_parameters.size(); ++i)
        {
            auto & [name, parameter] = m_parameters[i];

            if (parameter.isRequireGrad())
            {
                // Convert the parameter's data type to the optimizer's internal calculation type.
                auto gradFloat = parameter.grad().to(m_calculationDType);

                // Update biased first moment estimate.
                m_m[i] = m_beta1 * m_m[i] + float(1.0 - m_beta1) * gradFloat;

                // Update biased second raw moment estimate.
                m_v[i] = m_beta2 * m_v[i] + float(1.0 - m_beta2) * gradFloat * gradFloat;

                // Compute bias-corrected first moment estimate.
                TensorValue mHat = m_m[i] / float(1.0 - std::pow(m_beta1, m_timestep));

                // Compute bias-corrected second raw moment estimate.
                TensorValue vHat = m_v[i] / float(1.0 - std::pow(m_beta2, m_timestep));

                // Update parameter.
                parameter.value() -= (m_lr * mHat / (vHat.sqrt() + m_epsilon)).to(parameter.dataType());
            }
        }
    }

private:
    void initializeParameters()
    {
        for (const auto & [name, param] : m_parameters)
        {
            m_m.emplace_back(0.0f, param.shape(), param.value().device(), m_calculationDType);
            m_v.emplace_back(0.0f, param.shape(), param.value().device(), m_calculationDType);
        }
    }

    float m_lr{0.001f};         // Learning rate.
    float m_beta1{0.9f};        // Exponential decay rate for the first moment estimates.
    float m_beta2{0.999f};      // Exponential decay rate for the second moment estimates.
    float m_epsilon{1e-8f};     // Small constant for numerical stability.
    size_t m_timestep{0};       // Time step.
    std::vector<TensorValue> m_m;   // First moment vector.
    std::vector<TensorValue> m_v;   // Second moment vector.
};

}   // aix::optim namespace
