#pragma once

#include <cmath>

enum ActivationFunction
{
    LINEAR,
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    EXPONENTIAL
};

inline void relu(float &x)
{
    x = std::max(0.0f, x);
}

inline void leaky_relu(float &x)
{
    x = std::max(0.01f * x, x);
}

inline void sigmoid(float &x)
{
    x = 1 / (1 + exp(-x));
}

inline void tanh(float &x)
{
    x = std::tanh(x);
}

inline void exponential(float &x)
{
    x = exp(x);
}