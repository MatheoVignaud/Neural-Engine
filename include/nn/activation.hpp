#pragma once

#include <cmath>

enum ActivationFunction
{
    LINEAR,
    RELU,
    SIGMOID,
    TANH
};

inline void relu(float &x)
{
    x = x > 0 ? x : 0;
}

inline void sigmoid(float &x)
{
    x = 1 / (1 + exp(-x));
}

inline void tanh(float &x)
{
    x = std::tanh(x);
}