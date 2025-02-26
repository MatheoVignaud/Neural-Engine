#pragma once

#include <matrix/matrix.hpp>
#include <nn/activation.hpp>

enum LayerType
{
    INPUT,
    HIDDEN,
    OUTPUT,
};

struct Layer
{
    Matrix weights;
    Matrix biases;
    ActivationFunction activation;
    LayerType type;
};
