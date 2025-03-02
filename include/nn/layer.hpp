#pragma once

#include <matrix/matrix.hpp>
#include <nn/activation.hpp>

enum LayerType
{
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER,
};

struct Layer
{
    Matrix weights;
    Matrix biases;
    ActivationFunction activation;
    LayerType type;
};
