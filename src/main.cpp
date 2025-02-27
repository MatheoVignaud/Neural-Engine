#include <chrono>
#include <iostream>
#include <nn/network.hpp>

const uint16_t iterations = 1;
const uint16_t N = 1000;

int main(int argc, char *argv[])
{
    NeuralNetwork nn(N, ActivationFunction::RELU, N, ActivationFunction::RELU);
    nn.add_hidden_layer(N, ActivationFunction::RELU);
    nn.validate(0.0, 0.5);

    return 0;
}