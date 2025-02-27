#pragma once

#include <iostream>
#include <nn/layer.hpp>
#include <stdexcept>
#include <random>

class NeuralNetwork
{
public:
    NeuralNetwork(uint32_t input_size, ActivationFunction input_activation, uint32_t output_size, ActivationFunction output_activation);
    ~NeuralNetwork();

    void add_hidden_layer(uint32_t size, ActivationFunction activation);
    bool validate();
    bool validate(float moyenne, float ecart_type);
    bool mutate(float mutation_rate, float mutation_range);
    bool mutate(float mutation_rate, float mutation_range, float neuron_addition_rate, float layer_addition_rate);

    uint32_t get_input_size() { return this->input_layer.biases.get_cols(); }
    uint32_t get_output_size() { return this->output_layer.biases.get_cols(); }

    uint32_t get_hidden_layers_count() { return this->hidden_layers.size(); }
    uint32_t get_hidden_layer_size(uint32_t index) { return this->hidden_layers[index].biases.get_cols(); }

    uint32_t get_number_of_trainable_parameters();

    std::vector<float> forward(std::vector<float> input);

private:
    Layer input_layer;
    Layer output_layer;
    std::vector<Layer> hidden_layers;
    bool valid = false;
};