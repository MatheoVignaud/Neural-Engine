#include <nn/network.hpp>

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(uint32_t input_size, uint32_t output_size, ActivationFunction output_activation)
{
    this->input_layer.type = LayerType::INPUT_LAYER;
    this->input_layer.biases = Matrix(1, input_size); // just to know the size of the input

    this->output_layer.type = LayerType::OUTPUT_LAYER;
    this->output_layer.biases = Matrix(1, output_size);
    this->output_layer.activation = output_activation;
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::add_hidden_layer(uint32_t size, ActivationFunction activation)
{
    if (size > 0)
    {
        Layer hidden_layer;
        hidden_layer.type = LayerType::HIDDEN_LAYER;
        hidden_layer.biases = Matrix(1, size);
        hidden_layer.activation = activation;
        this->hidden_layers.push_back(hidden_layer);
    }
    else
    {
        throw std::invalid_argument("Hidden layer size must be greater than 0");
    }
}

bool NeuralNetwork::validate()
{
    if (this->hidden_layers.size() == 0)
    {
        this->valid = true;
        return false;
    }
    // create Matrix for weights
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        if (i == 0)
        {
            if (this->hidden_layers[i].weights.get_rows() == 0 || this->hidden_layers[i].weights.get_cols() == 0)
            {
                this->hidden_layers[i].weights = Matrix(this->input_layer.biases.get_rows(), this->hidden_layers[i].biases.get_rows(), 1);
            }
        }
        else
        {
            if (this->hidden_layers[i].weights.get_rows() == 0 || this->hidden_layers[i].weights.get_cols() == 0)
            {
                this->hidden_layers[i].weights = Matrix(this->hidden_layers[i - 1].biases.get_rows(), this->hidden_layers[i].biases.get_rows(), 1);
            }
        }
    }
    if (this->output_layer.weights.get_rows() == 0 || this->output_layer.weights.get_cols() == 0)
    {
        this->output_layer.weights = Matrix(this->hidden_layers[this->hidden_layers.size() - 1].biases.get_rows(), this->output_layer.biases.get_rows(), 1);
    }

    this->valid = true;
    return true;
}

bool NeuralNetwork::validate(float moyenne, float ecart_type)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(moyenne, ecart_type); // normal distribution

    if (this->hidden_layers.size() == 0)
    {
        this->valid = true;
        return false;
    }
    // create Matrix for weights
    this->input_layer.weights = Matrix(this->hidden_layers[0].biases.get_rows(), this->input_layer.biases.get_cols(), 1);
    for (int i = 0; i < this->input_layer.weights.get_rows(); i++)
    {
        for (int j = 0; j < this->input_layer.weights.get_cols(); j++)
        {
            this->input_layer.weights.set(j, i, distribution(gen));
        }
    }
    for (int i = 0; i < this->input_layer.biases.get_cols(); i++)
    {
        this->input_layer.biases.set(i, 0, distribution(gen));
    }
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        if (i == this->hidden_layers.size() - 1)
        {
            this->hidden_layers[i].weights = Matrix(this->output_layer.biases.get_rows(), this->hidden_layers[i].biases.get_cols(), 1);
        }
        else
        {
            this->hidden_layers[i].weights = Matrix(this->hidden_layers[i + 1].biases.get_rows(), this->hidden_layers[i].biases.get_cols(), 1);
        }
        for (int j = 0; j < this->hidden_layers[i].weights.get_rows(); j++)
        {
            for (int k = 0; k < this->hidden_layers[i].weights.get_cols(); k++)
            {
                this->hidden_layers[i].weights.set(k, j, distribution(gen));
            }
        }
        for (int j = 0; j < this->hidden_layers[i].biases.get_cols(); j++)
        {
            this->hidden_layers[i].biases.set(j, 0, distribution(gen));
        }
    }
    this->valid = true;
    return true;
}

bool NeuralNetwork::mutate(float mutation_rate, float mutation_range)
{
    return this->mutate(mutation_rate, mutation_range, 0, 0);
}

bool NeuralNetwork::mutate(float mutation_rate, float mutation_range, float neuron_addition_rate, float layer_addition_rate)
{
    if (!this->valid)
    {
        std::cout << "Validate the network first" << std::endl;
        return false;
    }

    bool add_neuron = (rand() % 100 < neuron_addition_rate * 100);
    bool add_layer = (rand() % 100 < layer_addition_rate * 100);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0, mutation_range); // normal distribution

    if (add_neuron)
    {
        // choose a random layer
        int layer_index = rand() % (this->hidden_layers.size());
        // add a neuron to the layer
        std::vector<float> new_biases;
        for (int i = 0; i < this->hidden_layers[layer_index].biases.get_cols(); i++)
        {
            new_biases.push_back(distribution(gen));
        }
        std::vector<float> new_weights;
        for (int i = 0; i < this->hidden_layers[layer_index].weights.get_cols(); i++)
        {
            new_weights.push_back(distribution(gen));
        }
        this->hidden_layers[layer_index].biases.add_row(new_biases);
        this->hidden_layers[layer_index].weights.add_row(new_weights);

        // modifiy the weights of the next layer
        if (layer_index < this->hidden_layers.size() - 1)
        {
            std::vector<float> new_weights;
            for (int i = 0; i < this->hidden_layers[layer_index + 1].weights.get_rows(); i++)
            {
                new_weights.push_back(distribution(gen));
            }
            this->hidden_layers[layer_index + 1].weights.add_col(new_weights);
        }
        else
        {
            std::vector<float> new_weights;
            for (int i = 0; i < this->output_layer.weights.get_rows(); i++)
            {
                new_weights.push_back(distribution(gen));
            }
            this->output_layer.weights.add_col(new_weights);
        }
    }

    for (int i = 0; i < this->input_layer.weights.get_rows(); i++)
    {
        for (int j = 0; j < this->input_layer.weights.get_cols(); j++)
        {
            if (rand() % 100 < mutation_rate * 100)
            {
                this->input_layer.weights.set(j, i, this->input_layer.weights.get(j, i) + distribution(gen));
            }
        }
    }

    for (int i = 0; i < this->input_layer.biases.get_cols(); i++)
    {
        if (rand() % 100 < mutation_rate * 100)
        {
            this->input_layer.biases.set(i, 0, this->input_layer.biases.get(i, 0) + distribution(gen));
        }
    }

    // hidden layers
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        for (int j = 0; j < this->hidden_layers[i].weights.get_rows(); j++)
        {
            for (int k = 0; k < this->hidden_layers[i].weights.get_cols(); k++)
            {
                if (rand() % 100 < mutation_rate * 100)
                {
                    this->hidden_layers[i].weights.set(k, j, this->hidden_layers[i].weights.get(k, j) + distribution(gen));
                }
            }
        }
        for (int j = 0; j < this->hidden_layers[i].biases.get_cols(); j++)
        {
            if (rand() % 100 < mutation_rate * 100)
            {
                this->hidden_layers[i].biases.set(j, 0, this->hidden_layers[i].biases.get(j, 0) + distribution(gen));
            }
        }
    }

    // output layer
    for (int i = 0; i < this->output_layer.weights.get_rows(); i++)
    {
        for (int j = 0; j < this->output_layer.weights.get_cols(); j++)
        {
            if (rand() % 100 < mutation_rate * 100)
            {
                this->output_layer.weights.set(j, i, this->output_layer.weights.get(j, i) + distribution(gen));
            }
        }
    }

    for (int i = 0; i < this->output_layer.biases.get_cols(); i++)
    {
        if (rand() % 100 < mutation_rate * 100)
        {
            this->output_layer.biases.set(i, 0, this->output_layer.biases.get(i, 0) + distribution(gen));
        }
    }

    return true;
}

uint32_t NeuralNetwork::get_number_of_trainable_parameters()
{
    if (!this->valid)
    {
        std::cout << "Validate the network first" << std::endl;
        return 0;
    }
    uint32_t total = 0;
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        total += this->hidden_layers[i].weights.get_rows() * this->hidden_layers[i].weights.get_cols();
        total += this->hidden_layers[i].biases.get_rows() * this->hidden_layers[i].biases.get_cols();
    }
    total += this->output_layer.weights.get_rows() * this->output_layer.weights.get_cols();
    total += this->output_layer.biases.get_rows() * this->output_layer.biases.get_cols();
    return total;
}

std::vector<float> NeuralNetwork::forward(std::vector<float> input)
{
    if (!this->valid)
    {
        std::cout << "Validate the network first" << std::endl;
        return {};
    }
    if (input.size() != this->input_layer.biases.get_rows())
    {
        std::cout << "Input size does not match the input layer size" << std::endl;
        return {};
    }

    Matrix output(1, input.size());
    output.set_data(input);

    // hidden layers
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        output = output * this->hidden_layers[i].weights + this->hidden_layers[i].biases;
        switch (this->hidden_layers[i].activation)
        {
        case ActivationFunction::RELU:
            output.function_on_elements(relu);
            break;
        case ActivationFunction::LEAKY_RELU:
            output.function_on_elements(leaky_relu);
            break;
        case ActivationFunction::SIGMOID:
            output.function_on_elements(sigmoid);
            break;
        case ActivationFunction::TANH:
            output.function_on_elements(tanh);
            break;
        case ActivationFunction::EXPONENTIAL:
            output.function_on_elements(exponential);
            break;
        default:
            break;
        }
    }

    // output layer
    output = output * this->output_layer.weights + this->output_layer.biases;
    switch (this->output_layer.activation)
    {
    case ActivationFunction::RELU:
        output.function_on_elements(relu);
        break;
    case ActivationFunction::LEAKY_RELU:
        output.function_on_elements(leaky_relu);
        break;
    case ActivationFunction::SIGMOID:
        output.function_on_elements(sigmoid);
        break;
    case ActivationFunction::TANH:
        output.function_on_elements(tanh);
        break;
    case ActivationFunction::EXPONENTIAL:
        output.function_on_elements(exponential);
        break;
    }

    std::vector<float> result;
    for (uint32_t i = 0; i < output.get_rows(); i++)
    {
        result.push_back(output.get(0, i));
    }
    return result;
}

std::vector<std::vector<float>> NeuralNetwork::forward_batch(std::vector<std::vector<float>> inputs)
{
    if (!this->valid)
    {
        std::cout << "Validate the network first" << std::endl;
        return {};
    }
    for (int i = 0; i < inputs.size(); i++)
    {
        if (inputs[i].size() != this->input_layer.biases.get_rows())
        {
            std::cout << "Input size does not match the input layer size" << std::endl;
            return {};
        }
    }

    Matrix output(inputs.size(), inputs[0].size());
    std::vector<float> input;
    for (int i = 0; i < inputs.size(); i++)
    {
        input.insert(input.end(), inputs[i].begin(), inputs[i].end());
    }

    output.set_data(input);

    // hidden layers
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        output = output * this->hidden_layers[i].weights;
        output.special_biases_addition_for_batched(this->hidden_layers[i].biases);
        switch (this->hidden_layers[i].activation)
        {
        case ActivationFunction::RELU:
            output.function_on_elements(relu);
            break;
        case ActivationFunction::LEAKY_RELU:
            output.function_on_elements(leaky_relu);
            break;
        case ActivationFunction::SIGMOID:
            output.function_on_elements(sigmoid);
            break;
        case ActivationFunction::TANH:
            output.function_on_elements(tanh);
            break;
        case ActivationFunction::EXPONENTIAL:
            output.function_on_elements(exponential);
            break;
        default:
            break;
        }
    }

    // output layer
    output = output * this->output_layer.weights;
    output.special_biases_addition_for_batched(this->output_layer.biases);
    switch (this->output_layer.activation)
    {
    case ActivationFunction::RELU:
        output.function_on_elements(relu);
        break;
    case ActivationFunction::LEAKY_RELU:
        output.function_on_elements(leaky_relu);
        break;
    case ActivationFunction::SIGMOID:
        output.function_on_elements(sigmoid);
        break;
    case ActivationFunction::TANH:
        output.function_on_elements(tanh);
        break;
    case ActivationFunction::EXPONENTIAL:
        output.function_on_elements(exponential);
        break;
    }

    std::vector<std::vector<float>> result;
    for (uint32_t i = 0; i < output.get_cols(); i++)
    {
        std::vector<float> row;
        for (uint32_t j = 0; j < output.get_rows(); j++)
        {
            row.push_back(output.get(i, j));
        }
        result.push_back(row);
    }
    return result;
}
NeuralNetwork NeuralNetwork::clone()
{
    NeuralNetwork nn;
    nn.input_layer = this->input_layer;
    nn.output_layer = this->output_layer;
    nn.hidden_layers = this->hidden_layers;
    nn.valid = this->valid;
    return nn;
}
