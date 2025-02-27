#include <nn/network.hpp>

NeuralNetwork::NeuralNetwork(uint32_t input_size, ActivationFunction input_activation, uint32_t output_size, ActivationFunction output_activation)
{
    this->input_layer.type = LayerType::INPUT;
    this->input_layer.biases = Matrix(input_size, 1); // just to know the size of the input
    this->input_layer.activation = input_activation;

    this->output_layer.type = LayerType::OUTPUT;
    this->output_layer.biases = Matrix(output_size, 1);
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
        hidden_layer.type = LayerType::HIDDEN;
        hidden_layer.biases = Matrix(size, 1);
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
    this->input_layer.weights = Matrix(this->hidden_layers[0].biases.get_rows(), this->input_layer.biases.get_cols(), 1);
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
    for (int i = 0; i < this->input_layer.biases.get_rows(); i++)
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
        for (int j = 0; j < this->hidden_layers[i].biases.get_rows(); j++)
        {
            this->hidden_layers[i].biases.set(j, 0, distribution(gen));
        }
    }
    this->valid = true;
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

    Matrix input_matrix(this->input_layer.biases.get_rows(), 1);
    input_matrix.set_data(input);

    // input layer
    Matrix output = this->input_layer.weights * input_matrix + this->input_layer.biases;
    switch (this->input_layer.activation)
    {
    case ActivationFunction::RELU:
        output.function_on_elements(relu);
        break;
    case ActivationFunction::SIGMOID:
        output.function_on_elements(sigmoid);
        break;
    case ActivationFunction::TANH:
        output.function_on_elements(tanh);
        break;
    }

    // hidden layers
    for (uint32_t i = 0; i < this->hidden_layers.size(); i++)
    {
        output = this->hidden_layers[i].weights * output + this->hidden_layers[i].biases;
        switch (this->hidden_layers[i].activation)
        {
        case ActivationFunction::RELU:
            output.function_on_elements(relu);
            break;
        case ActivationFunction::SIGMOID:
            output.function_on_elements(sigmoid);
            break;
        case ActivationFunction::TANH:
            output.function_on_elements(tanh);
            break;
        }
    }

    // output layer
    output = this->output_layer.weights * output + this->output_layer.biases;
    switch (this->output_layer.activation)
    {
    case ActivationFunction::RELU:
        output.function_on_elements(relu);
        break;
    case ActivationFunction::SIGMOID:
        output.function_on_elements(sigmoid);
        break;
    case ActivationFunction::TANH:
        output.function_on_elements(tanh);
        break;
    }

    std::vector<float> result;
    for (uint32_t i = 0; i < output.get_rows(); i++)
    {
        result.push_back(output.get(0, i));
    }
    return result;
}
