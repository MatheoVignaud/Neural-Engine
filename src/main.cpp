#include <SDL2/SDL.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <nn/network.hpp>

const uint16_t population_size = 50;
const uint16_t survivors = 5;
const uint16_t iterations = 100;

int main(int argc, char *argv[])
{
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // create a window
    SDL_Window *window = SDL_CreateWindow("Hello SDL", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 600, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // create a neural network
    std::vector<NeuralNetwork> population;
    std::vector<float> fitness(population_size, 0);
    for (uint16_t i = 0; i < population_size; i++)
    {
        NeuralNetwork nn(1, 1, ActivationFunction::LINEAR);
        nn.add_hidden_layer(1, ActivationFunction::EXPONENTIAL);
        nn.add_hidden_layer(4, ActivationFunction::TANH);
        nn.add_hidden_layer(1, ActivationFunction::RELU);
        nn.add_hidden_layer(1, ActivationFunction::EXPONENTIAL);
        nn.add_hidden_layer(1, ActivationFunction::LINEAR);
        nn.validate();
        // nn.mutate(0.1, 0.1, 0.0, 0.00);
        population.push_back(nn);
    }

    std::cout << "Population created" << std::endl;

    uint32_t generation = 0;

    // main loop
    bool quit = false;
    while (!quit)
    {
        // handle events
        SDL_Event e;
        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_QUIT)
            {
                quit = true;
            }
        }

        // clear the screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        // draw sinusoide
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (int i = 0; i < 800; i++)
        {
            float x = i;
            float y = 0.001 * pow(x, 2) + 0.1 * x + 2;
            SDL_RenderDrawPoint(renderer, i, 500 - y * 0.5);
        }

        std::vector<float> inputs;
        std::vector<float> result;

        for (int i = 0; i < iterations; i++)
        {
            // float x = rand() % 800;
            float x = i * 8;
            inputs.push_back(x);
            result.push_back(0.001 * pow(x, 2) + 0.1 * x + 2);
        }

        for (int i = 0; i < population_size; i++)
        {
            std::cout << i << "/" << population_size << std::endl;
            std::vector<float> output;
            for (int j = 0; j < iterations; j++)
            {
                output = population[i].forward({inputs[j]});
                fitness[i] += pow(output[0] - result[j], 2);
            }
            // delete last line of the console
            std::cout << "\033[A";
        }

        // sort the population based on fitness , lower is better
        bool sorted = false;
        while (!sorted)
        {
            sorted = true;
            for (int i = 0; i < population_size - 1; i++)
            {
                if (fitness[i] > fitness[i + 1])
                {
                    std::swap(population[i], population[i + 1]);
                    std::swap(fitness[i], fitness[i + 1]);
                    sorted = false;
                }
            }
        }

        // print the best fitness
        std::cout << "Best fitness of the generation: " << generation << " is " << fitness[0] << " with " << population[0].get_number_of_trainable_parameters() << " parameters" << std::endl;

        // draw the best network
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        for (int i = 0; i < 800; i++)
        {
            float x = i;
            float y = population[0].forward({x})[0];
            SDL_RenderDrawPoint(renderer, i, 500 - y * 0.5);
        }

        // create a new population
        std::vector<NeuralNetwork> new_population;
        for (int i = 0; i < survivors; i++)
        {
            new_population.push_back(population[i]);
        }
        for (int i = 0; i < population_size - survivors; i++)
        {
            NeuralNetwork nn = population[rand() % survivors];
            nn.mutate(0.05, 0.08, 0.3, 0.01);
            new_population.push_back(nn);
        }
        population = new_population;
        fitness = std::vector<float>(population_size, 0);

        // draw the screen
        SDL_RenderPresent(renderer);
        generation++;
    }

    return 0;
}