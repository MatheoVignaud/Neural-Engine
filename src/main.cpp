#include <SDL2/SDL.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <nn/network.hpp>
#include <thread>

const uint16_t population_size = 50;
const uint16_t survivors = 5;
const uint16_t iterations = 1600;
const uint16_t draw_interval = 1;
const uint16_t forward_per_thread = 10;

void thread_forward(std::vector<NeuralNetwork> *nn, std::vector<std::vector<float>> &inputs, std::vector<float> &results, std::vector<double> &fitness, int start, int end)
{

    for (int i = start; i < end; i++)
    {
        std::vector<std::vector<float>> output = nn->at(i).forward_batch(inputs);
        for (int j = 0; j < output.size(); j++)
        {
            fitness.at(i) += abs(output[j][0] - results[j]);
        }
    }
}

int main(int argc, char *argv[])
{
    // initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // create a window
    SDL_Window *window = SDL_CreateWindow("Neural Network", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 800, 600, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL)
    {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // create a neural network
    std::vector<NeuralNetwork> population;
    std::vector<double> fitness(population_size, 0);
    for (uint16_t i = 0; i < population_size; i++)
    {
        NeuralNetwork nn(1, 1, ActivationFunction::LINEAR);
        nn.add_hidden_layer(1, ActivationFunction::LEAKY_RELU);
        nn.add_hidden_layer(1, ActivationFunction::LEAKY_RELU);
        nn.add_hidden_layer(1, ActivationFunction::LEAKY_RELU);
        nn.add_hidden_layer(1, ActivationFunction::TANH);
        nn.add_hidden_layer(1, ActivationFunction::EXPONENTIAL);
        nn.add_hidden_layer(1, ActivationFunction::LEAKY_RELU);

        nn.validate();
        nn.mutate(0.1, 0.1, 0.5, 0.00);
        population.push_back(nn);
    }

    std::cout << "Population created" << std::endl;

    uint32_t generation = 0;
    float best_fitness = 0;

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

        std::vector<std::vector<float>> inputs;
        std::vector<float> result;

        for (int i = 0; i < iterations; i++)
        {
            // float x = rand() % 800;
            float x = i / 2.0;
            inputs.push_back({x});
            result.push_back(0.001 * pow(x, 2) + 0.1 * x + 2);
        }

        std::vector<std::thread> threads;
        // forward the population
        for (int i = 0; i < population_size; i += forward_per_thread)
        {
            if (i + forward_per_thread < population_size)
            {
                threads.push_back(std::thread(thread_forward, &population, std::ref(inputs), std::ref(result), std::ref(fitness), i, i + forward_per_thread));
            }
            else
            {
                threads.push_back(std::thread(thread_forward, &population, std::ref(inputs), std::ref(result), std::ref(fitness), i, population_size));
            }
        }

        for (int i = 0; i < threads.size(); i++)
        {
            threads[i].join();
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
        if (generation % draw_interval == 0)
        {
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

            // draw the best network
            std::vector<std::vector<float>> intputs_best;
            std::vector<std::vector<float>> outputs_best;
            for (int i = 0; i < 800; i++)
            {
                intputs_best.push_back({(float)i});
            }
            outputs_best = population[0].forward_batch(intputs_best);

            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
            for (int i = 0; i < 800; i++)
            {
                SDL_RenderDrawPoint(renderer, intputs_best[i][0], 500 - outputs_best[i][0] * 0.5);
            }

            SDL_RenderPresent(renderer);
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
            nn.mutate(0.03, 0.006, 0.1, 0.01);
            new_population.push_back(nn);
        }
        population = new_population;
        fitness = std::vector<double>(population_size, 0);

        // draw the screen

        generation++;
    }

    return 0;
}