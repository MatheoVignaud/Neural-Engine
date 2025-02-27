#include <SDL2/SDL.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <nn/network.hpp>

const uint16_t population_size = 100;

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
        nn.add_hidden_layer(2, ActivationFunction::SIGMOID);
        nn.validate();
        population.push_back(nn);
    }

    std::cout << "Population created" << std::endl;

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
            float x = i / 800.0 * 2 * 3.14;
            float y = sin(x);
            SDL_RenderDrawPoint(renderer, i, 300 - y * 100);
        }

        population[0].forward({0.5});

        // draw the screen
        SDL_RenderPresent(renderer);
    }

    return 0;
}