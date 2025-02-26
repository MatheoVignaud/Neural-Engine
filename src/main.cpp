#include <chrono>
#include <iostream>
#include <nn/network.hpp>

const uint16_t iterations = 10;
const uint16_t N = 5;

int main(int argc, char *argv[])
{
    Matrix a(4, 3);

    a.set_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    std::cout << "Matrix a: " << std::endl;
    a.print();

    Matrix b(3, 4);
    b.set_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    std::cout << "Matrix b: " << std::endl;
    b.print();

    Matrix c(2, 2);
    c = a * b;

    std::cout << "Matrix c: " << std::endl;
    c.print();

    return 0;
}