#pragma once
#include <cstdint>
#include <vector>

class Matrix
{
public:
    Matrix() {}
    Matrix(uint32_t cols, uint32_t rows, float init_val = 0);
    ~Matrix();

    uint32_t get_rows();
    uint32_t get_cols();
    float get(uint32_t col, uint32_t row);
    void set(uint32_t col, uint32_t row, float val);
    void set_data(std::vector<float> data);
    void function_on_elements(void (*func)(float &));
    void print();

    Matrix operator+(Matrix &b);
    Matrix operator-(Matrix &b);
    Matrix operator*(Matrix &b);
    Matrix operator~();

private:
    uint32_t rows;
    uint32_t cols;
    std::vector<float> data;
};