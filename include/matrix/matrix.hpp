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
    void add_row(std::vector<float> row);
    void add_col(std::vector<float> col);
    float get(uint32_t col, uint32_t row);
    void set(uint32_t col, uint32_t row, float val);
    void set_data(std::vector<float> data);
    void function_on_elements(void (*func)(float &));
    void print();

    Matrix operator+(Matrix &b);
    Matrix operator-(Matrix &b);
    Matrix operator*(Matrix &b);
    Matrix operator~();

    void special_biases_addition_for_batched(Matrix &b);

private:
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> data;

    friend Matrix add(Matrix &a, Matrix &b);
    friend Matrix subtract(Matrix &a, Matrix &b);
    friend Matrix multiply(Matrix &a, Matrix &b);
    friend Matrix transpose(Matrix &a);
};