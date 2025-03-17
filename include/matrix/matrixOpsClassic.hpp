#pragma once

#include <matrix/matrix.hpp>
#include <stdexcept>

inline Matrix add(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols())
    {
        Matrix result(a.get_cols(), b.get_rows());
        for (uint32_t i = 0; i < b.get_rows(); i++)
        {
            for (uint32_t j = 0; j < a.get_cols(); j++)
            {
                result.set(j, i, a.get(j, i) + b.get(j, i));
            }
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

inline Matrix subtract(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols())
    {
        Matrix result(a.get_cols(), b.get_rows());
        for (uint32_t i = 0; i < b.get_rows(); i++)
        {
            for (uint32_t j = 0; j < a.get_cols(); j++)
            {
                result.set(j, i, a.get(j, i) - b.get(j, i));
            }
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

inline Matrix multiply(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_cols())
    {
        Matrix result(a.get_cols(), b.get_rows());
        for (uint32_t i = 0; i < b.get_rows(); i++)
        {
            for (uint32_t j = 0; j < a.get_cols(); j++)
            {
                float sum = 0;
                for (uint32_t k = 0; k < a.get_rows(); k++)
                {
                    sum += a.get(j, k) * b.get(k, i);
                }
                result.set(j, i, sum);
            }
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

inline Matrix transpose(Matrix &a)
{
    Matrix result(a.get_rows(), a.get_cols());
    for (uint32_t i = 0; i < a.get_rows(); i++)
    {
        for (uint32_t j = 0; j < a.get_cols(); j++)
        {
            result.set(i, j, a.get(j, i));
        }
    }
    return result;
}