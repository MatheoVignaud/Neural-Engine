#pragma once

#include <matrix/matrix.hpp>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <vector>

inline void addThread(Matrix &a, Matrix &b, Matrix &result, uint32_t start, uint32_t end)
{
    for (uint32_t i = start; i < end; i++)
    {
        for (uint32_t j = 0; j < a.get_cols(); j++)
        {
            result.set(j, i, a.get(j, i) + b.get(j, i));
        }
    }
}

inline Matrix add(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols())
    {
        const uint32_t ops_per_thread = 100;
        Matrix result(a.get_cols(), b.get_rows());
        std::vector<std::thread> threads;
        for (uint32_t i = 0; i < b.get_rows(); i += ops_per_thread)
        {
            // create threads , but check if we are not out of bounds
            if (i + ops_per_thread < a.get_rows())
            {
                threads.push_back(std::thread(addThread, std::ref(a), std::ref(b), std::ref(result), i, i + ops_per_thread));
            }
            else
            {
                threads.push_back(std::thread(addThread, std::ref(a), std::ref(b), std::ref(result), i, a.get_rows()));
            }
        }
        for (auto &t : threads)
        {
            t.join();
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

inline void subtractThread(Matrix &a, Matrix &b, Matrix &result, uint32_t start, uint32_t end)
{
    for (uint32_t i = start; i < end; i++)
    {
        for (uint32_t j = 0; j < a.get_cols(); j++)
        {
            result.set(i, j, a.get(j, i) - b.get(j, i));
        }
    }
}

inline Matrix subtract(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_rows() && a.get_cols() == b.get_cols())
    {
        const uint32_t ops_per_thread = 100;
        Matrix result(a.get_cols(), b.get_rows());
        std::vector<std::thread> threads;
        for (uint32_t i = 0; i < b.get_rows(); i += ops_per_thread)
        {
            // create threads , but check if we are not out of bounds
            if (i + ops_per_thread < a.get_rows())
            {
                threads.push_back(std::thread(subtractThread, std::ref(a), std::ref(b), std::ref(result), i, i + ops_per_thread));
            }
            else
            {
                threads.push_back(std::thread(subtractThread, std::ref(a), std::ref(b), std::ref(result), i, a.get_rows()));
            }
        }
        for (auto &t : threads)
        {
            t.join();
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

inline void multiplyThread(Matrix &a, Matrix &b, Matrix &result, uint32_t start, uint32_t end)
{
    for (uint32_t i = start; i < end; i++)
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
}

inline Matrix multiply(Matrix &a, Matrix &b)
{
    if (a.get_rows() == b.get_cols())
    {
        const uint32_t ops_per_thread = 10;
        std::vector<std::thread> threads;
        Matrix result(a.get_cols(), b.get_rows());
        for (uint32_t i = 0; i < b.get_rows(); i += ops_per_thread)
        {
            // create threads , but check if we are not out of bounds
            if (i + ops_per_thread < b.get_rows())
            {
                threads.push_back(std::thread(multiplyThread, std::ref(a), std::ref(b), std::ref(result), i, i + ops_per_thread));
            }
            else
            {
                threads.push_back(std::thread(multiplyThread, std::ref(a), std::ref(b), std::ref(result), i, b.get_rows()));
            }
        }
        for (auto &t : threads)
        {
            t.join();
        }
        return result;
    }
    throw std::invalid_argument("Matrix dimensions do not match");
}

Matrix transpose(Matrix &a)
{
    Matrix result(a.get_cols(), a.get_rows());
    for (uint32_t i = 0; i < a.get_rows(); i++)
    {
        for (uint32_t j = 0; j < a.get_cols(); j++)
        {
            result.set(j, i, a.get(i, j));
        }
    }
    return result;
}
