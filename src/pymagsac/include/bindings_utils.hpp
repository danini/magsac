#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Used by the Python functions of graph-cut-ransac's estimators (see run_magsac.h).

typedef pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> DoubleArray;

// Copies an array of dims [n, cols], n >= min_rows, to a row-major vector.
inline std::vector<double> readRows(const DoubleArray& array, pybind11::ssize_t cols, pybind11::ssize_t min_rows, const char* what)
{
    if (array.ndim() != 2 || array.shape(1) != cols || array.shape(0) < min_rows)
        throw std::invalid_argument(std::string(what) + " should be an array with dims [n," +
            std::to_string(cols) + "], n>=" + std::to_string(min_rows));
    return std::vector<double>(array.data(), array.data() + array.size());
}

// The inlier probabilities of the n data points, used only by samplers 3 (NG-RANSAC) and
// 4 (AR-Sampler), which need one per point, and, for the AR-Sampler, a positive one.
inline std::vector<double> readProbabilities(int sampler, const DoubleArray& probabilities, pybind11::ssize_t n)
{
    if (sampler != 3 && sampler != 4)
        return {};
    if (probabilities.size() != n)
        throw std::invalid_argument("probabilities should have one value per data point (" +
            std::to_string(n) + ") with samplers 3 and 4, not " + std::to_string(probabilities.size()));
    std::vector<double> values(probabilities.data(), probabilities.data() + n);
    if (sampler == 4 && !(*std::max_element(values.begin(), values.end()) > 0))
        throw std::invalid_argument("probabilities should have a positive value with sampler 4");
    return values;
}

// The model as an array of the given shape, or None if no model was found.
inline pybind11::object toArray(int num_inliers, const std::vector<double>& values, std::vector<pybind11::ssize_t> shape)
{
    if (num_inliers == 0)
        return pybind11::none();
    DoubleArray array(shape);
    std::copy(values.begin(), values.end(), array.mutable_data());
    return std::move(array);
}

inline pybind11::array_t<bool> toMask(const std::vector<bool>& inliers)
{
    pybind11::array_t<bool> mask(inliers.size());
    std::copy(inliers.begin(), inliers.end(), mask.mutable_data());
    return mask;
}
