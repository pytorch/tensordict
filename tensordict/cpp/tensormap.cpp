#include "tensormap.h"
#include <stdexcept>

void TensorMap::set(std::string key, torch::Tensor value)
{
    map[key] = value;
}

void TensorMap::set(std::string key, TensorMap value)
{
    map[key] = value;
}

// void TensorMap::set(pybind11::tuple, torch::Tensor);
// void TensorMap::set(pybind11::tuple, TensorMap);

std::variant<torch::Tensor, TensorMap> TensorMap::get(std::string key)
{
    if(!map.contains(key))
        throw std::invalid_argument("invalid key: " + key);

    return map[key];
}

// std::variant<torch::Tensor, TensorMap> TensorMap::get(pybind11::tuple key);

// private void set_recursive(std::vector indices, std::v)
