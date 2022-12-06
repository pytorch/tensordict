#include "tensormap.h"
#include <stdexcept>
#include <string>

void TensorMap::set(std::string key, torch::Tensor value)
{
    this->map[key] = value;
}

void TensorMap::set(std::string key, TensorMap value)
{
    this->map[key] = value;
}

// void TensorMap::set(pybind11::tuple, torch::Tensor);
// void TensorMap::set(pybind11::tuple, TensorMap);

std::variant<torch::Tensor, TensorMap> TensorMap::get(std::string key)
{
    if (!this->map.contains(key))
        throw std::invalid_argument("Invalid key: " + key);

    return this->map[key];
}

// std::variant<torch::Tensor, TensorMap> TensorMap::get(pybind11::tuple key);

// TODO traversal can be outside in helper
void TensorMap::SetRecursive(
    std::map<std::string, std::variant<torch::Tensor, TensorMap>> map,
    std::vector<std::string> indices,
    std::variant<torch::Tensor, TensorMap> value,
    int index = 0)
{
    auto key = indices[index];
    if (!map.contains(key))
        throw std::invalid_argument("Invalid key " + key + " at index: " + std::to_string(index));

    if (index == indices.size() - 1)
    {
        map[key] = value;
        return;
    }

    if (std::holds_alternative<TensorMap>(map[key]))
    {
        auto currentMap = std::get<TensorMap>(map[key]);
        SetRecursive(currentMap.map, indices, value, index + 1);
    }
    else
        throw std::invalid_argument("Expected to have a Map at index " + std::to_string(index) + " but found tensor");
}
