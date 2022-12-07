#include "tensormap.h"
#include <exception>
#include <string>

void TensorMap::SetTensorAt(std::string key, torch::Tensor& value)
{
    // we want to pass objects as arguments but store the reference
    this->map[key] = &value;
}

void TensorMap::SetMapAt(std::string key, TensorMap& value)
{
    this->map[key] = &value;
}

void TensorMap::SetTensorAtPath(std::vector<std::string>& indices, torch::Tensor& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(this->map, indices, 0);
    auto key = indices[indices.size() - 1];

    lastMap[key] = &value;
}

void TensorMap::SetMapAtPath(std::vector<std::string>& indices, TensorMap& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(this->map, indices, 0);
    auto key = indices[indices.size() - 1];

    lastMap[key] = &value;
}

std::variant<torch::Tensor, TensorMap> TensorMap::GetAt(std::string key)
{
    if (this->map.count(key) == 0)
        throw std::invalid_argument("Invalid key: " + key);

    return UnboxVariant(this->map[key]);
}

std::variant<torch::Tensor, TensorMap> TensorMap::GetAtPath(std::vector<std::string>& indices)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(this->map, indices, 0);
    auto key = indices[indices.size() - 1];

    return UnboxVariant(lastMap[key]);
}

// Helper methods

std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& TensorMap::GetRecursive(
    std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& map,
    std::vector<std::string>& indices,
    int index)
{
    auto key = indices[index];
    if (map.count(key) == 0)
        throw std::invalid_argument("Invalid key " + key + " at index: " + std::to_string(index));

    if (index == indices.size() - 1)
    {
        return map;
    }

    if (std::holds_alternative<TensorMap*>(map[key]))
    {
        auto currentMap = std::get<TensorMap*>(map[key]);
        return GetRecursive(currentMap->map, indices, index + 1);
    }
    else
        throw std::invalid_argument("Expected to have a Map at index " + std::to_string(index) + " but found tensor");
}

std::variant<torch::Tensor, TensorMap> TensorMap::UnboxVariant(std::variant<torch::Tensor*, TensorMap*> pointer)
{
    if (std::holds_alternative<TensorMap*>(pointer))
    {
        TensorMap* value = std::get<TensorMap*>(pointer);
        return std::variant<torch::Tensor, TensorMap>(*value);
    }

    torch::Tensor* value = std::get<torch::Tensor*>(pointer);
    return std::variant<torch::Tensor, TensorMap>(*value);
}