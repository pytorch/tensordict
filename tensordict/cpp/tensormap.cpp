#include "tensormap.h"
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>

TensorMap::TensorMap()
{
    this->internalMap = std::make_shared<TensorMap::map>();
}

TensorMap::node TensorMap::GetAt(const std::string key) const
{
    auto _map = unsafeGetInternalMap();
    if (_map->count(key) == 0)
        throw std::invalid_argument("Invalid key: " + key);

    return _map->at(key);
}

void TensorMap::SetTensorAt(const std::string key, const torch::Tensor& value)
{
    unsafeGetInternalMap()->insert_or_assign(key, value);
}

void TensorMap::SetMapAt(const std::string key, const TensorMap& value)
{
    unsafeGetInternalMap()->insert_or_assign(key, value);
}

// Path

TensorMap::node TensorMap::GetAtPath(const std::vector<std::string>& indices)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(unsafeGetInternalMap(), indices, 0, false);
    auto key = indices[indices.size() - 1];

    return lastMap->at(key);
}

void TensorMap::SetTensorAtPath(const std::vector<std::string>& indices, const torch::Tensor& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(unsafeGetInternalMap(), indices, 0, true);
    auto key = indices[indices.size() - 1];

    lastMap->insert_or_assign(key, value);
}

/*
void TensorMap::SetMapAtPath(std::vector<std::string>& indices, TensorMap& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    auto lastMap = GetRecursive(this->map, indices, 0);
    auto key = indices[indices.size() - 1];

    lastMap[key] = &value;
}
*/


// Helper methods

TensorMap::map* TensorMap::GetRecursive(TensorMap::map* currentMap, const std::vector<std::string>& indices, const int index, const bool forcePath)
{
    if (index == indices.size() - 1) {
        return currentMap;
    }

    auto key = indices[index];
    if (currentMap->count(key) == 0) {
        if (forcePath) {
            currentMap->insert_or_assign(key, TensorMap()); // For now we insert maps by value
        }
        else {
            throw std::invalid_argument("Invalid key " + key + " at index: " + std::to_string(index));
        }
    }

    auto currentNode = currentMap->at(key);
    if (std::holds_alternative<TensorMap>(currentNode)) {
        auto nextMap = std::get<TensorMap>(currentNode);
        return GetRecursive(nextMap.unsafeGetInternalMap(), indices, index + 1, forcePath);
    }
    else {
        throw std::invalid_argument("Expected to have a Map at index " + std::to_string(index) + " but found tensor");

    }
}
