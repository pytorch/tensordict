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

    return GetRecursive(unsafeGetInternalMap(), indices, 0);
}

void TensorMap::SetTensorAtPath(const std::vector<std::string>& indices, const torch::Tensor& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    SetRecursive(unsafeGetInternalMap(), indices, 0, value);
}

void TensorMap::SetMapAtPath(std::vector<std::string>& indices, TensorMap& value)
{
    if (indices.size() == 0)
        throw std::invalid_argument("indices must have at least one element");

    SetRecursive(unsafeGetInternalMap(), indices, 0, value);
}


// Helper methods

TensorMap::node TensorMap::GetRecursive(TensorMap::map* currentMap, const std::vector<std::string>& indices, const int index)
{
    auto key = indices[index];
    if (index == indices.size() - 1) {
        return currentMap->at(key);
    }

    if (currentMap->count(key) == 0) {
        throw std::invalid_argument("Invalid key " + key + " at index: " + std::to_string(index));
    }

    auto currentNode = currentMap->at(key);
    if (std::holds_alternative<TensorMap>(currentNode)) {
        auto nextMap = std::get<TensorMap>(currentNode);
        return GetRecursive(nextMap.unsafeGetInternalMap(), indices, index + 1);
    }
    else {
        throw std::invalid_argument("Expected to have a Map at index " + std::to_string(index) + " but found tensor");
    }
}

void TensorMap::SetRecursive(TensorMap::map* currentMap, const std::vector<std::string>& indices, const int index, TensorMap::node value)
{
    auto key = indices[index];
    if (index == indices.size() - 1) {
        currentMap->insert_or_assign(key, value);
        return;
    }

    if (currentMap->count(key) == 0 || !std::holds_alternative<TensorMap>(currentMap->at(key))) {
        currentMap->insert_or_assign(key, TensorMap()); // For now we insert maps by value
    }

    auto nextMap = std::get<TensorMap>(currentMap->at(key));
    SetRecursive(nextMap.unsafeGetInternalMap(), indices, index + 1, value);
}
