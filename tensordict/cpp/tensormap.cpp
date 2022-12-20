#include "tensormap.h"
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

TensorMap::TensorMap()
{
    this->internalMap = std::make_shared<TensorMap::map>();
}

// Index Get - Set

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

// Path Get - Set

TensorMap::node TensorMap::GetAtPath(const py::tuple indices)
{
    if (py::len(indices) == 0)
        throw std::invalid_argument("indices must have at least one element");

    return GetRecursive(unsafeGetInternalMap(), indices, 0);
}

void TensorMap::SetTensorAtPath(const py::tuple indices, const torch::Tensor& value)
{
    if (py::len(indices) == 0)
        throw std::invalid_argument("indices must have at least one element");

    SetRecursive(unsafeGetInternalMap(), indices, 0, value);
}

void TensorMap::SetMapAtPath(const py::tuple indices, const TensorMap& value)
{
    if (py::len(indices) == 0)
        throw std::invalid_argument("indices must have at least one element");

    SetRecursive(unsafeGetInternalMap(), indices, 0, value);
}

// TODO: Should I return by ref and have python handle the ref count?
std::set<TensorMap::key> TensorMap::GetKeys(const bool includeNested, const bool leavesOnly) {
    std::set<TensorMap::key> result;
    if (includeNested) {
        py::tuple currentPath;
        GetKeysRecursive(result, currentPath, *this, leavesOnly);
    }
    else {
        GetKeysFirstLevel(result, leavesOnly);
    }

    return result;
}


// Helper methods

TensorMap::node TensorMap::GetRecursive(TensorMap::map* currentMap, const py::tuple indices, const int index)
{
    auto key = indices[index].cast<std::string>();
    if (index == py::len(indices) - 1) {
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

void TensorMap::SetRecursive(TensorMap::map* currentMap, const py::tuple indices, const int index, TensorMap::node value)
{
    auto key = indices[index].cast<std::string>();
    if (index == py::len(indices) - 1) {
        currentMap->insert_or_assign(key, value);
        return;
    }

    if (currentMap->count(key) == 0 || !std::holds_alternative<TensorMap>(currentMap->at(key))) {
        currentMap->insert_or_assign(key, TensorMap()); // For now we insert maps by value
    }

    auto nextMap = std::get<TensorMap>(currentMap->at(key));
    SetRecursive(nextMap.unsafeGetInternalMap(), indices, index + 1, value);
}

void TensorMap::GetKeysRecursive(std::set<TensorMap::key>& result, py::tuple currentPath, const node& currentNode, bool leavesOnly) {
    if (!std::holds_alternative<TensorMap>(currentNode))
    {
        TensorMap::key path = currentPath;
        if (py::len(currentPath) == 1)
            path = currentPath[0].cast<std::string>();

        result.insert(path);
        return;
    }

    auto currentTensorMap = std::get<TensorMap>(currentNode);
    auto currentMap = currentTensorMap.unsafeGetInternalMap();
    for (auto i : *currentMap)
    {
        auto nextStep =  py::make_tuple(i.first);
        TensorMap::GetKeysRecursive(result, currentPath + nextStep, i.second, leavesOnly);
    }
}

void TensorMap::GetKeysFirstLevel(std::set<TensorMap::key> &result, bool leavesOnly) {
    for (auto i : *unsafeGetInternalMap()) {
        if (!leavesOnly || !std::holds_alternative<TensorMap>(i.second)) {
            result.insert(i.first);
        }
    }
}


TensorMap::map* TensorMap::unsafeGetInternalMap() const {
    return internalMap.get();
}
