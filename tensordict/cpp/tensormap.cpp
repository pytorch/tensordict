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
std::set<std::vector<std::string> > TensorMap::GetKeys() {
    std::set<std::vector<std::string> > result;
    std::vector<std::string> currentPath;

    GetKeysRecursive(result, currentPath, *this);
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

void TensorMap::GetKeysRecursive(std::set<std::vector<std::string> >& result, std::vector<std::string>& currentPath, const node& currentNode) {
    if (!std::holds_alternative<TensorMap>(currentNode))
    {
        std::vector<std::string> copy(currentPath);
        result.insert(copy);
        return;
    }

    auto currentTensorMap = std::get<TensorMap>(currentNode);
    auto currentMap = currentTensorMap.unsafeGetInternalMap();
    for (auto i : *currentMap)
    {
        currentPath.push_back(i.first);
        TensorMap::GetKeysRecursive(result, currentPath, i.second);
        currentPath.pop_back();
    }
}
