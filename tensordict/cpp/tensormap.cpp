#include "tensormap.h"
#include <cstddef>
#include <exception>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

TensorMap::TensorMap(std::vector<int64_t> batchSize)
{
    this->internalMap = std::make_shared<TensorMap::map>();
    this->batchSize = batchSize;
}

// Index Get - Set

TensorMap::node TensorMap::GetAt(const std::string key) const
{
    if (!Contains(key))
        throw std::invalid_argument("Invalid key: " + key);

    return unsafeGetInternalMap()->at(key);
}

void TensorMap::SetAt(const std::string key, const TensorMap::node& value)
{
    if (std::holds_alternative<TensorMap>(value)) {
        auto tmap = std::get<TensorMap>(value);
        ValidateBatchSize(c10::IntArrayRef(tmap.batchSize));
    }
    else {
        auto tensor = std::get<torch::Tensor>(value); // can be adapted to use TensorBase and support multiple tensor types
        ValidateBatchSize(tensor.sizes());
        // TODO: handle case when dim == size; change tensor dim after valid size check
    }

    unsafeGetInternalMap()->insert_or_assign(key, value);
}

// Path Get - Set

TensorMap::node TensorMap::GetAtPath(const py::tuple indices)
{
    if (py::len(indices) == 0)
        throw std::invalid_argument("indices must have at least one element");

    return GetRecursive(this, indices, 0);
}

void TensorMap::SetAtPath(const py::tuple indices, const TensorMap::node& value)
{
    if (py::len(indices) == 0)
        throw std::invalid_argument("indices must have at least one element");

    SetRecursive(this, indices, 0, value);
}

// TODO: Should I return by ref and have python handle the ref count?
std::set<TensorMap::key> TensorMap::GetKeys(const bool includeNested, const bool leavesOnly) {
    std::set<TensorMap::key> result;
    if (includeNested) {
        py::tuple currentPath;
        if (leavesOnly)
            GetKeysRecursiveLeavesOnly(result, currentPath, *this);
        else
            GetKeysRecursiveAll(result, currentPath, *this);
    }
    else {
        GetKeysFirstLevel(result, leavesOnly);
    }

    return result;
}


// Helper methods

TensorMap::node TensorMap::GetRecursive(TensorMap* currentMap, const py::tuple indices, const int index)
{
    auto key = indices[index].cast<std::string>();
    if (index == py::len(indices) - 1) {
        return currentMap->GetAt(key);
    }

    if (!currentMap->Contains(key)) {
        throw std::invalid_argument("Invalid key " + key + " at index: " + std::to_string(index));
    }

    auto currentNode = currentMap->GetAt(key);
    if (std::holds_alternative<TensorMap>(currentNode)) {
        auto nextMap = std::get<TensorMap>(currentNode);
        return GetRecursive(&nextMap, indices, index + 1);
    }
    else {
        throw std::invalid_argument("Expected to have a Map at index " + std::to_string(index) + " but found tensor");
    }
}

void TensorMap::SetRecursive(TensorMap* currentMap, const py::tuple indices, const int index, TensorMap::node value)
{
    auto key = indices[index].cast<std::string>();
    if (index == py::len(indices) - 1) {
        currentMap->SetAt(key, value);
        return;
    }

    if (!currentMap->Contains(key) || ! currentMap->HoldsMap(key)) { // We overwrite tensors in case we encounter it in the path
        currentMap->SetAt(key, TensorMap(currentMap->batchSize)); // We insert new map with same batchsize as parent map
    }

    auto nextMap = std::get<TensorMap>(currentMap->GetAt(key));
    SetRecursive(&nextMap, indices, index + 1, value);
}

void TensorMap::GetKeysFirstLevel(std::set<TensorMap::key> &result, bool leavesOnly) {
    for (auto i : *unsafeGetInternalMap()) {
        if (!leavesOnly || !std::holds_alternative<TensorMap>(i.second)) {
            result.insert(i.first);
        }
    }
}

void TensorMap::GetKeysRecursiveAll(std::set<TensorMap::key>& result, py::tuple currentPath, const node& currentNode) {
    if (!std::holds_alternative<TensorMap>(currentNode))
        return;

    auto currentTensorMap = std::get<TensorMap>(currentNode);
    auto currentMap = currentTensorMap.unsafeGetInternalMap();
    for (auto i : *currentMap)
    {
        auto nextPath = currentPath + py::make_tuple(i.first);
        result.insert(GetCleanKey(nextPath));
        TensorMap::GetKeysRecursiveAll(result, nextPath, i.second);
    }

}

void TensorMap::GetKeysRecursiveLeavesOnly(std::set<TensorMap::key>& result, py::tuple currentPath, const node& currentNode) {
    if (!std::holds_alternative<TensorMap>(currentNode))
    {
        result.insert(GetCleanKey(currentPath));
        return;
    }

    auto currentTensorMap = std::get<TensorMap>(currentNode);
    auto currentMap = currentTensorMap.unsafeGetInternalMap();
    for (auto i : *currentMap)
    {
        auto nextStep =  py::make_tuple(i.first);
        TensorMap::GetKeysRecursiveLeavesOnly(result, currentPath + nextStep, i.second);
    }
}

TensorMap::key TensorMap::GetCleanKey(py::tuple path) {
    TensorMap::key cleanKey = path;
    if (py::len(path) == 1)
        cleanKey = path[0].cast<std::string>();
    return cleanKey;
}

void TensorMap::ValidateBatchSize(const c10::IntArrayRef shape) {
    c10::IntArrayRef validShape = c10::IntArrayRef(batchSize);
    if (shape.size() < validShape.size())
        throw std::invalid_argument("Tensor has less dimensions than batch size");

    std::string message = "(";
    for (size_t i = 0; i < validShape.size(); i++)
    {
        if (validShape.at(i) != shape.at(i)) {
            auto expected = message + std::to_string(validShape.at(i)) + ")";
            auto actual = message + std::to_string(shape.at(i)) + ")";
            throw std::invalid_argument("Tensor size is not conform to batch size. Expected: " + expected + ", but got: " + actual);
        }
        message += std::to_string(validShape.at(i)) + ", ";
    }
}

TensorMap::map* TensorMap::unsafeGetInternalMap() const {
    return internalMap.get();
}

bool TensorMap::Contains(std::string key) const {
    return unsafeGetInternalMap()->count(key) > 0;
}

bool TensorMap::HoldsMap(std::string key) const {
    auto node = GetAt(key);
    return std::holds_alternative<TensorMap>(node);
}
