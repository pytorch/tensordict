#include "tensormap.h"

TensorMap::TensorMap()
{
    tensors = new std::map<std::string, torch::Tensor>();
    maps = new std::map<std::string, TensorMap>();
}

void TensorMap::set(std::string key, torch::Tensor)
{
    if (tensors.contains(key))
        return;
}

void TensorMap::set(std::string, TensorMap);
void TensorMap::set(std::tuple, torch::Tensor);
void TensorMap::set(std::tuple, TensorMap);
std::variant<torch::Tensor, TensorMap> TensorMap::get(std::string key);
std::variant<torch::Tensor, TensorMap> TensorMap::get(std::tuple key);
