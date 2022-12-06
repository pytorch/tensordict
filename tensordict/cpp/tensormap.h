#include <torch/extension.h>

class TensorMap {
    private:
        std::map<std::string, torch::Tensor> tensors;
        std::map<std::string, TensorMap> maps;
        // TODO something about batch size
    public:
        TensorMap();
        void set(std::string key, torch::Tensor value);
        void set(std::string key, TensorMap value);
        void set(pybind11::tuple key, torch::Tensor value);
        void set(pybind11::tuple key, TensorMap value);
        std::variant<torch::Tensor, TensorMap> get(std::string key);
        std::variant<torch::Tensor, TensorMap> get(pybind11::tuple key);
        // TODO add keys - check iterator
};
