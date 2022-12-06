#include <torch/extension.h>

class TensorMap {
    private:
        std::map<std::string, std::variant<torch::Tensor, TensorMap>> map;
        void SetRecursive(
            std::map<std::string, std::variant<torch::Tensor, TensorMap>> map,
            std::vector<std::string> indices,
            std::variant<torch::Tensor, TensorMap> value,
            int index);
        // TODO something about batch size
    public:
        TensorMap();
        void set(std::string key, torch::Tensor value);
        void set(std::string key, TensorMap value);
        void set(std::vector<std::string> key, torch::Tensor value);
        void set(std::vector<std::string> key, TensorMap value);
        std::variant<torch::Tensor, TensorMap> get(std::string key);
        std::variant<torch::Tensor, TensorMap> get(std::vector<std::string> key);
        // TODO add keys - check iterator
};
