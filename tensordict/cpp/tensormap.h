#include <torch/extension.h>
#include <unordered_map>
#include <variant>

class TensorMap {
    typedef std::variant<torch::Tensor, TensorMap> node;
    private:
        std::unordered_map<std::string, node> internalMap;
        // std::variant<torch::Tensor, TensorMap> UnboxVariant(std::variant<torch::Tensor*, TensorMap*> pointer);
        // std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& GetRecursive(std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& map, std::vector<std::string>& indices, int index);
        // TODO something about batch size
    public:
        // TODO Add const to all input args
        void SetTensorAt(std::string key, torch::Tensor& value);
        node GetTensorAt(std::string key);

        // void SetMapAt(std::string key, TensorMap& value);
        // void SetTensorAtPath(std::vector<std::string>& key, torch::Tensor& value);
        // void SetMapAtPath(std::vector<std::string>& key, TensorMap& value);
        // std::variant<torch::Tensor, TensorMap> GetAtPath(std::vector<std::string>& key);
        // TODO add keys - check iterator
};
