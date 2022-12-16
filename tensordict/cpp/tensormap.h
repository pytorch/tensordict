#ifndef TensorMap_h
#define TensorMap_h

#include <memory>
#include <torch/extension.h>
#include <unordered_map>
#include <variant>
#include <string>
#include <iostream>

class TensorMap {
    typedef std::variant<torch::Tensor, TensorMap> node;
    private:
        std::shared_ptr<std::unordered_map<std::string, node> > internalMap;
        std::unordered_map<std::string, node>* unsafeGetInternalMap() const {
            return internalMap.get();
        }

        // std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& GetRecursive(std::map<std::string, std::variant<torch::Tensor*, TensorMap*>>& map, std::vector<std::string>& indices, int index);
        // TODO something about batch size
        // helpers

    public:
        TensorMap();
        ~TensorMap() = default;

        TensorMap(const TensorMap&) = default;
        TensorMap(TensorMap&&) = default;

        TensorMap& operator=(const TensorMap& other) & {
           internalMap = other.internalMap;
           return *this;
        }
        TensorMap& operator=(TensorMap& other) & {
            internalMap = std::move(other.internalMap);
            return *this;
        }

        bool is_same(const TensorMap& other) const noexcept {
            return this->internalMap == other.internalMap;
        }

        bool operator==(const TensorMap& other) const {
            return this->internalMap == other.internalMap;
        }

        bool operator!=(const TensorMap& other) const {
            return this->internalMap != other.internalMap;
        }

        // TODO Add const to all input args
        node GetAt(std::string key);
        void SetTensorAt(std::string key, torch::Tensor& value);
        void SetMapAt(std::string key, TensorMap& value);
        // void SetTensorAtPath(std::vector<std::string>& key, torch::Tensor& value);
        // void SetMapAtPath(std::vector<std::string>& key, TensorMap& value);
        // std::variant<torch::Tensor, TensorMap> GetAtPath(std::vector<std::string>& key);
        // TODO add keys - check iterator


};

#endif
