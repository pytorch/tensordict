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
    typedef std::unordered_map<std::string, node> map;
    private:
        std::shared_ptr<map> internalMap;
        map* unsafeGetInternalMap() const {
            return internalMap.get();
        }

        node GetRecursive(map* currentMap, const std::vector<std::string>& indices, const int index);
        void SetRecursive(map* currentMap, const std::vector<std::string>& indices, const int index, node value);
        // TODO something about batch size

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

        // Index Single Point
        node GetAt(const std::string key) const;
        void SetTensorAt(const std::string key, const torch::Tensor& value);
        void SetMapAt(const std::string key, const TensorMap& value);
        // Index Path
        node GetAtPath(const std::vector<std::string>& key);
        void SetTensorAtPath(const std::vector<std::string>& key, const torch::Tensor& value);
        void SetMapAtPath(std::vector<std::string>& key, TensorMap& value);
        // TODO add keys - check iterator


};

#endif
