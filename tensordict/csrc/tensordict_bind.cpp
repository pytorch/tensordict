#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/script.h>

#include <unordered_map>
#include <vector>

namespace tensordict_bind_internal {

using Tensor = at::Tensor;

// Flat, single-device TensorDict TorchBind class
struct TensorDictBind : torch::CustomClassHolder {
  TensorDictBind(const c10::Device& device)
      : device_(device) {}

  // Factory from key/value pairs with explicit batch_size and device
  static c10::intrusive_ptr<TensorDictBind> from_pairs(
      const std::vector<std::string>& keys,
      const std::vector<Tensor>& values,
      const std::vector<int64_t>& batch_size,
      const c10::Device& device) {
    if (keys.size() != values.size()) {
      TORCH_CHECK(false, "keys and values must have the same length");
    }
    auto obj = c10::make_intrusive<TensorDictBind>(device);
    obj->batch_size_ = batch_size;
    for (size_t i = 0; i < keys.size(); ++i) {
      obj->validate_tensor(values[i]);
      obj->validate_batch_size_prefix(values[i].sizes());
      obj->data_.emplace(keys[i], values[i]);
    }
    return obj;
  }

  bool has(const std::string& key) const {
    return data_.find(key) != data_.end();
  }

  Tensor get(const std::string& key) const {
    auto it = data_.find(key);
    TORCH_CHECK(it != data_.end(), "Key not found: ", key);
    return it->second;
  }

  void set(const std::string& key, const Tensor& value) {
    validate_tensor(value);
    validate_batch_size_prefix(value.sizes());
    data_[key] = value;
  }

  std::vector<std::string> keys() const {
    std::vector<std::string> out;
    out.reserve(data_.size());
    for (const auto& kv : data_) {
      out.push_back(kv.first);
    }
    return out;
  }

  std::vector<int64_t> batch_size() const { return batch_size_; }

  c10::Device device() const { return device_; }

  c10::intrusive_ptr<TensorDictBind> to(const c10::Device& new_device) const {
    auto obj = c10::make_intrusive<TensorDictBind>(new_device);
    obj->batch_size_ = batch_size_;
    for (const auto& kv : data_) {
      obj->data_.emplace(kv.first, kv.second.to(new_device));
    }
    return obj;
  }

  c10::intrusive_ptr<TensorDictBind> clone() const {
    auto obj = c10::make_intrusive<TensorDictBind>(device_);
    obj->batch_size_ = batch_size_;
    for (const auto& kv : data_) {
      obj->data_.emplace(kv.first, kv.second.clone());
    }
    return obj;
  }

 private:
  void validate_tensor(const Tensor& t) const {
    TORCH_CHECK(
        t.defined(),
        "TensorDictBind: cannot store an undefined tensor");
    TORCH_CHECK(
        t.device() == device_,
        "All tensors must be on device ", device_.str(), 
        ", but got ", t.device().str());
  }

  void validate_batch_size_prefix(at::IntArrayRef sizes) const {
    // If no batch_size specified, accept any.
    if (batch_size_.empty()) {
      return;
    }
    TORCH_CHECK(
        sizes.size() >= batch_size_.size(),
        "Tensor has fewer dims (", sizes.size(), ") than batch_size prefix (",
        batch_size_.size(), ")");
    for (size_t i = 0; i < batch_size_.size(); ++i) {
      TORCH_CHECK(
          sizes[i] == batch_size_[i],
          "Tensor batch dim ", i, " mismatch: expected ", batch_size_[i],
          ", got ", sizes[i]);
    }
  }

  std::unordered_map<std::string, Tensor> data_;
  std::vector<int64_t> batch_size_;
  c10::Device device_;
};

} // namespace tensordict_bind_internal

TORCH_LIBRARY(tensordict, m) {
  using tensordict_bind_internal::TensorDictBind;
  m.class_<TensorDictBind>("TensorDict")
      .def(torch::init<c10::Device>())
      .def_static("from_pairs", &TensorDictBind::from_pairs)
      .def("has", &TensorDictBind::has)
      .def("get", &TensorDictBind::get)
      .def("set", &TensorDictBind::set)
      .def("keys", &TensorDictBind::keys)
      .def("batch_size", &TensorDictBind::batch_size)
      .def("device", &TensorDictBind::device)
      .def("to", &TensorDictBind::to)
      .def("clone", &TensorDictBind::clone);
}




