// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

namespace py = pybind11;


py::tuple _unravel_key_to_tuple(const py::object& key) {
    bool is_tuple = py::isinstance<py::tuple>(key);
    bool is_str = py::isinstance<py::str>(key);

    if (is_tuple) {
        py::list newkey;
        for (const auto& subkey : key) {
            if (py::isinstance<py::str>(subkey)) {
                newkey.append(subkey);
            } else {
                auto _key = _unravel_key_to_tuple(subkey.cast<py::object>());
                if (_key.size() == 0) {
                    return py::make_tuple();
                }
                newkey += _key;
            }
        }
        return py::tuple(newkey);
    }
    if (is_str) {
        return py::make_tuple(key);
    } else {
        return py::make_tuple();
    }
}

py::object unravel_key(const py::object& key) {
    bool is_tuple = py::isinstance<py::tuple>(key);
    bool is_str = py::isinstance<py::str>(key);

    if (is_tuple) {
        py::list newkey;
        int count = 0;
        for (const auto& subkey : key) {
            if (py::isinstance<py::str>(subkey)) {
                newkey.append(subkey);
                count++;
            } else {
                auto _key = _unravel_key_to_tuple(subkey.cast<py::object>());
                count += _key.size();
                newkey += _key;
            }
        }
        if (count == 1) {
            return newkey[0];
        }
        return py::tuple(newkey);
    }
    if (is_str) {
        return key;
    } else {
        throw std::runtime_error("key should be a Sequence<NestedKey>");
    }
}

py::list unravel_key_list(const py::list& keys) {
    py::list newkeys;
    for (const auto& key : keys) {
        auto _key = unravel_key(key.cast<py::object>());
        newkeys.append(_key);
    }
    return newkeys;
}

py::list unravel_key_list(const py::tuple& keys) {
    return unravel_key_list(py::list(keys));
}

torch::Tensor _populate_index(torch::Tensor offsets, torch::Tensor offsets_cs) {
    int64_t total = offsets.sum().item<int64_t>();
    torch::Tensor out = torch::empty({total}, torch::dtype(torch::kLong));

    int64_t* out_data = out.data_ptr<int64_t>();
    int64_t cur_offset;
    int64_t count = -1;
    int64_t maxcount = -1;
    int64_t cur = -1;
    int64_t n = offsets.numel();
    for (int i = 0; i < total; ++i) {
        if (cur < n && count == maxcount) {
            cur++;
            count = -1;
            maxcount = offsets[cur].item<int64_t>() - 1;
            cur_offset = offsets_cs[cur].item<int64_t>();
        }
        count++;
        out_data[i] = cur_offset + count;
    }
    return out;
}
py::list _as_shape(torch::Tensor shape_tensor) {
    torch::Tensor shape_tensor_view = shape_tensor.reshape({-1, shape_tensor.size(-1)});
    torch::Tensor out = shape_tensor_view[0].clone();
    torch::Tensor unique = (shape_tensor_view == out).all(0);
    out.masked_fill_(torch::logical_not(unique), -1);
    std::vector<int64_t> shape_vector(shape_tensor.sizes().begin(), shape_tensor.sizes().end() - 1);
    // Extend 'shape_vector' with the values from 'out'.
    auto out_accessor = out.accessor<int64_t, 1>();
    for (int64_t i = 0; i < out_accessor.size(0); ++i) {
        shape_vector.push_back(out_accessor[i]);
    }

    py::list shape = py::cast(shape_vector);
    return shape;
}
