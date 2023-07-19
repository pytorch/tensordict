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

torch::Tensor _populate_index(torch::Tensor offsets, torch::Tensor offsets_cs, torch::Tensor index) {
    int64_t total = 0;
    for (int _cur = 0; _cur < index.numel(); ++_cur) {
        int64_t loc = index[_cur].item<int64_t>();
        int64_t incr = offsets[loc].item<int64_t>();
        total += incr;
    }
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
            int64_t cur_idx = index[cur].item<int64_t>();
            maxcount = offsets[cur_idx].item<int64_t>() - 1;
            cur_offset = offsets_cs[cur_idx].item<int64_t>();
        }
        count++;
        out_data[i] = cur_offset + count;
    }
    return out;
}
