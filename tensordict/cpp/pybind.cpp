// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <string>

#include <memory>

#include "tensormap.h"

namespace py = pybind11;

PYBIND11_MODULE(_tensor_map_cpp, m) {
    py::class_<tensordict::TensorMap>(m, "TensorMap")
        .def(py::init<>())
        .def("get", py::overload_cast<std::string>(&tensordict::TensorMap::get), "Get value at index")
        .def("get", py::overload_cast<std::vector<std::string>>(&tensordict::TensorMap::get), "Get value at path")
        .def("set", py::overload_cast<std::string, torch::Tensor>(&tensordict::TensorMap::set), "Set tensor as value at index")
        .def("set", py::overload_cast<std::string, tensordict::TensorMap>(&tensordict::TensorMap::set), "Set map as value at index")
        .def("set", py::overload_cast<std::vector<std::string>, torch::Tensor>(&tensordict::TensorMap::set), "Set tensor as value at path")
        .def("set", py::overload_cast<std::vector<std::string>, tensordict::TensorMap>(&tensordict::TensorMap::set), "Set map as value at path");
}
