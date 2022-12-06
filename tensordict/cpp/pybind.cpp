// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>

#include "tensormap.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor_map_cpp, m) {
    py::class_<TensorMap>(m, "TensorMap")
        .def(py::init<>())
        .def("get", py::overload_cast<std::string>(&TensorMap::get), "Get value at index")
        .def("get", py::overload_cast<std::vector<std::string>>(&TensorMap::get), "Get value at path")
        .def("set", py::overload_cast<std::string, torch::Tensor>(&TensorMap::set), "Set tensor as value at index")
        .def("set", py::overload_cast<std::string, TensorMap>(&TensorMap::set), "Set map as value at index")
        .def("set", py::overload_cast<std::vector<std::string>, torch::Tensor>(&TensorMap::set), "Set tensor as value at path")
        .def("set", py::overload_cast<std::vector<std::string>, TensorMap>(&TensorMap::set), "Set map as value at path");
}