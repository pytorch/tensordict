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
        .def("get", &TensorMap::GetAt, "Get value at index")
        .def("set", &TensorMap::SetTensorAt, "Set tensor as value at index")
        .def("set", &TensorMap::SetMapAt, "Set map as value at index");
        // path
//        .def("set", &TensorMap::SetTensorAtPath, "Set tensor as value at path")
//        .def("get", &TensorMap::GetAtPath, "Get value at path")
//        .def("set", &TensorMap::SetMapAtPath, "Set map as value at path");
}
