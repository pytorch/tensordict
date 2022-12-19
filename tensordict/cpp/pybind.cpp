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

PYBIND11_MODULE(_tensormap, m) {
    py::class_<TensorMap, std::shared_ptr<TensorMap> >(m, "TensorMap")
        .def(py::init<>())
        .def("__getitem__", &TensorMap::GetAt, "Get value at index")
        .def("__setitem__", &TensorMap::SetTensorAt, "Set tensor as value at index")
        .def("__setitem__", &TensorMap::SetMapAt, "Set map as value at index")
        // path
        .def("__getitem__", &TensorMap::GetAtPath, "Get value at path")
        .def("__setitem__", &TensorMap::SetTensorAtPath, "Set tensor as value at path")
        .def("__setitem__", &TensorMap::SetMapAtPath, "Set map as value at path")
        // keys
        .def("keys", &TensorMap::GetKeys, py::arg("includeNested") = false, py::arg("includeNested") = false, "Get Keys");
}
