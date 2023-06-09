// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>

#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_tensordict, m) {
  m.def("unravel_keys", py::overload_cast<const py::str&>(&unravel), py::arg("key"));
  m.def("unravel_keys", py::overload_cast<const py::tuple&>(&unravel), py::arg("key"));
//  m.def("unravel_keys", &unravel_keys, py::arg("key"), py::arg("make_tuple") = false);
//  m.def("unravel_keys", &unravel_keys, py::arg("key"));
}
