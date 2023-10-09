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
  m.def("unravel_keys", &unravel_key, py::arg("key")); // for bc compat
  m.def("unravel_key", &unravel_key, py::arg("key"));
  m.def("_unravel_key_to_tuple", &_unravel_key_to_tuple, py::arg("key"));
  m.def("unravel_key_list",
        py::overload_cast<const py::list &>(&unravel_key_list),
        py::arg("keys"));
  m.def("unravel_key_list",
        py::overload_cast<const py::tuple &>(&unravel_key_list),
        py::arg("keys"));
}
