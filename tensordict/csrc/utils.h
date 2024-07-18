/* @nolint */
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::tuple _unravel_key_to_tuple(const py::object &key);

py::object unravel_key(const py::object &key);

py::list unravel_key_list(const py::list &keys);

py::list unravel_key_list(const py::tuple &keys);
