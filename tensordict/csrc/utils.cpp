/* @nolint */
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.h"

namespace py = pybind11;

py::tuple _unravel_key_to_tuple(const py::object &key) {
  std::stack<py::object> stack;
  stack.push(key);
  py::list result;

  while (!stack.empty()) {
    py::object current = stack.top();
    stack.pop();

    if (py::isinstance<py::tuple>(current)) {
      py::tuple current_tuple = current.cast<py::tuple>();
      for (ssize_t i = current_tuple.size() - 1; i >= 0; --i) {
        stack.push(current_tuple[i]);
      }
    } else if (py::isinstance<py::str>(current)) {
      result.append(current);
    } else {
      return py::make_tuple();  // Return empty tuple if any non-string, non-tuple is encountered
    }
  }

  return py::tuple(result);
}

py::object unravel_key(const py::object &key) {
  if (!py::isinstance<py::tuple>(key) && !py::isinstance<py::str>(key)) {
    throw std::runtime_error("key should be a Sequence<NestedKey>");
  }

  std::stack<py::object> stack;
  stack.push(key);
  py::list result;
  int count = 0;

  while (!stack.empty()) {
    py::object current = stack.top();
    stack.pop();

    if (py::isinstance<py::tuple>(current)) {
      py::tuple current_tuple = current.cast<py::tuple>();
      for (ssize_t i = current_tuple.size() - 1; i >= 0; --i) {
        stack.push(current_tuple[i]);
      }
    } else if (py::isinstance<py::str>(current)) {
      result.append(current);
      count++;
    }
  }

  if (count == 1) {
    return result[0];
  }
  return py::tuple(result);
}

py::list unravel_key_list(const py::list &keys) {
  py::list newkeys;
  for (const auto &key : keys) {
    auto _key = unravel_key(key.cast<py::object>());
    newkeys.append(_key);
  }
  return newkeys;
}

py::list unravel_key_list(const py::tuple &keys) {
  return unravel_key_list(py::list(keys));
}
