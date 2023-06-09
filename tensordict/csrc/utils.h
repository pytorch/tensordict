// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//py::object unravel_keys(const py::object& key, bool make_tuple = false) {
//    if (py::isinstance<py::str>(key)) {
//        if (make_tuple) {
//            return py::make_tuple(key);
//        }
//        return key;
//    }
//    if (py::isinstance<py::tuple>(key)) {
//        py::list newkey;
//        for (const auto& subkey : key) {
//            if (py::isinstance<py::str>(subkey)) {
//                newkey.append(subkey);
//            } else {
//                auto _key = unravel_keys(subkey.cast<py::object>());
//                for (const auto& k : _key) {
//                    newkey.append(k);
//                }
//            }
//        }
//        return py::tuple(newkey);
//    } else {
//        throw std::runtime_error("key should be a Sequence<NestedKey>");
//    }
//}

//py::str unravel(const py::str& key) {
////    return py::make_tuple(key);
//    return key;
//}
//
//py::tuple unravel(const py::tuple& key) {
//    py::list newkey;
//    for (const auto& subkey : key) {
//        if (py::isinstance<py::str>(subkey)) {
//            newkey.append(subkey);
//        } else {
////            auto _key = unravel_keys_tuple(subkey);
//            auto _key = unravel(subkey.cast<py::tuple>());
//            for (const auto& k : _key) {
//                newkey.append(k);
//            }
//        }
//    }
//    return py::tuple(newkey);
//}

// This is the fastest implementation. Overaloading slows down str -> str
py::object unravel_keys(const py::object& key) {
    if (py::isinstance<py::tuple>(key)) {
        py::list newkey;
        for (const auto& subkey : key) {
            if (py::isinstance<py::str>(subkey)) {
                newkey.append(subkey);
            } else {
                auto _key = unravel_keys(subkey.cast<py::object>());
                for (const auto& k : _key) {
                    newkey.append(k);
                }
            }
        }
        return py::tuple(newkey);
    } else {
        return key;
    }
}
