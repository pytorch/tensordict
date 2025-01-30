# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import orjson as json
from typing import Callable, Any


TDTYPE_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}

class StructDtype:
    # def __new__(cls, map=None):
    #     if isinstance(map, StructDtype):
    #         return map
    #     return super().__new__(cls)
    def __init__(self, map=None):
        if map is None:
            map = {}
        assert isinstance(map, dict)
        self._maps = map

    @classmethod
    def from_td(cls, data: "TensorDictBase"):
        from tensordict.base import _is_tensor_collection
        self = cls()
        map = self._maps
        stack = deque()
        stack.append((self, data))
        while len(stack):
            sdtype, local_data = stack.popleft()
            map = sdtype._maps
            # TODO: handle lazy stacks here
            for k, v in local_data.items():
                cls = type(v)
                if _is_tensor_collection(cls):
                    # TODO: handle different dtypes here
                    # TODO: handle LazyStacks here
                    newmap = map[k] = StructDtype({})
                    stack.append((newmap, v))
                else:
                    map[k] = {
                        "shape": v.shape,
                        "dtype": v.dtype,
                    }
        return self

    def items(self, include_nested: bool=False, leaves_only: bool=False):
        stack = deque()
        stack.append(self)
        while len(stack):
            node = stack.popleft()
            for k, v in node._maps.items():
                if isinstance(v, StructDtype):
                    if include_nested:
                        stack.append(v)
                    if not leaves_only:
                        yield (k, v)
                else:
                    yield k, v

    def values(self, include_nested: bool=False, leaves_only: bool=False):
        yield from (_, v in self.items(include_nested=include_nested, leaves_only=leaves_only))

    def keys(self, include_nested: bool=False, leaves_only: bool=False):
        yield from (k, _ in self.items(include_nested=include_nested, leaves_only=leaves_only))

    # def json(self):
    #     return json.dumps(metadata_dict)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in TDTYPE_HANDLED_FUNCTIONS:
            return NotImplemented
        return TDTYPE_HANDLED_FUNCTIONS[func](*args, **kwargs)


    @classmethod
    def view(cls, tensor, dtype):
        from tensordict import TensorDict
        ns = []
        shapes = []
        dts = []
        keys = []
        stack = deque()
        stack.append((dtype.items(), ()))
        tensor_itemsize = tensor.dtype.itemsize
        while len(stack):
            items, prefix = stack.popleft()
            for k, dt in items:
                currentk = prefix + (k,)
                if isinstance(dt, StructDtype):
                    stack.append((dt.items(), currentk))
                    continue
                assert currentk not in keys, (currentk, keys)
                keys.append(currentk)
                s = dt["shape"]
                dt = dt["dtype"]
                shapes.append(s)
                dts.append(dt)
                nelts = (dt.itemsize * s.numel()) // tensor_itemsize
                ns.append(nelts)

        return TensorDict({k: v.view(dt).view(shape) for k, v, dt, shape in zip(keys, tensor.split(ns), dts, shapes, strict=True)})
