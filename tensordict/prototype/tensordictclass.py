import functools
from dataclasses import dataclass, field, make_dataclass
from platform import python_version
from typing import Callable, Dict, Optional, Tuple

import torch

from etils.array_types import typing
from packaging import version

from tensordict.tensordict import _accepted_classes, TensorDict, TensorDictBase

from torch import Tensor

PY37 = version.parse(python_version()) < version.parse("3.8")

# For __future__.annotations, we keep a dict of str -> class to call the class based on the string
CLASSES_DICT = {}


def tensordictclass(cls):
    TD_HANDLED_FUNCTIONS: Dict = {}

    name = cls.__name__
    datacls = make_dataclass(
        name,
        bases=(dataclass(cls),),
        fields=[("batch_size", torch.Size, field(default_factory=list))],
    )

    class _TensorDictClass(datacls):
        # TODO: (1) check type annotations and raise errors if Optional, Any or Union (?)
        # TODO: (2) optionally check that the keys of the _tensordict match the fields of the datacls using dataclasses.fields
        def __init__(self, *args, _tensordict=None, **kwargs):
            if _tensordict is not None:
                input_dict = {key: None for key in _tensordict.keys()}
                datacls.__init__(self, **input_dict, batch_size=_tensordict.batch_size)
                if args or kwargs:
                    raise ValueError("Cannot pass both args/kwargs and _tensordict.")
                self.tensordict = _tensordict
            else:
                # should we remove?
                if "batch_size" not in kwargs:
                    raise Exception(
                        "Keyword argument 'batch_size' is required for TensorDictClass."
                    )
                new_args = [None for _ in args]
                new_kwargs = {
                    key: None if key != "batch_size" else value
                    for key, value in kwargs.items()
                }
                datacls.__init__(self, *new_args, **new_kwargs)

                attributes = [key for key in self.__dict__ if key != "batch_size"]

                for attr in attributes:
                    if attr in dir(TensorDict):
                        raise Exception(
                            f"Attribute name {attr} can't be used for TensorDictClass"
                        )

                self.tensordict = TensorDict(
                    {
                        key: value
                        if isinstance(value, _accepted_classes)
                        else value.tensordict
                        for key, value in kwargs.items()
                        if key not in ("batch_size",)
                    },
                    batch_size=kwargs["batch_size"],
                )

        @staticmethod
        def _build_from_tensordict(tensordict):
            return _TensorDictClass(_tensordict=tensordict)

        @classmethod
        def __torch_function__(
            cls,
            func: Callable,
            types,
            args: Tuple = (),
            kwargs: Optional[dict] = None,
        ) -> Callable:
            if kwargs is None:
                kwargs = {}
            if func not in TD_HANDLED_FUNCTIONS or not all(
                issubclass(t, (Tensor, cls)) for t in types
            ):
                return NotImplemented
            return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

        def __getattribute__(self, item):
            if (
                not item.startswith("__")
                and "tensordict" in self.__dict__
                and item in self.__dict__["tensordict"].keys()
            ):
                out = self.__dict__["tensordict"][item]
                # from __future__ import annotations turns types in stings. For those we use CLASSES_DICT.
                # Otherwise, if the output is some TensorDictBase subclass, we check the type and if it
                # does not match, we map it. In all other cases, just return what has been gathered.
                field_def = datacls.__dataclass_fields__[item].type
                if isinstance(field_def, str) and field_def in CLASSES_DICT:
                    out = CLASSES_DICT[field_def](_tensordict=out)
                elif (
                    isinstance(field_def, type)
                    and not isinstance(out, field_def)
                    and isinstance(out, TensorDictBase)
                ):
                    out = field_def(_tensordict=out)
                return out
            return super().__getattribute__(item)

        def __getattr__(self, attr):
            res = getattr(self.tensordict, attr)
            if not callable(res):
                return res
            else:
                func = res

                def wrapped_func(*args, **kwargs):
                    res = func(*args, **kwargs)
                    if isinstance(res, TensorDictBase):
                        new = _TensorDictClass(_tensordict=res)
                        return new
                    else:
                        return res

                return wrapped_func

        def __getitem__(self, item):
            res = self.tensordict[item]
            return _TensorDictClass(
                **res,
                batch_size=res.batch_size,
            )  # device=res.device)

    def implements_for_tdc(torch_function: Callable) -> Callable:
        """Register a torch function override for TensorDictClass."""

        @functools.wraps(torch_function)
        def decorator(func):
            TD_HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    @implements_for_tdc(torch.stack)
    def _stack(list_of_tdc, dim):
        tensordict = torch.stack([tdc.tensordict for tdc in list_of_tdc], dim)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.cat)
    def _cat(list_of_tdc, dim):
        tensordict = torch.cat([tdc.tensordict for tdc in list_of_tdc], dim)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    CLASSES_DICT[name] = _TensorDictClass
    return _TensorDictClass
