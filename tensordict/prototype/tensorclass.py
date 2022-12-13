import dataclasses
import functools
import re
import typing
from dataclasses import dataclass, field, make_dataclass
from platform import python_version
from textwrap import indent
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from packaging import version

from tensordict import MetaTensor
from tensordict.tensordict import _accepted_classes, TensorDict, TensorDictBase

from torch import Tensor

PY37 = version.parse(python_version()) < version.parse("3.8")

# For __future__.annotations, we keep a dict of str -> class to call the class based on the string
CLASSES_DICT = {}


def _make_repr(key, item: MetaTensor, tensordict):
    if item.is_tensordict():
        return f"{key}={repr(tensordict[key])}"
    return f"{key}={item.get_repr()}"


def _td_fields(td: TensorDictBase) -> str:
    return indent(
        ",\n".join(
            sorted([_make_repr(key, item, td) for key, item in td.items_meta()])
        ),
        4 * " ",
    )


def tensorclass(cls):
    TD_HANDLED_FUNCTIONS: Dict = {}

    name = cls.__name__

    _datacls = dataclass(cls)

    datacls = make_dataclass(
        name,
        bases=(_datacls,),
        fields=[("batch_size", torch.Size, field(default_factory=list))],
    )

    EXPECTED_KEYS = set(datacls.__dataclass_fields__.keys())

    class _TensorDictClass(datacls):
        def __init__(self, *args, _tensordict=None, **kwargs):
            if _tensordict is not None:
                if args or kwargs:
                    raise ValueError("Cannot pass both args/kwargs and _tensordict.")
                if not all(key in EXPECTED_KEYS for key in _tensordict.keys()):
                    raise ValueError(
                        f"Keys from the tensordict ({set(_tensordict.keys())}) must correspond to the class attributes ({EXPECTED_KEYS - {'batch_size'} })."
                    )
                input_dict = {key: None for key in _tensordict.keys()}
                datacls.__init__(self, **input_dict, batch_size=_tensordict.batch_size)
                self.tensordict = _tensordict
            else:
                # get the default values
                for key in datacls.__dataclass_fields__:
                    default = datacls.__dataclass_fields__[key].default
                    default_factory = datacls.__dataclass_fields__[key].default_factory
                    if not isinstance(default_factory, dataclasses._MISSING_TYPE):
                        default = default_factory
                    if default is not None:
                        kwargs.setdefault(key, default)

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

                def _get_value(value):
                    if isinstance(
                        value, _accepted_classes + (dataclasses._MISSING_TYPE,)
                    ):
                        return value
                    elif type(value) in CLASSES_DICT.values():
                        return value.tensordict
                    else:
                        raise ValueError(f"{type(value)} is not an accepted class")

                self.tensordict = TensorDict(
                    {
                        key: _get_value(value)
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
            def check_td_out_type(field_def):
                if PY37:
                    field_def = str(field_def)
                if isinstance(field_def, str):
                    if re.search(r"Optional\[.*\]", field_def):
                        args = [re.search(r"Optional\[(.*?)\]", field_def).group(1)]
                    elif re.search(r"Union\[.*\]", field_def):
                        args = (
                            re.search(r"Union\[(.*?)\]", field_def).group(1).split(", ")
                        )
                    else:
                        args = None
                    # skip all Any or TensorDict or Optional[TensorDict] or Union[TensorDict] or Optional[Any]
                    if (
                        args is None
                        and (field_def == "Any" or re.search(r"TensorDict", field_def))
                    ) or (args and len(args) == 1 and args[0] == "Any"):
                        return None
                    if args and len(args) == 1 and re.search(r"TensorDict", args[0]):
                        return None
                    elif args:
                        # remove the NoneType from args
                        args = [arg for arg in args if arg not in ("NoneType",)]
                        if len(args) == 1 and args[0] in CLASSES_DICT:
                            return CLASSES_DICT[args[0]]
                        else:
                            raise TypeError(
                                f"{field_def} has args {args} which can't be deterministically cast."
                            )
                    else:
                        raise TypeError(
                            f"{field_def} has args {args} which can't be deterministically cast."
                        )
                else:
                    if typing.get_origin(field_def) is Union:
                        args = typing.get_args(field_def)
                        # remove the NoneType from args
                        args = [arg for arg in args if arg not in (type(None),)]
                        if len(args) == 1 and (
                            typing.Any is not args[0]
                            and args[0] is not TensorDictBase
                            and TensorDictBase not in args[0].__bases__
                        ):
                            # If there is just one type in Union or Optional, we return that type
                            return args[0]
                        elif len(args) == 1 and (
                            typing.Any is args[0]
                            or args[0] is TensorDictBase
                            or TensorDictBase in args[0].__bases__
                        ):
                            # Any or any TensorDictBase subclass are alway ok if alone
                            return None
                        else:
                            raise TypeError(
                                f"{field_def} has args {args} which can't be deterministically cast."
                            )
                    elif (
                        field_def is typing.Any
                        or field_def is TensorDictBase
                        or TensorDictBase in field_def.__bases__
                    ):
                        # Any or any TensorDictBase subclass are alway ok if alone
                        return None
                    else:
                        raise TypeError(f"{field_def} can't be deterministically cast.")

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
                elif isinstance(out, TensorDictBase):
                    dest_dtype = check_td_out_type(field_def)
                    if dest_dtype is not None:
                        print(dest_dtype)
                        out = dest_dtype(_tensordict=out)

                return out
            return super().__getattribute__(item)

        def __setattr__(self, key, value):
            if "tensordict" not in self.__dict__:
                return super().__setattr__(key, value)
            if key not in EXPECTED_KEYS:
                raise AttributeError(
                    f"Cannot set the attribute '{key}', expected attributes are {EXPECTED_KEYS}."
                )
            if type(value) in CLASSES_DICT.values():
                value = value.__dict__["tensordict"]
            self.__dict__["tensordict"][key] = value
            assert self.__dict__["tensordict"][key] is value

        def __getattr__(self, attr):
            res = getattr(self.tensordict, attr)
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
            if isinstance(item, str) or (
                isinstance(item, tuple)
                and all(isinstance(_item, str) for _item in item)
            ):
                raise ValueError("Invalid indexing arguments.")
            res = self.tensordict[item]
            return _TensorDictClass(_tensordict=res)  # device=res.device)

        def __setitem__(self, item, value):
            if isinstance(item, str) or (
                isinstance(item, tuple)
                and all(isinstance(_item, str) for _item in item)
            ):
                raise ValueError("Invalid indexing arguments.")
            if not isinstance(value, _TensorDictClass):
                raise ValueError(
                    "__setitem__ is only allowed for same-class assignement"
                )
            self.tensordict[item] = value.tensordict

        def __repr__(self) -> str:
            fields = _td_fields(self.tensordict)
            field_str = fields
            batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
            device_str = indent(f"device={self.device}", 4 * " ")
            is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
            string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
            return f"{name}(\n{string})"

    def implements_for_tdc(torch_function: Callable) -> Callable:
        """Register a torch function override for TensorDictClass."""

        @functools.wraps(torch_function)
        def decorator(func):
            TD_HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    @implements_for_tdc(torch.unbind)
    def _unbind(tdc, dim):
        tensordicts = torch.unbind(tdc.tensordict, dim)
        out = [_TensorDictClass(_tensordict=td) for td in tensordicts]
        return out

    @implements_for_tdc(torch.full_like)
    def _full_like(tdc, fill_value):
        tensordict = torch.full_like(tdc.tensordict, fill_value)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.zeros_like)
    def _zeros_like(tdc):
        return _full_like(tdc, 0.0)

    @implements_for_tdc(torch.zeros_like)
    def _ones_like(tdc):
        return _full_like(tdc, 1.0)

    @implements_for_tdc(torch.clone)
    def _clone(tdc):
        tensordict = torch.clone(tdc.tensordict)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.squeeze)
    def _squeeze(tdc):
        tensordict = torch.squeeze(tdc.tensordict)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.unsqueeze)
    def _unsqueeze(tdc, dim=0):
        tensordict = torch.unsqueeze(tdc.tensordict, dim)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.permute)
    def _permute(tdc, dims):
        tensordict = torch.permute(tdc.tensordict, dims)
        out = _TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.split)
    def _split(tdc, split_size_or_sections, dim=0):
        tensordicts = torch.split(tdc.tensordict, split_size_or_sections, dim)
        out = [_TensorDictClass(_tensordict=td) for td in tensordicts]
        return out

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
