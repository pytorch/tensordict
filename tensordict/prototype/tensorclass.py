import dataclasses
import functools
import re
import typing
from dataclasses import dataclass
from platform import python_version
from textwrap import indent
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from packaging import version

from tensordict import MetaTensor
from tensordict.tensordict import _accepted_classes, TensorDict, TensorDictBase
from tensordict.utils import DEVICE_TYPING

from torch import Tensor

T = typing.TypeVar("T", bound=TensorDictBase)
PY37 = version.parse(python_version()) < version.parse("3.8")

# For __future__.annotations, we keep a dict of str -> class to call the class based on the string
CLASSES_DICT = {}

# Regex precompiled patterns
OPTIONAL_PATTERN = re.compile(r"Optional\[(.*?)\]")
UNION_PATTERN = re.compile(r"Union\[(.*?)\]")


class _TensorClassMeta(type):
    def __new__(cls, clsname, bases, attrs, cls_repr=None):
        datacls = bases[0]
        attrs["_cls_repr"] = cls_repr
        for attr in datacls.__dataclass_fields__:
            if attr == "batch_size":
                continue
            if attr in dir(TensorDict):
                raise AttributeError(
                    f"Attribute name {attr} can't be used with @tensorclass"
                )
        return super().__new__(cls, datacls.__name__, bases, attrs)

    def __repr__(self):
        if self._cls_repr is not None:
            return self._cls_repr
        return super().__repr__()


def is_tensorclass(obj):
    """Returns True if obj is either a tensorclass or an instance of a tensorclass"""
    cls = obj if isinstance(obj, type) else type(obj)
    return dataclasses.is_dataclass(cls) and isinstance(cls, _TensorClassMeta)


def tensorclass(cls: T) -> T:
    """A decorator to create :obj:`tensorclass` classes.

    :obj:`tensorclass` classes are specialized :obj:`dataclass` instances that
    can execute some pre-defined tensor operations out of the box, such as
    indexing, item assignment, reshaping, casting to device or storage and many
    others.

    Examples:
        >>> from tensordict.prototype import tensorclass
        >>> import torch
        >>> from typing import Optional
        >>>
        >>> @tensorclass
        ... class MyData:
        ...     X: torch.Tensor
        ...     y: torch.Tensor
        ...     def expand_and_mask(self):
        ...         X = self.X.unsqueeze(-1).expand_as(self.y)
        ...         X = X[self.y]
        ...         return X
        ...
        >>> data = MyData(
        ...     X=torch.ones(3, 4, 1),
        ...     y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
        ...     batch_size=[3, 4])
        >>> print(data)
        MyData(
            X=Tensor(torch.Size([3, 4, 1]), dtype=torch.float32),
            y=Tensor(torch.Size([3, 4, 2, 2]), dtype=torch.bool),
            batch_size=[3, 4],
            device=None,
            is_shared=False)
        >>> print(data.expand_and_mask())
        tensor([])

    It is also possible to nest tensorclasses instances within each other:
        Examples:
        >>> from tensordict.prototype import tensorclass
        >>> import torch
        >>> from typing import Optional
        >>>
        >>> @tensorclass
        ... class NestingMyData:
        ...     nested: MyData
        ...
        >>> nesting_data = NestingMyData(nested=data, batch_size=[3, 4])
        >>> # although the data is stored as a TensorDict, the type hint helps us
        >>> # to appropriately cast the data to the right type
        >>> assert isinstance(nesting_data.nested, type(data))


    """
    TD_HANDLED_FUNCTIONS: Dict = {}

    name = cls.__name__
    # by capturing the representation of the original class, the representation of the
    # generated class can be made the same, including preserving information about
    # where the original class was defined
    cls_repr = repr(cls)
    datacls = dataclass(cls)

    EXPECTED_KEYS = set(datacls.__dataclass_fields__)

    class _TensorClass(datacls, metaclass=_TensorClassMeta, cls_repr=cls_repr):
        def __init__(self, *args, _tensordict=None, **kwargs):
            if (args or kwargs) and _tensordict is not None:
                raise ValueError("Cannot pass both args/kwargs and _tensordict.")

            if _tensordict is not None:
                if not all(key in EXPECTED_KEYS for key in _tensordict.keys()):
                    raise ValueError(
                        f"Keys from the tensordict ({set(_tensordict.keys())}) must correspond to the class attributes ({EXPECTED_KEYS})."
                    )
                input_dict = {key: None for key in _tensordict.keys()}
                super().__init__(**input_dict)
                self.tensordict = _tensordict
            else:
                device = kwargs.pop("device", None)
                if "batch_size" not in kwargs:
                    raise TypeError("Missing keyword argument batch_size")
                batch_size = kwargs.pop("batch_size")

                for value, key in zip(args, self.__dataclass_fields__):
                    if key in kwargs:
                        raise ValueError(f"The key {key} is already set in kwargs")
                    kwargs[key] = value
                args = []
                kwargs = self._set_default_values(kwargs)
                new_args = [None for _ in args]
                new_kwargs = {key: None for key in kwargs}

                super().__init__(*new_args, **new_kwargs)

                self.tensordict = TensorDict(
                    {
                        key: _get_typed_value(value)
                        for key, value in kwargs.items()
                        if key not in ("batch_size",)
                    },
                    batch_size=batch_size,
                    device=device,
                )

        @classmethod
        def _build_from_tensordict(cls, tensordict):
            return cls(_tensordict=tensordict)

        def _set_default_values(self, kwargs):
            for key in self.__dataclass_fields__:
                default = self.__dataclass_fields__[key].default
                default_factory = self.__dataclass_fields__[key].default_factory
                if not isinstance(default_factory, dataclasses._MISSING_TYPE):
                    default = default_factory
                if default is not None:
                    kwargs.setdefault(key, default)
            return kwargs

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
                expected_type = datacls.__dataclass_fields__[item].type
                out = _get_typed_output(out, expected_type)
                return out
            return super().__getattribute__(item)

        def __setattr__(self, key, value):
            if "tensordict" not in self.__dict__ or key in ("batch_size", "device"):
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
            if not callable(res):
                return res
            func = res

            def wrapped_func(*args, **kwargs):
                res = func(*args, **kwargs)
                if isinstance(res, TensorDictBase):
                    new = _TensorClass(_tensordict=res)
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
            return _TensorClass(_tensordict=res)  # device=res.device)

        def __setitem__(self, item, value):
            if isinstance(item, str) or (
                isinstance(item, tuple)
                and all(isinstance(_item, str) for _item in item)
            ):
                raise ValueError("Invalid indexing arguments.")
            if not isinstance(value, _TensorClass):
                raise ValueError(
                    "__setitem__ is only allowed for same-class assignement"
                )
            self.tensordict[item] = value.tensordict

        def __repr__(self) -> str:
            fields = _all_td_fields_as_str(self.tensordict)
            field_str = fields
            batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
            device_str = indent(f"device={self.device}", 4 * " ")
            is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
            string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
            return f"{name}(\n{string})"

        def to_tensordict(self) -> TensorDict:
            """Convert the tensorclass into a regular TensorDict.

            Makes a copy of all entries. Memmap and shared memory tensors are converted to
            regular tensors.

            Returns:
                A new TensorDict object containing the same values as the tensorclass.
            """
            return self.tensordict.to_tensordict()

        @property
        def device(self):
            return self.tensordict.device

        @device.setter
        def device(self, value: DEVICE_TYPING) -> None:
            raise RuntimeError(
                "device cannot be set using tensorclass.device = device, "
                "because device cannot be updated in-place. To update device, use "
                "tensorclass.to(new_device), which will return a new tensorclass "
                "on the new device."
            )

        @property
        def batch_size(self) -> torch.Size:
            return self.tensordict.batch_size

        @batch_size.setter
        def batch_size(self, new_size: torch.Size) -> None:
            self.tensordict._batch_size_setter(new_size)

    def implements_for_tdc(torch_function: Callable) -> Callable:
        """Register a torch function override for _TensorClass."""

        @functools.wraps(torch_function)
        def decorator(func):
            TD_HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    @implements_for_tdc(torch.unbind)
    def _unbind(tdc, dim):
        tensordicts = torch.unbind(tdc.tensordict, dim)
        out = [_TensorClass(_tensordict=td) for td in tensordicts]
        return out

    @implements_for_tdc(torch.full_like)
    def _full_like(tdc, fill_value):
        tensordict = torch.full_like(tdc.tensordict, fill_value)
        out = _TensorClass(_tensordict=tensordict)
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
        out = _TensorClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.squeeze)
    def _squeeze(tdc):
        tensordict = torch.squeeze(tdc.tensordict)
        out = _TensorClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.unsqueeze)
    def _unsqueeze(tdc, dim=0):
        tensordict = torch.unsqueeze(tdc.tensordict, dim)
        out = _TensorClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.permute)
    def _permute(tdc, dims):
        tensordict = torch.permute(tdc.tensordict, dims)
        out = _TensorClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.split)
    def _split(tdc, split_size_or_sections, dim=0):
        tensordicts = torch.split(tdc.tensordict, split_size_or_sections, dim)
        out = [_TensorClass(_tensordict=td) for td in tensordicts]
        return out

    @implements_for_tdc(torch.stack)
    def _stack(list_of_tdc, dim):
        tensordict = torch.stack([tdc.tensordict for tdc in list_of_tdc], dim)
        out = _TensorClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.cat)
    def _cat(list_of_tdc, dim):
        tensordict = torch.cat([tdc.tensordict for tdc in list_of_tdc], dim)
        out = _TensorClass(_tensordict=tensordict)
        return out

    CLASSES_DICT[name] = _TensorClass
    return _TensorClass


def _get_typed_value(value):
    if isinstance(value, _accepted_classes + (dataclasses._MISSING_TYPE,)):
        return value
    elif type(value) in CLASSES_DICT.values():
        return value.tensordict
    else:
        raise ValueError(f"{type(value)} is not an accepted class")


def _get_typed_output(out, expected_type):
    # from __future__ import annotations turns types in stings. For those we use CLASSES_DICT.
    # Otherwise, if the output is some TensorDictBase subclass, we check the type and if it
    # does not match, we map it. In all other cases, just return what has been gathered.
    if isinstance(expected_type, str) and expected_type in CLASSES_DICT:
        out = CLASSES_DICT[expected_type](_tensordict=out)
    elif (
        isinstance(expected_type, type)
        and not isinstance(out, expected_type)
        and isinstance(out, TensorDictBase)
    ):
        out = expected_type(_tensordict=out)
    elif isinstance(out, TensorDictBase):
        dest_dtype = _check_td_out_type(expected_type)
        if dest_dtype is not None:
            print(dest_dtype)
            out = dest_dtype(_tensordict=out)

    return out


def _single_td_field_as_str(key, item: MetaTensor, tensordict):
    if item.is_tensordict():
        return f"{key}={repr(tensordict[key])}"
    return f"{key}={item.get_repr()}"


def _all_td_fields_as_str(td: TensorDictBase) -> str:
    return indent(
        ",\n".join(
            sorted(
                [
                    _single_td_field_as_str(key, item, td)
                    for key, item in td.items_meta()
                ]
            )
        ),
        4 * " ",
    )


def _check_td_out_type(field_def):
    if PY37:
        field_def = str(field_def)
    if isinstance(field_def, str):
        optional_match = OPTIONAL_PATTERN.search(field_def)
        union_match = UNION_PATTERN.search(field_def)
        if optional_match is not None:
            args = [optional_match.group(1)]
        elif union_match is not None:
            args = union_match.group(1).split(", ")
        else:
            args = None
        if args:
            args = [arg for arg in args if arg not in ("NoneType",)]
        # skip all Any or TensorDict or Optional[TensorDict] or Union[TensorDict] or Optional[Any]
        if (args is None and (field_def == "Any" or "TensorDict" in field_def)) or (
            args and len(args) == 1 and args[0] == "Any"
        ):
            return None
        if args and len(args) == 1 and "TensorDict" in args[0]:
            return None
        elif args:
            # remove the NoneType from args
            if len(args) == 1 and args[0] in CLASSES_DICT:
                return CLASSES_DICT[args[0]]
            if len(args) == 1 and ("TensorDict" in args[0] or "Any" == args[0]):
                return None
            else:
                raise TypeError(
                    f"{field_def} has args {args} which can't be deterministically cast."
                )
        elif args is None:
            return None
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
