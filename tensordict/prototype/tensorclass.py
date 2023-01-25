import dataclasses
import functools
import inspect
import re
import typing
from dataclasses import dataclass
from platform import python_version
from textwrap import indent
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from packaging import version

from tensordict.tensordict import (
    _accepted_classes,
    get_repr,
    is_tensordict,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import DEVICE_TYPING

from torch import Tensor

T = typing.TypeVar("T", bound=TensorDictBase)
PY37 = version.parse(python_version()) < version.parse("3.8")

# For __future__.annotations, we keep a dict of str -> class to call the class based on the string
CLASSES_DICT = {}

# Regex precompiled patterns
OPTIONAL_PATTERN = re.compile(r"Optional\[(.*?)\]")
UNION_PATTERN = re.compile(r"Union\[(.*?)\]")


def is_tensorclass(obj):
    """Returns True if obj is either a tensorclass or an instance of a tensorclass"""
    cls = obj if isinstance(obj, type) else type(obj)
    return dataclasses.is_dataclass(cls) and cls.__name__ in CLASSES_DICT


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
    td_handled_functions: Dict = {}

    def implements_for_tdc(torch_function: Callable) -> Callable:
        """Register a torch function override for _TensorClass."""

        @functools.wraps(torch_function)
        def decorator(func):
            td_handled_functions[torch_function] = func
            return func

        return decorator

    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in td_handled_functions or not all(
            issubclass(t, (Tensor, cls)) for t in types
        ):
            return NotImplemented
        return td_handled_functions[func](*args, **kwargs)

    cls = dataclass(cls)
    expected_keys = set(cls.__dataclass_fields__)

    for attr in cls.__dataclass_fields__:
        if attr in dir(TensorDict):
            raise AttributeError(
                f"Attribute name {attr} can't be used with @tensorclass"
            )

    cls.__init__ = _init_wrapper(cls.__init__)
    cls.from_tensordict = classmethod(_from_tensordict_wrapper(expected_keys))
    cls.__torch_function__ = classmethod(__torch_function__)
    cls.__getstate__ = _getstate
    cls.__setstate__ = _setstate
    cls.__getattribute__ = _getattribute_wrapper(cls.__getattribute__)
    cls.__setattr__ = _setattr_wrapper(cls.__setattr__, expected_keys)
    cls.__getattr__ = _getattr
    cls.__getitem__ = _getitem
    cls.__setitem__ = _setitem
    cls.__repr__ = _repr
    cls.__len__ = _len
    cls.__eq__ = __eq__
    cls.__ne__ = __ne__
    cls.to_tensordict = _to_tensordict
    cls.device = property(_device, _device_setter)
    cls.batch_size = property(_batch_size, _batch_size_setter)

    implements_for_tdc(torch.unbind)(_unbind)
    implements_for_tdc(torch.full_like)(_full_like)
    implements_for_tdc(torch.zeros_like)(_zeros_like)
    implements_for_tdc(torch.zeros_like)(_ones_like)
    implements_for_tdc(torch.clone)(_clone)
    implements_for_tdc(torch.squeeze)(_squeeze)
    implements_for_tdc(torch.unsqueeze)(_unsqueeze)
    implements_for_tdc(torch.permute)(_permute)
    implements_for_tdc(torch.split)(_split)
    implements_for_tdc(torch.stack)(_stack)
    implements_for_tdc(torch.cat)(_cat)

    cls.__doc__ = f"{cls.__name__}{inspect.signature(cls)}"

    CLASSES_DICT[cls.__name__] = cls
    return cls


def _init_wrapper(init):
    init_sig = inspect.signature(init)
    params = list(init_sig.parameters.values())
    # drop first entry of params which corresponds to self and isn't passed by the user
    required_params = [p.name for p in params[1:] if p.default is inspect._empty]

    @functools.wraps(init)
    def wrapper(self, *args, batch_size, device=None, **kwargs):
        for value, key in zip(args, self.__dataclass_fields__):
            if key in kwargs:
                raise ValueError(f"The key {key} is already set in kwargs")
            kwargs[key] = value

        for key, field in self.__dataclass_fields__.items():
            if field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                default = field.default
            if default not in (None, dataclasses.MISSING):
                kwargs.setdefault(key, default)

        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            n_missing = len(missing_params)
            raise TypeError(
                f"{self.__class__.__name__}.__init__() missing {n_missing} "
                f"required positional argument{'' if n_missing == 1 else 's'}: "
                f"""{", ".join(f"'{name}'" for name in missing_params)}"""
            )

        self.tensordict = TensorDict(
            {}, batch_size=batch_size, device=device, _run_checks=False
        )
        init(self, **{key: _get_typed_value(value) for key, value in kwargs.items()})

    new_params = [
        inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    wrapper.__signature__ = init_sig.replace(parameters=params + new_params)

    return wrapper


def _from_tensordict_wrapper(expected_keys):
    def wrapper(cls, tensordict):
        if not all(key in expected_keys for key in tensordict.keys()):
            raise ValueError(
                f"Keys from the tensordict ({set(tensordict.keys())}) must "
                f"correspond to the class attributes ({expected_keys})."
            )
        # bypass initialisation. this means we don't incur any overhead creating an
        # empty tensordict and writing values to it. we can skip this because we already
        # have a tensordict to use as the underlying tensordict
        tc = cls.__new__(cls)
        tc.__dict__["tensordict"] = tensordict
        # since we aren't calling the dataclass init method, we need to manually check
        # whether a __post_init__ method has been defined and invoke it if so
        if hasattr(tc, "__post_init__"):
            tc.__post_init__()
        return tc

    return wrapper


def _getstate(self):
    return {"tensordict": self.tensordict}


def _setstate(self, state):
    self.tensordict = state.get("tensordict", None)


def _getattribute_wrapper(getattribute):
    def wrapper(self, item):
        if (
            not item.startswith("__")
            and "tensordict" in self.__dict__
            and item in self.__dict__["tensordict"].keys()
        ):
            out = self.__dict__["tensordict"][item]
            expected_type = self.__dataclass_fields__[item].type
            out = _get_typed_output(out, expected_type)
            return out
        return getattribute(self, item)

    return wrapper


def _setattr_wrapper(setattr_, expected_keys):
    def wrapper(self, key, value):
        if "tensordict" not in self.__dict__ or key in ("batch_size", "device"):
            return setattr_(self, key, value)
        if key not in expected_keys:
            raise AttributeError(
                f"Cannot set the attribute '{key}', expected attributes are {expected_keys}."
            )
        if type(value) in CLASSES_DICT.values():
            value = value.__dict__["tensordict"]
        self.__dict__["tensordict"][key] = value

    return wrapper


def _getattr(self, attr):
    res = getattr(self.tensordict, attr)
    if not callable(res):
        return res
    func = res

    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            new = self.from_tensordict(res)
            return new
        else:
            return res

    return wrapped_func


def _getitem(self, item):
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    res = self.tensordict[item]
    return self.from_tensordict(res)  # device=res.device)


def _setitem(self, item, value):
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    if not isinstance(value, self.__class__):
        raise ValueError("__setitem__ is only allowed for same-class assignement")
    self.tensordict[item] = value.tensordict


def _repr(self) -> str:
    fields = _all_td_fields_as_str(self.tensordict)
    field_str = fields
    batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
    device_str = indent(f"device={self.device}", 4 * " ")
    is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
    string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
    return f"{self.__class__.__name__}(\n{string})"


def _len(self) -> int:
    """Returns the length of first dimension, if there is, otherwise 0."""
    return len(self.tensordict)


def _to_tensordict(self) -> TensorDict:
    """Convert the tensorclass into a regular TensorDict.

    Makes a copy of all entries. Memmap and shared memory tensors are converted to
    regular tensors.

    Returns:
        A new TensorDict object containing the same values as the tensorclass.
    """
    return self.tensordict.to_tensordict()


def _device(self):
    return self.tensordict.device


def _device_setter(self, value: DEVICE_TYPING) -> None:
    raise RuntimeError(
        "device cannot be set using tensorclass.device = device, "
        "because device cannot be updated in-place. To update device, use "
        "tensorclass.to(new_device), which will return a new tensorclass "
        "on the new device."
    )


def _batch_size(self) -> torch.Size:
    return self.tensordict.batch_size


def _batch_size_setter(self, new_size: torch.Size) -> None:
    self.tensordict._batch_size_setter(new_size)


def __eq__(self, other):
    if not isinstance(other, self.__class__):
        return False
    return self.tensordict == other.tensordict


def __ne__(self, other):
    if not isinstance(other, self.__class__):
        return True
    return self.tensordict != other.tensordict


def _unbind(tdc, dim):
    tensordicts = torch.unbind(tdc.tensordict, dim)
    out = [tdc.from_tensordict(td) for td in tensordicts]
    return out


def _full_like(tdc, fill_value):
    tensordict = torch.full_like(tdc.tensordict, fill_value)
    out = tdc.from_tensordict(tensordict)
    return out


def _zeros_like(tdc):
    return _full_like(tdc, 0.0)


def _ones_like(tdc):
    return _full_like(tdc, 1.0)


def _clone(tdc):
    tensordict = torch.clone(tdc.tensordict)
    out = tdc.from_tensordict(tensordict)
    return out


def _squeeze(tdc):
    tensordict = torch.squeeze(tdc.tensordict)
    out = tdc.from_tensordict(tensordict)
    return out


def _unsqueeze(tdc, dim=0):
    tensordict = torch.unsqueeze(tdc.tensordict, dim)
    out = tdc.from_tensordict(tensordict)
    return out


def _permute(tdc, dims):
    tensordict = torch.permute(tdc.tensordict, dims)
    out = tdc.from_tensordict(tensordict)
    return out


def _split(tdc, split_size_or_sections, dim=0):
    tensordicts = torch.split(tdc.tensordict, split_size_or_sections, dim)
    out = [tdc.from_tensordict(td) for td in tensordicts]
    return out


def _stack(list_of_tdc, dim):
    tensordict = torch.stack([tdc.tensordict for tdc in list_of_tdc], dim)
    out = list_of_tdc[0].from_tensordict(tensordict)
    return out


def _cat(list_of_tdc, dim):
    tensordict = torch.cat([tdc.tensordict for tdc in list_of_tdc], dim)
    out = list_of_tdc[0].from_tensordict(tensordict)
    return out


def _get_typed_value(value):
    if isinstance(value, _accepted_classes) or value is dataclasses.MISSING:
        return value
    elif type(value) in CLASSES_DICT.values():
        return value.tensordict
    else:
        raise ValueError(f"{type(value)} is not an accepted class")


def _get_typed_output(out, expected_type):
    # from __future__ import annotations turns types in strings. For those we use CLASSES_DICT.
    # Otherwise, if the output is some TensorDictBase subclass, we check the type and if it
    # does not match, we map it. In all other cases, just return what has been gathered.
    if isinstance(expected_type, str) and expected_type in CLASSES_DICT:
        out = CLASSES_DICT[expected_type].from_tensordict(out)
    elif (
        isinstance(expected_type, type)
        and not isinstance(out, expected_type)
        and isinstance(out, TensorDictBase)
    ):
        out = expected_type.from_tensordict(out)
    elif isinstance(out, TensorDictBase):
        dest_dtype = _check_td_out_type(expected_type)
        if dest_dtype is not None:
            out = dest_dtype.from_tensordict(out)

    return out


def _single_td_field_as_str(key, item, tensordict):
    if is_tensordict(type(item)):
        return f"{key}={repr(tensordict[key])}"
    return f"{key}={get_repr(item)}"


def _all_td_fields_as_str(td: TensorDictBase) -> str:
    return indent(
        ",\n".join(
            sorted([_single_td_field_as_str(key, item, td) for key, item in td.items()])
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
