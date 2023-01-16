import dataclasses
import functools
import re
import typing
import collections
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
# For nested usecase, we keep dict to save key,value pairs of non-tensor data
TENSOR_CLASS_NON_TENSOR_DATA = collections.defaultdict(dict)

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
    others. Tensorclass objects can also save non-tensor data but all the operations
    are performed only on tensor attributes

    Examples:
        >>> from tensordict.prototype import tensorclass
        >>> import torch
        >>> from typing import Optional
        >>>
        >>> @tensorclass
        ... class MyData:
        ...     X: torch.Tensor
        ...     y: torch.Tensor
        ...     z: str
        ...     def expand_and_mask(self):
        ...         X = self.X.unsqueeze(-1).expand_as(self.y)
        ...         X = X[self.y]
        ...         return X
        ...
        >>> data = MyData(
        ...     X=torch.ones(3, 4, 1),
        ...     y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
        ...     z='test_tensorclass'
        ...     batch_size=[3, 4])
        >>> print(data)
        MyData(
            X=Tensor(torch.Size([3, 4, 1]), dtype=torch.float32),
            y=Tensor(torch.Size([3, 4, 2, 2]), dtype=torch.bool),
            z='test_tensorclass',
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

    cls.__init__ = _init_wrapper(cls.__init__, expected_keys)
    cls._build_from_tensordict = classmethod(_build_from_tensordict)
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

    CLASSES_DICT[cls.__name__] = cls
    return cls


def _init_wrapper(init, expected_keys):
    def wrapper(self, *args, batch_size=None, device=None, _tensordict=None, _non_tensordict=None, **kwargs):
        if (args or kwargs) and _tensordict is not None:
            raise ValueError("Cannot pass both args/kwargs and _tensordict.")

        if _tensordict is not None:
            if not all(key in expected_keys for key in _tensordict.keys()):
                raise ValueError(
                    f"Keys from the tensordict ({set(_tensordict.keys())}) must "
                    f"correspond to the class attributes ({expected_keys})."
                )
            # Handling Non-Tensor data
            if _non_tensordict:
                input_dict = _non_tensordict
            else:
                input_dict = {key: None for key in _tensordict.keys()}
            init(self, **input_dict)
            self.tensordict = _tensordict
        else:
            for value, key in zip(args, self.__dataclass_fields__):
                if key in kwargs:
                    raise ValueError(f"The key {key} is already set in kwargs")
                kwargs[key] = value

            for key, field in self.__dataclass_fields__.items():
                if not isinstance(field.default_factory, dataclasses._MISSING_TYPE):
                    default = field.default_factory()
                else:
                    default = field.default
                if default is not None:
                    kwargs.setdefault(key, default)

            new_kwargs = {}
            # Handling tensor, non-tensor and nested tensor classes differently
            for key, val in kwargs.items():
                if isinstance(val, _accepted_classes):
                    new_kwargs[key] = None
                elif is_tensorclass(val):
                    expected_type = self.__dataclass_fields__[key].type
                    TENSOR_CLASS_NON_TENSOR_DATA[expected_type] = {
                        k: v
                        for k, v in val.__dict__.items()
                        if not isinstance(v, _accepted_classes) and not is_tensorclass(v)
                    }
                    new_kwargs[key] = None
                else:
                    new_kwargs[key] = val

            init(self, **new_kwargs)
            self.tensordict = TensorDict(
                {
                    key: _get_typed_value(value)
                    for key, value in kwargs.items()
                    if key not in ("batch_size",) and (isinstance(value, _accepted_classes) or is_tensorclass(value))
                },
                batch_size=batch_size,
                device=device,
            )

    return wrapper


def _build_from_tensordict(cls, tensordict, non_tensordict):
    return cls(_tensordict=tensordict, _non_tensordict=non_tensordict)


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
        if "tensordict" not in self.__dict__ or key in ("batch_size", "device") or \
                    (not isinstance(value, _accepted_classes) and not is_tensorclass(value)):
            return setattr_(self, key, value)
        if key not in expected_keys:
            raise AttributeError(
                f"Cannot set the attribute '{key}', expected attributes are {expected_keys}."
            )
        if type(value) in CLASSES_DICT.values():
            value = value.__dict__["tensordict"]
        self.__dict__["tensordict"][key] = value
        assert self.__dict__["tensordict"][key] is value

    return wrapper


def _getattr(self, attr):
    res = getattr(self.tensordict, attr)
    if not callable(res):
        return res
    func = res
    non_tensor_dict = _get_non_tensor_dict(self.__dict__)

    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            new = self.__class__(_tensordict=res, _non_tensordict=non_tensor_dict)
            return new
        else:
            return res

    return wrapped_func


def _getitem(self, item):
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    tensor_res = self.tensordict[item]
    non_tensor_res = _get_non_tensor_dict(self.__dict__)
    return self.__class__(_tensordict=tensor_res, _non_tensordict=non_tensor_res)  # device=res.device)


def _setitem(self, item, value):
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    if not isinstance(value, self.__class__):
        raise ValueError("__setitem__ is only allowed for same-class assignement")
    for key, val in self.__dict__.items():
        if not isinstance(val, _accepted_classes):
            if val and val != value.__dict__[key]:
                raise ValueError(
                    f"Expecting {repr(val)} for {key} instead got {repr(value.__dict__[key])}"
                )
    self.tensordict[item] = value.tensordict


def _repr(self) -> str:
    fields = _all_td_fields_as_str(self.tensordict)
    field_str = fields
    non_tensor_fields = _all_non_td_fields_as_str(self.__dict__)
    batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
    device_str = indent(f"device={self.device}", 4 * " ")
    is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
    if len(non_tensor_fields):
        non_tensor_field_str = indent(",\n".join(non_tensor_fields), 4 * " ", )
        string = ",\n".join([field_str, non_tensor_field_str, batch_size_str, device_str, is_shared_str])
    else:
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


def _unbind(tdc, dim):
    tensordicts = torch.unbind(tdc.tensordict, dim)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = [tdc.__class__(_tensordict=td, _non_tensordict=non_tensor_dict) for td in tensordicts]
    return out


def _full_like(tdc, fill_value):
    tensordict = torch.full_like(tdc.tensordict, fill_value)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = tdc.__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
    return out


def _zeros_like(tdc):
    return _full_like(tdc, 0.0)


def _ones_like(tdc):
    return _full_like(tdc, 1.0)


def _clone(tdc):
    tensordict = torch.clone(tdc.tensordict)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = tdc.__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
    return out


def _squeeze(tdc):
    tensordict = torch.squeeze(tdc.tensordict)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = tdc.__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
    return out


def _unsqueeze(tdc, dim=0):
    tensordict = torch.unsqueeze(tdc.tensordict, dim)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = tdc.__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
    return out


def _permute(tdc, dims):
    tensordict = torch.permute(tdc.tensordict, dims)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = tdc.__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
    return out


def _split(tdc, split_size_or_sections, dim=0):
    tensordicts = torch.split(tdc.tensordict, split_size_or_sections, dim)
    non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
    out = [tdc.__class__(_tensordict=td, _non_tensordict=non_tensor_dict) for td in tensordicts]
    return out


def _stack(list_of_tdc, dim):
    if _validate_non_tensor_data(list_of_tdc):
        tensordict = torch.stack([tdc.tensordict for tdc in list_of_tdc], dim)
        tdc = list_of_tdc[0]
        non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
        out = list_of_tdc[0].__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
        return out
    else:
        raise ValueError(
            f"Cannot stack different tensor classes"
        )


def _cat(list_of_tdc, dim):
    if _validate_non_tensor_data(list_of_tdc):
        tensordict = torch.cat([tdc.tensordict for tdc in list_of_tdc], dim)
        tdc = list_of_tdc[0]
        non_tensor_dict = _get_non_tensor_dict(tdc.__dict__)
        out = list_of_tdc[0].__class__(_tensordict=tensordict, _non_tensordict=non_tensor_dict)
        return out
    else:
        raise ValueError(
            f"Cannot concatenate different tensor classes"
        )


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
    non_tensor_dict = None
    # Handling nested tensorclasses 
    if expected_type in TENSOR_CLASS_NON_TENSOR_DATA:
        non_tensor_dict = TENSOR_CLASS_NON_TENSOR_DATA[expected_type]
    if isinstance(expected_type, str) and expected_type in CLASSES_DICT:
        out = CLASSES_DICT[expected_type](_tensordict=out, _non_tensordict=non_tensor_dict)
    elif (
        isinstance(expected_type, type)
        and not isinstance(out, expected_type)
        and isinstance(out, TensorDictBase)
    ):
        out = expected_type(_tensordict=out, _non_tensordict=non_tensor_dict)
    elif isinstance(out, TensorDictBase):
        dest_dtype = _check_td_out_type(expected_type)
        if dest_dtype is not None:
            print(dest_dtype)
            out = dest_dtype(_tensordict=out, _non_tensordict=non_tensor_dict)

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


def _all_non_td_fields_as_str(dict) -> list:
    result = []
    for key, val in dict.items():
        if key not in 'tensordict' and val:
            result.append(f"{key}={repr(val)}")

    return result


def _get_non_tensor_dict(src_dict) -> dict:
    non_tensor_dict = {}
    for key, val in src_dict.items():
        if key != 'tensordict':
            non_tensor_dict[key] = val
    return non_tensor_dict


def _validate_non_tensor_data(list_tds) -> bool:
    list_tds_copy = list_tds.copy()
    td = list_tds_copy.pop()
    for key, val in td.__dict__.items():
        if key != 'tensordict' and val:
            for tds in list_tds_copy:
                if val != tds.__dict__[key]:
                    raise ValueError(
                        f"{repr(val)} and {repr(tds.__dict__[key])} for the attribute {repr(key)} are not matching"
                    )

    return True


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
