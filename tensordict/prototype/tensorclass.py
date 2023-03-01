# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
import inspect
import numbers
import re
import sys
import typing
import warnings
from copy import copy
from dataclasses import dataclass
from textwrap import indent
from typing import Any, Callable, Protocol, Sequence, TypeVar, Union

import tensordict as tensordict_lib

import torch
from tensordict.tensordict import (
    get_repr,
    is_tensor_collection,
    TD_HANDLED_FUNCTIONS,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import DeviceType, NestedKey
from torch import Tensor

T = TypeVar("T", bound=TensorDictBase)
PY37 = sys.version_info < (3, 8)


class TensorClassProtocol(Protocol):
    _tensordict: TensorDictBase
    _non_tensordict: dict[str, Any]


# We keep a dict of str -> class to call the class based on the string
CLASSES_DICT: dict[str, type] = {}

# Regex precompiled patterns
OPTIONAL_PATTERN = re.compile(r"Optional\[(.*?)\]")
UNION_PATTERN = re.compile(r"Union\[(.*?)\]")

# methods where non_tensordict data should be cleared in the return value
_CLEAR_METADATA = {"all", "any"}
# torch functions where we can wrap the corresponding TensorDict version
_TD_PASS_THROUGH = {
    torch.unbind,
    torch.full_like,
    torch.zeros_like,
    torch.ones_like,
    torch.clone,
    torch.squeeze,
    torch.unsqueeze,
    torch.split,
    torch.permute,
    torch.split,
    torch.stack,
    torch.cat,
    torch.gather,
}


def is_tensorclass(obj: type | Any) -> bool:
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
        ...     z: str
        ...     def expand_and_mask(self):
        ...         X = self.X.unsqueeze(-1).expand_as(self.y)
        ...         X = X[self.y]
        ...         return X
        ...
        >>> data = MyData(
        ...     X=torch.ones(3, 4, 1),
        ...     y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
        ...     z="test"
        ...     batch_size=[3, 4])
        >>> print(data)
        MyData(
            X=Tensor(torch.Size([3, 4, 1]), dtype=torch.float32),
            y=Tensor(torch.Size([3, 4, 2, 2]), dtype=torch.bool),
            z="test"
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
    tc_handled_functions: dict[Callable, Callable] = {}

    def implements_for_tdc(torch_function: Callable) -> Callable:
        """Register a torch function override for _TensorClass."""

        @functools.wraps(torch_function)
        def decorator(func: Callable) -> Callable:
            tc_handled_functions[torch_function] = func
            return func

        return decorator

    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if (
            func not in tc_handled_functions and func not in _TD_PASS_THROUGH
        ) or not all(issubclass(t, (Tensor, cls)) for t in types):
            return NotImplemented
        if kwargs is None:
            kwargs = {}
        if func in _TD_PASS_THROUGH:
            if len(args) > 0:
                input_ = args[0]
            else:
                input_ = kwargs["input"] if "input" in kwargs else kwargs["tensors"]
            if isinstance(input_, (list, tuple)):
                tc = input_[0]
                input_ = input_.__class__(tc._tensordict for tc in input_)
            else:
                tc = input_
                input_ = input_._tensordict

            out = kwargs.get("out")
            if out is not None:
                kwargs["out"] = out._tensordict

            res = TD_HANDLED_FUNCTIONS[func](input_, *args[1:], **kwargs)
            if isinstance(res, (list, tuple)):
                return res.__class__(_from_tensordict_with_copy(tc, td) for td in res)
            return _from_tensordict_with_copy(tc, res)

        return tc_handled_functions[func](*args, **kwargs)

    cls = dataclass(cls)
    expected_keys = set(cls.__dataclass_fields__)

    for attr in cls.__dataclass_fields__:
        if attr in dir(TensorDict):
            raise AttributeError(
                f"Attribute name {attr} can't be used with @tensorclass"
            )

    cls.__init__ = _init_wrapper(cls.__init__)
    cls._from_tensordict = classmethod(_from_tensordict_wrapper(expected_keys))
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
    cls.state_dict = _state_dict
    cls.load_state_dict = _load_state_dict
    cls.gather = _gather

    cls.to_tensordict = _to_tensordict
    cls.device = property(_device, _device_setter)
    cls.batch_size = property(_batch_size, _batch_size_setter)

    cls.__doc__ = f"{cls.__name__}{inspect.signature(cls)}"

    CLASSES_DICT[cls.__name__] = cls
    tensordict_lib.tensordict._ACCEPTED_CLASSES += [cls]
    return cls


def _from_tensordict_with_copy(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and a copy of the
    # non_tensordict data
    return tc._from_tensordict(
        tensordict=tensordict, non_tensordict=copy(tc._non_tensordict)
    )


def _from_tensordict_with_none(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and all non_tensordict entries
    # set to None
    return tc._from_tensordict(
        tensordict=tensordict,
        non_tensordict={key: None for key in tc._non_tensordict},
    )


def _init_wrapper(init: Callable) -> Callable:
    init_sig = inspect.signature(init)
    params = list(init_sig.parameters.values())
    # drop first entry of params which corresponds to self and isn't passed by the user
    required_params = [p.name for p in params[1:] if p.default is inspect._empty]

    @functools.wraps(init)
    def wrapper(
        self,
        *args: Any,
        batch_size: Sequence[int] | torch.Size | int,
        device: DeviceType | None = None,
        **kwargs,
    ):
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

        self._tensordict = TensorDict(
            {}, batch_size=batch_size, device=device, _run_checks=False
        )
        # To save non tensor data (Nested tensor classes also go here)
        self._non_tensordict = {}
        init(self, **{key: value for key, value in kwargs.items()})

    new_params = [
        inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    wrapper.__signature__ = init_sig.replace(parameters=params + new_params)

    return wrapper


def _from_tensordict_wrapper(expected_keys):
    def wrapper(cls, tensordict, non_tensordict=None):
        """Tensor class wrapper to instantiate a new tensor class object

        Args:
            tensordict (TensorDict): Dictionary of tensor types
            non_tensordict (dict): Dictionary with non-tensor and nested tensor class objects

        """
        if not isinstance(tensordict, TensorDictBase):
            raise RuntimeError(
                f"Expected a TensorDictBase instance but got {type(tensordict)}"
            )
        # Validating keys of tensordict
        for key in tensordict.keys():
            if key not in expected_keys:
                raise ValueError(
                    f"Keys from the tensordict ({set(tensordict.keys())}) must "
                    f"correspond to the class attributes ({expected_keys})."
                )

        # Validating non-tensor keys and for key clash
        tensor_keys = set(tensordict.keys())
        if non_tensordict is not None:
            for key in non_tensordict.keys():
                if key not in expected_keys:
                    raise ValueError(
                        f"Keys from the non-tensor data ({set(non_tensordict.keys())}) must "
                        f"correspond to the class attributes ({expected_keys})."
                    )
                if key in tensor_keys:
                    raise KeyError(
                        f"{key} is present in both tensor and non-tensor dicts"
                    )
        # bypass initialisation. this means we don't incur any overhead creating an
        # empty tensordict and writing values to it. we can skip this because we already
        # have a tensordict to use as the underlying tensordict
        tc = cls.__new__(cls)
        tc.__dict__["_tensordict"] = tensordict

        tc.__dict__["_non_tensordict"] = (
            non_tensordict if non_tensordict is not None else {}
        )
        # since we aren't calling the dataclass init method, we need to manually check
        # whether a __post_init__ method has been defined and invoke it if so
        if hasattr(tc, "__post_init__"):
            tc.__post_init__()
        return tc

    return wrapper


def _getstate(self) -> dict[str, Any]:
    """Returns a state dict which consists of tensor and non_tensor dicts for serialization.

    Returns:
        dictionary of state of tensor class

    """
    return {"tensordict": self._tensordict, "non_tensordict": self._non_tensordict}


def _setstate(self, state: dict[str, Any]) -> None:
    """Used to set the state of an object using state parameter

    Args:
        state (dict): State parameter to set the object
    """
    self._tensordict = state.get("tensordict", None)
    self._non_tensordict = state.get("non_tensordict", None)


def _getattribute_wrapper(getattribute: Callable) -> Callable:
    """Retrieve the value of an object's attribute or raise AttributeError

    Args:
        item (str) : name of the attribute to retrieve

    Returns:
        value of the attribute

    """

    @functools.wraps(getattribute)
    def wrapper(self, item: str) -> Any:
        if not item.startswith("__"):
            if (
                "_tensordict" in self.__dict__
                and item in self.__dict__["_tensordict"].keys()
            ):
                out = self._tensordict[item]
                expected_type = self.__dataclass_fields__[item].type
                out = _get_typed_output(out, expected_type)
                return out
            elif (
                "_non_tensordict" in self.__dict__
                and item in self.__dict__["_non_tensordict"]
            ):
                out = self._non_tensordict[item]
                return out
        return getattribute(self, item)

    return wrapper


def _setattr_wrapper(setattr_: Callable, expected_keys: set[str]) -> Callable:
    """Set the value of an attribute for the tensor class object

    Args:
        key (str): the name of the attribute to set
        value (any): the value to set for the attribute

    """

    @functools.wraps(setattr_)
    def wrapper(self, key: str, value: Any) -> None:
        if (
            "_tensordict" not in self.__dict__
            or "_non_tensordict" not in self.__dict__
            or key in ("batch_size", "device")
        ):
            return setattr_(self, key, value)
        if key not in expected_keys:
            raise AttributeError(
                f"Cannot set the attribute '{key}', expected attributes are {expected_keys}."
            )

        if isinstance(value, tuple(tensordict_lib.tensordict._ACCEPTED_CLASSES)):
            # Avoiding key clash, honoring the user input to assign tensor type data to the key
            if key in self._non_tensordict.keys():
                del self._non_tensordict[key]
            self._tensordict[key] = value
        else:
            # Avoiding key clash, honoring the user input to assign non-tensor data to the key
            if key in self._tensordict.keys():
                del self._tensordict[key]
            # Saving all non-tensor attributes
            self._non_tensordict[key] = value
        return None

    return wrapper


def _getattr(self, attr: str) -> Any:
    """Retrieve the value of an object's attribute, or a method output if attr is callable

    Args:
        attr: name of the attribute to retrieve or function to compute

    Returns:
        value of the attribute, or a method output applied on the instance

    """
    res = getattr(self._tensordict, attr)
    if not callable(res):
        return res
    func = res

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            if attr.endswith("_"):
                # in-place operation, return the current object
                return self
            elif attr in _CLEAR_METADATA:
                # this is an attribute where copying the metadata makes no sense, e.g.
                # .all or .any, so we replace all values with None
                return self._from_tensordict(
                    res, {k: None for k in self._non_tensordict}
                )
            # create a new tensorclass from res and copy the metadata from self
            return self._from_tensordict(res, copy(self._non_tensordict))
        return res

    return wrapped_func


def _getitem(self, item: NestedKey) -> Any:
    """Retrieve the class object at the given index. Indexing will happen for nested tensors as well

    Args:
       item (int or any other valid index type): index of the object to retrieve

    Returns:
        Tensor class object at the given index

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError(f"Invalid indexing arguments: {item}.")
    tensor_res = self._tensordict[item]
    return _from_tensordict_with_copy(self, tensor_res)  # device=res.device)


def _setitem(self, item: NestedKey, value: Any) -> None:
    """Set the value of the Tensor class object at the given index. Note that there is no strict validation on non-tensor values

    Args:
        item (int or any other valid index type): index of the object to set
        value (any): value to set for the item

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    if is_tensorclass(value) and not isinstance(value, self.__class__):
        self_keys = set().union(self._non_tensordict, self._tensordict.keys())
        value_keys = set().union(value._non_tensordict, value._tensordict.keys())
        if self_keys != value_keys:
            # if tensorclass but different class ensure that all keys are equal
            raise ValueError(
                "__setitem__ is only allowed for same-class or "
                "compatible class (i.e. same members) assignment"
            )

    if isinstance(value, (TensorDictBase, dict)):
        self._tensordict[item] = value
    else:
        # Validating the non-tensor data before setting the item
        for key, val in value._non_tensordict.items():
            # Raise a warning if non_tensor data doesn't match
            if (
                key in self._non_tensordict.keys()
                and val is not self._non_tensordict[key]
            ):
                warnings.warn(
                    f"Meta data at {repr(key)} may or may not be equal, this may result in "
                    f"undefined behaviours",
                    category=UserWarning,
                    stacklevel=2,
                )

        for key in value._tensordict.keys():
            # Making sure that the key-clashes won't happen, if the key is present in tensor data in value
            # we will honor that and remove the key-value pair from non-tensor data
            if key in self._non_tensordict.keys():
                del self._non_tensordict[key]

        self._tensordict[item] = value._tensordict


def _repr(self) -> str:
    """Return a string representation of Tensor class object"""
    fields = _all_td_fields_as_str(self._tensordict)
    field_str = fields
    non_tensor_fields = _all_non_td_fields_as_str(self._non_tensordict)
    batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
    device_str = indent(f"device={self.device}", 4 * " ")
    is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
    if len(non_tensor_fields) > 0:
        non_tensor_field_str = indent(
            ",\n".join(non_tensor_fields),
            4 * " ",
        )
        string = ",\n".join(
            [field_str, non_tensor_field_str, batch_size_str, device_str, is_shared_str]
        )
    else:
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
    return f"{self.__class__.__name__}(\n{string})"


def _len(self) -> int:
    """Returns the length of first dimension, if there is, otherwise 0."""
    return len(self._tensordict)


def _to_tensordict(self) -> TensorDict:
    """Convert the tensorclass into a regular TensorDict.

    Makes a copy of all entries. Memmap and shared memory tensors are converted to
    regular tensors.

    Returns:
        A new TensorDict object containing the same values as the tensorclass.

    """
    td = self._tensordict.to_tensordict()
    return td


def _device(self) -> torch.device:
    """Retrieves the device type of tensor class"""
    return self._tensordict.device


def _device_setter(self, value: DeviceType) -> None:
    raise RuntimeError(
        "device cannot be set using tensorclass.device = device, "
        "because device cannot be updated in-place. To update device, use "
        "tensorclass.to(new_device), which will return a new tensorclass "
        "on the new device."
    )


def _batch_size(self) -> torch.Size:
    """Retrieves the batch size for the tensor class

    Returns:
        batch size (torch.Size)

    """
    return self._tensordict.batch_size


def _batch_size_setter(self, new_size: torch.Size) -> None:
    """Set the value of batch_size

    Args:
        new_size (torch.Size): new_batch size to be set

    """
    self._tensordict._batch_size_setter(new_size)


def _state_dict(self) -> dict[str, Any]:
    """Returns a state_dict dictionary that can be used to save and load data from a tensorclass."""
    state_dict = {"_tensordict": self._tensordict.state_dict()}
    state_dict["_non_tensordict"] = copy(self._non_tensordict)
    return state_dict


def _load_state_dict(self, state_dict: dict[str, Any]):
    """Loads a state_dict attemptedly in-place on the destination tensorclass."""
    for key, item in state_dict.items():
        # keys will never be nested which facilitates everything, but let's
        # double check in case someone does something nasty
        if not isinstance(key, str):
            raise TypeError("Only str keys are allowed when calling load_state_dict.")
        if key == "_non_tensordict":
            for sub_key, sub_item in item.items():
                # sub_item is the state dict of a tensorclass
                if isinstance(sub_item, dict) and "_non_tensordict" in sub_item:
                    raise RuntimeError(
                        "Loading a saved tensorclass on a uninitialized tensorclass is not allowed"
                    )
                else:
                    # check that sub_key is part of the tensorclass
                    if sub_key not in self.__class__.__dataclass_fields__:
                        raise KeyError(
                            f"Key '{sub_key}' wasn't expected in the state-dict."
                        )
                    self._non_tensordict[sub_key] = sub_item
        elif key == "_tensordict":
            for sub_key in item.keys():
                if (
                    sub_key not in self.__class__.__dataclass_fields__
                    and sub_key not in ("__batch_size", "__device")
                ):
                    raise KeyError(
                        f"Key '{sub_key}' wasn't expected in the state-dict."
                    )

            self._tensordict.load_state_dict(item)
        else:
            raise KeyError(f"Key '{key}' wasn't expected in the state-dict.")

    return self


# this function appears to be required even though we fall back on `TensorDict.gather`
# because it's not clear how to deal with methods with an `out` argument generically
def _gather(
    self, dim: int, index: torch.Tensor, out: TensorClassProtocol | None = None
):
    """Gathers values along an axis specified by `dim`.

    Args:
        dim (int): the dimension along which collect the elements
        index (torch.Tensor): a long tensor which number of dimension matches
            the one of the tensordict with only one dimension differring between
            the two (the gathering dimension). Its elements refer to the
            index to be gathered along the required dimension.
        out (TensorDictBase, optional): a destination tensordict. It must
            have the same shape as the index.

    Examples:
        >>> @tensorclass
        ... class MyClass:
        ...     x: torch.Tensor
        ...     z: str
        ...     y: "MyClass1" = None  # future: drop quotes
        ...
        >>> c = MyClass(torch.randn(3, 4), "foo", MyClass(torch.randn(3, 4, 5), "bar", None, batch_size=[3, 4, 5]), batch_size=[3, 4])
        >>> dim = -1
        >>> index = torch.arange(3).expand(3, 3)
        >>> c_gather = c.gather(index=index, dim=dim)
        >>> print(c_gather)

    """
    if dim < 0:
        dim = self.batch_dims + dim

    return _from_tensordict_with_copy(
        self,
        self._tensordict.gather(
            dim=dim, index=index, out=out._tensordict if out is not None else None
        ),
    )


def __eq__(self, other: object) -> bool:
    """Compares the Tensor class object to another object for equality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        False if the objects are of different class types, Tensorclass of boolean
        values for tensor attributes and None for non-tensor attributes

    Examples:
        >>> @tensorclass
        ... class MyClass:
        ...     x: Tensor
        ...     y: "MyClass"
        ...     z: str
        ...
        >>> c1 = MyClass(
        ...     x=torch.randn(3, 4),
        ...     y=MyClass(
        ...         x=torch.randn(3, 4, 1),
        ...         y=None,
        ...         z="bar",
        ...         batch_size=[3, 4, 1],
        ...     ),
        ...     z="foo",
        ...     batch_size=[3, 4],
        ... )
        >>> c2 = c1.clone()
        >>> print(c1 == c2)
        MyClass(
            x=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
            y=MyClass(
                x=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                y=None,
                z=None,
                batch_size=torch.Size([3, 4, 1]),
                device=None,
                is_shared=False),
            z=None,
            batch_size=torch.Size([3, 4]),
            device=None,
            is_shared=False)
        >>> assert (c1 == c2).all()
        >>> assert (c1[:2] == c2[:2]).all()
        >>> assert not (c1 == c2.apply(lambda x: x+1)).all()

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        return False
    if is_tensorclass(other):
        tensor = self._tensordict == other._tensordict
    else:
        tensor = self._tensordict == other
    return _from_tensordict_with_none(self, tensor)


def __ne__(self, other: object) -> bool:
    """Compare the Tensor class object to another object for inequality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object

    Returns:
        False if the objects are of different class types, Tensorclass of boolean values for tensor attributes and None for non-tensor attributes

    Examples:
        >>> @tensorclass
        ... class MyClass:
        ...     x: Tensor
        ...     y: "MyClass"
        ...     z: str
        ...
        >>> c1 = MyClass(
        ...     x=torch.randn(3, 4),
        ...     y=MyClass(
        ...         x=torch.randn(3, 4, 1),
        ...         y=None,
        ...         z="bar",
        ...         batch_size=[3, 4, 1],
        ...     ),
        ...     z="foo",
        ...     batch_size=[3, 4],
        ... )
        >>> c2 = c1.clone()
        >>> print(c1 != c2)
        MyClass(
            x=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
            y=MyClass(
                x=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                y=None,
                z=None,
                batch_size=torch.Size([3, 4, 1]),
                device=None,
                is_shared=False),
            z=None,
            batch_size=torch.Size([3, 4]),
            device=None,
            is_shared=False)
        >>> c2 = c2.apply(lambda x: x+1)
        >>> assert (c1 != c2).all()

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor)
    ):
        return True
    if is_tensorclass(other):
        tensor = self._tensordict != other._tensordict
    else:
        tensor = self._tensordict != other
    return _from_tensordict_with_none(self, tensor)


def _get_typed_output(out, expected_type: str | type):
    # from __future__ import annotations turns types in strings. For those we use CLASSES_DICT.
    # Otherwise, if the output is some TensorDictBase subclass, we check the type and if it
    # does not match, we map it. In all other cases, just return what has been gathered.
    if isinstance(expected_type, str) and expected_type in CLASSES_DICT:
        if isinstance(out, CLASSES_DICT[expected_type]):
            return out
        out = CLASSES_DICT[expected_type]._from_tensordict(out)
    elif (
        isinstance(expected_type, type)
        and not isinstance(out, expected_type)
        and isinstance(out, TensorDictBase)
    ):
        out = expected_type._from_tensordict(out)
    elif isinstance(out, TensorDictBase):
        dest_dtype = _check_td_out_type(expected_type)
        if dest_dtype is not None:
            out = dest_dtype._from_tensordict(out)

    return out


def _single_td_field_as_str(key, item, tensordict):
    """Returns a string as a  key-value pair of tensordict

    Args:
        key (str): key of tensor dict item
        item (tensor type): value to be returned for key
        tensordict (Tensordict): Tensordict object

    Returns:
        String representation of a key-value pair

    """
    if is_tensor_collection(type(item)):
        return f"{key}={repr(tensordict[key])}"
    return f"{key}={get_repr(item)}"


def _all_td_fields_as_str(td: TensorDictBase) -> str:
    """Returns indented representation of tensor dict values as a key-value pairs

    Args:
        td (TensorDict) : Tensordict object

    Returns:
        String representation of all tensor data

    """
    return indent(
        ",\n".join(
            sorted([_single_td_field_as_str(key, item, td) for key, item in td.items()])
        ),
        4 * " ",
    )


def _all_non_td_fields_as_str(src_dict) -> list:
    """Returns a list of string representation of non-tensor key-value pairs

    Args:
        src_dict (dict): non_tensor_dict

    Returns:
        result (list): list of strings with key-value representation

    """
    result = []
    for key, val in src_dict.items():
        if not is_tensor_collection(val):
            result.append(f"{key}={repr(val)}")

    return result


def _check_td_out_type(field_def):
    """This function determines the type of attributes in the tensorclass,
    in order that results from calls to the underlying tensordict
    can be cast to the expected type before being returned
    """
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
            if len(args) == 1 and ("TensorDict" in args[0] or args[0] == "Any"):
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
                # Any or any TensorDictBase subclass are always ok if alone
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
