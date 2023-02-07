import dataclasses
import functools
import inspect
import re
import typing
import warnings
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


def _getstate(self):
    """Returns a state dict which consists of tensor and non_tensor dicts for serialization.

    Returns:
        dictionary of state of tensor class

    """
    return {"tensordict": self._tensordict, "non_tensordict": self._non_tensordict}


def _setstate(self, state) -> None:
    """Used to set the state of an object using state parameter

    Args:
        state (dict): State parameter to set the object
    """
    self._tensordict = state.get("tensordict", None)
    self._non_tensordict = state.get("non_tensordict", None)


def _getattribute_wrapper(getattribute):
    """Retrieve the value of an object's attribute or raise AttributeError

    Args:
        item (str) : name of the attribute to retrieve

    Returns:
        value of the attribute

    """

    @functools.wraps(getattribute)
    def wrapper(self, item):
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
                and item in self.__dict__["_non_tensordict"].keys()
            ):
                out = self._non_tensordict[item]
                return out
        return getattribute(self, item)

    return wrapper


def _setattr_wrapper(setattr_, expected_keys):
    """Set the value of an attribute for the tensor class object

    Args:
        key (str): the name of the attribute to set
        value (any): the value to set for the attribute

    """

    @functools.wraps(setattr_)
    def wrapper(self, key, value):
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

        if isinstance(value, _accepted_classes):
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

    return wrapper


def _getattr(self, attr):
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

    @functools.wraps(getattr)
    def wrapped_func(*args, **kwargs):
        res = func(*args, **kwargs)
        # Handling nested tensor class
        non_tensor_dict = {}
        for key, value in self._non_tensordict.items():
            if is_tensorclass(value):
                temp = getattr(value, attr)
                if callable(temp):
                    # Recursively calling for nested tensor classes
                    non_tensor_dict[key] = temp(*args, **kwargs)
                else:
                    non_tensor_dict[key] = temp
            else:
                non_tensor_dict[key] = value
        if isinstance(res, TensorDictBase):
            new = self._from_tensordict(res, non_tensor_dict)
            return new
        else:
            return res

    return wrapped_func


def _getitem(self, item):
    """Retrieve the class object at the given index. Indexing will happen for nested tensors as well

    Args:
       item (int or any other valid index type): index of the object to retrieve

    Returns:
        Tensor class object at the given index

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    tensor_res = self._tensordict[item]
    non_tensor_res = {}
    for key, value in self._non_tensordict.items():
        if is_tensorclass(value):
            non_tensor_res[key] = _getitem(value, item)
        else:
            non_tensor_res[key] = value

    return self._from_tensordict(tensor_res, non_tensor_res)  # device=res.device)


def _setitem(self, item, value):
    """Set the value of the Tensor class object at the given index. Note that there is no strict validation on non-tensor values

    Args:
        item (int or any other valid index type): index of the object to set
        value (any): value to set for the item

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError("Invalid indexing arguments.")
    if not isinstance(value, self.__class__):
        raise ValueError("__setitem__ is only allowed for same-class assignement")

    # Validating the non-tensor data before setting the item
    for key, val in value._non_tensordict.items():
        # Setting the item for nested tensor class
        if key in self._non_tensordict.keys() and is_tensorclass(val):
            _setitem(self._non_tensordict[key], item, val)
        else:
            # Raise a warning if non_tensor data doesn't match
            if (
                key in self._non_tensordict.keys()
                and val is not self._non_tensordict[key]
            ):
                warnings.warn(
                    f"Meta data at {repr(key)} may or may not be equal, this may result in "
                    f"undefined behaviours",
                    category=UserWarning,
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
    return self._tensordict.to_tensordict()


def _device(self):
    """Retrieves the device type of tensor class"""
    return self._tensordict.device


def _device_setter(self, value: DEVICE_TYPING) -> None:
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


def __eq__(self, other):
    """Compares the Tensor class object to another object for equality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object

    Returns:
        False if the objects are of different class types, Tensorclass of boolean values for tensor attributes and None for non-tensor attributes

    """
    if not isinstance(other, self.__class__):
        return False
    non_tensor = {}
    for key, value in self._non_tensordict.items():
        if is_tensorclass(value):
            non_tensor[key] = value == other._non_tensordict[key]
        else:
            non_tensor[key] = None
    tensor = self._tensordict == other._tensordict
    out = self._from_tensordict(tensor, non_tensor)
    return out


def __ne__(self, other):
    """Compare the Tensor class object to another object for inequality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object

    Returns:
        False if the objects are of different class types, Tensorclass of boolean values for tensor attributes and None for non-tensor attributes

    """
    if not isinstance(other, self.__class__):
        return True
    non_tensor = {}
    for key, value in self._non_tensordict.items():
        if is_tensorclass(value):
            non_tensor[key] = value != other._non_tensordict[key]
        else:
            non_tensor[key] = None
    tensor = self._tensordict != other._tensordict
    out = self._from_tensordict(tensor, non_tensor)
    return out


def _handle_non_tensor_dict(func, non_tensor_dict, *args, **kwargs):
    """Helper function to handle  non_tensor_dict in a given tensor class especially the nestor tensor objects

    Args:
        func (callable): Function to apply on nested Tensor classes
        non_tensor_dict (dict): Dictionary containing non-tensor and nestor tensor class data
        *args (tuple): Positional arguments to pass to the 'func'
        **kwargs (dict): Keyword arguments to pass to the 'func'

    Returns:
        non_tensor_dict (dict): non_tensor_dict after processing

    """

    for key, value in non_tensor_dict.items():
        if is_tensorclass(value):
            non_tensor_dict[key] = func(value, *args, **kwargs)
    return non_tensor_dict


def _handle_list_non_tensor_dict(func, list_of_tdc, *args, **kwargs):
    """Helper function to handle  list of non_tensor_dict in a given tensor class especially the nestor tensor objects

    Args:
        func (callable): Function to apply on nested Tensor classes
        list_of_tdc (list): list of tensor class objects
        *args (tuple): Positional arguments to pass to the 'func'
        **kwargs (dict): Keyword arguments to pass to the 'func'

    Returns:
        non_tensor_dict (dict): non_tensor_dict after processing

    """
    tdc = list_of_tdc[0]
    non_tensordict = tdc._non_tensordict
    for key, value in non_tensordict.items():
        if is_tensorclass(value):
            list_non_tdc = []
            for tdc in list_of_tdc:
                if not isinstance(value, type(tdc._non_tensordict[key])):
                    raise ValueError(
                        f"The values assigned for the attribute "
                        f"{repr(key)} are not matching"
                    )
                list_non_tdc.append(tdc._non_tensordict[key])
            non_tensordict[key] = func(list_non_tdc, *args, **kwargs)

    return non_tensordict


def _unbind(tdc, dim=0):
    """Unbind the tensor class object along a given dimension, the behavior is extended to nested tensor classes as well.(no impact to non-tensor data)

    Args:
        tdc: tensor class object
        dim (int): the dimension along which to unbind the tensor (default is 0)

    Returns:
        out (list): list of tensor class objects representing the unbound parts of the original tensor class object

    """
    tensordicts = torch.unbind(tdc._tensordict, dim)
    non_tensor_dict = _handle_non_tensor_dict(_unbind, tdc._non_tensordict, dim)
    out = [tdc._from_tensordict(td, non_tensor_dict) for td in tensordicts]
    return out


def _full_like(tdc, fill_value):
    """Fill the tensor types of tensor class object with the fill value, the behavior is extended to nested tensor classes as well (no impact to non-tensor data)

    Args:
        tdc: tensor class object
        fill_value (float): The value with which the filling happen

    Returns:
        out: the filled tensor class object

    """
    tensordict = torch.full_like(tdc._tensordict, fill_value)
    non_tensor_dict = _handle_non_tensor_dict(
        _full_like, tdc._non_tensordict, fill_value
    )
    out = tdc._from_tensordict(tensordict, non_tensor_dict)
    return out


def _zeros_like(tdc):
    """Fill the tensor types of tensor class object including nested tensor classes with zeros (no impact to non-tensor data)

    Args:
        tdc: tensor class object

    Returns:
        out: tensor class object filled with zeros

    """
    return _full_like(tdc, 0.0)


def _ones_like(tdc):
    """Fill the tensor types of tensor class object including nested tensor classes with ones (no impact to non-tensor data)

    Args:
        tdc: tensor class object

    Returns:
        out: tensor class object filled with ones

    """
    return _full_like(tdc, 1.0)


def _clone(tdc):
    """Create a shallow copy of the tensor class object, the behavior is extended to nested tensor classes as well

    Args:
        tdc: tensor class object

    Returns:
        out: a shallow copy of the tensor class object

    """
    tensordict = torch.clone(tdc._tensordict)
    non_tensor_dict = _handle_non_tensor_dict(_clone, tdc._non_tensordict)
    out = tdc._from_tensordict(tensordict, non_tensor_dict)
    return out


def _squeeze(tdc):
    """Remove single-dimensional entries from the shape of a tensors for the tensor class objects including nested tensor classes (no impact on non-tensor data)

    Args:
        tdc: tensor class object

    Returns:
        out: squeezed tensor class object

    """
    tensordict = torch.squeeze(tdc._tensordict)
    non_tensor_dict = _handle_non_tensor_dict(_squeeze, tdc._non_tensordict)
    out = tdc._from_tensordict(tensordict, non_tensor_dict)
    return out


def _unsqueeze(tdc, dim=0):
    """Insert a single-dimensional entry at the specified position in the shape of a tensor for tensor class objects including the nested tensor classes (no impact on non-tensor data)

    Args:
        tdc: tensor class object
        dim (int, optional): the position at which to insert the single-dimensional entry

    Returns:
        out: tensor class object with the single-dimensional entry inserted

    """
    tensordict = torch.unsqueeze(tdc._tensordict, dim)
    non_tensor_dict = _handle_non_tensor_dict(_unsqueeze, tdc._non_tensordict, dim)
    out = tdc._from_tensordict(tensordict, non_tensor_dict)
    return out


def _permute(tdc, dims):
    """Permute the dimensions of a tensor class object including nested tensor classes (no impact on non-tensor data)

    Args:
        tdc: tensor class object
        dims (int or tuple of ints): the desired order of the dimensions

    Returns:
        out: permuted tensor class object

    """
    tensordict = torch.permute(tdc._tensordict, dims)
    non_tensor_dict = _handle_non_tensor_dict(_permute, tdc._non_tensordict, dims)
    out = tdc._from_tensordict(tensordict, non_tensor_dict)
    return out


def _split(tdc, split_size_or_sections, dim=0):
    """
    Split a tensor class object into smaller tensor class objects along a given dimension.

    It extends the behavior to nested tensor classes (no impact on non-tensor data)

    Args:
       tdc: tensor class object
       split_size_or_sections (int or list): the size of each split
       dim (int, optional): the dimension along which to split the tensor (default is 0)

    Returns:
        out[list]: list of smaller tensor class objects


    """
    tensordicts = torch.split(tdc._tensordict, split_size_or_sections, dim)
    non_tensor_dict = _handle_non_tensor_dict(
        _split, tdc._non_tensordict, split_size_or_sections, dim
    )
    out = [tdc._from_tensordict(td, non_tensor_dict) for td in tensordicts]
    return out


def _stack(list_of_tdc, dim=0):
    """Stack tensor class objects along a given dimension, the behavior is extended to nested tensor classes. (no impact on non-tensor data)

    Args:
        list_of_tdc (list): list of  tensor class objects to stack
        dim (int, optional): the position of the new dimension (default is 0)

    Returns:
        out: stacked tensor class object

    """
    tensordict = torch.stack([tdc._tensordict for tdc in list_of_tdc], dim)
    non_tensordict = _handle_list_non_tensor_dict(_stack, list_of_tdc, dim)
    out = list_of_tdc[0]._from_tensordict(tensordict, non_tensordict)
    return out


def _cat(list_of_tdc, dim=0):
    """Concatenate tensor class objects along a given dimension, the behavior is extended to nested tensor classes as well.(no impact on non-tensor data)

    Args:
        list_of_tdc (list): list of  tensor class objects to concatenate
        dim (int, optional): the position of the new dimension (default is 0)

    Returns:
        out: concatenated tensor class object

    """
    tensordict = torch.cat([tdc._tensordict for tdc in list_of_tdc], dim)
    non_tensordict = _handle_list_non_tensor_dict(_cat, list_of_tdc, dim)
    out = list_of_tdc[0]._from_tensordict(tensordict, non_tensordict)
    return out


def _get_typed_output(out, expected_type):
    # from __future__ import annotations turns types in strings. For those we use CLASSES_DICT.
    # Otherwise, if the output is some TensorDictBase subclass, we check the type and if it
    # does not match, we map it. In all other cases, just return what has been gathered.
    if isinstance(expected_type, str) and expected_type in CLASSES_DICT:
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
    if is_tensordict(type(item)):
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
        if not is_tensordict(val):
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
