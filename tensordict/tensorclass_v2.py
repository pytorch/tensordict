from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    get_args,
    get_origin,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import tensordict.base

import torch
from pydantic import BaseModel, model_validator
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    MetaData,
    NonTensorData,
    NonTensorStack,
    set_list_to_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import DeviceType, IndexType, is_non_tensor
from torch import Size, Tensor

set_list_to_stack(True).set()

T = TypeVar("T")


class NestedList(Generic[T]):
    """A type that accepts either a single value or nested lists of values of type T.

    This type is used to represent data that can be either a single value or a nested structure
    of lists containing that value type. This is particularly useful for handling batch data
    and non-tensor data in TensorClass.

    Examples:
        >>> x: NestedList[int] = 1  # single value
        >>> x: NestedList[int] = [1, 2, 3]  # list of values
        >>> x: NestedList[int] = [[1, 2], [3, 4]]  # nested list of values
    """

    def __init__(self, value: Union[T, List["NestedList[T]"]]):
        self.value = value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, _handler):
        T = get_args(source_type)[0]  # Get the type parameter

        def validate_nested(v):
            if isinstance(v, (list, tuple)):
                return [validate_nested(item) for item in v]
            return v

        return {"type": "any", "mode": "python", "post_process": validate_nested}


class TensorClassMetaBase(type):
    """Base metaclass for TensorClass that handles inheritance of TensorDict methods."""

    _FALLBACK_METHOD_FROM_TD_NOWRAP = [
        "_check_batch_size",
        "_check_device",
        "_check_dim_name",
        "_check_unlock",
        "_default_get",
        "_get_at_str",
        "_get_at_tuple",
        "_get_names_idx",  # no wrap output
        "_get_str",
        "_get_tuple",
        "_get_tuple_maybe_non_tensor",
        "_has_names",
        "_items_list",
        "_maybe_names",
        "_multithread_apply_flat",
        "_multithread_apply_nest",
        "_multithread_rebuild",  # rebuild checks if self is a non tensor
        "_propagate_lock",
        "_propagate_unlock",
        "_reduce_get_metadata",
        "_values_list",
        "batch_dims",
        "batch_size",
        "bytes",
        "cat_tensors",
        "clear_refs_for_compile_",
        "data_ptr",
        "depth",
        "dim",
        "dtype",
        "entry_class",
        "get_item_shape",
        "get_non_tensor",
        "init_remote",
        "irecv",
        "is_consolidated",
        "is_contiguous",
        "is_cpu",
        "is_cuda",
        "is_empty",
        "is_floating_point",
        "is_locked",
        "is_memmap",
        "is_meta",
        "is_shared",
        "isend",
        "items",
        "keys",
        "make_memmap",
        "make_memmap_from_tensor",
        "names",
        "ndim",
        "ndimension",
        "numel",
        "numpy",
        "param_count",
        "pop",
        "recv",
        "reduce",
        "requires_grad",
        "saved_path",
        "send",
        "shape",
        "size",
        "sorted_keys",
        "to_struct_array",
        "tolist",
        "values",
    ]

    # Methods that need to be rewrapped after execution
    _FALLBACK_METHOD_FROM_TD = [
        "__abs__",
        "__add__",
        "__and__",
        "__bool__",
        "__eq__",
        "__iadd__",
        "__imul__",
        "__invert__",
        "__ipow__",
        "__isub__",
        "__itruediv__",
        "__mul__",
        "__ne__",
        "__neg__",
        "__or__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rmul__",
        "__rpow__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__sub__",
        "__truediv__",
        "__xor__",
        "_add_batch_dim",
        "_apply_nest",
        "_clone",
        "_clone_recurse",
        "_data",
        "_erase_names",
        "_exclude",
        "_fast_apply",
        "_flatten_keys_inplace",
        "_flatten_keys_outplace",
        "_get_sub_tensordict",
        "_grad",
        "_map",
        "_maybe_remove_batch_dim",
        "_memmap_",
        "_permute",
        "_remove_batch_dim",
        "_repeat",
        "_select",
        "_set_at_tuple",
        "_set_tuple",
        "_to_module",
        "abs",
        "abs_",
        "acos",
        "acos_",
        "add",
        "add_",
        "addcdiv",
        "addcdiv_",
        "addcmul",
        "addcmul_",
        "all",
        "amax",
        "amin",
        "any",
        "apply",
        "apply_",
        "as_tensor",
        "asin",
        "asin_",
        "atan",
        "atan_",
        "auto_batch_size_",
        "auto_device_",
        "bfloat16",
        "bitwise_and",
        "bool",
        "cat",
        "cat_from_tensordict",
        "ceil",
        "ceil_",
        "chunk",
        "clamp",
        "clamp_max",
        "clamp_max_",
        "clamp_min",
        "clamp_min_",
        "to",
        "unbind",
        "split",
        "clear",
        "clear_device_",
        "complex128",
        "complex32",
        "complex64",
        "consolidate",
        "contiguous",
        "copy_",
        "copy_at_",
        "cos",
        "cos_",
        "cosh",
        "cosh_",
        "cpu",
        "create_nested",
        "cuda",
        "cummax",
        "cummin",
        "densify",
        "detach",
        "detach_",
        "div",
        "div_",
        "double",
        "empty",
        "erf",
        "erf_",
        "erfc",
        "erfc_",
        "exclude",
        "exp",
        "exp_",
        "expand",
        "expand_as",
        "expm1",
        "expm1_",
        "extend",
        "fill_",
        "filter_empty_",
        "filter_non_tensor_data",
        "flatten",
        "flatten_keys",
        "float",
        "float16",
        "float32",
        "float64",
        "floor",
        "floor_",
        "frac",
        "frac_",
        "from_any",
        "from_consolidated",
        "from_dataclass",
        "from_h5",
        "from_modules",
        "from_namedtuple",
        "from_pytree",
        "from_remote_init",
        "from_struct_array",
        "from_tuple",
        "fromkeys",
        "gather",
        "gather_and_stack",
        "half",
        "int",
        "int16",
        "int32",
        "int64",
        "int8",
        "isfinite",
        "isnan",
        "isneginf",
        "isposinf",
        "isreal",
        "lazy_stack",
        "lerp",
        "lerp_",
        "lgamma",
        "lgamma_",
        "load_memmap_",
        "lock_",
        "log",
        "log10",
        "log10_",
        "log1p",
        "log1p_",
        "log2",
        "log2_",
        "log_",
        "logical_and",
        "logsumexp",
        "make_memmap_from_storage",
        "map",
        "map_iter",
        "masked_fill",
        "masked_fill_",
        "masked_select",
        "max",
        "maximum",
        "maximum_",
        "maybe_dense_stack",
        "mean",
        "min",
        "minimum",
        "minimum_",
        "mul",
        "mul_",
        "named_apply",
        "nanmean",
        "nansum",
        "neg",
        "neg_",
        "new_empty",
        "new_full",
        "new_ones",
        "new_tensor",
        "new_zeros",
        "norm",
        "permute",
        "pin_memory",
        "pin_memory_",
        "popitem",
        "pow",
        "pow_",
        "prod",
        "qint32",
        "qint8",
        "quint4x2",
        "quint8",
        "reciprocal",
        "reciprocal_",
        "record_stream",
        "refine_names",
        "rename",
        "rename_",
        "rename_key_",
        "repeat",
        "repeat_interleave",
        "replace",
        "requires_grad_",
        "reshape",
        "round",
        "round_",
        "rsub",
        "select",
        "separates",
        "set_",
        "set_non_tensor",
        "setdefault",
        "sigmoid",
        "sigmoid_",
        "sign",
        "sign_",
        "sin",
        "sin_",
        "sinh",
        "sinh_",
        "softmax",
        "split",
        "split_keys",
        "sqrt",
        "sqrt_",
        "squeeze",
    ]

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Get all TensorDict methods we want to inherit
        for method_name in dir(TensorDict):
            # Skip private methods and already defined ones
            if method_name.startswith("_") or method_name in namespace:
                continue

            # Get the method from TensorDict
            td_method = getattr(TensorDict, method_name)
            if method_name == "is_locked":
                print(
                    method_name,
                    td_method,
                    type(td_method),
                    callable(td_method),
                    isinstance(td_method, property),
                )
            if callable(td_method):
                # Create a wrapper that will delegate to _tensordict
                def wrap_method(m_name):
                    def wrapped(self, *args, m_name=m_name, **kwargs):
                        result = getattr(self._tensordict, m_name)(*args, **kwargs)
                        if m_name in mcs._FALLBACK_METHOD_FROM_TD_NOWRAP:
                            return result

                        # If this is a method that needs rewrapping, wrap the result
                        if m_name in mcs._FALLBACK_METHOD_FROM_TD:
                            if is_tensor_collection(result):
                                return self.from_tensordict(result)
                            elif isinstance(result, (list, tuple)) and all(
                                is_tensor_collection(x) for x in result
                            ):
                                return type(result)(
                                    self.from_tensordict(r) for r in result
                                )
                        return result

                    wrapped.__name__ = m_name
                    wrapped.__qualname__ = f"{cls.__name__}.{m_name}"
                    return wrapped

                setattr(cls, method_name, wrap_method(method_name))
            elif isinstance(td_method, property):
                print(method_name, "is a property")

                # Create a wrapper that will delegate to _tensordict
                def wrap_property(m_name):
                    def wrapped(self, *args, **kwargs):
                        print("Wrapped property")
                        result = getattr(self._tensordict, m_name)
                        if m_name in mcs._FALLBACK_METHOD_FROM_TD_NOWRAP:
                            return result

                        # If this is a method that needs rewrapping, wrap the result
                        if m_name in mcs._FALLBACK_METHOD_FROM_TD:
                            if is_tensor_collection(result):
                                return self.from_tensordict(result)
                            elif isinstance(result, (list, tuple)) and all(
                                is_tensor_collection(x) for x in result
                            ):
                                return type(result)(
                                    self.from_tensordict(r) for r in result
                                )
                        return result

                    wrapped.__name__ = m_name
                    wrapped.__qualname__ = f"{cls.__name__}.{m_name}"
                    return wrapped

                setattr(cls, method_name, wrap_property(method_name))
            else:
                continue

        return cls


class TensorClassMeta(type(BaseModel), TensorClassMetaBase):
    """Metaclass that combines Pydantic's ModelMetaclass with TensorDict method inheritance."""

    pass


class TensorClass(BaseModel, metaclass=TensorClassMeta):
    """A Pydantic-based implementation of TensorDict's TensorClass that combines the power of Pydantic's validation with TensorDict's functionality.

    This class allows you to define strongly-typed data structures that can handle both tensor
    and non-tensor data, with automatic validation, serialization, and tensor operations support.
    It inherits all TensorDict's methods while maintaining Pydantic's validation capabilities.

    Key Features:
        - Strong type checking and validation through Pydantic
        - Full TensorDict functionality (indexing, reshaping, device movement, etc.)
        - Support for nested TensorClass instances
        - Automatic handling of non-tensor data
        - Batch operations support
        - Device management
        - Memory sharing capabilities

    Args:
        **data: Keyword arguments for field values and configuration:
            - batch_size (Optional[Union[List[int], torch.Size]]): The batch dimensions
            - device (Optional[Union[str, torch.device]]): The device to store tensors on
            - Any field defined in the class with type hints

    Examples:
        >>> class MyTensorClass(TensorClass):
        ...     tensor: torch.Tensor
        ...     metadata: str
        ...     nested: Optional["MyTensorClass"] = None
        ...
        >>> # Create an instance with batch size
        >>> data = MyTensorClass(
        ...     tensor=torch.randn(3, 4),
        ...     metadata="example",
        ...     batch_size=[3]
        ... )
        >>> # Use TensorDict operations
        >>> data_gpu = data.to("cuda")
        >>> first_item = data[0]
        >>> # Nested structure
        >>> nested = MyTensorClass(
        ...     tensor=torch.randn(2, 2),
        ...     metadata="nested",
        ...     nested=data
        ... )

    Attributes:
        _tensordict (TensorDict): The underlying TensorDict instance that stores the data
        batch_size (torch.Size): The batch dimensions of the data
        device (torch.device): The device where the tensors are stored

    Configuration:
        The class behavior can be customized through the model_config dictionary:

        - arbitrary_types_allowed (bool): Allow fields with arbitrary types (required for tensors)
        - validate_assignment (bool): Enable/disable validation on field assignment
        - frozen (bool): Make the model immutable after creation
        - extra (str): How to handle extra fields ('allow', 'ignore', or 'forbid')
        - validate_default (bool): Whether to validate default values
        - json_schema_extra (dict): Additional JSON schema properties
        - json_encoders (dict): Custom JSON encoders for types
        - strict (bool): Enforce strict type checking

        Example:
            >>> class MyTensorClass(TensorClass):
            ...     model_config = {
            ...         "arbitrary_types_allowed": True,  # Required for tensor support
            ...         "validate_assignment": True,      # Validate on attribute assignment
            ...         "frozen": False,                  # Allow modifications
            ...         "extra": "forbid",               # Disallow extra fields
            ...     }

    Notes:
        - When subclassing, use type hints to define your fields
        - Tensor fields are automatically stored in the underlying TensorDict
        - Non-tensor data is handled through NonTensorData/MetaData
        - The class supports both eager and lazy operations
        - All TensorDict methods are available through method delegation
    """

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the underlying TensorDict if not already done
        if not hasattr(self, "_tensordict"):
            batch_size = data.get("batch_size", [])
            device = data.get("device", None)
            self._tensordict = TensorDict({}, batch_size=batch_size, device=device)

    model_config = {
        "arbitrary_types_allowed": True,  # Required for tensor support
    }

    def _sync_field(self, field: str) -> None:
        """Sync a single field from the model to the tensordict."""
        if not hasattr(self, field):
            return

        item = getattr(self, field)
        if isinstance(item, (torch.Tensor, TensorDictBase)):
            self._tensordict[field] = item
        else:
            # Get the field's type annotation
            field_type = self.__class__.model_fields[field].annotation

            def has_nested_list(type_):
                """Check if a type contains NestedList"""
                if get_origin(type_) is NestedList:
                    return True
                if get_origin(type_) in (Union, None):
                    args = get_args(type_)
                    return any(has_nested_list(arg) for arg in args)
                return False

            def get_primary_type(type_):
                """Get the primary type (non-NestedList) from a Union or simple type"""
                if get_origin(type_) in (Union, None):
                    for arg in get_args(type_):
                        if not has_nested_list(arg):
                            return arg
                return None

            # Handle NestedList types
            if has_nested_list(field_type):
                primary_type = get_primary_type(field_type)
                if primary_type and isinstance(item, primary_type):
                    self._tensordict[field] = NonTensorData(item)
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    # Convert sequence to NonTensorStack
                    self._tensordict[field] = NonTensorStack.from_list(item)
                else:
                    self._tensordict[field] = NonTensorData(item)
            else:
                self._tensordict[field] = MetaData(item)
        delattr(self, field)

    def update(
        self,
        input_dict_or_td: Union[Dict[str, Any], TensorDictBase],
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Optional[Sequence[str]] = None,
    ) -> "TensorClass":
        """Update the TensorClass with new values.

        Args:
            input_dict_or_td: Dictionary or TensorDict with new values
            clone: If True, clone tensors before updating
            inplace: If True, modify in place, otherwise return a new instance
            non_blocking: If True, try to perform the update asynchronously
            keys_to_update: Optional sequence of keys to update

        Returns:
            Updated TensorClass instance
        """
        if not inplace:
            new_instance = self.clone()
            new_instance.update(
                input_dict_or_td,
                clone=clone,
                inplace=True,
                non_blocking=non_blocking,
                keys_to_update=keys_to_update,
            )
            return new_instance

        # Convert dict to TensorDict if needed
        if isinstance(input_dict_or_td, dict):
            input_td = TensorDict(
                input_dict_or_td, batch_size=self.batch_size, device=self.device
            )
        else:
            input_td = input_dict_or_td

        # Update the tensordict
        self._tensordict.update(
            input_td,
            clone=clone,
            inplace=True,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )
        return self

    def to_tensordict(self) -> TensorDict:
        """Convert this instance to a TensorDict.

        Returns:
            TensorDict: A TensorDict containing all the data from this instance
        """
        return self._tensordict

    @classmethod
    def from_tensordict(cls, tensordict: TensorDictBase) -> "TensorClass":
        """Create a new instance from a TensorDict.

        This classmethod allows converting an existing TensorDict into a TensorClass instance,
        preserving all the data and batch dimensions.

        Args:
            tensordict (TensorDictBase): The source TensorDict to create the instance from

        Returns:
            A new TensorClass instance containing the data from the TensorDict

        Examples:
            >>> td = TensorDict({"x": torch.ones(3, 4)}, batch_size=[3])
            >>> instance = MyClass.from_tensordict(td)
        """
        instance = cls.__new__(cls)
        instance.__dict__["_tensordict"] = tensordict
        return instance

    @property
    def device(self) -> torch.device | None:
        """Get the device of the underlying TensorDict.

        Returns:
            torch.device | None: The device where the tensors are stored, or None if not on a specific device
        """
        return self._tensordict.device

    @device.setter
    def device(self, device: torch.device):
        self._tensordict.device = device

    @property
    def batch_size(self) -> torch.Size:
        """Get the batch size of the underlying TensorDict.

        Returns:
            torch.Size: The batch dimensions of the data
        """
        return self._tensordict.batch_size

    @batch_size.setter
    def batch_size(self, size):
        """Set the batch size of the underlying TensorDict"""
        td = self.__dict__["_tensordict"]
        td.batch_size = size

    @model_validator(mode="after")
    def sync_to_tensordict(self, data):
        # Ensure _tensordict exists with proper batch size
        if not hasattr(self, "_tensordict"):
            self.__dict__["_tensordict"] = TensorDict({}, batch_size=[])

        # Sync all fields to tensordict
        for field in self.__class__.model_fields:
            if hasattr(self, field):
                item = getattr(self, field)
                print(f"{field=} {item=}")
                self._tensordict[field] = item
        return self

    def __getattr__(self, name: str) -> Any:
        if name == "_tensordict":
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute {name}"
                )
        try:
            return self._tensordict[name]
        except (KeyError, AttributeError):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def __repr__(self):
        """Return a string representation of the TensorClass instance.

        The representation includes all fields and their values, as well as
        batch size and device information.
        """
        fields_repr = ", ".join(
            f"{field}={getattr(self, field, None)}"
            for field in self.__class__.model_fields
        )
        extra_fields = {
            "batch_size": self._tensordict.batch_size,
            "device": self._tensordict.device,
        }
        extra_repr = ", ".join(f"{k}={v}" for k, v in extra_fields.items())
        return f"{self.__class__.__name__}({fields_repr}, {extra_repr})"

    def __getitem__(self, index):
        """Support indexing operations like td[0] or td['key'].

        This method handles both integer/slice indexing for batch operations
        and string indexing for accessing fields.

        Args:
            index: Union[str, int, slice, Tuple] - The index to access

        Returns:
            The indexed data, maintaining the TensorClass type for batch operations
        """
        if isinstance(index, str):
            return self.__getattr__(index)
        # For slice/integer indexing, create a new instance with indexed tensordict
        new_instance = self.__class__()
        new_instance.__dict__["_tensordict"] = self._tensordict[index]
        return new_instance

    def __setitem__(self, index, value):
        """Support item assignment operations.

        This method handles both batch indexing and field assignment.

        Args:
            index: Union[str, int, slice, Tuple] - The index to assign to
            value: The value to assign
        """
        if isinstance(index, str):
            setattr(self, index, value)
        else:
            if isinstance(value, TensorClass):
                self._tensordict[index] = value._tensordict
            else:
                self._tensordict[index] = value

    def __len__(self):
        """Return the length of the first dimension of the batch size."""
        return len(self._tensordict)

    def __iter__(self):
        """Support iteration over the first dimension."""
        for field_name in self.__class__.model_fields:
            yield field_name, getattr(self, field_name)

    def iter_tensors(self):
        """Iterate over the first dimension of the batch size."""
        for i in range(len(self)):
            yield self[i]

    def clone(self):
        """Create a deep copy of the TensorClass instance."""
        new_instance = self.__class__()
        new_instance.__dict__["_tensordict"] = self._tensordict.clone()
        return new_instance

    def to(self, device):
        """Move the TensorClass to the specified device."""
        new_instance = self.__class__()
        new_instance.__dict__["_tensordict"] = self._tensordict.to(device)
        return new_instance

    def detach(self):
        """Detach all tensors in the TensorClass."""
        new_instance = self.__class__()
        new_instance.__dict__["_tensordict"] = self._tensordict.detach()
        return new_instance


class NonTensorStack(LazyStackedTensorDict):
    """A thin wrapper around LazyStackedTensorDict to make stack on non-tensor data easily recognizable.

    A ``NonTensorStack`` is returned whenever :func:`~torch.stack` is called on
    a list of :class:`~tensordict.NonTensorData` or ``NonTensorStack``.

    Examples:
        >>> from tensordict import NonTensorData
        >>> import torch
        >>> data = torch.stack([
        ...     torch.stack([NonTensorData(data=(i, j), batch_size=[]) for i in range(2)])
        ...    for j in range(3)])
        >>> print(data)
        NonTensorStack(
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, ...,
            batch_size=torch.Size([3, 2]),
            device=None)

    To obtain the values stored in a ``NonTensorStack``, call :class:`~.tolist`.
    """

    _is_non_tensor: bool = True

    def __init__(self, *args: TensorDictBase, **kwargs):
        args = [
            arg if is_tensor_collection(arg) else NonTensorData(arg) for arg in args
        ]
        super().__init__(*args, **kwargs)
        if not all(is_non_tensor(item) for item in self.tensordicts):
            raise RuntimeError("All tensordicts must be non-tensors.")

    @classmethod
    def from_list(cls, non_tensors: Sequence[Any]) -> "NonTensorStack":
        """Create a NonTensorStack from a list of non-tensor data.

        Args:
            non_tensors: A sequence of non-tensor data to stack

        Returns:
            A NonTensorStack containing the stacked non-tensor data
        """

        def _maybe_from_list(nontensor):
            if isinstance(nontensor, list):
                return cls.from_list(nontensor)
            if is_non_tensor(nontensor):
                return nontensor
            return NonTensorData(nontensor)

        return cls(*[_maybe_from_list(nontensor) for nontensor in non_tensors])

    def tolist(
        self,
        *,
        convert_tensors: bool = False,
        tolist_first: bool = False,
        convert_nodes: bool = False,
    ):
        """Extracts the content of a :class:`tensordict.tensorclass.NonTensorStack` in a nested list.

        Keyword Args:
            convert_tensors (bool): if ``True``, tensors will be converted to lists.
                Otherwise, they will remain as tensors. Default: ``False``.
            tolist_first (bool, optional): if ``True``, the tensordict will be converted to a list first when
                it has batch dimensions. Default: ``True``.
            convert_nodes (bool, optional): if ``True``, convert nodes to lists. Default: ``False``.

        Examples:
            >>> from tensordict import NonTensorData
            >>> import torch
            >>> data = torch.stack([
            ...     torch.stack([NonTensorData(data=(i, j), batch_size=[]) for i in range(2)])
            ...    for j in range(3)])
            >>> data.tolist()
            [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]]
        """
        iterator = self.tensordicts if self.stack_dim == 0 else self.unbind(0)
        return [
            td.tolist(
                convert_tensors=convert_tensors,
                tolist_first=tolist_first,
                convert_nodes=convert_nodes,
            )
            for td in iterator
        ]


tensordict.base._ACCEPTED_CLASSES = tensordict.base._ACCEPTED_CLASSES + (TensorClass,)
