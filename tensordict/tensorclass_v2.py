from typing import Any, Generic, List, Sequence, TypeVar, Union, get_args, get_origin

import torch
from pydantic import BaseModel, model_validator
from tensordict import (
    MetaData,
    NonTensorData,
    NonTensorStack,
    TensorDict,
    is_tensor_collection,
    set_list_to_stack,
)
from torch import Size, Tensor

set_list_to_stack(True).set()

T = TypeVar("T")


class NestedList(Generic[T]):
    """A type that accepts either a single value or nested lists of values of type T"""

    def __init__(self, value: Union[T, List["NestedList[T]"]]):
        self.value = value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, _handler):
        T = get_args(source_type)[0]  # Get the type parameter

        def validate_nested(v):
            if isinstance(v, (list, tuple)):
                return [validate_nested(x) for x in v]
            return T(v) if not isinstance(v, T) else v

        return {"type": "any", "mode": "before", "function": validate_nested}


class TensorClass(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra fields like batch_size
    }

    def __init__(self, **data):
        batch_size = data.pop("batch_size", None)
        device = data.pop("device", None)
        super().__init__(**data)
        self.__dict__["_tensordict"] = TensorDict(
            {}, batch_size=batch_size, device=device
        )
        # Initialize tensordict with current values
        for field in self.__class__.model_fields:
            if hasattr(self, field):
                item = getattr(self, field)
                if isinstance(item, torch.Tensor) or is_tensor_collection(item):
                    self._tensordict[field] = item
                else:
                    # Get the field's type annotation
                    field_type = self.__class__.model_fields[field].annotation

                    def has_nested_list(type_):
                        """Check if a type contains NestedList"""
                        if get_origin(type_) is NestedList:
                            return True
                        if get_origin(type_) in (Union, None):
                            # For Union types or simple types, check each argument
                            args = get_args(type_)
                            return any(has_nested_list(arg) for arg in args)
                        return False

                    def get_primary_type(type_):
                        """Get the primary type (non-NestedList) from a Union or simple type"""
                        if get_origin(type_) in (Union, None):
                            # Look through Union args for non-NestedList type
                            for arg in get_args(type_):
                                if not has_nested_list(arg):
                                    return arg
                        return None

                    # Check if it's a NestedList type or contains NestedList
                    if has_nested_list(field_type):
                        primary_type = get_primary_type(field_type)
                        # If it matches the primary type (e.g. str for b), treat as scalar
                        if primary_type and isinstance(item, primary_type):
                            self._tensordict[field] = NonTensorData(item)
                        # Otherwise if it's a sequence (but not str), treat as list
                        elif isinstance(item, Sequence) and not isinstance(
                            item, (str, bytes)
                        ):
                            stack = NonTensorStack.from_list(item)
                            self._tensordict[field] = stack
                        else:
                            self._tensordict[field] = NonTensorData(item)
                    else:
                        self._tensordict[field] = MetaData(item)
                delattr(self, field)

    @property
    def device(self) -> torch.device:
        return self._tensordict.device

    @device.setter
    def device(self, device: torch.device):
        self._tensordict.device = device

    @property
    def batch_size(self) -> Size:
        """Get the batch size of the underlying TensorDict"""
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
                self._tensordict[field] = getattr(self, field)
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
        fields_repr = ", ".join(
            f"{field}={getattr(self, field)}" for field in self.__class__.model_fields
        )
        extra_fields = {
            "batch_size": self.batch_size,
            "device": self._tensordict.device,
        }
        extra_repr = ", ".join(f"{k}={v}" for k, v in extra_fields.items())
        return f"{self.__class__.__name__}({fields_repr}, {extra_repr})"


if __name__ == "__main__":
    class MyClass(TensorClass):
        a: int | NestedList[int]  # NonTensorStack
        b: str | NestedList[str]  # Now accepts str or list[str] or list[list[str]] etc
        c: Tensor
        d: str  # MetaData


    # Default (empty) batch size
    model = MyClass(a=1, b="hello", c=torch.tensor([1.0, 2.0, 3.0]), d="a string")
    print(f"{model=}")
    print(f"Model attributes: a={model.a}, b={model.b}")
    print(f"Model tensor c={model.c}")
    print(f"TensorDict contents: {model._tensordict}")
    print(f"Default batch_size: {model.batch_size}")

    # Integer batch size
    model = MyClass(
        a=[1, 2],
        b=["hello", "world"],
        c=torch.tensor([[1.0], [2.0]]),  # 2x1 tensor
        d="a string",
        batch_size=2,
    )
    print(f"{model=}")
    print(f"{model._tensordict=}")
    print(f"Integer batch_size: {model.batch_size}")
