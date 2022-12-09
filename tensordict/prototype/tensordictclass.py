import functools
from dataclasses import dataclass, make_dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from tensordict.tensordict import _accepted_classes, TensorDict, TensorDictBase

from torch import Tensor


def tensordictclass(cls):
    TD_HANDLED_FUNCTIONS: Dict = {}

    name = cls.__name__
    datacls = make_dataclass(
        name, bases=(dataclass(cls),), fields=[("batch_size", torch.Size)]
    )

    class TensorDictClass(datacls):
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
            return TensorDictClass(_tensordict=tensordict)

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
                return self.__dict__["tensordict"][item]
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
                        new = TensorDictClass(_tensordict=res)
                        return new
                    else:
                        return res

                return wrapped_func

        def __getitem__(self, item):
            res = self.tensordict[item]
            return TensorDictClass(
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
        out = TensorDictClass(_tensordict=tensordict)
        return out

    @implements_for_tdc(torch.cat)
    def _cat(list_of_tdc, dim):
        tensordict = torch.cat([tdc.tensordict for tdc in list_of_tdc], dim)
        out = TensorDictClass(_tensordict=tensordict)
        return out

    return TensorDictClass
