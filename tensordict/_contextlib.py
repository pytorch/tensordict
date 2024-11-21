# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib

# This is a copy from https://github.com/pytorch/pytorch/blob/main/torch/utils/_contextlib.py#L120
# We use it for compatibility with torch >= 1.10 where the implementation fails
# for some tests in torchrl.

# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import sys
import warnings
from typing import Any, Callable, cast, TypeVar

import numpy as np

try:
    from torch.compiler import is_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling


# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """Like contextlib.ContextDecorator.

    Except:

    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise RuntimeError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allows a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else."
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return type(self)()


# TD cm functions
LAST_OP_MAPS = {}


def _reverse_lock(self, args, kwargs, out):
    return self.unlock_()


LAST_OP_MAPS["lock_"] = _reverse_lock


def _reverse_unlock(self, args, kwargs, out):
    return self.lock_()


LAST_OP_MAPS["unlock_"] = _reverse_unlock


def _reverse_transpose(self, args, kwargs, out):
    dim0, dim1 = args
    if not out.is_locked:
        return out.update(self.transpose(dim0, dim1), inplace=False)
    else:
        return out.update_(self.transpose(dim0, dim1))


LAST_OP_MAPS["transpose"] = _reverse_transpose


def _reverse_flatten_keys(self, args, kwargs, out):
    sep = args[0] if args else "."
    if not out.is_locked:
        return out.update(self.unflatten_keys(sep), inplace=False)
    else:
        return out.update_(self.unflatten_keys(sep))


LAST_OP_MAPS["flatten_keys"] = _reverse_flatten_keys


def _reverse_unflatten_keys(self, args, kwargs, out):
    sep = args[0] if args else "."
    if not out.is_locked:
        return out.update(self.flatten_keys(sep), inplace=False)
    else:
        return out.update_(self.flatten_keys(sep))


LAST_OP_MAPS["unflatten_keys"] = _reverse_unflatten_keys


def _reverse_flatten(self, args, kwargs, out):
    if len(args) == 2:
        dim0, dim1 = args
    elif len(args) == 1:
        dim0 = args[0]
        dim1 = kwargs.get("end_dim", -1)
    else:
        dim0 = kwargs.get("start_dim", 0)
        dim1 = kwargs.get("end_dim", -1)
    if dim1 < 0:
        dim1 = out.ndim + dim1
    if dim0 < 0:
        dim0 = out.ndim + dim0

    if not out.is_locked:
        return out.update(
            self.unflatten(dim0, out.shape[dim0 : dim1 + 1]), inplace=False
        )
    else:
        return out.update_(self.unflatten(dim0, out.shape[dim0 : dim1 + 1]))


LAST_OP_MAPS["flatten"] = _reverse_flatten


def _reverse_unflatten(self, args, kwargs, out):
    if args:
        dim0 = args[0]
        if len(args) > 1:
            unflattened_size = args[1]
        else:
            unflattened_size = kwargs.get("unflattened_size")
    else:
        dim0 = kwargs.get("dim")
        unflattened_size = kwargs.get("unflattened_size")
    if dim0 < 0:
        dim0 = out.ndim + dim0
    dim1 = dim0 + len(unflattened_size) - 1
    if not out.is_locked:
        unflattened = self.flatten(dim0, dim1)
        return out.update(unflattened, inplace=False)
    else:
        unflattened = self.flatten(dim0, dim1)
        return out.update_(unflattened)


LAST_OP_MAPS["unflatten"] = _reverse_unflatten


def _reverse_permute(self, args, kwargs, out):
    from tensordict.utils import _get_shape_from_args

    dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
    dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
    # inverse map
    inv_dims_list = np.argsort(dims_list)
    if not out.is_locked:
        return out.update(self.permute(inv_dims_list), inplace=False)
    else:
        return out.update_(self.permute(inv_dims_list))


LAST_OP_MAPS["permute"] = _reverse_permute


def _reverse_view(self, args, kwargs, out):
    if not out.is_locked:
        return out.update(self.view(out.shape), inplace=False)
    else:
        return out.update_(self.view(out.shape))


LAST_OP_MAPS["view"] = _reverse_view


def _reverse_unsqueeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.unsqueeze() as a decorator if the dimension is implicit."
        )
    if not out.is_locked:
        return out.update(self.squeeze(dim), inplace=False)
    else:
        return out.update_(self.squeeze(dim))


LAST_OP_MAPS["unsqueeze"] = _reverse_unsqueeze


def _reverse_squeeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.squeeze() as a decorator if the dimension is implicit."
        )
    if not out.is_locked:
        return out.update(self.unsqueeze(dim), inplace=False)
    else:
        return out.update_(self.unsqueeze(dim))


LAST_OP_MAPS["squeeze"] = _reverse_squeeze


def _reverse_to_module(self, args, kwargs, out):
    try:
        with out.unlock_() if not is_compiling() else contextlib.nullcontext():
            return self.to_module(*args, **kwargs, swap_dest=out)
    except AttributeError:
        # This is a bit unsafe but we assume that out won't have an unlock_() if it's not a TD
        raise RuntimeError(
            "to_module cannot be used as a decorator when return_swap=False."
        )


LAST_OP_MAPS["to_module"] = _reverse_to_module
