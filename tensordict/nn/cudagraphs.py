# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import functools
import warnings
import weakref
from functools import partial
from textwrap import indent
from typing import Any, Callable, List

import torch
from tensordict._nestedkey import NestedKey
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.nn.common import dispatch
from tensordict.nn.functional_modules import (
    PYTREE_REGISTERED_LAZY_TDS,
    PYTREE_REGISTERED_TDS,
)
from tensordict.nn.probabilistic import interaction_type, ProbabilisticTensorDictModule
from tensordict.utils import (
    _user_stacklevel,
    _zip_strict,
    logger as tensordict_logger,
    unravel_key,
)
from torch import nn, Tensor
from torch.utils._pytree import (
    tree_flatten as _pytree_flatten,
    tree_leaves as _pytree_leaves,
    tree_map as _pytree_map,
    tree_unflatten,
)

# CudaGraphModule copies tensordict inputs with ``copy_``, which matches the
# entries by key, and returns tensordict outputs with ``clone``: tensordicts are
# leaves of the trees it builds. They are kept as leaves with ``is_leaf``, so
# the global pytree registry is left as it is (pytorch/tensordict#2076).
# ``SUPPORTED_NODES`` is keyed by exact type, hence ``type(x)``.
_TD_LEAF_TYPES = frozenset(PYTREE_REGISTERED_TDS + PYTREE_REGISTERED_LAZY_TDS)


def _keys_differ(keys, module_keys) -> bool:
    """Whether ``keys``, given explicitly, are non-empty and differ from the module's own keys (pytorch/tensordict#2973).

    Only a str, or a list, tuple or set of nested keys, is compared; anything else returns ``False``.
    """
    collections = (str, list, tuple, set)
    if not isinstance(keys, collections) or not isinstance(module_keys, collections):
        return False
    if not len(keys):
        return False
    unordered = isinstance(keys, set) or isinstance(module_keys, set)
    try:
        keys, module_keys = (
            [unravel_key(key) for key in ([k] if isinstance(k, str) else k)]
            for k in (keys, module_keys)
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    if unordered:
        return set(keys) != set(module_keys)
    return keys != module_keys


def _is_td_leaf(x: Any) -> bool:
    return type(x) in _TD_LEAF_TYPES


def _tree_flatten(tree):
    return _pytree_flatten(tree, is_leaf=_is_td_leaf)


def _tree_leaves(tree):
    return _pytree_leaves(tree, is_leaf=_is_td_leaf)


def _tree_map(func, tree):
    return _pytree_map(func, tree, is_leaf=_is_td_leaf)


__all__ = ["CudaGraphModule"]


class CudaGraphModule:
    """Records a function, or a TensorDictModule, as a CUDA graph and replays it.

    A replay runs all the recorded kernels with one CPU call, which removes the
    CPU overhead of launching them one by one. The first ``warmup - 1`` calls
    run the function on a side stream. The next call records the graph, and
    later calls copy their inputs into the graph's buffers and replay it. Each
    call checks that the inputs match the recorded ones (see the rules below).
    See :ref:`compile-cudagraphs` for a worked example.

    Without CUDA, the constructor warns, and each call runs the function
    directly.

    .. warning::
        ``CudaGraphModule`` is a prototype feature and its API restrictions are likely to change in the future.

    .. warning::
        The wrapped function must meet the requirements below. ``CudaGraphModule``
        does not check all of them: a function that breaks one can give wrong
        results.

        - The function cannot have dynamic control flow. For instance, the following code snippet will fail to be
          wrapped in ``CudaGraphModule``:

          .. code-block:: python

            def func(x):
                if x.norm() > 1:
                    return x + 1
                else:
                    return x - 1

          Use :func:`torch.where` instead:

          .. code-block:: python

            def func(x):
                return torch.where(x.norm() > 1, x + 1, x - 1)

        - The function must execute a code that can be exactly re-run using the same buffers. This means that
          dynamic shapes (changing shape in the input or during the code execution) is not supported. In other words,
          the input must have a constant shape. After the capture, a tensor input with a different shape, or a
          tensordict input with a different batch size or an entry of a different shape, raises a ``ValueError``, and
          a tensordict input that lacks one of the ``in_keys`` raises a ``KeyError``.
        - The output of the function must be detached. If a call to the optimizers is required, put it in the input
          function. For instance, the following function is a valid operator:

          .. code-block:: python

            def func(x, y):
                optim.zero_grad()
                loss_val = loss_fn(x, y)
                loss_val.backward()
                optim.step()
                return loss_val.detach()

        - The inputs must not require gradients. If the function uses ``nn.Parameter`` instances (or other tensors
          that require gradients), do not pass them as arguments of the call. Read them from the enclosing scope, or
          bind them as default values, as in ``def func(x, w=w)``:

          .. code-block:: python

            w = nn.Parameter(torch.randn((), device="cuda"))
            optim = torch.optim.Adam([w], lr=1)

            def func(x):  # right: w is not an argument of the call
                optim.zero_grad()
                (x * w).backward()
                optim.step()

            def func(x, w):  # wrong: w is passed at each call
                ...

        - Tensor and tensordict arguments can change between calls, if their device and shape stay the same.
          Arguments that are not tensors (a ``float``, a ``str``) must not change after the capture. If the
          function takes tensors, a changed value raises a ``ValueError``. If it takes a tensordict, the replay uses
          the captured value, ignores the new one and emits a ``FutureWarning``. Starting with TensorDict 0.17, the
          default will change to raising a ``ValueError`` there too (pytorch/tensordict#2075).
          The keyword arguments of a tensordict module are its ``in_keys`` (see :func:`~tensordict.nn.dispatch`)
          and ``tensordict_out``.

        - If the module is a :class:`~tensordict.nn.TensorDictModuleBase` instance and the output id matches the input
          id, then this identity will be preserved during a call to ``CudaGraphModule``, and the outputs written into
          the input are clones of the graph's output buffers. In all other cases, the output will be cloned,
          irrespective of whether its elements match or do not match one of the inputs.

        - In-place writes to the input tensors inside the function (e.g. ``x.add_(1)``) do not reach the caller's
          inputs once the graph is captured: the graph reads and writes its own copies of the inputs.

        - The inputs are copied into the graph's buffers, and the graph is replayed, on the current stream. As for any
          CUDA operation, inputs written on another stream must be ordered before the call by the caller, e.g. with
          ``torch.cuda.current_stream().wait_stream(producer_stream)``. Consecutive calls share the graph's buffers:
          issue them from one stream, or order each call after the previous one in the same way.

        - The replay repeats the branch that each :class:`~tensordict.nn.ProbabilisticTensorDictModule` took at
          capture: a graph captured under ``set_interaction_type("random")`` keeps sampling under
          ``set_interaction_type("deterministic")``. The first replay under another interaction type emits a
          ``UserWarning``. The check covers only the :class:`~tensordict.nn.ProbabilisticTensorDictModule` instances
          inside the wrapped :class:`~torch.nn.Module`: other code that branches on
          :func:`~tensordict.nn.probabilistic.interaction_type` (for example an exploration module) and wrapped
          functions that are not modules are not checked; if the wrapped module holds no
          :class:`~tensordict.nn.ProbabilisticTensorDictModule`, the replay repeats the captured branch without a
          warning. Capture one ``CudaGraphModule`` per interaction type whenever the output depends on it.

    .. warning::
        ``CudaGraphModule`` is not an :class:`~torch.nn.Module` by design, to discourage gathering parameters
        of the input module and passing them to an optimizer.

    Args:
        module (Callable): a function that takes tensors (or a tensordict) and returns a tensor, a tensordict, or
            a pytree of them, such as a tuple or a dict. A tensordict module can also be called with keyword
            arguments (see the example).
        warmup (int, optional): the call that records the graph. The first ``warmup - 1`` calls run the function
            on a side stream, the next call records the graph, and later calls replay it. A compiled function needs
            enough calls to finish compiling before the capture, for example
            ``CudaGraphModule(torch.compile(module), warmup=3)``. Defaults to ``2``.
        in_keys (list of NestedKey, optional): the input keys, if the module takes a tensordict. If ``module``
            has an ``in_keys`` attribute, that attribute is used and this argument is ignored, with a
            ``UserWarning`` if the argument is a non-empty str, list, tuple or set of keys that differs from
            the attribute. An empty list marks a function that takes a tensordict, without keyword calls.
            Defaults to ``None``, which means that the function takes tensors.
        out_keys (list of NestedKey, optional): the output keys, if the module returns a tensordict. If
            ``module`` has an ``out_keys`` attribute, that attribute is used and this argument is ignored,
            with a ``UserWarning`` if the argument is a non-empty str, list, tuple or set of keys that differs
            from the attribute.
            Defaults to ``None``, which becomes ``[]`` when ``in_keys`` is given.
        device (torch.device, optional): the device of the CUDA streams that run the warmup and the capture.
            Defaults to ``None``, the current CUDA device.

    Examples:
        The example needs a CUDA device, so its lines are skipped when the
        docs are tested.

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import CudaGraphModule, TensorDictModule
        >>> # Wrap a simple function
        >>> def func(x):
        ...     return x + 1
        >>> func = CudaGraphModule(func)  # doctest: +SKIP
        >>> x = torch.rand((), device='cuda')  # doctest: +SKIP
        >>> out = func(x)  # doctest: +SKIP
        >>> assert isinstance(out, torch.Tensor)  # doctest: +SKIP
        >>> assert out == x+1  # doctest: +SKIP
        >>> # Wrap a tensordict module
        >>> func = TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["y"])
        >>> func = CudaGraphModule(func)  # doctest: +SKIP
        >>> # This can be called either with a TensorDict or regular keyword arguments alike
        >>> y = func(x=x)  # doctest: +SKIP
        >>> td = TensorDict(x=x)  # doctest: +SKIP
        >>> td = func(td)  # doctest: +SKIP

    .. note:: Tips on debugging CudaGraphModule errors:

        - Debug errors such as ``operation would make the legacy stream depend on a capturing blocking stream``
          on a version of your code that is not compiled first: compiled code hides the part of the code that
          creates the cross-stream dependency. The cause is often an operation between devices, which makes
          the capturing stream depend on other streams.
        - Errors such as ``Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture``, or other
          errors that come from the compiler (not the compiled code), usually mean that the graph was captured
          while the function was recompiling. List the recompiles with
          ``TORCH_LOGS="+recompiles" python myscript.py`` and remove them, or use more warmup calls.

    """

    _REQUIRES_GRAD_ERROR = (
        "CudaGraphModule cannot be part of a graph (or leaves variables). If you need tensors to "
        "require gradients, please pass them as constant inputs to the module instead (e.g. "
        "`def func(a, b, c=c): ...` where c is the tensor requiring gradients)."
    )

    def __init__(
        self,
        module: Callable[[List[Tensor] | TensorDictBase], None],
        warmup: int = 2,
        in_keys: List[NestedKey] = None,
        out_keys: List[NestedKey] = None,
        device: torch.device | None = None,
    ):
        self._has_cuda = torch.cuda.is_available()
        if not self._has_cuda:
            warnings.warn(
                "This module is instantiated on a machine without CUDA device available. "
                "We permit this usage, but calls to this instance will just pass through it and execute "
                "the provided function without additional performance gain."
            )
        self.module = module
        self.counter = 0
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be an integer greater than 0.")
        self._warmup = warmup
        if torch.cuda.is_available():
            self._warmup_stream = torch.cuda.Stream(device)
            self._capture_stream = torch.cuda.Stream(device)
            self._warmup_stream_cm = partial(torch.cuda.stream, self._warmup_stream)
        else:
            self._warmup_stream = None
            self._warmup_stream_cm = contextlib.nullcontext
        self.device = device
        tensordict_logger.info(
            f"Setting up CudaGraphModule with stream {self._warmup_stream} on device {self.device}."
        )

        for name, keys in (("in_keys", in_keys), ("out_keys", out_keys)):
            if hasattr(module, name) and _keys_differ(keys, getattr(module, name)):
                warnings.warn(
                    f"CudaGraphModule ignores {name}={keys} because the module has "
                    f"its own {name}={getattr(module, name)}, which it uses instead. "
                    f"Leave {name} unset, or pass an empty list, to silence this warning.",
                    UserWarning,
                    stacklevel=_user_stacklevel(),
                )
        if hasattr(module, "in_keys"):
            self.in_keys = module.in_keys
        else:
            self.in_keys = in_keys
        if hasattr(module, "out_keys"):
            self.out_keys = module.out_keys
        else:
            if out_keys is None and self.in_keys is not None:
                out_keys = []
            self.out_keys = out_keys
        self._is_tensordict_module = self.in_keys is not None
        self._out_matches_in = None
        self_ref = weakref.proxy(self)
        if self._is_tensordict_module:

            @dispatch(source=self.in_keys, dest=self.out_keys, auto_batch_size=False)
            def _call(
                tensordict: TensorDictBase,
                *args,
                tensordict_out: TensorDictBase | None = None,
                **kwargs: Any,
            ) -> Any:
                self = self_ref
                if self.counter >= self._warmup:
                    # Copy into the leaves the captured kernels read. The module may
                    # have rebound entries of ``self._tensordict`` to its outputs
                    # during capture (an output written under an input key), so
                    # ``self._tensordict`` is not a safe update target. Only the
                    # capture-time input leaves are copied; extra keys are ignored.
                    if args or kwargs or self._graph_args:
                        self._warn_if_args_changed(args, kwargs)
                    if self._prob_modules:
                        self._warn_if_interaction_type_changed(
                            self._args_warning_stacklevel
                        )
                    if tensordict.batch_size != self._graph_batch_size:
                        raise ValueError(
                            f"{self.__class__.__name__} was captured with batch_size="
                            f"{self._graph_batch_size} but got batch_size="
                            f"{tensordict.batch_size}. CUDA graphs replay fixed shapes."
                        )
                    src_keys, src_vals = tensordict._items_list(
                        True,
                        True,
                        sorting_keys=self._graph_input_keys,
                        default="intersection",
                    )
                    # The copies and the replay run on the current stream, so the
                    # replay reads the new inputs without a host synchronization. The
                    # copies are blocking so that the caller may overwrite a (pinned)
                    # CPU input as soon as the call returns.
                    if len(src_keys) == len(self._graph_input_keys):
                        if [val.shape for val in src_vals] != self._graph_input_shapes:
                            for key, val, shape in zip(
                                src_keys, src_vals, self._graph_input_shapes
                            ):
                                self._check_entry_shape(key, val.shape, shape)
                        if src_vals:
                            torch._foreach_copy_(self._graph_input_vals, src_vals)
                    else:
                        graph_keys = self._graph_inputs.keys(include_nested=True)
                        input_keys = tensordict.keys(include_nested=True)
                        missing = [
                            key
                            for key in self.in_keys
                            if key in graph_keys and key not in input_keys
                        ]
                        if missing:
                            raise KeyError(
                                f"{self.__class__.__name__} input is missing the in_keys "
                                f"{missing} that the captured graph reads."
                            )
                        # Other keys differ, or the layout does (e.g. a lazy stack
                        # against a dense tensordict): ``update_`` matches the leaves.
                        graph_leaves = self._graph_inputs.keys(
                            include_nested=True, leaves_only=True
                        )
                        for key in tensordict.keys(
                            include_nested=True, leaves_only=True
                        ):
                            if key in graph_leaves:
                                self._check_entry_shape(
                                    key,
                                    tensordict.get_item_shape(key),
                                    self._graph_inputs.get_item_shape(key),
                                )
                        self._graph_inputs.update_(tensordict)
                    self.graph.replay()
                    if self._out_matches_in:
                        # Clone: the graph overwrites ``self._out`` on the next replay.
                        result = tensordict.update(  # type-ignore[unreachable]
                            self._out, keys_to_update=self._selected_keys, clone=True
                        )
                    elif tensordict_out is not None:
                        result = tensordict_out.update(self._out, clone=True)
                    else:
                        result = self._out.clone() if self._out is not None else None
                    return result

                if not self._has_cuda or self.counter < self._warmup - 1:
                    # We must clone the data because providing non-contiguous data will fail later when we clone
                    if self._has_cuda:
                        self._warmup_stream.wait_stream(torch.cuda.current_stream())
                    with self._warmup_stream_cm():
                        tensordict.apply(self._clone, out=tensordict)
                        if tensordict_out is not None:
                            kwargs["tensordict_out"] = tensordict_out
                        out = self.module(tensordict, *args, **kwargs)
                        if self._out_matches_in is None:
                            self._out_matches_in = out is tensordict
                    if self._has_cuda:
                        torch.cuda.current_stream().wait_stream(self._warmup_stream)
                    self.counter += self._has_cuda
                    return out
                else:
                    tensordict_logger.info("Registering CUDA graph...")
                    if tensordict.device is None:
                        tensordict.apply(self._check_device_and_grad, filter_empty=True)
                    elif tensordict.device.type != "cuda":
                        raise ValueError(
                            "The input tensordict device must be of the 'cuda' type."
                        )

                    _tree_map(self._check_non_tensor, (args, kwargs))
                    # The replay reads the captured arguments: keep them to
                    # warn when a later call passes others.
                    self._graph_args = _tree_leaves((args, kwargs))
                    self._record_interaction_types()

                    torch.cuda.synchronize(self.device)
                    kwargs = kwargs.copy()
                    self._warmup_stream.wait_stream(
                        torch.cuda.current_stream(self.device)
                    )
                    with self._warmup_stream_cm():
                        tensordict.apply(self._clone, out=tensordict)
                        # A clone, not a copy: the caller keeps ``tensordict``, and the
                        # replays write their inputs into the leaves the graph reads.
                        self._tensordict = tensordict.clone()
                        # References to the input leaves as the graph will read
                        # them, kept apart from ``self._tensordict`` whose entries
                        # the module may rebind while it runs under capture.
                        self._graph_inputs = self._tensordict.copy()
                        (
                            self._graph_input_keys,
                            self._graph_input_vals,
                        ) = self._graph_inputs._items_list(True, True)
                        self._graph_input_shapes = [
                            val.shape for val in self._graph_input_vals
                        ]
                        self._graph_batch_size = self._graph_inputs.batch_size
                        if tensordict_out is not None:
                            td_out_save = tensordict_out.copy()
                            kwargs["tensordict_out"] = tensordict_out
                        this_out = self.module(tensordict, *args, **kwargs)
                    self._capture_stream.wait_stream(self._warmup_stream)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self._capture_stream):
                        if tensordict_out is not None:
                            kwargs["tensordict_out"] = td_out_save
                        out = self.module(self._tensordict, *args, **kwargs)
                    tensordict_logger.info("CUDA graph successfully registered.")

                    if not is_tensor_collection(out) and out is not None:
                        raise RuntimeError(
                            "The output of the function must be a tensordict, a tensorclass or None. Got "
                            f"type(out)={type(out)}."
                        )
                    if is_tensor_collection(out):
                        out.lock_()
                    self._out = out
                    self.counter += 1
                    if self._out_matches_in:
                        # We need to know what keys to update in out
                        if self.out_keys:
                            self._selected_keys = self.out_keys
                        else:
                            # Gather keys that have changed during execution
                            self._selected_keys = []

                            def check_tensor_id(name, t0, t1):
                                if t0 is not t1:
                                    self._selected_keys.append(name)

                            self._out.named_apply(
                                check_tensor_id,
                                self._graph_inputs,
                                default=None,
                                filter_empty=True,
                            )
                    return this_out

            # The frames between the warning and the user's call: this method,
            # ``_call``, the ``dispatch`` wrapper if there is one, ``__call__``.
            self._args_warning_stacklevel = 5 if hasattr(_call, "__wrapped__") else 4

        else:

            def _call(*args: torch.Tensor, **kwargs: torch.Tensor):
                self = self_ref
                if self.counter >= self._warmup:
                    srcs, dests = [], []
                    for arg_src, arg_dest in _zip_strict(
                        _tree_leaves((args, kwargs)), self._flat_tree
                    ):
                        self._maybe_copy_onto_(arg_src, arg_dest, srcs, dests)
                    if dests:
                        torch._foreach_copy_(dests, srcs)
                    if self._prob_modules:
                        self._warn_if_interaction_type_changed(4)
                    self.graph.replay()
                    if self._return_unchanged:
                        result = self._out
                    else:
                        result = tree_unflatten(
                            [
                                out.clone() if hasattr(out, "clone") else out
                                for out in self._out
                            ],
                            self._out_struct,
                        )
                    return result

                if not self._has_cuda or self.counter < self._warmup - 1:
                    args, kwargs = _tree_map(self._clone, (args, kwargs))
                    if self._has_cuda:
                        self._warmup_stream.wait_stream(
                            torch.cuda.current_stream(self.device)
                        )
                    with self._warmup_stream_cm():
                        out = self.module(*args, **kwargs)
                    if self._has_cuda:
                        torch.cuda.current_stream(self.device).wait_stream(
                            self._warmup_stream
                        )
                    self.counter += self._has_cuda
                    return out
                else:
                    tensordict_logger.info("Registering CUDA graph...")
                    self._record_interaction_types()
                    self._flat_tree, self._tree_spec = _tree_flatten((args, kwargs))

                    self._flat_tree = tuple(
                        self._check_device_and_clone(arg) for arg in self._flat_tree
                    )
                    args, kwargs = self._args, self._kwargs = tree_unflatten(
                        self._flat_tree, self._tree_spec
                    )

                    self._warmup_stream.wait_stream(
                        torch.cuda.current_stream(self.device)
                    )
                    with self._warmup_stream_cm():
                        # Run on fresh copies: a module that writes into a tensordict
                        # input must see the same structure during the capture.
                        args, kwargs = _tree_map(self._clone, (args, kwargs))
                        this_out = self.module(*args, **kwargs)
                    self._capture_stream.wait_stream(self._warmup_stream)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self._capture_stream):
                        out = self.module(*self._args, **self._kwargs)
                    tensordict_logger.info("CUDA graph successfully registered.")
                    self._out, self._out_struct = _tree_flatten(out)
                    self.counter += 1
                    # Check that there is not intersection between the indentity of inputs and outputs, otherwise warn
                    # user.
                    tree_leaves_input = _tree_leaves((self._args, self._kwargs))
                    tree_leaves_output = _tree_leaves(out)
                    if not isinstance(tree_leaves_output, tuple):
                        tree_leaves_output = (tree_leaves_output,)
                    for inp in tree_leaves_input:
                        if isinstance(inp, torch.Tensor) and inp in tree_leaves_output:
                            warnings.warn(
                                "An input tensor is also present in the output of the wrapped function. "
                                f"{type(self).__name__} will not copy tensors in place: the output will be cloned "
                                f"and the identity between input and output will not match anymore. "
                                f"Make sure you don't rely on input-output identity further in the code."
                            )
                    if not self._out:
                        self._return_unchanged = True
                    else:
                        self._out = [  # type: ignore[unreachable]
                            out.lock_() if is_tensor_collection(out) else out
                            for out in self._out
                        ]
                        self._return_unchanged = False
                    return this_out

        _call_func = functools.wraps(self.module)(_call)
        self._call_func = _call_func

    def _check_entry_shape(self, key, shape, captured_shape):
        if shape != captured_shape:
            raise ValueError(
                f"{self.__class__.__name__} was captured with the entry {key!r} of "
                f"shape {captured_shape} but got shape {shape}. CUDA graphs replay "
                "fixed shapes."
            )

    def _warn_if_args_changed(self, args, kwargs):
        # Compares the leaves, as the replay of a function that takes tensors
        # does in ``_maybe_copy_onto_``.
        leaves = _tree_leaves((args, kwargs))
        changed = len(leaves) != len(self._graph_args)
        for leaf, graph_leaf in zip(leaves, self._graph_args):
            if changed:
                break
            if leaf is graph_leaf or is_tensor_collection(graph_leaf):
                continue
            try:
                changed = bool(leaf != graph_leaf)
            except Exception:
                changed = True
        if changed:
            warnings.warn(
                f"{type(self).__name__} was captured with other non-tensor "
                "arguments than this call's. The replay ignores them and uses "
                "the captured values. Starting with TensorDict 0.17, the "
                "default will change to raising a ValueError, as for a "
                "function that takes tensors. Pass the values that change as "
                "entries of the tensordict, or use one CudaGraphModule per "
                "value (pytorch/tensordict#2075).",
                FutureWarning,
                stacklevel=self._args_warning_stacklevel,
            )

    def _interaction_types(self):
        # The interaction type each probabilistic module samples with.
        current = interaction_type()
        return [
            module.default_interaction_type if current is None else current
            for module in self._prob_modules
        ]

    def _record_interaction_types(self):
        modules = self.module.modules() if isinstance(self.module, nn.Module) else ()
        self._prob_modules = [
            module
            for module in modules
            if isinstance(module, ProbabilisticTensorDictModule)
        ]
        self._graph_interaction_types = self._interaction_types()

    def _warn_if_interaction_type_changed(self, stacklevel):
        interaction_types = self._interaction_types()
        if interaction_types != self._graph_interaction_types:
            # Warn once: later replays skip the check.
            self._prob_modules = []
            warnings.warn(
                f"{type(self).__name__} was captured with the interaction types "
                f"{[str(t) for t in self._graph_interaction_types]} but is called "
                f"with {[str(t) for t in interaction_types]}. The replay ignores "
                "the interaction type and repeats the captured branch. Capture one "
                f"{type(self).__name__} per interaction type "
                "(pytorch/tensordict#2547).",
                UserWarning,
                stacklevel=stacklevel,
            )

    @staticmethod
    def _maybe_copy_onto_(src, dest, srcs, dests):
        if isinstance(src, torch.Tensor):
            if src.shape != dest.shape:
                raise ValueError(
                    f"CudaGraphModule was captured with an input of shape {dest.shape} "
                    f"but got shape {src.shape}. CUDA graphs replay fixed shapes."
                )
            srcs.append(src)
            dests.append(dest)
            return
        if is_tensor_collection(src):
            if src.batch_size != dest.batch_size:
                raise ValueError(
                    f"CudaGraphModule was captured with batch_size={dest.batch_size} "
                    f"but got batch_size={src.batch_size}. CUDA graphs replay fixed shapes."
                )
            dest.copy_(src)
            return
        isdiff = False
        try:
            isdiff = src != dest
        except Exception as err:
            raise RuntimeError(
                "Couldn't assess input value. Make sure your function only takes tensor inputs or that "
                "the input value can be easily checked and is constant. For a better efficiency, avoid "
                "passing non-tensor inputs to your function."
            ) from err
        if isdiff:
            raise ValueError("Varying inputs must be torch.Tensor subclasses.")

    @classmethod
    def _check_device_and_clone(cls, x):
        if isinstance(x, torch.Tensor) or is_tensor_collection(x):
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)
            if x.device is None:
                # Check device of leaves of tensordict
                x.apply(cls._check_device_and_grad, filter_empty=True)

            elif x.device.type != "cuda":
                raise ValueError(
                    f"All tensors must be stored on CUDA. Got {x.device.type}."
                )

            return x.clone()
        return x

    @classmethod
    def _clone(cls, x):
        if isinstance(x, torch.Tensor) or is_tensor_collection(x):
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)
            return x.clone()
        return x

    @classmethod
    def _check_device_and_grad(cls, x):
        if isinstance(x, torch.Tensor):
            if x.device.type != "cuda":
                raise ValueError(
                    f"All tensors must be stored on CUDA. Got {x.device.type}."
                )
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)

    @staticmethod
    def _check_non_tensor(arg):
        if isinstance(arg, torch.Tensor):
            raise ValueError(
                "All tensors must be passed in the tensordict, not as arg or kwarg."
            )

    def __call__(self, *args, **kwargs):
        return self._call_func(*args, **kwargs)

    def __repr__(self):
        module = indent(f"module={self.module}", 4 * " ")
        warmup = indent(f"warmup={self._warmup}", 4 * " ")
        in_keys = indent(f"in_keys={self.in_keys}", 4 * " ")
        out_keys = indent(f"out_keys={self.out_keys}", 4 * " ")
        return f"{self.__class__.__name__}(\n{module}, \n{warmup}, \n{in_keys}, \n{out_keys}\n)"

    def state_dict(self):
        """Returns the ``state_dict`` of the wrapped module, or an empty dict if it has none."""
        try:
            return self.module.state_dict()
        except AttributeError:
            return {}

    def load_state_dict(self, state_dict):
        """Loads ``state_dict`` into the wrapped module. Does nothing if the module has no ``load_state_dict``."""
        try:
            self.module.load_state_dict(state_dict)
        except AttributeError:
            pass
