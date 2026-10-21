# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import sys
from copy import deepcopy
from typing import Any, Callable, Iterable, List, OrderedDict, overload, TYPE_CHECKING

from tensordict._nestedkey import NestedKey
from tensordict._td import TensorDict
from tensordict.nn.common import (
    dispatch,
    TensorDictModule,
    TensorDictModuleBase,
    WrapModule,
)
from tensordict.nn.utils import _set_skip_existing_None
from tensordict.tensordict import LazyStackedTensorDict, TensorDictBase
from tensordict.utils import _zip_strict, unravel_key_list
from torch import nn
from torch.compiler import is_compiling

_has_py311_or_greater = sys.version_info >= (3, 11)

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any


__all__ = ["TensorDictSequential"]


class TensorDictSequential(TensorDictModule):
    """Runs TensorDictModules one after the other on the same tensordict.

    Like :class:`torch.nn.Sequential`, each module receives the output of the
    previous one. Here the data is a tensordict, so a module can read any entry
    that an earlier module wrote.

    The ``in_keys`` of the sequence are the keys that a module reads and that no
    earlier module writes. The input tensordict must contain them. The
    ``out_keys`` are all the keys that the modules write, each listed once.
    To call the sequence with other parameters, use
    :meth:`~tensordict.TensorDictBase.to_module` (see the second example).
    See :ref:`modules-guide` for a guide.

    The modules are given as positional arguments, as one list, or as one
    dict (or ``OrderedDict``) that maps names to modules. ``seq[i]`` and
    ``seq["name"]`` return a module, and a slice returns a new sequence with
    the same options. Assigning or deleting ``seq[i]`` or ``seq["name"]``
    edits the sequence (a new name appends the module), and ``in_keys`` and
    ``out_keys`` are recomputed. A module registered under several names
    runs once for each name.

    Args:
        *modules (TensorDictModuleBase or Callable): the modules to run, in
            order. A callable without ``in_keys`` and ``out_keys``, such as a
            function that takes and returns a tensordict, is wrapped in a
            :class:`~tensordict.nn.WrapModule`. Its keys are then not part of
            the ``in_keys`` and ``out_keys`` of the sequence.

    Keyword Args:
        partial_tolerant (bool, optional): if ``True``, the input can lack
            some of the ``in_keys``: a module runs only if all its inputs are
            present, and is skipped otherwise. On a lazy stack, a module that
            cannot run on the whole stack runs on each tensordict of the stack
            that has its inputs, and its outputs are written into that
            tensordict. Defaults to ``False``.
        selected_out_keys (list of NestedKey, optional): the ``out_keys`` to
            keep in the output, as with :meth:`select_out_keys`. Each one must
            be in ``out_keys``. Defaults to ``None``, which keeps all of them.
        inplace (bool or str, optional): where the sequence writes its ``out_keys``
            after the last module ran.

            - ``True``: into the input tensordict, which is returned.
            - ``False``: into a new, empty :class:`~tensordict.TensorDict`.
            - ``"empty"``: into ``input.empty()``, which keeps the type, device
              and batch size of the input.
            - ``None``: the sequence returns what the last module returns,
              so the ``inplace`` value of each module decides.

            With ``False`` or ``"empty"``, the modules run on a shallow copy
            of the input, so the input does not change. See the last example.
            Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> _ = torch.manual_seed(0)
        >>> module = TensorDictSequential(
        ...     TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["x+1"]),
        ...     TensorDictModule(nn.Linear(3, 4), in_keys=["x+1"], out_keys=["w*(x+1)+b"]),
        ... )
        >>> # with tensordict input
        >>> print(module(TensorDict({"x": torch.zeros(3)}, [])))
        TensorDict(batch_size=[], fields={w*(x+1)+b: f32[4], x+1: f32[3], x: f32[3]})
        >>> module.in_keys, module.out_keys
        (['x'], ['x+1', 'w*(x+1)+b'])
        >>> # with tensor input: returns all the output keys in the order of the modules, ie "x+1" and "w*(x+1)+b"
        >>> module(x=torch.zeros(3))
        (tensor([1., 1., 1.]), tensor([-0.7214, -0.8748,  0.1571, -0.1138], grad_fn=<ViewBackward0>))
        >>> module(torch.zeros(3))
        (tensor([1., 1., 1.]), tensor([-0.7214, -0.8748,  0.1571, -0.1138], grad_fn=<ViewBackward0>))

    A sequence can hold probabilistic modules, and be called with other
    parameters through :meth:`~tensordict.TensorDictBase.to_module`, also
    under :func:`torch.vmap`:

        >>> from tensordict.nn import (
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ... )
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from torch.distributions import Normal
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> net1 = torch.nn.Linear(4, 8)
        >>> module1 = TensorDictModule(net1, in_keys=["input"], out_keys=["params"])
        >>> normal_params = TensorDictModule(
        ...      NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        ...  )
        >>> td_module1 = ProbabilisticTensorDictSequential(
        ...     module1,
        ...     normal_params,
        ...     ProbabilisticTensorDictModule(
        ...         in_keys=["loc", "scale"],
        ...         out_keys=["hidden"],
        ...         distribution_class=Normal,
        ...         return_log_prob=True,
        ...     )
        ... )
        >>> module2 = torch.nn.Linear(4, 8)
        >>> td_module2 = TensorDictModule(
        ...    module=module2, in_keys=["hidden"], out_keys=["output"]
        ... )
        >>> td_module = TensorDictSequential(td_module1, td_module2)
        >>> params = TensorDict.from_module(td_module)
        >>> with params.to_module(td_module):
        ...     _ = td_module(td)
        >>> sorted(td.keys())
        ['hidden', 'hidden_log_prob', 'input', 'loc', 'output', 'params', 'scale']
        >>> td["output"].shape
        torch.Size([3, 8])

    With :func:`torch.vmap` over 4 copies of the parameters, each entry gets
    a leading dimension of size 4:

        >>> from torch import vmap
        >>> params = params.expand(4)
        >>> def func(td, params):
        ...     with params.to_module(td_module):
        ...         return td_module(td)
        >>> td_vmap = vmap(func, (None, 0))(td, params)
        >>> td_vmap.batch_size
        torch.Size([4, 3])
        >>> td_vmap["output"].shape
        torch.Size([4, 3, 8])

    The ``inplace`` argument controls where the outputs go. In this example the
    inner modules use ``inplace=False``, so each one passes only its outputs to
    the next. The sequence uses ``inplace=True``, so it writes the final
    ``output`` into the input object. The intermediate entries are not written
    into it.

        >>> from tensordict import TensorClass
        >>> from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
        >>>
        >>> class MyClass(TensorClass):
        ...     input: torch.Tensor
        ...     output: torch.Tensor | None = None
        >>>
        >>> obj = MyClass(torch.randn(2, 3), batch_size=(2,))
        >>>
        >>> model = Seq(
        ...     Mod(
        ...         lambda x: (x + 1, x - 1),
        ...         in_keys=["input"],
        ...         out_keys=[("intermediate", "0"), ("intermediate", "1")],
        ...         inplace=False
        ...         ),
        ...     Mod(
        ...         lambda y0, y1: y0 * y1,
        ...         in_keys=[("intermediate", "0"), ("intermediate", "1")],
        ...         out_keys=["output"],
        ...         inplace=False
        ...         ),
        ...     inplace=True, )
        >>> print(model(obj))
        MyClass(batch_size=[2], fields={input: f32[3], output: f32[3]})

    """

    module: nn.ModuleList
    _select_before_return = False

    @property
    def out_keys(self):
        """The keys that the modules write, each listed once, or those chosen with :meth:`select_out_keys`."""
        return self._out_keys_apparent

    @out_keys.setter
    def out_keys(self, value: List[NestedKey]):
        self._out_keys = self._out_keys_apparent = unravel_key_list(list(value))
        self._complete_out_keys = list(self._out_keys)

    @overload
    def __init__(
        self,
        modules: OrderedDict[str, Callable[[TensorDictBase], TensorDictBase]],
        *,
        partial_tolerant: bool = False,
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        modules: List[Callable[[TensorDictBase], TensorDictBase]],
        *,
        partial_tolerant: bool = False,
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *modules: List[Callable[[TensorDictBase], TensorDictBase]],
        partial_tolerant: bool = False,
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | str | None = None,
    ) -> None: ...

    def __init__(
        self,
        *modules: Callable[[TensorDictBase], TensorDictBase],
        partial_tolerant: bool = False,
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | str | None = None,
    ) -> None:

        if len(modules) == 1 and isinstance(modules[0], collections.OrderedDict):
            modules_vals = self._convert_modules(modules[0].values())  # type: ignore[unreachable]
            in_keys, out_keys = self._compute_in_and_out_keys(modules_vals)
            self._complete_out_keys = list(out_keys)
            modules = collections.OrderedDict(_zip_strict(modules[0], modules_vals))
            super().__init__(
                module=nn.ModuleDict(modules),
                in_keys=in_keys,
                out_keys=out_keys,
            )
        elif len(modules) == 1 and isinstance(
            modules[0], collections.abc.MutableSequence
        ):
            modules = self._convert_modules(modules[0])  # type: ignore[unreachable]
            in_keys, out_keys = self._compute_in_and_out_keys(modules)
            self._complete_out_keys = list(out_keys)
            super().__init__(
                module=nn.ModuleList(modules),
                in_keys=in_keys,
                out_keys=out_keys,
            )
        elif len(modules) == 1 and isinstance(modules[0], dict):
            return self.__init__(  # type: ignore[unreachable]
                collections.OrderedDict(modules[0]),
                partial_tolerant=partial_tolerant,
                selected_out_keys=selected_out_keys,
                inplace=inplace,
            )
        else:
            modules = self._convert_modules(modules)
            in_keys, out_keys = self._compute_in_and_out_keys(modules)
            self._complete_out_keys = list(out_keys)
            super().__init__(
                module=nn.ModuleList(list(modules)),
                in_keys=in_keys,
                out_keys=out_keys,
            )

        self.inplace = inplace
        self.partial_tolerant = partial_tolerant
        if selected_out_keys:
            self._select_before_return = True
            selected_out_keys = unravel_key_list(selected_out_keys)
            if not all(key in self.out_keys for key in selected_out_keys):
                raise ValueError("All keys in selected_out_keys must be in out_keys.")
            self._out_keys_apparent = selected_out_keys
        else:
            self._select_before_return = False

    def reset_out_keys(self):
        """Undoes :meth:`select_out_keys`: the sequence writes all its ``out_keys`` again, and returns ``self``."""
        self.out_keys = list(self._complete_out_keys)
        return self

    @staticmethod
    def _convert_modules(modules):
        return [
            (
                WrapModule(module)
                if not TensorDictModuleBase.is_tdmodule_compatible(module)
                else module
            )
            for module in modules
        ]

    def _compute_in_and_out_keys(
        self, modules: list[TensorDictModule]
    ) -> tuple[list[NestedKey], list[NestedKey]]:
        in_keys = []  # type: ignore[var-annotated]
        out_keys = []  # type: ignore[var-annotated]
        for module in modules:
            # we sometimes use in_keys to select keys of a tensordict that are
            # necessary to run a TensorDictModule. If a key is an intermediary in
            # the chain, there is no reason why it should belong to the input
            # TensorDict.
            for in_key in module.in_keys:
                if in_key not in (out_keys + in_keys):
                    in_keys.append(in_key)
            out_keys += module.out_keys

        out_keys = [
            out_key
            for i, out_key in enumerate(out_keys)
            if out_key not in out_keys[i + 1 :]
        ]
        return in_keys, out_keys

    def select_out_keys(self, *selected_out_keys) -> TensorDictSequential:
        """Keeps only some of the ``out_keys`` in the output of the sequence, and returns the sequence.

        The sequence changes in place, and :meth:`reset_out_keys` undoes the
        change. The modules still compute all their outputs; the call then
        writes only the selected ones, and the entries of the input, into the
        output.

        Args:
            *selected_out_keys (NestedKey): the keys to keep. Each one must be
                in ``out_keys``.

        Returns:
            ``self``.

        Raises:
            ValueError: if a key in ``selected_out_keys`` is not part of ``out_keys``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
            >>> seq = Seq(
            ...     Mod(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
            ...     Mod(lambda y: y + 1, in_keys=["y"], out_keys=["z"]),
            ... )
            >>> _ = seq.select_out_keys("z")
            >>> seq.out_keys
            ['z']
            >>> td = seq(TensorDict(x=torch.zeros(())))
            >>> "y" in td
            False
            >>> td["z"]
            tensor(2.)
            >>> _ = seq.reset_out_keys()
            >>> seq.out_keys
            ['y', 'z']

        """
        self._select_before_return = True
        selected_out_keys = unravel_key_list(selected_out_keys)
        if not all(key in self.out_keys for key in selected_out_keys):
            raise ValueError("All keys in selected_out_keys must be in out_keys.")
        self._out_keys_apparent = selected_out_keys
        return self

    def select_subsequence(
        self,
        in_keys: Iterable[NestedKey] | None = None,
        out_keys: Iterable[NestedKey] | None = None,
    ) -> TensorDictSequential:
        """Returns a new sequence with only the modules that can run from ``in_keys`` and that ``out_keys`` need.

        Two filters are applied, in this order:

        - ``in_keys``: a module is kept if each of its inputs is in
          ``in_keys`` or is written by a module kept before it. The other
          modules, and those that read their outputs, are dropped.
        - ``out_keys``: of the modules left, a module is kept if it writes one
          of ``out_keys``, or an input of a module kept after it.

        A nested :class:`TensorDictSequential` is filtered the same way. The
        modules are not copied: the new sequence holds the same module
        objects. Modules of a named sequence keep their names, and the new
        sequence keeps ``partial_tolerant`` and ``inplace``.

        Args:
            in_keys (list of NestedKey, optional): the keys that the modules
                can start from. Defaults to ``None``, which uses the
                ``in_keys`` of the sequence.
            out_keys (list of NestedKey, optional): the keys to compute.
                Defaults to ``None``, which uses the ``out_keys`` of the
                sequence.

        Returns:
            A new sequence of the same type.

        Raises:
            ValueError: if no module is left.

        Examples:
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> idn = lambda x: x
            >>> module = Seq(
            ...     Mod(idn, in_keys=["a"], out_keys=["b"]),
            ...     Mod(idn, in_keys=["b"], out_keys=["c"]),
            ...     Mod(idn, in_keys=["c"], out_keys=["d"]),
            ...     Mod(idn, in_keys=["a"], out_keys=["e"]),
            ... )
            >>> # the modules that can run from "c"
            >>> module.select_subsequence(in_keys=["c"]).out_keys
            ['d']
            >>> # the modules that "c" depends on
            >>> sub = module.select_subsequence(out_keys=["c"])
            >>> sub.in_keys, sub.out_keys
            (['a'], ['b', 'c'])
            >>> sub[0] is module[0]
            True

        Nested sequences are filtered too:

            >>> module = Seq(
            ...     Seq(
            ...         Mod(idn, in_keys=["a"], out_keys=["b"]),
            ...         Mod(idn, in_keys=["b"], out_keys=["c"]),
            ...     ),
            ...     Seq(
            ...         Mod(idn, in_keys=["b"], out_keys=["d"]),
            ...         Mod(idn, in_keys=["d"], out_keys=["e"]),
            ...     ),
            ... )
            >>> sub = module.select_subsequence(in_keys=["b", "d"], out_keys=["e"])
            >>> len(sub), sub[0].out_keys
            (1, ['d', 'e'])

        """
        if in_keys is None:
            in_keys = deepcopy(self.in_keys)
        in_keys = unravel_key_list(in_keys)
        if out_keys is None:
            out_keys = deepcopy(self.out_keys)
        out_keys = unravel_key_list(out_keys)

        module_list = list(self._module_iter())
        id_to_keep = set(range(len(module_list)))
        for i, module in enumerate(module_list):
            if (
                type(module) is TensorDictSequential
            ):  # no isinstance because we don't want to mess up subclasses
                try:
                    module = module_list[i] = module.select_subsequence(in_keys=in_keys)
                except ValueError:
                    # then the module can be removed
                    id_to_keep.remove(i)
                    continue

            if all(key in in_keys for key in module.in_keys):
                in_keys.extend(module.out_keys)
            else:
                id_to_keep.remove(i)
        for i, module in reversed(list(enumerate(module_list))):
            if i in id_to_keep:
                if any(key in out_keys for key in module.out_keys):
                    if (
                        type(module) is TensorDictSequential
                    ):  # no isinstance because we don't want to mess up subclasses
                        module = module_list[i] = module.select_subsequence(
                            out_keys=out_keys
                        )
                    out_keys.extend(module.in_keys)
                else:
                    id_to_keep.remove(i)
        id_to_keep = sorted(id_to_keep)

        modules = [module_list[i] for i in id_to_keep]

        if modules == []:
            raise ValueError(
                "No modules left after selection. Make sure that in_keys and out_keys are coherent."
            )
        if isinstance(self.module, nn.ModuleList):
            return self._keep_options(type(self)(*modules))
        else:
            keys = list(self.module)
            modules_dict = collections.OrderedDict(
                (keys[i], module_list[i]) for i in id_to_keep
            )
            return self._keep_options(type(self)(modules_dict))

    def _keep_options(self, sequence: TensorDictSequential) -> TensorDictSequential:
        """Gives ``sequence``, built from some of this sequence's modules, the options of this sequence.

        The options are set after construction, because subclasses do not all
        take them as constructor arguments.
        """
        sequence.partial_tolerant = self.partial_tolerant
        sequence.inplace = self.inplace
        return sequence

    def _run_module(
        self,
        module: TensorDictModuleBase,
        tensordict: TensorDictBase,
        **kwargs: Any,
    ) -> Any:
        if not self.partial_tolerant or all(
            key in tensordict.keys(include_nested=True) for key in module.in_keys
        ):
            tensordict = module(tensordict, **kwargs)
        elif self.partial_tolerant and isinstance(tensordict, LazyStackedTensorDict):
            for sub_td in tensordict.tensordicts:
                if all(
                    key in sub_td.keys(include_nested=True) for key in module.in_keys
                ):
                    out = module(sub_td, **kwargs)
                    # Keep the outputs of modules that do not write in place.
                    if isinstance(out, TensorDictBase) and out is not sub_td:
                        out_keys = [
                            key
                            for key in module.out_keys
                            if key in out.keys(include_nested=True)
                        ]
                        if out_keys:
                            sub_td.update(out, keys_to_update=out_keys)
        return tensordict

    def _module_iter(self):
        if isinstance(self.module, nn.ModuleDict):
            yield from self.module.values()
        else:
            yield from self.module

    def _get_module_num_or_key(self, mod: nn.Module) -> int | str:
        if isinstance(self.module, nn.ModuleDict):
            for name, m in self.module.named_children():
                if m is mod:
                    return name
            else:
                raise RuntimeError("module not found.")
        else:
            for i, m in enumerate(self.module):
                if m is mod:
                    return i
            else:
                raise RuntimeError("module not found.")

    @dispatch(auto_batch_size=False)
    @_set_skip_existing_None()
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        """Runs the modules in order on ``tensordict`` and returns the result (see ``inplace``).

        If you call the sequence with tensors instead of a tensordict, the
        call returns the output tensors (see :func:`~tensordict.nn.dispatch`).

        Args:
            tensordict (TensorDictBase): the input tensordict.
            tensordict_out (TensorDictBase, optional): if given, the
                ``out_keys`` are written into it and it is returned, whatever
                the value of ``inplace``. Defaults to ``None``.

        Raises:
            RuntimeError: if other keyword arguments are given.
        """
        if (
            tensordict_out is not None
            or self._select_before_return
            or self.inplace is False
            or self.inplace == "empty"
        ):
            # With inplace=False or "empty", the modules must not write into the input.
            tensordict_exec = tensordict.copy()
        else:
            tensordict_exec = tensordict
        if tensordict_out is None:
            if self.inplace is True:
                tensordict_out = tensordict
            elif self.inplace is False:
                tensordict_out = TensorDict()
            elif self.inplace == "empty":
                tensordict_out = tensordict.empty()

        if not len(kwargs):
            for module in self._module_iter():
                try:
                    tensordict_exec = self._run_module(
                        module, tensordict_exec, **kwargs
                    )
                except Exception as e:
                    if _has_py311_or_greater:
                        module_num_or_key = self._get_module_num_or_key(module)
                        e.add_note(
                            f"Failed while executing module '{module_num_or_key}'."
                        )
                    raise
        else:
            raise RuntimeError(
                f"TensorDictSequential does not support keyword arguments other than 'tensordict_out' or in_keys: {self.in_keys}. Got {kwargs.keys()} instead."
            )
        if tensordict_out is not None:
            result = tensordict_out
            result.update(tensordict_exec, keys_to_update=self.out_keys)
        else:
            result = tensordict_exec
            if self._select_before_return:
                # We must also update any value that has been updated during the course of execution
                # from the input data.
                if is_compiling():
                    keys = [  # noqa: C416
                        k
                        for k in {k for k in self.out_keys}.union(  # noqa: C416
                            {k for k in tensordict.keys(True, True)}  # noqa: C416
                        )
                    ]
                else:
                    keys = list(set(self.out_keys).union(tensordict.keys(True, True)))
                return tensordict.update(result, keys_to_update=keys)
        return result

    def __len__(self) -> int:
        return len(self.module)

    def __getitem__(self, index: int | slice | str) -> Self | TensorDictModuleBase:
        if isinstance(index, (int, str)):
            return self.module.__getitem__(index)
        else:
            return self._keep_options(type(self)(*self.module.__getitem__(index)))

    def __setitem__(
        self, index: int | slice | str, tensordict_module: TensorDictModuleBase
    ) -> None:
        if callable(tensordict_module):
            # wrap a plain callable as the constructor does
            (tensordict_module,) = self._convert_modules([tensordict_module])
        self._edit_modules(
            lambda modules: modules.__setitem__(index, tensordict_module)
        )

    def __delitem__(self, index: int | slice | str) -> None:
        self._edit_modules(lambda modules: modules.__delitem__(index))

    def _edit_modules(self, edit: Callable[[nn.Module], None]) -> None:
        """Applies ``edit`` to the module container and updates the keys.

        The edit is first tried on a copy of the container, so that an edit
        that fails, or that :meth:`_check_modules` rejects, leaves the sequence
        unchanged.
        """
        if isinstance(self.module, nn.ModuleDict):
            candidate = nn.ModuleDict(collections.OrderedDict(self.module.items()))
            edit(candidate)
            modules = list(candidate.values())
        else:
            candidate = nn.ModuleList(list(self.module))
            edit(candidate)
            modules = list(candidate)
        self._check_modules(modules)
        edit(self.module)
        self._refresh_keys()

    def _check_modules(self, modules: list[nn.Module]) -> None:
        """Raises if the sequence cannot hold ``modules`` after an edit."""
        if self._select_before_return:
            _, out_keys = self._compute_in_and_out_keys(modules)
            missing = [key for key in self._out_keys_apparent if key not in out_keys]
            if missing:
                raise ValueError(
                    f"The edit would remove the selected out_keys {missing}. Call "
                    "reset_out_keys() before the edit, or select other keys."
                )

    def _refresh_keys(self) -> None:
        """Recomputes ``in_keys`` and ``out_keys`` from the modules after an edit."""
        in_keys, out_keys = self._compute_in_and_out_keys(list(self._module_iter()))
        selected = self._out_keys_apparent if self._select_before_return else None
        self.in_keys = in_keys
        self.out_keys = out_keys
        if selected is not None:
            self._out_keys_apparent = selected
