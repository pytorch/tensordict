# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Point-to-point and collective communication with torch.distributed of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch
from tensordict.base import (
    _check_p2p_peer,
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _resolve_tensorclass_type,
    NO_DEFAULT,
    Self,
    T,
)
from tensordict.utils import _int_generator
from torch import Tensor

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase


class _Distributed:
    """Point-to-point and collective communication with torch.distributed."""

    # Distributed functionality
    def gather_and_stack(
        self, dst: int, group: "torch.distributed.ProcessGroup" | None = None
    ) -> Self | None:
        """Stacks the tensordicts of the other ranks into the tensordict of rank ``dst``, in place.

        Every rank of the group must call it. On rank ``dst``, the tensordict must
        have one row per other rank: its first batch dimension is
        ``world_size - 1``, and its own data is not in the result. The
        tensordicts are sent with :func:`torch.distributed.gather_object`, which
        pickles them: use it only with ranks you trust. See
        :ref:`distributed-collectives`.

        Args:
            dst (int): the rank that receives the stack.
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.

        Returns:
            ``self`` on rank ``dst``, ``None`` on the other ranks.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def gather_and_stack(rank, world_size):
                    if rank == 0:
                        out = TensorDict(a=torch.zeros(world_size - 1, 3), batch_size=[world_size - 1])
                        out.gather_and_stack(dst=0)
                        print(out["a"][:, 0])  # with 2 ranks: tensor([1.])
                    else:
                        td = TensorDict(a=torch.full((3,), float(rank)), batch_size=[3])
                        td.gather_and_stack(dst=0)
        """
        from torch import distributed as dist

        output = (
            [None for _ in range(dist.get_world_size(group=group))]
            if dst == dist.get_rank(group=group)
            else None
        )
        dist.gather_object(self, output, dst=dst, group=group)
        if dst == dist.get_rank(group=group):
            # remove self from output
            output = [item for i, item in enumerate(output) if i != dst]
            self.update(torch.stack(output, 0), inplace=True)
            return self
        return None

    def send(
        self,
        dst: int | "TensorDictPipe" | None = None,  # noqa: F821
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_dst: int | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        consolidated: bool = False,
    ) -> None:  # noqa: D417
        """Sends the leaves of the tensordict to another rank, with :mod:`torch.distributed`.

        The receiver calls :meth:`~.recv` on a tensordict with the same keys,
        shapes and dtypes, which writes the values into its tensors. By default,
        each leaf is one message, marked with a tag. To send to a rank that does
        not know the structure, use :meth:`~.init_remote`. See :ref:`distributed`.

        Args:
            dst (int or TensorDictPipe, optional): the global rank of the
                receiver, or a :class:`~tensordict._ucxx.TensorDictPipe` to send
                through with UCXX, which ignores the other arguments. Exclusive
                with ``group_dst``: pass exactly one of them. Defaults to ``None``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            group_dst (int, optional): the rank of the receiver in ``group``,
                which must be given. Exclusive with ``dst``. The messages then go
                through the backend of ``group`` directly, so ``group`` can be a
                standalone :class:`~torch.distributed.ProcessGroup` built on a store,
                never registered by :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`. Defaults to ``None``.
            init_tag (int, optional): the tag of the first leaf. The next leaves
                use ``init_tag + 1``, ``init_tag + 2``, and so on. The receiver must
                pass the same value. Give two transfers that run at the same time
                tag ranges that do not overlap. Defaults to ``0``.
            pseudo_rand (bool, optional): if ``True``, the tags are pseudo-random
                numbers seeded by ``init_tag``, so that transfers from several
                ranks do not overlap. Generating them costs about 10
                microseconds per leaf. The receiver must pass the same value.
                Defaults to ``False``.
            consolidated (bool, optional): if ``True``, sends all the leaves as one
                message: the tensordict is consolidated first (see
                :meth:`~.consolidate`), unless it is already. The receiver must call
                ``recv(consolidated=True)`` on a consolidated tensordict with the
                same structure. ``init_tag`` and ``pseudo_rand`` are ignored.
                Defaults to ``False``.

        Raises:
            ValueError: if both or none of ``dst`` and ``group_dst`` are given, or
                if ``group_dst`` is given without ``group``.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def send_recv(rank, world_size):
                    if rank == 0:
                        td = TensorDict(a=torch.ones(4), b=torch.ones(4, 3), batch_size=[4])
                        td.send(dst=1)
                    elif rank == 1:
                        td = TensorDict(a=torch.zeros(4), b=torch.zeros(4, 3), batch_size=[4])
                        td.recv(src=0)
                        print(td["b"].sum())  # tensor(12.)
        """
        from tensordict._ucxx import TensorDictPipe

        if isinstance(dst, TensorDictPipe):
            if group_dst is not None:
                raise ValueError(
                    "`group_dst` cannot be used when `dst` is a TensorDictPipe."
                )
            dst.send(self)
            return
        _check_p2p_peer(dst, group_dst, group, "dst", "group_dst")
        if consolidated:
            from torch import distributed as dist

            td_c = self if self.is_consolidated() else self.consolidate(metadata=True)
            storage = td_c._consolidated["storage"]
            if group_dst is not None:
                group.send([storage], group_dst, 0).wait()
            else:
                dist.send(storage, dst=dst, group=group)
            return
        self._send(
            dst,
            _tag=init_tag - 1,
            pseudo_rand=pseudo_rand,
            group=group,
            group_dst=group_dst,
        )

    def _send(
        self,
        dst: int | None,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_dst: int | None = None,
    ) -> int:
        from torch import distributed as dist

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(type(value)):
                _tag = value._send(
                    dst,
                    _tag=_tag,
                    pseudo_rand=pseudo_rand,
                    group=group,
                    group_dst=group_dst,
                )
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = _int_generator(_tag + 1)
            if group_dst is not None:
                # Direct backend call: works for raw (unregistered) groups,
                # which the functional API rejects.
                group.send([value], group_dst, _tag).wait()
            else:
                dist.send(value, dst=dst, tag=_tag, group=group)

        return _tag

    def recv(
        self,
        src: int | "TensorDictPipe" | None = None,  # noqa: F821
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_src: int | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        consolidated: bool = False,
    ) -> int:  # noqa: D417
        """Receives the leaves that :meth:`~.send` sends, and writes them into the tensordict in place.

        The tensordict must have the keys, shapes and dtypes of the tensordict
        that the sender sends. See :meth:`~.send` for an example, and
        :ref:`distributed`.

        Args:
            src (int or TensorDictPipe, optional): the global rank of the sender,
                or a :class:`~tensordict._ucxx.TensorDictPipe` to receive from
                with UCXX, which ignores the other arguments. Exclusive with
                ``group_src``: pass exactly one of them. Defaults to ``None``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            group_src (int, optional): the rank of the sender in ``group``, which
                must be given. Exclusive with ``src``. The messages then go through
                the backend of ``group`` directly, so ``group`` can be a standalone
                :class:`~torch.distributed.ProcessGroup` built on a store, never
                registered by :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`. Defaults to ``None``.
            init_tag (int, optional): the ``init_tag`` that the sender passed.
                Defaults to ``0``.
            pseudo_rand (bool, optional): the ``pseudo_rand`` that the sender
                passed. Defaults to ``False``.
            consolidated (bool, optional): if ``True``, receives one message into
                the consolidated storage of the tensordict, which updates all its
                leaves, as sent by ``send(consolidated=True)``. The tensordict must
                be consolidated already, for example by
                ``consolidate(metadata=True)`` or :meth:`~.from_remote_init`;
                otherwise an ``AttributeError`` is raised. ``init_tag`` and
                ``pseudo_rand`` are ignored. Defaults to ``False``.

        Returns:
            With a pipe, the received tensordict, which is ``self`` only if the
            tensordict is consolidated with the layout of the data, so use the
            return value. With ``consolidated=True``, ``None``. Otherwise, the
            tag of the last leaf received.

        Raises:
            ValueError: if both or none of ``src`` and ``group_src`` are given, or
                if ``group_src`` is given without ``group``.
        """
        from tensordict._ucxx import TensorDictPipe

        if isinstance(src, TensorDictPipe):
            if group_src is not None:
                raise ValueError(
                    "`group_src` cannot be used when `src` is a TensorDictPipe."
                )
            return src.recv(self)
        _check_p2p_peer(src, group_src, group, "src", "group_src")
        if consolidated:
            from torch import distributed as dist

            storage = self._consolidated["storage"]
            if group_src is not None:
                group.recv([storage], group_src, 0).wait()
            else:
                dist.recv(storage, src=src, group=group)
            return
        return self._recv(
            src,
            _tag=init_tag - 1,
            pseudo_rand=pseudo_rand,
            group=group,
            group_src=group_src,
        )

    async def asend(self, dst: "TensorDictPipe") -> None:  # noqa: F821
        """Sends the tensordict through a UCXX pipe, asynchronously.

        Awaits ``dst.asend(self)``. See :class:`~tensordict._ucxx.TensorDictPipe`,
        which needs the optional ``ucxx`` package.

        Args:
            dst (TensorDictPipe): the pipe to send through.

        .. seealso:: :meth:`~.send`, which also takes a pipe, and works without
            ``await``.

        Examples:
            This example needs ``ucxx`` and a receiver on the other end of the
            pipe:

            .. code-block:: python

                async def produce(pipe):
                    for _ in range(10):
                        await TensorDict(obs=torch.randn(64, 84)).asend(pipe)
        """
        await dst.asend(self)

    async def arecv(
        self,
        src: "TensorDictPipe",  # noqa: F821
        *,
        device: torch.device | str | None = None,
    ) -> "TensorDictBase":
        """Receives a tensordict from a UCXX pipe, into the tensordict if possible, asynchronously.

        Awaits ``src.arecv(self, device=device)``. If the tensordict is
        consolidated with the layout of the received data, the data is written
        into its storage. Otherwise, the pipe returns another tensordict: use the
        return value. See :class:`~tensordict._ucxx.TensorDictPipe`, which needs
        the optional ``ucxx`` package.

        Args:
            src (TensorDictPipe): the pipe to receive from.

        Keyword Args:
            device (torch.device, str or int, optional): the device of the storage
                that the pipe allocates for a new structure. Defaults to ``None``,
                which uses the device of the sender with a consolidated pipe, and
                the CPU otherwise.

        Returns:
            The received tensordict, ``self`` or another tensordict.

        .. seealso:: :meth:`~.recv`, which also takes a pipe, and works without
            ``await``.

        Examples:
            This example needs ``ucxx`` and a sender on the other end of the pipe:

            .. code-block:: python

                async def consume(pipe):
                    td = await TensorDict().arecv(pipe)  # first receive: allocates
                    for _ in range(9):
                        td = await td.arecv(pipe)  # same structure: writes into td
        """
        return await src.arecv(self, device=device)

    def _recv(
        self,
        src: int | None,
        _tag: int = -1,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
        non_blocking: bool = False,
        group_src: int | None = None,
    ) -> int:
        from torch import distributed as dist

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(type(value)):
                _tag = value._recv(
                    src,
                    _tag=_tag,
                    pseudo_rand=pseudo_rand,
                    group=group,
                    group_src=group_src,
                )
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = _int_generator(_tag + 1)
            if group_src is not None:
                # Direct backend call: works for raw (unregistered) groups,
                # which the functional API rejects.
                group.recv([value], group_src, _tag).wait()
            else:
                dist.recv(value, src=src, tag=_tag, group=group)
            self._set_str(
                key, value, inplace=True, validated=True, non_blocking=non_blocking
            )

        return _tag

    def init_remote(
        self,
        dst: int | None = None,
        group: "ProcessGroup" | None = None,  # noqa: F821
        device: torch.device | None = None,
        use_broadcast: bool = False,
        _tensorclass_type: str | None = None,
    ) -> None:
        """Sends the structure and the content of the tensordict to a rank that does not know them.

        The receiver calls :meth:`~.from_remote_init`, which builds a new
        consolidated tensordict. The tensordict is consolidated (see
        :meth:`~.consolidate`), then its metadata is sent with
        :func:`torch.distributed.send_object_list`, which pickles it, and its
        storage as one tensor. Send the next batches with
        ``send(consolidated=True)``, into the tensordict that
        :meth:`~.from_remote_init` returned. See :ref:`distributed`.

        With ``use_broadcast=True``, it calls :meth:`~.broadcast` instead, and
        every rank of the group must take part: the others call
        ``from_remote_init(src=..., use_broadcast=True)``.

        Args:
            dst (int, optional): the rank of the receiver. Required unless
                ``use_broadcast=True``, which ignores it. Defaults to ``None``.
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            device (torch.device, str or int, optional): point-to-point, the device
                that the metadata goes through, as in
                :func:`torch.distributed.send_object_list` (with ``nccl``, a CUDA
                device). With ``use_broadcast=True``, the device of the
                consolidated storage, as in :meth:`~.broadcast`. Defaults to
                ``None``.
            use_broadcast (bool, optional): if ``True``, broadcasts the tensordict
                to every rank of the group. Defaults to ``False``.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def cold_start(rank, world_size):
                    if rank == 0:
                        td = TensorDict(obs=torch.ones(2, 84), reward=torch.ones(2, 1), batch_size=[2])
                        td.init_remote(dst=1)
                        td["obs"].add_(1)
                        td.send(dst=1, consolidated=True)
                    elif rank == 1:
                        td = TensorDict.from_remote_init(src=0)
                        print(td.batch_size, td.is_consolidated())  # torch.Size([2]) True
                        td.recv(src=0, consolidated=True)
                        print(td["obs"][0, 0])  # tensor(2.)
        """
        if use_broadcast:
            rank = torch.distributed.get_rank(group=group)
            self.broadcast(
                src=rank,
                group=group,
                device=device,
                _tensorclass_type=_tensorclass_type,
            )
            return

        td_c = self.consolidate(metadata=True)
        storage = td_c._consolidated["storage"]
        metadata = td_c._consolidated["metadata"]
        metadata["_total_bytes"] = storage.numel()
        if _tensorclass_type is not None:
            metadata["_tensorclass_type"] = _tensorclass_type
        torch.distributed.send_object_list(
            [metadata],
            dst=dst,
            group=group,
            device=device,
        )
        torch.distributed.send(storage, dst=dst, group=group)

    @classmethod
    def from_remote_init(
        cls: T,
        src: int,
        group: "ProcessGroup" | None = None,  # noqa: F821
        device: torch.device | None = None,
        use_broadcast: bool = False,
    ) -> Self:
        """Receives a tensordict that :meth:`~.init_remote` sends, and returns it.

        The result is consolidated (see :meth:`~.consolidate`): receive the next
        batches into it with ``recv(consolidated=True)``, which allocates no new
        memory. A tensorclass arrives as the same tensorclass, if this process can
        import its class. The metadata is unpickled: use it only with ranks you
        trust. See :meth:`~.init_remote` for an example, and :ref:`distributed`.

        Args:
            src (int): the rank of the sender.
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            device (torch.device, str or int, optional): the device of the result.
                Point-to-point, it is also the device that the metadata goes
                through, as in :func:`torch.distributed.recv_object_list`.
                Defaults to ``None``, which is the CPU point-to-point, and the
                device of the sender's storage with ``use_broadcast=True``.
            use_broadcast (bool, optional): if ``True``, receives with
                :meth:`~.broadcast`. It must match the sender's value. Defaults to
                ``False``.

        Returns:
            A new consolidated tensordict, of the class that was sent.
        """
        if use_broadcast:
            return cls({}).broadcast(src=src, group=group, device=device)

        from tensordict._reductions import _rebuild_tensordict_files_consolidated

        meta = [None]
        torch.distributed.recv_object_list(
            meta,
            src=src,
            group=group,
            device=device,
        )
        metadata = meta[0]
        total_bytes = metadata.pop("_total_bytes")
        tc_type_str = metadata.pop("_tensorclass_type", None)
        storage = torch.empty(total_bytes, dtype=torch.uint8, device=device or "cpu")
        torch.distributed.recv(storage, src=src, group=group)
        result = _rebuild_tensordict_files_consolidated(metadata, storage)
        if tc_type_str is not None:
            tc_cls = _resolve_tensorclass_type(tc_type_str)
            # Like a load: the data comes from another process
            result = tc_cls._from_tensordict(result, rebuild=False)
        return result

    def isend(
        self,
        dst: int | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,  # noqa: F821
        group_dst: int | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        return_early: bool = False,
        consolidated: bool = False,
    ) -> int | List["Work"] | None:  # noqa: D417, F821
        """Sends the leaves of the tensordict to another rank, one asynchronous transfer per leaf.

        As :meth:`~.send`, but the transfers of all the leaves run at the same
        time. By default, the call waits until they are done. With
        ``return_early=True``, it returns at once: call ``wait()`` on each
        returned ``Work`` object before you change the tensordict. The receiver
        calls :meth:`~.irecv` or :meth:`~.recv`. See :ref:`distributed`.

        Args:
            dst (int, optional): the global rank of the receiver. Exclusive with
                ``group_dst``: pass exactly one of them. Defaults to ``None``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            group_dst (int, optional): the rank of the receiver in ``group``,
                which must be given. Exclusive with ``dst``. As in :meth:`~.send`,
                ``group`` can then be a standalone process group. Defaults to
                ``None``.
            init_tag (int, optional): the tag of the first leaf. The next leaves
                use ``init_tag + 1``, ``init_tag + 2``, and so on. The receiver must
                pass the same value. Defaults to ``0``.
            pseudo_rand (bool, optional): if ``True``, the tags are pseudo-random
                numbers seeded by ``init_tag``. The receiver must pass the same
                value. Defaults to ``False``.
            return_early (bool, optional): if ``True``, returns the list of
                ``Work`` objects of the transfers without waiting for them.
                Defaults to ``False``.
            consolidated (bool, optional): if ``True``, sends all the leaves as one
                message, as ``send(consolidated=True)`` does. A tensordict that is
                not consolidated is consolidated on each call, which copies every
                leaf: keep a consolidated copy and refresh it with
                :meth:`~.update_` to avoid the copy. The receiver must call
                ``irecv(consolidated=True)`` or ``recv(consolidated=True)``.
                ``init_tag`` and ``pseudo_rand`` are ignored. Defaults to ``False``.

        Returns:
            With ``return_early=True``, a list of ``Work`` objects. Otherwise, the
            tag of the last leaf sent, or ``None`` with ``consolidated=True``.

        Raises:
            ValueError: if both or none of ``dst`` and ``group_dst`` are given, or
                if ``group_dst`` is given without ``group``.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def isend_irecv(rank, world_size):
                    if rank == 0:
                        td = TensorDict(a=torch.ones(4), b=torch.ones(4, 3), batch_size=[4])
                        works = td.isend(dst=1, return_early=True)
                        # ... other work, which does not change td ...
                        for work in works:
                            work.wait()
                    elif rank == 1:
                        td = TensorDict(a=torch.zeros(4), b=torch.zeros(4, 3), batch_size=[4])
                        works = td.irecv(src=0, return_premature=True)
                        # ... other work, which does not read td ...
                        for work in works:
                            work.wait()
                        print(td["b"].sum())  # tensor(12.)
        """
        _check_p2p_peer(dst, group_dst, group, "dst", "group_dst")
        if consolidated:
            from torch import distributed as dist

            td_c = self if self.is_consolidated() else self.consolidate(metadata=True)
            storage = td_c._consolidated["storage"]
            if group_dst is not None:
                work = group.send([storage], group_dst, 0)
            else:
                work = dist.isend(storage, dst=dst, group=group)
            if return_early:
                return [work]
            work.wait()
            return
        return self._isend(
            dst,
            _tag=init_tag - 1,
            pseudo_rand=pseudo_rand,
            group=group,
            group_dst=group_dst,
            return_early=return_early,
        )

    def _isend(
        self,
        dst: int | None,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
        return_early: bool = False,
        group_dst: int | None = None,
    ) -> int:
        from torch import distributed as dist

        root = False
        if _futures is None:
            root = True
            _futures = []
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _tag = value._isend(
                    dst,
                    _tag=_tag,
                    pseudo_rand=pseudo_rand,
                    _futures=_futures,
                    group=group,
                    return_early=return_early,
                    group_dst=group_dst,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = _int_generator(_tag + 1)
            if group_dst is not None:
                # Direct backend call: works for raw (unregistered) groups,
                # which the functional API rejects.
                _future = group.send([value], group_dst, _tag)
            else:
                _future = dist.isend(value, dst=dst, tag=_tag, group=group)
            _futures.append(_future)
        if root and not return_early:
            for _future in _futures:
                _future.wait()
        elif root and return_early:
            return _futures
        return _tag

    def irecv(
        self,
        src: int | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_src: int | None = None,
        return_premature: bool = False,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        consolidated: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        """Receives the leaves that :meth:`~.isend` or :meth:`~.send` sends, one asynchronous transfer per leaf.

        The tensordict must have the keys, shapes and dtypes of the tensordict
        that the sender sends; the values are written into its tensors in place.
        By default, the call waits until all the transfers are done. With
        ``return_premature=True``, it returns at once: call ``wait()`` on each
        returned ``Work`` object before you read the tensordict. See
        :meth:`~.isend` for an example, and :ref:`distributed`.

        Args:
            src (int, optional): the global rank of the sender. Exclusive with
                ``group_src``: pass exactly one of them. Defaults to ``None``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            group_src (int, optional): the rank of the sender in ``group``, which
                must be given. Exclusive with ``src``. As in :meth:`~.recv`,
                ``group`` can then be a standalone process group. Defaults to
                ``None``.
            return_premature (bool, optional): if ``True``, returns the list of
                ``Work`` objects of the transfers without waiting for them.
                Defaults to ``False``.
            init_tag (int, optional): the ``init_tag`` that the sender passed.
                Defaults to ``0``.
            pseudo_rand (bool, optional): the ``pseudo_rand`` that the sender
                passed. Defaults to ``False``.
            consolidated (bool, optional): if ``True``, receives one message into
                the consolidated storage of the tensordict, as
                ``recv(consolidated=True)`` does. The tensordict must be
                consolidated already. ``init_tag`` and ``pseudo_rand`` are ignored.
                Defaults to ``False``.

        Returns:
            With ``return_premature=True``, a list of ``Work`` objects. Otherwise,
            ``None``.

        Raises:
            ValueError: if both or none of ``src`` and ``group_src`` are given, or
                if ``group_src`` is given without ``group``.
        """
        _check_p2p_peer(src, group_src, group, "src", "group_src")
        if consolidated:
            from torch import distributed as dist

            storage = self._consolidated["storage"]
            if group_src is not None:
                work = group.recv([storage], group_src, 0)
            else:
                work = dist.irecv(storage, src=src, group=group)
            if return_premature:
                return [work]
            work.wait()
            return
        return self._irecv(
            src,
            return_premature=return_premature,
            _tag=init_tag - 1,
            pseudo_rand=pseudo_rand,
            group=group,
            group_src=group_src,
        )

    def _irecv(
        self,
        src: int | None,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_src: int | None = None,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        from torch import distributed as dist

        root = False
        if _future_list is None:
            _future_list = []
            root = True

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _tag, _future_list = value._irecv(
                    src,
                    _tag=_tag,
                    _future_list=_future_list,
                    pseudo_rand=pseudo_rand,
                    group=group,
                    group_src=group_src,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = _int_generator(_tag + 1)
            if group_src is not None:
                # Direct backend call: works for raw (unregistered) groups,
                # which the functional API rejects.
                _future_list.append(group.recv([value], group_src, _tag))
            else:
                _future_list.append(dist.irecv(value, src=src, tag=_tag, group=group))
        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    def reduce(
        self,
        dst,
        op=None,
        async_op=False,
        return_premature=False,
        group=None,
    ) -> None:
        """Reduces each leaf over all the ranks of the group, into the tensordict of rank ``dst``.

        Every rank of the group must call it, with a tensordict of the same
        structure. Each leaf is reduced in place with
        :func:`torch.distributed.reduce`. After the call, rank ``dst`` holds the
        result. On the other ranks, the values of the tensordict are not defined
        (with ``gloo``, they can hold partial results): clone the tensordict
        before the call if you need them. A lazy stack reduces each stacked
        leaf, like a dense tensordict of the same keys and shapes, and rank
        ``dst`` writes the result back into its members: after the last
        reduction, or with ``async_op=True``, after each reduction completes.
        That write raises if a member leaf cannot be written in place, such as
        an expanded tensor, and the call waits for every reduction before it
        raises. See :ref:`distributed-collectives`.

        Args:
            dst (int): the rank that receives the result.
            op (torch.distributed.ReduceOp, optional): the reduction. Defaults to
                ``None``, which is ``torch.distributed.ReduceOp.SUM``.
            async_op (bool, optional): if ``True``, the reductions of all the
                leaves run at the same time, and the call waits for them unless
                ``return_premature=True``. Defaults to ``False``.
            return_premature (bool, optional): with ``async_op=True``, if
                ``True``, returns the list of ``Work`` objects of the reductions
                without waiting for them: call ``wait()`` on each one before you use
                the tensordict, even if an earlier one raises. On rank ``dst``,
                the leaves of a lazy stack give a
                :class:`torch.futures.Future` instead, which also writes the
                result back. Defaults to ``False``.
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.

        Returns:
            With ``async_op=True`` and ``return_premature=True``, a list of
            ``Work`` objects. Otherwise, ``None``.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def reduce(rank, world_size):
                    td = TensorDict(a=torch.ones(3) * (rank + 1), batch_size=[3])
                    td.reduce(dst=0, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        print(td["a"])  # with 2 ranks: tensor([3., 3., 3.])
        """
        from torch import distributed as dist

        if op is None:
            op = dist.ReduceOp.SUM
        return self._reduce(dst, op, async_op, return_premature, group=group)

    def _reduce(
        self,
        dst,
        op=None,
        async_op=False,
        return_premature=False,
        _future_list=None,
        group=None,
    ):
        """Issues the reductions of :meth:`reduce`, one per leaf, in ``sorted_keys`` order.

        The root call (``_future_list=None``) creates the list that nested calls
        append to: the results of :func:`torch.distributed.reduce` (``Work``
        objects with ``async_op=True``, ``None`` otherwise). Every rank must
        issue the collectives of a dense tensordict of the same keys and shapes,
        in the same order. An override whose leaves are copies writes the
        result back on rank ``dst``: in a callback of the future of the ``Work``
        with ``async_op=True``, and otherwise by appending a callable that the
        root runs after the last collective. Returns the list for a nested call
        or with ``return_premature=True``. Otherwise the root waits for every
        entry with :meth:`_finish_reduce` and returns ``None``.
        """
        from torch import distributed as dist

        if op is None:
            op = dist.ReduceOp.SUM
        root = False
        if _future_list is None:
            _future_list = []
            root = True
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(type(value)):
                _future_list = value._reduce(
                    dst=dst,
                    op=op,
                    async_op=async_op,
                    _future_list=_future_list,
                    group=group,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            _future_list.append(
                dist.reduce(value, dst=dst, op=op, async_op=async_op, group=group)
            )
        if not root:
            return _future_list
        elif async_op and return_premature:
            return _future_list
        self._finish_reduce(_future_list, async_op)

    @staticmethod
    def _finish_reduce(future_list: list, async_op: bool) -> None:
        """Waits for every reduction (``async_op``) or runs every pending write-back, then raises the first error.

        A lazy stack on rank dst writes its results back in a callback of the
        reduction (``async_op``), or leaves the write-back in the list (sync
        mode). Stopping at the first error would leave reductions still writing
        into the tensordict, or members not written back.
        """
        error = None
        for item in future_list:
            try:
                if async_op:
                    item.wait()
                elif item is not None:
                    item()
            except Exception as err:
                if error is None:
                    error = err
        if error is not None:
            raise error

    def broadcast(
        self,
        src: int,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        device: torch.device | str | None = None,
        _tensorclass_type: str | None = None,
    ) -> Self:
        """Sends the tensordict of rank ``src`` to every rank of the group, and returns it on each rank.

        Every rank of the group must call it. The other ranks do not need to know
        the structure: they can call it on an empty tensordict. Rank ``src``
        consolidates its tensordict (see :meth:`~.consolidate`), and broadcasts the
        metadata with :func:`torch.distributed.broadcast_object_list`, which
        pickles it, then the storage as one tensor. The other ranks build a new
        tensordict from them. Use it only with ranks you trust. See
        :ref:`distributed-collectives`.

        Args:
            src (int): the rank that sends its tensordict.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            device (torch.device, str or int, optional): on rank ``src``, the
                device of the consolidated storage, if the tensordict is not
                consolidated yet. On the other ranks, the device of the result.
                With ``nccl``, pass a CUDA device. Defaults to ``None``, which keeps
                the device of the tensordict on rank ``src``, and uses the same
                device on the other ranks.

        Returns:
            The consolidated tensordict of rank ``src``. On rank ``src``, it is
            the tensordict itself if it is consolidated already, and a
            consolidated copy otherwise. On the other ranks, it is a new
            tensordict.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def broadcast(rank, world_size):
                    td = TensorDict(a=torch.arange(3.0), batch_size=[3]) if rank == 0 else TensorDict()
                    td = td.broadcast(src=0)
                    print(rank, td["a"])  # tensor([0., 1., 2.]) on every rank
        """
        from tensordict._reductions import _rebuild_tensordict_files_consolidated
        from torch import distributed as dist

        rank = dist.get_rank(group=group)
        if rank == src:
            kwargs = {"metadata": True}
            if device is not None:
                kwargs["device"] = torch.device(device)
            td_c = self if self.is_consolidated() else self.consolidate(**kwargs)
            storage = td_c._consolidated["storage"]
            metadata = td_c._consolidated["metadata"]
            metadata["_total_bytes"] = storage.numel()
            metadata["_storage_device"] = str(storage.device)
            if _tensorclass_type is not None:
                metadata["_tensorclass_type"] = _tensorclass_type
            dist.broadcast_object_list([metadata], src=src, group=group)
            dist.broadcast(storage, src=src, group=group)
            return td_c
        else:
            meta = [None]
            dist.broadcast_object_list(meta, src=src, group=group)
            metadata = meta[0]
            total_bytes = metadata.pop("_total_bytes")
            storage_device = metadata.pop("_storage_device")
            tc_type_str = metadata.pop("_tensorclass_type", None)
            recv_device = device if device is not None else torch.device(storage_device)
            storage = torch.empty(total_bytes, dtype=torch.uint8, device=recv_device)
            dist.broadcast(storage, src=src, group=group)
            result = _rebuild_tensordict_files_consolidated(metadata, storage)
            if tc_type_str is not None:
                tc_cls = _resolve_tensorclass_type(tc_type_str)
                # Like a load: the data comes from another process
                result = tc_cls._from_tensordict(result, rebuild=False)
            return result

    def all_reduce(
        self,
        op=None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        async_op: bool = False,
    ) -> None:
        """Reduces each leaf over all the ranks of the group, in place, so that every rank holds the result.

        Every rank of the group must call it, with a tensordict of the same
        structure. Each leaf is reduced with :func:`torch.distributed.all_reduce`,
        for example to sum or average gradients. See
        :ref:`distributed-collectives`.

        Args:
            op (torch.distributed.ReduceOp, optional): the reduction. Defaults to
                ``None``, which is ``torch.distributed.ReduceOp.SUM``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.
            async_op (bool, optional): if ``True``, returns the list of ``Work``
                objects of the reductions without waiting for them: call ``wait()``
                on each one before you use the tensordict. Defaults to ``False``.

        Returns:
            With ``async_op=True``, a list of ``Work`` objects. Otherwise, ``None``.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def all_reduce(rank, world_size):
                    td = TensorDict(a=torch.ones(3) * rank, batch_size=[3])
                    td.all_reduce(op=dist.ReduceOp.SUM)
                    print(rank, td["a"])  # with 2 ranks: tensor([1., 1., 1.]) on every rank
        """
        from torch import distributed as dist

        if op is None:
            op = dist.ReduceOp.SUM
        futures = []

        def _all_reduce(value):
            futures.append(
                dist.all_reduce(value, op=op, group=group, async_op=async_op)
            )

        self._fast_apply(
            _all_reduce,
            is_leaf=_NESTED_TENSORS_AS_LISTS,
        )
        if async_op:
            return futures
        return

    def all_gather(
        self,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
    ) -> list:
        """Returns a list with the tensordict of each rank of the group, on every rank.

        Every rank of the group must call it. The tensordicts can have different
        structures. Each rank consolidates its tensordict (see
        :meth:`~.consolidate`), the metadata is gathered with
        :func:`torch.distributed.all_gather_object`, which pickles it, and the
        storages with :func:`torch.distributed.all_gather`. Use it only with ranks
        you trust. See :ref:`distributed-collectives`.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): the process group.
                Defaults to ``None``, the default process group.

        Returns:
            A list of new consolidated tensordicts, one per rank, in the order of
            the ranks.

        Examples:
            This example runs in two processes: add it to the
            :ref:`runner script of the distributed guide <distributed>` and
            pass it to ``run``.

            .. code-block:: python

                def all_gather(rank, world_size):
                    local = TensorDict(a=torch.full((3,), float(rank)), batch_size=[3])
                    tds = local.all_gather()
                    print(rank, [td["a"][0].item() for td in tds])  # [0.0, 1.0] on every rank
        """
        from tensordict._reductions import _rebuild_tensordict_files_consolidated
        from torch import distributed as dist

        world_size = dist.get_world_size(group=group)
        td_c = self if self.is_consolidated() else self.consolidate(metadata=True)
        storage = td_c._consolidated["storage"]
        metadata = td_c._consolidated["metadata"]
        metadata["_total_bytes"] = storage.numel()

        all_meta = [None] * world_size
        dist.all_gather_object(all_meta, metadata, group=group)

        all_sizes = [m["_total_bytes"] for m in all_meta]
        max_size = max(all_sizes)
        padded = torch.zeros(max_size, dtype=torch.uint8, device=storage.device)
        padded[: storage.numel()] = storage
        gathered = [
            torch.empty(max_size, dtype=torch.uint8, device=storage.device)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, padded, group=group)

        result = []
        for i in range(world_size):
            m = all_meta[i]
            sz = m.pop("_total_bytes")
            result.append(_rebuild_tensordict_files_consolidated(m, gathered[i][:sz]))
        return result
