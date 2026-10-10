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

from typing import Any, List, TYPE_CHECKING

import torch
from tensordict.base import (
    _check_p2p_peer,
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _resolve_tensorclass_type,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import _int_generator
from torch import Tensor

if TYPE_CHECKING:
    from tensordict._ucxx import TensorDictPipe
    from tensordict.base import TensorDictBase


class _Distributed:
    """Point-to-point and collective communication with torch.distributed."""

    # Distributed functionality
    def gather_and_stack(
        self, dst: int, group: "torch.distributed.ProcessGroup" | None = None
    ) -> Self | None:
        """Gathers tensordicts from various workers and stacks them onto self in the destination worker.

        Args:
            dst (int): the rank of the destination worker where :func:`gather_and_stack` will be called.
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from tensordict import TensorDict
            >>> import torch
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Create a single tensordict to be sent to server
            ...     td = TensorDict(
            ...         {("a", "b"): torch.randn(2),
            ...          "c": torch.randn(2)}, [2]
            ...     )
            ...     td.gather_and_stack(0)
            ...
            >>> def server():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Creates the destination tensordict on server.
            ...     # The first dim must be equal to world_size-1
            ...     td = TensorDict(
            ...         {("a", "b"): torch.zeros(2),
            ...          "c": torch.zeros(2)}, [2]
            ...     ).expand(1, 2).contiguous()
            ...     td.gather_and_stack(0)
            ...     assert td["a", "b"] != 0
            ...     print("yuppie")
            ...
            >>> if __name__ == "__main__":
            ...     mp.set_start_method("spawn")
            ...
            ...     main_worker = mp.Process(target=server)
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...
            ...     main_worker.join()
            ...     secondary_worker.join()
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
        dst: int | TensorDictPipe | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_dst: int | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        consolidated: bool = False,
    ) -> None:  # noqa: D417
        """Sends the content of a tensordict to a distant worker.

        Args:
            dst (int or TensorDictPipe, optional): the global rank of the
                destination worker where the content should be sent, or a
                :class:`~tensordict._ucxx.TensorDictPipe` for UCXX-based
                transport. Mutually exclusive with ``group_dst``; exactly one
                of the two must be provided.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
            group_dst (int, optional): the rank of the destination worker
                *relative to* ``group``. Requires ``group`` to be passed and is
                mutually exclusive with ``dst``. When set, the p2p calls are
                issued directly on the group's backend, so ``group`` may be a
                standalone :class:`~torch.distributed.ProcessGroup` built
                against a store (never registered through
                :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`).
                Defaults to ``None``.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.
            consolidated (bool): if True, sends the consolidated storage as a
                single tensor (1 message). The tensordict is consolidated first
                if needed. The receiver must use ``recv(consolidated=True)`` and
                must already hold a consolidated tensordict with matching schema.
                Defaults to ``False``.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from tensordict import TensorDict
            >>> import torch
            >>>
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.send(0)
            ...
            >>>
            >>> def server(queue):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.recv(1)
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__=="__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(target=server, args=(queue,))
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

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
        src: int | TensorDictPipe | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        group_src: int | None = None,
        init_tag: int = 0,
        pseudo_rand: bool = False,
        consolidated: bool = False,
    ) -> int:  # noqa: D417
        """Receives the content of a tensordict and updates content with it.

        Check the example in the `send` method for context.

        Args:
            src (int or TensorDictPipe, optional): the global rank of the
                source worker, or a
                :class:`~tensordict._ucxx.TensorDictPipe` for UCXX-based
                transport.  When a pipe is passed, the data is received
                in-place into this tensordict's consolidated storage (if
                available). Mutually exclusive with ``group_src``; exactly one
                of the two must be provided.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
            group_src (int, optional): the rank of the source worker *relative
                to* ``group``. Requires ``group`` to be passed and is mutually
                exclusive with ``src``. When set, the p2p calls are issued
                directly on the group's backend, so ``group`` may be a
                standalone :class:`~torch.distributed.ProcessGroup` built
                against a store (never registered through
                :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`).
                Defaults to ``None``.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`send`.
                Defaults to ``False``.
            consolidated (bool): if True, receives a single consolidated storage
                tensor directly into ``self._consolidated["storage"]``. The
                tensordict must already be consolidated (e.g. from a prior
                :meth:`~.from_remote_init` call). All leaf tensor views update
                in-place automatically.
                Defaults to ``False``.
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

    async def asend(self, dst: TensorDictPipe) -> None:
        """Sends the content of a tensordict through a UCXX pipe (async).

        Args:
            dst (TensorDictPipe): the pipe to send through.

        .. seealso:: :meth:`send` for the synchronous variant and
            torch.distributed-based transport.
        """
        await dst.asend(self)

    async def arecv(
        self,
        src: TensorDictPipe,
        *,
        device: torch.device | str | None = None,
    ) -> "TensorDictBase":
        """Receives content into this tensordict from a UCXX pipe (async).

        Args:
            src (TensorDictPipe): the pipe to receive from.

        Keyword Args:
            device: device for storage allocation on first receive.

        Returns:
            The received TensorDict (may be ``self`` for in-place updates).

        .. seealso:: :meth:`recv` for the synchronous variant and
            torch.distributed-based transport.
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
        """Initializes a remote tensordict by sending its metadata and content.

        Two transport modes are available:

        - **Point-to-point** (default): uses ``send_object_list`` +
          ``dist.send`` to transfer metadata and storage to a single
          destination rank. Only the sender and receiver need to call the
          pair ``init_remote`` / ``from_remote_init``.
        - **Broadcast** (``use_broadcast=True``): delegates to
          :meth:`~.broadcast`.  Metadata and storage are broadcast to
          *every* rank in the group.  All ranks must participate (receivers
          call ``from_remote_init(src=..., use_broadcast=True)``).  ``dst``
          is ignored in this mode.

        Args:
            dst (int, optional): The rank of the destination process. Required
                when ``use_broadcast=False`` (the default).  Ignored when
                ``use_broadcast=True``.
            group ("ProcessGroup", optional): The process group to use for communication. Defaults to None.
            device (torch.device, optional): The device to use for tensor operations. Defaults to None.
            use_broadcast (bool): If ``True``, use :meth:`~.broadcast` instead
                of point-to-point send.  Defaults to ``False``.

        .. seealso::
            The receiving process should call `~.from_remote_init` or an equivalent method to receive and initialize a new tensordict based on the sent metadata.

        Examples:
            >>> import os
            >>> import torch
            >>> import torch.distributed as dist
            >>> from tensordict import TensorDict, MemoryMappedTensor
            >>> import multiprocessing as mp
            >>>
            >>> def server(queue):
            ...     # Set environment variables for distributed communication
            ...     os.environ["MASTER_ADDR"] = "localhost"
            ...     os.environ["MASTER_PORT"] = "29505"
            ...
            ...     # Initialize the distributed backend
            ...     dist.init_process_group("gloo", rank=0, world_size=2)
            ...
            ...     # Create a sample tensordict
            ...     td = (
            ...         TensorDict(
            ...             {
            ...                 ("a", "b"): torch.ones(2),
            ...                 "c": torch.ones(2),
            ...                 ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.ones(2, 2)),
            ...             },
            ...             [2],
            ...         )
            ...         .expand(1, 2)
            ...         .contiguous()
            ...     )
            ...
            ...     # Send the tensordict metadata and content to the client
            ...     td.init_remote(dst=1)
            ...
            >>> def client(queue):
            ...     # Set environment variables for distributed communication
            ...     os.environ["MASTER_ADDR"] = "localhost"
            ...     os.environ["MASTER_PORT"] = "29505"
            ...
            ...     # Initialize the distributed backend
            ...     dist.init_process_group("gloo", rank=1, world_size=2)
            ...
            ...     # Receive the tensordict metadata and content from the server
            ...     received_td = TensorDict.from_remote_init(src=0)
            ...
            ...     # Verify that the received tensordict matches the expected structure and values
            ...     assert set(received_td.keys()) == {"a", "c", "d"}
            ...     assert (received_td == 1).all()
            ...
            ...     # Signal that the test has completed successfully
            ...     queue.put("yuppie")
            >>>
            >>> if __name__ == "__main__":
            ...     queue = mp.Queue(1)
            ...
            ...     # Create and start the server and client processes
            ...     main_worker = mp.Process(target=server, args=(queue,))
            ...     secondary_worker = mp.Process(target=client, args=(queue,))
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...
            ...     try:
            ...         out = queue.get(timeout=10)  # Wait for the signal with a timeout
            ...         print(out)  # Should print "yuppie"
            ...     finally:
            ...         queue.close()
            ...         main_worker.join(timeout=10)
            ...         secondary_worker.join(timeout=10)
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
        cls,
        src: int,
        group: "torch.distributed.ProcessGroup" | None = None,
        device: torch.device | None = None,
        use_broadcast: bool = False,
    ) -> Self:
        """Creates a new tensordict instance initialized from remotely sent metadata.

        This class method receives consolidated metadata and a single storage buffer
        sent by :meth:`~.init_remote`, then reconstructs the full tensordict.

        Two transport modes are available (must match the sender's choice):

        - **Point-to-point** (default): uses ``recv_object_list`` +
          ``dist.recv``.  Only sender and receiver participate.
        - **Broadcast** (``use_broadcast=True``): delegates to
          :meth:`~.broadcast`.  All ranks must participate.

        Args:
            src (int): The rank of the source process that sent the metadata.
            group ("ProcessGroup", optional): The process group to use for communication. Defaults to None.
            device (torch.device, optional): The device to use for tensor operations. Defaults to None.
            use_broadcast (bool): If ``True``, use :meth:`~.broadcast` instead
                of point-to-point recv.  Must match the sender's setting.
                Defaults to ``False``.

        Returns:
            TensorDict: A new tensordict instance initialized with the received metadata and content.

        .. seealso::
            The sending process should have called `~.init_remote` to send the metadata and content.
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
            result = tc_cls._from_tensordict(result)
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
    ) -> int | List["Work"]:  # noqa: D417, F821
        """Sends the content of the tensordict asynchronously.

        Args:
            dst (int, optional): the global rank of the destination worker
                where the content should be sent. Mutually exclusive with
                ``group_dst``; exactly one of the two must be provided.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
            group_dst (int, optional): the rank of the destination worker
                *relative to* ``group``. Requires ``group`` to be passed and is
                mutually exclusive with ``dst``. When set, the p2p calls are
                issued directly on the group's backend, so ``group`` may be a
                standalone :class:`~torch.distributed.ProcessGroup` built
                against a store (never registered through
                :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`).
                Defaults to ``None``.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.
            return_early (bool, optional): if True, a list of futures
                will be returned instead of the tag of the last tensor sent.
                Defaults to ``False``.

        Example:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torch import multiprocessing as mp
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.isend(0)
            ...
            >>>
            >>> def server(queue, return_premature=True):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     out = td.irecv(1, return_premature=return_premature)
            ...     if return_premature:
            ...         for fut in out:
            ...             fut.wait()
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__ == "__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(
            ...         target=server,
            ...         args=(queue, )
            ...         )
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

        """
        _check_p2p_peer(dst, group_dst, group, "dst", "group_dst")
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
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        """Receives the content of a tensordict and updates content with it asynchronously.

        Check the example in the :meth:`~.isend` method for context.

        Args:
            src (int, optional): the global rank of the source worker. Mutually
                exclusive with ``group_src``; exactly one of the two must be
                provided.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): if set, the specified process group
                will be used for communication. Otherwise, the default process group
                will be used.
                Defaults to ``None``.
            group_src (int, optional): the rank of the source worker *relative
                to* ``group``. Requires ``group`` to be passed and is mutually
                exclusive with ``src``. When set, the p2p calls are issued
                directly on the group's backend, so ``group`` may be a
                standalone :class:`~torch.distributed.ProcessGroup` built
                against a store (never registered through
                :func:`~torch.distributed.init_process_group` or
                :func:`~torch.distributed.new_group`).
                Defaults to ``None``.
            return_premature (bool): if ``True``, returns a list of futures to wait
                upon until the tensordict is updated. Defaults to ``False``,
                i.e. waits until update is completed withing the call.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`isend`.
                Defaults to ``False``.

        Returns:
            if ``return_premature=True``, a list of futures to wait
                upon until the tensordict is updated.
        """
        _check_p2p_peer(src, group_src, group, "src", "group_src")
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
        op: Any | None = None,
        async_op: bool = False,
        return_premature: bool = False,
        group: Any | None = None,
    ) -> Any:
        """Reduces the tensordict across all machines.

        Only the process with ``rank`` dst is going to receive the final result.

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
        elif async_op:
            for future in _future_list:
                future.wait()
            return

    def broadcast(
        self,
        src: int,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        device: torch.device | str | None = None,
        _tensorclass_type: str | None = None,
    ) -> Self:
        """Broadcasts a tensordict from ``src`` to all ranks.

        Uses consolidated transport: the source rank consolidates the tensordict
        and broadcasts metadata (via ``broadcast_object_list``) followed by the
        contiguous storage tensor (via ``dist.broadcast``). Receiving ranks
        allocate a buffer and reconstruct the full tensordict.

        Args:
            src (int): The rank of the source process.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): The process group
                to use. Defaults to ``None`` (default group).
            device (torch.device or str, optional): The device on which to
                allocate the receive buffer.  If ``None``, the device is
                inferred from the source's consolidated storage (transmitted
                via metadata).  Useful when the backend is ``nccl`` and
                buffers must live on CUDA.  Defaults to ``None``.

        Returns:
            TensorDict: the broadcast tensordict on all ranks (consolidated).
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
                result = tc_cls._from_tensordict(result)
            return result

    def all_reduce(
        self,
        op: Any | None = None,
        *,
        group: "torch.distributed.ProcessGroup" | None = None,
        async_op: bool = False,
    ) -> None:
        """All-reduces the tensordict across all ranks in-place.

        Each leaf tensor is reduced individually via ``dist.all_reduce``.
        After the call, every rank holds the reduced values.

        Args:
            op (dist.ReduceOp, optional): The reduce operation (e.g.
                ``ReduceOp.SUM``). Defaults to ``ReduceOp.SUM``.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): The process group
                to use. Defaults to ``None`` (default group).
            async_op (bool): if ``True``, returns a list of futures.
                Defaults to ``False``.

        Returns:
            None, or a list of futures if ``async_op=True``.
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
        """All-gathers tensordicts from every rank.

        Each rank consolidates its tensordict, then metadata is gathered via
        ``all_gather_object`` and storage via ``dist.all_gather``. Returns a
        list of tensordicts, one per rank.

        Keyword Args:
            group (torch.distributed.ProcessGroup, optional): The process group
                to use. Defaults to ``None`` (default group).

        Returns:
            list[TensorDict]: A list of tensordicts from all ranks.
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
