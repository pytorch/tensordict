from __future__ import annotations

import warnings

from typing import Sequence

import torch

from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import _is_tensor_collection, CompatibleType, T, TensorDictBase
from tensordict.utils import _check_keys, _shape, DeviceType


def pad(tensordict: T, pad_size: Sequence[int], value: float = 0.0) -> T:
    """Pads all tensors in a tensordict along the batch dimensions with a constant value, returning a new tensordict.

    Args:
         tensordict (TensorDict): The tensordict to pad
         pad_size (Sequence[int]): The padding size by which to pad some batch
            dimensions of the tensordict, starting from the first dimension and
            moving forward. [len(pad_size) / 2] dimensions of the batch size will
            be padded. For example to pad only the first dimension, pad has the form
            (padding_left, padding_right). To pad two dimensions,
            (padding_left, padding_right, padding_top, padding_bottom) and so on.
            pad_size must be even and less than or equal to twice the number of batch dimensions.
         value (float, optional): The fill value to pad by, default 0.0

    Returns:
        A new TensorDict padded along the batch dimensions

    Examples:
        >>> from tensordict import TensorDict, pad
        >>> import torch
        >>> td = TensorDict({'a': torch.ones(3, 4, 1),
        ...     'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
        >>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        >>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
        >>> print(padded_td.batch_size)
        torch.Size([4, 6])
        >>> print(padded_td.get("a").shape)
        torch.Size([4, 6, 1])
        >>> print(padded_td.get("b").shape)
        torch.Size([4, 6, 1, 1])

    """
    if len(pad_size) > 2 * len(tensordict.batch_size):
        raise RuntimeError(
            "The length of pad_size must be <= 2 * the number of batch dimensions"
        )

    if len(pad_size) % 2:
        raise RuntimeError("pad_size must have an even number of dimensions")

    new_batch_size = list(tensordict.batch_size)
    for i in range(len(pad_size)):
        new_batch_size[i // 2] += pad_size[i]

    reverse_pad = pad_size[::-1]
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    out = TensorDict(
        {}, torch.Size(new_batch_size), device=tensordict.device, _run_checks=False
    )
    for key, tensor in tensordict.items():
        cur_pad = reverse_pad
        if len(pad_size) < len(_shape(tensor)) * 2:
            cur_pad = [0] * (len(_shape(tensor)) * 2 - len(pad_size)) + reverse_pad

        if _is_tensor_collection(tensor.__class__):
            padded = pad(tensor, pad_size, value)
        else:
            padded = torch.nn.functional.pad(tensor, cur_pad, value=value)
        out.set(key, padded)

    return out


def pad_sequence(
    list_of_tensordicts: Sequence[T],
    batch_first: bool | None = None,
    pad_dim: int = 0,
    padding_value: float = 0.0,
    out: T | None = None,
    device: DeviceType | None = None,
    return_mask: bool | None = False,
) -> T:
    """Pads a list of tensordicts in order for them to be stacked together in a contiguous format.

    Args:
        list_of_tensordicts (List[TensorDictBase]): the list of instances to pad and stack.
        pad_dim (int, optional): the ``pad_dim`` indicates the dimension to pad all the keys in the tensordict.
            Defaults to ``0``.
        padding_value (number, optional): the padding value. Defaults to ``0.0``.
        out (TensorDictBase, optional): if provided, the destination where the data will be
            written.
        device (device compatible type, optional): if provded, the device where the
            TensorDict output will be created.
        return_mask (bool, optional): if ``True``, a "masks" entry will be returned.
            It contains a tensordict with the same structure as the stacked tensordict where every entry contains the mask of valid values with size ``torch.Size([stack_len, *new_shape])``,
            where `new_shape[pad_dim] = max_seq_length` and the rest of the `new_shape` matches the previous shape of the contained tensors.

    Examples:
        >>> list_td = [
        ...     TensorDict({"a": torch.zeros((3, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
        ...     TensorDict({"a": torch.zeros((5, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
        ...     ]
        >>> padded_td = pad_sequence(list_td, return_mask=True)
        >>> print(padded_td)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([2, 4, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([2, 5, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                masks: TensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.bool, is_shared=False),
                        b: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)
    """
    if batch_first is not None:
        warnings.warn(
            "The batch_first argument is deprecated and will be removed in a future release. The output will always be batch_first.",
            category=DeprecationWarning,
        )

    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")

    # check that all tensordict match
    update_batch_size = True
    max_seq_length = float("-inf")
    keys = _check_keys(list_of_tensordicts, leaves_only=True, include_nested=True)
    tmp_list_of_tensordicts = []
    for td in list_of_tensordicts:

        if return_mask:
            tmp_list_of_tensordicts.append(td.clone(False))

        for key in keys:
            tensor_shape = td.get(key).shape
            pos_pad_dim = pad_dim if pad_dim >= 0 else len(tensor_shape) + pad_dim

            # track the maximum sequence length to update batch_size accordingly
            if tensor_shape[pos_pad_dim] > max_seq_length:
                max_seq_length = tensor_shape[pos_pad_dim]

            # The mask should always contain the batch_size of the TensorDict
            mask_shape = td.shape

            # if the pad_dim is past the batch_size of the TensorDict, we need to add the new dimension to the mask
            if pos_pad_dim >= td.ndim:
                mask_shape += torch.Size([tensor_shape[pos_pad_dim]])
                update_batch_size = False

            if return_mask:
                tmp_list_of_tensordicts[-1].set(
                    ("masks", key),
                    torch.ones(mask_shape, dtype=torch.bool),
                )
    if return_mask:
        list_of_tensordicts = tmp_list_of_tensordicts

    keys = _check_keys(list_of_tensordicts, leaves_only=True, include_nested=True)

    old_batch_size = list(list_of_tensordicts[0].batch_size)
    if update_batch_size and len(old_batch_size) > 0:
        old_batch_size[pad_dim] = max_seq_length
    shape = [
        len(list_of_tensordicts),
    ] + old_batch_size

    if out is None:
        out = list_of_tensordicts[0].empty(recurse=True).reshape(torch.Size(shape))

    for key in keys:
        try:
            tensor_shape = list_of_tensordicts[0].get(key).shape
            pos_pad_dim = (
                (pad_dim if pad_dim >= 0 else len(tensor_shape) + pad_dim)
                if len(tensor_shape) > 1
                else 0  # handles the case when the masks are 1-dimensional
            )
            out.set(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [
                        td.get(key).transpose(0, pos_pad_dim)
                        for td in list_of_tensordicts
                    ],
                    batch_first=True,
                    padding_value=padding_value,
                ).transpose(1, pos_pad_dim + 1),
                inplace=True,
            )
        except Exception as err:
            raise RuntimeError(f"pad_sequence failed for key {key}") from err
    return out


def merge_tensordicts(*tensordicts: T) -> T:
    """Merges tensordicts together."""
    if len(tensordicts) < 2:
        raise RuntimeError(
            f"at least 2 tensordicts must be provided, got" f" {len(tensordicts)}"
        )
    d = tensordicts[0].to_dict()
    batch_size = tensordicts[0].batch_size
    for td in tensordicts[1:]:
        d.update(td.to_dict())
        if td.batch_dims < len(batch_size):
            batch_size = td.batch_size
    return TensorDict(d, batch_size, device=td.device, _run_checks=False)


def dense_stack_tds(
    td_list: Sequence[TensorDictBase] | LazyStackedTensorDict,
    dim: int = None,
) -> T:
    """Densely stack a list of :class:`~tensordict.TensorDictBase` objects (or a :class:`~tensordict.LazyStackedTensorDict`) given that they have the same structure.

    This function is called with a list of :class:`~tensordict.TensorDictBase` (either passed directly or obtrained from
    a :class:`~tensordict.LazyStackedTensorDict`).
    Instead of calling ``torch.stack(td_list)``, which would return a :class:`~tensordict.LazyStackedTensorDict`,
    this function expands the first element of the input list and stacks the input list onto that element.
    This works only when all the elements of the input list have the same structure.
    The :class:`~tensordict.TensorDictBase` returned will have the same type of the elements of the input list.

    This function is useful when some of the :class:`~tensordict.TensorDictBase` objects that need to be stacked
    are :class:`~tensordict.LazyStackedTensorDict` or have :class:`~tensordict.LazyStackedTensorDict`
    among entries (or nested entries).
    In those cases, calling ``torch.stack(td_list).to_tensordict()`` is infeasible.
    Thus, this function provides an alternative for densely stacking the list provided.

    Args:
        td_list (List of TensorDictBase or LazyStackedTensorDict): the tds to stack.
        dim (int, optional): the dimension to stack them.
            If td_list is a LazyStackedTensorDict, it will be retrieved automatically.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict import dense_stack_tds
        >>> from tensordict.tensordict import assert_allclose_td
        >>> td0 = TensorDict({"a": torch.zeros(3)},[])
        >>> td1 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(2)},[])
        >>> td_lazy = torch.stack([td0, td1], dim=0)
        >>> td_container = TensorDict({"lazy": td_lazy}, [])
        >>> td_container_clone = td_container.clone()
        >>> td_stack = torch.stack([td_container, td_container_clone], dim=0)
        >>> td_stack
        LazyStackedTensorDict(
            fields={
                lazy: LazyStackedTensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 2, -1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2, 2]),
                    device=None,
                    is_shared=False,
                    stack_dim=0)},
            exclusive_fields={
            },
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> td_stack = dense_stack_tds(td_stack) # Automatically use the LazyStackedTensorDict stack_dim
        TensorDict(
            fields={
                lazy: LazyStackedTensorDict(
                    fields={
                        a: Tensor(shape=torch.Size([2, 2, -1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    exclusive_fields={
                        1 ->
                            b: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2, 2]),
                    device=None,
                    is_shared=False,
                    stack_dim=1)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)
        # Note that
        # (1) td_stack is now a TensorDict
        # (2) this has pushed the stack_dim of "lazy" (0 -> 1)
        # (3) this has revealed the exclusive keys.
        >>> assert_allclose_td(td_stack, dense_stack_tds([td_container, td_container_clone], dim=0))
        # This shows it is the same to pass a list or a LazyStackedTensorDict

    """
    if isinstance(td_list, LazyStackedTensorDict):
        dim = td_list.stack_dim
        td_list = td_list.tensordicts
    elif dim is None:
        raise ValueError(
            "If a list of tensordicts is provided, stack_dim must not be None"
        )
    shape = list(td_list[0].shape)
    shape.insert(dim, len(td_list))

    out = td_list[0].unsqueeze(dim).expand(shape).clone()
    return torch.stack(td_list, dim=dim, out=out)


def make_tensordict(
    input_dict: dict[str, CompatibleType] | None = None,
    batch_size: Sequence[int] | torch.Size | int | None = None,
    device: DeviceType | None = None,
    **kwargs: CompatibleType,  # source
) -> TensorDict:
    """Returns a TensorDict created from the keyword arguments or an input dictionary.

    If ``batch_size`` is not specified, returns the maximum batch size possible.

    This function works on nested dictionaries too, or can be used to determine the
    batch-size of a nested tensordict.

    Args:
        input_dict (dictionary, optional): a dictionary to use as a data source
            (nested keys compatible).
        **kwargs (TensorDict or torch.Tensor): keyword arguments as data source
            (incompatible with nested keys).
        batch_size (iterable of int, optional): a batch size for the tensordict.
        device (torch.device or compatible type, optional): a device for the TensorDict.

    Examples:
        >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
        >>> print(make_tensordict(input_dict))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # alternatively
        >>> td = make_tensordict(**input_dict)
        >>> # nested dict: the nested TensorDict can have a different batch-size
        >>> # as long as its leading dims match.
        >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
        >>> print(make_tensordict(input_dict))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # we can also use this to work out the batch sie of a tensordict
        >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
        >>> print(make_tensordict(input_td))
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                b: TensorDict(
                    fields={
                        c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3, 4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
    """
    if input_dict is not None:
        kwargs.update(input_dict)
    return TensorDict.from_dict(kwargs, batch_size=batch_size, device=device)
