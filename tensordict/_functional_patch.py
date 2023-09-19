from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from tensordict import TensorDictBase
from torch import Tensor
from torch._functorch.utils import exposed_in
from torch.nn.utils._named_member_accessor import _MISSING

from torch.func import functional_call
from functorch import dim as funcdim

__all__ = []


@exposed_in("torch.func")
def functional_call_patched(
    module: "torch.nn.Module",
    parameter_and_buffer_dicts: Union[Dict[str, Tensor], Sequence[Dict[str, Tensor]]],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    if isinstance(parameter_and_buffer_dicts, TensorDictBase):
        # Ideally, we would like to make the whole stack compatible with tensordict,
        # from functional_call to NamedMemberAccessor.
        # Withing tensordict library, this would an awful lot of monkey patching
        # so we'll do the quickest hack instead.
        # It's quite inefficient as we flatten the parameter structure to conventinal names
        # to then reconstruct the structure from scratch, but it works.
        parameter_and_buffer_dicts = dict(parameter_and_buffer_dicts.flatten_keys("."))
    return functional_call(
        module,
        parameter_and_buffer_dicts,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )

def swap_tensor(
    module: "torch.nn.Module",
    name: str,
    tensor: torch.Tensor,
    allow_missing: bool = False,
) -> torch.Tensor:
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if (
        tensor is not _MISSING
        and not isinstance(tensor, (torch.Tensor, funcdim.Tensor))  # <= adding funcdim.Tensor
        and tensor is not None
    ):
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')

    orig_tensor: torch.Tensor
    if name in module._parameters:
        orig_tensor = module._parameters[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._parameters[name] = tensor  # type: ignore[assignment]
        else:
            del module._parameters[name]
    elif name in module._buffers:
        orig_tensor = module._buffers[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]
    else:
        try:
            orig_tensor = getattr(module, name)
        except AttributeError as ex:
            if not allow_missing:
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{name}`"
                ) from ex
            orig_tensor = _MISSING
        if (
            orig_tensor is not _MISSING
            and not isinstance(orig_tensor, torch.Tensor)
            and orig_tensor is not None
        ):
            raise TypeError(
                f"attribute `{name}`: {orig_tensor} is not an instance of torch.Tensor"
            )
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)
    return orig_tensor

torch.nn.utils._named_member_accessor.swap_tensor = swap_tensor
torch._functorch.functional_call.functional_call = functional_call_patched
torch.func.functional_call = functional_call_patched
