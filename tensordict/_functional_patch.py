from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from tensordict import TensorDictBase
from torch import Tensor
from torch._functorch.utils import exposed_in
from torch.func import functional_call

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


torch._functorch.functional_call.functional_call = functional_call_patched
torch.func.functional_call = functional_call_patched
