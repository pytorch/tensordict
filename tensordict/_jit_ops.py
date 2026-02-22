# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""TorchScript-compatible ops for TensorDict operations."""

import torch

# Ensure C++ extension is loaded
try:
    import tensordict._C  # noqa: F401
except ImportError:
    pass


def _get_tb_class():
    """Get the TorchBind TensorDict class if available."""
    try:
        clsns = getattr(torch, "classes", None)
        tdns = getattr(clsns, "tensordict", None) if clsns is not None else None
        return getattr(tdns, "TensorDict", None) if tdns is not None else None
    except Exception:
        return None


_TB_TensorDict = _get_tb_class()

# Script functions that work on TorchBind class
if _TB_TensorDict is not None:

    @torch.jit.script
    def _td_get_scripted(td: torch.classes.tensordict.TensorDict, key: str) -> torch.Tensor:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.get()."""
        return torch.ops.tensordict.get(td, key)

    @torch.jit.script
    def _td_set_scripted(td: torch.classes.tensordict.TensorDict, key: str, value: torch.Tensor) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.set()."""
        return torch.ops.tensordict.set(td, key, value)

    @torch.jit.script
    def _td_has_scripted(td: torch.classes.tensordict.TensorDict, key: str) -> bool:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.has()."""
        return torch.ops.tensordict.has(td, key)

    @torch.jit.script
    def _td_keys_scripted(td: torch.classes.tensordict.TensorDict) -> list[str]:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.keys()."""
        return torch.ops.tensordict.keys(td)

    @torch.jit.script
    def _td_device_scripted(td: torch.classes.tensordict.TensorDict) -> torch.device:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.device."""
        return torch.ops.tensordict.device(td)

    @torch.jit.script
    def _td_to_scripted(td: torch.classes.tensordict.TensorDict, device: torch.device) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.to()."""
        return torch.ops.tensordict.to(td, device)

    @torch.jit.script
    def _td_clone_scripted(td: torch.classes.tensordict.TensorDict) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        """Scripted version of TensorDict.clone()."""
        return torch.ops.tensordict.clone(td)

