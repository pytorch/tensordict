# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch

from tensordict import TensorDict

MODES = ["stack_out", "set_at_loop", "copy_at", "manual_leaf_copy"]
if hasattr(TensorDict, "prepare_copy_at_"):
    MODES.insert(-1, "copy_at_writer")


def _maybe_synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _make_rollout(num_envs, time_steps, device):
    return TensorDict(
        {
            "observation": torch.empty(num_envs, time_steps, 60, device=device),
            "action": torch.empty(num_envs, time_steps, 8, device=device),
            "done": torch.empty(
                num_envs, time_steps, 1, dtype=torch.bool, device=device
            ),
            "terminated": torch.empty(
                num_envs, time_steps, 1, dtype=torch.bool, device=device
            ),
            "truncated": torch.empty(
                num_envs, time_steps, 1, dtype=torch.bool, device=device
            ),
            "collector": TensorDict(
                {
                    "traj_ids": torch.empty(
                        num_envs, time_steps, dtype=torch.long, device=device
                    )
                },
                batch_size=(num_envs, time_steps),
                device=device,
            ),
            "next": TensorDict(
                {
                    "observation": torch.empty(num_envs, time_steps, 60, device=device),
                    "reward": torch.empty(num_envs, time_steps, 1, device=device),
                    "done": torch.empty(
                        num_envs, time_steps, 1, dtype=torch.bool, device=device
                    ),
                    "terminated": torch.empty(
                        num_envs, time_steps, 1, dtype=torch.bool, device=device
                    ),
                    "truncated": torch.empty(
                        num_envs, time_steps, 1, dtype=torch.bool, device=device
                    ),
                },
                batch_size=(num_envs, time_steps),
                device=device,
            ),
        },
        batch_size=(num_envs, time_steps),
        device=device,
    )


def _make_step(num_envs, device):
    return TensorDict(
        {
            "observation": torch.randn(num_envs, 60, device=device),
            "action": torch.randn(num_envs, 8, device=device),
            "done": torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            "terminated": torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            "truncated": torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            "collector": TensorDict(
                {"traj_ids": torch.arange(num_envs, device=device)},
                batch_size=(num_envs,),
                device=device,
            ),
            "next": TensorDict(
                {
                    "observation": torch.randn(num_envs, 60, device=device),
                    "reward": torch.randn(num_envs, 1, device=device),
                    "done": torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
                    "terminated": torch.zeros(
                        num_envs, 1, dtype=torch.bool, device=device
                    ),
                    "truncated": torch.zeros(
                        num_envs, 1, dtype=torch.bool, device=device
                    ),
                },
                batch_size=(num_envs,),
                device=device,
            ),
        },
        batch_size=(num_envs,),
        device=device,
    )


def _make_steps(num_envs, time_steps, device):
    return [_make_step(num_envs, device) for _ in range(time_steps)]


def _wide_key(index):
    if index % 2:
        return (f"root{index % 5}", f"mid{index % 3}", f"value{index}")
    return (f"root{index % 5}", f"value{index}")


def _wide_width(index):
    return (1, 2, 4, 8, 16)[index % 5]


def _make_wide_rollout(batch, time_steps, device, n_leaves=25):
    rollout = TensorDict({}, batch_size=(batch, time_steps), device=device)
    for i in range(n_leaves):
        rollout.set(
            _wide_key(i),
            torch.empty(batch, time_steps, _wide_width(i), device=device),
        )
    return rollout


def _make_wide_step(batch, device, n_leaves=25):
    step = TensorDict({}, batch_size=(batch,), device=device)
    for i in range(n_leaves):
        step.set(
            _wide_key(i),
            torch.randn(batch, _wide_width(i), device=device),
        )
    return step


def _make_wide_steps(batch, time_steps, device):
    return [_make_wide_step(batch, device) for _ in range(time_steps)]


def _stack_out(rollout, steps, device):
    torch.stack(steps, dim=1, out=rollout)
    _maybe_synchronize(device)


def _copy_at(rollout, steps, device):
    for i, step in enumerate(steps):
        rollout.copy_at_(step, idx=(slice(None), i), fast=True)
    _maybe_synchronize(device)


def _set_at_loop(rollout, steps, device):
    for i, step in enumerate(steps):
        idx = (slice(None), i)
        for key, value in step.items():
            rollout.set_at_(key, value, idx)
    _maybe_synchronize(device)


def _copy_at_writer(writer, steps, device):
    for i, step in enumerate(steps):
        writer.copy_(step, index=i)
    _maybe_synchronize(device)


def _manual_leaf_copy(rollout, step_values, dest_values, device):
    for i, values in enumerate(step_values):
        for dest, source in zip(dest_values, values):
            dest[:, i].copy_(source)
    _maybe_synchronize(device)


@pytest.mark.parametrize("time_steps", [2, 8, 32])
@pytest.mark.parametrize("num_envs", [32, 4096])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", MODES)
def test_collector_preallocated_write(benchmark, num_envs, time_steps, device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    rollout = _make_rollout(num_envs, time_steps, device)
    steps = _make_steps(num_envs, time_steps, device)
    keys, dest_values = rollout._items_list(True, True)
    step_values = [
        step._items_list(True, True, sorting_keys=keys, default=None)[1]
        for step in steps
    ]
    _maybe_synchronize(device)

    if mode == "stack_out":
        benchmark(_stack_out, rollout, steps, device)
    elif mode == "set_at_loop":
        benchmark(_set_at_loop, rollout, steps, device)
    elif mode == "copy_at":
        benchmark(_copy_at, rollout, steps, device)
    elif mode == "copy_at_writer":
        writer = rollout.prepare_copy_at_(dim=1, source=steps[0])
        benchmark(_copy_at_writer, writer, steps, device)
    elif mode == "manual_leaf_copy":
        benchmark(_manual_leaf_copy, rollout, step_values, dest_values, device)


@pytest.mark.parametrize("time_steps", [128, 256])
@pytest.mark.parametrize("batch", [128, 4096])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", MODES)
def test_wide_nested_preallocated_write(benchmark, batch, time_steps, device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    rollout = _make_wide_rollout(batch, time_steps, device)
    steps = _make_wide_steps(batch, time_steps, device)
    keys, dest_values = rollout._items_list(True, True)
    step_values = [
        step._items_list(True, True, sorting_keys=keys, default=None)[1]
        for step in steps
    ]
    _maybe_synchronize(device)

    if mode == "stack_out":
        benchmark(_stack_out, rollout, steps, device)
    elif mode == "set_at_loop":
        benchmark(_set_at_loop, rollout, steps, device)
    elif mode == "copy_at":
        benchmark(_copy_at, rollout, steps, device)
    elif mode == "copy_at_writer":
        writer = rollout.prepare_copy_at_(dim=1, source=steps[0])
        benchmark(_copy_at_writer, writer, steps, device)
    elif mode == "manual_leaf_copy":
        benchmark(_manual_leaf_copy, rollout, step_values, dest_values, device)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
