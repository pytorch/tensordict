#!/usr/bin/env python3
"""Manual test to verify to_torchbind context manager works."""

import torch
from tensordict import TensorDict

# Ensure C++ extension is loaded
try:
    import tensordict._C  # noqa: F401
except Exception:
    print("TorchBind extension not available, skipping test")
    exit(0)

# Check if TorchBind class is available
try:
    tb_class = torch.classes.tensordict.TensorDict
    print(f"✓ TorchBind class available: {tb_class}")
except Exception as e:
    print(f"✗ TorchBind class not available: {e}")
    exit(0)

# Test 1: Basic context manager usage
print("\n=== Test 1: Basic Context Manager ===")
td = TensorDict(
    {"x": torch.tensor(1.0), "y": torch.tensor(2.0)},
    batch_size=[],
    device="cpu"
)
print(f"Before context: {dict(td.items())}")

with td.to_torchbind() as tb:
    print(f"Inside context (TorchBind): device={tb.device()}, keys={tb.keys()}")
    # Modify in TorchBind
    tb.set("x", torch.tensor(10.0))
    tb.set("z", torch.tensor(30.0))
    print(f"After modification: x={tb.get('x').item()}, z={tb.get('z').item()}")

print(f"After context: {dict(td.items())}")
assert td.get("x").item() == 10.0, "x should be updated"
assert td.get("z").item() == 30.0, "z should be added"
print("✓ Test 1 passed")

# Test 2: Using scripted function
print("\n=== Test 2: With Scripted Function ===")
td2 = TensorDict(
    {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
    batch_size=[],
    device="cpu"
)

@torch.jit.script
def process_tb(tb: torch.classes.tensordict.TensorDict):  # type: ignore[attr-defined]
    a = tb.get("a")
    b = tb.get("b")
    tb.set("c", a + b)

print(f"Before: {dict(td2.items())}")
with td2.to_torchbind() as tb:
    process_tb(tb)
    print(f"Inside: c exists = {tb.has('c')}")

print(f"After: {dict(td2.items())}")
assert "c" in td2.keys(), "c should be added"
assert torch.allclose(td2.get("c"), torch.tensor([4.0, 6.0])), "c should be a + b"
print("✓ Test 2 passed")

print("\n✅ All tests passed!")





