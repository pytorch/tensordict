import pytest
import torch
from tensordict import TensorDict

# Ensure C++ extension is loaded so TorchBind registration runs
import tensordict._C  # noqa: F401


def _tb_class_available():
    try:
        clsns = torch.classes
        return clsns.tensordict.TensorDict is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _tb_class_available(), reason="TorchBind tensordict class not available"
)


@torch.jit.script
def _make_td_scripted(x: torch.Tensor) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
    keys = ["x", "y"]
    vals = [x, x + 1]
    return torch.classes.tensordict.TensorDict.from_pairs(keys, vals, [], x.device)  # type: ignore[attr-defined]


def test_scripted_factory_roundtrip():
    obj = _make_td_scripted(torch.tensor(3))
    # Wrap back to Python TensorDict
    from tensordict import TensorDict

    td = TensorDict.from_torchbind(obj)
    assert set(td.keys()) == {"x", "y"}
    assert td.get("x").item() == 3
    assert td.get("y").item() == 4


def test_fork_wait_roundtrip():
    f1 = torch.jit.fork(_make_td_scripted, torch.tensor(1))
    f2 = torch.jit.fork(_make_td_scripted, torch.tensor(2))
    obj1 = torch.jit.wait(f1)
    obj2 = torch.jit.wait(f2)

    from tensordict import TensorDict

    td1 = TensorDict.from_torchbind(obj1)
    td2 = TensorDict.from_torchbind(obj2)
    assert td1.get("x").item() == 1 and td1.get("y").item() == 2
    assert td2.get("x").item() == 2 and td2.get("y").item() == 3


def test_original_example_tensordict():
    """Test the original example from user query - TensorDict fork/wait."""
    from tensordict import TensorDict

    @torch.jit.script
    def make_td(x: torch.Tensor) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        keys = ["x", "y"]
        vals = [x, x + 1]
        return torch.classes.tensordict.TensorDict.from_pairs(keys, vals, [], x.device)  # type: ignore[attr-defined]

    def parallel():
        fut1 = torch.jit.fork(make_td, torch.tensor(1))
        fut2 = torch.jit.fork(make_td, torch.tensor(2))
        obj1 = torch.jit.wait(fut1)
        obj2 = torch.jit.wait(fut2)
        # Convert back to Python TensorDict
        td1 = TensorDict.from_torchbind(obj1)
        td2 = TensorDict.from_torchbind(obj2)
        return td1, td2

    td1, td2 = parallel()
    assert td1.get("x").item() == 1
    assert td1.get("y").item() == 2
    assert td2.get("x").item() == 2
    assert td2.get("y").item() == 3


def test_torchbind_methods_in_scripted():
    """Test that TorchBind methods work inside scripted functions."""
    @torch.jit.script
    def process_td(td: torch.classes.tensordict.TensorDict) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        x = td.get("x")
        td.set("y", x + 10)
        return td

    tb = torch.classes.tensordict.TensorDict(torch.device("cpu"))  # type: ignore[attr-defined]
    tb.set("x", torch.tensor(5))
    result = process_td(tb)
    assert result.get("y").item() == 15


def test_to_torchbind_requires_device():
    from tensordict import TensorDict, tensordict as td_mod

    td = TensorDict({"x": torch.tensor(1), "y": torch.tensor(2)}, batch_size=[])
    # Device is None by default when not enforced at construction
    assert td.device is None
    with pytest.raises(RuntimeError, match="requires a non-None device"):
        td_mod.to_torchbind(td)


def test_to_torchbind_with_device():
    """Test conversion to TorchBind when device is set."""
    from tensordict import TensorDict

    td = TensorDict({"x": torch.tensor(1), "y": torch.tensor(2)}, batch_size=[], device="cpu")
    tb = td.to_torchbind()
    assert tb.get("x").item() == 1
    assert tb.get("y").item() == 2
    assert tb.device() == torch.device("cpu")


def test_batch_size_validation():
    # batch_size [] should accept 0-d tensors
    obj = _make_td_scripted(torch.tensor(0))
    from tensordict import TensorDict

    td = TensorDict.from_torchbind(obj)
    assert tuple(td.batch_size) == ()


def test_torchbind_keys():
    """Test keys() method on TorchBind class."""
    tb = torch.classes.tensordict.TensorDict(torch.device("cpu"))  # type: ignore[attr-defined]
    tb.set("a", torch.ones(1))
    tb.set("b", torch.ones(2))
    keys = tb.keys()
    assert set(keys) == {"a", "b"}


def test_torchbind_clone():
    """Test clone() method on TorchBind class."""
    tb = torch.classes.tensordict.TensorDict(torch.device("cpu"))  # type: ignore[attr-defined]
    tb.set("x", torch.ones(1))
    tb_clone = tb.clone()
    assert tb_clone.get("x").item() == 1
    # Modify clone shouldn't affect original
    tb_clone.set("x", torch.tensor(2.0))
    assert tb.get("x").item() == 1
    assert tb_clone.get("x").item() == 2


def test_torchbind_to_device():
    """Test to() method for device conversion."""
    tb = torch.classes.tensordict.TensorDict(torch.device("cpu"))  # type: ignore[attr-defined]
    tb.set("x", torch.ones(1))
    assert tb.device() == torch.device("cpu")
    # Note: CUDA test is separate below


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_validation_cuda():
    # Create a CPU-class then try to set CUDA tensor -> should error
    cpu_device = torch.device("cpu")
    tb = torch.classes.tensordict.TensorDict(cpu_device)  # type: ignore[attr-defined]
    cpu_t = torch.ones(1)
    tb.set("x", cpu_t)  # OK
    cuda_t = torch.ones(1, device="cuda")
    with pytest.raises(RuntimeError, match="All tensors must be on device"):
        tb.set("y", cuda_t)


def test_jit_script_end_to_end():
    """End-to-end test demonstrating JIT scripting with TensorDict.
    
    This test shows the complete workflow:
    1. Create a Python TensorDict
    2. Convert to TorchBind format
    3. Use in a scripted function
    4. Convert back to Python TensorDict
    5. Verify results
    """

    # Step 1: Create a Python TensorDict with some data
    td = TensorDict(
        {
            "input": torch.tensor([1.0, 2.0, 3.0]),
            "weight": torch.tensor([0.5, 1.5, 2.5]),
        },
        batch_size=[],
        device="cpu",
    )

    # Step 2: Convert Python TensorDict to TorchBind format
    tb = td.to_torchbind()
    assert tb.device() == torch.device("cpu")
    assert set(tb.keys()) == {"input", "weight"}

    # Step 3: Define a scripted function that processes the TensorDict
    @torch.jit.script
    def process_tensordict(
        td: torch.classes.tensordict.TensorDict,  # type: ignore[attr-defined]
        scale: float,
    ) -> torch.classes.tensordict.TensorDict:  # type: ignore[attr-defined]
        """Process TensorDict: multiply input by weight, add scale, store result."""
        input_val = td.get("input")
        weight_val = td.get("weight")
        # Compute result
        result = input_val * weight_val + scale
        # Store result back in TensorDict
        td.set("output", result)
        return td

    # Step 4: Call the scripted function with TorchBind TensorDict
    result_tb = process_tensordict(tb, scale=10.0)

    # Step 5: Convert back to Python TensorDict for verification
    result_td = TensorDict.from_torchbind(result_tb)

    # Step 6: Verify the results
    assert set(result_td.keys()) == {"input", "weight", "output"}
    assert torch.allclose(result_td.get("input"), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(result_td.get("weight"), torch.tensor([0.5, 1.5, 2.5]))
    # output = input * weight + scale = [1*0.5, 2*1.5, 3*2.5] + 10 = [0.5, 3.0, 7.5] + 10 = [10.5, 13.0, 17.5]
    expected_output = torch.tensor([10.5, 13.0, 17.5])
    assert torch.allclose(result_td.get("output"), expected_output)


def test_to_torchbind_context_manager():
    """Test that to_torchbind() works as a context manager, automatically converting back."""
    from tensordict import TensorDict

    # Create a Python TensorDict
    td = TensorDict(
        {"x": torch.tensor(1.0), "y": torch.tensor(2.0)}, batch_size=[], device="cpu"
    )

    # Define a scripted function that modifies the TorchBind TensorDict
    @torch.jit.script
    def modify_tb(tb: torch.classes.tensordict.TensorDict) -> None:  # type: ignore[attr-defined]
        x = tb.get("x")
        tb.set("x", x + 10)
        tb.set("z", torch.tensor(30.0))

    # Use as context manager
    with td.to_torchbind() as tb:
        # Verify we have a TorchBind object inside the context
        assert tb.device() == torch.device("cpu")
        
        # Modify the TorchBind TensorDict
        modify_tb(tb)
        
        # Verify modifications in TorchBind object
        assert tb.get("x").item() == 11.0
        assert tb.get("y").item() == 2.0
        assert tb.get("z").item() == 30.0

    # After exiting context, td should be automatically updated from TorchBind
    # Verify td is still a Python TensorDict
    assert isinstance(td, TensorDict)
    
    # Verify the modifications were propagated back
    assert td.get("x").item() == 11.0
    assert td.get("y").item() == 2.0
    assert td.get("z").item() == 30.0
    
    # Verify we can still use Python TensorDict methods
    assert set(td.keys()) == {"x", "y", "z"}




