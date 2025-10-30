import importlib
from pathlib import Path

import pytest
import torch
from tensordict.nn import TensorClassModuleBase
from tensordict.tensorclass import TensorClass
from torch import Tensor

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None


class InputTensorClass(TensorClass):
    """Test input TensorClass with two tensor fields."""

    a: Tensor
    b: Tensor


class AddDiffResult(TensorClass):
    """Test output TensorClass for add/diff operations."""

    added: Tensor
    substracted: Tensor


class OutputTensorClass(TensorClass):
    """Test output TensorClass with nested structure."""

    input: InputTensorClass
    result: AddDiffResult


class AddDiffModule(TensorClassModuleBase[InputTensorClass, AddDiffResult]):
    """Test module that adds and subtracts two tensors."""

    def forward(self, x: InputTensorClass) -> AddDiffResult:
        return AddDiffResult(
            added=(x.a + x.b), substracted=(x.a - x.b), batch_size=x.batch_size
        )


class TestTensorClassModule(TensorClassModuleBase[InputTensorClass, OutputTensorClass]):
    """Test module with nested TensorClass output."""

    def __init__(self) -> None:
        super().__init__()
        self.add_diff = AddDiffModule()

    def forward(self, x: InputTensorClass) -> OutputTensorClass:
        return OutputTensorClass(
            input=x, result=self.add_diff(x), batch_size=x.batch_size
        )


class TestTensorClassModuleForward:
    """Tests for TensorClassModule forward pass."""

    def test_forward(self) -> None:
        """Test basic forward pass with TensorClass input."""
        module = TestTensorClassModule()
        value = InputTensorClass(a=10, b=5, batch_size=[])
        output = module.forward(value)
        assert isinstance(output, OutputTensorClass)
        assert output.result.added == 15
        assert output.result.substracted == 5

    def test_td_forward(self) -> None:
        """Test forward pass with TensorDict input via wrapper."""
        td_module = TestTensorClassModule().as_td_module()
        value = InputTensorClass(a=10, b=5, batch_size=[])
        td_output = td_module(value.to_tensordict())
        assert td_output["result", "added"] == 15
        assert td_output["result", "substracted"] == 5

    def test_wrapper_keys(self) -> None:
        """Test that wrapper correctly extracts in_keys and out_keys."""
        module = TestTensorClassModule()
        td_module = module.as_td_module()
        assert set(td_module.in_keys) == {"a", "b"}
        assert set(td_module.out_keys) == {
            ("input", "a"),
            ("input", "b"),
            ("result", "added"),
            ("result", "substracted"),
        }


@pytest.mark.skipif(not _has_onnx, reason="ONNX is not available")
class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_onnx_export_module(self, tmp_path: Path) -> None:
        """Test ONNX export of TensorClassModule."""
        tc_module = TestTensorClassModule()
        tc_module.eval()
        tc_input = InputTensorClass(
            a=torch.tensor([10.0], dtype=torch.float),
            b=torch.tensor([5.0], dtype=torch.float),
            batch_size=[1],
        )
        torch_input = tc_input.to_tensordict().to_dict()

        td_module = tc_module.as_td_module().select_out_keys(
            ("result", "added"), ("result", "substracted")
        )
        output_names = [
            v if isinstance(v, str) else "_".join(v) for v in td_module.out_keys
        ]

        onnx_program = torch.onnx.export(
            model=td_module, kwargs=torch_input, output_names=output_names, dynamo=True
        )

        path = tmp_path / "file.onnx"
        onnx_program.save(str(path))

        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        output_names = [output.name for output in ort_session.get_outputs()]

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        onnxruntime_output_dict = dict(zip(output_names, onnxruntime_outputs))

        tc_outputs = tc_module(tc_input)

        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_output_dict["result_added"]),
            tc_outputs.result.added,
        )
        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_output_dict["result_substracted"]),
            tc_outputs.result.substracted,
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_tensorclass_conversion_error(self) -> None:
        """Test that conversion to TensorDictModule fails for non-TensorClass types."""

        class BadModule(TensorClassModuleBase[Tensor, Tensor]):
            def forward(self, x: Tensor) -> Tensor:
                return x + 1

        module = BadModule()
        with pytest.raises(
            ValueError,
            match="Only TensorClassModuleBase implementations with both input and output type as TensorClass",
        ):
            module.as_td_module()

    def test_batch_size_preservation(self) -> None:
        """Test that batch size is correctly preserved through forward pass."""
        module = AddDiffModule()
        batch_sizes = [[], [3], [2, 3], [1, 2, 3]]

        for batch_size in batch_sizes:
            if batch_size:
                input_tc = InputTensorClass(
                    a=torch.randn(*batch_size),
                    b=torch.randn(*batch_size),
                    batch_size=batch_size,
                )
            else:
                input_tc = InputTensorClass(
                    a=torch.randn(()),
                    b=torch.randn(()),
                    batch_size=batch_size,
                )
            output = module(input_tc)
            assert output.batch_size == torch.Size(batch_size)
            assert output.added.shape == torch.Size(batch_size)
            assert output.substracted.shape == torch.Size(batch_size)
