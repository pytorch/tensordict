"""Tests for tabular data import/export (pandas, CSV, Parquet, JSON)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import torch

from tensordict import TensorDict, TensorDictBase

_has_pandas = importlib.util.find_spec("pandas") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestFromPandas:
    def test_numeric_columns(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        td = TensorDict.from_pandas(df)
        assert td.batch_size == torch.Size([3])
        assert td["a"].dtype == torch.int64
        assert td["b"].dtype == torch.float64
        assert (td["a"] == torch.tensor([1, 2, 3])).all()
        assert (td["b"] == torch.tensor([4.0, 5.0, 6.0])).all()

    def test_string_columns(self):
        import pandas as pd

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"], "age": [25, 30, 35]})
        td = TensorDict.from_pandas(df)
        assert td.batch_size == torch.Size([3])
        assert td["age"].dtype == torch.int64
        name_val = td["name"]
        from tensordict.tensorclass import NonTensorData

        assert isinstance(name_val, NonTensorData) or hasattr(name_val, "tolist")

    def test_bool_columns(self):
        import pandas as pd

        df = pd.DataFrame({"flag": [True, False, True], "value": [1, 2, 3]})
        td = TensorDict.from_pandas(df)
        assert td["flag"].dtype == torch.bool
        assert (td["flag"] == torch.tensor([True, False, True])).all()

    def test_batch_size_default(self):
        import pandas as pd

        df = pd.DataFrame({"x": range(10)})
        td = TensorDict.from_pandas(df)
        assert td.batch_size == torch.Size([10])

    def test_batch_size_explicit(self):
        import pandas as pd

        df = pd.DataFrame({"x": range(6)})
        td = TensorDict.from_pandas(df, batch_size=torch.Size([6]))
        assert td.batch_size == torch.Size([6])

    def test_auto_batch_size(self):
        import pandas as pd

        df = pd.DataFrame({"x": range(5)})
        td = TensorDict.from_pandas(df, auto_batch_size=True)
        assert td.batch_size == torch.Size([5])

    def test_auto_batch_size_conflicts_batch_size(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(TypeError, match="Conflicting"):
            TensorDict.from_pandas(df, auto_batch_size=True, batch_size=[2])

    def test_device(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1.0, 2.0]})
        td = TensorDict.from_pandas(df, device="cpu")
        assert td.device == torch.device("cpu")

    def test_separator_nested(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "obs.x": [1.0, 2.0],
                "obs.y": [3.0, 4.0],
                "action": [0, 1],
            }
        )
        td = TensorDict.from_pandas(df, separator=".")
        assert "obs" in td.keys()
        assert (td["obs", "x"] == torch.tensor([1.0, 2.0])).all()
        assert (td["obs", "y"] == torch.tensor([3.0, 4.0])).all()
        assert (td["action"] == torch.tensor([0, 1])).all()

    def test_dtype_override(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        td = TensorDict.from_pandas(df, dtype=torch.float32)
        assert td["a"].dtype == torch.float32
        assert td["b"].dtype == torch.float32

    def test_empty_dataframe(self):
        import pandas as pd

        df = pd.DataFrame(
            {"a": pd.array([], dtype="int64"), "b": pd.array([], dtype="float64")}
        )
        td = TensorDict.from_pandas(df)
        assert td.batch_size == torch.Size([0])

    def test_categorical_columns(self):
        import pandas as pd

        df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "c"])})
        td = TensorDict.from_pandas(df)
        assert td["cat"].dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        assert td["cat"].shape == torch.Size([4])

    def test_module_level_function(self):
        import pandas as pd

        from tensordict import from_pandas

        df = pd.DataFrame({"x": [1, 2, 3]})
        td = from_pandas(df)
        assert td.batch_size == torch.Size([3])
        assert (td["x"] == torch.tensor([1, 2, 3])).all()

    def test_classmethod_on_base(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2]})
        td = TensorDictBase.from_pandas(df)
        assert isinstance(td, TensorDict)


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestToPandas:
    def test_numeric_only(self):
        td = TensorDict({"a": torch.arange(3), "b": torch.zeros(3)}, [3])
        df = td.to_pandas()
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3
        assert (df["a"].values == np.array([0, 1, 2])).all()

    def test_nested_with_separator(self):
        td = TensorDict(
            {
                "obs": TensorDict({"x": torch.ones(2), "y": torch.zeros(2)}, [2]),
                "action": torch.tensor([0, 1]),
            },
            [2],
        )
        df = td.to_pandas(separator=".")
        assert "obs.x" in df.columns
        assert "obs.y" in df.columns
        assert "action" in df.columns

    def test_nested_without_separator_raises(self):
        td = TensorDict(
            {"nested": TensorDict({"x": torch.ones(2)}, [2])},
            [2],
        )
        with pytest.raises(ValueError, match="separator"):
            td.to_pandas()

    def test_multidim_tensor_raises(self):
        td = TensorDict({"matrix": torch.ones(3, 4)}, [3])
        with pytest.raises(ValueError, match="Cannot convert tensor"):
            td.to_pandas()


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestRoundtrip:
    def test_pandas_roundtrip_numeric(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        td = TensorDict.from_pandas(df)
        df2 = td.to_pandas()
        assert (df2["a"].values == df["a"].values).all()
        assert (df2["b"].values == df["b"].values).all()

    def test_pandas_roundtrip_nested(self):
        import pandas as pd

        df = pd.DataFrame({"obs.x": [1.0, 2.0], "obs.y": [3.0, 4.0]})
        td = TensorDict.from_pandas(df, separator=".")
        df2 = td.to_pandas(separator=".")
        assert "obs.x" in df2.columns
        assert (df2["obs.x"].values == df["obs.x"].values).all()


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestCSV:
    def test_from_csv(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_csv(
            csv_path, index=False
        )
        td = TensorDict.from_csv(csv_path)
        assert td.batch_size == torch.Size([3])
        assert (td["a"] == torch.tensor([1, 2, 3])).all()

    def test_to_csv(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "out.csv"
        td = TensorDict({"x": torch.arange(3), "y": torch.zeros(3)}, [3])
        td.to_csv(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "x" in df.columns

    def test_csv_roundtrip(self, tmp_path):
        csv_path = tmp_path / "round.csv"
        td = TensorDict(
            {
                "a": torch.tensor([1, 2, 3]),
                "b": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
            },
            [3],
        )
        td.to_csv(csv_path)
        td2 = TensorDict.from_csv(csv_path)
        assert (td2["a"] == td["a"]).all()
        assert torch.allclose(td2["b"], td["b"])

    def test_module_level_function(self, tmp_path):
        import pandas as pd

        from tensordict import from_csv

        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_path, index=False)
        td = from_csv(csv_path)
        assert td.batch_size == torch.Size([2])


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not found")
class TestParquet:
    def test_from_parquet(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "test.parquet"
        table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        pq.write_table(table, str(path))

        td = TensorDict.from_parquet(path)
        assert td.batch_size == torch.Size([3])
        assert (td["a"] == torch.tensor([1, 2, 3])).all()

    def test_parquet_column_selection(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "test.parquet"
        table = pa.table({"a": [1, 2], "b": [3.0, 4.0], "c": [5, 6]})
        pq.write_table(table, str(path))

        td = TensorDict.from_parquet(path, columns=["a", "c"])
        assert "a" in td.keys()
        assert "c" in td.keys()
        assert "b" not in td.keys()

    @pytest.mark.skipif(not _has_pandas, reason="pandas not found")
    def test_to_parquet(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "out.parquet"
        td = TensorDict({"x": torch.arange(3), "y": torch.zeros(3)}, [3])
        td.to_parquet(path)
        table = pq.read_table(str(path))
        assert table.num_rows == 3
        assert "x" in table.column_names

    @pytest.mark.skipif(not _has_pandas, reason="pandas not found")
    def test_parquet_roundtrip(self, tmp_path):
        path = tmp_path / "round.parquet"
        td = TensorDict(
            {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4.0, 5.0, 6.0])},
            [3],
        )
        td.to_parquet(path)
        td2 = TensorDict.from_parquet(path)
        assert (td2["a"] == td["a"]).all()
        assert torch.allclose(td2["b"], td["b"])

    def test_module_level_function(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from tensordict import from_parquet

        path = tmp_path / "test.parquet"
        table = pa.table({"x": [1, 2]})
        pq.write_table(table, str(path))
        td = from_parquet(path)
        assert td.batch_size == torch.Size([2])


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestJSON:
    def test_from_json_records(self, tmp_path):
        path = tmp_path / "test.json"
        data = [{"a": 1, "b": 4.0}, {"a": 2, "b": 5.0}, {"a": 3, "b": 6.0}]
        path.write_text(json.dumps(data))

        td = TensorDict.from_json(path)
        assert td.batch_size == torch.Size([3])

    def test_from_json_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        lines = [json.dumps({"x": i, "y": float(i)}) for i in range(4)]
        path.write_text("\n".join(lines))

        td = TensorDict.from_json(path, lines=True)
        assert td.batch_size == torch.Size([4])

    def test_to_json(self, tmp_path):
        path = tmp_path / "out.json"
        td = TensorDict({"x": torch.arange(3), "y": torch.zeros(3)}, [3])
        td.to_json(path)
        data = json.loads(path.read_text())
        assert len(data) == 3

    def test_to_json_lines(self, tmp_path):
        path = tmp_path / "out.jsonl"
        td = TensorDict({"x": torch.arange(3)}, [3])
        td.to_json(path, lines=True)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_module_level_function(self, tmp_path):
        from tensordict import from_json

        path = tmp_path / "test.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        td = from_json(path)
        assert td.batch_size == torch.Size([2])


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestFromAnyIntegration:
    def test_from_any_pandas(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        td = TensorDict.from_any(df)
        assert isinstance(td, TensorDict)
        assert td.batch_size == torch.Size([3])
        assert (td["a"] == torch.tensor([1, 2, 3])).all()

    def test_from_any_pandas_with_device(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0]})
        td = TensorDict.from_any(df, device="cpu")
        assert td.device == torch.device("cpu")
