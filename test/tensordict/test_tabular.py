"""Tests for tabular data import/export (pandas, CSV, Parquet, JSON)."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings

import numpy as np
import pytest
import torch
from tensordict import (
    from_csv,
    from_json,
    from_pandas,
    from_parquet,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack

_has_pandas = importlib.util.find_spec("pandas") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None


def test_tabular_module_is_deprecated():
    import tensordict._tabular

    sys.modules.pop("tensordict.tabular", None)
    with pytest.warns(
        DeprecationWarning,
        match=r"^tensordict\.tabular is deprecated and will be removed in TensorDict "
        r"0\.17\. Use TensorDict\.from_pandas, from_csv, from_parquet, from_json, "
        r"to_pandas, to_csv, to_parquet and to_json instead\.$",
    ) as record:
        import tensordict.tabular as tabular
    assert record[0].filename == __file__
    assert tabular._read_csv is tensordict._tabular._read_csv
    with pytest.raises(AttributeError):
        tabular.not_a_name


@tensorclass
class TabularTensorClass:
    x: torch.Tensor
    y: torch.Tensor


@tensorclass
class TabularNestedTensorClass:
    a: torch.Tensor
    inner: TabularTensorClass


def _nested_tensorclass():
    return TabularNestedTensorClass(
        a=torch.tensor([1.0, 2.0, 3.0]),
        inner=TabularTensorClass(
            x=torch.tensor([4, 5, 6]), y=torch.tensor([7.0, 8.0, 9.0]), batch_size=[3]
        ),
        batch_size=[3],
    )


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

    def test_to_parquet_without_pandas(self, tmp_path, monkeypatch):
        # The parquet extra only installs pyarrow: to_parquet must not need pandas.
        import pyarrow as pa
        import tensordict._tabular

        # pyarrow looks for pandas once, on first use, and remembers the result:
        # let it do so before pandas is hidden, so other tests are not affected.
        pa.table({"a": [1]})
        monkeypatch.setattr(tensordict._tabular, "_has_pandas", lambda: False)
        monkeypatch.setitem(sys.modules, "pandas", None)
        path = tmp_path / "no_pandas.parquet"
        td = TensorDict(
            a=torch.tensor([1.0, 2.0, 3.0]),
            b=torch.tensor([True, False, True]),
            s=NonTensorStack("x", "y", "z"),
            obs=TensorDict(x=torch.tensor([4, 5, 6]), batch_size=[3]),
            batch_size=[3],
        )
        td.to_parquet(path, separator=".")
        td2 = TensorDict.from_parquet(path, separator=".")
        assert td2.batch_size == torch.Size([3])
        assert (td2["a"] == td["a"]).all()
        assert (td2["b"] == td["b"]).all()
        assert (td2["obs", "x"] == td["obs", "x"]).all()
        assert td2["s"] == ["x", "y", "z"]

    @pytest.mark.parametrize("with_pandas", [True, False])
    @pytest.mark.parametrize("in_tensordict", [True, False])
    def test_to_parquet_nested_tensorclass(
        self, tmp_path, monkeypatch, with_pandas, in_tensordict
    ):
        # A nested tensorclass gets one column per leaf, as a nested TensorDict.
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tensordict._tabular

        if with_pandas and not _has_pandas:
            pytest.skip("pandas not found")
        pa.table({"a": [1]})
        if not with_pandas:
            monkeypatch.setattr(tensordict._tabular, "_has_pandas", lambda: False)
            monkeypatch.setitem(sys.modules, "pandas", None)
        tc = _nested_tensorclass()
        data = TensorDict(tc=tc, batch_size=[3]) if in_tensordict else tc
        path = tmp_path / "nested.parquet"
        data.to_parquet(path, separator=".")
        prefix = ("tc",) if in_tensordict else ()
        assert pq.read_table(str(path)).column_names == [
            ".".join((*prefix, "a")),
            ".".join((*prefix, "inner", "x")),
            ".".join((*prefix, "inner", "y")),
        ]
        td2 = TensorDict.from_parquet(path, separator=".")
        assert (td2[(*prefix, "a")] == tc.a).all()
        assert (td2[(*prefix, "inner", "x")] == tc.inner.x).all()
        assert (td2[(*prefix, "inner", "y")] == tc.inner.y).all()

    def test_module_level_function(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

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
        path = tmp_path / "test.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        td = from_json(path)
        assert td.batch_size == torch.Size([2])

    @pytest.mark.parametrize("lines", [False, True])
    def test_from_json_without_pandas_keeps_later_fields(
        self, tmp_path, monkeypatch, lines
    ):
        # https://github.com/pytorch/tensordict/issues/3063
        # Without pandas, the fields that the first record lacks were dropped.
        import tensordict._tabular

        monkeypatch.setattr(tensordict._tabular, "_has_pandas", lambda: False)
        monkeypatch.setitem(sys.modules, "pandas", None)

        def read(records):
            path = tmp_path / ("test.jsonl" if lines else "test.json")
            if lines:
                path.write_text("\n".join(json.dumps(r) for r in records))
            else:
                path.write_text(json.dumps(records))
            return TensorDict.from_json(path, lines=lines)

        # "c" is a number field with missing values, which warns (#3069)
        with pytest.warns(FutureWarning, match="TensorDict 0.17"):
            td = read([{"a": "x"}, {"a": "y", "b": "z"}, {"c": 3.5, "a": "w"}])
        assert td.batch_size == torch.Size([3])
        assert list(td.keys()) == ["a", "b", "c"]
        assert td["a"] == ["x", "y", "w"]
        assert td["b"] == [None, "z", None]
        assert td["c"][2] == 3.5
        td = read([{}, {"b": "later"}])
        assert td.batch_size == torch.Size([2])
        assert list(td.keys()) == ["b"]
        assert td["b"] == [None, "later"]

    @pytest.mark.parametrize("lines", [False, True])
    def test_from_json_without_pandas_warns_for_missing_numbers(
        self, tmp_path, monkeypatch, lines
    ):
        # https://github.com/pytorch/tensordict/issues/3069
        # Without pandas, a number field with missing or null values is a
        # non-tensor entry with None; from TensorDict 0.17 it is a float tensor
        # with NaN, as with pandas.
        import tensordict._tabular

        monkeypatch.setattr(tensordict._tabular, "_has_pandas", lambda: False)
        monkeypatch.setitem(sys.modules, "pandas", None)

        def read(records):
            path = tmp_path / ("test.jsonl" if lines else "test.json")
            if lines:
                path.write_text("\n".join(json.dumps(r) for r in records))
            else:
                path.write_text(json.dumps(records))
            return TensorDict.from_json(path, lines=lines)

        with pytest.warns(FutureWarning, match="TensorDict 0.17") as record:
            td = read([{"a": 1, "b": 2, "c": "x"}, {"a": 3, "d": None}])
        (warning,) = [w for w in record if "TensorDict 0.17" in str(w.message)]
        assert "'b', 'd'" in str(warning.message)
        assert warning.filename == __file__
        assert td["b"] == [2, None]
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            read([{"a": 1}, {"a": 2.5}])
            read([{"s": "x"}, {"t": [1]}])


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


@pytest.mark.skipif(not _has_pandas, reason="pandas not found")
class TestTensorClassTabular:
    def test_from_pandas_returns_tensorclass(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        tc = TabularTensorClass.from_pandas(df)
        assert isinstance(tc, TabularTensorClass)
        assert tc.batch_size == torch.Size([3])
        assert (tc.x == torch.tensor([1, 2, 3])).all()
        assert (tc.y == torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)).all()

    def test_to_pandas(self):
        tc = TabularTensorClass(
            x=torch.tensor([1, 2, 3]),
            y=torch.tensor([4.0, 5.0, 6.0]),
            batch_size=[3],
        )
        df = tc.to_pandas()
        assert list(df.columns) == ["x", "y"]
        assert (df["x"].values == np.array([1, 2, 3])).all()
        assert (df["y"].values == np.array([4.0, 5.0, 6.0], dtype=np.float32)).all()

    def test_nested_tensorclass_columns(self, tmp_path):
        # A nested tensorclass gets one column per leaf, as a nested TensorDict.
        import pandas as pd

        tc = _nested_tensorclass()
        columns = ["a", "inner.x", "inner.y"]
        assert list(tc.to_tensordict().to_pandas(separator=".").columns) == columns
        df = tc.to_pandas(separator=".")
        assert list(df.columns) == columns
        assert (df["inner.x"].values == np.array([4, 5, 6])).all()
        csv_path = tmp_path / "nested.csv"
        tc.to_csv(csv_path, separator=".")
        assert list(pd.read_csv(csv_path).columns) == columns
        td2 = TensorDict.from_csv(csv_path, separator=".")
        assert (td2["inner", "x"] == tc.inner.x).all()

    def test_nested_tensorclass_multi_dim_leaf(self, tmp_path):
        # A nested tensorclass with a leaf that does not fit one column stays
        # one object column, where a nested TensorDict raises.
        import pandas as pd

        tc = _nested_tensorclass()
        tc.inner.x = torch.ones(3, 2)
        with pytest.raises(ValueError, match="Cannot convert tensor with shape"):
            tc.to_tensordict().to_pandas(separator=".")
        df = tc.to_pandas(separator=".")
        assert list(df.columns) == ["a", "inner"]
        assert isinstance(df["inner"][0], TabularTensorClass)
        assert (df["inner"][0].x == tc.inner.x[0]).all()
        df_outer = TensorDict(tc=tc, batch_size=[3]).to_pandas(separator=".")
        assert list(df_outer.columns) == ["tc"]
        csv_path = tmp_path / "nested.csv"
        tc.to_csv(csv_path, separator=".")
        df2 = pd.read_csv(csv_path)
        assert list(df2.columns) == ["a", "inner"]
        assert df2["inner"][0] == str(df["inner"][0])

    def test_nested_tensorclass_without_leaves(self):
        # A nested tensorclass whose fields are all None has nothing to flatten
        # and stays one object column.
        tc = TabularNestedTensorClass(
            a=torch.tensor([1.0, 2.0, 3.0]),
            inner=TabularTensorClass(x=None, y=None, batch_size=[3]),
            batch_size=[3],
        )
        df = tc.to_pandas(separator=".")
        assert list(df.columns) == ["a", "inner"]
        assert isinstance(df["inner"][0], TabularTensorClass)

    def test_csv_roundtrip(self, tmp_path):
        csv_path = tmp_path / "tensorclass.csv"
        tc = TabularTensorClass(
            x=torch.tensor([1, 2, 3]),
            y=torch.tensor([4.0, 5.0, 6.0]),
            batch_size=[3],
        )
        tc.to_csv(csv_path)
        tc2 = TabularTensorClass.from_csv(csv_path)
        assert isinstance(tc2, TabularTensorClass)
        assert (tc2.x == tc.x).all()
        assert torch.allclose(tc2.y.to(tc.y.dtype), tc.y)

    def test_json_roundtrip(self, tmp_path):
        path = tmp_path / "tensorclass.json"
        tc = TabularTensorClass(
            x=torch.tensor([1, 2, 3]),
            y=torch.tensor([4.0, 5.0, 6.0]),
            batch_size=[3],
        )
        tc.to_json(path)
        tc2 = TabularTensorClass.from_json(path)
        assert isinstance(tc2, TabularTensorClass)
        assert (tc2.x == tc.x).all()
        assert torch.allclose(tc2.y.to(tc.y.dtype), tc.y)

    @pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not found")
    def test_parquet_roundtrip(self, tmp_path):
        path = tmp_path / "tensorclass.parquet"
        tc = TabularTensorClass(
            x=torch.tensor([1, 2, 3]),
            y=torch.tensor([4.0, 5.0, 6.0]),
            batch_size=[3],
        )
        tc.to_parquet(path)
        tc2 = TabularTensorClass.from_parquet(path)
        assert isinstance(tc2, TabularTensorClass)
        assert (tc2.x == tc.x).all()
        assert torch.allclose(tc2.y.to(tc.y.dtype), tc.y)
