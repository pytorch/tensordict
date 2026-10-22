"""Tests for tabular data import/export (pandas, CSV, Parquet, JSON, Arrow)."""

from __future__ import annotations

import ast
import datetime
import decimal
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import warnings

import _tabular_cases
import numpy as np
import pytest
import torch
from tensordict import (
    from_arrow,
    from_csv,
    from_json,
    from_pandas,
    from_parquet,
    tensorclass,
    TensorDict,
    TensorDictBase,
    to_arrow,
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

    @pytest.mark.parametrize("row_group_size", [None, 3])
    def test_from_parquet_fixed_shape_tensor(self, tmp_path, row_group_size):
        # A fixed_shape_tensor column keeps its per-row shape (#3101).
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not hasattr(pa, "FixedShapeTensorArray"):
            pytest.skip("pyarrow has no fixed_shape_tensor type")
        values = np.arange(48, dtype=np.float32).reshape(8, 2, 3)
        path = tmp_path / "tensor.parquet"
        table = pa.table({"t": pa.FixedShapeTensorArray.from_numpy_ndarray(values)})
        pq.write_table(table, str(path), row_group_size=row_group_size)
        if row_group_size is not None:
            assert pq.read_table(str(path)).column("t").num_chunks > 1

        td = TensorDict.from_parquet(path)
        assert td.batch_size == torch.Size([8])
        assert td["t"].shape == torch.Size([8, 2, 3])
        assert td["t"].dtype == torch.float32
        assert (td["t"] == torch.from_numpy(values)).all()

    def test_from_parquet_fixed_shape_tensor_bool(self, tmp_path):
        # to_numpy_ndarray() rejects bool, so these still come back flat.
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not hasattr(pa, "FixedShapeTensorArray"):
            pytest.skip("pyarrow has no fixed_shape_tensor type")
        values = np.arange(12).reshape(3, 2, 2) % 3 == 0
        path = tmp_path / "mask.parquet"
        table = pa.table({"m": pa.FixedShapeTensorArray.from_numpy_ndarray(values)})
        pq.write_table(table, str(path))

        td = TensorDict.from_parquet(path)
        assert td["m"].dtype == torch.bool
        assert (td["m"] == torch.from_numpy(values).reshape(3, 4)).all()

    def test_from_parquet_fixed_size_list(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "list.parquet"
        column = pa.array([[1, 2, 3], [4, 5, 6]], type=pa.list_(pa.int32(), 3))
        pq.write_table(pa.table({"f": column}), str(path))

        td = TensorDict.from_parquet(path)
        assert td["f"].dtype == torch.int64
        assert (td["f"] == torch.tensor([[1, 2, 3], [4, 5, 6]])).all()

    def test_module_level_function(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "test.parquet"
        table = pa.table({"x": [1, 2]})
        pq.write_table(table, str(path))
        td = from_parquet(path)
        assert td.batch_size == torch.Size([2])


def _arrow_table():
    import pyarrow as pa

    n = 5
    values = pa.array(np.arange(9, dtype=np.float32))
    offsets = pa.array(np.array([0, 1, 3, 3, 6, 9], dtype=np.int64))
    ragged = pa.LargeListArray.from_arrays(offsets, values)
    return pa.table(
        {
            "f": pa.array(np.arange(n, dtype=np.float32)),
            "u": pa.array(np.arange(n, dtype=np.uint8)),
            "b": pa.array([True, False, True, True, False]),
            "ragged": ragged,
            "ragged32": pa.array(
                [[1], [2, 3], [], [4, 5, 6], [7]], type=pa.list_(pa.int32())
            ),
            "fsl": pa.array(
                [[i, i + 1] for i in range(n)], type=pa.list_(pa.int64(), 2)
            ),
            "fst": pa.FixedShapeTensorArray.from_numpy_ndarray(
                np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3)
            ),
            "st": pa.StructArray.from_arrays(
                [pa.array(np.arange(n)), ragged], names=["x", "z"]
            ),
            "s": pa.array(["a", "bb", "ccc", "d", "e"]),
            "strings": pa.array([["a"], [], ["b", "c"], ["d"], []]),
            "date": pa.array([datetime.date(2024, 1, i + 1) for i in range(n)]),
        }
    )


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not found")
@pytest.mark.filterwarnings("error::DeprecationWarning")
class TestArrow:
    def test_from_arrow_types(self):
        td = from_arrow(_arrow_table())
        assert td.batch_size == torch.Size([5])
        assert torch.equal(td["f"], torch.arange(5, dtype=torch.float32))
        assert torch.equal(td["u"], torch.arange(5, dtype=torch.uint8))
        assert torch.equal(td["b"], torch.tensor([True, False, True, True, False]))
        assert td["ragged"].is_nested and td["ragged"].layout is torch.jagged
        assert [x.tolist() for x in td["ragged"].unbind(0)] == [
            [0.0],
            [1.0, 2.0],
            [],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
        assert td["ragged32"].dtype == torch.int32
        assert [x.tolist() for x in td["ragged32"].unbind(0)] == [
            [1],
            [2, 3],
            [],
            [4, 5, 6],
            [7],
        ]
        assert torch.equal(td["fsl"], torch.arange(5)[:, None] + torch.arange(2))
        assert torch.equal(
            td["fst"], torch.arange(30, dtype=torch.float32).view(5, 2, 3)
        )
        assert isinstance(td["st"], TensorDict)
        assert torch.equal(td["st", "x"], torch.arange(5))
        assert td["st", "z"].is_nested
        assert isinstance(td.get("s"), NonTensorStack)
        assert td.get("s").tolist() == ["a", "bb", "ccc", "d", "e"]
        assert td.get("strings").tolist() == [["a"], [], ["b", "c"], ["d"], []]
        assert td.get("date").tolist()[1] == datetime.date(2024, 1, 2)

    def test_from_arrow_views(self):
        table = _arrow_table()

        def address(name):
            return table.column(name).chunk(0).buffers()[1].address

        td = from_arrow(table)
        assert td["f"].data_ptr() == address("f")
        ragged = table.column("ragged").chunk(0)
        assert td["ragged"].values().data_ptr() == ragged.values.buffers()[1].address
        assert td["ragged"].offsets().data_ptr() == ragged.offsets.buffers()[1].address
        assert td["st", "x"].data_ptr() == (
            table.column("st").chunk(0).field(0).buffers()[1].address
        )
        fst = table.column("fst").chunk(0).storage.values
        assert td["fst"].data_ptr() == fst.buffers()[1].address
        td = from_arrow(table, copy=True)
        assert td["f"].data_ptr() != address("f")
        assert td["ragged"].values().data_ptr() != ragged.values.buffers()[1].address
        assert torch.equal(td["f"], torch.arange(5, dtype=torch.float32))

    def test_from_arrow_sliced(self):
        table = _arrow_table()
        full = from_arrow(table)
        td = from_arrow(table.slice(1, 3))
        assert td.batch_size == torch.Size([3])
        for key in ("f", "u", "b", "fsl", "fst", ("st", "x")):
            assert torch.equal(td[key], full[key][1:4]), key
        assert td["f"].data_ptr() == full["f"].data_ptr() + 4
        for key in ("ragged", "ragged32", ("st", "z")):
            assert [x.tolist() for x in td[key].unbind(0)] == [
                x.tolist() for x in full[key].unbind(0)[1:4]
            ], key
        assert td.get("s").tolist() == ["bb", "ccc", "d"]

    def test_from_arrow_chunked(self):
        import pyarrow as pa

        table = pa.table(
            {
                "x": pa.chunked_array([[1.0, 2.0], [3.0]]),
                "ids": pa.chunked_array([[[1], [2, 3]], [[4, 5, 6]]]),
            }
        )
        td = from_arrow(table)
        assert torch.equal(td["x"], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert [x.tolist() for x in td["ids"].unbind(0)] == [[1], [2, 3], [4, 5, 6]]
        empty = from_arrow(pa.table({"x": pa.chunked_array([], type=pa.float32())}))
        assert empty.batch_size == torch.Size([0])
        assert empty["x"].shape == torch.Size([0])

    def test_from_arrow_inputs(self):
        import pyarrow as pa

        table = pa.table({"x": [1, 2, 3]})

        class Stream:
            def __arrow_c_stream__(self, requested_schema=None):
                return table.__arrow_c_stream__(requested_schema)

        for data in (table.to_batches()[0], Stream()):
            td = from_arrow(data, device="cpu")
            assert td.device == torch.device("cpu")
            assert torch.equal(td["x"], torch.tensor([1, 2, 3]))

    def test_from_arrow_permuted_tensor(self):
        array = np.arange(24).reshape(2, 3, 4).transpose(0, 2, 1)
        import pyarrow as pa

        column = pa.FixedShapeTensorArray.from_numpy_ndarray(array)
        assert column.type.permutation == [1, 0]
        td = from_arrow(pa.table({"t": column}))
        assert torch.equal(td["t"], torch.from_numpy(np.ascontiguousarray(array)))

    def _null_table(self):
        import pyarrow as pa

        return pa.table(
            {
                "f": pa.array([1.0, None, 3.0]),
                "i": pa.array([None, 2, 3]),
                "b": pa.array([True, None, True]),
                "ragged": pa.array([[1.0], None, [2.0, 3.0]]),
                "fsl": pa.array([[1, 2], None, [5, 6]], type=pa.list_(pa.int64(), 2)),
                "st": pa.array([{"x": 1, "y": 2.0}, None, {"x": 3, "y": None}]),
                "s": pa.array(["a", None, "c"]),
                "full": pa.array([1, 2, 3]),
            }
        )

    def test_from_arrow_nulls_raise(self):
        import pyarrow as pa

        with pytest.raises(ValueError, match=r"\('f',\) has 1 null value"):
            from_arrow(self._null_table())
        with pytest.raises(ValueError, match="nulls must be one of"):
            from_arrow(self._null_table(), nulls="drop")
        with pytest.raises(ValueError, match=r"\('l',\) has 1 null"):
            from_arrow(pa.table({"l": pa.array([[1.0, None], [2.0]])}))
        with pytest.raises(ValueError, match=r"\('st', 'x'\) has 1 null"):
            from_arrow(pa.table({"st": pa.array([{"x": 1}, None])}))

    @pytest.mark.parametrize("nulls", ["fill", "mask"])
    def test_from_arrow_nulls(self, nulls):
        import pyarrow as pa

        td = from_arrow(self._null_table(), nulls=nulls)
        nan = float("nan")
        torch.testing.assert_close(
            td["f"], torch.tensor([1.0, nan, 3.0], dtype=torch.float64), equal_nan=True
        )
        assert torch.equal(td["i"], torch.tensor([0, 2, 3]))
        assert torch.equal(td["b"], torch.tensor([True, False, True]))
        assert [x.tolist() for x in td["ragged"].unbind(0)] == [[1.0], [], [2.0, 3.0]]
        assert torch.equal(td["fsl"], torch.tensor([[1, 2], [0, 0], [5, 6]]))
        assert torch.equal(td["st", "x"], torch.tensor([1, 0, 3]))
        torch.testing.assert_close(
            td["st", "y"],
            torch.tensor([2.0, nan, nan], dtype=torch.float64),
            equal_nan=True,
        )
        assert td.get("s").tolist() == ["a", None, "c"]
        if nulls == "fill":
            assert "_mask" not in td.keys()
        else:
            assert set(td["_mask"].keys(include_nested=True, leaves_only=True)) == {
                "f",
                "i",
                "b",
                "ragged",
                "fsl",
                ("st", "x"),
                ("st", "y"),
                "s",
            }
            assert torch.equal(td["_mask", "f"], torch.tensor([True, False, True]))
            assert torch.equal(td["_mask", "i"], torch.tensor([False, True, True]))
            assert torch.equal(
                td["_mask", "st", "y"], torch.tensor([True, False, False])
            )
            with pytest.raises(ValueError, match="under the key '_mask'"):
                from_arrow(pa.table({"_mask": [1]}), nulls="mask")
        # Nulls inside a list are filled.
        td = from_arrow(pa.table({"l": pa.array([[1, None], [2]])}), nulls=nulls)
        assert [x.tolist() for x in td["l"].unbind(0)] == [[1, 0], [2]]

    def test_from_arrow_memory_map(self, tmp_path):
        # An in-place write into a view of a memory-mapped file kills the
        # process, so run it in a subprocess.
        import pyarrow as pa

        path = tmp_path / "table.arrow"
        table = pa.table(
            {"x": pa.array(np.arange(5)), "l": pa.array([[1], [2, 3]] * 2 + [[]])}
        )
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        code = (
            "import sys, pyarrow as pa\n"
            "from tensordict import from_arrow\n"
            f"table = pa.ipc.open_file(pa.memory_map({str(path)!r})).read_all()\n"
            "address = table.column('x').chunk(0).buffers()[1].address\n"
            "td = from_arrow(table, copy=False)\n"
            "assert td['x'].data_ptr() == address\n"
            "td = from_arrow(table)\n"
            "assert td['x'].data_ptr() != address\n"
            "td['x'].add_(1)\n"
            "td['l'].values().add_(1)\n"
            "assert table.column('x').to_pylist() == [0, 1, 2, 3, 4]\n"
            "assert table.column('l').to_pylist()[1] == [2, 3]\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_to_arrow_roundtrip(self):
        ragged = torch.nested.nested_tensor_from_jagged(
            torch.arange(5.0), torch.tensor([0, 1, 3, 5])
        )
        ragged_2d = torch.nested.nested_tensor_from_jagged(
            torch.arange(10.0).view(5, 2), torch.tensor([0, 2, 2, 5])
        )
        td = TensorDict(
            f=torch.arange(3, dtype=torch.float32),
            b=torch.tensor([True, False, True]),
            image=torch.arange(18).view(3, 2, 3),
            ragged=ragged,
            ragged_2d=ragged_2d,
            obs=TensorDict(x=torch.tensor([4, 5, 6]), batch_size=[3]),
            empty=TensorDict(batch_size=[3]),
            s=NonTensorStack("x", "y", "z"),
            batch_size=[3],
        )
        table = to_arrow(td)
        assert table.num_rows == 3
        assert table.column("f").chunk(0).buffers()[1].address == td["f"].data_ptr()
        assert str(table.schema.field("image").type).startswith(
            "extension<arrow.fixed_shape_tensor"
        )
        td2 = from_arrow(table)
        assert set(td2.keys(True, True)) == set(td.keys(True, True))
        for key in ("f", "b", "image", ("obs", "x")):
            assert torch.equal(td2[key], td[key]), key
            assert td2[key].dtype == td[key].dtype
        for key in ("ragged", "ragged_2d"):
            assert td2[key].shape[0] == 3
            for x, y in zip(td2[key].unbind(0), td[key].unbind(0)):
                assert torch.equal(x, y), key
        assert td2.get("s").tolist() == ["x", "y", "z"]

    def test_arrow_empty(self):
        table = _arrow_table().slice(0, 0)
        td = from_arrow(table)
        assert td.batch_size == torch.Size([0])
        for key in td.keys(True, True):
            assert td.get(key).shape[0] == 0, key
        assert td.get("s").batch_size == torch.Size([0])
        assert td["fst"].shape == torch.Size([0, 2, 3])
        table2 = to_arrow(td)
        assert table2.num_rows == 0
        assert table2.schema.field("fst").type == table.schema.field("fst").type

    def test_to_arrow_batch_dims(self):
        td = TensorDict(x=torch.zeros(2, 3), batch_size=[2, 3])
        with pytest.raises(ValueError, match="one batch dimension"):
            to_arrow(td)
        assert to_arrow(td.reshape(-1)).num_rows == 6


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


# The type-by-null matrix (#3234, #3238): a column of each Arrow type, with no
# nulls, with null rows, and with null values inside rows, through each reader
# and writer. The cases are in _tabular_cases.py; tabular_v0_15_2.json records
# what v0.15.2 returned for the readers and writers it had.
_V0_15_2 = json.loads(
    (pathlib.Path(__file__).parent / "tabular_v0_15_2.json").read_text()
)["runs"]


def _fst_as_tensor(dtype, shape):
    # #3109 reads a fixed_shape_tensor column without nulls as a tensor of its
    # shape and dtype, where v0.15.2 gave a flat int64 or float32 tensor.
    values = torch.arange(6 * 3, dtype=dtype).view(3, *shape)
    return {
        "c": {
            "kind": "tensor",
            "dtype": str(dtype),
            "shape": list(values.shape),
            "values": values.tolist(),
        }
    }


# The results that change on purpose since v0.15.2, with the change that made them.
_CHANGED_SINCE_V0_15_2 = {
    ("fixed_shape_tensor_float32-no_nulls", "from_parquet"): _fst_as_tensor(
        torch.float32, (2, 3)
    ),
    ("fixed_shape_tensor_int32-no_nulls", "from_parquet"): _fst_as_tensor(
        torch.int32, (3, 2)
    ),
}


def _v0_15_2_result(case_id, reader, *, any_pandas):
    """What v0.15.2 returned, or ``None`` if it was not recorded with this pandas major version."""
    if (case_id, reader) in _CHANGED_SINCE_V0_15_2:
        return _CHANGED_SINCE_V0_15_2[case_id, reader]
    major = (
        importlib.import_module("pandas").__version__.split(".")[0]
        if _has_pandas
        else None
    )
    runs = [_V0_15_2[major]] if major in _V0_15_2 else []
    if any_pandas:
        runs += [run for key, run in sorted(_V0_15_2.items()) if key != major]
    for run in runs:
        result = run["cases"].get(case_id, {}).get(reader)
        if result is not None and result != _tabular_cases.PYARROW_CANNOT:
            return result
    return None


def _plain_value(value):
    """A Python value of a leaf in a description, with nulls as ``None`` and bools apart from 1 and 0."""
    if isinstance(value, dict):
        if "tensor" in value or "ndarray" in value:
            return _plain_value(value["values"])
        return {k: _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    if isinstance(value, str):
        if value in ("nan", "None", "NaT"):
            return None
        match = re.fullmatch(r"np\.\w+\((.*)\)|Decimal\((.*)\)", value)
        if match and match[2] is not None:
            return decimal.Decimal(ast.literal_eval(match[2]))
        try:
            value = ast.literal_eval(match[1] if match else value)
        except (ValueError, SyntaxError):
            return value
        return value if isinstance(value, str) else _plain_value(value)
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    return value


def _flat(row):
    if isinstance(row, list):
        return [leaf for value in row for leaf in _flat(value)]
    return [row]


def _same_row(actual, expected):
    if expected is None and isinstance(actual, list):
        # A null row read as a row of nulls (NaN) is fine.
        return all(leaf is None for leaf in _flat(actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _same_row(actual[k], expected[k]) for k in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        actual, expected = _flat(actual), _flat(expected)
        return len(actual) == len(expected) and all(
            _same_row(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def _entry_rows(entry):
    """The rows of a described entry, or ``None`` if it does not hold one value per row."""
    kind = entry["kind"]
    if kind in ("tensor", "NonTensorStack"):
        return _plain_value(entry["values"])
    if kind == "jagged":
        return _plain_value(entry["rows"])
    if "entries" in entry:
        fields = {k: _entry_rows(v) for k, v in entry["entries"].items()}
        if any(rows is None for rows in fields.values()):
            return None
        return [dict(zip(fields, row)) for row in zip(*fields.values())]
    # NonTensorData: one value for all the rows.
    return None


def _holds_column(result, column):
    """Whether a result holds the rows of the Arrow column, with nulls as None or NaN."""
    if "c" not in result:
        return False
    rows = _entry_rows(result["c"])
    expected = _plain_value(column.to_pylist())
    return (
        rows is not None
        and len(rows) == len(expected)
        and all(_same_row(a, e) for a, e in zip(rows, expected))
    )


def _check_v0_15_2(case_id, reader, actual, column, *, any_pandas=False):
    # Where v0.15.2 was right, the result must not change. Where v0.15.2 raised
    # or lost the rows of the column, a result that holds them passes too, so
    # that fixing an old bug does not need a change of this test.
    if actual == _tabular_cases.PYARROW_CANNOT:
        pytest.skip("this pyarrow cannot write this column to Parquet and read it back")
    expected = _v0_15_2_result(case_id, reader, any_pandas=any_pandas)
    if expected is None:
        pytest.skip("no v0.15.2 result recorded with this version of pandas")
    if actual == expected:
        return
    if (case_id, reader) in _CHANGED_SINCE_V0_15_2 or _holds_column(expected, column):
        pytest.fail(
            f"{reader} of an Arrow column {case_id} gives\n  {actual}\n"
            f"where v0.15.2 gave\n  {expected}"
        )
    assert _holds_column(actual, column), (
        f"{reader} of an Arrow column {case_id} gives\n  {actual}\n"
        f"which neither is what v0.15.2 gave\n  {expected}\nnor holds the rows "
        f"of the column, {column.to_pylist()}"
    )


def _make_column(case_id):
    import pyarrow as pa

    column = _tabular_cases.make_column(pa, case_id)
    if column is None:
        pytest.skip("this pyarrow cannot build this column")
    return column


def _from_arrow_reference(pa, typ, values):
    """The description of the entry that from_arrow documents for a column of ``typ`` holding ``values``."""
    if pa.types.is_struct(typ):
        return {
            "kind": "TensorDict",
            "entries": {
                field.name: _from_arrow_reference(
                    pa,
                    field.type,
                    [None if row is None else row[field.name] for row in values],
                )
                for field in typ
            },
        }
    shape = ()
    leaf = typ
    if isinstance(typ, getattr(pa, "FixedShapeTensorType", ())):
        shape, leaf = tuple(typ.shape), typ.value_type
    elif pa.types.is_fixed_size_list(typ):
        shape, leaf = (typ.list_size,), typ.value_type
    jagged = pa.types.is_list(typ) or pa.types.is_large_list(typ)
    if jagged:
        leaf = typ.value_type
    if not (
        pa.types.is_integer(leaf)
        or pa.types.is_floating(leaf)
        or pa.types.is_boolean(leaf)
    ):
        return {"kind": "NonTensorStack", "values": _tabular_cases._plain(values)}
    dtype = torch.from_numpy(np.zeros(0, leaf.to_pandas_dtype())).dtype
    if dtype.is_floating_point:
        fill = float("nan")
    else:
        fill = False if dtype == torch.bool else 0

    def filled(row, row_shape):
        if row is None:
            return torch.full(row_shape, fill, dtype=dtype).tolist()
        return [fill if v is None else v for v in row]

    if jagged:
        rows = [[] if row is None else filled(row, ()) for row in values]
        return {
            "kind": "jagged",
            "dtype": str(dtype),
            "rows": _tabular_cases._tensor_values(rows),
        }
    if shape:
        rows = [filled(row, (int(np.prod(shape)),)) for row in values]
        tensor = torch.tensor(rows, dtype=dtype).view(len(values), *shape)
    else:
        tensor = torch.tensor([fill if v is None else v for v in values], dtype=dtype)
    return {
        "kind": "tensor",
        "dtype": str(dtype),
        "shape": list(tensor.shape),
        "values": _tabular_cases._tensor_values(tensor.tolist()),
    }


def _has_nulls(pa, arr):
    """Whether an Arrow array has a null row, or a null value inside a row."""
    if arr.null_count:
        return True
    if isinstance(arr, pa.ExtensionArray):
        return _has_nulls(pa, arr.storage)
    if pa.types.is_struct(arr.type):
        return any(_has_nulls(pa, child) for child in arr.flatten())
    if (
        pa.types.is_list(arr.type)
        or pa.types.is_large_list(arr.type)
        or pa.types.is_fixed_size_list(arr.type)
    ):
        return _has_nulls(pa, arr.flatten())
    return False


def _expected_masks(pa, arr, key):
    """The masks that from_arrow(nulls="mask") documents: one per entry with null rows, True where the row is not null."""
    if pa.types.is_struct(arr.type):
        # flatten() gives the fields the nulls of the struct.
        masks = {}
        for field, child in zip(arr.type, arr.flatten()):
            masks.update(_expected_masks(pa, child, (*key, field.name)))
        return masks
    if not arr.null_count:
        return {}
    return {key: arr.is_valid().to_pylist()}


def _skip_old_pyarrow_from_arrow():
    import pyarrow as pa

    if int(pa.__version__.split(".")[0]) < 15:
        pytest.skip("from_arrow needs pyarrow 15 or later (#3238)")


_COLUMN = "the_column"


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not found")
class TestTabularTypeMatrix:
    """Each Arrow type, with and without nulls, through each reader and writer (#3234, #3238)."""

    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_from_parquet(self, case_id, tmp_path):
        # Readers that v0.15.2 had give the same result as v0.15.2, apart from
        # the changes listed in _CHANGED_SINCE_V0_15_2.
        import pyarrow as pa
        import pyarrow.parquet as pq

        result = _tabular_cases.run_parquet(pa, pq, _make_column(case_id), tmp_path)
        _check_v0_15_2(
            case_id,
            "from_parquet",
            result["from_parquet"],
            _make_column(case_id),
            any_pandas=True,
        )

    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_to_parquet(self, case_id, tmp_path):
        # to_parquet of what from_parquet read, read back with from_parquet.
        import pyarrow as pa
        import pyarrow.parquet as pq

        result = _tabular_cases.run_parquet(pa, pq, _make_column(case_id), tmp_path)
        _check_v0_15_2(
            case_id, "to_parquet", result["to_parquet"], _make_column(case_id)
        )

    @pytest.mark.skipif(not _has_pandas, reason="pandas not found")
    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_from_pandas(self, case_id):
        # from_pandas of pyarrow.Table.to_pandas().
        import pyarrow as pa

        result = _tabular_cases.run_pandas(pa, _make_column(case_id))
        _check_v0_15_2(
            case_id, "from_pandas", result["from_pandas"], _make_column(case_id)
        )

    @pytest.mark.skipif(not _has_pandas, reason="pandas not found")
    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_to_pandas(self, case_id):
        # to_pandas of what from_pandas read, read back with from_pandas.
        import pyarrow as pa

        result = _tabular_cases.run_pandas(pa, _make_column(case_id))
        _check_v0_15_2(case_id, "to_pandas", result["to_pandas"], _make_column(case_id))

    @pytest.mark.parametrize("nulls", ["raise", "fill", "mask"])
    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_from_arrow(self, case_id, nulls):
        # from_arrow is new in 0.15.3: it gives what its docstring documents,
        # or a clear error that names the column, never wrong values.
        import pyarrow as pa

        _skip_old_pyarrow_from_arrow()
        column = _make_column(case_id)
        table = pa.table({_COLUMN: column})
        has_nulls = _has_nulls(pa, column)
        try:
            td = from_arrow(table, nulls=nulls)
        except Exception as err:
            assert _COLUMN in str(err), (
                f"from_arrow(nulls={nulls!r}) of an Arrow column {case_id} raises an "
                f"error that does not name the column: {type(err).__name__}: {err}"
            )
            if nulls == "raise" and has_nulls:
                assert isinstance(err, ValueError), (
                    f"from_arrow(nulls='raise') of an Arrow column {case_id} with "
                    f"nulls raises {type(err).__name__}, not ValueError: {err}"
                )
            return
        assert not (nulls == "raise" and has_nulls), (
            f"from_arrow(nulls='raise') of an Arrow column {case_id} with nulls "
            f"does not raise, and gives {_tabular_cases.describe(td)}"
        )
        expected = _from_arrow_reference(pa, column.type, column.to_pylist())
        actual = _tabular_cases.describe_entry(td.get(_COLUMN))
        assert actual == expected, (
            f"from_arrow(nulls={nulls!r}) of an Arrow column {case_id} gives\n"
            f"  {actual}\nwhere its docstring gives\n  {expected}"
        )
        masks = td.get("_mask", None)
        if nulls != "mask":
            assert masks is None, (
                f"from_arrow(nulls={nulls!r}) of an Arrow column {case_id} adds "
                f"the masks {masks}"
            )
            return
        expected = _expected_masks(pa, column, (_COLUMN,))
        actual = {
            (key,) if isinstance(key, str) else key: value.tolist()
            for key, value in ({} if masks is None else masks.items(True, True))
        }
        assert actual == expected, (
            f"from_arrow(nulls='mask') of an Arrow column {case_id} gives the masks "
            f"{actual}, where its docstring gives {expected}"
        )

    @pytest.mark.parametrize("case_id", _tabular_cases.CASE_IDS)
    def test_to_arrow(self, case_id):
        # to_arrow of what from_arrow read, read back with from_arrow, gives
        # the same entries, or to_arrow raises a clear error.
        import pyarrow as pa

        _skip_old_pyarrow_from_arrow()
        table = pa.table({_COLUMN: _make_column(case_id)})
        try:
            td = from_arrow(table, nulls="fill")
        except Exception:
            pytest.skip("from_arrow cannot read this column: see test_from_arrow")
        try:
            table2 = to_arrow(td)
        except Exception as err:
            assert _COLUMN in str(err), (
                f"to_arrow of the entry from_arrow read from an Arrow column "
                f"{case_id} raises an error that does not name it: "
                f"{type(err).__name__}: {err}"
            )
            return
        expected = _tabular_cases.describe(td)
        try:
            actual = _tabular_cases.describe(from_arrow(table2, nulls="fill"))
        except Exception as err:
            raise AssertionError(
                f"from_arrow cannot read back what to_arrow wrote for an Arrow column "
                f"{case_id}, of type {table2.schema.field(_COLUMN).type}: "
                f"{type(err).__name__}: {err}"
            ) from err
        assert actual == expected, (
            f"to_arrow then from_arrow of an Arrow column {case_id} gives\n"
            f"  {actual}\nwhere from_arrow first gave\n  {expected}"
        )


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not found")
class TestArrowMinimumVersion:
    """from_arrow needs pyarrow 15, while the parquet extra allows pyarrow 10 (#3238)."""

    @pytest.mark.parametrize("version", ["10.0.1", "14.0.2"])
    def test_from_arrow_old_pyarrow_raises_import_error(self, monkeypatch, version):
        import pyarrow as pa

        table = pa.table({"x": [1, 2]})
        monkeypatch.setattr(pa, "__version__", version)
        try:
            from_arrow(table)
        except ImportError as err:
            assert "15" in str(err) and "pyarrow" in str(err), (
                f"from_arrow on pyarrow {version} raises an ImportError that does "
                f"not name the minimum, pyarrow 15: {err}"
            )
        else:
            pytest.fail(
                f"from_arrow runs on pyarrow {version}, where it fails or gives "
                "wrong results; it must raise an ImportError that names the "
                "minimum, pyarrow 15 (#3238)"
            )

    def test_from_arrow_pyarrow_15_runs(self, monkeypatch):
        import pyarrow as pa

        _skip_old_pyarrow_from_arrow()
        table = pa.table({"x": [1, 2]})
        monkeypatch.setattr(pa, "__version__", "15.0.0")
        assert torch.equal(from_arrow(table)["x"], torch.tensor([1, 2]))

    def test_parquet_runs_on_pyarrow_10(self, monkeypatch, tmp_path):
        # The check of from_arrow must not reach from_parquet and to_parquet,
        # which the parquet extra allows from pyarrow 10.
        import pyarrow as pa

        monkeypatch.setattr(pa, "__version__", "10.0.1")
        td = TensorDict(x=torch.tensor([1.0, 2.0]), batch_size=[2])
        td.to_parquet(tmp_path / "x.parquet")
        assert torch.equal(
            TensorDict.from_parquet(tmp_path / "x.parquet")["x"], td["x"]
        )
