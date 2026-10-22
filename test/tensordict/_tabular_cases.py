# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Arrow columns of each type, with and without nulls, for the tabular matrix tests in test_tabular.py.

This module imports only ``TensorDict`` from tensordict, so that it also runs
with older releases. Run it as a script to record what a release returns:

    git archive v0.15.2 tensordict | tar -x -C <dir>
    PYTHONPATH=<dir> python test/tensordict/_tabular_cases.py \
        test/tensordict/tabular_v0_15_2.json v0.15.2

with the CI variables of ``.github/unittest/linux/scripts/run_test.sh`` set,
once with pandas 2 and once with pandas 3: the results of objects in pandas
columns depend on its major version. tabular_v0_15_2.json was recorded with
pandas 2.3.3, pyarrow 25.0.1, numpy 1.26.4 on Python 3.10, and with pandas
3.0.6, pyarrow 26.0.0, numpy 2.5.4 on Python 3.12.
"""

from __future__ import annotations

import decimal
import json
import math
import sys
import warnings

import numpy as np
import torch
from tensordict import TensorDict

N_ROWS = 3

# The three variants of each type: no nulls, null rows, and null values inside
# rows (for the types whose rows hold more than one value).
VARIANTS = ("no_nulls", "null_rows", "null_values")


def _values(name, variant):
    """The Python values of column ``name`` in ``variant``, or ``None`` if it does not apply."""
    scalar = {
        "int": [1, 2, 3],
        "float": [1.5, 2.5, 3.5],
        "bool": [True, False, True],
        "string": ["a", "bb", "<ccc>"],
        "decimal": [decimal.Decimal(v) for v in ("1.10", "2.20", "3.30")],
        "timestamp": [1_700_000_000_000_000 + i for i in range(N_ROWS)],
    }
    rows = {
        "list_int": [[1, 2], [3], []],
        "list_float": [[1.5, 2.5], [3.5], []],
        "list_string": [["a"], ["b", "c"], []],
        "fsl_int": [[1, 2], [3, 4], [5, 6]],
        "fsl_float": [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]],
        "fst": [list(range(6 * i, 6 * i + 6)) for i in range(N_ROWS)],
        "fst_bool": [[True, False, True, False]] * N_ROWS,
        "struct": [
            {"x": 1, "y": 1.5, "s": "a"},
            {"x": 2, "y": 2.5, "s": "b"},
            {"x": 3, "y": 3.5, "s": "c"},
        ],
    }
    if name in scalar:
        if variant == "null_values":
            return None
        values = list(scalar[name])
        if variant == "null_rows":
            values[1] = None
        return values
    values = [list(v) if isinstance(v, list) else dict(v) for v in rows[name]]
    if variant == "null_rows":
        values[1] = None
    elif variant == "null_values":
        if name == "struct":
            values[0]["y"] = None
            values[1]["s"] = None
        else:
            values[0][0] = None
    return values


def _fst(value_type, shape):
    def build(pa, values):
        if not hasattr(pa, "fixed_shape_tensor"):
            return None
        typ = pa.fixed_shape_tensor(getattr(pa, value_type)(), shape)
        return pa.ExtensionArray.from_storage(typ, pa.array(values, typ.storage_type))

    return build


def _simple(make_type):
    return lambda pa, values: pa.array(values, type=make_type(pa))


def _half(pa, values):
    array = np.array([np.nan if v is None else v for v in values], np.float16)
    return pa.array(array, mask=np.array([v is None for v in values]))


def _dictionary(pa, values):
    return pa.array(values, type=pa.string()).dictionary_encode()


# Maps each type to the name of its values in ``_values``, and a function of
# ``(pyarrow, values)`` that builds the column, or returns ``None`` if this
# version of pyarrow cannot.
TYPES = {
    **{
        name: ("int", _simple(lambda pa, name=name: getattr(pa, name)()))
        for name in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        )
    },
    "float16": ("float", _half),
    "float32": ("float", _simple(lambda pa: pa.float32())),
    "float64": ("float", _simple(lambda pa: pa.float64())),
    "bool": ("bool", _simple(lambda pa: pa.bool_())),
    "string": ("string", _simple(lambda pa: pa.string())),
    "large_string": ("string", _simple(lambda pa: pa.large_string())),
    "dictionary": ("string", _dictionary),
    "timestamp_us": ("timestamp", _simple(lambda pa: pa.timestamp("us"))),
    "timestamp_us_utc": (
        "timestamp",
        _simple(lambda pa: pa.timestamp("us", tz="UTC")),
    ),
    "decimal128": ("decimal", _simple(lambda pa: pa.decimal128(5, 2))),
    "list_int64": ("list_int", _simple(lambda pa: pa.list_(pa.int64()))),
    "list_float32": ("list_float", _simple(lambda pa: pa.list_(pa.float32()))),
    "large_list_float64": (
        "list_float",
        _simple(lambda pa: pa.large_list(pa.float64())),
    ),
    "list_string": ("list_string", _simple(lambda pa: pa.list_(pa.string()))),
    "fixed_size_list_int64": (
        "fsl_int",
        _simple(lambda pa: pa.list_(pa.int64(), 2)),
    ),
    "fixed_size_list_float32": (
        "fsl_float",
        _simple(lambda pa: pa.list_(pa.float32(), 2)),
    ),
    "fixed_shape_tensor_float32": ("fst", _fst("float32", [2, 3])),
    "fixed_shape_tensor_int32": ("fst", _fst("int32", [3, 2])),
    "fixed_shape_tensor_bool": ("fst_bool", _fst("bool_", [2, 2])),
    "struct": (
        "struct",
        _simple(
            lambda pa: pa.struct(
                [("x", pa.int64()), ("y", pa.float64()), ("s", pa.string())]
            )
        ),
    ),
}

# The ids "<type>-<variant>" of all the cases that apply.
CASE_IDS = [
    f"{name}-{variant}"
    for name, (values_name, _) in TYPES.items()
    for variant in VARIANTS
    if _values(values_name, variant) is not None
]


def make_column(pa, case_id):
    """The pyarrow.Array of a case, or ``None`` if this pyarrow cannot build it."""
    name, variant = case_id.rsplit("-", 1)
    values_name, build = TYPES[name]
    return build(pa, _values(values_name, variant))


def _scalar(value):
    """A JSON value that tells apart types that compare equal, such as 1 and 1.0."""
    if isinstance(value, np.generic):
        return f"np.{value.dtype}({value.item()!r})"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return repr(value)


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return {"ndarray": str(value.dtype), "values": _plain(value.tolist())}
    if isinstance(value, torch.Tensor):
        return {"tensor": str(value.dtype), "values": _tensor_values(value.tolist())}
    return _scalar(value)


def _tensor_values(value):
    if isinstance(value, list):
        return [_tensor_values(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def describe_entry(value):
    """A JSON description of one entry of a tensordict: its type, dtype, shape and values."""
    from tensordict import is_tensor_collection

    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return {
                "kind": "jagged",
                "dtype": str(value.dtype),
                "rows": [_tensor_values(row.tolist()) for row in value.unbind(0)],
            }
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _tensor_values(value.tolist()),
        }
    name = type(value).__name__
    if name == "NonTensorStack":
        return {"kind": name, "values": _plain(value.tolist())}
    if name == "NonTensorData":
        return {"kind": name, "value": _plain(value.data)}
    if is_tensor_collection(value):
        return {"kind": name, "entries": describe(value)}
    return {"kind": name, "value": _plain(value)}


def describe(td):
    """A JSON description of a tensordict, entry by entry."""
    return {key: describe_entry(td.get(key)) for key in td.keys()}


def _error(err):
    return {"error": type(err).__name__}


# The result of a case that this pyarrow cannot write to Parquet and read back
# by itself, before tensordict sees the data (pyarrow 25 cannot read back a
# fixed_size_list column with a null row, for example).
PYARROW_CANNOT = {"pyarrow_cannot": "write and read back this column"}


def run_parquet(pa, pq, column, tmp_path):
    """Describes ``from_parquet``, and ``to_parquet`` followed by ``from_parquet``."""
    path = tmp_path / "in.parquet"
    try:
        pq.write_table(pa.table({"c": column}), str(path))
        pq.read_table(str(path))
    except Exception:
        return {"from_parquet": PYARROW_CANNOT, "to_parquet": PYARROW_CANNOT}
    try:
        td = TensorDict.from_parquet(path)
    except Exception as err:
        return {"from_parquet": _error(err), "to_parquet": _error(err)}
    result = {"from_parquet": describe(td)}
    try:
        td.to_parquet(tmp_path / "out.parquet")
        result["to_parquet"] = describe(
            TensorDict.from_parquet(tmp_path / "out.parquet")
        )
    except Exception as err:
        result["to_parquet"] = _error(err)
    return result


def run_pandas(pa, column):
    """Describes ``from_pandas`` of ``to_pandas()`` of the table, and ``to_pandas`` followed by ``from_pandas``."""
    try:
        df = pa.table({"c": column}).to_pandas()
    except Exception as err:
        return {"from_pandas": _error(err), "to_pandas": _error(err)}
    result = {}
    try:
        td = TensorDict.from_pandas(df)
    except Exception as err:
        return {"from_pandas": _error(err), "to_pandas": _error(err)}
    result["from_pandas"] = describe(td)
    try:
        result["to_pandas"] = describe(TensorDict.from_pandas(td.to_pandas()))
    except Exception as err:
        result["to_pandas"] = _error(err)
    return result


def _dump(record):
    """JSON text with one line per case, so that a diff shows which cases changed."""
    lines = ["{", f' "tensordict": {json.dumps(record["tensordict"])},', ' "runs": {']
    runs = sorted(record["runs"].items())
    for i, (major, run) in enumerate(runs):
        lines.append(f"  {json.dumps(major)}: {{")
        lines.append(f'   "versions": {json.dumps(run["versions"], sort_keys=True)},')
        lines.append('   "cases": {')
        cases = sorted(run["cases"].items())
        for j, (case_id, result) in enumerate(cases):
            comma = "," if j < len(cases) - 1 else ""
            lines.append(
                f"    {json.dumps(case_id)}: {json.dumps(result, sort_keys=True)}{comma}"
            )
        lines.append("   }")
        lines.append("  }" + ("," if i < len(runs) - 1 else ""))
    lines += [" }", "}", ""]
    return "\n".join(lines)


def _main(out, label):
    """Records the results of this environment in ``out``, next to those of other pandas major versions."""
    import pathlib
    import tempfile

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    warnings.simplefilter("ignore")
    out = pathlib.Path(out)
    record = (
        json.loads(out.read_text())
        if out.exists()
        else {"tensordict": label, "runs": {}}
    )
    assert record["tensordict"] == label, record["tensordict"]
    cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case_id in CASE_IDS:
            column = make_column(pa, case_id)
            if column is None:
                continue
            path = pathlib.Path(tmp) / case_id
            path.mkdir()
            cases[case_id] = {
                **run_parquet(pa, pq, column, path),
                **run_pandas(pa, column),
            }
    record["runs"][pd.__version__.split(".")[0]] = {
        "versions": {
            "pyarrow": pa.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "cases": cases,
    }
    out.write_text(_dump(record))


if __name__ == "__main__":
    _main(*sys.argv[1:3])
