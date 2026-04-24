"""Tabular data import/export for TensorDict.

Supports conversion between TensorDict and tabular formats:
pandas DataFrames, CSV, Parquet, and JSON files.

All external dependencies (pandas, pyarrow) are optional.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

_has_pandas = importlib.util.find_spec("pandas") is not None
_has_pyarrow = importlib.util.find_spec("pyarrow") is not None


def _unflatten_columns(flat_dict: dict, separator: str) -> dict:
    """Convert flat column dict to nested dict via separator splitting.

    {"obs.x": arr, "obs.y": arr, "action": arr}
    -> {"obs": {"x": arr, "y": arr}, "action": arr}
    """
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(
                    f"Key conflict: '{part}' is both a leaf and a prefix in column names"
                )
            current = current[part]
        current[parts[-1]] = value
    return result


def _flatten_keys(td, separator: str) -> dict[str, Any]:
    """Flatten a TensorDict into a dict with separated key names."""
    from tensordict.base import _is_tensor_collection, is_non_tensor

    result = {}
    for key, value in td.items():
        if _is_tensor_collection(type(value)) and not is_non_tensor(value):
            sub = _flatten_keys(value, separator)
            for sub_key, sub_val in sub.items():
                result[f"{key}{separator}{sub_key}"] = sub_val
        else:
            result[key] = value
    return result


def _columns_to_tensordict(
    columns: dict[str, np.ndarray | list],
    *,
    cls,
    device: torch.device | None,
    batch_size: torch.Size | None,
    separator: str | None,
    dtype: torch.dtype | None,
    num_rows: int,
):
    """Convert a dict of column arrays into a TensorDict."""
    data = {}
    for col_name, arr in columns.items():
        if isinstance(arr, np.ndarray) and arr.dtype.kind in ("i", "f", "u", "b"):
            if dtype is not None:
                data[col_name] = torch.as_tensor(arr.copy()).to(dtype)
            else:
                data[col_name] = arr.copy()
        elif isinstance(arr, np.ndarray) and arr.dtype.kind == "M":
            data[col_name] = arr.astype("datetime64[ns]").astype(np.int64)
        else:
            data[col_name] = arr

    if separator is not None:
        data = _unflatten_columns(data, separator)

    if batch_size is None:
        batch_size = torch.Size([num_rows])

    return cls(data, batch_size=batch_size, device=device)


def _dataframe_to_tensordict(
    dataframe,
    *,
    cls,
    device: torch.device | None,
    batch_size: torch.Size | None,
    separator: str | None,
    dtype: torch.dtype | None,
):
    """Convert a pandas DataFrame to a TensorDict."""
    num_rows = len(dataframe)
    columns = {}

    for col in dataframe.columns:
        series = dataframe[col]
        if hasattr(series, "cat") and series.cat is not None:
            columns[str(col)] = series.cat.codes.to_numpy().copy()
        else:
            arr = series.to_numpy()
            columns[str(col)] = arr

    return _columns_to_tensordict(
        columns,
        cls=cls,
        device=device,
        batch_size=batch_size,
        separator=separator,
        dtype=dtype,
        num_rows=num_rows,
    )


def _tensordict_to_dataframe(td, *, separator: str | None):
    """Convert a TensorDict to a pandas DataFrame."""
    import pandas as pd

    from tensordict.base import _is_tensor_collection, is_non_tensor

    if separator is not None:
        flat = _flatten_keys(td, separator)
    else:
        flat = {}
        for key, value in td.items():
            if _is_tensor_collection(type(value)) and not is_non_tensor(value):
                raise ValueError(
                    f"Nested TensorDict at key '{key}' requires a separator parameter "
                    "to flatten to DataFrame columns. Use to_pandas(separator='.')."
                )
            flat[key] = value

    data = {}
    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            if value.ndim > 1:
                raise ValueError(
                    f"Cannot convert tensor with shape {value.shape} at key '{key}' "
                    f"to a single DataFrame column. Only 1D tensors (matching the "
                    f"batch size) are supported."
                )
            data[key] = value.detach().cpu().numpy()
        elif is_non_tensor(value):
            if hasattr(value, "tolist"):
                data[key] = value.tolist()
            else:
                data[key] = [value.data] * len(td)
        elif isinstance(value, np.ndarray):
            data[key] = value
        else:
            data[key] = value

    return pd.DataFrame(data)


def _pyarrow_table_to_columns(table) -> tuple[dict[str, np.ndarray | list], int]:
    """Convert a pyarrow Table to a dict of numpy arrays / lists."""
    import pyarrow as pa

    columns = {}
    for col_name in table.column_names:
        column = table.column(col_name)
        if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
            columns[col_name] = column.to_numpy(zero_copy_only=False)
        elif pa.types.is_boolean(column.type):
            columns[col_name] = column.to_numpy(zero_copy_only=False)
        elif pa.types.is_timestamp(column.type):
            columns[col_name] = column.to_numpy(zero_copy_only=False)
        else:
            columns[col_name] = column.to_pylist()

    return columns, table.num_rows


def _read_csv(path, **kwargs) -> tuple[dict[str, np.ndarray | list], int]:
    """Read a CSV file and return a column dict."""
    if _has_pandas:
        import pandas as pd

        df = pd.read_csv(path, **kwargs)
        columns = {}
        for col in df.columns:
            series = df[col]
            if hasattr(series, "cat") and series.cat is not None:
                columns[str(col)] = series.cat.codes.to_numpy().copy()
            else:
                columns[str(col)] = series.to_numpy()
        return columns, len(df)
    elif _has_pyarrow:
        import pyarrow.csv as pcsv

        table = pcsv.read_csv(str(path), **kwargs)
        return _pyarrow_table_to_columns(table)
    else:
        raise ImportError(
            "Either pandas or pyarrow is required for from_csv. "
            "Install with: pip install pandas  or  pip install pyarrow"
        )


def _read_parquet(
    path, columns: list[str] | None = None, **kwargs
) -> tuple[dict[str, np.ndarray | list], int]:
    """Read a Parquet file and return a column dict."""
    if _has_pyarrow:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path), columns=columns, **kwargs)
        return _pyarrow_table_to_columns(table)
    elif _has_pandas:
        import pandas as pd

        df = pd.read_parquet(path, columns=columns, **kwargs)
        col_dict = {}
        for col in df.columns:
            col_dict[str(col)] = df[col].to_numpy()
        return col_dict, len(df)
    else:
        raise ImportError(
            "Either pyarrow or pandas is required for from_parquet. "
            "Install with: pip install pyarrow  or  pip install pandas"
        )


def _read_json(path, lines: bool = False, **kwargs) -> tuple[dict[str, np.ndarray | list], int]:
    """Read a JSON file and return a column dict."""
    if _has_pandas:
        import pandas as pd

        df = pd.read_json(path, lines=lines, **kwargs)
        columns = {}
        for col in df.columns:
            columns[str(col)] = df[col].to_numpy()
        return columns, len(df)
    else:
        import json
        from pathlib import Path

        text = Path(path).read_text()
        if lines:
            records = [json.loads(line) for line in text.strip().splitlines() if line.strip()]
        else:
            data = json.loads(text)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError(
                    "JSON file must contain an array of records for tabular import. "
                    "For nested JSON objects, use from_dict instead."
                )

        if not records:
            return {}, 0

        all_keys = list(records[0].keys())
        columns = {key: [r.get(key) for r in records] for key in all_keys}

        for key, vals in columns.items():
            try:
                arr = np.array(vals)
                if arr.dtype.kind in ("i", "f", "u", "b"):
                    columns[key] = arr
            except (ValueError, TypeError):
                pass

        return columns, len(records)


def _write_csv(td, path, separator: str | None, **kwargs):
    """Write a TensorDict to a CSV file."""
    import pandas as pd

    df = _tensordict_to_dataframe(td, separator=separator)
    df.to_csv(path, index=False, **kwargs)


def _write_parquet(td, path, separator: str | None, **kwargs):
    """Write a TensorDict to a Parquet file."""
    if _has_pyarrow:
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = _tensordict_to_dataframe(td, separator=separator)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path), **kwargs)
    elif _has_pandas:
        import pandas as pd

        df = _tensordict_to_dataframe(td, separator=separator)
        df.to_parquet(path, **kwargs)
    else:
        raise ImportError(
            "Either pyarrow or pandas is required for to_parquet. "
            "Install with: pip install pyarrow  or  pip install pandas"
        )


def _write_json(td, path, separator: str | None, lines: bool = False, **kwargs):
    """Write a TensorDict to a JSON file."""
    if _has_pandas:
        import pandas as pd

        df = _tensordict_to_dataframe(td, separator=separator)
        df.to_json(path, orient="records", lines=lines, **kwargs)
    else:
        import json

        from tensordict.base import _is_tensor_collection, is_non_tensor

        if separator is not None:
            flat = _flatten_keys(td, separator)
        else:
            flat = dict(td.items())

        records = []
        n = len(td)
        for i in range(n):
            record = {}
            for key, value in flat.items():
                if isinstance(value, torch.Tensor):
                    record[key] = value[i].item()
                elif is_non_tensor(value):
                    if hasattr(value, "tolist"):
                        record[key] = value.tolist()[i] if hasattr(value.tolist(), "__getitem__") else value.data
                    else:
                        record[key] = value.data
                else:
                    record[key] = value[i] if hasattr(value, "__getitem__") else value
            records.append(record)

        text = json.dumps(records, default=str)
        if lines:
            text = "\n".join(json.dumps(r, default=str) for r in records)

        Path(path).write_text(text)
