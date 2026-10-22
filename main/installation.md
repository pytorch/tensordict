# Installation

Install the latest release from PyPI:

```
pip install tensordict
```

tensordict needs Python 3.10+ and PyTorch 2.11+. To make sure that the
install worked, print the version:

```
python -c "import tensordict; print(tensordict.__version__)"
```

## Requirements

tensordict is pure Python: it has no compiled extension. Each change is
tested with PyTorch 2.11, the latest PyTorch release, and PyTorch nightly, on
Python 3.10 to 3.15 and the free-threaded 3.14t and 3.15t.

If PyTorch is not installed, `pip` installs the default build from PyPI
with tensordict. If you need a specific build, for example for CUDA or for a
CPU-only machine, install PyTorch first. The [PyTorch Get Started page](https://pytorch.org/get-started/locally/) gives the command for each
build.

The other runtime dependencies are `numpy`, `cloudpickle`, `pyvers`, and
`typing_extensions>=4.0` on Python 3.10. `pip` and `conda` install them
with tensordict.

## Optional dependencies

Some features need other packages. Install them with an extra, for example
`pip install "tensordict[h5,zarr]"`:

| Extra | Installs | For |
| --- | --- | --- |
| `h5` | `h5py>=3.9` | HDF5 files: [`to_h5()`](reference/generated/tensordict.TensorDictBase.to_h5.html#tensordict.TensorDictBase.to_h5), [`from_h5()`](reference/generated/tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5), and [`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). |
| `zarr` | `zarr>=3.0.8` | Zarr stores: [`to_zarr()`](reference/generated/tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr) and [`from_zarr()`](reference/generated/tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr). zarr 3 needs Python 3.11+. On Python 3.10, this extra installs nothing. |
| `pandas` | `pandas>=1.5` | DataFrames and CSV files: [`to_pandas()`](reference/generated/tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas), [`from_pandas()`](reference/generated/tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas), [`to_csv()`](reference/generated/tensordict.TensorDictBase.to_csv.html#tensordict.TensorDictBase.to_csv), and [`from_csv()`](reference/generated/tensordict.TensorDictBase.from_csv.html#tensordict.TensorDictBase.from_csv). |
| `parquet` | `pyarrow>=10.0` | Parquet files: [`to_parquet()`](reference/generated/tensordict.TensorDictBase.to_parquet.html#tensordict.TensorDictBase.to_parquet) and [`from_parquet()`](reference/generated/tensordict.TensorDictBase.from_parquet.html#tensordict.TensorDictBase.from_parquet). These methods use pyarrow, or pandas if pyarrow is not installed. Arrow tables: [`from_arrow()`](reference/generated/tensordict.from_arrow.html#tensordict.from_arrow) and [`to_arrow()`](reference/generated/tensordict.to_arrow.html#tensordict.to_arrow), which need pyarrow 15 or later. |
| `tabular` | `pandas>=1.5` and `pyarrow>=10.0` | The `pandas` and `parquet` extras together. |
| `safetensors` | `safetensors>=0.4.5` | safetensors files: [`to_safetensors()`](reference/generated/tensordict.to_safetensors.html#tensordict.to_safetensors) and [`from_safetensors()`](reference/generated/tensordict.from_safetensors.html#tensordict.from_safetensors). |
| `onnx` | `onnx`, `onnxscript`, and `onnxruntime` | Export of a [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) to ONNX with [`torch.onnx.export()`](https://docs.pytorch.org/docs/stable/onnx_export.html#torch.onnx.export), and running the result. |

[Large datasets and storage backends](storage.html#storage) explains the HDF5, Zarr, and tabular formats.

## Install a nightly build

The `tensordict-nightly` package is built from the main branch every day.
It installs the same `tensordict` directory as the release. If both
packages are installed, the one installed last overwrites the files of the
other. Uninstalling one of them then deletes files that the other needs. So
uninstall the release first:

```
pip uninstall tensordict
pip install tensordict-nightly
```

If both are already installed, uninstall both, then install one of them
again.

A nightly build of tensordict can use features that only PyTorch nightly
has, so install the two together. Recent PyTorch nightly builds need Python
3.11+. This example installs the CPU build of PyTorch nightly:

```
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install tensordict-nightly
```

## Install with conda

The conda-forge channel has the releases:

```
conda install -c conda-forge tensordict
```

## Install from source

Install from source to contribute to tensordict, or to use a change that is
not in a nightly build yet. The build needs no compiler. `pip` installs the
build requirements, `setuptools>=77` and `setuptools_scm`, unless you
pass `--no-build-isolation`.

```
git clone https://github.com/pytorch/tensordict
cd tensordict
pip install -e .
```

`-e` installs tensordict in editable mode: Python imports it from your
clone, so your changes apply without a new install.

If you do not want `pip` to change your PyTorch installation, pass
`--no-deps`. `--no-deps` also skips the other runtime dependencies, so
install them first:

```
pip install numpy cloudpickle "pyvers>=0.2,<0.3" "typing_extensions>=4.0"
pip install -e . --no-deps
```

Note

If you use `uv` with a PyTorch nightly from the PyTorch nightly wheel
index, `uv` can replace the nightly `torch` with a release from PyPI
when it resolves the dependencies of tensordict. To keep your PyTorch
build, use one of these commands:

```
# Option 1: do not resolve dependencies (install the runtime dependencies first)
uv pip install -e . --no-deps
# Option 2: resolve against the PyTorch nightly index
uv pip install -e . --prerelease=allow -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
```

## tensordict and TorchRL versions

Each [TorchRL](https://github.com/pytorch/rl) release requires the
tensordict release with the same minor version. For example, TorchRL 0.14
requires tensordict 0.14, and does not accept 0.15. The nightly package
`torchrl-nightly` requires `tensordict-nightly`.

To get a matching pair, install TorchRL and let `pip` choose tensordict:

```
pip install torchrl
```

Do not mix a nightly build of one library with a release of the other. A
mismatch can show as an `ImportError: cannot import name ...` when you
import `torchrl`.

## Next steps

[Getting started](getting_started.html#getting-started) shows the main features of tensordict in one hour.