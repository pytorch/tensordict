"""
Storing TensorDicts in zarr
===========================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this tutorial you will learn how to store tensordicts in `zarr
<https://zarr.dev>`_ stores, when to prefer zarr over the memory-mapped
and HDF5 backends, and how to tune chunking and compression for your
workload.

**What you will learn:**

- saving and loading tensordicts with
  :meth:`~tensordict.TensorDictBase.to_zarr` and
  :meth:`~tensordict.TensorDictBase.from_zarr`;
- reading slices of a store without loading it in memory;
- chunking and compression;
- packing a store into a single zip file;
- pre-allocating a zarr-backed buffer with
  :meth:`~tensordict.TensorDictBase.from_schema`.

The zarr backend requires ``zarr>=3.0``, installable with
``pip install "tensordict[zarr]"``.
"""

##############################################################################
# Why zarr?
# ---------
# zarr is a chunked, compressed array format designed for parallel and
# cloud-native workloads. Where the memory-mapped format (see the
# :ref:`saving tutorial <saving>`) is the fastest choice for local
# checkpoints, zarr shines when the data lives somewhere else: the same
# hierarchy of groups and arrays can be written to a local directory, a
# single zip file, or an S3-style object store, and every array is split
# into chunks that can be read, written and compressed independently.
# It is also the native storage of the scientific Python stack (xarray,
# dask), so a tensordict saved as zarr can be opened by those tools and
# vice versa.

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

root = Path(tempfile.mkdtemp())

##############################################################################
# Saving and loading
# ------------------
# :meth:`~tensordict.TensorDictBase.to_zarr` writes the tensordict to a
# store and returns a :class:`~tensordict.PersistentTensorDict` pointing at
# it, exactly like :meth:`~tensordict.TensorDictBase.to_h5` does for HDF5.

td = TensorDict(
    {
        "images": torch.randn(128, 3, 32, 32),
        "labels": torch.randint(10, (128,)),
        "meta": {"description": "a small dataset"},
    },
    batch_size=[128],
)
td_zarr = td.to_zarr(root / "dataset.zarr")
td_zarr

##############################################################################
# The batch size, dimension names and non-tensor entries (like the
# ``"description"`` string above) are all persisted.
# :meth:`~tensordict.TensorDictBase.from_zarr` restores them without any
# inference:

td_back = TensorDict.from_zarr(root / "dataset.zarr")
assert td_back.batch_size == torch.Size([128])
assert td_back["meta", "description"] == "a small dataset"

##############################################################################
# Stores written by other tools (xarray, raw zarr) lack the tensordict
# metadata; in that case ``from_zarr`` infers the batch size from the
# leading dimensions shared by all arrays, like ``from_h5`` does.
#
# Lazy access
# -----------
# ``from_zarr`` does not read any tensor data: entries are fetched from the
# store when accessed. :meth:`~tensordict.TensorDictBase.get_at` reads a
# sub-region of an entry, which is the idiomatic way of sampling from a
# dataset that does not fit in memory:

row = td_back.get_at("images", 3)
batch = td_back.get_at("images", slice(0, 16))
picks = td_back.get_at("images", torch.tensor([1, 17, 121]))

##############################################################################
# Chunking
# --------
# By default, tensordict writes each tensor as a single uncompressed chunk:
# for checkpoint-style save/load this is the fastest layout, and it makes
# the timings directly comparable with the other formats (see the
# :ref:`serialization speed tutorial <sphx_glr_tutorials_serialization_speed.py>`).
# A single chunk, however, means ``get_at`` must read the whole array. For
# dataset-style access, pass ``chunks=`` (forwarded to
# :meth:`zarr.Group.create_array`) so that a row read only touches the
# chunks it needs:

td.to_zarr(root / "chunked.zarr", chunks=(16, 3, 32, 32))
td_chunked = TensorDict.from_zarr(root / "chunked.zarr")
# this now reads a single 16-image chunk from disk:
row = td_chunked.get_at("images", 3)

##############################################################################
# Compression
# -----------
# Any zarr codec can be passed through ``compressors=``. Compression is
# per-chunk: combined with chunking, it gives compact storage that still
# supports partial reads.

from zarr.codecs import ZstdCodec

td.to_zarr(root / "compressed.zarr", chunks=(16, 3, 32, 32), compressors=ZstdCodec())

##############################################################################
# Alternative stores
# ------------------
# The ``filename`` argument accepts any ``zarr.abc.store.Store``. A
# ``ZipStore`` packs the whole hierarchy into one file -- convenient for
# shipping a dataset around (zarr zip stores are read-mostly: prefer
# writing once and reading many times):

from zarr.storage import ZipStore

store = ZipStore(root / "dataset.zarr.zip", mode="w")
td.to_zarr(store).close()

td_zip = TensorDict.from_zarr(ZipStore(root / "dataset.zarr.zip", mode="r"))
assert (td_zip["labels"] == td["labels"]).all()
td_zip.close()

##############################################################################
# Remote object stores work the same way through zarr's fsspec integration:
# ``zarr.storage.FsspecStore.from_url("s3://bucket/dataset.zarr")`` can be
# passed as ``filename`` to write directly to S3-compatible storage.
#
# Pre-allocation
# --------------
# Like the other backends, zarr supports the
# :meth:`~tensordict.TensorDictBase.from_schema` pattern: declare the
# layout up-front, then fill the buffer row by row without ever holding the
# full dataset in memory.

buffer = TensorDict.from_schema(
    {"obs": ([4], torch.float32), "reward": ([], torch.float32)},
    batch_size=[1000],
    storage="zarr",
    filename=root / "buffer.zarr",
)
for i in range(4):
    buffer[i] = TensorDict(obs=torch.randn(4), reward=torch.randn(()), batch_size=[])

##############################################################################
# Conclusion
# ----------
# The zarr backend rounds out tensordict's storage options with a chunked,
# compressed, cloud-friendly format behind the familiar
# :class:`~tensordict.PersistentTensorDict` interface. Use
# ``to_zarr``/``from_zarr`` when your data needs to live on object storage,
# interoperate with xarray/dask, or benefit from per-chunk compression;
# stick to the memory-mapped format for the fastest local checkpoints.
#
# Further reading
# ---------------
# - The :ref:`storage page <storage>` compares all backends and details the
#   zarr-specific keyword arguments.
# - The :ref:`serialization speed tutorial
#   <sphx_glr_tutorials_serialization_speed.py>` benchmarks zarr against
#   the other on-disk formats.
# - The `zarr documentation <https://zarr.readthedocs.io>`_ covers stores,
#   codecs and the v3 format in depth.
