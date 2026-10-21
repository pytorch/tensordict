# Tutorials

Each tutorial solves one task with a Python script in `sphinx_tuto/`. The
documentation builds each script into a page and runs it, so the output on
the page comes from the code:
[tutorials on docs.pytorch.org](https://docs.pytorch.org/tensordict/stable/).
Each page also has links to download the script and a Jupyter notebook.

If you are new to tensordict, read
[Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html)
first. Then read the tutorials in this order.

## Basics

1. [Getting, setting, and nesting keys](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_keys.html)
   (`tensordict_keys.py`): set, get, iterate over, and flatten keys, nested keys included.
2. [Slicing, indexing, and masking](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_slicing.html)
   (`tensordict_slicing.py`): index a TensorDict along its batch dimensions.
3. [Changing the shape of a TensorDict](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_shapes.html)
   (`tensordict_shapes.py`): the batch size, reshape, stack, and concatenate.
4. [Filling a TensorDict row by row](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_preallocation.html)
   (`tensordict_preallocation.py`): the first write allocates every tensor.
5. [Building tensordicts from streams](https://docs.pytorch.org/tensordict/stable/tutorials/streamed_tensordict.html)
   (`streamed_tensordict.py`): group a stream of data of different sizes into batches.

## Storage

6. [Devices and memory-mapped tensors](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_memory.html)
   (`tensordict_memory.py`): move a TensorDict to a device, and store it in memory-mapped files.
7. [Storing TensorDicts in zarr](https://docs.pytorch.org/tensordict/stable/tutorials/zarr_storage.html)
   (`zarr_storage.py`): Zarr stores, chunks, and compression.
8. [Benchmarking TensorDict serialization speed](https://docs.pytorch.org/tensordict/stable/tutorials/serialization_speed.html)
   (`serialization_speed.py`): `memmap`, `torch.save`, safetensors, and Zarr, compared.

## tensordict.nn

9. [Building models with TensorDictModule](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_module.html)
   (`tensordict_module.py`): modules that read and write TensorDicts.
10. [Functional calls and ensembles with TensorDict](https://docs.pytorch.org/tensordict/stable/tutorials/functional.html)
    (`functional.py`): parameter TensorDicts, `torch.vmap` over model ensembles, and `functional_call`.
11. [Exporting tensordict modules](https://docs.pytorch.org/tensordict/stable/tutorials/export.html)
    (`export.py`): `torch.export` of a `TensorDictModule`.

## Data loading

12. [Using TensorDict for datasets](https://docs.pytorch.org/tensordict/stable/tutorials/data_fashion.html)
    (`data_fashion.py`): a FashionMNIST training loop with a memory-mapped TensorDict.
13. [Using tensorclasses for datasets](https://docs.pytorch.org/tensordict/stable/tutorials/tensorclass_fashion.html)
    (`tensorclass_fashion.py`): the same loop with a tensorclass.
14. [Batched data loading with tensorclasses](https://docs.pytorch.org/tensordict/stable/tutorials/tensorclass_imagenet.html)
    (`tensorclass_imagenet.py`): load batches of images from memory-mapped tensors.

## Scientific computing

15. [Simulating particles with a tensorclass](https://docs.pytorch.org/tensordict/stable/tutorials/tensorclass_particles.html)
    (`tensorclass_particles.py`): a batched simulation step, compiled, with a trajectory saved to disk.

## Run a tutorial

Install tensordict, then run the script from the repository root:

```bash
python tutorials/sphinx_tuto/tensordict_keys.py
```

Some tutorials need more packages:

- `zarr_storage.py`: `pip install "tensordict[zarr]"` (Python 3.11+).
- `serialization_speed.py`: `matplotlib`, and `safetensors` and `zarr` for all the comparisons.
- `data_fashion.py`, `tensorclass_fashion.py`, and `tensorclass_imagenet.py`:
  `torchvision`. The first two download FashionMNIST into `./data`.
  `tensorclass_imagenet.py` reads the
  [hymenoptera dataset](https://download.pytorch.org/tutorial/hymenoptera_data.zip),
  unzipped into `./data/hymenoptera_data`.
