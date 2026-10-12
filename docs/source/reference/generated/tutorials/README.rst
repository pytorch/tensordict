Tutorials
=========

Each tutorial solves one task with a script that runs from top to bottom on a
CPU. The documentation build runs every script, so the output on each page
comes from the code above it. Each page has links to download the script and
a Jupyter notebook, and ends with a "Further reading" list of the guides and
API pages that go deeper. If you are new to tensordict, read
:ref:`getting-started` first.

The sidebar lists the tutorials in four groups, in reading order:

- Basics: keys, indexing, shapes, row-by-row filling, and lazy stacks of
  streamed data.
- Storage: devices and memory-mapped tensors, the zarr backend, and a
  benchmark of the serialization formats.
- tensordict.nn: modules that read and write tensordicts, functional calls and
  ensembles, and export with ``torch.export``.
- Data loading: datasets in a ``TensorDict`` or a ``TensorClass``, and batched
  loading of an image dataset.
