.. _installation:

Installation
============

Install the latest release from PyPI:

.. code-block:: bash

   pip install tensordict

tensordict needs Python 3.10+ and PyTorch 2.11+. To make sure that the
install worked, print the version:

.. code-block:: bash

   python -c "import tensordict; print(tensordict.__version__)"

Requirements
------------

tensordict is pure Python: it has no compiled extension. Each change is
tested with PyTorch 2.11, the latest PyTorch release, and PyTorch nightly, on
Python 3.10 to 3.15 and the free-threaded 3.14t and 3.15t.

If PyTorch is not installed, ``pip`` installs the default build from PyPI
with tensordict. If you need a specific build, for example for CUDA or for a
CPU-only machine, install PyTorch first. The `PyTorch Get Started page
<https://pytorch.org/get-started/locally/>`__ gives the command for each
build.

The other runtime dependencies are ``numpy``, ``cloudpickle``, ``pyvers``, and
``typing_extensions>=4.0`` on Python 3.10. ``pip`` and ``conda`` install them
with tensordict.

Optional dependencies
---------------------

Some features need other packages. Install them with an extra, for example
``pip install "tensordict[h5,zarr]"``:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Extra
     - Installs
     - For
   * - ``h5``
     - ``h5py>=3.9``
     - HDF5 files: :meth:`~tensordict.TensorDict.to_h5`,
       :meth:`~tensordict.TensorDict.from_h5`, and
       :class:`~tensordict.PersistentTensorDict`.
   * - ``zarr``
     - ``zarr>=3.0.8``
     - Zarr stores: :meth:`~tensordict.TensorDict.to_zarr` and
       :meth:`~tensordict.TensorDict.from_zarr`. zarr 3 needs Python 3.11+.
       On Python 3.10, this extra installs nothing.
   * - ``pandas``
     - ``pandas>=1.5``
     - DataFrames and CSV files: :meth:`~tensordict.TensorDict.to_pandas`,
       :meth:`~tensordict.TensorDict.from_pandas`,
       :meth:`~tensordict.TensorDict.to_csv`, and
       :meth:`~tensordict.TensorDict.from_csv`.
   * - ``parquet``
     - ``pyarrow>=10.0``
     - Parquet files: :meth:`~tensordict.TensorDict.to_parquet` and
       :meth:`~tensordict.TensorDict.from_parquet`. These methods use
       pyarrow, or pandas if pyarrow is not installed.
       Arrow tables: :func:`~tensordict.from_arrow` and
       :func:`~tensordict.to_arrow`, which need pyarrow 15 or later.
   * - ``tabular``
     - ``pandas>=1.5`` and ``pyarrow>=10.0``
     - The ``pandas`` and ``parquet`` extras together.
   * - ``safetensors``
     - ``safetensors>=0.7.0``
     - safetensors files: :func:`~tensordict.to_safetensors` and
       :func:`~tensordict.from_safetensors`.
   * - ``onnx``
     - ``onnx``, ``onnxscript``, and ``onnxruntime``
     - Export of a :class:`~tensordict.nn.TensorDictModule` to ONNX with
       :func:`torch.onnx.export`, and running the result.

:ref:`storage` explains the HDF5, Zarr, and tabular formats.

Install a nightly build
-----------------------

The ``tensordict-nightly`` package is built from the main branch every day.
It installs the same ``tensordict`` directory as the release. If both
packages are installed, the one installed last overwrites the files of the
other. Uninstalling one of them then deletes files that the other needs. So
uninstall the release first:

.. code-block:: bash

   pip uninstall tensordict
   pip install tensordict-nightly

If both are already installed, uninstall both, then install one of them
again.

A nightly build of tensordict can use features that only PyTorch nightly
has, so install the two together. Recent PyTorch nightly builds need Python
3.11+. This example installs the CPU build of PyTorch nightly:

.. code-block:: bash

   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
   pip install tensordict-nightly

Install with conda
------------------

The conda-forge channel has the releases:

.. code-block:: bash

   conda install -c conda-forge tensordict

Install from source
-------------------

Install from source to contribute to tensordict, or to use a change that is
not in a nightly build yet. The build needs no compiler. ``pip`` installs the
build requirements, ``setuptools>=77`` and ``setuptools_scm``, unless you
pass ``--no-build-isolation``.

.. code-block:: bash

   git clone https://github.com/pytorch/tensordict
   cd tensordict
   pip install -e .

``-e`` installs tensordict in editable mode: Python imports it from your
clone, so your changes apply without a new install.

If you do not want ``pip`` to change your PyTorch installation, pass
``--no-deps``. ``--no-deps`` also skips the other runtime dependencies, so
install them first:

.. code-block:: bash

   pip install numpy cloudpickle "pyvers>=0.2,<0.3" "typing_extensions>=4.0"
   pip install -e . --no-deps

.. note::

   If you use ``uv`` with a PyTorch nightly from the PyTorch nightly wheel
   index, ``uv`` can replace the nightly ``torch`` with a release from PyPI
   when it resolves the dependencies of tensordict. To keep your PyTorch
   build, use one of these commands:

   .. code-block:: bash

      # Option 1: do not resolve dependencies (install the runtime dependencies first)
      uv pip install -e . --no-deps
      # Option 2: resolve against the PyTorch nightly index
      uv pip install -e . --prerelease=allow -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"

tensordict and TorchRL versions
-------------------------------

Each `TorchRL <https://github.com/pytorch/rl>`__ release requires the
tensordict release with the same minor version. For example, TorchRL 0.14
requires tensordict 0.14, and does not accept 0.15. The nightly package
``torchrl-nightly`` requires ``tensordict-nightly``.

To get a matching pair, install TorchRL and let ``pip`` choose tensordict:

.. code-block:: bash

   pip install torchrl

Do not mix a nightly build of one library with a release of the other. A
mismatch can show as an ``ImportError: cannot import name ...`` when you
import ``torchrl``.

Next steps
----------

:ref:`getting-started` shows the main features of tensordict in one hour.
