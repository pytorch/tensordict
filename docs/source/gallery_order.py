"""Order of the tutorials on the gallery index page (sphinx-gallery).

The sidebar lists the tutorials in ``index.rst``; this keeps the gallery index
in the same teaching order instead of sorting by the number of code lines.
"""

from __future__ import annotations

from sphinx_gallery.sorting import FileNameSortKey

TUTORIAL_ORDER = [
    # Basics
    "tensordict_keys",
    "tensordict_slicing",
    "tensordict_shapes",
    "tensordict_preallocation",
    "streamed_tensordict",
    # Storage
    "tensordict_memory",
    "zarr_storage",
    "serialization_speed",
    # tensordict.nn
    "tensordict_module",
    "functional",
    "export",
    # Data loading
    "data_fashion",
    "tensorclass_fashion",
    "tensorclass_imagenet",
    # Scientific computing
    "tensorclass_particles",
]


class TutorialOrder(FileNameSortKey):
    """Sort the tutorials by their place in ``TUTORIAL_ORDER``, then by name."""

    def __call__(self, filename):
        stem = filename.rsplit(".", 1)[0]
        if stem in TUTORIAL_ORDER:
            return (TUTORIAL_ORDER.index(stem), filename)
        return (len(TUTORIAL_ORDER), filename)
