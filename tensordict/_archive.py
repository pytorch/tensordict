# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Zip-archive backend for memory-mapped tensordicts.

A memmap archive is a standard zip file whose entries mirror the memmap
directory layout byte-for-byte (``meta.json`` per level, one ``*.memmap``
file per leaf).  Tensor payloads are STORED (uncompressed) and aligned to
64 bytes so that the whole archive can be memory-mapped once and every
leaf exposed as a zero-copy view into the mapping -- the same mechanism
:meth:`~tensordict.TensorDictBase.consolidate` uses for its single-file
format, and the same trick ``torch.save`` relies on for ``mmap=True``.

Because the entry tree *is* the directory tree, ``unzip archive.tdz``
produces a directory that :meth:`~tensordict.TensorDictBase.load_memmap`
accepts, and ``zip -0 -r`` over an existing memmap directory produces a
loadable archive (unaligned entries silently fall back to a copying read).
"""
from __future__ import annotations

import io
import os
import posixpath
import shutil
import struct
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import torch

from tensordict.memmap import MemoryMappedTensor

# Suffix that triggers archive mode by default in ``save``/``memmap``.
TENSORDICT_ARCHIVE_SUFFIX = ".tdz"

# Payload alignment for STORED tensor entries. 64 bytes matches
# ``torch.save``'s default storage alignment and is a multiple of every
# tensor itemsize, so mmap'ed slices can always be viewed as their dtype.
_ALIGNMENT = 64

# Id of the zip "extra" block used to pad local headers to the alignment
# boundary. Spells b"td" in little-endian.
_PAD_EXTRA_ID = 0x6474

# struct format of the fixed part of a zip local file header.
_LOCAL_HEADER_FMT = "<4s5H3I2H"
_LOCAL_HEADER_SIZE = struct.calcsize(_LOCAL_HEADER_FMT)
_LOCAL_HEADER_SIG = b"PK\x03\x04"
# Size of the zip64 extra block appended to the local header by
# ``zipfile`` when an entry is opened with ``force_zip64=True``.
_ZIP64_LOCAL_EXTRA_SIZE = 20

_COMPRESSION_ALIASES = {
    None: zipfile.ZIP_STORED,
    "stored": zipfile.ZIP_STORED,
    "deflate": zipfile.ZIP_DEFLATED,
    "deflated": zipfile.ZIP_DEFLATED,
    "bzip2": zipfile.ZIP_BZIP2,
    "lzma": zipfile.ZIP_LZMA,
}

_COPY_CHUNK_SIZE = 16 * 1024 * 1024


def _resolve_compression(compression: str | int | None) -> int:
    if isinstance(compression, int):
        return compression
    try:
        return _COMPRESSION_ALIASES[
            compression.lower() if isinstance(compression, str) else compression
        ]
    except KeyError:
        raise ValueError(
            f"Unknown compression {compression!r}. Expected one of "
            f"{sorted(k for k in _COMPRESSION_ALIASES if isinstance(k, str))} "
            f"or a zipfile compression constant."
        ) from None


def is_memmap_archive(path: str | Path) -> bool:
    """Checks whether ``path`` points to a memmap tensordict archive.

    Returns ``True`` if ``path`` is a zip file containing a top-level
    ``meta.json`` entry (the marker of the memmap format), ``False``
    otherwise. Directories always return ``False``.
    """
    path = Path(path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as zf:
        return "meta.json" in zf.namelist()


def _iter_memmap_dir(prefix: Path):
    """Yields the files of a memmap directory in a deterministic order.

    ``meta.json`` comes first at every level so that metadata sits ahead of
    the payload it describes within each subtree.
    """
    entries = sorted(prefix.iterdir(), key=lambda p: p.name)
    files = [p for p in entries if p.is_file()]
    dirs = [p for p in entries if p.is_dir()]
    yield from (p for p in files if p.name == "meta.json")
    yield from (p for p in files if p.name != "meta.json")
    for d in dirs:
        yield from _iter_memmap_dir(d)


def _padding_extra(header_offset: int, name_len: int, alignment: int) -> bytes:
    """Builds a zip extra block padding the payload to ``alignment`` bytes.

    The payload of a STORED entry starts at
    ``header_offset + 30 + name_len + len(extra)`` where ``extra`` is the
    local-header extra field: our padding block (``4 + n`` bytes) followed
    by the 20-byte zip64 block that ``zipfile`` appends under
    ``force_zip64=True``.
    """
    payload_start = (
        header_offset + _LOCAL_HEADER_SIZE + name_len + 4 + _ZIP64_LOCAL_EXTRA_SIZE
    )
    pad = -payload_start % alignment
    return struct.pack("<HH", _PAD_EXTRA_ID, pad) + b"\x00" * pad


def _write_entry(
    zf: zipfile.ZipFile,
    arcname: str,
    src,
    size: int,
    *,
    compress_type: int,
    compresslevel: int | None,
    align: bool,
) -> None:
    zinfo = zipfile.ZipInfo(arcname, date_time=time.localtime()[:6])
    zinfo.compress_type = compress_type
    zinfo.file_size = size
    zinfo.external_attr = 0o644 << 16
    if compresslevel is not None:
        zinfo._compresslevel = compresslevel
    align = align and compress_type == zipfile.ZIP_STORED
    if align:
        name_len = len(zinfo.filename.encode("utf-8"))
        zinfo.extra = _padding_extra(zf.start_dir, name_len, _ALIGNMENT)
    with zf.open(zinfo, "w", force_zip64=True) as dst:
        if align and zf.fp.tell() % _ALIGNMENT:
            raise RuntimeError(
                "Failed to align the archive entry payload. This indicates a "
                "change in the zipfile local header layout; please file an "
                "issue on the tensordict repository."
            )
        shutil.copyfileobj(src, dst, _COPY_CHUNK_SIZE)


def pack_memmap(
    prefix: str | Path,
    archive_path: str | Path,
    *,
    compression: str | int | None = None,
    compresslevel: int | None = None,
) -> Path:
    """Packs a memory-mapped tensordict directory into a single-file zip archive.

    The archive entries replicate the directory layout, so the operation is
    the exact inverse of :func:`unpack_memmap` (and of a plain ``unzip``).
    Tensor payloads are stored uncompressed by default and aligned to
    64 bytes, which lets :meth:`~tensordict.TensorDictBase.load_memmap`
    memory-map the archive and expose every leaf as a zero-copy view.

    Args:
        prefix (str or Path): path to a directory previously produced by
            :meth:`~tensordict.TensorDictBase.memmap` and similar methods.
        archive_path (str or Path): path of the archive to create.

    Keyword Args:
        compression (str or int, optional): one of ``"stored"`` (default),
            ``"deflate"``, ``"bzip2"``, ``"lzma"`` or a
            :mod:`zipfile` compression constant. Any value other than
            ``"stored"`` trades memory-mapped (zero-copy) loading for a
            smaller file: compressed leaves are decompressed in memory when
            accessed.
        compresslevel (int, optional): compression level forwarded to
            :mod:`zipfile`.

    Returns:
        the path to the archive.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, pack_memmap
        >>> td = TensorDict(a=torch.randn(3), b=TensorDict(c=torch.zeros(3, 2)))
        >>> td.memmap("./saved_td")
        >>> pack_memmap("./saved_td", "./saved_td.tdz")
        >>> td2 = TensorDict.load_memmap("./saved_td.tdz")
        >>> assert (td == td2).all()
    """
    prefix = Path(prefix)
    archive_path = Path(archive_path)
    if not (prefix / "meta.json").exists():
        raise ValueError(
            f"{prefix} does not look like a memory-mapped tensordict directory "
            f"(no meta.json found)."
        )
    compress_type = _resolve_compression(compression)
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as zf:
        for filepath in _iter_memmap_dir(prefix):
            arcname = filepath.relative_to(prefix).as_posix()
            align = filepath.suffix == ".memmap"
            entry_compression = compress_type if align else zipfile.ZIP_STORED
            with open(filepath, "rb") as src:
                _write_entry(
                    zf,
                    arcname,
                    src,
                    filepath.stat().st_size,
                    compress_type=entry_compression,
                    compresslevel=compresslevel,
                    align=align,
                )
    return archive_path


def unpack_memmap(archive_path: str | Path, prefix: str | Path) -> Path:
    """Extracts a memmap tensordict archive into a memmap directory.

    This is the inverse of :func:`pack_memmap` and is equivalent to
    ``unzip``: the resulting directory can be passed to
    :meth:`~tensordict.TensorDictBase.load_memmap` or memory-mapped into
    in-place.

    Args:
        archive_path (str or Path): path to an archive produced by
            :func:`pack_memmap` or :meth:`~tensordict.TensorDictBase.save`.
        prefix (str or Path): directory where the content should be
            extracted.

    Returns:
        the path to the directory.
    """
    archive_path = Path(archive_path)
    prefix = Path(prefix)
    if not is_memmap_archive(archive_path):
        raise ValueError(f"{archive_path} is not a memory-mapped tensordict archive.")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(prefix)
    return prefix


class _ArchiveEntry:
    __slots__ = ("offset", "size", "compress_type")

    def __init__(self, offset: int, size: int, compress_type: int):
        self.offset = offset
        self.size = size
        self.compress_type = compress_type


class _ArchiveReader:
    """Index over a memmap archive plus a lazily-created mmap of the file.

    The reader parses the zip central directory once, resolves the payload
    offset of every entry (by reading each local header) and exposes leaves
    as zero-copy views into a single memory-mapped ``uint8`` tensor.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zf = zipfile.ZipFile(self.path)
        self.entries: dict[str, _ArchiveEntry] = {}
        self.dirs: set[str] = {""}
        self.children: dict[str, set[str]] = {"": set()}
        self._storage = None
        with open(self.path, "rb") as f:
            for info in self._zf.infolist():
                if info.flag_bits & 0x1:
                    raise RuntimeError(
                        f"Cannot load encrypted archive entry {info.filename!r}."
                    )
                name = info.filename
                if name.endswith("/"):
                    self._register_dir(name.rstrip("/"))
                    continue
                f.seek(info.header_offset)
                header = f.read(_LOCAL_HEADER_SIZE)
                if len(header) != _LOCAL_HEADER_SIZE or header[:4] != _LOCAL_HEADER_SIG:
                    raise RuntimeError(
                        f"Corrupted local header for entry {name!r} in {self.path}."
                    )
                name_len, extra_len = struct.unpack_from("<2H", header, 26)
                payload_offset = (
                    info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len
                )
                self.entries[name] = _ArchiveEntry(
                    payload_offset, info.file_size, info.compress_type
                )
                parent = posixpath.dirname(name)
                self._register_dir(parent)
                self.children[parent].add(posixpath.basename(name))

    def _register_dir(self, name: str) -> None:
        while name not in self.dirs:
            self.dirs.add(name)
            self.children.setdefault(name, set())
            parent = posixpath.dirname(name)
            self.children.setdefault(parent, set()).add(posixpath.basename(name))
            name = parent

    @property
    def storage(self) -> torch.Tensor:
        # A single MAP_PRIVATE mapping of the whole archive. Slicing it is
        # free; pages are only read from disk when a leaf is accessed.
        # Writes are allowed but copy-on-write: they do not reach the file
        # (which would invalidate the archive checksums).
        if self._storage is None:
            self._storage = torch.from_file(
                str(self.path),
                dtype=torch.uint8,
                size=os.path.getsize(self.path),
                shared=False,
                # needed when device ctx differs
                device=torch.device("cpu"),
            )
        return self._storage

    def read_bytes(self, name: str) -> bytes:
        return self._zf.read(name)

    def flat_view(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Returns the payload of ``name`` as a flat tensor of ``dtype``.

        Zero-copy whenever the entry is STORED and its payload offset is
        compatible with the dtype alignment; falls back to reading a copy in
        memory otherwise (e.g. compressed entries or archives written
        without alignment by external zip tools).
        """
        entry = self.entries[name]
        if entry.compress_type != zipfile.ZIP_STORED:
            data = self.read_bytes(name)
            if not data:
                return torch.empty(0, dtype=dtype)
            return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype)
        flat = self.storage[entry.offset : entry.offset + entry.size]
        if entry.offset % dtype.itemsize:
            # Foreign archive without aligned payloads: viewing would fail,
            # copy into fresh (offset-0) storage instead.
            flat = flat.clone()
        return flat.view(dtype)

    def leaf_tensor(
        self, name: str, dtype: torch.dtype, shape: torch.Size | torch.Tensor
    ) -> torch.Tensor:
        """Builds the tensor stored at entry ``name``.

        ``shape`` may be a nested-size tensor, in which case a (jagged)
        nested-tensor view is returned, mirroring
        :meth:`~tensordict.MemoryMappedTensor.from_filename`.
        """
        flat = self.flat_view(name, dtype)
        if isinstance(shape, torch.Tensor):
            func_offset_stride = getattr(
                torch, "_nested_compute_contiguous_strides_offsets", None
            )
            if func_offset_stride is None:
                raise RuntimeError(
                    "The PyTorch version isn't compatible with memmap "
                    "nested tensors. Please upgrade to a more recent "
                    "version."
                )
            numel = shape.prod(-1).sum().int()
            # _nested_view_from_buffer addresses the *storage* of the buffer
            # from its base, ignoring the view offset, so the buffer must own
            # its storage: nested leaves are materialized rather than viewed.
            return torch._nested_view_from_buffer(
                flat[:numel].clone(),
                shape,
                *func_offset_stride(shape),
            )
        return flat.view(torch.Size(shape))


class _ArchivePath:
    """A :class:`~pathlib.Path`-like view inside a memmap archive.

    Implements the subset of the ``Path`` API used by the memmap loaders
    (``/``, ``exists``, ``is_dir``, ``is_file``, ``iterdir``, ``open``,
    ``with_suffix``, ``name``, ``parts``), so the recursive
    ``_load_memmap`` implementations can traverse an archive exactly as
    they traverse a directory tree.
    """

    __slots__ = ("reader", "at")

    def __init__(self, reader: _ArchiveReader, at: str = ""):
        self.reader = reader
        self.at = at

    @classmethod
    def root(cls, path: str | Path) -> _ArchivePath:
        return cls(_ArchiveReader(path))

    def __truediv__(self, other) -> _ArchivePath:
        other = str(other)
        return _ArchivePath(self.reader, f"{self.at}/{other}" if self.at else other)

    def exists(self) -> bool:
        return self.at in self.reader.entries or self.at in self.reader.dirs

    def is_file(self) -> bool:
        return self.at in self.reader.entries

    def is_dir(self) -> bool:
        return self.at in self.reader.dirs

    def iterdir(self):
        for child in sorted(self.reader.children.get(self.at, ())):
            yield self / child

    def open(self, mode: str = "rb"):
        if mode not in ("rb", "r"):
            raise ValueError(
                f"Archived tensordicts are read-only, cannot open {self} "
                f"with mode {mode!r}."
            )
        data = self.reader.read_bytes(self.at)
        return io.BytesIO(data) if "b" in mode else io.StringIO(data.decode("utf-8"))

    def with_suffix(self, suffix: str) -> _ArchivePath:
        return _ArchivePath(
            self.reader, str(PurePosixPath(self.at).with_suffix(suffix))
        )

    @property
    def name(self) -> str:
        return posixpath.basename(self.at)

    @property
    def parts(self) -> tuple[str, ...]:
        return (
            (str(self.reader.path), *self.at.split("/"))
            if self.at
            else (str(self.reader.path),)
        )

    def __str__(self) -> str:
        return f"{self.reader.path}::{self.at}"

    def __repr__(self) -> str:
        return f"_ArchivePath({str(self)!r})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _ArchivePath)
            and self.reader is other.reader
            and self.at == other.at
        )

    def __hash__(self):
        return hash((id(self.reader), self.at))


def _memmap_tensor_from_path(
    path, *, dtype: torch.dtype, shape: torch.Size | torch.Tensor
) -> torch.Tensor:
    """Loads a memmap leaf from either a real file or an archive entry.

    Real files yield :class:`~tensordict.MemoryMappedTensor` instances
    backed by their own file; archive entries yield zero-copy views into
    the shared mapping of the archive.
    """
    if isinstance(path, _ArchivePath):
        return path.reader.leaf_tensor(path.at, dtype, shape)
    return MemoryMappedTensor.from_filename(
        filename=str(path), dtype=dtype, shape=shape
    )


def _save_as_archive(
    td,
    archive_path: str | Path,
    *,
    num_threads: int = 0,
    compression: str | int | None = None,
    copy_existing: bool = False,
    share_non_tensor: bool = False,
    existsok: bool = True,
    robust_key: bool | None = True,
) -> Path:
    """Writes ``td`` to a single-file memmap archive.

    If ``td`` is already memory-mapped on disk, its directory is packed
    directly. Otherwise the tensordict is staged to a temporary memmap
    directory next to ``archive_path`` (so both live on the same
    filesystem) and packed, after which the staging directory is removed.
    """
    archive_path = Path(archive_path)
    if archive_path.is_dir():
        raise ValueError(
            f"Cannot write an archive at {archive_path}: it is an existing "
            f"directory."
        )
    if archive_path.exists() and not existsok:
        raise RuntimeError(
            f"A file already exists at {archive_path}, cannot save the "
            f"tensordict there. Set existsok=True to overwrite."
        )
    saved_prefix = getattr(td, "_memmap_prefix", None)
    if (
        td.is_memmap()
        and isinstance(saved_prefix, Path)
        and (saved_prefix / "meta.json").exists()
    ):
        return pack_memmap(saved_prefix, archive_path, compression=compression)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.mkdtemp(dir=archive_path.parent, prefix=f".{archive_path.name}.")
    try:
        td.memmap(
            prefix=staging,
            copy_existing=copy_existing,
            num_threads=num_threads,
            share_non_tensor=share_non_tensor,
            robust_key=robust_key,
        )
        return pack_memmap(staging, archive_path, compression=compression)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
