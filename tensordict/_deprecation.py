# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers that deprecate public names.

Every deprecation names the TensorDict release that removes it.
``test_deprecation_deadlines`` in ``test/utils/test_utils.py`` fails once
``version.txt`` reaches that release, so that the removal is not forgotten.
It reads the ``removal=`` argument of these helpers, and the phrases
"removed in TensorDict X.Y", "Starting with TensorDict X.Y, the default will
change" and "the default will change (or will become, or becomes) ... in X.Y"
in messages, docstrings and docs.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def deprecation_message(
    what: str, *, removal: str, replacement: str | None = None
) -> str:
    """Returns the message of the warning that :func:`warn_deprecated` emits."""
    message = f"{what} is deprecated and will be removed in TensorDict {removal}."
    if replacement is not None:
        message += f" Use {replacement} instead."
    return message


def warn_deprecated(
    what: str, *, removal: str, replacement: str | None = None, stacklevel: int = 2
) -> None:
    """Emits a :class:`DeprecationWarning` that says when ``what`` goes away.

    Args:
        what (str): the deprecated name, written as users write it, for
            instance ``"tensordict.utils.cache"`` or ``"set_list_to_stack(False)"``.

    Keyword Args:
        removal (str): the release that removes ``what``, for instance ``"0.17"``.
        replacement (str, optional): what to use instead.
        stacklevel (int, optional): the frame that the warning points to,
            counted as in :func:`warnings.warn` from the caller of this
            function. The default, ``2``, points to the code that called the
            deprecated function.
    """
    warnings.warn(
        deprecation_message(what, removal=removal, replacement=replacement),
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def warn_deprecated_env_var(
    name: str, value: str, *, removal: str, replacement: str | None = None
) -> None:
    """Emits a :class:`FutureWarning` that says when a value of an environment variable goes away.

    tensordict reads its environment variables when it is imported, so a
    :class:`DeprecationWarning` would be attributed to tensordict itself, and
    Python's default warning filters would hide it. Users set these variables
    in their shell or job scripts, so the warning is a :class:`FutureWarning`,
    which Python shows by default.

    Call it from the module-level code that reads the variable: the warning
    points to that code.

    Args:
        name (str): the name of the environment variable, for instance
            ``"LIST_TO_STACK"``.
        value (str): the deprecated value, as read from :data:`os.environ`.

    Keyword Args:
        removal (str): the release that removes support for ``name=value``.
        replacement (str, optional): what to use instead.
    """
    warnings.warn(
        deprecation_message(
            f"{name}={value}", removal=removal, replacement=replacement
        ),
        FutureWarning,
        stacklevel=2,
    )


def deprecated(
    what: str, *, removal: str, replacement: str | None = None
) -> Callable[[_F], _F]:
    """Decorates a function or method so that each call warns.

    To deprecate a property, decorate its getter (and setter) and then wrap
    the result in :class:`property`.

    Args:
        what (str): the deprecated name, as in :func:`warn_deprecated`.

    Keyword Args:
        removal (str): the release that removes ``what``.
        replacement (str, optional): what to use instead.
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(what, removal=removal, replacement=replacement)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_attributes(
    module: str,
    attributes: Mapping[str, tuple[Any, str | None]],
    *,
    removal: str,
) -> Callable[[str], Any]:
    """Returns a module ``__getattr__`` that serves deprecated attributes.

    Assign the result to ``__getattr__`` at the end of the module (PEP 562).
    Reading a deprecated attribute, or importing it with
    ``from module import name``, warns and returns its value. Any other
    missing attribute raises :class:`AttributeError` as usual.

    Args:
        module (str): the ``__name__`` of the module.
        attributes (mapping): maps each deprecated name to ``(value, replacement)``,
            where ``replacement`` says what to use instead, or is ``None``.

    Keyword Args:
        removal (str): the release that removes the attributes.
    """

    def __getattr__(name: str) -> Any:
        if name not in attributes:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        value, replacement = attributes[name]
        warn_deprecated(f"{module}.{name}", removal=removal, replacement=replacement)
        return value

    return __getattr__
