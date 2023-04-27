import warnings
from functools import wraps

from tensordict import (  # no_qa
    is_tensorclass as is_tensorclass_true,
    tensorclass as tensorclass_true,
)


@wraps(tensorclass_true)
def tensorclass(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "tensorclass is not a prototype anymore and can be imported directly from tensordict root.",
        category=DeprecationWarning,
    )
    return tensorclass_true(*args, **kwargs)


@wraps(is_tensorclass_true)
def is_tensorclass(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "is_tensorclass is not a prototype anymore and can be imported directly from tensordict root.",
        category=DeprecationWarning,
    )
    return is_tensorclass_true(*args, **kwargs)
