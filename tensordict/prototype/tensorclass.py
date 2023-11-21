import warnings
from functools import wraps

from tensordict._td import is_tensorclass as is_tensorclass_true  # no_qa
from tensordict.tensorclass import tensorclass as tensorclass_true  # no_qa


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
