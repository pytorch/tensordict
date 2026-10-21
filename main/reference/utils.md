# Utilities

This page lists the names of `tensordict.utils` that the `tensordict`
module does not export: helpers for tensors and tensordicts, the types
that some methods return, and type aliases. Import them from
`tensordict.utils`. The names that `tensordict` also exports, such as
[`unravel_key()`](generated/tensordict.unravel_key.html#tensordict.unravel_key) or [`assert_close()`](generated/tensordict.assert_close.html#tensordict.assert_close), are
on [tensordict: containers and functions](td.html#api-tensordict).

## Tensor and tensordict helpers

`expand_right` and `expand_as_right` add dimensions on the right of a
tensor, where broadcasting adds them on the left. `isin` and
`remove_duplicates` compare the values of one entry along a batch
dimension.

| [`expand_right`](generated/tensordict.utils.expand_right.html#tensordict.utils.expand_right)(tensor, shape) | Expands a tensor on the right to `shape`. |
| --- | --- |
| [`expand_as_right`](generated/tensordict.utils.expand_as_right.html#tensordict.utils.expand_as_right)(tensor, dest) | Expands a tensor on the right to the shape of `dest`. |
| [`isin`](generated/tensordict.utils.isin.html#tensordict.utils.isin)(input, reference, key[, dim]) | Returns which elements of `input` along `dim` are also in `reference`. |
| [`remove_duplicates`](generated/tensordict.utils.remove_duplicates.html#tensordict.utils.remove_duplicates)(input, key[, dim, ...]) | Returns a tensordict without the elements whose `key` value repeats an earlier one. |

## Returned types

The methods that have a `return_early` argument, such as
[`memmap()`](generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap), return a
[`TensorDictFuture`](generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) if you pass `return_early=True`.
[`tolist()`](generated/tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist) returns
a [`LinkedList`](generated/tensordict.utils.LinkedList.html#tensordict.utils.LinkedList) if you pass `as_linked_list=True`.

| [`TensorDictFuture`](generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture)(futures, resulting_td) | A tensordict that is still being written by background threads. |
| --- | --- |
| [`LinkedList`](generated/tensordict.utils.LinkedList.html#tensordict.utils.LinkedList)(*args[, td]) | A list of the values of a non-tensor entry that writes item assignments back to the tensordict. |

## Type aliases

`DeviceType` is the type of the `device` arguments, and `IndexType`
is the type of the indices that a tensordict accepts.

| [`DeviceType`](generated/tensordict.utils.DeviceType.html#tensordict.utils.DeviceType) | The type of the `device` arguments. |
| --- | --- |
| [`IndexType`](generated/tensordict.utils.IndexType.html#tensordict.utils.IndexType) | The type of the indices that a tensordict accepts in `td[index]`. |

## Debugging and profiling

`print_directory_tree` prints the files of a directory, for example
the directory of a saved tensordict. `timeit` and `logger` are the
timer and the logger that tensordict uses in its own code. `Buffer` is
[`torch.nn.Buffer`](generated/tensordict.utils.Buffer.html#tensordict.utils.Buffer), kept here so that old
imports of `tensordict.utils.Buffer` still work.

| [`print_directory_tree`](generated/tensordict.utils.print_directory_tree.html#tensordict.utils.print_directory_tree)(path[, indent, ...]) | Logs the tree of files and directories under `path`. |
| --- | --- |
| [`timeit`](generated/tensordict.utils.timeit.html#tensordict.utils.timeit)(name) | Measures the time spent in a function or a block of code, by name. |
| [`Buffer`](generated/tensordict.utils.Buffer.html#tensordict.utils.Buffer)([data, persistent]) | A kind of Tensor that should not be considered a model parameter. |

tensordict.utils.logger

The [`logging.Logger`](https://docs.python.org/3/library/logging.html#logging.Logger) named `"tensordict"`. To show fewer
messages from tensordict, raise the level of this logger, for example
`logging.getLogger("tensordict").setLevel(logging.WARNING)`.

## See also

- [tensordict: containers and functions](td.html#api-tensordict): the functions of the `tensordict` module.
- [Save and load tensordicts](../saving.html#saving): `memmap(return_early=True)` and the files of a saved
tensordict.