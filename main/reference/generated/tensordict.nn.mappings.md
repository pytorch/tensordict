# mappings

tensordict.nn.mappings(*key: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)

Returns the function named by `key`, which maps a tensor elementwise, for example to positive values.

Modules such as `NormalParamExtractor` use it to
turn the output of a network into a scale. All the functions except
`"none"` (the identity) map real numbers to non-negative numbers.
`"relu"` can return 0, the others return positive numbers.

Parameters:

**key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - one of `"softplus"`, `"exp"`, `"relu"`,
`"expln"` (see [`expln()`](tensordict.nn.utils.expln.html#tensordict.nn.utils.expln)),
`"biased_softplus"` (a [`biased_softplus`](tensordict.nn.biased_softplus.html#tensordict.nn.biased_softplus) with
`bias=1.0`), `"none"`, or a name added with
[`add_custom_mapping()`](tensordict.nn.add_custom_mapping.html#tensordict.nn.add_custom_mapping). `"biased_softplus_{bias}"` and
`"biased_softplus_{bias}_{min_val}"` return a
[`biased_softplus`](tensordict.nn.biased_softplus.html#tensordict.nn.biased_softplus) with these arguments, for example
`"biased_softplus_2.0_0.1"`.

Returns:

A callable that takes a tensor and returns a tensor.

Raises:

- [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) - if `key` is not a known name.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a `"biased_softplus_..."` key does not have the form
 `"biased_softplus_{bias}"` or `"biased_softplus_{bias}_{min_val}"`.

Examples

```
>>> import torch
>>> from tensordict.nn import mappings
>>> mappings("exp")(torch.zeros(2))
tensor([1., 1.])
>>> mappings("biased_softplus_2.0_0.1")(torch.zeros(2))
tensor([2., 2.])
```