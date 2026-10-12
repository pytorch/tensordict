# dispatch

*class*tensordict.nn.dispatch(*separator='_'*, *source: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = 'in_keys'*, *dest: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = 'out_keys'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Allows for a function expecting a TensorDict to be called using kwargs.

`dispatch()` must be used within modules that have an `in_keys` (or
another source of keys indicated by the `source` keyword argument) and
`out_keys` (or another `dest` key list) attributes indicating what keys
to be read and written from the tensordict. The wrapped function should
also have a `tensordict` leading argument.

The resulting function will return a single tensor (if there is a single
element in out_keys), otherwise it will return a tuple sorted as the `out_keys`
of the module.

`dispatch()` can be used either as a method or as a class when extra arguments
need to be passed.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - separator that combines sub-keys together
for `in_keys` that are tuples of strings.
Defaults to `"_"`.
- **source** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**keys**,**optional*) - if a string is provided,
it points to the module attribute that contains the
list of input keys to be used. If a list is provided instead, it
will contain the keys used as input to the module.
Defaults to `"in_keys"` which is the attribute name of
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) list of input keys.
- **dest** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**keys**,**optional*) - if a string is provided,
it points to the module attribute that contains the
list of output keys to be used. If a list is provided instead, it
will contain the keys used as output to the module.
Defaults to `"out_keys"` which is the attribute name of
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) list of output keys.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch-size of the
input tensordict is determined automatically as the maximum number
of common dimensions across all the input tensors.
Defaults to `True`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict.nn import dispatch
>>> class MyModule(nn.Module):
... in_keys = ["a"]
... out_keys = ["b"]
...
... @dispatch
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a'] + 1
... return tensordict
...
>>> module = MyModule()
>>> b = module(a=torch.zeros(1, 2))
>>> assert (b == 1).all()
>>> # equivalently
>>> class MyModule(nn.Module):
... keys_in = ["a"]
... keys_out = ["b"]
...
... @dispatch(source="keys_in", dest="keys_out")
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a'] + 1
... return tensordict
...
>>> module = MyModule()
>>> b = module(a=torch.zeros(1, 2))
>>> assert (b == 1).all()
>>> # or this
>>> class MyModule(nn.Module):
... @dispatch(source=["a"], dest=["b"])
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a'] + 1
... return tensordict
...
>>> module = MyModule()
>>> b = module(a=torch.zeros(1, 2))
>>> assert (b == 1).all()
```

`dispatch()` will also work with nested keys with the default
`"_"` separator.

Examples

```
>>> class MyModuleNest(nn.Module):
... in_keys = [("a", "c")]
... out_keys = ["b"]
...
... @dispatch
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a', 'c'] + 1
... return tensordict
...
>>> module = MyModuleNest()
>>> b = module(a_c=torch.zeros(1, 2))
>>> assert (b == 1).all()
```

If another separator is wanted, it can be indicated with the `separator`
argument in the constructor:

Examples

```
>>> class MyModuleNest(nn.Module):
... in_keys = [("a", "c")]
... out_keys = ["b"]
...
... @dispatch(separator="sep")
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a', 'c'] + 1
... return tensordict
...
>>> module = MyModuleNest()
>>> b = module(asepc=torch.zeros(1, 2))
>>> assert (b == 1).all()
```

`dispatch()` also accepts positional arguments. They are matched to
the input keys in the order of `in_keys`. If you pass the same key as a
positional and as a keyword argument, the call raises a `RuntimeError`.

Note

If the first argument is a [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) instance, it is
assumed that dispatch is not being used and that this tensordict
contains all the necessary information to be run through the module.
In other words, one cannot decompose a tensordict with the first key
of the module inputs pointing to a tensordict instance.
In general, it is preferred to use `dispatch()` with tensordict
leaves only.

Examples

```
>>> class MyModuleNest(nn.Module):
... in_keys = [("a", "c"), "d"]
... out_keys = ["b"]
...
... @dispatch
... def forward(self, tensordict):
... tensordict['b'] = tensordict['a', 'c'] + tensordict["d"]
... return tensordict
...
>>> module = MyModuleNest()
>>> b = module(torch.zeros(1, 2), d=torch.ones(1, 2)) # works
>>> assert (b == 1).all()
>>> b = module(torch.zeros(1, 2), torch.ones(1, 2)) # works
>>> assert (b == 1).all()
>>> module(torch.zeros(1, 2), a_c=torch.ones(1, 2)) # ("a", "c") is passed twice
Traceback (most recent call last):
 ...
RuntimeError: Duplicated argument in args and kwargs.
```

Note

To turn keyword calls off for every module, set the environment
variable `DISPATCH_TDNN_MODULES=0` before you import tensordict.
The decorated functions then accept a tensordict only, and the call
does not go through the wrapper of `dispatch()`. See
[torch.compile, torch.func, torch.export and CUDA graphs](../../compile.html#compile) for how modules behave under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

get_source(*func*, *self_func*)

Returns the input keys that `source` points to.

Deprecated since version 0.15: This method will be removed in TensorDict 0.17. It has no
replacement.

## Tutorials that use `dispatch`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule