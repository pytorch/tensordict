# dispatch

*class*tensordict.nn.dispatch(*separator='_'*, *source: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = 'in_keys'*, *dest: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = 'out_keys'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Lets a function that takes a tensordict be called with tensors as arguments.

Decorate a function or a method whose first argument (after `self`) is
named `tensordict`. When you call it with tensors, the wrapper puts them
in a new tensordict at the keys of `source`, calls the function, and
returns the entries at the keys of `dest`: one tensor if there is one
key, else a tuple in the order of the keys. An output key `"_"` is
skipped. When you call it with a tensordict as first argument, or as the
keyword argument `tensordict`, the wrapper passes it on unchanged.

Tensors are matched to the input keys in two ways:

- Positional arguments, in the order of the keys.
- Keyword arguments, named after the key. A nested key such as
`("a", "c")` is joined with `separator`: `a_c=...`.

Other keyword arguments are passed on to the function. Use the class as
`@dispatch` or, with arguments, as `@dispatch(source=..., dest=...)`.
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) and the other modules of
`tensordict.nn` decorate their `forward` with it.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the string that joins the parts of a
nested key into a keyword argument name. Defaults to `"_"`.
- **source** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the input keys. A
string is the name of the attribute of the module that holds
them. Defaults to `"in_keys"`.
- **dest** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the output keys. A string
is the name of the attribute of the module that holds them.
Defaults to `"out_keys"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch size of the
tensordict built from the arguments is the largest leading shape
that all the inputs share. If `False`, it is empty. Defaults to
`True`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - when the function is decorated, if its first argument
 is not named `tensordict`; when it is called, if one input is
 given both as a positional and as a keyword argument.

Note

An input key cannot hold a tensordict passed as the first positional
argument: that tensordict is taken as the input. The keyword
argument `tensordict` is taken as the input too, unless an input key
is named `"tensordict"`.

Note

To turn keyword calls off for every module, set the environment
variable `DISPATCH_TDNN_MODULES=0` before you import tensordict.
The decorated functions then accept a tensordict only, and the call
does not go through the wrapper of `dispatch()`. See
[torch.compile, torch.func, torch.export, and CUDA graphs](../../compile.html#compile) for how modules behave under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

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
... tensordict["b"] = tensordict["a"] + 1
... return tensordict
...
>>> module = MyModule()
>>> module(a=torch.zeros(2))
tensor([1., 1.])
```

The keys can also be given to the decorator, as a list or as the names
of other attributes (`@dispatch(source="keys_in", dest="keys_out")`):

```
>>> class MyModule(nn.Module):
... @dispatch(source=["a"], dest=["b"])
... def forward(self, tensordict):
... tensordict["b"] = tensordict["a"] + 1
... return tensordict
...
>>> MyModule()(a=torch.zeros(2))
tensor([1., 1.])
```

A nested key is passed by joining its parts with `separator`. Inputs
can also be positional; one input given both ways raises an error:

```
>>> class MyModuleNest(nn.Module):
... in_keys = [("a", "c"), "d"]
... out_keys = ["b"]
...
... @dispatch(separator="__")
... def forward(self, tensordict):
... tensordict["b"] = tensordict["a", "c"] + tensordict["d"]
... return tensordict
...
>>> module = MyModuleNest()
>>> module(a__c=torch.zeros(2), d=torch.ones(2))
tensor([1., 1.])
>>> module(torch.zeros(2), torch.ones(2))
tensor([1., 1.])
>>> module(torch.zeros(2), a__c=torch.ones(2))
Traceback (most recent call last):
 ...
RuntimeError: Duplicated argument in args and kwargs.
```

get_source(*func*, *self_func*)

Returns the input keys that `source` points to.

A string `source` is read as an attribute of `self_func`. `func`
is not used.

Deprecated since version 0.15: This method will be removed in TensorDict 0.17. It has no
replacement.

## Tutorials that use `dispatch`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule