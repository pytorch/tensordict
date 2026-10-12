# set_skip_existing

*class*tensordict.nn.set_skip_existing(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *in_key_attr='in_keys'*, *out_key_attr='out_keys'*)

Bases: `_DecoratorContextManager`

A context manager for skipping existing nodes in a TensorDict graph.

When used as a context manager, it will set the skip_existing() value
to the `mode` indicated, leaving the user able to code up methods that
will check the global value and execute the code accordingly.

When used as a method decorator, it will check the tensordict input keys
and skip the method if the current mode applies to all output keys and they
are already present.
This is not expected to be used as a decorator for methods that do not
respect the following signature: `def fun(self, tensordict, *args, **kwargs)`.

Parameters:

- **mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - If `True`, it indicates that existing entries in the graph
won't be overwritten, unless they are only partially present. [`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing)
will return `True`.
If `False`, no check will be performed.
If a list of keys, only modules whose output keys are all in the
list can be skipped. The keys follow TensorDict's nested-key
conventions. If any output key is not in the list, the module runs
and recomputes all of its outputs.
If `None`, the value of [`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) will not be
changed. This is intended to be used exclusively for decorating
methods and allow their behaviour to depend on the same class
when used as a context manager (see example below).
Defaults to `True`.
- **in_key_attr** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the input key list attribute
in the module's method being decorated. Defaults to `in_keys`.
- **out_key_attr** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the output key list attribute
in the module's method being decorated. Defaults to `out_keys`.

Examples

```
>>> import torch
>>> from tensordict.nn import set_skip_existing, skip_existing
>>> with set_skip_existing():
... if skip_existing():
... print("True")
... else:
... print("False")
...
True
>>> print("calling from outside:", skip_existing())
calling from outside: False
```

This class can also be used as a decorator:

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import set_skip_existing, skip_existing, TensorDictModuleBase
>>> class MyModule(TensorDictModuleBase):
... in_keys = []
... out_keys = ["out"]
... @set_skip_existing()
... def forward(self, tensordict):
... print("hello")
... tensordict.set("out", torch.zeros(()))
... return tensordict
>>> module = MyModule()
>>> module(TensorDict({"out": torch.zeros(())}, [])) # does not print anything
TensorDict(
 fields={
 out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> module(TensorDict()) # prints hello
hello
TensorDict(
 fields={
 out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Decorating a method with the mode set to `None` is useful whenever one
wants to let the context manager take care of skipping things from the outside:

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import set_skip_existing, skip_existing, TensorDictModuleBase
>>> class MyModule(TensorDictModuleBase):
... in_keys = []
... out_keys = ["out"]
... @set_skip_existing(None)
... def forward(self, tensordict):
... print("hello")
... tensordict.set("out", torch.zeros(()))
... return tensordict
>>> module = MyModule()
>>> _ = module(TensorDict({"out": torch.zeros(())}, [])) # prints "hello"
hello
>>> with set_skip_existing(True):
... _ = module(TensorDict({"out": torch.zeros(())}, [])) # no print
```

Note

To allow for modules to have the same input and output keys and not
mistakenly ignoring subgraphs, `@set_skip_existing(True)` will be
deactivated whenever the output keys are also the input keys:

```
>>> class MyModule(TensorDictModuleBase):
... in_keys = ["out"]
... out_keys = ["out"]
... @set_skip_existing()
... def forward(self, tensordict):
... print("calling the method!")
... return tensordict
...
>>> module = MyModule()
>>> module(TensorDict({"out": torch.zeros(())}, [])) # runs: in_keys and out_keys overlap
calling the method!
TensorDict(
 fields={
 out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```