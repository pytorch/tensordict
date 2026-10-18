# set_skip_existing

*class*tensordict.nn.set_skip_existing(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *in_key_attr='in_keys'*, *out_key_attr='out_keys'*, ***, *thread_local: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Bases: `_DecoratorContextManager`

Makes modules skip their computation when their outputs are already in the tensordict.

As a context manager, it sets the value that [`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing)
returns inside the block. [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule),
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) and the probabilistic
modules read that value: when it is on, a module whose `out_keys` are
all in the input returns the input without running. A module whose
`in_keys` and `out_keys` share a key always runs.

As a decorator of a method `forward(self, tensordict, ...)`, it applies
the same check before the method runs, with `mode` as the value.

Parameters:

- **mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*, or**None**,**optional*) - 

the value of
[`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing).

- `True`: a module is skipped if all its `out_keys` are in the
input.
- `False`: no module is skipped.
- A list of keys: only the modules whose `out_keys` are all in
the list can be skipped.
- `None`: for a decorator only; the method follows the value of
the enclosing block (see the third example). As a context
manager, `None` raises a `RuntimeError`.

Defaults to `True`.
- **in_key_attr** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - for a decorator, the name of the
attribute of the module that holds its input keys. Defaults to
`"in_keys"`.
- **out_key_attr** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - for a decorator, the name of the
attribute of the module that holds its output keys. Defaults to
`"out_keys"`.

Keyword Arguments:

**thread_local** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the mode applies to the
calling thread only, and blocks that other threads enter do not
change it. Inside the block, blocks without this flag that the
same thread enters stay in that thread too. Threads started inside
the block see the process-wide mode. If `False`, the mode applies
to the whole process. Defaults to `False`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a block with a mode other than `False` or `None`
 is entered under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import set_skip_existing, skip_existing, TensorDictModule
>>> module = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
>>> td = TensorDict(x=torch.zeros(()), y=torch.zeros(()))
>>> with set_skip_existing():
... print(skip_existing())
... print(module(td)["y"]) # "y" exists: the module does not run
True
tensor(0.)
>>> module(td)["y"]
tensor(1.)
```

As a decorator, it skips the method when the outputs exist:

```
>>> from tensordict.nn import TensorDictModuleBase
>>> class MyModule(TensorDictModuleBase):
... in_keys = []
... out_keys = ["out"]
... @set_skip_existing()
... def forward(self, tensordict):
... print("hello")
... tensordict.set("out", torch.zeros(()))
... return tensordict
>>> module = MyModule()
>>> _ = module(TensorDict({"out": torch.zeros(())})) # does not print anything
>>> _ = module(TensorDict())
hello
```

With `mode=None`, the decorated method runs unless the caller opens a
`set_skip_existing` block:

```
>>> class MyModule(TensorDictModuleBase):
... in_keys = []
... out_keys = ["out"]
... @set_skip_existing(None)
... def forward(self, tensordict):
... print("hello")
... tensordict.set("out", torch.zeros(()))
... return tensordict
>>> module = MyModule()
>>> _ = module(TensorDict({"out": torch.zeros(())}))
hello
>>> with set_skip_existing(True):
... _ = module(TensorDict({"out": torch.zeros(())})) # does not print anything
```

A module whose input is also an output always runs:

```
>>> class MyModule(TensorDictModuleBase):
... in_keys = ["out"]
... out_keys = ["out"]
... @set_skip_existing()
... def forward(self, tensordict):
... print("calling the method!")
... return tensordict
...
>>> _ = MyModule()(TensorDict({"out": torch.zeros(())}))
calling the method!
```