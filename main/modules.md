# Modules: TensorDictModule and tensordict.nn

A module of [`tensordict.nn`](reference/nn.html#module-tensordict.nn) reads its inputs from a tensordict by key and
writes its outputs back into it. You can then chain modules that take several
inputs. Generic code, such as a training loop, can call any model with one
tensordict, without knowing its inputs. For what a tensordict is, see
[What is TensorDict?](overview.html#overview).

## Wrap a module: TensorDictModule

[`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) wraps a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), or
any other callable, and gives it a list of input keys (`in_keys`) and output
keys (`out_keys`). When you call it with a tensordict, it reads the entries
at `in_keys`, calls the wrapped module with them, and writes the results at
`out_keys`. In this example, the tensordict holds the positions of 5
particles:

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> module = TensorDictModule(nn.Linear(3, 4), in_keys=["pos"], out_keys=["hidden"])
>>> td = TensorDict({"pos": torch.randn(5, 3)}, batch_size=[5])
>>> out = module(td)
>>> out is td
True
>>> td
TensorDict(batch_size=[5], fields={hidden: f32[4], pos: f32[3]})
```

By default, the module writes into the input tensordict and returns it.
`out` and `td` are two names for one object, so `module(td)` without an
assignment has the same effect as `td = module(td)`. A module can read any
entry of the tensordict, so modules that share keys make a graph.

A [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) is a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
Save it with `state_dict()` and move it with `to()`, as any other module.
The parameters of the wrapped module have the prefix `module.`, for
example `module.weight`.

### Input and output keys

If `in_keys` is a list, the module passes the values to the wrapped module
as positional arguments, in the order of the list. `out_keys` has one key
for each output of the wrapped module, in order. A key can be a nested key: a
tuple of strings that names an entry inside a nested tensordict (see
[Keys, nested keys, and locking](concepts/keys.html#keys)):

```
>>> shift = TensorDictModule(
... lambda x: (x + 1, x - 1),
... in_keys=[("robot", "joints")],
... out_keys=[("robot", "up"), ("robot", "down")],
... )
>>> td = shift(TensorDict({("robot", "joints"): torch.zeros(2, 3)}, batch_size=[2]))
>>> list(td["robot"].keys())
['joints', 'up', 'down']
```

If `in_keys` is a dict, the module passes the values as keyword arguments.
By default, the dict maps each argument name to a key of the tensordict:
`in_keys={"x": "pos"}` calls `module(x=td["pos"])`. One key can feed
several arguments. With `out_to_in_map=False`, the dict maps each key of the
tensordict to an argument name instead:

```
>>> diff = TensorDictModule(lambda x, y: x - y, in_keys={"x": "a", "y": "b"}, out_keys=["c"])
>>> diff(TensorDict({"a": torch.tensor(5.0), "b": torch.tensor(2.0)}))["c"]
tensor(3.)
>>> diff = TensorDictModule(
... lambda x, y: x - y, in_keys={"a": "x", "b": "y"}, out_keys=["c"], out_to_in_map=False
... )
>>> diff(TensorDict({"a": torch.tensor(5.0), "b": torch.tensor(2.0)}))["c"]
tensor(3.)
```

The wrapped module can return one tensor, a tuple with one tensor for each
key of `out_keys`, or a dict. The module reads a dict output by the names in
`out_keys`. If the number of tensors and the number of `out_keys` are not
equal, the module raises `RuntimeError: There are more tensors (...) than
out_keys (...)` or `There are more out_keys than tensors`. To discard an
output, give it the out-key `"_"`:

```
>>> first_only = TensorDictModule(lambda x: (x + 1, x - 1), in_keys=["x"], out_keys=["y", "_"])
>>> list(first_only(TensorDict({"x": torch.zeros(2)})).keys())
['x', 'y']
```

### Missing inputs and errors

If an entry of `in_keys` is not in the input, the module passes `None` in
its place. A wrapped module that cannot take `None` then fails, and the
error names the missing keys:

```
>>> module(TensorDict({"other": torch.zeros(5, 3)}, batch_size=[5]))
Traceback (most recent call last):
 ...
KeyError: "Some tensors ... the following inputs are None: {'pos'}."
```

With `strict=True`, the module raises `KeyError` for the first missing key
before it calls the wrapped module.

When the wrapped module raises an error, [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)
raises the same error again. Its cause is a `RuntimeError` with the text
`TensorDictModule failed with operation`, followed by the wrapped module, its
`in_keys`, and its `out_keys`. Python prints the cause first, so the
original error is the last one in the traceback.

### Where the outputs go

The `inplace` argument sets where the module writes its outputs:

- `True` (default): into the input tensordict, which the module returns.
- `False`: into a new [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) that holds only the
outputs, with an empty batch size and no device. The input does not change.
- `"empty"`: into `tensordict.empty()`, a new tensordict with no entries
and the type, batch size, and device of the input.

In this example, the module writes into a new tensordict:

```
>>> module = TensorDictModule(nn.Linear(3, 4), in_keys=["pos"], out_keys=["hidden"], inplace=False)
>>> td = TensorDict({"pos": torch.randn(5, 3)}, batch_size=[5])
>>> out = module(td)
>>> list(out.keys()), out.batch_size, list(td.keys())
(['hidden'], torch.Size([]), ['pos'])
```

If you pass a tensordict as `tensordict_out`, the module writes its outputs
into it, whatever the value of `inplace`, and returns it:

```
>>> dest = TensorDict(batch_size=[5])
>>> module(td, tensordict_out=dest) is dest
True
>>> list(dest.keys())
['hidden']
```

### Call a module with tensors

You can also call a module with tensors, from code that does not use
tensordicts. Pass each input as a keyword argument named after its key in
`in_keys`. The module builds a tensordict from the arguments and runs. It
returns the output tensors instead of a tensordict: one tensor if
`out_keys` has one key, else a tuple in the order of `out_keys`.

```
>>> class Net(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(100, 1)
...
... def forward(self, x):
... logits = self.linear(x)
... return logits, torch.sigmoid(logits)
>>> net = TensorDictModule(
... Net(),
... in_keys=["input"],
... out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
... )
>>> logits, probabilities = net(input=torch.randn(32, 100))
>>> probabilities.shape
torch.Size([32, 1])
```

These rules apply to keyword calls:

- The keyword of a nested key joins its parts with `"_"`: the in-key
`("robot", "joints")` becomes `robot_joints=`.
- You can also pass the tensors as positional arguments, in the order of
`in_keys`. If you pass the same input both ways, the module raises
`RuntimeError: Duplicated argument in args and kwargs.`
- A keyword call skips the outputs with the out-key `"_"`, and returns
the other outputs.
- The tensordict that [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) builds from
the arguments has an empty batch size.

For example, with the nested in-key `("robot", "joints")` of the `shift`
module above:

```
>>> shift(robot_joints=torch.zeros(2, 3))[0].shape
torch.Size([2, 3])
```

The [export tutorial](tutorials/export.html) uses keyword calls to export a
module that takes and returns plain tensors.

A decorator on `forward`, [`dispatch`](reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch), gives
[`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) its keyword calls. To turn keyword
calls off for all modules, set the environment variable
`DISPATCH_TDNN_MODULES=0` before you import tensordict. The modules then
take only tensordicts, a keyword call raises `TypeError`, and each call
skips the `dispatch` wrapper.

### Write your own module

To write a module that works on the tensordict itself, subclass
[`TensorDictModuleBase`](reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase). Give the class `in_keys` and
`out_keys`, and a `forward` that takes a tensordict, writes the
`out_keys` into it, and returns it. To accept keyword calls too, decorate
`forward` with [`dispatch`](reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch):

```
>>> from tensordict.nn import TensorDictModuleBase, dispatch
>>> class Normalize(TensorDictModuleBase):
... in_keys = ["signal"]
... out_keys = ["signal_norm"]
...
... @dispatch
... def forward(self, tensordict):
... signal = tensordict["signal"]
... tensordict["signal_norm"] = (signal - signal.mean()) / signal.std()
... return tensordict
>>> normalize = Normalize()
>>> normalize(TensorDict({"signal": torch.arange(4.0)}, batch_size=[4]))["signal_norm"]
tensor([-1.1619, -0.3873, 0.3873, 1.1619])
>>> normalize(signal=torch.arange(4.0))
tensor([-1.1619, -0.3873, 0.3873, 1.1619])
```

A plain [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) with `in_keys` and `out_keys`
attributes also works in a sequence (see Chain modules:
TensorDictSequential), if its `forward` takes a tensordict, writes its
outputs into it, and returns it.

### Run a module on a tensorclass

A module also runs on a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) instance, the class
form of a tensordict with typed fields (see [TensorClass](tensorclass.html#tensorclass-guide)). It
reads and writes the fields named in `in_keys` and `out_keys`, and
returns the same instance:

```
>>> from tensordict import TensorClass
>>> class Particles(TensorClass):
... pos: torch.Tensor
... hidden: torch.Tensor | None = None
>>> encoder = TensorDictModule(nn.Linear(3, 4), in_keys=["pos"], out_keys=["hidden"])
>>> particles = Particles(pos=torch.randn(5, 3), batch_size=[5])
>>> encoder(particles) is particles
True
>>> particles.hidden.shape
torch.Size([5, 4])
```

Each out-key must be a field of the class. A module that writes another key
raises `AttributeError: Cannot set the attribute ...: Particles has no such
field`.

To give `forward` typed inputs and outputs instead of keys, subclass
[`TensorClassModuleBase`](reference/generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase) (see Typed modules:
TensorClassModuleBase).

## Chain modules: TensorDictSequential

[`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential) gives the output of each module to the next
module as its only input, so it cannot hold a module with several inputs.
[`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) runs a sequence of tensordict
modules on one tensordict. Each module reads its `in_keys` from that
tensordict and writes its `out_keys` into it. A module can therefore read
entries that no earlier module wrote, and it can ignore the outputs of
earlier modules. In this example, `masker` reads the output of `net` and
the entry `("input", "mask")`:

```
>>> from tensordict.nn import TensorDictSequential
>>> class MLP(nn.Module):
... def __init__(self, input_size=100, hidden_size=50, output_size=10):
... super().__init__()
... self.fc1 = nn.Linear(input_size, hidden_size)
... self.fc2 = nn.Linear(hidden_size, output_size)
...
... def forward(self, x):
... x = torch.relu(self.fc1(x))
... return self.fc2(x)
>>> class Masker(nn.Module):
... def forward(self, x, mask):
... return torch.softmax(x * mask, dim=1)
>>> net = TensorDictModule(MLP(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")])
>>> masker = TensorDictModule(
... Masker(),
... in_keys=[("intermediate", "x"), ("input", "mask")],
... out_keys=[("output", "probabilities")],
... )
>>> model = TensorDictSequential(net, masker)
>>> td = TensorDict(
... {"input": {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))}},
... batch_size=[32],
... )
>>> td = model(td)
>>> td["output", "probabilities"].shape
torch.Size([32, 10])
```

### How a sequence gets its keys

A [`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) computes its keys from its
modules, in order:

- `in_keys` holds the keys that a module reads and that no earlier module
writes. The input must contain them.
- `out_keys` holds all the keys that the modules write, each one once. A key
that two modules write is at the position of the last write.

In this sequence, the last module writes `"h"` again:

```
>>> from tensordict.nn import TensorDictModule as Mod
>>> seq = TensorDictSequential(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["h"]),
... Mod(lambda h, mask: h * mask, in_keys=["h", "mask"], out_keys=["y"]),
... Mod(lambda x: -x, in_keys=["x"], out_keys=["z"]),
... Mod(lambda h: h * 10, in_keys=["h"], out_keys=["h"]),
... )
>>> seq.in_keys
['x', 'mask']
>>> seq.out_keys
['y', 'z', 'h']
```

The modules of a sequence can be any callable that takes and returns a
tensordict. The sequence does not track the keys of a callable that has no
`in_keys` and `out_keys`. A sequence called with tensors returns all its
`out_keys`, in order (see Call a module with tensors):

```
>>> seq(x=torch.ones(3), mask=torch.tensor([1.0, 0.0, 1.0]))
(tensor([2., 0., 2.]), tensor([-1., -1., -1.]), tensor([20., 20., 20.]))
```

A [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential) of tensordict modules runs on a tensordict,
because each module returns the tensordict. But it has no `in_keys` and
`out_keys`, so keyword calls, `select_out_keys`, and
`select_subsequence` do not work on it. Use
[`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) instead.

### Run part of a sequence

[`select_subsequence()`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential.select_subsequence) returns a new
[`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) that keeps only the modules that
are necessary to compute `out_keys` from `in_keys`. Both arguments are
optional, and default to the keys of the sequence:

```
>>> seq.select_subsequence(out_keys=["y"]).in_keys
['x', 'mask']
>>> len(seq.select_subsequence(out_keys=["y"]))
2
>>> len(seq.select_subsequence(out_keys=["z"]))
1
>>> seq.select_subsequence(in_keys=["h", "mask"]).out_keys
['y', 'h']
```

With `in_keys`, the new sequence keeps the modules that read only these keys
or keys that a kept module writes. It raises `ValueError: No modules left
after selection` if no module is left. Indexing also returns a new sequence:
`seq[:2]` holds the first two modules. To give the modules names, pass a
dict: `TensorDictSequential({"encoder": enc, "head": head})`, then read
`seq["encoder"]`.

Build the subsequence once, outside a compiled function. Dynamo cannot build
it inside the graph (see [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile)).

### Where a sequence writes

A [`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) also takes `inplace` and
`tensordict_out`. With `tensordict_out`, the modules run on a shallow
copy of the input, and the sequence writes its `out_keys` into
`tensordict_out`. The input does not change.

With `inplace=False` or `inplace="empty"`, the modules also run on a
shallow copy of the input. The sequence returns a new tensordict with its
`out_keys`, and the input does not change:

```
>>> seq = TensorDictSequential(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["h"]),
... Mod(lambda h: h * 2, in_keys=["h"], out_keys=["y"]),
... inplace=False,
... )
>>> td = TensorDict({"x": torch.zeros(3)}, batch_size=[3])
>>> sorted(seq(td).keys())
['h', 'y']
>>> sorted(td.keys())
['x']
```

## Control the outputs

To save memory, keep only some of the outputs of a model. To save work, skip
the modules whose outputs are already in the input.

### Keep fewer outputs

Each module adds its outputs to the tensordict, so the tensordict grows with
every layer. When autograd records the graph, it keeps these tensors anyway.
Without autograd, for example at inference, you can drop them. Call
[`select_out_keys()`](reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.select_out_keys) with the keys to
keep. The method changes the module in place and returns it. After that, each
call removes the other output keys from the result:

```
>>> seq = TensorDictSequential(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["h"]),
... Mod(lambda h: h * 2, in_keys=["h"], out_keys=["y"]),
... )
>>> _ = seq.select_out_keys("y")
>>> seq.out_keys
['y']
>>> td = seq(TensorDict({"x": torch.zeros(3), "other": torch.ones(3)}, batch_size=[3]))
>>> sorted(td.keys())
['other', 'x', 'y']
>>> seq(x=torch.zeros(3))
tensor([2., 2., 2.])
```

The `selected_out_keys` argument of
[`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) does the same at construction.
[`reset_out_keys()`](reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.reset_out_keys) restores all the
output keys. The `out_keys_source` attribute keeps the full list.

For a sequence, the result holds the entries of the input and the selected
keys. If an entry of the input has the name of an intermediate key (`"h"`
above), the sequence writes the new value into it.

A [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) follows the same rule. It removes
only the output keys that you did not select, and the other entries of the
input stay:

```
>>> two = Mod(lambda x: (x + 1, x + 2), in_keys=["x"], out_keys=["a", "b"]).select_out_keys("b")
>>> sorted(two(TensorDict({"x": torch.zeros(3), "other": torch.ones(3)}, batch_size=[3])).keys())
['b', 'other', 'x']
```

### Skip outputs that are already there

By default, a module computes its outputs and overwrites the entries that are
already in the input. Inside [`set_skip_existing`](reference/generated/tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing), a
module whose `out_keys` are all in the input returns the input unchanged.
The rule applies to the `out_keys` of a sequence first, then to the
`out_keys` of each of its modules:

```
>>> seq = TensorDictSequential(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["h"]),
... Mod(lambda h: h * 2, in_keys=["h"], out_keys=["y"]),
... )
>>> from tensordict.nn import set_skip_existing
>>> td = TensorDict({"x": torch.ones(2), "h": torch.zeros(2)}, batch_size=[2])
>>> with set_skip_existing(True):
... td = seq(td)
>>> td["h"], td["y"]
(tensor([0., 0.]), tensor([0., 0.]))
>>> seq(td)["y"]
tensor([4., 4.])
```

The first module keeps the `"h"` that is in the input, and the second module
computes `"y"` from it. These rules apply:

- A module that reads one of its own `out_keys` always runs.
- `set_skip_existing(["h"])` skips only the modules whose `out_keys` are
all in the list.
- [`skip_existing()`](reference/generated/tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) returns the current value. It is
`False` outside the block.
- Skipping is not compatible with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). With
`fullgraph=True`, the call raises an error.

`set_skip_existing` also works as a decorator on the `forward` of a
[`TensorDictModuleBase`](reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) subclass (see its reference).

## Typed modules: TensorClassModuleBase

[`TensorClassModuleBase`](reference/generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase) defines a module whose input
and output are [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) instances instead of
tensordicts with string keys (see [TensorClass](tensorclass.html#tensorclass-guide)). The class gives
its input and output types as generic parameters. A type checker or an IDE
then knows the fields that `forward` reads and writes. If you rename a
field, the type checker finds the code that still uses the old name.

```
>>> from tensordict import TensorClass
>>> from tensordict.nn import TensorClassModuleBase
>>> class InputTC(TensorClass):
... a: torch.Tensor
... b: torch.Tensor
>>> class OutputTC(TensorClass):
... total: torch.Tensor
... difference: torch.Tensor
>>> class SumDiff(TensorClassModuleBase[InputTC, OutputTC]):
... def forward(self, x: InputTC) -> OutputTC:
... return OutputTC(total=x.a + x.b, difference=x.a - x.b, batch_size=x.batch_size)
>>> sum_diff = SumDiff()
>>> sum_diff(InputTC(a=torch.tensor([1.0, 2.0]), b=torch.tensor([3.0, 4.0]), batch_size=[2])).total
tensor([4., 6.])
```

[`as_td_module()`](reference/generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase.as_td_module) returns a
[`TensorClassModuleWrapper`](reference/generated/tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper), which reads and writes
tensordicts and can go into a [`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential).
The wrapper returns a new tensordict that holds only the output fields:

```
>>> td_sum_diff = sum_diff.as_td_module()
>>> td = TensorDict({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}, batch_size=[2])
>>> sorted(td_sum_diff(td).keys())
['difference', 'total']
```

In a [`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential), the modules after the
wrapper therefore see only these outputs, not its input keys. By default, the
result of the sequence also holds only these outputs. Nested
[`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) fields map to nested keys.

## Probabilistic modules

A [`ProbabilisticTensorDictModule`](reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) builds a distribution
from entries of a tensordict and writes a value from it. It has no
parameters. It reads the parameters of the distribution at `in_keys`, builds
`distribution_class(**params, **distribution_kwargs)`, and writes one value
at each key of `out_keys`. The last part of each key in `in_keys` must be
the name of an argument of the distribution, for example `"loc"` and
`"scale"` for [`Normal`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal). A dict `in_keys` maps
each argument name to a key of the tensordict.

To compute the parameters with a network, put one or more
[`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) before it, in a
[`ProbabilisticTensorDictSequential`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential). In this policy,
[`NormalParamExtractor`](reference/generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor) splits the output
of the network into a location and a positive scale:

```
>>> from torch import distributions as d
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... )
>>> from tensordict.nn.distributions import NormalParamExtractor
>>> policy = ProbabilisticTensorDictSequential(
... TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["params"]),
... TensorDictModule(NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]),
... ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=d.Normal,
... return_log_prob=True,
... ),
... )
>>> policy.out_keys
['params', 'loc', 'scale', 'action', 'action_log_prob']
>>> td = policy(TensorDict({"obs": torch.randn(5, 3)}, batch_size=[5]))
>>> td["action"].shape, td["action_log_prob"].shape
(torch.Size([5, 2]), torch.Size([5, 2]))
```

The default `distribution_class` is
[`Delta`](reference/generated/tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta), which always returns its
parameter. With it, the module copies its input to `out_keys`.

`distribution_class` can also be your own
[`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) subclass, or any
callable that returns a distribution. The module calls it with the entries
of `in_keys` as keyword arguments. To set the value that the
`DETERMINISTIC` interaction type returns (see below), give the class a
`deterministic_sample` property:

```
>>> class ClippedNormal(d.Normal):
... @property
... def deterministic_sample(self):
... return self.loc.clamp(-1.0, 1.0)
>>> clipped = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=ClippedNormal
... )
>>> clipped(TensorDict({"loc": torch.tensor([3.0, -0.5]), "scale": torch.ones(2)}, batch_size=[2]))["action"]
tensor([ 1.0000, -0.5000])
```

A [`ProbabilisticTensorDictSequential`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) also has these
methods:

- [`get_dist()`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential.get_dist) returns
the distribution.
- [`get_dist_params()`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential.get_dist_params)
runs the modules before the last one, and returns the tensordict with the
parameters.
- [`log_prob()`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential.log_prob) returns
the log-probability of the values in the tensordict.

If the sequence returns the distribution of its last module (see
`return_composite` below), these methods write the intermediate entries
(`"params"`, `"loc"`, and `"scale"` here) into the input tensordict:

```
>>> td = TensorDict({"obs": torch.randn(5, 3)}, batch_size=[5])
>>> policy.get_dist(td)
Normal(loc: torch.Size([5, 2]), scale: torch.Size([5, 2]))
>>> list(td.keys())
['obs', 'params', 'loc', 'scale']
```

A [`ProbabilisticTensorDictModule`](reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) samples again and
overwrites `out_keys` that are already in the input. To keep them, call it
inside `set_skip_existing(True)` (see Skip outputs that are already
there).

### Interaction types

The interaction type is the rule that a probabilistic module uses to get a
value from its distribution. [`InteractionType`](reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) has
these values:

- `RANDOM`: a sample, from `rsample()` if the distribution has it, else
from `sample()`.
- `MODE`, `MEDIAN`, `MEAN`: the attribute of the same name. If the
distribution has no `mean`, `MEAN` averages `n_empirical_estimate`
samples (1000 by default). If it has no `mode` or `median`, the module
raises `NotImplementedError`.
- `DETERMINISTIC`: the `deterministic_sample` attribute of the
distribution. If the distribution does not have one, the module uses the
mode for distributions with enumerable support (for example
[`Categorical`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical) and
[`Bernoulli`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.bernoulli.Bernoulli)) and the mean for the others. For a
distribution class that tensordict does not know, the module chooses from
its support and warns.

Each module has a `default_interaction_type` (`DETERMINISTIC` by
default). Inside [`set_interaction_type`](reference/generated/tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type), all modules use
the type of the block instead. It also works as a decorator, and it accepts
strings in any case:

```
>>> from tensordict.nn import InteractionType, set_interaction_type
>>> sampler = ProbabilisticTensorDictModule(
... in_keys=["logits"], out_keys=["choice"], distribution_class=d.Categorical
... )
>>> td = TensorDict({"logits": torch.tensor([[0.0, 1.0, 0.0]]).expand(8, 3)}, batch_size=[8])
>>> sampler(td)["choice"]
tensor([1, 1, 1, 1, 1, 1, 1, 1])
>>> _ = torch.manual_seed(0)
>>> with set_interaction_type("random"):
... choice = sampler(td)["choice"]
>>> choice
tensor([1, 1, 1, 2, 1, 1, 2, 1])
>>> @set_interaction_type(InteractionType.MODE)
... def act(td):
... return sampler(td)["choice"]
>>> act(td)
tensor([1, 1, 1, 1, 1, 1, 1, 1])
```

`set_interaction_type()` with no argument sets `DETERMINISTIC`.
`set_interaction_type(None)` restores the `default_interaction_type` of
each module. The function `tensordict.nn.interaction_type()`
returns the type of the current block, or `None` outside a block. TorchRL
collectors sample with `RANDOM` by default (their `exploration_type`
argument).

To draw several samples in one call, pass `num_samples`. The module then
returns a new tensordict with `num_samples` as leading batch dimensions,
and the input does not change. In `RANDOM` mode, each sample is a new
draw. With the other interaction types, the module repeats its one
deterministic value `num_samples` times.

```
>>> multi = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=d.Normal,
... num_samples=4,
... default_interaction_type="random",
... )
>>> multi(TensorDict({"loc": torch.zeros(3, 2), "scale": torch.ones(3, 2)}, batch_size=[3])).batch_size
torch.Size([4, 3])
```

The `generator` argument of
[`ProbabilisticTensorDictModule`](reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) makes the samples
reproducible without a change to the global random state. Under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), the module ignores it. See its reference.

### Where the log-probability goes

With `return_log_prob=True`, the module also writes the log-probability of
the value. The name of the entry depends on the global setting
[`composite_lp_aggregate()`](reference/generated/tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate), which is `False` by default.
Change it with [`set_composite_lp_aggregate`](reference/generated/tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate) or with the
`COMPOSITE_LP_AGGREGATE` environment variable. The table gives the keys
that the module writes:

| Case | `composite_lp_aggregate()` is `False` (default) | `composite_lp_aggregate()` is `True` |
| --- | --- | --- |
| One out-key, `"action"` | `"action_log_prob"` | `"sample_log_prob"` |
| Nested out-key, `("agent", "action")` | `("agent", "action_log_prob")` | `"sample_log_prob"` |
| One out-key and `log_prob_key="lp"` | `"lp"` | `"lp"` |
| One out-key and `log_prob_keys=["lp"]` | `"lp"` | The constructor raises `RuntimeError`. |
| [`CompositeDistribution`](reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) with the out-keys `"cont"` and `("nested", "disc")` | `"cont_log_prob"` and `("nested", "disc_log_prob")` | The same two entries, and `"sample_log_prob"`: their sum, with the batch shape |
| Two probabilistic modules in one sequence, with the out-keys `"a"` and `"b"` | `"a_log_prob"` and `"b_log_prob"` | One `"sample_log_prob"`: the second module overwrites the value of the first |

The rule for the default: the module adds `"_log_prob"` to the last part of
each out-key. Each log-probability has the shape that the distribution
gives. A [`Normal`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal) with a value of shape `[5, 2]`
gives a log-probability of shape `[5, 2]`. To get one log-probability for
each of the 5 rows, give a `distribution_class` that returns
`Independent(Normal(loc, scale), 1)`.
[`Independent`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.independent.Independent) sums the
log-probability over the last dimension.

The `log_prob` methods follow the same setting. For one distribution,
they return a tensor. For a
[`CompositeDistribution`](reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution), they return a
tensordict of `<name>_log_prob` entries by default. With `True`, they
return one tensor: the sum over all the distributions and their event
dimensions.

A module reads the setting when you build it. If the setting is different
when you call a module with `return_log_prob=True`, the module raises
`RuntimeError: composite_lp_aggregate is set to ...`. Choose the setting
before you build the modules, and keep it.

### Several heads: CompositeDistribution

A policy with several outputs, for example a continuous movement and a
discrete grip, needs one distribution for each output.
[`CompositeDistribution`](reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) groups them. Its
parameters are a tensordict with one nested tensordict for each output. Its
samples and log-probabilities are tensordicts with one entry for each output.
To use it in a module, read the parameters at one key (`"params"` below),
and give `distribution_map` in `distribution_kwargs`: a dict from each
output name to its distribution class. The module then takes its
`out_keys` from the names of `distribution_map`:

```
>>> from tensordict.nn import CompositeDistribution
>>> class Heads(nn.Module):
... def __init__(self):
... super().__init__()
... self.linear = nn.Linear(3, 2 + 2 + 4)
...
... def forward(self, obs):
... out = self.linear(obs)
... return TensorDict(
... {
... "move": {"loc": out[..., :2], "scale": out[..., 2:4].exp()},
... "grip": {"logits": out[..., 4:]},
... },
... batch_size=obs.shape[:-1],
... )
>>> policy = ProbabilisticTensorDictSequential(
... TensorDictModule(Heads(), in_keys=["obs"], out_keys=["params"]),
... ProbabilisticTensorDictModule(
... in_keys=["params"],
... distribution_class=CompositeDistribution,
... distribution_kwargs={"distribution_map": {"move": d.Normal, "grip": d.Categorical}},
... return_log_prob=True,
... default_interaction_type="random",
... ),
... )
>>> td = policy(TensorDict({"obs": torch.randn(5, 3)}, batch_size=[5]))
>>> td["move"].shape, td["grip"].shape
(torch.Size([5, 2]), torch.Size([5]))
>>> list(policy.log_prob(td).keys())
['move_log_prob', 'grip_log_prob']
```

The `name_map` argument in `distribution_kwargs` writes the samples at
other keys. You can also chain several
[`ProbabilisticTensorDictModule`](reference/generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) instances in one
[`ProbabilisticTensorDictSequential`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential). A later module can
then read the sample of an earlier one, and
[`get_dist()`](reference/generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential.get_dist) returns a
[`CompositeDistribution`](reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) of all of them.
The `return_composite` argument controls this. By default, it is `True`
if there is more than one probabilistic module or if the last module is not
probabilistic.

## Parameters as tensordicts

The parameters of a model fit in a tensordict, with one nested key for each
submodule. With such a tensordict, you can make a functional call: a call of
a module with parameters that the module does not own. Use functional calls
for ensembles, target networks, meta-learning, and per-sample gradients.

### Extract and swap parameters

[`from_module()`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.from_module) returns the parameters and buffers
of a module in a tensordict. The entries are the module's own
[`Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) objects, so an in-place change to them changes
the module. [`to_module()`](reference/generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) puts the entries of a
tensordict into a module. In a `with` block, it puts the original
parameters back at the end of the block:

```
>>> mlp = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
>>> params = TensorDict.from_module(mlp)
>>> list(params.keys(include_nested=True, leaves_only=True))
[('0', 'weight'), ('0', 'bias'), ('2', 'weight'), ('2', 'bias')]
>>> params["0", "weight"] is mlp[0].weight
True
>>> zeros = params.detach().clone().zero_()
>>> with zeros.to_module(mlp):
... y = mlp(torch.randn(2, 3))
>>> bool((y == 0).all())
True
>>> bool(mlp[0].weight.abs().sum() > 0)
True
```

After the block, `mlp` has its own parameters again. Without `with`,
`to_module` keeps the new parameters in the module, and
returns a tensordict with the old ones.

### Ensembles with from_modules and vmap

[`from_modules()`](reference/generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules) stacks the parameters of several
modules of the same structure into one tensordict, with a leading batch
dimension. [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) over this tensordict calls one module with each
set of parameters:

```
>>> nets = [nn.Linear(3, 1) for _ in range(4)]
>>> stacked = TensorDict.from_modules(*nets)
>>> stacked.batch_size
torch.Size([4])
>>> base = nn.Linear(3, 1)
>>> def call(params, x):
... with params.to_module(base):
... return base(x)
>>> x = torch.randn(10, 3)
>>> y = torch.vmap(call, (0, None))(stacked, x)
>>> y.shape
torch.Size([4, 10, 1])
>>> torch.testing.assert_close(y[2], nets[2](x))
```

The [functional tutorial](tutorials/functional.html) explains this pattern
in more detail. [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile) shows it under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) and with
[`torch.func.grad()`](https://docs.pytorch.org/docs/stable/generated/torch.func.grad.html#torch.func.grad) for per-sample gradients.

[`EnsembleModule`](reference/generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule) does the same for a
[`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule). It makes `num_copies` copies of
the parameters of the module, initializes each copy again, and calls the
module on the input with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap). The output has a leading batch
dimension of size `num_copies`:

```
>>> from tensordict.nn import EnsembleModule
>>> ensemble = EnsembleModule(
... TensorDictModule(nn.Linear(3, 2), in_keys=["x"], out_keys=["y"]), num_copies=4
... )
>>> ensemble(TensorDict({"x": torch.randn(10, 3)}, batch_size=[10]))["y"].shape
torch.Size([4, 10, 2])
>>> ensemble.params_td.batch_size
torch.Size([4])
```

The copies live in `ensemble.params_td`, a
[`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams), so `ensemble.parameters()` returns
them for the optimizer. `ensemble.parameters()` also returns the parameters
of the module that you passed. The forward pass does not use them, and they
get no gradient. To chain two ensembles, build the second one with
`expand_input=False`. Each of its copies then reads the matching slice of
the output of the first one.

### Keep parameters on a module: TensorDictParams

A plain [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) attribute of a module is not seen by
`module.parameters()`, `module.state_dict()`, or `module.to()`. An
optimizer then does not train it, and a checkpoint does not save it.
[`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) is a tensordict whose entries are
registered as parameters and buffers of the module that holds it. Use it to
keep a tensordict of parameters on a module, for example the stacked weights
of an ensemble or a copy of the weights for a target network.

A target network must not be trained by the optimizer. In this example,
`no_convert=True` registers the copy as buffers: `state_dict()` and
`to()` see them, `parameters()` does not.

```
>>> from tensordict import TensorDictParams
>>> class Critic(nn.Module):
... def __init__(self, net):
... super().__init__()
... self.net = net
... target = TensorDict.from_module(net).detach().clone()
... self.target_params = TensorDictParams(target, no_convert=True)
>>> critic = Critic(nn.Linear(2, 1))
>>> list(critic.state_dict().keys())
['net.weight', 'net.bias', 'target_params.weight', 'target_params.bias']
>>> [name for name, _ in critic.named_parameters()]
['net.weight', 'net.bias']
>>> critic = critic.to(torch.float64)
>>> critic.target_params["weight"].dtype
torch.float64
>>> with critic.target_params.to_module(critic.net):
... target_value = critic.net(torch.zeros(1, 2, dtype=torch.float64))
```

A [`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) behaves like the tensordict that it
wraps: you can index it, set entries, and call `to_module`. These rules
apply:

- When you set an entry, `float16`, `float32`, and `float64` tensors
become [`Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter). Tensors of all other dtypes, including
`bfloat16`, complex, and `bool`, become buffers. An optimizer does not see
a buffer. With `no_convert=True`, all tensors that are not already
parameters become buffers.
- `named_parameters()` names each entry by its nested key, joined with
`"."`, for example `"0.weight"`.
- The constructor does not copy the tensordict that you pass. It turns the
leaves of that tensordict into parameters and buffers, in place. If you
later add or replace an entry of that tensordict, indexing sees the new
value, but `parameters()` and `buffers()` do not. Write through the
[`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) instead.

In this example, only the `float32` entry becomes a parameter:

```
>>> p = TensorDictParams(
... TensorDict(
... {
... "f32": torch.zeros(2),
... "bf16": torch.zeros(2, dtype=torch.bfloat16),
... "flags": torch.zeros(2, dtype=torch.bool),
... },
... )
... )
>>> [name for name, _ in p.named_parameters()]
['f32']
>>> sorted(name for name, _ in p.named_buffers())
['bf16', 'flags']
```

## Compile and capture modules

The modules of [`tensordict.nn`](reference/nn.html#module-tensordict.nn) work with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile),
[`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func), and [`torch.export.export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export). A module that writes into
its input recompiles when the key set of the input changes. For this limit,
the others, and their workarounds, see [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile). To record a module as
a CUDA graph and replay it, use [`CudaGraphModule`](reference/generated/tensordict.nn.CudaGraphModule.html#tensordict.nn.CudaGraphModule) (see
[CUDA graphs](compile.html#compile-cudagraphs)).

## See also

- [tensordict.nn](reference/nn.html#api-nn): the reference of every class and function on this page.
- [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile): [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func),
[`torch.export.export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export), and CUDA graphs with tensordict modules.
- The [TensorDictModule tutorial](tutorials/tensordict_module.html): a
residual block built step by step from tensordict modules, plain functions
in a sequence, and a first probabilistic module.
- The [functional tutorial](tutorials/functional.html): functional calls,
ensembles, and `vmap` with `from_module` and `to_module`.
- The [export tutorial](tutorials/export.html): export a tensordict module
with [`torch.export.export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export), and select its outputs first.
- [TorchRL](https://pytorch.org/rl/): policies, value networks, and losses
built from these modules.