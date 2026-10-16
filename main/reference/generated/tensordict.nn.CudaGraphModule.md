# CudaGraphModule

*class*tensordict.nn.CudaGraphModule(*module: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [None](https://docs.python.org/3/builtins/constants.html#None)]*, *warmup: [int](https://docs.python.org/3/builtins/functions.html#int) = 2*, *in_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = None*, *out_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Records a function, or a TensorDictModule, as a CUDA graph and replays it.

A replay runs all the recorded kernels with one CPU call, which removes the
CPU overhead of launching them one by one. The first `warmup - 1` calls
run the function on a side stream. The next call records the graph, and
later calls copy their inputs into the graph's buffers and replay it. Each
call checks that the inputs match the recorded ones (see the rules below).
See [CUDA graphs](../../compile.html#compile-cudagraphs) for a worked example.

Without CUDA, the constructor warns, and each call runs the function
directly.

Warning

`CudaGraphModule` is a prototype feature and its API restrictions are likely to change in the future.

Warning

This module requires the wrapped function to meet a few requirements. It is the user responsibility
to make sure that all of these are fulfilled.

> - The function cannot have dynamic control flow. For instance, the following code snippet will fail to be
> wrapped in CudaGraphModule:
> 
> 
> 
> 
> ```
> def func(x):
> if x.norm() > 1:
> return x + 1
> else:
> return x - 1
> ```
> 
> 
> 
> 
> 
> Use [`torch.where()`](https://docs.pytorch.org/docs/stable/generated/torch.where.html#torch.where) instead:
> 
> 
> 
> 
> ```
> def func(x):
> return torch.where(x.norm() > 1, x + 1, x - 1)
> ```
> - The function must execute a code that can be exactly re-run using the same buffers. This means that
> dynamic shapes (changing shape in the input or during the code execution) is not supported. In other words,
> the input must have a constant shape. After the capture, a tensor input with a different shape, or a
> tensordict input with a different batch size or an entry of a different shape, raises a `ValueError`, and
> a tensordict input that lacks one of the `in_keys` raises a `KeyError`.
> - The output of the function must be detached. If a call to the optimizers is required, put it in the input
> function. For instance, the following function is a valid operator:
> 
> 
> 
> 
> ```
> def func(x, y):
> optim.zero_grad()
> loss_val = loss_fn(x, y)
> loss_val.backward()
> optim.step()
> return loss_val.detach()
> ```
> - The inputs must not require gradients. If the function uses `nn.Parameter` instances (or other tensors
> that require gradients), do not pass them as arguments of the call. Read them from the enclosing scope, or
> bind them as default values, as in `def func(x, w=w)`:
> 
> 
> 
> 
> ```
> w = nn.Parameter(torch.randn((), device="cuda"))
> optim = torch.optim.Adam([w], lr=1)
> 
> def func(x): # right: w is not an argument of the call
> optim.zero_grad()
> (x * w).backward()
> optim.step()
> 
> def func(x, w): # wrong: w is passed at each call
> ...
> ```
> - Tensor and tensordict arguments can change between calls, if their device and shape stay the same.
> Arguments that are not tensors (a `float`, a `str`) must not change after the capture. If the
> function takes tensors, a changed value raises a `ValueError`. If it takes a tensordict, the replay uses
> the captured value, ignores the new one and emits a `FutureWarning`. Starting with TensorDict 0.17, the
> default will change to raising a `ValueError` there too (pytorch/tensordict#2075).
> The keyword arguments of a tensordict module are its `in_keys` (see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch))
> and `tensordict_out`.
> - If the module is a [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) instance and the output id matches the input
> id, then this identity will be preserved during a call to `CudaGraphModule`, and the outputs written into
> the input are clones of the graph's output buffers. In all other cases, the output will be cloned,
> irrespective of whether its elements match or do not match one of the inputs.
> - In-place writes to the input tensors inside the function (e.g. `x.add_(1)`) do not reach the caller's
> inputs once the graph is captured: the graph reads and writes its own copies of the inputs.
> - The inputs are copied into the graph's buffers, and the graph is replayed, on the current stream. As for any
> CUDA operation, inputs written on another stream must be ordered before the call by the caller, e.g. with
> `torch.cuda.current_stream().wait_stream(producer_stream)`. Consecutive calls share the graph's buffers:
> issue them from one stream, or order each call after the previous one in the same way.
> - The replay repeats the branch that each [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) took at
> capture: a graph captured under `set_interaction_type("random")` keeps sampling under
> `set_interaction_type("deterministic")`. The first replay under another interaction type emits a
> `UserWarning`. The check covers only the [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) instances
> inside the wrapped [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module): other code that branches on
> `interaction_type()` (for example an exploration module) and wrapped
> functions that are not modules are not checked; if the wrapped module holds no
> [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule), the replay repeats the captured branch without a
> warning. Capture one `CudaGraphModule` per interaction type whenever the output depends on it.

Warning

`CudaGraphModule` is not an [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) by design, to discourage gathering parameters
of the input module and passing them to an optimizer.

Parameters:

- **module** (*Callable*) - a function that receives tensors (or tensordict) as input and outputs a
PyTreeable collection of tensors. If a tensordict is provided, the module can be run with keyword arguments
too (see example below).
- **warmup** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the call that records the graph. The first `warmup - 1` calls run the function
on a side stream, the next call records the graph, and later calls replay it. A compiled function needs
enough calls to finish compiling before the capture, for example
`CudaGraphModule(torch.compile(module), warmup=3)`. Defaults to `2`.
- **in_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**NestedKeys*) - 

the input keys, if the module takes a TensorDict as input.
Defaults to `module.in_keys` if this value exists, otherwise `None`.

Note

If `in_keys` is provided but empty, the module is assumed to receive a tensordict as input.
This is sufficient to make `CudaGraphModule` aware that the function should be treated as a
TensorDictModule, but keyword arguments will not be dispatched. See below for some examples.
- **out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**NestedKeys*) - the output keys, if the module takes and outputs TensorDict as output.
Defaults to `module.out_keys` if this value exists, otherwise `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of the stream to use.

Examples

The example needs a CUDA device, so its lines are skipped when the
docs are tested.

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import CudaGraphModule, TensorDictModule
>>> # Wrap a simple function
>>> def func(x):
... return x + 1
>>> func = CudaGraphModule(func) 
>>> x = torch.rand((), device='cuda') 
>>> out = func(x) 
>>> assert isinstance(out, torch.Tensor) 
>>> assert out == x+1 
>>> # Wrap a tensordict module
>>> func = TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["y"])
>>> func = CudaGraphModule(func) 
>>> # This can be called either with a TensorDict or regular keyword arguments alike
>>> y = func(x=x) 
>>> td = TensorDict(x=x) 
>>> td = func(td)
```

Note

Tips on debugging CudaGraphModule errors:

- Errors of the sort of operation would make the legacy stream depend on a capturing blocking stream are

to be debugged using a non-compiled version of your code first (compiled code will hide the part of the
code that is responsible for the cross-stream dependency). It can happen because you are
doing inter-device operations which make the capturing stream dependent on other streams.
- Errors like Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture or other errors originating

from the compiler (not the compiled code!) probably point to a capture of the graph while recompiling.
Capture the recompiles with TORCH_LOGS="+recompiles" python myscript.py and try to fix these.
In general, make sure you are using enough warmup steps.
Please file an issue if you're struggling with these.

## Tutorials that use `CudaGraphModule`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule