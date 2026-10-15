# Buffer

*class*tensordict.utils.Buffer(*data=None*, ***, *persistent=True*)

A kind of Tensor that should not be considered a model
parameter. For example, BatchNorm's `running_mean` is not a parameter, but is part of the module's state.

Buffers are [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) subclasses, that have a
very special property when used with `Module` s - when they're
assigned as Module attributes they are automatically added to the list of
its buffers, and will appear e.g. in [`buffers()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.buffers) iterator.
Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
the [`register_buffer()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) function.

Parameters:

- **data** (*Tensor*) - buffer tensor.
- **persistent** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the buffer is part of the module's
`state_dict`. Default: `True`