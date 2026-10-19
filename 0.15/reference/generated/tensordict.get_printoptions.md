# get_printoptions

tensordict.get_printoptions() → [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)

Returns the current print options of tensordicts as a dict.

The keys are the options of [`set_printoptions`](tensordict.set_printoptions.html#tensordict.set_printoptions). See
[Printing and display](../../printing.html#printing).

Examples

```
>>> from tensordict import get_printoptions
>>> get_printoptions()["show_dtype"]
True
```