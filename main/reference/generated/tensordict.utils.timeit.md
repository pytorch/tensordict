# timeit

*class*tensordict.utils.timeit(*name*)

Measures the time spent in a function or a block of code, by name.

Use it as a decorator or as a context manager. Each name accumulates
its call count, its total time and its mean time.
`timeit.print()` logs them with the tensordict logger, and
`timeit.erase()` resets them.

Parameters:

**name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the name under which the time is recorded.

Examples

```
>>> from tensordict.utils import timeit
>>> @timeit("step")
... def step():
... return 1
>>> _ = step()
>>> with timeit("block"):
... _ = step()
>>> timeit.print() # logs "block took ... msec" and "step took ... msec"
>>> timeit.erase()
```