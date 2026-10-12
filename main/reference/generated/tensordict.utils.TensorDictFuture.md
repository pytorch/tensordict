# TensorDictFuture

*class*tensordict.utils.TensorDictFuture(*futures*, *resulting_td*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A custom future class for TensorDict multithreaded operations.

Parameters:

- **futures** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**futures*) - a list of concurrent.futures.Future objects to wait for.
- **resulting_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - instance that will result from the futures
completing.

result()

Wait and returns the resulting tensordict.