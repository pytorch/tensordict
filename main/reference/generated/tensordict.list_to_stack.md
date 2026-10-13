# list_to_stack

tensordict.list_to_stack(*allow_none=False*)

Returns whether assigning a list to a key stores a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack).

See [`set_list_to_stack`](tensordict.set_list_to_stack.html#tensordict.set_list_to_stack).

Parameters:

**allow_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the option was never
set, `None` is returned instead of the default. Defaults to
`False`.