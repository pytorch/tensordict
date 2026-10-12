# print_directory_tree

tensordict.utils.print_directory_tree(*path*, *indent=''*, *display_metadata=True*) → [str](https://docs.python.org/3/builtins/stdtypes.html#str)

Prints the directory tree starting from the specified path.

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The path of the directory to print.
- **indent** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The current indentation level for formatting.
- **display_metadata** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, metadata of the dir will be
displayed too.

Returns:

the string printed with the logger.