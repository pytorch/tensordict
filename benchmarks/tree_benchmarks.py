import timeit
from itertools import combinations_with_replacement
from string import ascii_lowercase

import torch
from tensordict import TensorDict
from tensordict.prototype import make_tree

if __name__ == "__main__":
    td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    keys = []
    for depth in range(1, 10):
        keys.extend(combinations_with_replacement(("left", "right"), depth))

    for key in keys:
        td[key] = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

    root = make_tree(td, n_nodes=2048)
    keys = [(*k, "a") for k in keys]

    print("get_multiple_items")
    print(
        "tensordict:",
        timeit.timeit(
            "stack([td[key] for key in keys], 0)",
            globals={"td": td, "keys": keys, "stack": torch.stack},
            number=10_000,
        ),
    )

    print(
        "tree:",
        timeit.timeit(
            "root.get_multiple_items(*keys)",
            globals={"root": root, "keys": keys},
            number=10_000,
        ),
        end="\n\n",
    )

    td_deep = TensorDict({"data": torch.rand(2, 3, 4)}, [2, 3])
    for i in range(len(ascii_lowercase)):
        td_deep[tuple(ascii_lowercase[: i + 1])] = TensorDict(
            {"data": torch.rand(2, 3, 4)}, [2, 3]
        )

    root_deep = make_tree(td_deep, n_nodes=100)
    keys_deep = [
        tuple(ascii_lowercase[:i]) + ("data",) for i in range(len(ascii_lowercase) + 1)
    ]

    print("get_multiple_items_deep")
    print(
        "tensordict",
        timeit.timeit(
            "stack([td[key] for key in keys], 0)",
            globals={"td": td_deep, "keys": keys_deep, "stack": torch.stack},
            number=10_000,
        ),
    )

    print(
        "tree",
        timeit.timeit(
            "root.get_multiple_items(*keys)",
            globals={"root": root_deep, "keys": keys_deep},
            number=10_000,
        ),
    )
