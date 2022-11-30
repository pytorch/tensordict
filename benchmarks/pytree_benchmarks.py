import timeit

import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map

if __name__ == "__main__":
    nested_dict = {
        "a": {"b": torch.randn(3, 4, 1), "c": {"d": torch.rand(3, 4, 5, 6)}},
        "c": torch.rand(3, 4, 1),
    }
    nested_td = TensorDict(nested_dict, [3, 4], _run_checks=False)

    # # reshape
    # print(
    #     "reshape, pytree",
    #     timeit.timeit(
    #         "tree_map(lambda x: x.reshape(12, *x.shape[2:]), nested_dict)",
    #         globals={"tree_map": tree_map, "nested_dict": nested_dict},
    #         number=10000,
    #     ),
    # )
    # print(
    #     "reshape, td",
    #     timeit.timeit(
    #         "nested_td.reshape(12)", globals={"nested_td": nested_td}, number=10000
    #     ),
    # )
    #
    # # view
    # print(
    #     "view, pytree",
    #     timeit.timeit(
    #         "tree_map(lambda x: x.view(12, *x.shape[2:]), nested_dict)",
    #         globals={"tree_map": tree_map, "nested_dict": nested_dict},
    #         number=10000,
    #     ),
    # )
    # print(
    #     "view, td",
    #     timeit.timeit(
    #         "nested_td.view(12)", globals={"nested_td": nested_td}, number=10000
    #     ),
    # )
    #
    # # unbind
    # print(
    #     "unbind, pytree",
    #     timeit.timeit(
    #         "tree_map(lambda x: x.unbind(0), nested_dict)",
    #         globals={"tree_map": tree_map, "nested_dict": nested_dict},
    #         number=10000,
    #     ),
    # )
    # print(
    #     "unbind, td",
    #     timeit.timeit(
    #         "nested_td.unbind(0)", globals={"nested_td": nested_td}, number=10000
    #     ),
    # )
    #
    # # split
    # print(
    #     "split, pytree",
    #     timeit.timeit(
    #         "tree_map(lambda x: x.split([1, 2], 0), nested_dict)",
    #         globals={"tree_map": tree_map, "nested_dict": nested_dict},
    #         number=10000,
    #     ),
    # )
    # print(
    #     "split, td",
    #     timeit.timeit(
    #         "nested_td.split([1, 2], 0)", globals={"nested_td": nested_td}, number=10000
    #     ),
    # )
    #
    # # add
    # print(
    #     "add, pytree",
    #     timeit.timeit(
    #         "tree_map(lambda x: x + 1, nested_dict)",
    #         globals={"tree_map": tree_map, "nested_dict": nested_dict},
    #         number=10000,
    #     ),
    # )
    print(
        "add, td",
        timeit.timeit(
            "nested_td.apply(lambda x: x+1)",
            globals={"nested_td": nested_td},
            number=100000,
        ),
    )
