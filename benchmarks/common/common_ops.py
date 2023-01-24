# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import timeit

from argparse import ArgumentParser

import git

import torch
from tensordict import TensorDict


# use this to checkout a branch
def checkout(branch):

    repo = git.Repo("../..")
    current = git.Repo("..").git.branch("--show-current")
    repo.git.checkout("speed")
    return current


def main():
    # creation (empty)
    cmd = "TensorDict({}, [3, 4])"
    print(
        cmd, min(*timeit.repeat(cmd, globals={"TensorDict": TensorDict}, number=10000))
    )

    # creation (empty)
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    cmd = "TensorDict({'a': a, 'b': b}, [3, 4])"
    print(
        cmd,
        min(
            *timeit.repeat(
                cmd, globals={"TensorDict": TensorDict, "a": a, "b": b}, number=10000
            )
        ),
    )

    # creation nested 1
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    cmd = "TensorDict({'a': a, ('b', 'b1'): b}, [3, 4])"
    print(
        cmd,
        min(
            *timeit.repeat(
                cmd, globals={"TensorDict": TensorDict, "a": a, "b": b}, number=10000
            )
        ),
    )

    # creation nested 2
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    cmd = "TensorDict({'a': a, 'b': {'b1': b}}, [3, 4])"
    print(
        cmd,
        min(
            *timeit.repeat(
                cmd, globals={"TensorDict": TensorDict, "a": a, "b": b}, number=10000
            )
        ),
    )

    # clone
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone()"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td}, number=10000)))

    # __setitem__
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone();tdc['c'] = c"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td, "c": c}, number=10000)))

    # set
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone();tdc.set('c', c)"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td, "c": c}, number=10000)))

    # set shared
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone();tdc.share_memory_()"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td}, number=10000)))

    # update
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": b}, [3, 4])
    td2 = td.clone()
    cmd = "tdc = td.clone();tdc.update(td2)"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td, "td2": td2}, number=10000)))

    # update nested
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    td2 = td.clone()
    cmd = "tdc = td.clone();tdc.update(td2)"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td, "td2": td2}, number=10000)))

    # set nested
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone();tdc['b', 'b1'] = b"
    print(
        cmd, min(*timeit.repeat(cmd, globals={"td": td, "b": b.clone()}, number=10000))
    )

    # set nested new
    a = torch.zeros(3, 4, 5)
    b = torch.zeros(3, 4, 5)
    c = torch.zeros(3, 4, 5)
    td = TensorDict({"a": a, "b": {"b1": b}}, [3, 4])
    cmd = "tdc = td.clone();tdc['c', 'c', 'c'] = c"
    print(cmd, min(*timeit.repeat(cmd, globals={"td": td, "c": c}, number=10000)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--branch", default="main", help="branch where to run the benchmark"
    )
    args = parser.parse_args()
    # current = checkout(args.branch)
    main()
    # checkout(current)
