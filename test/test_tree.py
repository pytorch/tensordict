import pytest
import torch
from tensordict import TensorDict
from tensordict.prototype import make_tree
from tensordict.prototype.tree import _TensorDictNode


@pytest.mark.parametrize("n_nodes", [10, 20, 100])
def test_create_root(n_nodes):
    a = torch.rand(2, 3, 4)
    td = TensorDict({"a": a}, [2, 3])
    root = make_tree(td, n_nodes=n_nodes)

    torch.testing.assert_close(a, root["a"])
    assert root.source.batch_size == torch.Size([n_nodes, 2, 3])
    torch.testing.assert_close(a, root.source["a"][0])
    assert root.source.numel() == 1
    assert root.source.cursor == 1
    assert root.source._node_indices == {0}


def test_tree_from_nested_tensordict():
    left_left = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    left_right = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    left = TensorDict(
        {"a": torch.rand(2, 3, 4), "left": left_left, "right": left_right}, [2, 3]
    )
    right = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    td = TensorDict({"a": torch.rand(2, 3, 4), "left": left, "right": right}, [2, 3])
    root = make_tree(td, n_nodes=10)
    source = root.source

    assert source.numel() == 5
    assert set(root.keys(include_nested=True, leaves_only=True)) == {
        "a",
        ("left", "a"),
        ("left", "left", "a"),
        ("left", "right", "a"),
        ("right", "a"),
    }
    torch.testing.assert_close(root["left", "right", "a"], left_right["a"])


def test_add_nested_value():
    a = torch.rand(2, 3, 4)
    td = TensorDict({"a": a}, [2, 3])
    root = make_tree(td, n_nodes=10)

    root["left"] = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

    assert root.source.numel() == 2
    assert isinstance(root["left"], _TensorDictNode)
    assert (root["left", "a"] == root["left"]["a"]).all()
    torch.testing.assert_close(root["left", "a"], root.source["a"][root["left"].idx])


def test_add_tensordict():
    a = torch.rand(2, 3, 4)
    td = TensorDict({"a": a}, [2, 3])
    root = make_tree(td, n_nodes=10)

    left = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    root["left"] = TensorDict({"a": torch.rand(2, 3, 4), "left": left}, [2, 3])
    source = root.source

    assert source.numel() == 3
    torch.testing.assert_close(left["a"], root["left", "left", "a"])


def test_delete_nodes():
    td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    root = make_tree(td, n_nodes=10)

    # collect the indicies of each node in the left branch
    indices = set()
    for key in ["left", ("left", "left"), ("left", "right")]:
        root[key] = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
        indices.add(root[key].idx[0])

    assert root.source.numel() == 4
    del root["left"]
    assert root.source.numel() == 1
    # none of the left branch indices should be in the tree anymore
    assert len(root.source._node_indices & indices) == 0

    with pytest.raises(KeyError):
        root["left"]


def test_get_multiple_items():
    td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])
    root = make_tree(td, n_nodes=10)

    keys = [
        ("left",),
        ("left", "left"),
        ("left", "right"),
        ("right",),
        ("right", "right"),
    ]
    for key in keys:
        root[key] = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

    keys = [k + ("a",) for k in keys]

    assert root.source.numel() == 6
    assert (
        root.get_multiple_items(*keys) == torch.stack([root[k] for k in keys])
    ).all()
    # get_multiple_items is relative
    left_keys = [k for k in keys if k[0] == "left"]
    left = root["left"]
    assert (
        root.get_multiple_items(*left_keys)
        == left.get_multiple_items(*(k[1:] for k in left_keys))
    ).all()


def test_apply():
    td = TensorDict({"a": -2 * torch.ones(2, 3, 4)}, [2, 3])
    root = make_tree(td, n_nodes=10)

    root.apply_(torch.abs)

    assert (root["a"] == 2).all()


def test_key_checks():
    td = TensorDict({"a": torch.ones(2, 3, 4), "b": torch.zeros(2, 3, 4)}, [2, 3])
    root = make_tree(td, n_nodes=10)

    with pytest.raises(
        KeyError, match="All nodes must have the same leaf keys as the root node."
    ):
        root["left"] = TensorDict({"b": torch.rand(2, 3, 4)}, [2, 3])

    with pytest.raises(
        KeyError, match="All nodes must have the same leaf keys as the root node."
    ):
        root["left"] = torch.rand(2, 3, 4)

    with pytest.raises(
        KeyError, match="All nodes must have the same leaf keys as the root node."
    ):
        root["left"] = TensorDict(
            {
                "a": torch.rand(2, 3, 4),
                "b": torch.rand(2, 3, 4),
                "c": torch.rand(2, 3, 4),
            },
            [2, 3],
        )
