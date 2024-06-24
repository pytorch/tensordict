# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from tensordict import TensorDict
from tensordict.nn.storage import (
    BinaryToDecimal,
    DynamicStorage,
    FixedStorage,
    QueryModule,
    SipHash,
    TensorDictStorage,
)


def test_embedding_memory():
    embedding_storage = FixedStorage(
        torch.nn.Embedding(num_embeddings=10, embedding_dim=2),
        lambda x: torch.nn.init.constant_(x, 0),
    )

    index = torch.Tensor([1, 2]).long()
    assert len(embedding_storage) == 0
    assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()

    embedding_storage[index] = torch.ones(size=(2, 2))
    assert torch.sum(embedding_storage.contain(index)).item() == 2

    assert (embedding_storage[index] == torch.ones(size=(2, 2))).all()

    assert len(embedding_storage) == 2
    embedding_storage.clear()
    assert len(embedding_storage) == 0
    assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()


def test_binary_to_decimal():
    binary_to_decimal = BinaryToDecimal(
        num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
    )
    binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
    decimal = binary_to_decimal(binary)

    assert decimal.shape == (2,)
    assert (decimal == torch.Tensor([3, 2])).all()


def test_query():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    query = TensorDict(
        {
            "key1": torch.Tensor([[1], [1], [1], [2]]),
            "key2": torch.Tensor([[3], [3], [2], [3]]),
        },
        batch_size=(4,),
    )
    res = query_module(query)
    assert res["index"][0] == res["index"][1]
    for i in range(1, 3):
        assert res["index"][i].item() != res["index"][i + 1].item(), (
            f"{i} = ({query[i]['key1']}, {query[i]['key2']}) s index and {i + 1} = ({query[i + 1]['key1']}, "
            f"{query[i + 1]['key2']})'s index are the same!"
        )


def test_query_module():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    embedding_storage = FixedStorage(
        torch.nn.Embedding(num_embeddings=23, embedding_dim=1),
        lambda x: torch.nn.init.constant_(x, 0),
    )

    tensor_dict_storage = TensorDictStorage(
        query_module=query_module,
        key_to_storage={"index": embedding_storage},
    )

    index = TensorDict(
        {
            "key1": torch.Tensor([[-1], [1], [3], [-3]]),
            "key2": torch.Tensor([[0], [2], [4], [-4]]),
        },
        batch_size=(4,),
    )

    value = TensorDict(
        {"index": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
    )

    tensor_dict_storage[index] = value
    assert torch.sum(tensor_dict_storage.contain(index)).item() == 4

    new_index = index.clone(True)
    new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
    retrieve_value = tensor_dict_storage[new_index]

    assert (retrieve_value["index"] == value["index"]).all()


def test_storage():
    query_module = QueryModule(
        in_keys=["key1", "key2"],
        index_key="index",
        hash_module=SipHash(),
    )

    embedding_storage = DynamicStorage(default_tensor=torch.zeros((1,)))

    tensor_dict_storage = TensorDictStorage(
        query_module=query_module,
        key_to_storage={"index": embedding_storage},
    )

    index = TensorDict(
        {
            "key1": torch.Tensor([[-1], [1], [3], [-3]]),
            "key2": torch.Tensor([[0], [2], [4], [-4]]),
        },
        batch_size=(4,),
    )

    value = TensorDict(
        {"index": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
    )

    tensor_dict_storage[index] = value
    assert torch.sum(tensor_dict_storage.contain(index)).item() == 4

    new_index = index.clone(True)
    new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
    retrieve_value = tensor_dict_storage[new_index]

    assert (retrieve_value["index"] == value["index"]).all()
