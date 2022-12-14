import heapq

import torch
from tensordict.metatensor import MetaTensor
from tensordict.tensordict import (
    _TensorDictKeysView,
    SubTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import _is_shared

KEY_ERR = (
    "All nodes must have the same leaf keys as the root node. The leaf keys on the "
    "root node are {leaf_keys}, but you are trying to set the key(s) {key}."
)


class _TensorDictNodeKeysView(_TensorDictKeysView):
    # a custom keys view class for tensordict nodes which merges keys
    # from the tensordict and keys corresponding to the children
    def _keys(self):
        return (*self.tensordict._source.keys(), *self.tensordict._children)

    def _items(self, tensordict=None):
        if tensordict is None:
            tensordict = self.tensordict
        if isinstance(tensordict, _TensorDictNode):
            return (*tensordict._source.items(), *tensordict._children.items())
        return super()._items(tensordict)


class _TensorDictNode(SubTensorDict):
    # a basic node class that inherits from SubTensorDict and implements simple
    # get and set operations, as well as `keys` and `_make_meta` to make sure
    # that __repr__ displays the right thing
    def __init__(self, source, idx):
        super().__init__(source, idx)
        self._children = {}

    def keys(self, include_nested=False, leaves_only=False):
        return _TensorDictNodeKeysView(self, include_nested, leaves_only)

    @property
    def source(self):
        return self._source

    @source.setter
    def _(self):
        raise RuntimeError(
            "Source should not be updated manually, it is managed by the nodes and "
            "`make_tree`."
        )

    def _make_meta(self, key):
        if key in self._children:
            # entries of self._children are always nodes, so we can simplify slightly
            # compared to the logic in other implementations of _make_meta
            out = self._children[key]
            is_memmap = self._is_memmap if self._is_memmap is not None else False
            is_shared = (
                self._is_shared if self._is_shared is not None else _is_shared(out)
            )
            return MetaTensor(
                out,
                device=out.device,
                _is_memmap=is_memmap,
                _is_shared=is_shared,
                _is_tensordict=True,
            )
        return super()._make_meta(key)

    def __setitem__(self, key, item):
        key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
        if isinstance(key, tuple) or isinstance(item, TensorDictBase):
            if isinstance(key, tuple):
                key, subkey = key[0], key[1:]
            else:
                subkey = ()
            if key in self._children:
                node = self._children[key]
            else:
                node = self._source.add_node()
                self._children[key] = node
            if subkey:
                node[subkey] = item
            elif isinstance(item, TensorDictBase):
                item_keys = set(item.keys(leaves_only=True))
                if item_keys != self.source._leaf_keys:
                    raise KeyError(
                        KEY_ERR.format(leaf_keys=self.source._leaf_keys, key=item_keys)
                    )
                for k, v in item.items():
                    node[k] = v
        else:
            if key not in self.source._leaf_keys:
                raise KeyError(
                    KEY_ERR.format(leaf_keys=self.source._leaf_keys, key=key)
                )
            super().__setitem__(key, item)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, subkey = key[0], key[1:]
        else:
            subkey = ()
        if key in self._children:
            if subkey:
                return self._children[key][subkey]
            return self._children[key]
        if subkey:
            raise KeyError("key not valid")
        return super().__getitem__(key)

    def __delitem__(self, key):
        if isinstance(key, tuple):
            prekey, key = key[:-1], key[-1]
        else:
            prekey = ()

        if prekey:
            del self[prekey][key]
        else:
            if key in self._children:
                for idx in self._children[key]._child_indices():
                    self._source.remove_node(idx)
                del self._children[key]
            else:
                super().__delitem__(key)

    def _child_indices(self):
        yield self.idx[0]
        for child in self._children.values():
            if isinstance(child, _TensorDictNode):
                yield from child._child_indices()

    def apply_(self, fn):
        indices = list(self._child_indices())
        for k in self._source.keys():
            self._source[k][indices] = fn(self._source[k][indices])

    def get_multiple_items(self, *keys):
        keys = [(key,) if isinstance(key, str) else key for key in keys]
        key = keys[0][-1]
        assert {k[-1] for k in keys} == {key}
        indices = [self[k[:-1]].idx[0] if k[:-1] else self.idx[0] for k in keys]
        return self._source[key][indices]


class _TensorDictTreeSource(TensorDict):
    # a simple source class that keeps track of available indices
    # and can add / remove nodes
    def __init__(
        self,
        source,
        batch_size=None,
        device=None,
        *,
        _n_nodes=1000,
        _meta_source=None,
        _run_checks=None,
        _is_shared=None,
        _is_memmap=None,
    ):
        super().__init__(
            source,
            batch_size,
            device=device,
            _meta_source=_meta_source,
            _run_checks=_run_checks,
            _is_shared=_is_shared,
            _is_memmap=_is_memmap,
        )
        self._leaf_keys = set(source)
        self._n_nodes = _n_nodes
        self._available_indices = list(range(self._n_nodes))
        heapq.heapify(self._available_indices)
        self._node_indices = set()

    def numel(self):
        return len(self._node_indices)

    @property
    def cursor(self):
        return self._available_indices[0]

    def add_node(self):
        idx = heapq.heappop(self._available_indices)
        self._node_indices.add(idx)
        return _TensorDictNode(self, idx)

    def remove_node(self, idx):
        heapq.heappush(self._available_indices, idx)
        self._node_indices.remove(idx)


def make_tree(tensordict, n_nodes=1_000):
    contents = {
        key: torch.zeros(n_nodes, *value.shape)
        for key, value in tensordict.items(leaves_only=True)
    }
    source = _TensorDictTreeSource(
        contents, torch.Size([n_nodes, *tensordict.batch_size])
    )
    root = source.add_node()
    for key, value in tensordict.items(include_nested=True, leaves_only=True):
        root[key] = value
    return root
