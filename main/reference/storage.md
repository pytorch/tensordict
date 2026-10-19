# Key-value stores and UCXX

This page lists the tensordicts that keep their entries on a key-value
server, and the classes and functions that stream tensordicts between
processes over UCXX. For how to use them, see [Large datasets and storage backends](../storage.html#storage) and
[Distributed tensordicts](../distributed.html#distributed).

## Key-value stores

A [`TensorDictStore`](generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) keeps its entries on a server
that speaks the Redis protocol, such as Redis or Dragonfly. The data is on
the server, and each read gets it from the server. A
[`LazyStackedTensorDictStore`](generated/tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) is a lazy stack of
tensordicts on such a server, so that its entries can have different
shapes. You need the `redis` package and a running server.

Warning

`LazyStackedTensorDictStore` is experimental: its API can change in a
minor release without a deprecation cycle.

Both classes have the methods of [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](td.html#api-tensordict-methods). Their pages list the
methods that they add, such as `from_store`, `close`, and
`clear_redis`. `STORE_BACKENDS` is the type of their `backend`
argument.

| [`TensorDictStore`](generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore)(*[, backend, host, port, ...]) | A tensordict whose entries live on a Redis-compatible server. |
| --- | --- |
| [`LazyStackedTensorDictStore`](generated/tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore)(*[, backend, ...]) | A lazy stack of tensordicts whose entries live on a Redis-compatible server. |

| [`STORE_BACKENDS`](generated/tensordict.store.STORE_BACKENDS.html#tensordict.store.STORE_BACKENDS) | alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] |
| --- | --- |

## UCXX transport

Warning

`TensorDictPipe`, `TensorDictServer`, `send_tensordict`, and
`recv_tensordict` are experimental: their API can change in a minor
release without a deprecation cycle.

UCXX is the Python interface of UCX, a communication library that
chooses the transport between two processes, such as shared memory,
InfiniBand, or TCP. These classes and functions need the optional `ucxx`
package.

- A [`TensorDictPipe`](generated/tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) is a channel between two
processes. After the first tensordict, it sends only the data of the
next tensordicts with the same structure. The receiver writes each
tensordict into the same buffer, so clone a received tensordict if you
need it after the next receive.
- A [`TensorDictServer`](generated/tensordict._ucxx.TensorDictServer.html#tensordict._ucxx.TensorDictServer) accepts connections and
returns one pipe for each client.
- [`send_tensordict()`](generated/tensordict._ucxx.send_tensordict.html#tensordict._ucxx.send_tensordict) and
[`recv_tensordict()`](generated/tensordict._ucxx.recv_tensordict.html#tensordict._ucxx.recv_tensordict) send or receive one
tensordict over a UCXX endpoint, with its metadata each time.

[`TensorDictBase.asend()`](generated/tensordict.TensorDictBase.asend.html#tensordict.TensorDictBase.asend) and
[`TensorDictBase.arecv()`](generated/tensordict.TensorDictBase.arecv.html#tensordict.TensorDictBase.arecv) send
and receive through a pipe.

Import these names from `tensordict._ucxx`. It is their only import
path for now, although the module name starts with an underscore.

| [`TensorDictPipe`](generated/tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)(endpoint, *[, consolidated, ...]) | A channel between two processes that streams tensordicts over UCXX. |
| --- | --- |
| [`TensorDictServer`](generated/tensordict._ucxx.TensorDictServer.html#tensordict._ucxx.TensorDictServer)(port, *[, consolidated]) | A listener that accepts several clients, and gives a [`TensorDictPipe`](generated/tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) for each. |

| [`send_tensordict`](generated/tensordict._ucxx.send_tensordict.html#tensordict._ucxx.send_tensordict)(endpoint, td, *[, consolidated]) | Sends a tensordict over a UCXX endpoint, with its metadata, in one call. |
| --- | --- |
| [`recv_tensordict`](generated/tensordict._ucxx.recv_tensordict.html#tensordict._ucxx.recv_tensordict)(endpoint[, td, device, ...]) | Receives a tensordict that [`send_tensordict()`](generated/tensordict._ucxx.send_tensordict.html#tensordict._ucxx.send_tensordict) sends over a UCXX endpoint. |

## See also

- [Large datasets and storage backends](../storage.html#storage): start a server, write a tensordict to it, and read it
back.
- [Containers on storage backends](../compatibility.html#compatibility): which containers work on a store.
- [Distributed tensordicts](../distributed.html#distributed): send tensordicts between processes.
- [Distributed and remote](td.html#api-methods-distributed): the send and receive methods of every
tensordict.