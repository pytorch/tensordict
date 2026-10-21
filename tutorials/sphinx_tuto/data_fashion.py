"""
Using TensorDict for datasets
=============================

A ``torchvision`` dataset applies its transform to each image again at every
epoch. The ``DataLoader`` reads the samples one by one to build a batch. A
:class:`~tensordict.TensorDict` of memory-mapped tensors stores the converted
images once, and the ``DataLoader`` reads each batch from it with one index.
This tutorial does this for FashionMNIST, with the model and the training loop
of the `PyTorch Quickstart Tutorial
<https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__. It
is for readers who train a model from a map-style dataset.

What you will learn:

- How to convert a ``torchvision`` dataset into a tensordict of
  :class:`~tensordict.MemoryMappedTensor` entries, once.
- How to pass a tensordict to a ``DataLoader`` as a map-style dataset, and
  what ``collate_fn`` it takes.
- How to read the batches in a training loop.

Before you start, read :ref:`sphx_glr_tutorials_tensordict_memory.py` for
memory-mapped tensors. :ref:`sphx_glr_tutorials_tensorclass_fashion.py` does
the same with a tensorclass.
"""

import time

import torch
import torch.nn as nn

from tensordict import MemoryMappedTensor, TensorDict
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


###############################################################################
# The ``torchvision.datasets`` module contains many datasets that are ready to
# use. This tutorial uses FashionMNIST. Each image shows an item of clothing,
# and the task is to classify the type of clothing in the image (for example
# "Bag" or "Sneaker").

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

###############################################################################
# Create two tensordicts, one for the training data and one for the test
# data. Their entries are memory-mapped tensors. ``MemoryMappedTensor.empty``
# without a file name creates an anonymous memory map: on Linux, it is in
# shared memory under ``/dev/shm`` when it has room. To store the data in a
# file on disk, pass ``filename=``. The ``ToTensor`` transform runs once per
# image, when you fill the tensordicts, and not at each epoch.
#
# The tensordicts stay on the CPU, so that their entries stay memory-mapped
# tensors. The training loop moves each batch to the device. If you pass
# ``device="cuda"`` to the ``TensorDict`` constructor, the constructor copies
# the whole dataset to the GPU, and the entries are no longer memory-mapped.


def make_memmap_tensordict(dataset):
    image_shape = dataset[0][0].squeeze().shape
    return TensorDict(
        {
            "images": MemoryMappedTensor.empty(
                (len(dataset), *image_shape), dtype=torch.float32
            ),
            "targets": MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
        },
        batch_size=[len(dataset)],
    )


training_data_td = make_memmap_tensordict(training_data)
test_data_td = make_memmap_tensordict(test_data)

###############################################################################
# Fill the memory-mapped tensors. A ``DataLoader`` reads the dataset in
# batches of 1000 images, and each batch is written into the tensordict with
# one index. This step reads and converts every image once.


def fill(data_td, dataset):
    i = 0
    for images, targets in DataLoader(dataset, batch_size=1000):  # noqa: TOR401
        n = images.shape[0]
        data_td[i : i + n] = TensorDict(
            {"images": images.squeeze(1), "targets": targets}, [n]
        )
        i += n


fill(training_data_td, training_data)
fill(test_data_td, test_data)
print(training_data_td)

###############################################################################
# DataLoaders
# -----------
#
# Create two pairs of ``DataLoader``: one pair from the ``torchvision``
# datasets, and one pair from the memory-mapped tensordicts.
#
# A ``TensorDict`` implements ``__len__``, ``__getitem__``, and
# ``__getitems__``, so you can use it as a map-style dataset. The
# ``DataLoader`` then reads each batch with one batched index, and the result
# is already a ``TensorDict``. Pass ``collate_fn=lambda x: x`` to keep that
# batch as it is; the default collate function also returns it unchanged. If
# your dataset is a list of tensordicts, the default collate function stacks
# them, densely if their shapes match and lazily otherwise.

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)  # noqa: TOR401
test_dataloader = DataLoader(test_data, batch_size=batch_size)  # noqa: TOR401

train_dataloader_td = DataLoader(  # noqa: TOR401
    training_data_td, batch_size=batch_size, collate_fn=lambda x: x
)
test_dataloader_td = DataLoader(  # noqa: TOR401
    test_data_td, batch_size=batch_size, collate_fn=lambda x: x
)

###############################################################################
# Each batch of the tensordict ``DataLoader`` is a ``TensorDict`` with
# ``batch_size=[64]``:

print(next(iter(train_dataloader_td)))

###############################################################################
# Time one pass over the training data with each data loader, without a
# model. This measures the data loading alone:


def time_one_pass(dataloader):
    t0 = time.perf_counter()
    for _ in dataloader:
        pass
    return time.perf_counter() - t0


print(f"torchvision DataLoader: {time_one_pass(train_dataloader):.2f} s")
print(f"TensorDict DataLoader:  {time_one_pass(train_dataloader_td):.2f} s")

###############################################################################
# Model
# -----
#
# The model is the same as in the
# `Quickstart Tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__.
#


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


torch.manual_seed(0)
model = Net().to(device)
model_td = Net().to(device)
model_td.load_state_dict(model.state_dict())

###############################################################################
# Optimizing the parameters
# -------------------------
#
# Optimize the parameters of the model with stochastic gradient descent and
# the cross-entropy loss. Both models start from the same parameters.
#

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer_td = torch.optim.SGD(model_td.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 300 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


###############################################################################
# The training loop for the ``TensorDict`` data loader reads the images and
# the targets by key, and moves them to the device.


def train_td(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, data in enumerate(dataloader):
        X, y = data["images"].to(device), data["targets"].to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 300 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def test_td(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch["images"].to(device), batch["targets"].to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


###############################################################################
# Train each model for one epoch, and test it once. The first run uses the
# ``TensorDict`` data loaders, the second run uses the ``torchvision`` data
# loaders.

epochs = 1

t0 = time.perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train_td(train_dataloader_td, model_td, loss_fn, optimizer_td)
    test_td(test_dataloader_td, model_td, loss_fn)
print(f"TensorDict training done! time: {time.perf_counter() - t0: 4.4f} s")

t0 = time.perf_counter()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print(f"Training done! time: {time.perf_counter() - t0: 4.4f} s")

###############################################################################
# Conclusion
# ----------
#
# You can now convert a dataset into a tensordict of memory-mapped tensors
# once. Then you train a model from it with a ``DataLoader`` and
# ``collate_fn=lambda x: x``.
#
# The two training runs use the same model, the same initial parameters, and
# the same optimizer, so their times differ only by the data loading. The
# one-pass timings before the training show the data loading alone. The
# ``torchvision`` data loader applies ``ToTensor`` to each image, and collates
# 64 samples into each batch. The ``TensorDict`` data loader reads each batch
# from the memory-mapped tensors with one index. The times above come from
# the machine that built this page. They depend on the hardware and on the
# load of the machine, so measure them on your own machine.
#
# Further reading
# ---------------
#
# - :ref:`data-loading` explains the ways to load data with tensordict and
#   to share it between processes.
# - :ref:`sphx_glr_tutorials_tensorclass_fashion.py` does the same with a
#   tensorclass.
# - :ref:`sphx_glr_tutorials_tensorclass_imagenet.py` applies random
#   transforms to whole batches, for a larger image dataset.
# - :ref:`saving` explains how to save the memory-mapped tensordict to disk
#   and load it again.
