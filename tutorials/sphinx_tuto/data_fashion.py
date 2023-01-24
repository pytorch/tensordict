# -*- coding: utf-8 -*-
"""
Using TensorDict for datasets
=============================
"""


##############################################################################
# In this tutorial we demonstrate how ``TensorDict`` can be used to
# efficiently and transparently load and manage data inside a training
# pipeline. The tutorial is based heavily on the `PyTorch Quickstart
# Tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__,
# but modified to demonstrate use of ``TensorDict``.


import torch
import torch.nn as nn
from tensordict import MemmapTensor, TensorDict
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


###############################################################################
# The ``torchvision.datasets`` module contains a number of convenient pre-prepared
# datasets. In this tutorial we'll use the relatively simple FashionMNIST dataset. Each
# image is an item of clothing, the objective is to classify the type of clothing in
# the image (e.g. "Bag", "Sneaker" etc.).

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
# We will create two tensordicts, one each for the training and test data. We create
# memory-mapped tensors to hold the data. This will allow us to efficiently load
# batches of transformed data from disk rather than repeatedly load and transform
# individual images.
#
# First we create the ``MemmapTensor`` containers.


training_data_td = TensorDict(
    {
        "images": MemmapTensor(
            len(training_data),
            *training_data[0][0].squeeze().shape,
            dtype=torch.float32,
        ),
        "targets": MemmapTensor(len(training_data), dtype=torch.int64),
    },
    batch_size=[len(training_data)],
    device=device,
)
test_data_td = TensorDict(
    {
        "images": MemmapTensor(
            len(test_data), *test_data[0][0].squeeze().shape, dtype=torch.float32
        ),
        "targets": MemmapTensor(len(test_data), dtype=torch.int64),
    },
    batch_size=[len(test_data)],
    device=device,
)

###############################################################################
# Then we can iterate over the data to populate the memory-mapped tensors. This takes a
# bit of time, but performing the transforms up-front will save repeated effort during
# training later.

for i, (img, label) in enumerate(training_data):
    training_data_td[i] = TensorDict({"images": img, "targets": label}, [])

for i, (img, label) in enumerate(test_data):
    test_data_td[i] = TensorDict({"images": img, "targets": label}, [])

###############################################################################
# DataLoaders
# ----------------
#
# We'll create DataLoaders from the ``torchvision``-provided Datasets, as well as from
# our memory-mapped TensorDicts.
#
# Since ``TensorDict`` implements ``__len__`` and ``__getitem__`` (and also
# ``__getitems__``) we can use it like a map-style Dataset and create a ``DataLoader``
# directly from it. Note that because ``TensorDict`` can already handle batched indices,
# there is no need for collation, so we pass the identity function as ``collate_fn``.

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

train_dataloader_td = DataLoader(
    training_data_td, batch_size=batch_size, collate_fn=lambda x: x
)
test_dataloader_td = DataLoader(
    test_data_td, batch_size=batch_size, collate_fn=lambda x: x
)

###############################################################################
# Model
# -------
#
# We use the same model from the
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


model = Net().to(device)
model_td = Net().to(device)
model, model_td

###############################################################################
# Optimizing the parameters
# ---------------------------------
#
# We'll optimise the parameters of the model using stochastic gradient descent and
# cross-entropy loss.
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


###############################################################################
# The training loop for our ``TensorDict``-based DataLoader is very similar, we just
# adjust how we unpack the data to the more explicit key-based retrieval offered by
# ``TensorDict``. The ``.contiguous()`` method loads the data stored in the memmap tensor.


def train_td(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, data in enumerate(dataloader):
        X, y = data["images"].contiguous(), data["targets"].contiguous()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
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
            X, y = batch["images"].contiguous(), batch["targets"].contiguous()

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


for d in train_dataloader_td:
    print(d)
    break

import time

t0 = time.time()
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train_td(train_dataloader_td, model_td, loss_fn, optimizer_td)
    test_td(test_dataloader_td, model_td, loss_fn)
print(f"TensorDict training done! time: {time.time() - t0: 4.4f} s")

t0 = time.time()
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print(f"Training done! time: {time.time() - t0: 4.4f} s")
