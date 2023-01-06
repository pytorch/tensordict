# -*- coding: utf-8 -*-
"""
Using tensorclasses for datasets
=============================
"""


##############################################################################
# In this tutorial we demonstrate how tensorclasses can be used to
# efficiently and transparently load and manage data inside a training
# pipeline. The tutorial is based heavily on the `PyTorch Quickstart
# Tutorial <https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>`__,
# but modified to demonstrate use of tensorclass. See the related tutorial using
# ``TensorDict``.


import torch
import torch.nn as nn
from tensordict import MemmapTensor
from tensordict.prototype import tensorclass
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
# Tensorclasses are dataclasses that expose dedicated tensor methods over
# its contents much like ``TensorDict``. They are a good choice when the
# structure of the data you want to store is fixed and predictable.
#
# As well as specifying the contents, we can also encapsulate related
# logic as custom methods when defining the class. In this case we’ll
# write a ``from_dataset`` classmethod that takes a dataset as input and
# creates a tensorclass containing the data from the dataset. We create
# memory-mapped tensors to hold the data. This will allow us to
# efficiently load batches of transformed data from disk rather than
# repeatedly load and transform individual images.


@tensorclass
class FashionMNISTData:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, device=None):
        data = cls(
            images=MemmapTensor(
                len(dataset), *dataset[0][0].squeeze().shape, dtype=torch.float32
            ),
            targets=MemmapTensor(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
            device=device,
        )
        for i, (image, target) in enumerate(dataset):
            data[i] = cls(images=image, targets=torch.tensor(target), batch_size=[])
        return data

    def __len__(self):
        # need to define explicitly for `len()` to work
        return self.batch_size[0] if self.batch_dims else 0


###############################################################################
# We will create two tensorclasses, one each for the training and test data. Note that
# we incur some overhead here as we are looping over the entire dataset, transforming
# and saving to disk.

training_data_tc = FashionMNISTData.from_dataset(training_data, device=device)
test_data_tc = FashionMNISTData.from_dataset(test_data, device=device)

###############################################################################
# DataLoaders
# ----------------
#
# We’ll create DataLoaders from the ``torchvision``-provided Datasets, as
# well as from our memory-mapped TensorDicts.
#
# Since ``TensorDict`` implements ``__len__`` and ``__getitem__`` (and
# also ``__getitems__``) we can use it like a map-style Dataset and create
# a ``DataLoader`` directly from it. Note that because ``TensorDict`` can
# already handle batched indices, there is no need for collation, so we
# pass the identity function as ``collate_fn``.

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

train_dataloader_tc = DataLoader(
    training_data_tc, batch_size=batch_size, collate_fn=lambda x: x
)
test_dataloader_tc = DataLoader(
    test_data_tc, batch_size=batch_size, collate_fn=lambda x: x
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
model_tc = Net().to(device)
model, model_tc

###############################################################################
# Optimizing the parameters
# ---------------------------------
#
# We'll optimise the parameters of the model using stochastic gradient descent and
# cross-entropy loss.
#

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer_tc = torch.optim.SGD(model_tc.parameters(), lr=1e-3)


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
# The training loop for our tensorclass-based DataLoader is very similar, we just
# adjust how we unpack the data to the more explicit attribute-based retrieval offered
# by the tensorclass. The ``.contiguous()`` method loads the data stored in the memmap
# tensor.


def train_tc(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, data in enumerate(dataloader):
        X, y = data.images.contiguous(), data.targets.contiguous()

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


def test_tc(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch.images.contiguous(), batch.targets.contiguous()

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


for d in train_dataloader_tc:
    print(d)
    break

import time

t0 = time.time()
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train_tc(train_dataloader_tc, model_tc, loss_fn, optimizer_tc)
    test_tc(test_dataloader_tc, model_tc, loss_fn)
print(f"Tensorclass training done! time: {time.time() - t0: 4.4f} s")

t0 = time.time()
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print(f"Training done! time: {time.time() - t0: 4.4f} s")
