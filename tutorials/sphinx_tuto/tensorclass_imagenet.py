# -*- coding: utf-8 -*-
"""
Batched data loading with tensorclasses
=======================================
"""

##############################################################################
# In this tutorial we demonstrate how tensorclasses and memory-mapped
# tensors can be used together to efficiently and transparently load data
# from disk inside a model training pipeline.
#
# The basic idea is that we pre-load the entire dataset into a
# memory-mapped tensors, applying any non-random transformations before
# saving to disk. This means that not only do we avoid performing repeated
# computation each time we iterate through the data, we also are able to
# efficiently load data from the memory-mapped tensor in batches, rather
# than sequentially from the raw image files.
#
# We’ll use the same subset of imagenet used in `this transfer learning
# tutorial <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>`__.
#
# .. note:: Download the data from
#   `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__
#   and extract it. We assume in this tutorial that the extracted data is
#   saved in the subdirectory ``data/``.
#

from pathlib import Path

import torch
from kornia import augmentation
from tensordict import MemmapTensor
from tensordict.prototype import tensorclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

##############################################################################
# First define train and val transforms that will be applied to train and
# val examples respectively. Note that there are random components in the
# train transform to prevent overfitting to training data over multiple
# epochs.

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

##############################################################################
# We use ``torchvision.datasets.ImageFolder`` to conveniently load and
# transform the data from disk.
#

data_dir = Path("data/hymenoptera_data/")


train_data = datasets.ImageFolder(root=data_dir / "train", transform=train_transform)
val_data = datasets.ImageFolder(root=data_dir / "val", transform=val_transform)

##############################################################################
# We’ll also create a dataset of the raw training data that simply resizes
# the image to a common size and converts to tensor. We’ll use this to
# load the data into memory-mapped tensors. The random transformations
# need to be different each time we iterate through the data, so they
# cannot be pre-computed.

train_data_raw = datasets.ImageFolder(
    root=data_dir / "train",
    transform=transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    ),
)

##############################################################################
# We will use augmentations from the
# `Kornia <https://github.com/kornia/kornia>`__ library to apply the
# random transformations to batched tensors.


batch_train_transform = augmentation.AugmentationSequential(
    augmentation.RandomResizedCrop((224, 224)),
    augmentation.RandomHorizontalFlip(),
    augmentation.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
)

##############################################################################
# Tensorclasses are a good choice when the structure of your data is known
# apriori. They are dataclasses that expose dedicated tensor methods over
# their contents much like a ``TensorDict``.
#
# As well as specifying the contents (in this case ``images`` and
# ``targets``) we can also encapsulate related logic as custom methods
# when defining the class. Here we add a classmethod that takes a dataset
# and creates a tensorclass containing the data by iterating over the
# dataset. We create memory-mapped tensors to hold the data so that they
# can be efficiently loaded in batches later.


@tensorclass
class ImageNetData:
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
        return self.batch_size[0] if self.batch_dims else 0


##############################################################################
# We create two tensorclasses, one for the training and on for the
# validation data. Note that while this step can be slightly expensive, it
# allows us to save repeated computation later during training.

train_data_tc = ImageNetData.from_dataset(train_data_raw, device=device)
val_data_tc = ImageNetData.from_dataset(val_data, device=device)

##############################################################################
# DataLoaders
# -----------
#
# We can create dataloaders both from the ``torchvision``-provided
# Datasets, as well as from our memory-mapped tensorclasses.
#
# Since tensorclasses implement ``__len__`` and ``__getitem__`` (and also
# ``__getitems__``) we can use them like a map-style Dataset and create a
# ``DataLoader`` directly from them.
#
# Note that because the tensorclass can handle batched indices, there is
# no need for additional collation, so we pass the identity function as
# the ``collate_fn``.

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

train_dataloader_tc = DataLoader(
    train_data_tc, batch_size=batch_size, collate_fn=lambda x: x
)
val_dataloader_tc = DataLoader(
    val_data_tc, batch_size=batch_size, collate_fn=lambda x: x
)

##############################################################################
# We can now compare how long it takes to iterate once over the data in
# each case. The regular dataloader loads images one by one from disk,
# applies the transform sequentially and then stacks the results.

import time

t0 = time.time()
for image, target in train_dataloader:
    image, target = image.to(device), target.to(device)
print(f"One iteration over dataloader done! Time: {time.time() - t0:4.4f}s")


##############################################################################
# Our tensorclass-based dataloader instead loads data from the
# memory-mapped tensor in batches. We then apply the batched random
# transformations to the batched images.

t0 = time.time()
for batch in train_dataloader_tc:
    image, target = (
        batch_train_transform(batch.images.contiguous()),
        batch.targets.contiguous(),
    )
print(f"One iteration over tensorclass dataloader done! Time: {time.time() - t0:4.4f}s")

##############################################################################
# In the case of the validation set, we see an even bigger performance
# improvement, because there are no random transformations, so we can save
# the fully transformed data in the memory-mapped tensor, eliminating the
# need for additional transformations as we load from disk.

t0 = time.time()
for image, target in val_dataloader:
    image, target = image.to(device), target.to(device)
print(f"One iteration over val data. Time: {time.time() - t0:4.4f}s")

t0 = time.time()
for batch in val_dataloader_tc:
    image, target = batch.images.contiguous(), batch.targets.contiguous()
print(f"One iteration over tensorclass val data. Time: {time.time() - t0:4.4f}s")
