Note

Go to the end
to download the full example code.

# Using tensorclasses for datasets

A [`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) is the class form of a tensordict. You
declare its fields with types, and you write methods that run on one sample or
on a batch. This tutorial stores FashionMNIST in a tensorclass with an
`images` field, a `targets` field, and a method that loads a dataset into
memory-mapped tensors. Then it trains the model of the [PyTorch Quickstart
Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
from it. It is for readers who want named, typed fields and methods for their
data.

What you will learn:

- How to define a [`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) with a typed field for each
part of a sample, and a method that loads a dataset into it.
- How to pass a tensorclass to a `DataLoader` as a map-style dataset.
- How to move a whole batch to the device with one call, and read its fields
as attributes.

[Using TensorDict for datasets](data_fashion.html#sphx-glr-tutorials-data-fashion-py) does the same with a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), and compares the loading time with the
`torchvision` data loader. Read it first.

```
import time

import torch
import torch.nn as nn

from tensordict import MemoryMappedTensor, TensorClass
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

```
Using device: cpu
```

This tutorial uses the FashionMNIST dataset of `torchvision`. Each image
shows an item of clothing, and the task is to classify the type of clothing
in the image (for example "Bag" or "Sneaker").

```
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
```

Like a `TensorDict`, a tensorclass has a batch size, and you can index it,
move it to a device, or memory-map it. Use a tensorclass when you know the
fields of your data in advance. To define one, subclass
[`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) and annotate its fields. The
`@tensorclass` decorator is the older form of the same thing.

A tensorclass can also have methods. The `from_dataset` class method below
takes a dataset and returns a tensorclass that holds all its data in
memory-mapped tensors. `MemoryMappedTensor.empty` without a file name
creates an anonymous memory map: on Linux, it is in shared memory under
`/dev/shm` when it has room. The method fills the tensors in batches of
1000 images, so the `ToTensor` transform runs once per image, and not at
each epoch.

```
class FashionMNISTData(TensorClass):
 images: torch.Tensor
 targets: torch.Tensor

 @classmethod
 def from_dataset(cls, dataset):
 data = cls(
 images=MemoryMappedTensor.empty(
 (len(dataset), *dataset[0][0].squeeze().shape), dtype=torch.float32
 ),
 targets=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
 batch_size=[len(dataset)],
 )
 i = 0
 for images, targets in DataLoader(dataset, batch_size=1000): # noqa: TOR401
 n = images.shape[0]
 data[i : i + n] = cls(
 images=images.squeeze(1), targets=targets, batch_size=[n]
 )
 i += n
 return data
```

Create two tensorclasses, one for the training data and one for the test
data. This step reads and converts every image once. The tensorclasses stay
on the CPU, so that their fields stay memory-mapped tensors. The training
loop moves each batch to the device.

```
training_data_tc = FashionMNISTData.from_dataset(training_data)
test_data_tc = FashionMNISTData.from_dataset(test_data)
print(training_data_tc)
```

```
FashionMNISTData(
 images=MemoryMappedTensor(shape=torch.Size([60000, 28, 28]), device=cpu, dtype=torch.float32, is_shared=False),
 targets=MemoryMappedTensor(shape=torch.Size([60000]), device=cpu, dtype=torch.int64, is_shared=False),
 batch_size=torch.Size([60000]),
 device=None,
 is_shared=False)
```

## DataLoaders

A tensorclass implements `__len__`, `__getitem__`, and `__getitems__`,
so you can use it as a map-style dataset. The `DataLoader` then reads each
batch with one batched index, and the result is already a
`FashionMNISTData` batch. Pass `collate_fn=lambda x: x`: the default
collate function cannot read a tensorclass.

```
batch_size = 64

train_dataloader_tc = DataLoader( # noqa: TOR401
 training_data_tc, batch_size=batch_size, collate_fn=lambda x: x
)
test_dataloader_tc = DataLoader( # noqa: TOR401
 test_data_tc, batch_size=batch_size, collate_fn=lambda x: x
)
```

Each batch is a `FashionMNISTData` with `batch_size=[64]`:

```
print(next(iter(train_dataloader_tc)))
```

```
FashionMNISTData(
 images=Tensor(shape=torch.Size([64, 28, 28]), device=cpu, dtype=torch.float32, is_shared=False),
 targets=Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.int64, is_shared=False),
 batch_size=torch.Size([64]),
 device=None,
 is_shared=False)
```

## Model

The model is the same as in the
[Quickstart Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).

```
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
model_tc = Net().to(device)
```

## Optimizing the parameters

Optimize the parameters of the model with stochastic gradient descent and
the cross-entropy loss.

The training loop moves the whole batch to the device with one `.to()`
call, then reads the images and the targets as attributes.

```
loss_fn = nn.CrossEntropyLoss()
optimizer_tc = torch.optim.SGD(model_tc.parameters(), lr=1e-3)

def train_tc(dataloader, model, loss_fn, optimizer):
 size = len(dataloader.dataset)
 model.train()

 for batch, data in enumerate(dataloader):
 data = data.to(device)
 X, y = data.images, data.targets

 pred = model(X)
 loss = loss_fn(pred, y)

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 if batch % 300 == 0:
 loss, current = loss.item(), batch * len(X)
 print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_tc(dataloader, model, loss_fn):
 size = len(dataloader.dataset)
 num_batches = len(dataloader)
 model.eval()
 test_loss, correct = 0, 0
 with torch.no_grad():
 for batch in dataloader:
 batch = batch.to(device)
 X, y = batch.images, batch.targets

 pred = model(X)

 test_loss += loss_fn(pred, y).item()
 correct += (pred.argmax(1) == y).type(torch.float).sum().item()

 test_loss /= num_batches
 correct /= size

 print(
 f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
 )
```

Train the model for one epoch, and test it once:

```
epochs = 1

t0 = time.perf_counter()
for t in range(epochs):
 print(f"Epoch {t + 1}\n-------------------------")
 train_tc(train_dataloader_tc, model_tc, loss_fn, optimizer_tc)
 test_tc(test_dataloader_tc, model_tc, loss_fn)
print(f"Tensorclass training done! time: {time.perf_counter() - t0: 4.4f} s")
```

```
Epoch 1
-------------------------
loss: 2.305311 [ 0/60000]
loss: 2.275867 [19200/60000]
loss: 2.230498 [38400/60000]
loss: 2.172683 [57600/60000]
Test Error:
 Accuracy: 49.5%, Avg loss: 2.162513

Tensorclass training done! time: 2.2759 s
```

## Conclusion

You can now define a [`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) with typed fields, and
a `from_dataset` method that loads a dataset into memory-mapped tensors
once. You can pass the tensorclass to a `DataLoader` with
`collate_fn=lambda x: x`. One call moves each batch to the device, and you
read its fields as attributes.

## Further reading

- [TensorClass](../tensorclass.html#tensorclass-guide) explains how to define and use tensorclasses.
- [Which container should I use?](../choosing_a_container.html#choosing-a-container) compares `TensorDict`, `TensorClass`, and
`TypedTensorDict`.
- [Load data and share it between processes](../data_loading.html#data-loading) explains the ways to load data with tensordict and to
share it between processes.
- [Batched data loading with tensorclasses](tensorclass_imagenet.html#sphx-glr-tutorials-tensorclass-imagenet-py) applies random
transforms to whole batches, for a larger image dataset.

**Total running time of the script:** (0 minutes 4.728 seconds)

[`Download Jupyter notebook: tensorclass_fashion.ipynb`](../_downloads/5a8f2d6b38b250cbfadc2a295255dde6/tensorclass_fashion.ipynb)

[`Download Python source code: tensorclass_fashion.py`](../_downloads/7d676e84a77f97a44ec65055635f58ee/tensorclass_fashion.py)

[`Download zipped: tensorclass_fashion.zip`](../_downloads/777fb49680f0121bef451f089b823d24/tensorclass_fashion.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)