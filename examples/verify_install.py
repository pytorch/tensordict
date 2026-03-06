#!/usr/bin/env python3
import torch
import tensordict

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"TensorDict: {tensordict.__version__}")

from tensordict._dtensor import _compute_transfer_plan
print("_compute_transfer_plan imported OK")

from tensordict import TensorDict
td = TensorDict(a=torch.randn(3))
assert hasattr(td, "dtensor_send")
assert hasattr(td, "dtensor_recv")
print("dtensor_send/dtensor_recv methods exist")

print("\nAll checks passed!")
