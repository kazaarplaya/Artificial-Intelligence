# Student Name: Hans Pujalte
# Student FAN: PUJA0009
# File: [question5.py]
# Date: [5-08-2026]
# Description: Convert a NumPy array to a PyTorch tensor and perform a basic operation.

import numpy as np
import torch

numpy_arr = np.array([1,2,3,4,5])

tensor_from_numpy = torch.from_numpy(numpy_arr)
print("Tensor from NumPy array:", tensor_from_numpy)

tensor_addition = tensor_from_numpy + 10
print("Tensor after addition:", tensor_addition)
