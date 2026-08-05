# Student Name: Hans Pujalte
# Student FAN: PUJA0009
# File: [question3.py]
# Date: [5-08-2026]
# Description: Perform element-wise addition and multiplication on two tensors

import torch

tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([4, 5, 6])

tensor_add = tensor_a + tensor_b
print("Element-wise addition:", tensor_add)

tensor_mul = tensor_a * tensor_b
print("Element-wise multiplication:", tensor_mul)