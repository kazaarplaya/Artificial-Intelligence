# Student Name: Hans Pujalte
# Student FAN: PUJA0009
# File: [question4.py]
# Date: [5-08-2026]
# Description: Create a 2x3 tensor and a 3x2 tensor, then perform matrix multiplication

import torch

tensor_c = torch.rand(2,3)
tensor_d = torch.rand(3,2)

tensor_matmul = torch.matmul(tensor_c, tensor_d)
print("Matrix multiplication results:", tensor_matmul)