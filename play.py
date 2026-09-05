import torch

tensor = torch.randn(2, 3, 4)

print('before transpose:')
print(tensor)

print('after transpose:')
transposed_tensor = tensor.transpose(0, 1)
print(transposed_tensor)
