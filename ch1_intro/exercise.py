import torch

"""Ex1: 2D Tensor and add 1 dimension to dimension 0."""
a = torch.rand(3, 3)
print(a)
a = a.unsqueeze(0)
print(a)

"""Ex2: delete the added dimension."""
b = a.squeeze(0)
print(b)

"""Ex3: random Tensor with shape 5*3 in range [3, 7)"""
rand_tensor = (7 - 3) * torch.rand((5, 3)) + 3
print(rand_tensor)

"""Ex4: a tensor with normal distribution."""
normal_tensor = torch.normal(mean=0, std=1, size=(3, 4))
print(normal_tensor)

"""Ex5: get the indices of all not zero values in [1, 1, 1, 0, 1]"""
'''(tensor == target_value).nonzero(as_tuple=True)'''
x = torch.Tensor([1, 1, 1, 0, 1])
print((x != 0).nonzero(as_tuple=True))
print(x.nonzero(as_tuple=False))
print(x.nonzero(as_tuple=False).squeeze_(1))

"""Ex6: random (3, 1) tensor and stack 4 copies of it."""
x = torch.rand(3, 1)
print(x)
print(torch.stack((x, x, x, x), 1))
# Answer
print(x.expand(3, 4))
# print(x.unsqueeze(0).expand(4, 3, 4))

"""Ex7: batch-matrix-matrix-product of 2D matrixes 
a = torch.rand(3, 4, 5) and b = torch.rand(3, 5, 4)."""
a = torch.rand(3, 4, 5)
b = torch.rand(3, 5, 4)
print(torch.bmm(a, b))

"""Ex8: batch-m-m-production of a=torch.rand(3, 4, 5)
and b=torch,rand(5, 4)."""
a = torch.rand(3, 4, 5)
b = torch.rand(5, 4)
b1 = torch.stack((b, b, b), 0)
print(torch.bmm(a, b1))
# answer
print(torch.bmm(a, b.unsqueeze(0).expand(a.size(0), *b.size())))