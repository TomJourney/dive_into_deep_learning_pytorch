import torch

# 点积：2个向量相同位置的元素乘积的和
y = torch.ones(4, dtype=torch.float32)
x = torch.arange(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
# tensor([0., 1., 2., 3.]) tensor([1., 1., 1., 1.]) tensor(6.)

# 可以通过按元素乘法，然后求和来表示两个向量的点积
print(torch.sum(x * y))
# tensor(6.)

