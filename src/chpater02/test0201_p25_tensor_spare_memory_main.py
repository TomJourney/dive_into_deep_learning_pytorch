import torch

# 测试案例-节省内存
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("X = ", X)
Y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("Y = ", Y)
# X =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
# Y =  tensor([[2, 1, 4, 3],
#         [1, 2, 3, 4],
#         [4, 3, 2, 1]])

# 张量运算后，内存地址保持不变，不会开辟新内存
Z = torch.zeros_like(Y)
print("id(Z) = ", id(Z))
Z[:] = X + Y
print("id(Z) = ", id(Z))
# id(Z) =  1928199112736
# id(Z) =  1928199112736

# 也可以使用 X[:] = X + Y 或者 X += Y 减少内存开销
before = id(X)
X += Y
print("id(X) == before => ", id(X) == before)
# id(X) == before =>  True

# 使用 X = X + Y 会开辟新内存
before = id(X)
X = X + Y
print("id(X) == before => ", id(X) == before)
# id(X) == before =>  False