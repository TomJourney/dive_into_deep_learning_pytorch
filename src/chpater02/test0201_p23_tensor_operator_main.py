import torch

# 测试案例-运算符
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # 幂运算
# tensor([ 3,  4,  6, 10])
# tensor([-1,  0,  2,  6])
# tensor([ 2,  4,  8, 16])
# tensor([0.5000, 1.0000, 2.0000, 4.0000])
# tensor([ 1,  4, 16, 64])

# 向量点积与矩阵乘法
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

# 矩阵连接
print("按轴0连接 = ", torch.cat((X, Y), dim=0))
print("按轴1连接 = ", torch.cat((X, Y), dim=1))
# 按轴0连接 =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])
# 按轴1连接 =  tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

# 通过逻辑运算符构建二元张量
print("X == Y => ", X == Y)
# X == Y =>  tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])

# 对张量中所有元素求和
print("X.sum() = ", X.sum())
# X.sum() =  tensor(66.)

