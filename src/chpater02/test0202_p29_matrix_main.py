import torch

# 创建一个矩阵
A = torch.arange(20).reshape(5, 4)
print("A = ", A)
# A =  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])

# 矩阵转置 =》 A.T
print("A.T = ", A.T)
# A.T =  tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])

# 定义对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("B = ", B)
# 比较对称矩阵与它的转置
print(B == B.T)
# B =  tensor([[1, 2, 3],
#         [2, 0, 4],
#         [3, 4, 5]])
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])

