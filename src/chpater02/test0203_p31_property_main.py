import torch

# 性质：给定具有相同形状的任意两个张量， 任何按照元素二元运算的结果都将是相同形状的张量；
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)

# 张量克隆：通过分配新内存，把A的一个副本分配给B
B = A.clone()
print("A = ", A)
print("A+B = ", A + B)
# A =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# A+B =  tensor([[ 0.,  2.,  4.,  6.],
#         [ 8., 10., 12., 14.],
#         [16., 18., 20., 22.],
#         [24., 26., 28., 30.],
#         [32., 34., 36., 38.]])

# 两个矩阵按元素相乘称为哈达玛积
print("A * B = ", A * B)
# A * B =  tensor([[  0.,   1.,   4.,   9.],
#         [ 16.,  25.,  36.,  49.],
#         [ 64.,  81., 100., 121.],
#         [144., 169., 196., 225.],
#         [256., 289., 324., 361.]])

# 张量加上或乘以一个标量不会改变张量的形状，其中张量的每个元素都与标量相加或相乘
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print("X = ", X)
# X =  tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

print("a + X = ", a + X)
print("(a * X).shape = ", (a * X).shape)
# a + X =  tensor([[[ 2,  3,  4,  5],
#          [ 6,  7,  8,  9],
#          [10, 11, 12, 13]],
#
#         [[14, 15, 16, 17],
#          [18, 19, 20, 21],
#          [22, 23, 24, 25]]])
# (a * X).shape =  torch.Size([2, 3, 4])

