import torch

# 测试案例-广播机制
# 按照数组中长度为1的轴进行广播
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print("a = ", a)
print("b = ", b)
# a =  tensor([[0],
#         [1],
#         [2]])
# b =  tensor([[0, 1]])

# 计算两个矩阵： a + b (a维度为3*1，b维度为1*2，下面操作把两个矩阵广播为维度为3*2的矩阵)
print("a + b = ", a + b)
# a + b =  tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
