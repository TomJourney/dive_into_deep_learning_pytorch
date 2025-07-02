import torch

# 求和：一种降维方式
x = torch.arange(4, dtype=torch.float32)
print("x = ", x)
print("x.sum() = ", x.sum())
# x =  tensor([0., 1., 2., 3.])
# x.sum() =  tensor(6.)

# 计算矩阵A的元素和
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A.shape = ", A.shape)
print("A.sum() = ", A.sum())
# A.shape =  torch.Size([5, 4])
# A.sum() =  tensor(190.)

#