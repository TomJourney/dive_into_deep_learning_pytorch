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

# 指定张量沿着哪一个轴来通过求和降低维度
print("A = ", A)
# A =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])

# 对轴0上的元素求和来降维，轴0表示行，即每列对应行的元素求和，行维度消失，保留列维度
A_sum_axis0 = A.sum(axis=0)
print("A_sum_axis0 = ", A_sum_axis0)
print("A_sum_axis0.shape = ", A_sum_axis0.shape)
# A_sum_axis0 =  tensor([40., 45., 50., 55.])
# A_sum_axis0.shape =  torch.Size([4])

# 对轴1上的元素求和来降维，轴1表示列，即每行对应列的元素求和，列维度消失，保留行维度
A_sum_axis1 = A.sum(axis=1)
print("A_sum_axis1 = ", A_sum_axis1)
print("A_sum_axis1.shape = ", A_sum_axis1.shape)
# A_sum_axis1 =  tensor([ 6., 22., 38., 54., 70.])
# A_sum_axis1.shape =  torch.Size([5])

# 沿着行和列对矩阵求和，等价于对矩阵的所有元素求和
A.sum(axis=[0, 1])
