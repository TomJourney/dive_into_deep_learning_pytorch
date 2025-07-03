import torch

# 求和：一种降维方式
x = torch.arange(4, dtype=torch.float32)
print("x = ", x)
print("x.sum() = ", x.sum())
# x =  tensor([0., 1., 2., 3.])
# x.sum() =  tensor(6.)

# 计算矩阵A的元素和
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A = ", A)
# A =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])

print("A.shape = ", A.shape)
print("A.sum() = ", A.sum())
# A.shape =  torch.Size([5, 4])
# A.sum() =  tensor(190.)

# 指定张量沿着哪一个轴来通过求和降低维度

# 对轴0上的元素求和来降维，轴0表示行，即每列所有行元素求和，行维度消失，保留列维度
A_sum_axis0 = A.sum(axis=0)
print("A_sum_axis0 = ", A_sum_axis0)
print("A_sum_axis0.shape = ", A_sum_axis0.shape)
# A_sum_axis0 =  tensor([40., 45., 50., 55.])
# A_sum_axis0.shape =  torch.Size([4])

# 对轴1上的元素求和来降维，轴1表示列，即每行所有列元素求和，列维度消失，保留行维度
A_sum_axis1 = A.sum(axis=1)
print("A_sum_axis1 = ", A_sum_axis1)
print("A_sum_axis1.shape = ", A_sum_axis1.shape)
# A_sum_axis1 =  tensor([ 6., 22., 38., 54., 70.])
# A_sum_axis1.shape =  torch.Size([5])

# 沿着行和列对矩阵求和，等价于对矩阵的所有元素求和
A_sum = A.sum(axis=[0, 1])
print("A_sum = ", A_sum)
print("A.sum() = ", A.sum())
# A_sum =  tensor(190.)
# A.sum() =  tensor(190.)

# === 计算平均值
print("A.mean() = ", A.mean(), "  A.sum()/A.numel() = ", A.sum()/A.numel())
# A.mean() =  tensor(9.5000) A.sum()/A.numel() =  tensor(9.5000)

# 同样，计算平均值由额可以沿着指定的轴降低张量的维度
# 计算每行所有列元素的均值
print("A.mean(axis=0) = ", A.mean(axis=0))
print("A.sum(axis=0)/A.shape[0] = ", A.sum(axis=0)/A.shape[0])
# A.mean(axis=0) =  tensor([ 8.,  9., 10., 11.])
# A.sum(axis=0)/A.shape[0] =  tensor([ 8.,  9., 10., 11.])

# === 非降维求和
# 对每列的所有行元素求和，且保持维度不变
sum_A = A.sum(axis=1, keepdims = True)
print("sum_A = ", sum_A)
# sum_A =  tensor([[ 6.],
#         [22.],
#         [38.],
#         [54.],
#         [70.]])

# === 通过广播操作，计算每行元素的均值
print("A / sum_A = ", A / sum_A)
# A / sum_A =  tensor([[0.0000, 0.1667, 0.3333, 0.5000],
#         [0.1818, 0.2273, 0.2727, 0.3182],
#         [0.2105, 0.2368, 0.2632, 0.2895],
#         [0.2222, 0.2407, 0.2593, 0.2778],
#         [0.2286, 0.2429, 0.2571, 0.2714]])

# === 沿着轴0计算累计总和(每行列元素的累计总和)，注意是累计总和
print("A.cumsum(axis=0) = ", A.cumsum(axis=0))
# A.cumsum(axis=0) =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.],
#         [24., 28., 32., 36.],
#         [40., 45., 50., 55.]])
