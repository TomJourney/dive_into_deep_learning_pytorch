import torch

# 2.3.8 矩阵向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print(A.shape, x.shape, torch.mv(A, x))
# torch.Size([5, 4]) torch.Size([4]) tensor([ 14.,  38.,  62.,  86., 110.])

# 2.3.9 矩阵与矩阵乘法
B = torch.ones(4, 3, dtype=torch.float32)
print(torch.mm(A, B))
# tensor([[ 6.,  6.,  6.],
#         [22., 22., 22.],
#         [38., 38., 38.],
#         [54., 54., 54.],
#         [70., 70., 70.]])

# 2.3.10 范数
# 计算向量的L2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# tensor(5.)

# 计算向量的L1范数
print(torch.abs(u).sum())
# tensor(7.)

# 计算矩阵的费罗贝尼乌斯范数
print(torch.norm(torch.ones((4, 9))))
# tensor(6.)

