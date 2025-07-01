import torch

# 测试案例-转换为其他python对象
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

A = X.numpy()
B = torch.tensor(A)
print("type(A) = ", type(A))
print("type(B) = ", type(B))
# type(A) =  <class 'numpy.ndarray'>
# type(B) =  <class 'torch.Tensor'>

# 要将大小为1的张量转换为python标量，可以调用item函数或python的内置函数
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
# tensor([3.5000]) 3.5 3.5 3
