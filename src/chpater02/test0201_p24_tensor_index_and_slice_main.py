import torch

# 测试案例-索引与切片
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("X = ", X)
# X =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# 使用索引选择张量元素
print("X[-1] = ", X[-1]) # 访问最后一个元素
print("X[1:3] = ", X[1:3]) # 访问第2个，第3个元素
# X[-1] =  tensor([ 8.,  9., 10., 11.])
# X[1:3] =  tensor([[ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# 通过指定索引把元素写入矩阵
X[1, 2] = 99
print("X = ", X)
# X =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  99.,  7.],
#         [ 8.,  9., 10., 11.]])

# 为多个元素赋予相同的值。我们只需要索引所有元素，然后给他们赋值。
# 如 [0:2, :] 访问第1行和第2行，: 代表轴1（列）的所有元素。
X[0:2, :] = 100
print("X = ", X)
# X =  tensor([[100., 100., 100., 100.],
#         [100., 100., 100., 100.],
#         [  8.,   9.,  10.,  11.]])

