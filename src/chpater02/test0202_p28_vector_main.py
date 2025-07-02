import torch

x = torch.arange(4)
print("x = ", x)
# x =  tensor([0, 1, 2, 3])

# 通过张量索引访问任一元素
print("x[3] = ", x[3])
# x[3] =  tensor(3)

# 调用len函数获取张量的长度
print("len(x) = ", len(x))
# len(x) =  4

# 通过.shape属性访问向量的长度
print("x.shape = ", x.shape)
# x.shape =  torch.Size([4])
