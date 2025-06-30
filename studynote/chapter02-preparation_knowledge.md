[toc]

# 【README】

本文总结自《动手学深度学习PyTorch版》，书籍官方地址[https://zh.d2l.ai/chapter_installation/index.html](https://zh.d2l.ai/chapter_installation/index.html)；

---

# 【1】数据操作

## 【1.1】入门

张量定义：表示一个由数值组成的数组，这个数组由多个维度（轴）。

- 向量：具有一个轴的张量；
- 矩阵：具有两个轴的张量；

元素：张量中的每个值称为张量的元素； 

【test0201_p21_test.py】张量操作测试案例

```python
import torch

# 使用arange创建一个行向量
x = torch.arange(12)
print("x = ", x)
# x =  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 访问张量的形状
print("x.shape = ", x.shape)
# x.shape =  torch.Size([12])

# 计算张量中所有元素的总数，即形状的所有元素乘积
print("x.numel() = ", x.numel())
# x.numel() =  12

# 改变张量的形状，使用reshape
x2 = x.reshape(3, 4)
print("x2 = ", x2)
# x2 =  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 获取值全0或全1的张量
print("\n===torch.zeros(2, 3, 4) = ", torch.zeros(2, 3, 4))
print("\n===torch.ones(2, 3, 4) = ", torch.ones(2, 3, 4))
# torch.zeros(2, 3, 4) =  tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

# 通过某个特定的概率分组随机采样得到张量中的每个元素的值
# 从标准高斯分布中随机采样，均值为0，标准差为1
print("\n=== torch.randn(3,4) = ", torch.randn(3, 4))
# tensor([[ 0.2323,  0.6953, -0.9805,  1.1222],
#         [ 0.3032,  1.1561, -0.4181,  1.7556],
#         [ 0.0511,  0.2409,  0.3492, -0.0601]])

# 为每个张量的元素赋值
x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("\nx = ", x)
# x =  tensor([[2, 1, 4, 3],
#         [1, 2, 3, 4],
#         [4, 3, 2, 1]])
```

<br>

---

## 【2.1】运算符 

1）张量运算：可以在同一形状的任意两个张量上执行按元素操作； 



