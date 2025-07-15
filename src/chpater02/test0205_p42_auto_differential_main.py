# 微积分
# 解释导数
import torch

# 自动微分：深度学习框架通过自动计算导数加快求导

# 【1】业务需求： 对 y=2x^Tx 关于列向量x求导
x = torch.arange(4.0)
print(x)
# tensor([0., 1., 2., 3.])

x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)
print(x.grad)
# None

# 计算y
y = 2 * torch.dot(x, x)
print(y)
# tensor(28., grad_fn=<MulBackward0>)

# 调用反向传播函数来自动计算y关于x的每个分量的梯度，并打印这些梯度
y.backward()
print(x.grad)
# tensor([ 0.,  4.,  8., 12.])

# 函数y=2x^Tx关于x的梯度是4x，验证这个梯度是否正确
print(x.grad == 4 * x)
# tensor([True, True, True, True])

# 【2】业务需求：计算第2个函数的梯度
# 默认情况下， PyTorch会累积梯度，我们需要清楚之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
# tensor([1., 1., 1., 1.])


# 2.5.2 非标量变量的反向传播
# 当y不是标量而是一个向量时，向量y关于向量x的导数的最自然解释是一个矩阵
# 对于高阶和高维的y和x，求导的结果可以是一个高阶张量
# 单独计算批量中每个样本的偏导数之和
x.grad.zero_()
y = x * x
y.sum().backward() # 等价于 y.backward(torch.ones(len(x)))
print(x.grad)
# tensor([0., 2., 4., 6.])