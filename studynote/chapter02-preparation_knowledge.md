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

# 创建值全0或全1的张量
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

## 【1.2】运算符 

1）张量运算：可以在同一形状的任意两个张量上执行按元素操作； 

【test0201_p23_tensor_operator_main.py】测试案例-运算符 

```python
import torch

# 测试案例-运算符
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # 幂运算
# tensor([ 3,  4,  6, 10])
# tensor([-1,  0,  2,  6])
# tensor([ 2,  4,  8, 16])
# tensor([0.5000, 1.0000, 2.0000, 4.0000])
# tensor([ 1,  4, 16, 64])

# 向量点积与矩阵乘法
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

# 矩阵连接
print("按轴0连接 = ", torch.cat((X, Y), dim=0))
print("按轴1连接 = ", torch.cat((X, Y), dim=1))
# 按轴0连接 =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])
# 按轴1连接 =  tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

# 通过逻辑运算符构建二元张量
print("X == Y => ", X == Y)
# X == Y =>  tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])

# 对张量中所有元素求和
print("X.sum() = ", X.sum())
# X.sum() =  tensor(66.)
```

【补充】轴0表示行，轴1表示列；

<br>

---

## 【1.3】广播机制 

1）广播机制：形状不同的2个张量之间执行按元素运算；有两种方式，如下：

- 通过适当复制元素来扩展一个或两个数组，转换后使得两个张量具有相同形状；
- 对生成的数组执行按元素操作；

【test0201_p24_tensor_propagation_main.py】测试案例-广播机制

```python
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
```

【说明】

a维度为3X1，b维度为1X2，下面操作把两个矩阵广播为维度为3X2的矩阵，即矩阵a复制列，矩阵b复制行，然后按元素相加；

<br>

---

## 【1.4】索引和切片

1）索引：张量的第1个元素的索引是0，最后一个元素的索引是-1 ；可以指定范围包含第1个元素和最后一个之前的元素；

【test0201_p24_tensor_index_and_slice_main.py】

```python
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
```

<br>

为多个元素赋予相同的值。我们只需要索引所有元素，然后给他们赋值。如 [0:2, :] 访问第1行和第2行（<font color=red>索引0表示第1行，索引1表示第2行，所以[0:2]不包含索引2的数据 </font>），: 代表轴1（列）的所有元素。

```python
# 为多个元素赋予相同的值。我们只需要索引所有元素，然后给他们赋值。
# 如 [0:2, :] 访问第1行和第2行，: 代表轴1（列）的所有元素。
X[0:2, :] = 100
print("X = ", X)
# X =  tensor([[100., 100., 100., 100.],
#         [100., 100., 100., 100.],
#         [  8.,   9.,  10.,  11.]])
```

<br>

---

## 【1.5】节省内存

1）张量运算后，内存地址保持不变，不会开辟新内存； 

【test0201_p25_tensor_spare_memory_main.py】测试案例-节省内存 

```python
import torch

# 测试案例-节省内存
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

# 张量运算后，内存地址保持不变，不会开辟新内存
Z = torch.zeros_like(Y)
print("id(Z) = ", id(Z))
Z[:] = X + Y
print("id(Z) = ", id(Z))
# id(Z) =  1928199112736
# id(Z) =  1928199112736

# 也可以使用 X[:] = X + Y 或者 X += Y 减少内存开销
before = id(X)
X += Y
print("id(X) == before => ", id(X) == before)
# id(X) == before =>  True

# 使用 X = X + Y 会开辟新内存
before = id(X)
X = X + Y
print("id(X) == before => ", id(X) == before)
# id(X) == before =>  False
```

【小结】节省内存的写法：使用 X[:] = X + Y 或者 X += Y 或者 Z[:] = X + Y 减少内存开销；

<br>

---

## 【1.6】转换为其他python对象 

1）把深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之同样容易； torch张量和numpy数组将共享他们的底层内存，就地操作更改一个张量会同时更改另一个张量；  

【test0201_p25_tensor_transfer_other_obj_main.py】转换为其他python对象 

```python
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
```

<br>

---

## 【数据操作总结】

深度学习存储和操作数据的主要接口是张量（n维数组）。

它提供各种功能，包括基本数据运算， 广播，索引，切片，内存节省和转换为其他python对象。

<br>

---

# 【2】数据预处理

## 【2.1】读取数据集

【test0202_p26_read_dataset_main.py】测试案例-使用pandas包读取数据集

```python
import os
import pandas as pd

# 测试案例-读取数据集
data_file = os.path.join("dataset", "house_tiny.csv")
data = pd.read_csv(data_file)
print("data = ", data)
# data =     NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
```

<br>

---

## 【2.2】处理缺失值 

1）NaN代表缺失值，处理缺失值的方法有两种：

- 插值法：用一个代替值代替缺失值；
- 删除法：直接忽略缺失值； 

【例】构建输入与输出。

通过位置索引iloc，把data分为inputs和outputs，其中前者是data的前两列，后者为data的最后一列。

对于inputs中的缺失值，我们用同一列的均值来替换NaN。

【test0202_p26_proc_missing_value_main.py】测试用例-处理缺失值

```python
import os

import pandas as pd

data_file = os.path.join("dataset", "house_tiny.csv")
data = pd.read_csv(data_file)
print("data = ", data)
# data =     NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000

# 测试案例-处理缺失值
print("\n\n=== 测试案例-处理缺失值")
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2:3]
print("inputs = ", inputs)
print("outputs = ", outputs)
# inputs =     NumRooms Alley 
# 0       NaN  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       NaN   NaN
# outputs =      Price
# 0  127500
# 1  106000
# 2  178100
# 3  140000

# 处理缺失值：用均值填充第1列的NaN
inputs.iloc[:, 0] = inputs.fillna(inputs.iloc[:, 0].mean())
print("inputs = ", inputs)
# inputs =     NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN

# 处理缺失值：使用pandas把Alley转为两列：Alley_Pave, Alley_nan
inputs = pd.get_dummies(inputs, dummy_na=True)  # dummies=仿制品
print(type(inputs))
# <class 'pandas.core.frame.DataFrame'>

cols_to_convert = ['Alley_Pave', 'Alley_nan']
inputs[cols_to_convert] = inputs[cols_to_convert].astype(int)
print("inputs = ", inputs)
# inputs =     NumRooms  Alley_Pave  Alley_nan
# 0       3.0           1          0
# 1       2.0           0          1
# 2       4.0           0          1
# 3       3.0           0          1

```

<br>

---

## 【2.3】转换为张量格式

【test0202_p26_proc_missing_value_main.py】测试案例-转换为张量格式

```python
# 测试案例-转换为张量格式
X = torch.tensor(inputs.values)
print("X = ", X)
y = torch.tensor(outputs.values)
print("y = ", y)
# X =  tensor([[3., 1., 0.],
#         [2., 0., 1.],
#         [4., 0., 1.],
#         [3., 0., 1.]], dtype=torch.float64)
# y =  tensor([[127500],
#         [106000],
#         [178100],
#         [140000]])
```

<br>

---

# 【3】线性代数

## 【3.1】向量

【test0202_p28_vector_main.py】测试案例-向量 

```python
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

```

【维度的含义】

向量或轴的维度：表示向量或轴的长度，即向量或轴的元素数量； 

张量的维度：张量的轴数；如二维矩阵是二维张量，轴数为2；

<br>

----

## 【3.2】矩阵

【test0202_p29_matrix_main.py】测试案例-创建矩阵+对称矩阵+矩阵转置

```python
import torch

# 创建一个矩阵
A = torch.arange(20).reshape(5, 4)
print("A = ", A)
# A =  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])

# 矩阵转置 =》 A.T
print("A.T = ", A.T)
# A.T =  tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])

# 定义对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("B = ", B)
# 比较对称矩阵与它的转置
print(B == B.T)
# B =  tensor([[1, 2, 3],
#         [2, 0, 4],
#         [3, 4, 5]])
# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])
```

<br>

---

## 【3.3】张量（特别重要：张量定义）

1）向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有多个轴的数据结构；

2）<font color=red>张量定义：具有多个轴的数据结构，即描述具有多个轴的n维数组的通用方法 </font>； 

特别的， 向量是一阶张量， 矩阵是二阶张量；

【test0202_p29_tensor_main.py】测试案例-把一个向量转换为三阶张量（三维张量）

```python
import torch

# 把一个向量转换为三阶张量（三维张量）
X = torch.arange(24)
X = X.reshape(2, 3, 4)
print("X = ", X)
# X =  tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
```

<br>

---

## 【3.4】张量的性质

1）性质：给定具有相同形状的任意两个张量， 任何按照元素二元运算的结果都将是相同形状的张量；

2）两个矩阵按元素相乘称为哈达玛积；

```python
import torch

# 性质：给定具有相同形状的任意两个张量， 任何按照元素二元运算的结果都将是相同形状的张量；
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)

# 张量克隆：通过分配新内存，把A的一个副本分配给B
B = A.clone()
print("A = ", A)
print("A+B = ", A + B)
# A =  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.],
#         [16., 17., 18., 19.]])
# A+B =  tensor([[ 0.,  2.,  4.,  6.],
#         [ 8., 10., 12., 14.],
#         [16., 18., 20., 22.],
#         [24., 26., 28., 30.],
#         [32., 34., 36., 38.]])

# 两个矩阵按元素相乘称为哈达玛积
print("A * B = ", A * B)
# A * B =  tensor([[  0.,   1.,   4.,   9.],
#         [ 16.,  25.,  36.,  49.],
#         [ 64.,  81., 100., 121.],
#         [144., 169., 196., 225.],
#         [256., 289., 324., 361.]])
```

### 【3.4.1】张量加上或乘以一个标量

张量加上或乘以一个标量不会改变张量的形状，其中张量的每个元素都与标量相加或相乘； 

```python
# 张量加上或乘以一个标量不会改变张量的形状，其中张量的每个元素都与标量相加或相乘
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print("X = ", X)
# X =  tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

print("a + X = ", a + X)
print("(a * X).shape = ", (a * X).shape)
# a + X =  tensor([[[ 2,  3,  4,  5],
#          [ 6,  7,  8,  9],
#          [10, 11, 12, 13]],
#
#         [[14, 15, 16, 17],
#          [18, 19, 20, 21],
#          [22, 23, 24, 25]]])
# (a * X).shape =  torch.Size([2, 3, 4])
```

<br>

---

## 【2.5】降维

