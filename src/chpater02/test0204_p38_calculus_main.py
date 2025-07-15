# 微积分
# 解释导数
import numpy as np
import torch
from matplotlib_inline import backend_inline

# 数值函数（当x=1时，导数为2）
def f(x):
    return 3 * x ** 2 - 4 * x

# 数值极限
def numerical_lim(f, x, h):
    return (f(x+h)-f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit = {numerical_lim(f, 1, h):.5f}')
    h *= 0.1
# h=0.10000, numerical limit = 2.30000
# h=0.01000, numerical limit = 2.03000
# h=0.00100, numerical limit = 2.00300
# h=0.00010, numerical limit = 2.00030
# h=0.00001, numerical limit = 2.00003


