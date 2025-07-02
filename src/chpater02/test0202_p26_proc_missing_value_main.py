import os

import pandas as pd
import torch

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
