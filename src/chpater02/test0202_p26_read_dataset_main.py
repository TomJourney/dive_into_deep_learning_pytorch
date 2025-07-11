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

inputs.iloc[:, 0] = inputs.fillna(inputs.iloc[:, 0].mean())
print("inputs = ", inputs)
# inputs =     NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN