# import pandas as pd

# # 假设表格在 'input_1' 文件夹中，并命名为 'data.csv'
# data_dir = 'input_1/merged_2018_1.csv'
# df = pd.read_csv(data_dir)

# # 提取 "Cancelled" 列中数值为 1 的所有行
# cancelled_rows = df[df['Cancelled'] == 1]

# # 将提取出来的行重复 10 次
# repeated_cancelled_rows = pd.concat([cancelled_rows] * 10, ignore_index=True)

# # 将重复的行加到整个表格的末尾
# result_df = pd.concat([df, repeated_cancelled_rows], ignore_index=True)


# # 使用 sample 方法重新打乱行顺序，设置 `frac=1` 表示打乱所有行
# shuffled_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # 保存重新打乱顺序的表格
# shuffled_df.to_csv('shuffled_output.csv', index=False)


import os
import pandas as pd

# 设置文件夹路径
input_folder = 'input'
output_folder = 'input_oversample_1'

# 如果输出文件夹不存在，则创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历 input_folder 中的所有 CSV 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取每个 CSV 文件
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # 提取 "Cancelled" 列中数值为 1 的所有行
        cancelled_rows = df[df['Cancelled'] == 1]

        # 将提取出来的行重复 10 次
        repeated_cancelled_rows = pd.concat([cancelled_rows] * 10, ignore_index=True)

        # 将重复的行加到整个表格的末尾
        result_df = pd.concat([df, repeated_cancelled_rows], ignore_index=True)

        # 使用 sample 方法重新打乱行顺序
        shuffled_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 设置新的文件名，并保存到 output_folder
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_oversample.csv")
        shuffled_df.to_csv(output_file_path, index=False)

print("所有文件已处理并保存到 input_oversample 文件夹中。")
