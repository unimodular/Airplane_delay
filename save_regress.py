# import os
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import joblib
# import warnings

# # 设置文件目录路径
# data_dir = 'input'

# # 定义目标和特征列
# target_column = 'ArrDelay'
# flight_columns = [
#     'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
#     'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime'
# ]
# weather_columns = [
#       'HourlyDewPointTemperature', 
#     'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPressureChange', 
#     'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 'HourlyVisibility', 
#     'HourlyWindSpeed'
# ]

# # 将 HHMM 格式的时间转换为小时数
# def convert_to_hours(time):
#     hours = time // 100  # 提取小时部分
#     minutes = time % 100  # 提取分钟部分
#     return hours + minutes / 60.0  # 转换为小时数

# # 初始化模型和标准化器
# model = LinearRegression(fit_intercept=True)
# scaler = StandardScaler(with_mean=True)

# # 预扫描：获取每个类别特征的完整值域
# all_years, all_months = set(), set()
# all_origins, all_dests = set(), set()
# all_airlines = set()
# all_days_of_month, all_days_of_week = set(), set()

# for file_name in os.listdir(data_dir):
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(data_dir, file_name)
#         df = pd.read_csv(file_path, usecols=flight_columns, na_values=["", " ", "NA", "N/A"], low_memory=False)
        
#         all_years.update(df['Year'].unique())
#         all_months.update(df['Month'].unique())
#         all_origins.update(df['Origin'].unique())
#         all_dests.update(df['Dest'].unique())
#         all_airlines.update(df['Marketing_Airline_Network'].unique())
#         all_days_of_month.update(df['DayofMonth'].unique())
#         all_days_of_week.update(df['DayOfWeek'].unique())

# # 排序类别以确保一致性
# all_years = sorted(all_years)
# all_months = sorted(all_months)
# all_origins = sorted(all_origins)
# all_dests = sorted(all_dests)
# all_airlines = sorted(all_airlines)
# all_days_of_month = sorted(all_days_of_month)
# all_days_of_week = sorted(all_days_of_week)

# # 设置 OneHotEncoder，使用完整类别值域
# encoder = OneHotEncoder(
#     sparse_output=False,
#     handle_unknown='ignore',
#     categories=[
#         all_years, all_months, all_days_of_month, all_days_of_week, 
#         all_airlines, all_origins, all_dests
#     ]
# )

# encoder_fitted = False
# numeric_columns = weather_columns + ['CRSDepTime', 'CRSArrTime']
# categorical_feature_names = []

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=UserWarning)

#     X_combined_all = []
#     y_all = []

#     for file_name in os.listdir(data_dir):
#         if file_name.endswith('.csv'):
#             file_path = os.path.join(data_dir, file_name)
            
#             # 随机选取十分之一的数据
#             df = pd.read_csv(file_path, na_values=["", " ", "NA", "N/A"], low_memory=False)
#             df = df.sample(frac=0.2, random_state=42)

#             # 删除取消航班的数据
#             df = df[df['Cancelled'] != 1]

#             # 填充 ArrDelay 中的 NaN 值为 0
#             df[target_column] = df[target_column].fillna(0)

#             # 分离目标变量和特征
#             y_batch = df[target_column].values.ravel()
#             X_flight = df[flight_columns].fillna('missing')  # 用 'missing' 填充类别特征
#             X_weather = df[weather_columns]

#             # 转换时间列
#             X_flight['CRSDepTime'] = X_flight['CRSDepTime'].apply(convert_to_hours)
#             X_flight['CRSArrTime'] = X_flight['CRSArrTime'].apply(convert_to_hours)

#             # 将每列转换为数值类型，无法转换的值会被设置为 NaN
#             X_weather = X_weather.apply(pd.to_numeric, errors='coerce')

#             # 使用均值填充天气数据中的 NaN 值
#             X_weather.fillna(X_weather.mean(), inplace=True)

#             # 数值特征标准化
#             if not encoder_fitted:
#                 X_numeric = scaler.fit_transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))
#             else:
#                 X_numeric = scaler.transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))

#             # 类别特征编码
#             if not encoder_fitted:
#                 X_categorical = encoder.fit_transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
#                                                                 'Marketing_Airline_Network', 'Origin', 'Dest']])
#                 categorical_feature_names = encoder.get_feature_names_out(['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
#                                                                            'Marketing_Airline_Network', 'Origin', 'Dest'])
#                 encoder_fitted = True
#             else:
#                 X_categorical = encoder.transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
#                                                             'Marketing_Airline_Network', 'Origin', 'Dest']])

#             # 合并数值和类别特征
#             X_combined = np.hstack([X_numeric, X_categorical])

#             # 将当前批次数据存储
#             X_combined_all.append(X_combined)
#             y_all.extend(y_batch)

#     # 合并所有数据并进行一次性训练
#     X_combined_all = np.vstack(X_combined_all)
#     y_all = np.array(y_all)
#     model.fit(X_combined_all, y_all)

# print("训练完成。")

# # 将数值特征和编码后的类别特征名称合并
# all_feature_names = numeric_columns + list(categorical_feature_names)

# # 提取模型的系数
# coefficients = model.coef_.flatten()

# # 将特征名称、对应的系数和类型输出到 txt 文件
# output_file = 'feature_coefficients_with_type_ols.txt'
# with open(output_file, 'w') as f:
#     f.write("Feature Name\tCoefficient\tType\n")
#     for feature, coef in zip(all_feature_names, coefficients):
#         feature_type = "numeric" if feature in numeric_columns else "object"
#         f.write(f"{feature}\t{coef}\t{feature_type}\n")

# print(f"特征名称、系数及类型已保存到 {output_file}")

# # 保存模型
# model_filename = 'ols_regressor_model.joblib'
# joblib.dump(model, model_filename)
# print(f"模型已保存到 {model_filename}")

# # 保存 scaler 和 encoder 对象
# scaler_filename = 'scaler_model_delay.joblib'
# encoder_filename = 'encoder_model_delay.joblib'
# joblib.dump(scaler, scaler_filename)
# joblib.dump(encoder, encoder_filename)
# print(f"scaler 和 encoder 对象已保存到 {scaler_filename} 和 {encoder_filename}")


import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import warnings
from tqdm import tqdm

# 设置文件目录路径
data_dir = 'input'

# 定义目标和特征列
target_column = 'ArrDelay'
flight_columns = [
    'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
    'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime'
]
weather_columns = [
    'HourlyDewPointTemperature', 
    'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPressureChange', 
    'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 'HourlyVisibility', 
    'HourlyWindSpeed'
]

# 将 HHMM 格式的时间转换为小时数
def convert_to_hours(time):
    hours = time // 100  # 提取小时部分
    minutes = time % 100  # 提取分钟部分
    return hours + minutes / 60.0  # 转换为小时数

# 初始化模型和标准化器
model = LinearRegression(fit_intercept=True)
scaler = StandardScaler(with_mean=True)

# 预扫描：获取每个类别特征的完整值域
all_years, all_months = set(), set()
all_origins, all_dests = set(), set()
all_airlines = set()
all_days_of_month, all_days_of_week = set(), set()

for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, usecols=flight_columns, na_values=["", " ", "NA", "N/A"], low_memory=False)
        
        all_years.update(df['Year'].unique())
        all_months.update(df['Month'].unique())
        all_origins.update(df['Origin'].unique())
        all_dests.update(df['Dest'].unique())
        all_airlines.update(df['Marketing_Airline_Network'].unique())
        all_days_of_month.update(df['DayofMonth'].unique())
        all_days_of_week.update(df['DayOfWeek'].unique())

# 排序类别以确保一致性
all_years = sorted(all_years)
all_months = sorted(all_months)
all_origins = sorted(all_origins)
all_dests = sorted(all_dests)
all_airlines = sorted(all_airlines)
all_days_of_month = sorted(all_days_of_month)
all_days_of_week = sorted(all_days_of_week)

# 设置 OneHotEncoder，使用完整类别值域
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore',
    categories=[
        all_years, all_months, all_days_of_month, all_days_of_week, 
        all_airlines, all_origins, all_dests
    ]
)

encoder_fitted = False
numeric_columns = weather_columns + ['CRSDepTime', 'CRSArrTime']
categorical_feature_names = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

    X_combined_all = []
    y_all = []

    # 添加进度条以显示文件处理进度
    for file_name in tqdm(os.listdir(data_dir), desc="Processing files"):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            
            # 随机选取十分之一的数据
            df = pd.read_csv(file_path, na_values=["", " ", "NA", "N/A"], low_memory=False)
            df = df.sample(frac=0.04, random_state=42)

            # 删除取消航班的数据
            df = df[df['Cancelled'] != 1]

            # 填充 ArrDelay 中的 NaN 值为 0
            df[target_column] = df[target_column].fillna(0)

            # 分离目标变量和特征
            y_batch = df[target_column].values.ravel()
            X_flight = df[flight_columns].fillna('missing')  # 用 'missing' 填充类别特征
            X_weather = df[weather_columns]

            # 转换时间列
            X_flight['CRSDepTime'] = X_flight['CRSDepTime'].apply(convert_to_hours)
            X_flight['CRSArrTime'] = X_flight['CRSArrTime'].apply(convert_to_hours)

            # 将每列转换为数值类型，无法转换的值会被设置为 NaN
            X_weather = X_weather.apply(pd.to_numeric, errors='coerce')

            # 使用均值填充天气数据中的 NaN 值
            X_weather.fillna(X_weather.mean(), inplace=True)

            # 数值特征标准化
            if not encoder_fitted:
                X_numeric = scaler.fit_transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))
            else:
                X_numeric = scaler.transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))

            # 类别特征编码
            if not encoder_fitted:
                X_categorical = encoder.fit_transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                                'Marketing_Airline_Network', 'Origin', 'Dest']])
                categorical_feature_names = encoder.get_feature_names_out(['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                                           'Marketing_Airline_Network', 'Origin', 'Dest'])
                encoder_fitted = True
            else:
                X_categorical = encoder.transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                            'Marketing_Airline_Network', 'Origin', 'Dest']])

            # 合并数值和类别特征
            X_combined = np.hstack([X_numeric, X_categorical])

            # 将当前批次数据存储
            X_combined_all.append(X_combined)
            y_all.extend(y_batch)

    # 合并所有数据并进行一次性训练
    X_combined_all = np.vstack(X_combined_all)
    y_all = np.array(y_all)
    model.fit(X_combined_all, y_all)

print("训练完成。")

# 将数值特征和编码后的类别特征名称合并
all_feature_names = numeric_columns + list(categorical_feature_names)

# 提取模型的系数
coefficients = model.coef_.flatten()

# 将特征名称、对应的系数和类型输出到 txt 文件
output_file = 'feature_coefficients_with_type_ols.txt'
with open(output_file, 'w') as f:
    f.write("Feature Name\tCoefficient\tType\n")
    for feature, coef in zip(all_feature_names, coefficients):
        feature_type = "numeric" if feature in numeric_columns else "object"
        f.write(f"{feature}\t{coef}\t{feature_type}\n")

print(f"特征名称、系数及类型已保存到 {output_file}")

# 保存模型
model_filename = 'ols_regressor_model.joblib'
joblib.dump(model, model_filename)
print(f"模型已保存到 {model_filename}")

# 保存 scaler 和 encoder 对象
scaler_filename = 'scaler_model_delay.joblib'
encoder_filename = 'encoder_model_delay.joblib'
joblib.dump(scaler, scaler_filename)
joblib.dump(encoder, encoder_filename)
print(f"scaler 和 encoder 对象已保存到 {scaler_filename} 和 {encoder_filename}")
