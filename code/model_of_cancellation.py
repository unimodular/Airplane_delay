import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import warnings
import joblib

# 设置文件目录路径
data_dir = 'input_oversample_1'

# 定义目标和特征列
target_columns = ['Cancelled']
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
model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, max_iter=1, tol=None)
scaler = StandardScaler(with_mean=False)

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
        # all_flight_numbers.update(df['Flight_Number_Marketing_Airline'].unique())
        all_days_of_month.update(df['DayofMonth'].unique())
        all_days_of_week.update(df['DayOfWeek'].unique())

# 排序类别以确保一致性
all_years = sorted(all_years)
all_months = sorted(all_months)
all_origins = sorted(all_origins)
all_dests = sorted(all_dests)
all_airlines = sorted(all_airlines)
# all_flight_numbers = sorted(all_flight_numbers)
all_days_of_month = sorted(all_days_of_month)
all_days_of_week = sorted(all_days_of_week)

# 设置 OneHotEncoder，使用完整类别值域
encoder = OneHotEncoder(
    sparse_output=True,
    handle_unknown='ignore',
    categories=[
        all_years, all_months, all_days_of_month, all_days_of_week, 
        all_airlines, all_origins, all_dests
    ]
)

# 遍历目录中的文件，逐个读取和训练
encoder_fitted = False
numeric_columns = weather_columns + ['CRSDepTime', 'CRSArrTime']
categorical_feature_names = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

    for file_name in tqdm(os.listdir(data_dir), desc="Processing files"):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            
            # 使用 chunksize 逐批读取数据
            chunk_size = 200000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, na_values=["", " ", "NA", "N/A"], low_memory=False):
                
                # 删除包含目标列缺失值的行
                chunk.dropna(subset=target_columns, inplace=True)

                # 分离目标变量和特征
                y_batch = chunk[target_columns].values.ravel()
                X_flight = chunk[flight_columns].fillna('missing')  # 用 'missing' 填充类别特征
                X_weather = chunk[weather_columns]

                # 转换时间列
                X_flight['CRSDepTime'] = X_flight['CRSDepTime'].apply(convert_to_hours)
                X_flight['CRSArrTime'] = X_flight['CRSArrTime'].apply(convert_to_hours)

                # 将每列转换为数值类型，无法转换的值会被设置为 NaN
                X_weather = X_weather.apply(pd.to_numeric, errors='coerce')

                # 使用均值填充天气数据中的 NaN 值
                # X_weather = X_weather.fillna(0)
                for col in X_weather.columns:
                  if X_weather[col].dtype in ['float64', 'int64']:  # 确保是数值类型
                      # 使用均值填充，如果整列是 NaN，则改为 0 填充
                      X_weather[col] = X_weather[col].fillna(X_weather[col].mean() if not X_weather[col].mean() is np.nan else 0)
                  else:
                      # 对非数值类型的列用 0 填充
                      X_weather[col] = X_weather[col].fillna(0)
                # 确保没有 NaN 值
                if X_weather.isna().any().any() or X_flight.isna().any().any():
                    print("仍然存在 NaN 值，检查列：")
                    print("X_weather 缺失列：", X_weather.columns[X_weather.isna().any()])
                    print("X_flight 缺失列：", X_flight.columns[X_flight.isna().any()])
                    exit()

                # 数值特征标准化
                if not encoder_fitted:
                    X_numeric = scaler.fit_transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))
                else:
                    X_numeric = scaler.transform(pd.concat([X_weather, X_flight[['CRSDepTime', 'CRSArrTime']]], axis=1))

                # 类别特征编码
                if not encoder_fitted:
                    X_categorical = encoder.fit_transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                                    'Marketing_Airline_Network', 
                                                                    'Origin', 'Dest']])
                    categorical_feature_names = encoder.get_feature_names_out(['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                                               'Marketing_Airline_Network', 
                                                                               'Origin', 'Dest'])
                    encoder_fitted = True
                else:
                    X_categorical = encoder.transform(X_flight[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 
                                                                'Marketing_Airline_Network', 
                                                                'Origin', 'Dest']])

                # 合并数值和类别稀疏矩阵
                X_sparse = hstack([csr_matrix(X_numeric), X_categorical])



                #修改：保存经过 scaler 和 encoder 处理后的数据

                processed_data_file = 'processed_data.npz'

                from scipy.sparse import save_npz
                save_npz(processed_data_file, X_sparse)
                #print(f"处理后的数据已保存到 {processed_data_file}")



                # 最终检查 NaN 值
                if np.isnan(X_sparse.data).any():
                    print("NaN values found in sparse matrix data after merging.")
                    exit()

                # 增量训练
                model.partial_fit(X_sparse, y_batch, classes=[0, 1])

print("增量训练完成。")

# 将数值特征和编码后的类别特征名称合并
all_feature_names = numeric_columns + list(categorical_feature_names)

# 提取模型的系数
coefficients = model.coef_.flatten()

# 将特征名称、对应的系数和类型输出到 txt 文件
output_file = 'feature_coefficients_with_type.txt'
with open(output_file, 'w') as f:
    f.write("Feature Name\tCoefficient\tType\n")
    for feature, coef in zip(all_feature_names, coefficients):
        feature_type = "numeric" if feature in numeric_columns else "object"
        f.write(f"{feature}\t{coef}\t{feature_type}\n")

print(f"特征名称、系数及类型已保存到 {output_file}")

# 保存模型
model_filename = 'sgd_classifier_model.joblib'
joblib.dump(model, model_filename)
print(f"模型已保存到 {model_filename}")

# 保存 scaler 和 encoder 对象
scaler_filename = 'scaler_model.joblib'
encoder_filename = 'encoder_model.joblib'
joblib.dump(scaler, scaler_filename)
joblib.dump(encoder, encoder_filename)
print(f"scaler 和 encoder 对象已保存到 {scaler_filename} 和 {encoder_filename}")

# 加载 scaler 对象
scaler = joblib.load('scaler_model.joblib')
encoder = joblib.load('encoder_model.joblib')
# 检查 scaler 的特征名称
if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
    print("Scaler的特征名称顺序:", feature_names)


if hasattr(encoder, "feature_names_in_"):
    feature_names = encoder.feature_names_in_
    print("encoder的特征名称顺序:", feature_names)
