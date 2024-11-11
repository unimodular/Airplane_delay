# import os
# import pandas as pd
# import joblib
# from scipy.sparse import hstack, csr_matrix

# # 加载模型、scaler 和 encoder
# model = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\sgd_classifier_model.joblib')
# scaler = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\scaler_model.joblib')
# encoder = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\encoder_model.joblib')

# # 设置文件目录路径
# data_dir = 'C:\\Users\\Patron\\Desktop\\module_3_model\\input_1'
# # 加载 scaler 对象

# # 检查 scaler 的特征名称
# if hasattr(scaler, "feature_names_in_"):
#     feature_names = scaler.feature_names_in_
#     print("Scaler的特征名称顺序:", feature_names)


# if hasattr(encoder, "feature_names_in_"):
#     feature_names = encoder.feature_names_in_
#     print("encoder的特征名称顺序:", feature_names)



# # 找到第四个文件
# files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
# files.sort()  # 确保文件顺序一致
# fourth_file_path = os.path.join(data_dir, files[0])  # 获取第四个文件的路径

# # 读取第四个文件并提取前100个样本
# df = pd.read_csv(fourth_file_path, na_values=["", " ", "NA", "N/A"], low_memory=False)

# # 如果文件为空或没有足够的行，提示错误
# if df.empty or len(df) < 100:
#     print("第四个文件为空或没有足够的数据行")
# else:
#     samples = df.iloc[:100]  # 获取前100个样本
    
#     # 提取所需的特征列
#     feature_columns = [
#         'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
#         'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime',
#         'ELEVATION', 'HourlyAltimeterSetting', 'HourlyDewPointTemperature', 
#         'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPressureChange', 
#         'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 'HourlyVisibility', 
#         'HourlyWindSpeed'
#     ]

#     # 只保留特征列
#     sample_features = samples[feature_columns].copy()

#     # 处理时间列的格式（假设时间是以 HHMM 形式存储）
#     def convert_to_hours(time):
#         hours = time // 100
#         minutes = time % 100
#         return hours + minutes / 60.0

#     sample_features['CRSDepTime'] = sample_features['CRSDepTime'].apply(convert_to_hours)
#     sample_features['CRSArrTime'] = sample_features['CRSArrTime'].apply(convert_to_hours)

#     # 分离数值和类别特征
#     numeric_features = sample_features[['ELEVATION', 'HourlyAltimeterSetting', 
#                                         'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 
#                                         'HourlyPressureChange', 'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 
#                                         'HourlyVisibility', 'HourlyWindSpeed', 'CRSDepTime', 'CRSArrTime' ]]
#     categorical_features = sample_features[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
#                                             'Origin', 'Dest']]

#     # 数值特征标准化
#     X_numeric = scaler.transform(numeric_features)

#     # 类别特征编码
#     X_categorical = encoder.transform(categorical_features)

#     # 合并数值和类别特征
#     X_sparse = hstack([csr_matrix(X_numeric), X_categorical])

#     # 使用加载的模型对样本进行预测
#     predictions = model.predict(X_sparse)

#     # 输出预测结果
#     print("第四个文件前100个样本的取消预测：", predictions)




import os
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

# 加载模型、scaler 和 encoder
model = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\sgd_classifier_model.joblib')
scaler = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\scaler_model.joblib')
encoder = joblib.load('C:\\Users\\Patron\\Desktop\\module_3_model\\encoder_model.joblib')

# 设置文件目录路径
data_dir = 'C:\\Users\\Patron\\Desktop\\module_3_model\\input_prediction'

# 检查 scaler 和 encoder 的特征名称顺序（可选）
if hasattr(scaler, "feature_names_in_"):
    print("Scaler的特征名称顺序:", scaler.feature_names_in_)

if hasattr(encoder, "feature_names_in_"):
    print("Encoder的特征名称顺序:", encoder.feature_names_in_)

# 找到第四个文件
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
files.sort()  # 确保文件顺序一致
fourth_file_path = os.path.join(data_dir, files[0])  # 获取第四个文件的路径

# 读取第四个文件并提取前600000个样本
df = pd.read_csv(fourth_file_path, na_values=["", " ", "NA", "N/A"], low_memory=False)

# 如果文件为空或没有足够的行，提示错误
if df.empty or len(df) < 100:
    print("第四个文件为空或没有足够的数据行")
else:
    samples = df.iloc[:600000]  # 获取前600000个样本
    
    # 提取所需的特征列
    feature_columns = [
        'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
        'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime',
         'HourlyDewPointTemperature', 
        'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyPressureChange', 
        'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 'HourlyVisibility', 
        'HourlyWindSpeed'
    ]

    # 只保留特征列
    sample_features = samples[feature_columns].copy()

    # 处理时间列的格式（假设时间是以 HHMM 形式存储）
    def convert_to_hours(time):
        hours = time // 100
        minutes = time % 100
        return hours + minutes / 60.0

    sample_features['CRSDepTime'] = sample_features['CRSDepTime'].apply(convert_to_hours)
    sample_features['CRSArrTime'] = sample_features['CRSArrTime'].apply(convert_to_hours)

    # 分离数值和类别特征
    numeric_features = sample_features[[ 
                                        'HourlyDewPointTemperature', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 
                                        'HourlyPressureChange', 'HourlyRelativeHumidity', 'HourlySeaLevelPressure', 
                                        'HourlyVisibility', 'HourlyWindSpeed', 'CRSDepTime', 'CRSArrTime']]
    categorical_features = sample_features[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'Marketing_Airline_Network', 
                                            'Origin', 'Dest']]

    # 确保所有数值特征均为数值类型并填充缺失值
    numeric_features = numeric_features.apply(pd.to_numeric, errors='coerce')

    for col in numeric_features.columns:
        if numeric_features[col].dtype in ['float64', 'int64']:  # 确保是数值类型
            # 使用均值填充，如果整列是 NaN，则改为 0 填充
            numeric_features[col] = numeric_features[col].fillna(numeric_features[col].mean() if not pd.isna(numeric_features[col].mean()) else 0)
        else:
            # 对非数值类型的列用 0 填充
            numeric_features[col] = numeric_features[col].fillna(0).astype(float)  # 转换为浮点数

    # 数值特征标准化
    X_numeric = scaler.transform(numeric_features)

    # 类别特征编码
    X_categorical = encoder.transform(categorical_features)

    # 合并数值和类别特征
    X_sparse = hstack([csr_matrix(X_numeric), X_categorical])

    # 使用加载的模型对样本进行预测
    predictions = model.predict(X_sparse)

    from sklearn.metrics import mean_squared_error

    # 假设 y_true 是真实标签值，predictions 是模型的预测值
    y_true = samples['Cancelled'].iloc[:600000].values
    mse = mean_squared_error(y_true, predictions)
    print("MSE:", mse)

    sum_predictions = sum(predictions)
    # 输出预测结果
    #print("第四个文件前100个样本的取消预测：", predictions)
    print("第四个文件前600000个样本的取消预测：", sum_predictions)
