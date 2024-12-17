import pandas as pd

def process_csv(file_path):
    try:
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 删除连续重复的行
        data = data.loc[(data.shift() != data).any(axis=1)]
        
        # 获取最后100行，如果少于100行则获取全部
        last_rows = data.tail(100)
        
        # 获取最后一列的名称
        last_column_name = last_rows.columns[20]
        fps_column = last_rows.columns[-1]
        
        # 输出最后一列的describe()结果
        describe_result = last_rows[fps_column].describe()
        print("Statistics for the last column:")
        print(describe_result)
        
        # 统计每个唯一值的数量并按降序排序
        value_counts = last_rows[last_column_name].value_counts().sort_values(ascending=False)
        print("\nValue counts for the last column (sorted by count):")
        print(value_counts)
        
    except Exception as e:
        print(f"Error processing file: {e}")

# 替换 'your_file.csv' 为实际的文件路径
file_path = 'data_file.csv'
process_csv(file_path)
