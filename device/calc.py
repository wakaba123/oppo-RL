import pandas as pd
import numpy as np
import argparse


# 定义功耗曲线函数
def power_curve_little(x):
    a = 5.241558774794333e-15
    b = 2.5017801973228364
    c = 3.4619889386290694
    return a * np.power(x, b) + c
    
def power_curve_big(x):
    a = 4.261717048425323e-20
    b = 3.3944174181971385
    c = 17.785960069546174
    return a * np.power(x, b) + c

# 读取 CSV 文件
def read_csv_and_calculate_power(file_path):
    # 读取csv文件
    df = pd.read_csv(file_path)
    
    # 确保 big_cpu_freq 和 little_cpu_freq 列存在
    if 'big_cpu_freq' not in df.columns or 'little_cpu_freq' not in df.columns:
        raise ValueError("CSV文件中缺少 'big_cpu_freq' 或 'little_cpu_freq' 列")

    # 计算功耗
    df['little_power'] = df['little_cpu_freq'].apply(power_curve_little)
    df['big_power'] = df['big_cpu_freq'].apply(power_curve_big)

    # 计算总功耗
    df['total_power'] = df['little_power'] + df['big_power']

    return df[['big_cpu_freq', 'little_cpu_freq', 'little_power', 'big_power', 'total_power', 'fps']]

# 主函数，读取文件并输出功耗结果
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help='输入输出文件的名称')
    args = parser.parse_args()
    file_path = f"{args.name}/{args.name}.csv"  # 替换为你的CSV文件路径
    power_data = read_csv_and_calculate_power(file_path)
    
    # 打印计算结果
    print(np.mean(power_data['total_power'][-300:]))
    print(np.mean(power_data['fps'][-300:]))

    # 如果需要保存结果到新CSV文件
    # power_data.to_csv('power_calculated1.csv', index=False)
