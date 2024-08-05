import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_columns(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 获取列名
    columns = data.columns
    
    # 为每一列生成折线图
    for column in columns:
        plt.figure()
        plt.plot(data[column])
        plt.title(f"Line plot for {column}")
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.grid(True)
        plt.savefig('pics/' + column + '.png')
        # plt.show()

# 示例使用
file_path = 'output.csv'  # 替换为你的CSV文件路径
plot_csv_columns(file_path)
