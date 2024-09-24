import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# mark = "douyin_with_douyin_model_2"
# file_path = f"output_{mark}.csv"

def plot_csv_columns(dir, file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 获取列名
    columns = data.columns
    # 判断文件夹是否存在
    if not os.path.exists(f'{dir}/pics'):
        # 如果文件夹不存在，则创建它
        os.makedirs(f'{dir}/pics')
        print(f"文件夹 pics 已创建。")
    else:
        print(f"文件夹 pics 已存在。")
    
    # 为每一列生成折线图
    for column in columns:
        plt.figure()
        plt.plot(data[column][:-300])
        plt.title(f"Line plot for {column}")
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.grid(True)
        plt.savefig(f'{dir}/pics/' + column + '.png')
        # plt.show()

    # plt.figure()
    # plt.plot(data['loss'])
    # plt.title(f"Line plot for loss")
    # plt.ylim(0,0.2)
    # plt.xlabel('Index')
    # plt.ylabel(column)
    # plt.grid(True)
    # plt.savefig('pics/' +'loss2'+ '.png')
    
    

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='输入输出文件的名称')
args = parser.parse_args()
file_path = f'{args.name}/{args.name}.csv'
# 示例使用
plot_csv_columns(args.name, file_path)

df = pd.read_csv(file_path)

# 检查行数是否足够
if len(df) < 300:
    print("数据行数不足300行")
else:
    # 计算每一列的后300行的平均值
    averages = df.iloc[-300:].mean()

    # 输出每个字段及其对应的平均值
    print("后300行的平均值:")
    for column, avg in averages.items():
        print(f"{column}: {avg}")