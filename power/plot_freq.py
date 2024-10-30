import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 文件列表
csv_files = ['cluster_0.txt', 'cluster_1.txt', 'cluster_2.txt', 'cluster_3.txt']

# 创建一个 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 调整figsize来设置整体图的大小
axes = axes.flatten()  # 将 2x2 的子图数组展开为一维，方便遍历

# 遍历每个文件和对应的子图
for i, csv_file in enumerate(csv_files):
    data = pd.read_csv(csv_file)
    
    # 创建索引列（0 到 n-1）
    data['index'] = range(len(data))
    
    # 在子图上绘制柱状图
    sns.barplot(data=data, x='index', y='time', ax=axes[i])
    
    # 设置标题和轴标签
    axes[i].set_title(f"Cluster {i}")
    axes[i].set_xlabel("Index")  # 横坐标设为索引
    axes[i].set_ylabel("Time")

# 调整子图布局
plt.tight_layout()
plt.show()
