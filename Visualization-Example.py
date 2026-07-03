import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据（让 Pandas 自动将第一行识别为表头/X轴刻度）
# 确保文件名与你的文件路径一致
df = pd.read_csv('tBLG_raw_processed.csv')

# 2. 提取 X 轴数值
# df.columns[1:] 获取除第一列 'Target' 之外的所有列名，即 1200 到 3000 的刻度
x_values = df.columns[1:].astype(float).values

# 3. 修复类别列（Target）的类型转换问题
# 先转成 float 解决科学计数法文本问题，再转成 int
df['Target'] = df['Target'].astype(float).astype(int)

# 设置随机种子以确保每次运行抽取的随机曲线一致（如果不需要一致，可以删除这行）
np.random.seed(42)

# 定义所有分类
categories = [0, 1, 2, 3]

# =======================================================
# 图片 1 到 图片 4：类别 0, 1, 2, 3 分别随机抽取 3 条曲线
# =======================================================
for cat in categories:
    # 创建独立画布
    fig, ax = plt.subplots(figsize=(8, 5))

    # 筛选出当前类别的数据
    cat_data = df[df['Target'] == cat]

    # 随机抽取 3 行（如果某类别数据少于3条，则有多少取多少）
    n_samples = min(5, len(cat_data))
    sampled_data = cat_data.sample(n=n_samples)

    # 绘制曲线
    for i, (_, row) in enumerate(sampled_data.iterrows()):
        # 提取当前行的 Y 轴响应值
        y_values = row.iloc[1:].astype(float).values
        ax.plot(x_values, y_values, label=f'Sample {i + 1}')

    ax.set_title(f'Category {cat} - 3 Random Curves')
    ax.set_xlabel('X Axis (1200 - 3000)')
    ax.set_ylabel('Y Axis')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存图片（会分别保存为 category_0_random_3.png, category_1_random_3.png 等）
    plt.savefig(f'category_{cat}_random_3.png', dpi=300)
    plt.close(fig)

# =======================================================
# 图片 5：四个类别各随机选一条曲线在同一张图上对比
# =======================================================
fig, ax = plt.subplots(figsize=(10, 6))

for cat in categories:
    cat_data = df[df['Target'] == cat]

    if not cat_data.empty:
        # 每个类别随机抽取 1 行
        sampled_row = cat_data.sample(n=1).iloc[0]
        y_values = sampled_row.iloc[1:].astype(float).values
        ax.plot(x_values, y_values, label=f'Category {cat}')

ax.set_title('Comparison of 4 Categories (1 Random Curve Each)')
ax.set_xlabel('X Axis (1200 - 3000)')
ax.set_ylabel('Y Axis')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存最后一张对比图
plt.savefig('categories_comparison.png', dpi=300)
plt.close(fig)

print("所有图片已成功生成并保存！")