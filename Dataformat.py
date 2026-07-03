import pandas as pd

def process_dataset(file_path, output_path):
    # 1. 读取数据，因为是用分号分隔的，所以使用 sep=';'
    # header=None 表示原始文件没有列名
    df = pd.read_csv(file_path, sep=";", header=None)

    # 2. 提取最后一列作为标签 (Target)
    # df.iloc[:, -1] 代表最后一列
    target_column = df.iloc[:, -1]

    # 3. 提取除最后一列之外的所有特征数据
    feature_data = df.iloc[:, :-1]

    # 4. 将标签插入到第一列 (位置 0)，并为特征列重命名（可选，方便查看）
    # 创建一个新的 DataFrame，第一列是 Label，后面是数据
    processed_df = pd.DataFrame(index=df.index)
    processed_df["Label"] = target_column

    # 将特征数据拼接到后面
    processed_df = pd.concat([processed_df, feature_data], axis=1)

    # 5. 保存为标准的 CSV 文件（默认用逗号分隔，每个数据一个单元格）
    # index=False 表示不保存行索引
    processed_df.to_csv(output_path, index=False)
    print(f"处理完成！文件已成功保存至: {output_path}")


# --- 运行处理 ---

# 处理数据 (raw)
process_dataset("GANtBLGTotal.csv", "GANtBLGTotal_processed.csv")

