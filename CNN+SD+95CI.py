import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as stats  # 导入用于计算置信区间的统计模块
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

df = pd.read_csv('GANtBLGTotal_processed.csv')

# 提取标签 y (第一列) 与 特征 X (后序所有数值列)
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

print(f"数据加载成功！总样本数: {X.shape[0]}, 每条曲线特征点数: {X.shape[1]}")

# 自动处理标签：不管原始标签是字符串还是非连续数字(如1,2,3,4)，都严格编码为标准的 0, 1, 2, 3
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"检测到共有 {num_classes} 个不同的类别。标签已被自动映射为 0 至 {num_classes - 1}")
print(f"原始数据集整体类别分布: {Counter(y)}")


# =====================================================================
# 2. 定义源自 models.py 的 CNN 模型结构
# =====================================================================
def Classifier_CNN(input_length, num_classes=4):
    model = tf.keras.Sequential()
    # 输入形状：(特征点数, 1通道)
    model.add(layers.Input(shape=[input_length, 1]))

    # Layer 1
    model.add(layers.Conv1D(kernel_size=3, strides=2, filters=32, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # Layer 2
    model.add(layers.Conv1D(kernel_size=3, strides=2, filters=64, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # Layer 3
    model.add(layers.Conv1D(kernel_size=3, strides=2, filters=128, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # Layer 4
    model.add(layers.Conv1D(kernel_size=3, strides=2, filters=256, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))

    # Flatten & Dense 输出层
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))

    # 编译模型：这里针对整型标签使用 sparse_categorical_crossentropy
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# =====================================================================
# 3. 严格的分层k折交叉验证
# =====================================================================
n_splits = 10
# StratifiedKFold 会严格保证每一折的训练集和测试集中，各类别比例与整体数据一致
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
all_y_test = []
all_y_pred = []

print(f"\n开始执行分层 {n_splits} 折交叉验证...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n" + "=" * 50)
    print(f" >>> 当前执行：第 {fold + 1} 折 / 共 {n_splits} 折 <<< ")
    print("=" * 50)

    # 按照分层索引切分数据
    X_train_f, X_test_f = X[train_idx], X[test_idx]
    y_train_f, y_test_f = y[train_idx], y[test_idx]

    # 【显式验证】打印当前折中的类别分布，确保每一折都包含了所有不同类别的数据
    print(f"本折【训练集】类别分布: {dict(sorted(Counter(y_train_f).items()))}")
    print(f"本折【测试集】类别分布: {dict(sorted(Counter(y_test_f).items()))}")

    # 每一折在内部独立进行 MinMax 归一化，严谨防止测试集信息泄露
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_f_scaled = scaler.fit_transform(X_train_f)
    X_test_f_scaled = scaler.transform(X_test_f)

    # 转换成符合 Keras 1D CNN 的三维张量输入格式: (样本数, 特征数, 1)
    X_train_cnn = np.expand_dims(X_train_f_scaled, axis=-1)
    X_test_cnn = np.expand_dims(X_test_f_scaled, axis=-1)

    # 每一折都必须重新实例化一个干净、未被训练过的网络
    model = Classifier_CNN(input_length=X_train_cnn.shape[1], num_classes=num_classes)

    # 模型训练
    model.fit(
        X_train_cnn, y_train_f,
        epochs=40,  # 针对 209 条样本的轻量化设定，可根据实际拟合情况调整
        batch_size=16,
        verbose=0  # 隐藏逐个Epoch的训练日志，保持控制台整洁
    )

    # 预测当前测试折
    y_pred_probs = model.predict(X_test_cnn, verbose=0)
    y_pred_f = np.argmax(y_pred_probs, axis=1)

    # 计算当前折准确率
    fold_acc = accuracy_score(y_test_f, y_pred_f)
    fold_accuracies.append(fold_acc)
    print(f"第 {fold + 1} 折测试验证完成，准确率(Accuracy): {fold_acc * 100:.2f}%")

    # 收集当前折的真实标签与预测标签，用于最后的统计
    all_y_test.extend(y_test_f)
    all_y_pred.extend(y_pred_f)

# =====================================================================
# 4. 交叉验证最终统计与报告
# =====================================================================
print("\n" + "=====" * 12)
print(" 最终分层交叉验证汇总统计 ")
print("=====" * 12)
print(f"每一折的准确率明细: {[f'Fold {i + 1}: {acc * 100:.2f}%' for i, acc in enumerate(fold_accuracies)]}")

# 计算基本统计量
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies, ddof=1)  # 使用 ddof=1 计算无偏样本标准差

# 基于 t 分布计算 95% 置信区间 (Confidence Interval)
# 自由度 df = 交叉验证折数 - 1
ci_bound = stats.t.ppf(1 - 0.05 / 2, df=n_splits - 1) * (std_acc / np.sqrt(n_splits))
ci_lower = max(0.0, mean_acc - ci_bound)  # 下限不低于 0%
ci_upper = min(1.0, mean_acc + ci_bound)  # 上限不超过 100%

print(f"{n_splits}折平均准确率 (Mean Accuracy): {mean_acc * 100:.2f}% (标准差: ±{std_acc * 100:.2f}%)")
print(f"95% 置信区间 (95% Confidence Interval): [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]")

print("\n总体分类报告 (汇总全部预测结果):")
# target_names 可以自动将 0,1,2,3 映射回原先的标签名称
print(classification_report(all_y_test, all_y_pred, target_names=[str(c) for c in le.classes_]))

print("总体混淆矩阵:")
print(confusion_matrix(all_y_test, all_y_pred))