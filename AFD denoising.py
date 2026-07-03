import numpy as np
import pandas as pd
from scipy.signal import hilbert
from numpy.fft import fft, ifft
from numpy.matlib import repmat


# =====================================================================
# 1. 复制自 2_AFD分解.ipynb 的核心基础函数
# =====================================================================

def e_a(a, t):
    """评估函数"""
    return ((1 - np.absolute(a) ** 2) ** 0.5) / (1 - np.conjugate(a) * (np.e ** (1j * t)))


def weight(n, order):
    """数值积分的权重函数"""
    return np.ones((n, 1), 'complex')


def intg(f, g, W):
    if np.ndim(g) == 1:
        g = np.array([g])
    y = f.dot(g.T * W)
    if np.ndim(y) != 1:
        y = y[0, 0]
    return y / (np.shape(f)[1])


def FFT_AFD(s, max_level=10, M=20):
    """
    基于FFT的自适应傅里叶分解 (AFD)
    """
    if np.ndim(s) == 1:
        s = np.array([s])
    K = np.shape(s)[1]
    t = np.array([np.arange(0, 2 * np.pi, 2 * np.pi / K)])

    # 转换为解析信号
    if np.isreal(s).all():
        G = hilbert(s)
    else:
        G = s.copy()

    # 生成 a_n 字典
    if np.size(M) == 1:
        abs_a = np.array([np.arange(0, 1, 1.0 / M)])
    else:
        abs_a = np.array([M]) if np.ndim(M) == 1 else M.copy()

    temp = np.zeros((1, np.size(abs_a)), 'complex')
    for k in np.arange(0, np.size(abs_a)):
        temp[0, k] = complex(abs_a[0, k])
    abs_a = temp.copy()

    # 生成评估器基
    Base = np.zeros((np.size(abs_a), np.size(t)), 'complex')
    for k in np.arange(0, np.shape(Base)[0]):
        Base[k, :] = fft(e_a(abs_a[0, k], t), np.size(t))

    Weight = weight(K, 6)
    an = np.zeros((1, max_level + 1), 'complex')
    coef = np.zeros((1, max_level + 1), 'complex')
    coef[0, 0] = intg(G, np.ones((1, np.size(t))), Weight)

    for n in np.arange(1, np.size(an)):
        e_an = e_a(an[0, n - 1], t)
        G = (G - coef[0, n - 1] * e_an) * (1 - np.conjugate(an[0, n - 1]) * (np.e ** (1j * t))) / (
                    np.e ** (1j * t) - an[0, n - 1])
        S1 = ifft(repmat(fft(G * Weight.conj().T, np.size(t)), np.shape(Base)[0], 1) * Base, np.size(t), 1)
        max_loc = np.nonzero(np.absolute(S1) == np.absolute(S1).max())
        an[0, n] = abs_a[0, max_loc[0][0]] * np.e ** (1j * t[0, max_loc[1][0]])
        coef[0, n] = np.conjugate(e_a(an[0, n], t).dot(G.conj().T * Weight))[0, 0] / K

    return 1, an, coef, t


def inverse_AFD(an, coef, t, standard='level', standard_value=float("inf")):
    """
    逆积分重构信号 (AFD去噪重构)
    """
    if np.ndim(an) == 1:
        an = np.array([an])
    if np.ndim(coef) == 1:
        coef = np.array([coef])
    if np.ndim(t) == 1:
        t = np.array([t])

    Weight = weight(np.size(t), 6)
    tem_B = (np.sqrt(1 - np.absolute(an[0, 0]) ** 2) / (1 - np.conjugate(an[0, 0]) * np.e ** (t * 1j)))
    G_recovery = coef[0, 0] * tem_B
    n = 0

    if standard.lower() == 'level':
        current_value = 0
        target_value = min((np.size(an) - 1, standard_value))
    elif standard.lower() == 'energy':
        current_value = intg(np.real(G_recovery), np.real(G_recovery), Weight)
        target_value = standard_value

    while n < np.size(an) - 1 and current_value < target_value:
        n = n + 1
        tem_B = (np.sqrt(1 - np.absolute(an[0, n]) ** 2) / (1 - np.conjugate(an[0, n]) * np.e ** (t * 1j))) * (
                (np.e ** (1j * t) - an[0, n - 1]) / (np.sqrt(1 - np.absolute(an[0, n - 1]) ** 2))) * tem_B
        G_recovery = G_recovery + coef[0, n] * tem_B
        if standard.lower() == 'level':
            current_value = n
        elif standard.lower() == 'energy':
            current_value = intg(np.real(G_recovery), np.real(G_recovery), Weight)

    return G_recovery, n


# =====================================================================
# 2. 数据读取与批处理流程
# =====================================================================

# 读取横轴和纵轴数据
df_raw = pd.read_csv('tBLG_raw_processed.csv')
df_snr0 = pd.read_csv('tBLG_SNR0_processed.csv')

# 提取标签列
labels = df_snr0.iloc[:, 0].values

# 提取数值矩阵 (除去第一列的 Label)
# 注：根据你的描述，raw提供x轴，snr0提供y轴。AFD是在区间上对 y 进行正交分解重构。
y_data = df_snr0.iloc[:, 1:].values  # 形状为 (209, 采样点数)

# 创建一个存储重构去噪后数据的矩阵
y_denoised = np.zeros_like(y_data)

# 循环遍历 209 条曲线
num_curves = y_data.shape[0]
print(f"开始对 {num_curves} 条曲线进行 AFD 分解去噪...")

# 参数设定：max_level控制保留的高频/低频分量级数，可根据实际去噪要求微调
MAX_LEVEL = 10

for i in range(num_curves):
    signal = y_data[i, :]

    # 1. 运行快速自适应傅里叶分解
    state, an, coef, t = FFT_AFD(signal, max_level=MAX_LEVEL, M=20)

    # 2. 逆变换重构信号（提取实部作为去噪后的信号）
    G_recovery, _ = inverse_AFD(an, coef, t, standard='level', standard_value=MAX_LEVEL)
    signal_rec = np.real(G_recovery).flatten()

    # 3. 存入结果矩阵
    y_denoised[i, :] = signal_rec

    if (i + 1) % 50 == 0 or (i + 1) == num_curves:
        print(f"已处理完成: {i + 1}/{num_curves}")

# =====================================================================
# 3. 保存去噪重构后的数据集
# =====================================================================

# 将重构后的 y 数据重新与标签拼接
columns = df_snr0.columns
df_denoised_output = pd.DataFrame(y_denoised, columns=columns[1:])
df_denoised_output.insert(0, columns[0], labels)

# 保存为新文件
output_filename = 'tBLG_SNR0_AFD_denoised.csv'
df_denoised_output.to_csv(output_filename, index=False)
print(f"去噪重构成功！结果已保存至: {output_filename}")