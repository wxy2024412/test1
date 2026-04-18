import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import expon
import math


def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """带通滤波器"""
    if highcut >= 0.5 * fs:
        raise ValueError("高频截止频率超过奈奎斯特频率")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def calculate_bcsm(signal, lam, fs, lowcut=None, highcut=None):
    """计算Box-Cox稀疏测度（BCSM）"""
    # 带通滤波（如果指定频段）
    if lowcut is not None and highcut is not None:
        if highcut <= lowcut:
            raise ValueError("高频截止频率必须大于低频")
        signal = bandpass_filter(signal, lowcut, highcut, fs)

    # 希尔伯特变换与包络计算
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    SE = envelope ** 2  # 平方包络

    # 归一化处理
    SE_sum = np.sum(SE)
    if SE_sum <= 1e-10:
        return 0  # 避免除以零  直接返回基线值，避免后续计算
    normalized_SE = SE / SE_sum

    # 计算权重项
    N = len(signal)
    weighted_terms = []
    epsilon = 1e-20  # 新增：极小值保护    避免后续的term = N * se为0   log(0)的出现

    for se in normalized_SE:
        term = N * se

        # 数值稳定处理（统一处理所有情况）
        term = max(term, epsilon)  # 确保term始终>=epsilon

        if lam != 0:
            weighted = term ** lam
        else:
            weighted = np.log(term)
        weighted_terms.append(weighted)

    # 计算参数C（严格遵循论文公式）
    if lam > 0:
        if float(lam).is_integer():
            # 整数情况：C = (λ+1)!
            C = math.factorial(int(lam) + 1)
        else:
            # 非整数情况：C = E[Y^{λ+1}] / (E[Y])^{λ+1}
            rate = 0.5  # 指数分布参数λ=0.5
            scale = 1 / rate  # scale=2
            k = lam + 1
            moment = math.gamma(k) * (scale ** k )# E[Y^{λ+1}]
            E_Y = scale  # E[Y] = 1/rate = 2
            C = moment / (E_Y ** (lam + 1))
    else:  # λ=0
        # C = 1 - γ (γ为欧拉-马歇罗尼常数)
        C = 1 - np.euler_gamma

    # 计算BCSM值
    bcsm = np.sum(normalized_SE * weighted_terms) - C
    return bcsm


def find_optimal_band(signal, fs, freq_bands, lam):
    """在不同频段中寻找最优BCSM值"""
    max_bcsm = -np.inf
    optimal_band = None

    for (low, high) in freq_bands:
        bcsm = calculate_bcsm(signal, lam, fs, lowcut=low, highcut=high)
        if bcsm > max_bcsm and not np.isinf(bcsm):
            max_bcsm = bcsm
            optimal_band = (low, high)

    return optimal_band, max_bcsm


# 使用示例
if __name__ == "__main__":
    # 生成测试信号（含脉冲成分）
    fs = 20000
    t = np.linspace(0, 1, fs)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 基础信号
    signal += np.random.normal(0, 0.1, len(t))  # 添加噪声
    signal[::500] += 2.0  # 添加周期性脉冲

    # 定义频段划分（根据实际应用调整）
    freq_bands = [(1000, 3000), (3000, 5000), (5000, 7000)]

    # 计算λ=0时的最优频段
    optimal_band, bcsm_value = find_optimal_band(signal, fs, freq_bands, lam=0)
    print(f"最优频段: {optimal_band}, BCSM值: {bcsm_value:.4f}")

    # 验证整数λ情况（λ=2）
    _, bcsm_kurtosis = find_optimal_band(signal, fs, freq_bands, lam=1)
    print(f"Kurtosis(λ=1)基准值: {bcsm_kurtosis:.4f}")




