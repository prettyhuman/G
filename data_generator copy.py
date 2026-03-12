"""
data_generator.py
=================
为镍顶吹炉故障诊断生成仿真传感器数据。

变量定义（共20个）:
  X1 ~ X3  : 炉内温度传感器 T1-T3  (℃, 正常区间 1200-1350)
  X4 ~ X6  : 压力传感器 P1-P3      (kPa, 正常区间 120-180)
  X7 ~ X9  : 流量传感器 F1-F3      (Nm³/h, 正常区间 800-1200)
  X10 ~ X12: 振动/声学传感器 V1-V3  (mm/s, 正常区间 0.5-2.0)
  X13 ~ X15: 烟气成分 CO/CO2/SO2   (%, 正常区间 各不同)
  X16 ~ X18: 炉口温度/炉壳温度/喷枪位置 (正常区间 各不同)
  L1       : 潜在变量1 (炉况综合指数)
  L2       : 潜在变量2 (设备磨损指数)

6类标签:
  0: 正常运行
  1: 炉衬侵蚀/穿炉  — 炉内温度异常升高，热点迁移
  2: 喷枪堵塞/烧损  — 供氧压力突升后骤降，流量异常
  3: 供氧异常(过吹) — 氧气流量持续偏高，CO2升高
  4: 炉口积渣/喷溅  — 炉口温度升高，振动增大
  5: 冷却系统异常   — 冷却水流量下降，炉壳温度升高

每类采样数:
  正常: 4000   故障1-5: 各800
  每个样本 = 一段时间窗口(window_size=64步) → 提取统计特征 → 图节点特征
"""

import numpy as np

# -----------------------------------------------------------------------
# 变量基准值与正常范围
# -----------------------------------------------------------------------

VARIABLE_NAMES = [
    "T1(炉温区1)", "T2(炉温区2)", "T3(炉温区3)",   # X1-X3
    "P1(炉压)",   "P2(喷枪压)",  "P3(供氧压)",      # X4-X6
    "F1(氧流量)", "F2(冷却水)",  "F3(氮气流)",       # X7-X9
    "V1(振动1)",  "V2(振动2)",   "V3(声学)",         # X10-X12
    "CO(%)",      "CO2(%)",      "SO2(%)",            # X13-X15
    "T_炉口",     "T_炉壳",      "喷枪位置",          # X16-X18
    "L1(炉况)",   "L2(磨损)",                         # L1, L2
]

N_VARS = 20

# 正常状态基准均值
NORMAL_MEAN = np.array([
    1280, 1260, 1270,   # T1-T3  (℃)
    150,  160,  155,    # P1-P3  (kPa)
    1000, 600,  200,    # F1-F3  (Nm³/h)
    1.2,  1.0,  0.8,   # V1-V3  (mm/s)
    2.5,  18.0, 0.3,   # CO, CO2, SO2  (%)
    900,  80,   3.5,   # T_炉口(℃), T_炉壳(℃), 喷枪位置(m)
    0.5,  0.2,         # L1, L2  (归一化指数)
], dtype=np.float32)

# 正常状态标准差（噪声水平）
NORMAL_STD = np.array([
    15,  15,  15,
    5,   5,   5,
    30,  20,  10,
    0.2, 0.2, 0.1,
    0.3, 1.0, 0.05,
    20,  3,   0.1,
    0.05, 0.03,
], dtype=np.float32)

# 故障偏移幅度（相对于正常均值的偏移，以NORMAL_STD为单位）
# 每行: [fault_idx(1-5), variable_indices, shift_multipliers]
FAULT_PROFILES = {
    # 故障1: 炉衬侵蚀 — T1-T3升高，T_炉口升高，T_炉壳升高，L1漂移
    1: {
        "shift":  {0: +8, 1: +10, 2: +12,   # T1+120℃, T2+150℃, T3+180℃
                   15: +6, 16: +5,            # T_炉口+120℃, T_炉壳+15℃
                   3: +3,                      # P1升高
                   18: +6},                    # L1漂移
        "std_mult": {0: 1.5, 1: 1.8, 2: 2.0, 15: 1.5},
    },
    # 故障2: 喷枪堵塞 — P2异常(先升后降)，F1下降，喷枪位置异常，V1-V2增大
    2: {
        "shift":  {4: +8, 6: -6,              # P2+40kPa, F1-180Nm³/h
                   9: +4, 10: +3,              # V1+0.8, V2+0.6
                   17: -5,                     # 喷枪位置偏移
                   19: +5},                    # L2(磨损)升高
        "std_mult": {4: 2.5, 6: 2.0, 9: 2.0, 10: 1.8},
    },
    # 故障3: 供氧异常(过吹) — F1持续偏高，CO2升高，CO降低，P3升高
    3: {
        "shift":  {6: +8,                     # F1+240Nm³/h (过吹)
                   5: +5,                      # P3升高
                   12: -4,                     # CO降低 (过氧化)
                   13: +5,                     # CO2升高
                   14: +6,                     # SO2升高
                   0: +3, 1: +3, 2: +3},       # 温度轻微升高
        "std_mult": {6: 1.5, 13: 1.5, 14: 2.0},
    },
    # 故障4: 炉口积渣/喷溅 — T_炉口升高，V1-V3剧增，CO升高，喷枪位置波动
    4: {
        "shift":  {15: +8,                    # T_炉口+160℃
                   9: +6, 10: +7, 11: +8,     # 振动剧增
                   12: +5,                     # CO升高 (不完全燃烧)
                   17: +0,                     # 喷枪位置(波动增大)
                   3: +4},                     # P1升高
        "std_mult": {9: 3.0, 10: 3.5, 11: 3.0, 15: 2.0, 17: 4.0},
    },
    # 故障5: 冷却系统异常 — F2(冷却水)下降，T_炉壳升高，T1-T3升高，L2升高
    5: {
        "shift":  {7: -8,                     # F2冷却水-160Nm³/h
                   16: +10,                    # T_炉壳+30℃
                   0: +5, 1: +4, 2: +5,       # T1-T3升高
                   15: +4,                     # T_炉口升高
                   19: +8},                    # L2(磨损)高
        "std_mult": {7: 2.0, 16: 1.8, 0: 1.3, 1: 1.3},
    },
}

# 变量间物理相关矩阵（用于生成协相关噪声）
# 简化为分组相关：同组变量间相关系数0.6
CORR_GROUPS = [
    [0, 1, 2, 15, 16, 18],   # 温度组
    [3, 4, 5, 6],             # 压力-流量组
    [7, 8],                   # 冷却组
    [9, 10, 11],              # 振动组
    [12, 13, 14],             # 烟气组
    [17, 19],                 # 喷枪-潜变量组
]

def _build_corr_matrix(n=N_VARS, rho=0.6):
    C = np.eye(n, dtype=np.float32)
    for group in CORR_GROUPS:
        for i in group:
            for j in group:
                if i != j:
                    C[i, j] = rho
    # Make it valid PSD via nearest PSD
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-6, None)
    C = (eigvecs * eigvals) @ eigvecs.T
    return C

CORR_MATRIX = _build_corr_matrix()
CHOL        = np.linalg.cholesky(CORR_MATRIX).astype(np.float32)


# -----------------------------------------------------------------------
# 生成单个样本 (window_size 步时序 → shape [window_size, N_VARS])
# -----------------------------------------------------------------------

def generate_window(label: int, window_size: int = 64,
                    rng: np.random.Generator = None) -> np.ndarray:
    """
    生成一段时间窗口的多变量传感器数据。
    返回 shape [window_size, N_VARS] 的 float32 数组。
    """
    if rng is None:
        rng = np.random.default_rng()

    mean = NORMAL_MEAN.copy()
    std  = NORMAL_STD.copy()

    if label > 0:
        profile = FAULT_PROFILES[label]
        for idx, mult in profile["shift"].items():
            mean[idx] += mult * NORMAL_STD[idx]
        for idx, mult in profile["std_mult"].items():
            std[idx] *= mult

    # 生成相关噪声
    white = rng.standard_normal((window_size, N_VARS)).astype(np.float32)
    correlated = white @ CHOL.T   # [window_size, N_VARS]

    # 添加时序漂移 (缓慢随机游走)
    drift = np.cumsum(rng.normal(0, 0.02, (window_size, N_VARS))
                      .astype(np.float32), axis=0)

    data = mean + std * correlated + std * 0.1 * drift
    return data


# -----------------------------------------------------------------------
# 提取统计特征 (window → 节点特征向量)
# -----------------------------------------------------------------------

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    从 [window_size, N_VARS] 时序数据提取每个变量的统计特征。
    每个变量提取8个特征: mean, std, max, min, range, skew, kurt, rms
    返回 shape [N_VARS, 8] 的特征矩阵 → 作为图的节点特征。
    """
    from scipy import stats as sp_stats

    T, N = window.shape
    feats = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        x = window[:, i]
        feats[i, 0] = np.mean(x)
        feats[i, 1] = np.std(x) + 1e-8
        feats[i, 2] = np.max(x)
        feats[i, 3] = np.min(x)
        feats[i, 4] = feats[i, 2] - feats[i, 3]          # range
        feats[i, 5] = float(sp_stats.skew(x))
        feats[i, 6] = float(sp_stats.kurtosis(x))
        feats[i, 7] = float(np.sqrt(np.mean(x ** 2)))     # RMS

    return feats


# -----------------------------------------------------------------------
# 生成整个数据集
# -----------------------------------------------------------------------

CLASS_COUNTS = {
    0: 4000,   # 正常
    1: 800,    # 炉衬侵蚀
    2: 800,    # 喷枪堵塞
    3: 800,    # 供氧异常
    4: 800,    # 炉口积渣
    5: 800,    # 冷却系统异常
}

CLASS_NAMES = {
    0: "正常运行",
    1: "炉衬侵蚀/穿炉",
    2: "喷枪堵塞/烧损",
    3: "供氧异常(过吹)",
    4: "炉口积渣/喷溅",
    5: "冷却系统异常",
}

def generate_dataset(window_size: int = 64, seed: int = 42):
    """
    生成所有样本。
    返回 (all_features, all_labels) 其中:
      all_features: list of np.ndarray shape [N_VARS, 8]
      all_labels:   list of int
    """
    rng = np.random.default_rng(seed)
    features_list = []
    labels_list   = []

    print("生成仿真数据集...")
    for label, count in CLASS_COUNTS.items():
        print(f"  类别 {label} ({CLASS_NAMES[label]}): {count} 个样本")
        for _ in range(count):
            window = generate_window(label, window_size, rng)
            feats  = extract_features(window)
            features_list.append(feats)
            labels_list.append(label)

    print(f"  总计: {len(labels_list)} 个样本")
    return features_list, labels_list
