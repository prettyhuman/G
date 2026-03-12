"""
data_generator.py
=================
镍顶吹炉仿真数据（含混淆变量）

核心设计：
  真实因果变量 (causal)   : 每类故障有专属的少数几个传感器真正异常
  混淆变量   (spurious)   : 某些背景传感器会随标签虚假波动，
                            但与故障无物理因果关系
                            → 纯统计方法(GIN/GATv2等)会被误导
                            → CI-GNN 通过 alpha/beta 解耦识别并屏蔽它们

这正是 CI-GNN 论文的核心应用场景。
"""

import numpy as np

VARIABLE_NAMES = [
    "T1(Zone1Temp)", "T2(Zone2Temp)", "T3(Zone3Temp)",
    "P1(FurnPres)",  "P2(LancePres)", "P3(O2Pres)",
    "F1(O2Flow)",    "F2(CoolWater)", "F3(N2Flow)",
    "V1(Vibr1)",     "V2(Vibr2)",     "V3(Acoustic)",
    "CO(%)",         "CO2(%)",         "SO2(%)",
    "T_Mouth",       "T_Shell",        "LancePos",
    "L1(FurnIdx)",   "L2(WearIdx)",
]

N_VARS = 20

NORMAL_MEAN = np.array([
    1280, 1260, 1270,
    150,  160,  155,
    1000, 600,  200,
    1.2,  1.0,  0.8,
    2.5,  18.0, 0.3,
    900,  80,   3.5,
    0.5,  0.2,
], dtype=np.float32)

NORMAL_STD = np.array([
    15,  15,  15,
    5,   5,   5,
    30,  20,  10,
    0.2, 0.2, 0.1,
    0.3, 1.0, 0.05,
    20,  3,   0.1,
    0.05, 0.03,
], dtype=np.float32)

# causal_vars  : 真正发生物理异常的传感器
# spurious_vars: 虚假跟随标签的混淆传感器（对纯统计方法是陷阱）
FAULT_PROFILES = {
    1: {
        "causal_shift":   {0: +9, 1: +11, 2: +13, 15: +7, 16: +5, 18: +6},
        "causal_std":     {0: 1.5, 1: 1.8, 2: 2.0, 15: 1.5},
        "spurious_vars":  [7, 8, 11],
        "spurious_shift": 3.5,
    },
    2: {
        "causal_shift":   {4: +9, 6: -7, 9: +5, 10: +4, 17: -5, 19: +5},
        "causal_std":     {4: 2.5, 6: 2.0, 9: 2.0, 10: 1.8},
        "spurious_vars":  [0, 1, 13],
        "spurious_shift": 3.0,
    },
    3: {
        "causal_shift":   {6: +9, 5: +5, 12: -4, 13: +6, 14: +7, 0: +3},
        "causal_std":     {6: 1.5, 13: 1.5, 14: 2.0},
        "spurious_vars":  [9, 10, 16],
        "spurious_shift": 3.0,
    },
    4: {
        "causal_shift":   {15: +9, 9: +7, 10: +8, 11: +9, 12: +5, 3: +4},
        "causal_std":     {9: 3.0, 10: 3.5, 11: 3.0, 15: 2.0, 17: 4.0},
        "spurious_vars":  [7, 13, 18],
        "spurious_shift": 3.5,
    },
    5: {
        "causal_shift":   {7: -9, 16: +11, 0: +5, 1: +4, 2: +5, 15: +4, 19: +8},
        "causal_std":     {7: 2.0, 16: 1.8, 0: 1.3},
        "spurious_vars":  [4, 11, 14],
        "spurious_shift": 3.0,
    },
}

CORR_GROUPS = [
    [0, 1, 2, 15, 16, 18],
    [3, 4, 5, 6],
    [7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [17, 19],
]

def _build_chol(n=N_VARS, rho=0.6):
    C = np.eye(n, dtype=np.float32)
    for grp in CORR_GROUPS:
        for i in grp:
            for j in grp:
                if i != j:
                    C[i, j] = rho
    vals, vecs = np.linalg.eigh(C)
    C = (vecs * np.clip(vals, 1e-6, None)) @ vecs.T
    return np.linalg.cholesky(C).astype(np.float32)

CHOL = _build_chol()


def generate_window(label: int, window_size: int = 64,
                    rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    mean = NORMAL_MEAN.copy()
    std  = NORMAL_STD.copy()

    if label > 0:
        prof = FAULT_PROFILES[label]
        for idx, mult in prof["causal_shift"].items():
            mean[idx] += mult * NORMAL_STD[idx]
        for idx, mult in prof["causal_std"].items():
            std[idx] *= mult
        # 混淆变量：70% 概率虚假偏移，方向随机
        for idx in prof["spurious_vars"]:
            if rng.random() < 0.70:
                mean[idx] += rng.choice([-1, 1]) * prof["spurious_shift"] * NORMAL_STD[idx]
                std[idx]  *= 1.5

    white = rng.standard_normal((window_size, N_VARS)).astype(np.float32)
    drift = np.cumsum(rng.normal(0, 0.02, (window_size, N_VARS)).astype(np.float32), axis=0)
    return mean + std * (white @ CHOL.T) + std * 0.1 * drift


def extract_features(window: np.ndarray) -> np.ndarray:
    from scipy import stats as sp
    T, N = window.shape
    feats = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        x = window[:, i]
        feats[i, 0] = np.mean(x)
        feats[i, 1] = np.std(x) + 1e-8
        feats[i, 2] = np.max(x)
        feats[i, 3] = np.min(x)
        feats[i, 4] = feats[i, 2] - feats[i, 3]
        feats[i, 5] = float(sp.skew(x))
        feats[i, 6] = float(sp.kurtosis(x))
        feats[i, 7] = float(np.sqrt(np.mean(x ** 2)))
    return feats


CLASS_COUNTS = {0: 4000, 1: 800, 2: 800, 3: 800, 4: 800, 5: 800}

CLASS_NAMES = {
    0: "Normal",
    1: "Lining Erosion",
    2: "Lance Blockage",
    3: "Over-Blowing",
    4: "Mouth Slagging",
    5: "Cooling Failure",
}


def generate_dataset(window_size: int = 64, seed: int = 42):
    rng = np.random.default_rng(seed)
    features_list, labels_list = [], []
    print("Generating dataset (causal signal + spurious correlations)...")
    for label, count in CLASS_COUNTS.items():
        for _ in range(count):
            features_list.append(extract_features(generate_window(label, window_size, rng)))
            labels_list.append(label)
    print(f"  Total: {len(labels_list)} samples")
    return features_list, labels_list
