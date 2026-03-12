"""
graph_builder.py
================
将传感器统计特征矩阵 [N_VARS, 8] 转换为 PyTorch Geometric 的 Data 对象。

图结构设计:
  节点: 20个 (每个传感器/变量一个节点)
  边  : 两种连接方式:
    1) 物理先验边: 根据镍顶吹炉工艺知识定义的因果/相关关系
    2) 数据驱动边: 皮尔逊相关系数 > threshold 的变量对

节点特征: 8维统计特征 [mean, std, max, min, range, skew, kurt, rms]
         归一化后输入模型

可解释性:
  CI-GNN 的 alpha (因果因子) 会高亮与故障最相关的节点/边
  predict_with_explanation() 会返回每个节点的重要性分数
"""

import numpy as np
import torch
from torch_geometric.data import Data

from data_generator import VARIABLE_NAMES, N_VARS, NORMAL_MEAN, NORMAL_STD

# -----------------------------------------------------------------------
# 物理先验边（有向，基于工艺知识）
# -----------------------------------------------------------------------
# 格式: (源节点, 目标节点) 均为 0-indexed (X1=0, ..., L2=19)

PHYSICAL_EDGES = [
    # 供氧系统: 供氧压P3(5) → 氧流量F1(6) → 炉温T1-T3(0,1,2)
    (5, 6), (6, 0), (6, 1), (6, 2),
    # 炉温 → 烟气成分
    (0, 12), (0, 13), (1, 13), (2, 14),
    # 炉压P1(3) ↔ 炉内温度
    (3, 0), (3, 1), (3, 2),
    # 喷枪压P2(4) → 喷枪位置(17)
    (4, 17),
    # 冷却水F2(7) → 炉壳温度T_炉壳(16) → 炉温
    (7, 16), (16, 0), (16, 1),
    # 炉温 → 炉口温度T_炉口(15)
    (0, 15), (1, 15), (2, 15),
    # 振动V1-V3(9,10,11) ← 炉口温度/喷溅
    (15, 9), (15, 10), (15, 11),
    # 潜在变量L1(18) ← 多个温度; L2(19) ← 磨损相关
    (0, 18), (1, 18), (2, 18), (16, 19), (9, 19), (10, 19),
    # 氮气F3(8) → 炉压
    (8, 3),
]

def build_edge_index_physical() -> torch.Tensor:
    """返回物理先验边的 edge_index [2, E]（无向：双向添加）。"""
    src = [u for u, v in PHYSICAL_EDGES] + [v for u, v in PHYSICAL_EDGES]
    dst = [v for u, v in PHYSICAL_EDGES] + [u for u, v in PHYSICAL_EDGES]
    # 去重
    edges = list(set(zip(src, dst)))
    # 加自环
    for i in range(N_VARS):
        edges.append((i, i))
    src, dst = zip(*edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)

PHYSICAL_EDGE_INDEX = build_edge_index_physical()

# -----------------------------------------------------------------------
# 特征归一化参数（基于正常状态均值/标准差，8个统计量）
# -----------------------------------------------------------------------
# 粗略归一化：用 NORMAL_MEAN 和 NORMAL_STD 缩放 mean 特征，
# 其余特征相对比例缩放

def normalize_features(feats: np.ndarray) -> np.ndarray:
    """
    feats: [N_VARS, 8]
    归一化: 每个变量的每个统计量 / (该变量正常均值的绝对值 + eps)
    使特征量纲统一到 [-5, 5] 左右的范围。
    """
    scale = np.abs(NORMAL_MEAN).reshape(-1, 1) + 1e-6   # [N_VARS, 1]
    return (feats / scale).astype(np.float32)

# -----------------------------------------------------------------------
# 核心转换函数
# -----------------------------------------------------------------------

def features_to_graph(feats: np.ndarray, label: int,
                       corr_threshold: float = 0.5) -> Data:
    """
    将 [N_VARS, 8] 统计特征转换为 PyG Data。

    Args:
        feats          : shape [N_VARS, 8]
        label          : 类别标签 (0-5)
        corr_threshold : 数据驱动边的皮尔逊相关阈值

    Returns:
        torch_geometric.data.Data
    """
    # 1. 节点特征归一化
    x = torch.tensor(normalize_features(feats), dtype=torch.float)  # [20, 8]

    # 2. 边: 物理先验边 + 数据驱动边
    # 数据驱动边: 相关系数用 mean 特征列计算
    mean_vec = feats[:, 0]   # [N_VARS]
    corr_mat = np.corrcoef(feats)  # [N_VARS, N_VARS] 用全部8维特征
    # 处理 NaN
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    data_src, data_dst = np.where(
        (np.abs(corr_mat) > corr_threshold) & (~np.eye(N_VARS, dtype=bool))
    )

    # 合并物理边 + 数据驱动边
    extra_src = torch.tensor(data_src, dtype=torch.long)
    extra_dst = torch.tensor(data_dst, dtype=torch.long)

    all_src = torch.cat([PHYSICAL_EDGE_INDEX[0], extra_src])
    all_dst = torch.cat([PHYSICAL_EDGE_INDEX[1], extra_dst])

    # 去重
    edge_set = set(zip(all_src.tolist(), all_dst.tolist()))
    if len(edge_set) > 0:
        s, d = zip(*edge_set)
        edge_index = torch.tensor([list(s), list(d)], dtype=torch.long)
    else:
        edge_index = PHYSICAL_EDGE_INDEX.clone()

    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y,
                num_nodes=N_VARS)


def build_dataset(features_list: list, labels_list: list,
                  corr_threshold: float = 0.5) -> list:
    """
    将所有样本转换为 PyG Data 列表。
    """
    graphs = []
    n = len(features_list)
    for i, (feats, label) in enumerate(zip(features_list, labels_list)):
        g = features_to_graph(feats, label, corr_threshold)
        graphs.append(g)
        if (i + 1) % 1000 == 0:
            print(f"  转换进度: {i+1}/{n}")
    return graphs
