"""数据归一化类"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Normalizer:
    """特征和目标变量的归一化器"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, target_mean: float, target_std: float):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.target_mean = float(target_mean)
        self.target_std = float(max(target_std, 1e-6))

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_cols: list[str], train_end: pd.Timestamp, target_col: str = "total_power") -> "Normalizer":
        """从训练数据拟合归一化参数"""
        train = df[df["ts"] <= train_end]
        values = train[feature_cols].to_numpy(dtype=np.float32)
        target = train[target_col].to_numpy(dtype=np.float32)
        return cls(
            mean=np.nanmean(values, axis=0),
            std=np.nanstd(values, axis=0),
            target_mean=float(np.nanmean(target)),
            target_std=float(np.nanstd(target)),
        )

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        """归一化输入特征"""
        return (values.astype(np.float32) - self.mean) / self.std

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        """归一化目标变量"""
        return (values.astype(np.float32) - self.target_mean) / self.target_std

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        """反归一化目标变量（预测值转回原始尺度）"""
        return values.astype(float) * self.target_std + self.target_mean
