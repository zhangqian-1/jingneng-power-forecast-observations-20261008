"""工具函数：时间特征提取"""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_feature_frame(ts: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """从时间戳提取周期性时间特征"""
    idx = pd.DatetimeIndex(ts)
    minute = idx.hour * 60 + idx.minute
    day_angle = 2 * np.pi * minute / (24 * 60)
    week_angle = 2 * np.pi * idx.dayofweek / 7
    month_angle = 2 * np.pi * (idx.month - 1) / 12
    year_angle = 2 * np.pi * (idx.dayofyear - 1) / 366

    return pd.DataFrame(
        {
            "hour_sin": np.sin(day_angle),
            "hour_cos": np.cos(day_angle),
            "weekday_sin": np.sin(week_angle),
            "weekday_cos": np.cos(week_angle),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "doy_sin": np.sin(year_angle),
            "doy_cos": np.cos(year_angle),
            "is_weekend": (idx.dayofweek >= 5).astype(float),
            "is_heating_season": np.isin(idx.month, [11, 12, 1, 2, 3]).astype(float),
        }
    )


def time_slot_index(ts: pd.Series | pd.DatetimeIndex | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算时间槽索引（用于状态特征查找）"""
    idx = pd.DatetimeIndex(ts)
    minute_idx = (idx.hour * 60 + idx.minute) // 15
    week_slot = idx.dayofweek.to_numpy(dtype=int) * 96 + minute_idx.to_numpy(dtype=int)
    return week_slot, minute_idx.to_numpy(dtype=int)
