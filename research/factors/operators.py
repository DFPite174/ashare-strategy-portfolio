"""
量化因子核心向量化算子库 (Quantitative Vectorized Operators)
包含常用时序算子 (Time-Series) 与截面算子 (Cross-Sectional)，支持高效无前视偏差计算。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================================
# 时序算子 (Time-Series Operators: 针对每个标的时序独立计算)
# ============================================================================

def ts_delay(df: pd.DataFrame, col: str, period: int = 1) -> pd.Series:
    """时序滞后 (前移) 算子"""
    return df.groupby("symbol")[col].shift(period)


def ts_delta(df: pd.DataFrame, col: str, period: int = 1) -> pd.Series:
    """时序差分算子: x_t - x_{t-period}"""
    return df.groupby("symbol")[col].diff(period)


def ts_pct_change(df: pd.DataFrame, col: str, period: int = 1) -> pd.Series:
    """时序变化率算子: (x_t - x_{t-period}) / x_{t-period}"""
    return df.groupby("symbol")[col].pct_change(period)


def ts_mean(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序移动平均"""
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(period, min_periods=max(1, period // 2)).mean())


def ts_std(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序滚动标准差"""
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(period, min_periods=max(2, period // 2)).std())


def ts_max(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序滚动最大值"""
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(period, min_periods=max(1, period // 2)).max())


def ts_min(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序滚动最小值"""
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(period, min_periods=max(1, period // 2)).min())


def ts_sum(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序滚动求和"""
    return df.groupby("symbol")[col].transform(lambda s: s.rolling(period, min_periods=max(1, period // 2)).sum())


def ts_rank(df: pd.DataFrame, col: str, period: int = 20) -> pd.Series:
    """时序滚动分位数排名 (归一化到 0 ~ 1)"""
    def _rolling_rank(s: pd.Series) -> pd.Series:
        return s.rolling(period, min_periods=max(2, period // 2)).apply(
            lambda x: (pd.Series(x).rank(pct=True).iloc[-1]) if len(x) > 0 else np.nan,
            raw=False
        )
    return df.groupby("symbol")[col].transform(_rolling_rank)


def ts_decay_linear(df: pd.DataFrame, col: str, period: int = 10) -> pd.Series:
    """线性衰减加权移动平均: weight_i = i / sum(1..n)"""
    weights = np.arange(1, period + 1)
    weight_sum = weights.sum()

    def _decay(window):
        if len(window) < period:
            w = np.arange(1, len(window) + 1)
            return np.dot(window, w) / w.sum()
        return np.dot(window, weights) / weight_sum

    return df.groupby("symbol")[col].transform(
        lambda s: s.rolling(period, min_periods=1).apply(_decay, raw=True)
    )


# ============================================================================
# 截面算子 (Cross-Sectional Operators: 针对每个交易日全市场所有标的统一计算)
# ============================================================================

def cs_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """截面百分比排名 (0 ~ 1)"""
    return df.groupby("trade_date")[col].rank(pct=True)


def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """截面标准化 Z-Score: (x - mean) / std"""
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    return df.groupby("trade_date")[col].transform(_zscore)


def cs_demean(df: pd.DataFrame, col: str) -> pd.Series:
    """截面去均值"""
    return df.groupby("trade_date")[col].transform(lambda s: s - s.mean())
