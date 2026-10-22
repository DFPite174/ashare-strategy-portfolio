"""
主流开源与经典量化因子库 (Built-in Factor Library)
包含常用技术指标 (RSI, MACD, KDJ, 布林带, 真实波幅), WorldQuant Alpha 101 经典子集与 PIT 基本面因子。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from research.factors.base import BaseFactor, FactorDefinition


# ============================================================================
# 1. 经典技术指标因子库 (Technical Indicators Suite)
# ============================================================================

class RSIFactor(BaseFactor):
    """相对强弱指标 (Relative Strength Index, RSI)"""

    def __init__(self, period: int = 14):
        super().__init__(FactorDefinition(
            name=f"rsi_{period}",
            category="TECHNICAL",
            description=f"Relative Strength Index over {period} periods",
            lookback_days=period * 2,
            direction=1,
        ))
        self.period = period

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        def _calc_rsi(group: pd.Series) -> pd.Series:
            delta = group.diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(self.period, min_periods=self.period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(self.period, min_periods=self.period).mean()
            rs = gain / (loss + 1e-12)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

        df["rsi"] = df.groupby("symbol")["close"].transform(_calc_rsi)
        factor_series = df.set_index(["trade_date", "symbol"])["rsi"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class MACDDiffFactor(BaseFactor):
    """指数平滑异同移动平均线差值 (MACD DIF: EMA(fast) - EMA(slow))"""

    def __init__(self, fast: int = 12, slow: int = 26):
        super().__init__(FactorDefinition(
            name=f"macd_dif_{fast}_{slow}",
            category="TECHNICAL",
            description=f"MACD DIF line ({fast}-{slow})",
            lookback_days=slow * 2,
            direction=1,
        ))
        self.fast = fast
        self.slow = slow

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        def _calc_dif(group: pd.Series) -> pd.Series:
            ema_fast = group.ewm(span=self.fast, adjust=False).mean()
            ema_slow = group.ewm(span=self.slow, adjust=False).mean()
            return (ema_fast - ema_slow) / group  # 归一化为相对百分比

        df["dif"] = df.groupby("symbol")["close"].transform(_calc_dif)
        factor_series = df.set_index(["trade_date", "symbol"])["dif"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class BollingerBandsPctBFactor(BaseFactor):
    """布林带 %B 摆动因子: (Close - Lower) / (Upper - Lower)"""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(FactorDefinition(
            name=f"boll_pct_b_{period}",
            category="VOLATILITY",
            description=f"Bollinger Bands %B ({period}d, {num_std} std)",
            lookback_days=period,
            direction=-1,  # 均值回归反转属性
        ))
        self.period = period
        self.num_std = num_std

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        def _calc_pct_b(group: pd.Series) -> pd.Series:
            mid = group.rolling(self.period).mean()
            std = group.rolling(self.period).std()
            upper = mid + self.num_std * std
            lower = mid - self.num_std * std
            band = upper - lower
            return (group - lower) / (band + 1e-12)

        df["pct_b"] = df.groupby("symbol")["close"].transform(_calc_pct_b)
        factor_series = df.set_index(["trade_date", "symbol"])["pct_b"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class ATRVolatilityFactor(BaseFactor):
    """真实波幅相对指标 (Average True Range, ATR / Close)"""

    def __init__(self, period: int = 14):
        super().__init__(FactorDefinition(
            name=f"atr_vol_{period}",
            category="VOLATILITY",
            description=f"Normalized ATR volatility over {period} periods",
            lookback_days=period,
            direction=-1,  # 低波动异象 (Low Volatility Anomaly)
        ))
        self.period = period

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        prev_close = df.groupby("symbol")["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = df.groupby("symbol")["tr"].transform(lambda s: s.rolling(self.period).mean())
        df["norm_atr"] = atr / df["close"]

        factor_series = df.set_index(["trade_date", "symbol"])["norm_atr"].dropna()
        factor_series.name = self.definition.name
        return factor_series


# ============================================================================
# 2. WorldQuant Alpha 101 经典子集 (Alpha 101 Suite)
# ============================================================================

class Alpha006Factor(BaseFactor):
    """
    WorldQuant Alpha 101: Alpha#6
    公式: -1 * correlation(open, volume, 10)
    """

    def __init__(self, period: int = 10):
        super().__init__(FactorDefinition(
            name=f"wq_alpha_006_{period}",
            category="ALPHA101",
            description="WorldQuant Alpha#6: -1 * correlation(open, volume, 10)",
            lookback_days=period,
            direction=1,
        ))
        self.period = period

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        def _rolling_corr(group: pd.DataFrame) -> pd.Series:
            return -1.0 * group["open"].rolling(self.period).corr(group["volume"])

        df["alpha006"] = df.groupby("symbol")[["open", "volume"]].apply(_rolling_corr).reset_index(level=0, drop=True)
        factor_series = df.set_index(["trade_date", "symbol"])["alpha006"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class Alpha012Factor(BaseFactor):
    """
    WorldQuant Alpha 101: Alpha#12
    公式: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    """

    def __init__(self):
        super().__init__(FactorDefinition(
            name="wq_alpha_012",
            category="ALPHA101",
            description="WorldQuant Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))",
            lookback_days=2,
            direction=1,
        ))

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        d_vol = df.groupby("symbol")["volume"].diff(1)
        d_close = df.groupby("symbol")["close"].diff(1)
        df["alpha012"] = np.sign(d_vol) * (-1.0 * d_close)

        factor_series = df.set_index(["trade_date", "symbol"])["alpha012"].dropna()
        factor_series.name = self.definition.name
        return factor_series
