"""
因子定义与计算基类 (Factor Abstraction Protocol)
标准化 (date, asset) 多重索引因子输出，支持技术面动量与 PIT 基本面因子。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorDefinition:
    """因子元数据定义"""
    name: str
    category: str  # 'MOMENTUM', 'REVERSAL', 'VALUE', 'QUALITY', 'NOISE', 'TRAP'
    description: str
    lookback_days: int
    direction: int = 1  # 1 为正向因子 (值越大越好), -1 为反向因子


class BaseFactor(ABC):
    """因子计算抽象基类"""

    def __init__(self, definition: FactorDefinition):
        self.definition = definition

    @abstractmethod
    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        计算因子值
        :param bars_df: 行情数据，包含 symbol, trade_date, open, high, low, close, volume, adj_factor
        :param fundamentals_df: PIT 财务数据 (可选)
        :return: pd.Series, MultiIndex 为 (trade_date, symbol), 值为 float
        """
        pass


class MomentumFactor(BaseFactor):
    """经典时序动量因子 (N日收益率)"""

    def __init__(self, lookback: int = 20):
        super().__init__(FactorDefinition(
            name=f"momentum_{lookback}d",
            category="MOMENTUM",
            description=f"Past {lookback} days cumulative adjusted return",
            lookback_days=lookback,
            direction=1,
        ))
        self.lookback = lookback

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 使用复权收盘价计算真实收益 (若无复权因子则默认使用收盘价)
        if "adj_factor" in df.columns:
            df["adj_close"] = df["close"] * df["adj_factor"]
        else:
            df["adj_close"] = df["close"]
        
        # 按 symbol 分组并排序计算动量
        df = df.sort_values(["symbol", "trade_date"])
        df["momentum"] = df.groupby("symbol")["adj_close"].pct_change(self.lookback)
        
        # 整理为 MultiIndex (trade_date, symbol)
        factor_series = df.set_index(["trade_date", "symbol"])["momentum"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class MeanReversionFactor(BaseFactor):
    """短期均值回归/反转因子 (-N日收益率)"""

    def __init__(self, lookback: int = 5):
        super().__init__(FactorDefinition(
            name=f"reversal_{lookback}d",
            category="REVERSAL",
            description=f"Negative past {lookback} days return",
            lookback_days=lookback,
            direction=-1,
        ))
        self.lookback = lookback

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if "adj_factor" in df.columns:
            df["adj_close"] = df["close"] * df["adj_factor"]
        else:
            df["adj_close"] = df["close"]
        df = df.sort_values(["symbol", "trade_date"])
        df["reversal"] = -1.0 * df.groupby("symbol")["adj_close"].pct_change(self.lookback)
        
        factor_series = df.set_index(["trade_date", "symbol"])["reversal"].dropna()
        factor_series.name = self.definition.name
        return factor_series


class ROEValueFactor(BaseFactor):
    """PIT 基本面净资产收益率因子 (ROE)"""

    def __init__(self):
        super().__init__(FactorDefinition(
            name="pit_roe",
            category="QUALITY",
            description="Point-In-Time ROE using publicly announced reports",
            lookback_days=90,
            direction=1,
        ))

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        if fundamentals_df is None or fundamentals_df.empty:
            raise ValueError("ROEValueFactor requires non-empty PIT fundamentals_df")

        # 将财报公布日与行情日期对齐
        bars = bars_df[["symbol", "trade_date"]].copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"])
        
        fina = fundamentals_df.copy()
        fina["published_at"] = pd.to_datetime(fina["published_at"])
        fina = fina.sort_values(["symbol", "published_at"]).drop_duplicates(subset=["symbol", "published_at"], keep="last")

        # 针对每个行情日，通过 asof merge 仅获取在 trade_date 之前最新已公开的 ROE
        merged_list = []
        for sym, grp_bars in bars.groupby("symbol"):
            grp_fina = fina[fina["symbol"] == sym].sort_values("published_at")
            if grp_fina.empty:
                continue
            grp_bars = grp_bars.sort_values("trade_date")
            merged = pd.merge_asof(
                grp_bars,
                grp_fina[["published_at", "roe"]],
                left_on="trade_date",
                right_on="published_at",
                direction="backward"
            )
            merged_list.append(merged)

        if not merged_list:
            return pd.Series(dtype=float, name=self.definition.name)

        full_df = pd.concat(merged_list, ignore_index=True).dropna(subset=["roe"])
        factor_series = full_df.set_index(["trade_date", "symbol"])["roe"]
        factor_series.name = self.definition.name
        return factor_series


class RandomNoiseFactor(BaseFactor):
    """纯随机高斯噪声因子 (基线对照组，用于 FDR 显著性多重检验)"""

    def __init__(self, seed: int = 42):
        super().__init__(FactorDefinition(
            name=f"random_noise_{seed}",
            category="NOISE",
            description="Pure random Gaussian noise without alpha",
            lookback_days=1,
            direction=1,
        ))
        self.seed = seed

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        df = bars_df[["symbol", "trade_date"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["noise"] = rng.standard_normal(len(df))
        factor_series = df.set_index(["trade_date", "symbol"])["noise"]
        factor_series.name = self.definition.name
        return factor_series


class LookaheadLeakageFactor(BaseFactor):
    """未来函数恶意测试因子 (故意偷窥 T+1 价格，用于系统红队负向防御检验)"""

    def __init__(self):
        super().__init__(FactorDefinition(
            name="lookahead_leakage_trap",
            category="TRAP",
            description="Malicious lookahead factor reading T+1 close returns",
            lookback_days=1,
            direction=1,
        ))

    def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"])
        # 恶意向前平移，使用明日收益作为今日因子
        df["future_return"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0
        factor_series = df.set_index(["trade_date", "symbol"])["future_return"].dropna()
        factor_series.name = self.definition.name
        return factor_series
