"""
策略库与目标权重生成单元测试 (Self-contained Strategy Zoo & Target Weights Tests)
"""

import pytest
import numpy as np
import pandas as pd

from strategy.zoo import (
    StrategyRegistry,
    StrategyMetadata,
    StrategyLifecycle,
    MultiFactorStrategy,
)
from strategy.optimizers import HierarchicalRiskParityOptimizer, MeanVarianceOptimizer


def test_strategy_registry_lifecycle():
    reg = StrategyRegistry()
    meta = StrategyMetadata(
        strategy_id="strat_001",
        name="多因子趋势增强策略",
        lifecycle_status=StrategyLifecycle.CANDIDATE,
        tags=["momentum", "multi_factor"],
    )
    mock_df = pd.DataFrame()
    strat = MultiFactorStrategy(metadata=meta, composite_signal_df=mock_df)

    reg.register_strategy(strat)
    assert reg.get_strategy("strat_001").metadata.name == "多因子趋势增强策略"
    assert len(reg.list_strategies(status=StrategyLifecycle.CANDIDATE)) == 1
    assert len(reg.filter_by_tag("momentum")) == 1


def test_multi_factor_strategy_target_weights_generation():
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = ["000001.SZ", "600519.SH", "601088.SH", "688981.SH"]
    rows = []
    np.random.seed(42)
    for d in dates:
        for s in symbols:
            rows.append({
                "trade_date": d,
                "symbol": s,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0 + np.random.randn() * 2,
                "volume": 100000,
            })
    bars_df = pd.DataFrame(rows)
    pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
    composite_signal = pivoted_close.pct_change(10).fillna(0.0)

    meta = StrategyMetadata(
        strategy_id="hrp_top2_strat",
        name="HRP 风险平价动量 Top2 策略",
        lifecycle_status=StrategyLifecycle.AUDITED,
    )
    strategy = MultiFactorStrategy(
        metadata=meta,
        composite_signal_df=composite_signal,
        optimizer=HierarchicalRiskParityOptimizer(),
        top_k=2,
        cov_lookback=20,
    )

    weights_df = strategy.generate_target_weights(bars_df=bars_df, rebalance_interval=5)
    assert weights_df.shape == (len(dates), len(symbols))
    # 验证调仓日活跃权重
    active_weights = weights_df.iloc[-1]
    assert abs(active_weights.sum() - 1.0) < 1e-4
    # 验证选股 top_k
    assert np.count_nonzero(active_weights > 1e-5) <= 2
