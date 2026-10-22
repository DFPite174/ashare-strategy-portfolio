"""
企业级量化投研平台 - 复合多因子动态优化策略实现 (Multi-Factor Strategy)
将假说、自动因子信号与组合优化器无缝装配为生产级标准策略。
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from strategy.zoo.base_strategy import BaseStrategy, StrategyMetadata
from strategy.optimizers.base import PortfolioOptimizer, OptimizationConstraints
from strategy.optimizers.hrp import HierarchicalRiskParityOptimizer
from strategy.optimizers.equal_weight import EqualWeightOptimizer


class MultiFactorStrategy(BaseStrategy):
    """
    通用多因子加权与组合优化策略
    """

    def __init__(
        self,
        metadata: StrategyMetadata,
        composite_signal_df: pd.DataFrame,
        optimizer: Optional[PortfolioOptimizer] = None,
        constraints: Optional[OptimizationConstraints] = None,
        top_k: int = 2,
        cov_lookback: int = 40,
    ):
        super().__init__(metadata)
        self.composite_signal_df = composite_signal_df
        self.optimizer = optimizer or HierarchicalRiskParityOptimizer()
        self.constraints = constraints or OptimizationConstraints()
        self.top_k = top_k
        self.cov_lookback = cov_lookback

    def generate_target_weights(
        self,
        bars_df: pd.DataFrame,
        rebalance_interval: int = 10,
    ) -> pd.DataFrame:
        pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
        timeline = sorted(list(pivoted_close.index))
        symbols = sorted(list(pivoted_close.columns))
        n_days = len(timeline)

        daily_returns = pivoted_close.pct_change().fillna(0.0)
        target_weights_df = pd.DataFrame(0.0, index=timeline, columns=symbols)

        curr_weights = np.zeros(len(symbols))

        # 调仓循环
        start_idx = max(self.cov_lookback, 10)
        for i in range(start_idx, n_days, rebalance_interval):
            dt = timeline[i]
            if dt not in self.composite_signal_df.index:
                continue

            scores_today = self.composite_signal_df.loc[dt].dropna()
            if scores_today.empty or len(scores_today) < self.top_k:
                continue

            # 选取 Top-K 标的
            top_syms = scores_today.nlargest(self.top_k).index.tolist()
            sub_indices = [symbols.index(s) for s in top_syms]

            # 提取回看区间的收益率序列用于计算协方差
            sub_rets = daily_returns.iloc[max(0, i - self.cov_lookback):i][top_syms]
            cov_mat = sub_rets.cov().values if len(sub_rets) > 5 else np.eye(len(top_syms))
            exp_rets = scores_today.loc[top_syms].values

            # 求解局部组合优化权重
            opt_weights = self.optimizer.optimize(
                expected_returns=exp_rets,
                cov_matrix=cov_mat,
                current_weights=curr_weights[sub_indices],
                constraints=self.constraints,
            )

            # 映射回全资产权重
            full_w = np.zeros(len(symbols))
            for idx_in_top, sym_idx in enumerate(sub_indices):
                full_w[sym_idx] = opt_weights[idx_in_top]

            curr_weights = full_w

            # 填充到后续持有期
            end_step = min(i + rebalance_interval, n_days)
            for step in range(i, end_step):
                target_weights_df.iloc[step] = curr_weights

        return target_weights_df
