"""
企业级量化投研平台 - 基准等权与信号得分加权优化器 (Equal / Score Weighting)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from strategy.optimizers.base import PortfolioOptimizer, OptimizationConstraints


class EqualWeightOptimizer(PortfolioOptimizer):
    """基准等权分配器"""

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        n = len(expected_returns)
        if n == 0:
            return np.array([])
        return np.ones(n) / n


class ScoreWeightedOptimizer(PortfolioOptimizer):
    """按预测得分截面排名正向加权分配器"""

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        cons = constraints or OptimizationConstraints()
        n = len(expected_returns)
        if n == 0:
            return np.array([])

        # 仅对预期收益为正的资产分配权重
        pos_scores = np.maximum(expected_returns, 0.0)
        s = np.sum(pos_scores)
        if s <= 1e-8:
            return np.ones(n) / n

        raw_weights = pos_scores / s
        clipped = np.clip(raw_weights, cons.min_weight, cons.max_weight)
        cs = np.sum(clipped)
        return clipped / cs if cs > 0 else np.ones(n) / n
