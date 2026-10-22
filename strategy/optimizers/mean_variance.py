"""
企业级量化投研平台 - 马科维茨均值-方差优化器 (Mean-Variance Optimizer via cvxpy)
基于凸优化求解带换手率约束、单票上限约束的标准 Markowitz 二次规划。
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import cvxpy as cp

from strategy.optimizers.base import PortfolioOptimizer, OptimizationConstraints


class MeanVarianceOptimizer(PortfolioOptimizer):
    """马科维茨二次规划求解器"""

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        cons = constraints or OptimizationConstraints()
        n = len(expected_returns)

        # 协方差矩阵对角微扰，确保半正定
        cov_regularized = cov_matrix + np.eye(n) * 1e-6

        w = cp.Variable(n)
        curr_w = current_weights if current_weights is not None else np.zeros(n)

        # 目标函数: max mu^T w - 0.5 * gamma * w^T Sigma w - turnover_penalty * |w - w_curr|
        ret_term = expected_returns @ w
        risk_term = 0.5 * cons.risk_aversion * cp.quad_form(w, cov_regularized)
        turnover_term = cons.turnover_penalty * cp.norm(w - curr_w, 1)

        objective = cp.Maximize(ret_term - risk_term - turnover_term)

        # 约束条件
        conditions = [
            cp.sum(w) == 1.0,
            w >= cons.min_weight,
            w <= cons.max_weight,
        ]

        if cons.max_turnover is not None:
            conditions.append(cp.norm(w - curr_w, 1) <= cons.max_turnover)

        prob = cp.Problem(objective, conditions)
        try:
            # 优先使用 OSQP 或 Clarabel 求解
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
                weights = np.array(w.value).flatten()
                # 截断微小数值误差
                weights = np.clip(weights, 0.0, 1.0)
                s = weights.sum()
                return weights / s if s > 0 else np.ones(n) / n
        except Exception:
            pass

        # 求解失败退化为等权
        return np.ones(n) / n
