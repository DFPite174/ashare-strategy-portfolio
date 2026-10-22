"""
企业级量化投研平台 - 风险平价优化器 (Equal Risk Contribution / Risk Parity Optimizer via scipy)
通过非线性规划求解资产风险贡献相等的权重配置。
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import minimize

from strategy.optimizers.base import PortfolioOptimizer, OptimizationConstraints


class RiskParityOptimizer(PortfolioOptimizer):
    """风险平价 (ERC) 优化器"""

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        cons = constraints or OptimizationConstraints()
        n = len(expected_returns)
        cov = cov_matrix + np.eye(n) * 1e-6

        # 风险贡献计算目标函数: min sum_i ( RC_i - sigma_p / n )^2
        def objective(weights: np.ndarray) -> float:
            w = weights.reshape(-1, 1)
            port_var = float(np.squeeze(w.T @ cov @ w))
            if port_var <= 1e-12:
                return 0.0
            port_std = np.sqrt(port_var)
            # 边际风险贡献 MRC = Cov @ w / sigma_p
            mrc = (cov @ w) / port_std
            # 风险贡献 RC = w * MRC
            rc = (w * mrc).flatten()
            target_rc = port_std / n
            return float(np.sum((rc - target_rc) ** 2))

        # 初始点为等权
        w0 = np.ones(n) / n
        bounds = [(cons.min_weight, cons.max_weight) for _ in range(n)]
        constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            fun=objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 300, "ftol": 1e-9},
        )

        if res.success and res.x is not None:
            w_sol = np.clip(res.x, 0.0, 1.0)
            s = np.sum(w_sol)
            return w_sol / s if s > 0 else w0

        return w0
