"""
企业级量化投研平台 - 组合优化器抽象基类 (Portfolio Optimizer Base)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationConstraints:
    """组合优化约束条件"""
    max_weight: float = 0.40           # 单标的持仓权重上限 (默认 40%)
    min_weight: float = 0.00           # 单标的持仓权重下限 (默认多头 0%)
    turnover_penalty: float = 0.02     # 换手率惩罚系数 gamma (默认 0.02 抑制频繁调仓)
    max_turnover: Optional[float] = None # 单次调仓最大换手上限
    target_return: Optional[float] = None
    risk_aversion: float = 1.0         # 风险厌恶系数
    use_ledoit_wolf: bool = True       # 是否默认启用 Ledoit-Wolf 收缩协方差估计


class PortfolioOptimizer(ABC):
    """组合优化器基础接口"""

    @abstractmethod
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        """
        求解最优资产配置权重向量
        :param expected_returns: (n_assets,) 预期收益率向量
        :param cov_matrix: (n_assets, n_assets) 资产协方差矩阵
        :param current_weights: (n_assets,) 当前持仓权重
        :param constraints: 业务与风险约束
        :return: (n_assets,) 最优权重向量，和等于 1.0
        """
        pass
