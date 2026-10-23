"""
企业级量化投研平台 - 协方差估计器与降噪模块 (Covariance Estimator & Shrinkage)
提供 Ledoit-Wolf 收缩估计 (Shrinkage Covariance) 与经验样本协方差，
压缩小样本估计误差与极端特征值，增强二次规划与风险平价权重的样本外稳定性。
"""

from __future__ import annotations
from typing import Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf


class CovarianceEstimator:
    """资产收益率协方差矩阵稳健估计器"""

    @staticmethod
    def ledoit_wolf_shrinkage(
        returns: Union[pd.DataFrame, np.ndarray],
        assume_centered: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf 渐近最优线性收缩协方差估计
        :param returns: (n_samples, n_assets) 收益率矩阵
        :param assume_centered: 是否假设收益率均值为 0
        :return: (shrunk_cov, shrinkage_coefficient)
        """
        arr = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D returns array, got shape {arr.shape}")

        n_samples, n_assets = arr.shape
        if n_assets == 0:
            return np.empty((0, 0)), 0.0

        if n_samples < 3 or n_assets == 1:
            # 样本极少时回退为带微小正则化的经验协方差
            raw_cov = np.cov(arr, rowvar=False)
            if np.isscalar(raw_cov):
                raw_cov = np.array([[float(raw_cov)]])
            return raw_cov + np.eye(n_assets) * 1e-6, 0.0

        try:
            shrunk_cov, shrinkage = ledoit_wolf(arr, assume_centered=assume_centered)
            # 确保对角元非负与半正定数值稳定性
            shrunk_cov = (shrunk_cov + shrunk_cov.T) / 2.0
            shrunk_cov += np.eye(n_assets) * 1e-7
            return shrunk_cov, float(shrinkage)
        except Exception:
            raw_cov = np.cov(arr, rowvar=False) + np.eye(n_assets) * 1e-6
            return raw_cov, 0.0

    @staticmethod
    def sample_covariance(
        returns: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """标准样本协方差矩阵 (带半正定数值微扰)"""
        arr = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
        n_assets = arr.shape[1] if arr.ndim == 2 else 1
        if arr.ndim != 2 or arr.shape[0] < 2:
            return np.eye(n_assets) * 1e-4

        cov = np.cov(arr, rowvar=False)
        if np.isscalar(cov):
            cov = np.array([[float(cov)]])
        cov = (cov + cov.T) / 2.0
        cov += np.eye(n_assets) * 1e-6
        return cov

    @classmethod
    def estimate(
        cls,
        returns: Union[pd.DataFrame, np.ndarray],
        method: Literal["ledoit_wolf", "sample"] = "ledoit_wolf",
    ) -> np.ndarray:
        """
        统一协方差估计算法入口
        """
        if method == "ledoit_wolf":
            cov, _ = cls.ledoit_wolf_shrinkage(returns)
            return cov
        elif method == "sample":
            return cls.sample_covariance(returns)
        else:
            raise ValueError(f"Unsupported covariance estimation method: {method}")
