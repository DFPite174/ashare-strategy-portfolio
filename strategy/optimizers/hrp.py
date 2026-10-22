"""
企业级量化投研平台 - 层次风险平价优化器 (Hierarchical Risk Parity / HRP)
基于 Marcos López de Prado (AFML) 机器学习聚类树形配置理论:
1. 层次聚类 (Tree Clustering)
2. 拟对角化重排 (Quasi-Diagonalization)
3. 递归二分权重分配 (Recursive Bisection)
完全规避协方差矩阵求逆 (No Matrix Inversion)，对奇异性与病态条件数具备极强鲁棒性。
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from strategy.optimizers.base import PortfolioOptimizer, OptimizationConstraints


class HierarchicalRiskParityOptimizer(PortfolioOptimizer):
    """层次风险平价 (HRP) 优化器"""

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """按聚类层级重排标的索引 (Quasi-Diagonalization)"""
        link = link.astype(int)
        sort_ix = pd_series = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]

        while max(sort_ix) >= num_items:
            sort_ix_new = []
            for item in sort_ix:
                if item >= num_items:
                    orig_ix = item - num_items
                    sort_ix_new.append(link[orig_ix, 0])
                    sort_ix_new.append(link[orig_ix, 1])
                else:
                    sort_ix_new.append(item)
            sort_ix = sort_ix_new
        return sort_ix

    def _get_cluster_var(self, cov: np.ndarray, c_items: List[int]) -> float:
        """计算子聚类方差"""
        sub_cov = cov[np.ix_(c_items, c_items)]
        # 子聚类内部采用逆方差加权
        diag = np.diag(sub_cov)
        inv_diag = 1.0 / np.maximum(diag, 1e-8)
        w = inv_diag / np.sum(inv_diag)
        c_var = float(w.T @ sub_cov @ w)
        return c_var

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> np.ndarray:
        cons = constraints or OptimizationConstraints()
        n = len(expected_returns)
        if n <= 1:
            return np.ones(n)

        # 协方差转相关系数
        std = np.sqrt(np.diag(cov_matrix))
        std[std < 1e-8] = 1e-8
        corr = cov_matrix / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)

        # 1. 距离矩阵: d_ij = sqrt( 0.5 * (1 - rho_ij) )
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # 2. 层次聚类与拟对角化
        condensed_dist = squareform(dist, checks=False)
        link = sch.linkage(condensed_dist, method="single")
        sort_ix = self._get_quasi_diag(link)

        # 3. 递归二分分配权重 (Recursive Bisection)
        weights = np.ones(n)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters_new = []
            for c in clusters:
                if len(c) > 1:
                    mid = len(c) // 2
                    c1 = c[:mid]
                    c2 = c[mid:]

                    var1 = self._get_cluster_var(cov_matrix, c1)
                    var2 = self._get_cluster_var(cov_matrix, c2)

                    # 分配因子 alpha1 = 1 - var1 / (var1 + var2)
                    tot_var = var1 + var2
                    alpha1 = 1.0 - (var1 / tot_var) if tot_var > 1e-12 else 0.5
                    alpha2 = 1.0 - alpha1

                    for idx in c1:
                        weights[idx] *= alpha1
                    for idx in c2:
                        weights[idx] *= alpha2

                    clusters_new.append(c1)
                    clusters_new.append(c2)
            clusters = clusters_new

        # 4. 遵守单票最大持仓约束 (迭代削峰填谷分配)
        for _ in range(25):
            if np.all(weights <= cons.max_weight + 1e-5):
                break
            excess = np.sum(np.maximum(weights - cons.max_weight, 0.0))
            weights = np.minimum(weights, cons.max_weight)
            uncapped_mask = weights < cons.max_weight - 1e-6
            if not np.any(uncapped_mask) or np.sum(weights[uncapped_mask]) <= 1e-12:
                break
            weights[uncapped_mask] += excess * (weights[uncapped_mask] / np.sum(weights[uncapped_mask]))

        weights = np.clip(weights, cons.min_weight, cons.max_weight)
        s = np.sum(weights)
        return weights / s if s > 0 else np.ones(n) / n
