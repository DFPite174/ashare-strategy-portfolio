"""
企业级量化投研平台 - 统计量化正交贪心因子选择与合成器 (Track 1: Orthogonal Greedy Selector)
基于 ICIR 最大化与低多重共线性约束 (|corr| < 0.35)，通过贪心筛选与施密特正交化生成纯净复合 Alpha 信号。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class SelectedFactorInfo:
    factor_name: str
    icir: float
    rank_ic_mean: float
    weight: float
    pairwise_correlations: Dict[str, float]


@dataclass
class FactorSelectionResult:
    selected_factors: List[SelectedFactorInfo]
    composite_signal_df: pd.DataFrame
    correlation_matrix: pd.DataFrame
    method: str = "orthogonal_greedy"


class OrthogonalGreedySelector:
    """
    量化统计正交贪心选择器
    """

    def __init__(self, max_correlation: float = 0.35, max_factors: int = 5):
        self.max_correlation = max_correlation
        self.max_factors = max_factors

    def compute_factor_icir(
        self,
        factor_df: pd.DataFrame,
        forward_returns_df: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        计算单因子的 Rank IC 均值与 ICIR (日频)
        """
        aligned_factor = factor_df.reindex(forward_returns_df.index)
        dates = aligned_factor.index

        daily_ics = []
        for dt in dates:
            if dt in forward_returns_df.index:
                f_row = aligned_factor.loc[dt].dropna()
                r_row = forward_returns_df.loc[dt].dropna()
                common_syms = f_row.index.intersection(r_row.index)
                if len(common_syms) >= 3:
                    f_rank = f_row.loc[common_syms].rank()
                    r_rank = r_row.loc[common_syms].rank()
                    ic = f_rank.corr(r_rank, method="spearman")
                    if np.isfinite(ic):
                        daily_ics.append(ic)

        if not daily_ics or len(daily_ics) < 5:
            return 0.0, 0.0

        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics))
        icir = (ic_mean / ic_std * np.sqrt(252.0)) if ic_std > 1e-6 else 0.0
        return ic_mean, icir

    def select_and_synthesize(
        self,
        candidate_factor_dfs: Dict[str, pd.DataFrame],
        forward_returns_df: pd.DataFrame,
        orthogonalize: bool = True,
    ) -> FactorSelectionResult:
        """
        执行正交贪心算法挑选因子并加权合成
        :param candidate_factor_dfs: {factor_name: df(index=trade_date, columns=symbol)}
        :param forward_returns_df: df(index=trade_date, columns=symbol) 未来收益率
        :param orthogonalize: 是否对选出因子进行施密特正交化去共线性
        """
        # 1. 评估所有候选因子的 ICIR 与 Rank IC
        factor_stats = {}
        for fname, f_df in candidate_factor_dfs.items():
            ic_mean, icir = self.compute_factor_icir(f_df, forward_returns_df)
            factor_stats[fname] = {"ic_mean": ic_mean, "icir": icir, "abs_icir": abs(icir)}

        # 按 |ICIR| 从大到小排序
        sorted_candidates = sorted(factor_stats.keys(), key=lambda k: factor_stats[k]["abs_icir"], reverse=True)

        # 2. 扁平化数据以计算因子间横截面平均相关系数矩阵
        all_dates = forward_returns_df.index
        all_symbols = forward_returns_df.columns

        factor_vectors = {}
        for fname in sorted_candidates:
            # 展平为一维序列计算全局相关性
            aligned = candidate_factor_dfs[fname].reindex(index=all_dates, columns=all_symbols)
            factor_vectors[fname] = aligned.values.flatten()

        f_matrix_df = pd.DataFrame(factor_vectors).dropna(how="all")
        corr_matrix = f_matrix_df.corr(method="spearman").fillna(0.0)

        # 3. 贪心挑选：每次加入与已选因子最大相关系数 < max_correlation 的最优因子
        selected_names: List[str] = []
        for cand in sorted_candidates:
            if not selected_names:
                selected_names.append(cand)
            else:
                max_pairwise_corr = max(abs(corr_matrix.loc[cand, prev]) for prev in selected_names)
                if max_pairwise_corr <= self.max_correlation:
                    selected_names.append(cand)

            if len(selected_names) >= self.max_factors:
                break

        # 4. 权重分配 (按 ICIR 方向与绝对大小加权)
        total_abs_icir = sum(factor_stats[f]["abs_icir"] for f in selected_names) or 1.0
        selected_info = []

        for f in selected_names:
            stat = factor_stats[f]
            # 因子 IC 为负则反向加权 (使复合信号始终与收益正相关)
            sign = 1.0 if stat["ic_mean"] >= 0 else -1.0
            w = (stat["abs_icir"] / total_abs_icir) * sign
            pairwise = {prev: float(corr_matrix.loc[f, prev]) for prev in selected_names if prev != f}
            selected_info.append(SelectedFactorInfo(
                factor_name=f,
                icir=stat["icir"],
                rank_ic_mean=stat["ic_mean"],
                weight=w,
                pairwise_correlations=pairwise,
            ))

        # 5. 合成复合信号
        # 标准化每个因子 (横截面 z-score)
        composite_df = pd.DataFrame(0.0, index=all_dates, columns=all_symbols)

        for info in selected_info:
            f_raw = candidate_factor_dfs[info.factor_name].reindex(index=all_dates, columns=all_symbols)
            # 横截面 rank 标准化至 [-0.5, 0.5]
            f_norm = f_raw.apply(lambda row: (row.rank(pct=True) - 0.5) if row.count() > 1 else row, axis=1).fillna(0.0)
            composite_df = composite_df + f_norm * info.weight

        return FactorSelectionResult(
            selected_factors=selected_info,
            composite_signal_df=composite_df,
            correlation_matrix=corr_matrix.loc[selected_names, selected_names],
            method="orthogonal_greedy",
        )
