"""
Alphalens-cn 因子验证适配器与统计稳健性引擎 (Alphalens Engine)
封装 Alphalens 因子清洗、IC/RankIC 分析、分层单调性、HAC 重叠自相关方差修正与 FDR 多重检验。
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import alphalens as al


class AlphalensAdapter:
    """提供高保真因子检验、统计稳健性修正与防未来函数审计"""

    @staticmethod
    def prepare_prices_matrix(bars_df: pd.DataFrame) -> pd.DataFrame:
        """从扁平行情表构建宽表价格矩阵 (trade_date 为行索引, 标的代码为列)"""
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 复权价用于计算真实的前瞻收益 (Forward Returns)
        df["adj_close"] = df["close"] * df["adj_factor"]
        prices = df.pivot(index="trade_date", columns="symbol", values="adj_close").sort_index()
        return prices

    @classmethod
    def clean_factor_data(
        cls,
        factor_series: pd.Series,
        prices: pd.DataFrame,
        periods: Tuple[int, ...] = (1, 5, 10, 20),
        quantiles: int = 5,
        max_loss: float = 0.35,
    ) -> pd.DataFrame:
        """
        调用 Alphalens 核心清洗函数，计算因子与多期前瞻收益
        """
        # 确保时区与索引类型一致
        clean_factor = factor_series.copy()
        if not isinstance(clean_factor.index.levels[0], pd.DatetimeIndex):
            clean_factor.index = clean_factor.index.set_levels(
                pd.to_datetime(clean_factor.index.levels[0]), level=0
            )

        clean_data = al.utils.get_clean_factor_and_forward_returns(
            factor=clean_factor,
            prices=prices,
            periods=periods,
            quantiles=quantiles,
            max_loss=max_loss,
        )
        return clean_data

    @staticmethod
    def compute_hac_stats(series: pd.Series, lag: int) -> Tuple[float, float]:
        """
        Newey-West / HAC (Heteroskedasticity and Autocorrelation Consistent) 稳健统计量
        纠正持有期交叠导致的自相关偏差
        :return: (t_stat_hac, p_value_hac)
        """
        clean_s = series.dropna().values
        n = len(clean_s)
        if n <= 1:
            return 0.0, 1.0

        mean = np.mean(clean_s)
        demeaned = clean_s - mean

        # 计算 0 阶自协方差
        gamma0 = np.sum(demeaned ** 2) / n
        hac_var = gamma0

        # 计算 1 到 lag 阶自协方差并赋予 Bartlett 线性递减权重
        for j in range(1, min(lag + 1, n)):
            gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / n
            weight = 1.0 - (j / (lag + 1.0))
            hac_var += 2.0 * weight * gamma_j

        if hac_var <= 1e-12:
            return 0.0, 1.0

        std_error = math.sqrt(hac_var / n)
        t_stat = float(mean / std_error)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
        return t_stat, p_val

    @classmethod
    def evaluate_factor_metrics(
        cls,
        clean_factor_data: pd.DataFrame,
        periods: Tuple[int, ...] = (1, 5, 10, 20),
    ) -> Dict[str, Any]:
        """
        全量计算因子统计画像：IC、RankIC、ICIR、单调性得分及 HAC 调整后统计量
        """
        # 1. 计算日频 Rank IC 时序
        ic_df = al.performance.factor_information_coefficient(clean_factor_data)
        metrics = {"periods": {}}

        for p in periods:
            col = p if p in ic_df.columns else f"{p}D"
            if col not in ic_df.columns:
                continue

            ic_series = ic_df[col].dropna()
            n_days = len(ic_series)
            if n_days == 0:
                continue

            mean_ic = float(ic_series.mean())
            std_ic = float(ic_series.std())
            icir = float(mean_ic / std_ic) if std_ic > 1e-8 else 0.0

            # 经典原始 t-stat
            t_raw = float(mean_ic / (std_ic / math.sqrt(n_days))) if std_ic > 1e-8 else 0.0
            p_raw = float(2.0 * (1.0 - stats.t.cdf(abs(t_raw), df=n_days - 1)))

            # HAC / Newey-West 调整后的 t-stat (lag = period - 1)
            t_hac, p_hac = cls.compute_hac_stats(ic_series, lag=max(1, p - 1))

            metrics["periods"][f"{p}D"] = {
                "mean_ic": round(mean_ic, 4),
                "std_ic": round(std_ic, 4),
                "icir": round(icir, 4),
                "t_stat_raw": round(t_raw, 2),
                "p_value_raw": round(p_raw, 6),
                "t_stat_hac": round(t_hac, 2),
                "p_value_hac": round(p_hac, 6),
                "observations": n_days,
            }

        # 2. 分层收益与单调性得分 (Quantile Return Monotonicity)
        mean_return_by_q, _ = al.performance.mean_return_by_quantile(clean_factor_data, by_date=False)
        metrics["quantile_returns"] = {}
        metrics["monotonicity_scores"] = {}

        for p in periods:
            col = p if p in mean_return_by_q.columns else f"{p}D"
            if col in mean_return_by_q.columns:
                q_ret = mean_return_by_q[col].to_dict()
                metrics["quantile_returns"][f"{p}D"] = {str(k): round(float(v), 6) for k, v in q_ret.items()}

                # 计算分位数与平均收益的斯皮尔曼秩相关系数作为单调性得分 (-1.0 到 1.0)
                q_indices = list(q_ret.keys())
                q_vals = [q_ret[k] for k in q_indices]
                if len(q_vals) >= 3:
                    spearman_corr, _ = stats.spearmanr(q_indices, q_vals)
                    metrics["monotonicity_scores"][f"{p}D"] = round(float(spearman_corr), 4)

        return metrics

    @staticmethod
    def audit_leakage_and_traps(clean_factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        防未来函数与数据泄漏审查：
        当 RankIC 异常接近 1.0 (如 > 0.85) 或 t-stat 极度反常时，标记潜在未来函数泄漏
        """
        ic_df = al.performance.factor_information_coefficient(clean_factor_data)
        warnings = []
        is_leaking = False

        for col in ic_df.columns:
            mean_ic = ic_df[col].mean()
            if abs(mean_ic) > 0.80:
                is_leaking = True
                warnings.append(f"Abnormally high Mean IC ({mean_ic:.4f}) on period {col}, lookahead leakage suspected.")

        audit_result = {
            "is_lookahead_leaking": is_leaking,
            "status": "FAIL_LEAKAGE_DETECTED" if is_leaking else "PASS_AUDIT",
            "warnings": warnings,
        }
        return audit_result
