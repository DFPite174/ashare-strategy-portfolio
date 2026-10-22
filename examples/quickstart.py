"""
ashare-strategy-portfolio 快速入门示例 (Quickstart Demo)
演示：
1. 经济学假说选择 (Hypothesis Definition)
2. 双轨因子挑选 (Dual-track Factor Selection)
3. 四大组合优化器求解权重对比 (Mean-Variance, Risk Parity, HRP, Equal Weight)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from strategy.hypotheses import get_hypothesis, STANDARD_HYPOTHESES
from strategy.selection import OrthogonalGreedySelector, SemanticFactorSelector
from strategy.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HierarchicalRiskParityOptimizer,
    OptimizationConstraints,
)
from research.factors.registry import FactorRegistry


def main():
    print("=" * 70)
    print("  ashare-strategy-portfolio: 策略与组合优化器端到端演示")
    print("=" * 70)

    # 1. 结构化经济假说
    hypo = get_hypothesis("H_MOM_01")
    print(f"\n[1] 载入结构化假说: {hypo.title} (ID: {hypo.hypothesis_id})")
    print(f"    经济学逻辑: {hypo.economic_rationale[:50]}...")
    print(f"    标签集合: {hypo.tags}")

    # 2. 模拟构建 4 支股票的收益率与协方差矩阵
    np.random.seed(42)
    assets = ["000001.SZ", "600519.SH", "601088.SH", "688981.SH"]
    n = len(assets)
    returns = np.array([0.15, 0.12, 0.08, 0.22])  # 预期年化收益
    volatilities = np.array([0.25, 0.20, 0.15, 0.35])
    corr = np.array([
        [1.0, 0.3, 0.1, 0.4],
        [0.3, 1.0, 0.2, 0.25],
        [0.1, 0.2, 1.0, 0.05],
        [0.4, 0.25, 0.05, 1.0],
    ])
    cov = np.outer(volatilities, volatilities) * corr

    cons = OptimizationConstraints(min_weight=0.0, max_weight=0.50, risk_aversion=2.0)

    print("\n[2] 组合优化器求解目标权重配置对比 (单票上限 50%):")
    # Equal Weight
    ew = EqualWeightOptimizer().optimize(returns, cov, constraints=cons)
    # Mean-Variance
    mv = MeanVarianceOptimizer().optimize(returns, cov, constraints=cons)
    # Risk Parity (ERC)
    rp = RiskParityOptimizer().optimize(returns, cov, constraints=cons)
    # Hierarchical Risk Parity (HRP)
    hrp = HierarchicalRiskParityOptimizer().optimize(returns, cov, constraints=cons)

    comparison_df = pd.DataFrame({
        "Equal Weight": ew,
        "Mean-Variance": mv,
        "Risk Parity": rp,
        "HRP": hrp,
    }, index=assets).round(4)

    print(comparison_df.to_string())

    print("\n[3] 因子库与语义挑选测试:")
    registry = FactorRegistry()
    selector = SemanticFactorSelector(registry)
    matched = selector.select_factors_by_hypothesis(hypo, max_factors=3)
    print(f"    基于假说匹配到的候选因子: {[m.name for m in matched]}")
    print("\n演示完成！全部组件正常运行。")


if __name__ == "__main__":
    main()
