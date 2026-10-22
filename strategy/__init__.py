"""
企业级量化投研平台 - 策略与组合中枢 (Strategy & Portfolio Optimization Module)
包含：经济假说生成、双轨因子挑选与自动合成、组合优化器（HRP/风险平价/均值方差）、策略库管理。
"""

from strategy.hypotheses import Hypothesis, HypothesisTheme, MarketRegime, STANDARD_HYPOTHESES, get_hypothesis
from strategy.selection import (
    OrthogonalGreedySelector,
    SemanticFactorSelector,
    SelectedFactorInfo,
    FactorSelectionResult,
)
from strategy.optimizers import (
    PortfolioOptimizer,
    OptimizationConstraints,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HierarchicalRiskParityOptimizer,
    EqualWeightOptimizer,
    ScoreWeightedOptimizer,
)
from strategy.zoo import (
    BaseStrategy,
    StrategyMetadata,
    StrategyLifecycle,
    MultiFactorStrategy,
    StrategyRegistry,
)

__all__ = [
    "Hypothesis",
    "HypothesisTheme",
    "MarketRegime",
    "STANDARD_HYPOTHESES",
    "get_hypothesis",
    "OrthogonalGreedySelector",
    "SemanticFactorSelector",
    "SelectedFactorInfo",
    "FactorSelectionResult",
    "PortfolioOptimizer",
    "OptimizationConstraints",
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "HierarchicalRiskParityOptimizer",
    "EqualWeightOptimizer",
    "ScoreWeightedOptimizer",
    "BaseStrategy",
    "StrategyMetadata",
    "StrategyLifecycle",
    "MultiFactorStrategy",
    "StrategyRegistry",
]
