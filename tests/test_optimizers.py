"""
组合优化器单元测试 - 覆盖 cvxpy 均值方差、scipy 风险平价、HRP 层次风险平价与基准分配器
"""

import pytest
import numpy as np

from strategy.optimizers import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    HierarchicalRiskParityOptimizer,
    EqualWeightOptimizer,
    ScoreWeightedOptimizer,
    OptimizationConstraints,
)


@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    n_assets = 4
    exp_returns = np.array([0.15, 0.10, 0.12, 0.08])
    # 构建正定协方差矩阵
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = A.T @ A * 0.05 + np.eye(n_assets) * 0.02
    return exp_returns, cov_matrix


def test_equal_weight_optimizer(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = EqualWeightOptimizer()
    w = opt.optimize(exp_rets, cov)
    assert len(w) == 4
    assert np.allclose(w, [0.25, 0.25, 0.25, 0.25])
    assert abs(np.sum(w) - 1.0) < 1e-6


def test_score_weighted_optimizer(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = ScoreWeightedOptimizer()
    cons = OptimizationConstraints(max_weight=0.50)
    w = opt.optimize(exp_rets, cov, constraints=cons)
    assert len(w) == 4
    assert abs(np.sum(w) - 1.0) < 1e-6
    assert np.all(w >= 0.0)
    assert np.all(w <= 0.50 + 1e-6)
    # 预期收益最高的资产权重应最高
    assert np.argmax(w) == np.argmax(exp_rets)


def test_mean_variance_cvxpy_optimizer(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = MeanVarianceOptimizer()
    cons = OptimizationConstraints(max_weight=0.45, min_weight=0.05, turnover_penalty=0.001)
    curr_w = np.array([0.25, 0.25, 0.25, 0.25])

    w = opt.optimize(exp_rets, cov, current_weights=curr_w, constraints=cons)
    assert len(w) == 4
    assert abs(np.sum(w) - 1.0) < 1e-5
    assert np.all(w >= 0.05 - 1e-5)
    assert np.all(w <= 0.45 + 1e-5)


def test_risk_parity_scipy_optimizer(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = RiskParityOptimizer()
    cons = OptimizationConstraints(max_weight=0.60, min_weight=0.05)

    w = opt.optimize(exp_rets, cov, constraints=cons)
    assert len(w) == 4
    assert abs(np.sum(w) - 1.0) < 1e-5
    assert np.all(w >= 0.05 - 1e-5)
    assert np.all(w <= 0.60 + 1e-5)

    # 验证各资产风险贡献趋近均衡
    port_std = np.sqrt(w.T @ cov @ w)
    rc = w * (cov @ w) / port_std
    # 相对极差应很小
    assert (np.max(rc) - np.min(rc)) / np.mean(rc) < 0.25


def test_hrp_optimizer(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = HierarchicalRiskParityOptimizer()
    cons = OptimizationConstraints(max_weight=0.50, min_weight=0.0)

    w = opt.optimize(exp_rets, cov, constraints=cons)
    assert len(w) == 4
    assert abs(np.sum(w) - 1.0) < 1e-5
    assert np.all(w >= 0.0)
    assert np.all(w <= 0.50 + 1e-5)
