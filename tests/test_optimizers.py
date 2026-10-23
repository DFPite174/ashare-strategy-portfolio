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
    CovarianceEstimator,
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


def test_covariance_estimator_ledoit_wolf():
    np.random.seed(42)
    # 模拟 30 天 4 只股票的日收益率
    rets = np.random.normal(0.0005, 0.015, size=(30, 4))
    shrunk_cov, shrinkage = CovarianceEstimator.ledoit_wolf_shrinkage(rets)
    assert shrunk_cov.shape == (4, 4)
    assert 0.0 <= shrinkage <= 1.0
    # 校验对称性与严格正定性
    assert np.allclose(shrunk_cov, shrunk_cov.T)
    eigenvalues = np.linalg.eigvalsh(shrunk_cov)
    assert np.all(eigenvalues > 0)

    # 统一接口
    cov_out = CovarianceEstimator.estimate(rets, method="ledoit_wolf")
    assert np.allclose(cov_out, shrunk_cov)


def test_turnover_penalty_restricts_rebalancing(sample_market_data):
    exp_rets, cov = sample_market_data
    opt = MeanVarianceOptimizer()
    curr_w = np.array([0.40, 0.40, 0.10, 0.10])

    # 无换手率惩罚时
    cons_no_penalty = OptimizationConstraints(max_weight=0.50, turnover_penalty=0.0)
    w_free = opt.optimize(exp_rets, cov, current_weights=curr_w, constraints=cons_no_penalty)
    turnover_free = np.sum(np.abs(w_free - curr_w))

    # 强换手率惩罚时 (gamma = 0.20)
    cons_strong_penalty = OptimizationConstraints(max_weight=0.50, turnover_penalty=0.20)
    w_penalized = opt.optimize(exp_rets, cov, current_weights=curr_w, constraints=cons_strong_penalty)
    turnover_penalized = np.sum(np.abs(w_penalized - curr_w))

    # 强惩罚下的换手率应显著低于自由优化
    assert turnover_penalized <= turnover_free
    assert abs(np.sum(w_penalized) - 1.0) < 1e-5
