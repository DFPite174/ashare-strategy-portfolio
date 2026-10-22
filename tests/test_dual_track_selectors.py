"""
双轨因子挑选与自动合成单元测试 - 检验统计量化正交贪心 (Track 1) 与语义 Agent 合成 (Track 2)
"""

import pytest
import numpy as np
import pandas as pd

from strategy.hypotheses import get_hypothesis
from strategy.selection import OrthogonalGreedySelector, SemanticFactorSelector
from research.factors.registry import FactorRegistry


@pytest.fixture
def mock_candidate_factors_data():
    dates = pd.date_range("2025-01-01", periods=30, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = ["000001.SZ", "600519.SH", "688981.SH", "601088.SH"]
    np.random.seed(42)

    f1 = pd.DataFrame(np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols)
    # f2 与 f1 高度共线性 (相关系数 > 0.9)
    f2 = f1 * 0.95 + np.random.randn(len(dates), len(symbols)) * 0.05
    # f3 与 f1 相互独立 (相关系数 ~ 0)
    f3 = pd.DataFrame(np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols)

    # 构造未来收益率，使 f1 具备较强预测力
    forward_rets = f1.shift(-1) * 0.02 + np.random.randn(len(dates), len(symbols)) * 0.01
    forward_rets = forward_rets.dropna()

    return {"factor_alpha": f1, "factor_alpha_collinear": f2, "factor_beta": f3}, forward_rets


def test_track1_orthogonal_greedy_selector(mock_candidate_factors_data):
    factors_dict, forward_rets = mock_candidate_factors_data
    selector = OrthogonalGreedySelector(max_correlation=0.40, max_factors=2)

    res = selector.select_and_synthesize(factors_dict, forward_rets)

    selected_names = [info.factor_name for info in res.selected_factors]
    assert len(selected_names) <= 2
    # 高度共线性的两个因子不应同时被选中
    assert not ("factor_alpha" in selected_names and "factor_alpha_collinear" in selected_names)
    assert not res.composite_signal_df.empty
    assert res.method == "orthogonal_greedy"


def test_track2_semantic_agent_selector():
    dates = pd.date_range("2025-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
    bars_list = []
    for d in dates:
        bars_list.append({"trade_date": d, "symbol": "000001.SZ", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2, "volume": 100000})
        bars_list.append({"trade_date": d, "symbol": "600519.SH", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 100000})
    bars_df = pd.DataFrame(bars_list)

    registry = FactorRegistry()
    selector = SemanticFactorSelector(registry)

    # 1. 意图解析测试
    keywords = selector.parse_user_intent_keywords("我想挖掘基于中期动量和低波动特征的稳健趋势策略")
    assert "momentum" in keywords or "trend" in keywords

    # 2. 语义选择与自动合成闭环测试
    hypo = get_hypothesis("H_MOM_01")
    matched = selector.select_factors_by_hypothesis(hypo, max_factors=2)
    assert len(matched) >= 1

    res = selector.run_semantic_pipeline(hypo, bars_df, composite_factor_name="test_semantic_composite")
    assert not res.composite_signal_df.empty
    assert res.method == "semantic_agent_synthesizer"
    # 确认新合成因子成功注册到因子库
    assert registry.has_factor("test_semantic_composite")
