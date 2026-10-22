"""
策略假设单元测试
"""

import pytest
from strategy.hypotheses import Hypothesis, HypothesisTheme, MarketRegime, get_hypothesis, STANDARD_HYPOTHESES


def test_hypothesis_immutability():
    h = Hypothesis(
        hypothesis_id="TEST_01",
        title="测试动量假说",
        economic_rationale="投资者反应不足导致中期动量溢价持续存在",
        theme=HypothesisTheme.MOMENTUM_TREND,
        target_regime=MarketRegime.BULL,
        expected_holding_days=10,
        preferred_factor_tags=["momentum", "trend"],
    )
    assert h.hypothesis_id == "TEST_01"
    with pytest.raises(Exception):
        h.title = "已修改"  # 契约不可变


def test_standard_hypotheses_catalog():
    assert len(STANDARD_HYPOTHESES) >= 3
    h_mom = get_hypothesis("H_MOM_01")
    assert h_mom.theme == HypothesisTheme.MOMENTUM_TREND
    assert h_mom.expected_holding_days == 10

    with pytest.raises(KeyError):
        get_hypothesis("NON_EXISTING")
