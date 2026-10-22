"""
企业级量化投研平台 - 典型量化经济假说模板库 (Hypothesis Templates)
"""

from strategy.hypotheses.models import Hypothesis, HypothesisTheme, MarketRegime


STANDARD_HYPOTHESES = [
    Hypothesis(
        hypothesis_id="H_MOM_01",
        title="中期动量趋势强化假说",
        economic_rationale="A 股投资者存在注意力有限与处置效应，业绩稳健向好股票在中期（20~60日）呈现持续正反馈动量溢价。",
        theme=HypothesisTheme.MOMENTUM_TREND,
        target_regime=MarketRegime.BULL,
        expected_holding_days=10,
        preferred_factor_tags=["momentum", "trend", "technical"],
        tags=["momentum", "trend", "alpha"],
    ),
    Hypothesis(
        hypothesis_id="H_VAL_QUALITY_02",
        title="高盈利质量蓝筹估值修复假说",
        economic_rationale="高净资产收益率（ROE）且估值合理的行业龙头在震荡和下行周期具备高安全边际与稳健股息支撑。",
        theme=HypothesisTheme.QUALITY_PROFITABILITY,
        target_regime=MarketRegime.OSCILLATION,
        expected_holding_days=20,
        preferred_factor_tags=["fundamental", "profitability", "value", "roe"],
        tags=["quality", "value", "defensive"],
    ),
    Hypothesis(
        hypothesis_id="H_LOW_VOL_03",
        title="低特异质波动率防御溢价假说",
        economic_rationale="散户彩票偏好导致高波动投机股被普遍高估，低波动股票在长期风险调整后收益显著超越市场基准。",
        theme=HypothesisTheme.LOW_VOLATILITY,
        target_regime=MarketRegime.BEAR,
        expected_holding_days=20,
        preferred_factor_tags=["volatility", "risk", "technical"],
        tags=["low_vol", "defensive", "smart_beta"],
    ),
]


def get_hypothesis(hypo_id: str) -> Hypothesis:
    for h in STANDARD_HYPOTHESES:
        if h.hypothesis_id == hypo_id:
            return h
    raise KeyError(f"Hypothesis {hypo_id} not found in catalog")
