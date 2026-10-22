"""
企业级量化投研平台 - 语义与大模型驱动因子挑选与合成器 (Track 2: Semantic & Agent Selector)
根据用户提出的策略需求或经济假说，自动在因子库中进行语义匹配，并智能生成/合成新的复合因子表达式。
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from strategy.hypotheses.models import Hypothesis, HypothesisTheme
from strategy.selection.orthogonal_selector import SelectedFactorInfo, FactorSelectionResult
from research.factors.registry import FactorRegistry, FactorMetadata
from research.factors.formula_engine import FormulaEngine


THEME_KEYWORD_MAP = {
    HypothesisTheme.MOMENTUM_TREND: ["momentum", "trend", "breakout", "ts_pct_change", "macd"],
    HypothesisTheme.VALUE_REVERSION: ["value", "reversal", "pe", "pb", "mean_reversion", "rsi"],
    HypothesisTheme.QUALITY_PROFITABILITY: ["quality", "roe", "profitability", "margin", "fundamental"],
    HypothesisTheme.LOW_VOLATILITY: ["volatility", "atr", "std", "low_vol", "bollinger"],
    HypothesisTheme.LIQUIDITY_PREMIUM: ["liquidity", "turnover", "volume", "amount"],
    HypothesisTheme.COMPOSITE_MULTI_STYLE: ["composite", "multi_factor", "alpha"],
}


class SemanticFactorSelector:
    """
    语义与 Agent 假说驱动的因子选择与自动合成器
    """

    def __init__(self, registry: Optional[FactorRegistry] = None):
        self.registry = registry or FactorRegistry()
        self.formula_engine = FormulaEngine()

    def parse_user_intent_keywords(self, prompt: str) -> List[str]:
        """
        从用户自然语言描述中提取核心量化意图关键词
        """
        keywords = []
        p = prompt.lower()
        mapping = {
            "动量": "momentum",
            "趋势": "trend",
            "价值": "value",
            "估值": "value",
            "反转": "reversal",
            "超跌": "reversal",
            "盈利": "quality",
            "质量": "quality",
            "roe": "roe",
            "低波": "volatility",
            "稳健": "low_vol",
            "成交量": "volume",
            "换手率": "turnover",
            "rsi": "rsi",
            "macd": "macd",
        }
        for cn_key, en_tag in mapping.items():
            if cn_key in p or en_tag in p:
                keywords.append(en_tag)

        # 英文词正则匹配
        words = re.findall(r"[a-zA-Z_]+", p)
        keywords.extend([w for w in words if len(w) > 2])
        return list(set(keywords))

    def select_factors_by_hypothesis(
        self,
        hypothesis: Hypothesis,
        max_factors: int = 3,
    ) -> List[FactorMetadata]:
        """
        依据结构化假说匹配最优候选因子
        """
        theme_keys = THEME_KEYWORD_MAP.get(hypothesis.theme, [])
        query_tags = set(hypothesis.preferred_factor_tags + hypothesis.tags + theme_keys)

        candidates = self.registry.list_factors()
        scored: List[Tuple[FactorMetadata, float]] = []

        for meta in candidates:
            score = 0.0
            meta_tags = set(meta.tags)
            overlap = query_tags.intersection(meta_tags)
            score += len(overlap) * 2.0

            # 描述匹配
            desc_lower = meta.description.lower()
            for q in query_tags:
                if q in desc_lower or q in meta.name.lower():
                    score += 1.0

            scored.append((meta, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [meta for meta, score in scored[:max_factors] if score > 0]

    def synthesize_new_composite_factor(
        self,
        factors: List[FactorMetadata],
        composite_name: str,
        hypothesis: Hypothesis,
    ) -> str:
        """
        依据假说与已选因子自动构建新的 AST 复合因子公式并入库
        例如: cs_rank(rsi_14) * 0.5 + cs_rank(ts_pct_change(close, 20)) * 0.5
        """
        if not factors:
            raise ValueError("No candidate factors provided to synthesize composite factor")

        n = len(factors)
        weight_each = round(1.0 / n, 2)
        terms = []

        for f in factors:
            # 若因子自带公式则嵌套，否则以截面排序包装
            formula_base = f.formula or f.name
            terms.append(f"cs_rank({formula_base}) * {weight_each}")

        composite_formula = " + ".join(terms)

        # 动态注册进 FactorRegistry
        self.registry.register(
            name=composite_name,
            formula=composite_formula,
            category="COMPOSITE",
            description=f"Auto-synthesized composite factor for hypothesis: {hypothesis.title}",
            tags=["composite", "auto_synthesized"] + hypothesis.tags,
            author="SemanticAgent",
        )

        return composite_formula

    def run_semantic_pipeline(
        self,
        hypothesis: Hypothesis,
        bars_df: pd.DataFrame,
        composite_factor_name: str = "auto_composite_alpha",
    ) -> FactorSelectionResult:
        """
        运行完整的语义驱动挑选、合成与信号生成闭环
        """
        matched_metas = self.select_factors_by_hypothesis(hypothesis, max_factors=3)
        if not matched_metas:
            # 默认兜底因子
            matched_metas = [self.registry.get_metadata("momentum_20d") or self.registry.list_factors()[0]]

        # 合成新因子并注册入库
        formula = self.synthesize_new_composite_factor(matched_metas, composite_factor_name, hypothesis)

        # 执行子因子计算并进行截面排名加权融合
        sub_signals = []
        w_per = 1.0 / len(matched_metas)
        for m in matched_metas:
            fac = self.registry.get(m.name)
            s = fac.compute(bars_df)
            if isinstance(s, pd.Series) and isinstance(s.index, pd.MultiIndex):
                wide = s.unstack(level="symbol")
            elif isinstance(s, pd.DataFrame):
                wide = s
            else:
                wide = pd.DataFrame(s)
            ranked = wide.rank(axis=1, pct=True)
            sub_signals.append(ranked * w_per)

        composite_signal_df = sum(sub_signals)

        selected_info = []
        w_per = 1.0 / len(matched_metas)
        for m in matched_metas:
            selected_info.append(SelectedFactorInfo(
                factor_name=m.name,
                icir=1.0,  # 语义先验置信度
                rank_ic_mean=0.05,
                weight=w_per,
                pairwise_correlations={},
            ))

        corr_df = pd.DataFrame(np.eye(len(matched_metas)), index=[m.name for m in matched_metas], columns=[m.name for m in matched_metas])

        return FactorSelectionResult(
            selected_factors=selected_info,
            composite_signal_df=composite_signal_df,
            correlation_matrix=corr_df,
            method="semantic_agent_synthesizer",
        )
