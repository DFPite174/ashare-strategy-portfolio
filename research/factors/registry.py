"""
企业级因子库管理与注册中台 (Factor Registry & Zoo)
支持因子集中注册、元数据检索、标签语义检索 (为后续策略模块自动挑因子预留接口) 以及第三方因子库导入。
"""

from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union, Callable, Any
import pandas as pd

from research.factors.base import BaseFactor, FactorDefinition
from research.factors.formula_engine import FormulaEngine


@dataclass
class FactorMetadata:
    """因子元数据条目"""
    name: str
    category: str
    description: str
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    version: str = "1.0.0"
    formula: Optional[str] = None
    factor_class: Optional[Type[BaseFactor]] = None


class FactorRegistry:
    """企业级因子资产注册中台 (Factor Zoo)"""

    _registry: Dict[str, FactorMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factor_class: Optional[Type[BaseFactor]] = None,
        formula: Optional[str] = None,
        category: str = "CUSTOM",
        tags: Optional[List[str]] = None,
        description: str = "",
        author: str = "researcher",
        version: str = "1.0.0",
    ) -> None:
        """
        向因子库注册一个新因子 (支持 Python 类或公式字符串)
        """
        if not factor_class and not formula:
            raise ValueError(f"Factor '{name}' must provide either a factor_class or a formula string.")

        entry = FactorMetadata(
            name=name,
            category=category.upper(),
            description=description,
            tags=[t.lower() for t in (tags or [])],
            author=author,
            version=version,
            formula=formula,
            factor_class=factor_class,
        )
        cls._registry[name] = entry

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseFactor:
        """从因子库中实例化获取指定因子"""
        if name not in cls._registry:
            raise KeyError(f"Factor '{name}' not found in registry. Available: {list(cls._registry.keys())}")

        meta = cls._registry[name]
        if meta.factor_class:
            return meta.factor_class(**kwargs)
        elif meta.formula:
            # 动态公式因子包装类
            formula_str = meta.formula
            class _FormulaFactor(BaseFactor):
                def __init__(self):
                    super().__init__(FactorDefinition(
                        name=meta.name,
                        category=meta.category,
                        description=meta.description,
                        lookback_days=20,
                    ))
                def compute(self, bars_df: pd.DataFrame, fundamentals_df: Optional[pd.DataFrame] = None) -> pd.Series:
                    res = FormulaEngine.evaluate(formula_str, bars_df)
                    res.name = self.definition.name
                    return res
            return _FormulaFactor()
        raise RuntimeError(f"Corrupted metadata for factor '{name}'")

    @classmethod
    def has_factor(cls, name: str) -> bool:
        """检查因子是否已注册"""
        return name in cls._registry

    @classmethod
    def get_metadata(cls, name: str) -> Optional[FactorMetadata]:
        """获取指定因子的元数据"""
        return cls._registry.get(name)

    @classmethod
    def list_factors(
        cls,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[FactorMetadata]:
        """按类别或标签筛选因子"""
        results = list(cls._registry.values())
        if category:
            cat_upper = category.upper()
            results = [r for r in results if r.category == cat_upper]
        if tags:
            tag_set = {t.lower() for t in tags}
            results = [r for r in results if tag_set.intersection(r.tags)]
        return results

    @classmethod
    def search(cls, query: str) -> List[FactorMetadata]:
        """
        模糊关键词与语义搜索 (服务于未来策略模块自动挑因子的交互需求)
        """
        q = query.lower()
        matched = []
        for meta in cls._registry.values():
            if (
                q in meta.name.lower()
                or q in meta.description.lower()
                or any(q in t for t in meta.tags)
                or (meta.formula and q in meta.formula.lower())
            ):
                matched.append(meta)
        return matched

    @classmethod
    def import_library(cls, library_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        批量导入外部因子库配置 (如 YAML/JSON 定义的第三方开源库)
        """
        for name, spec in library_dict.items():
            cls.register(
                name=name,
                factor_class=spec.get("factor_class"),
                formula=spec.get("formula"),
                category=spec.get("category", "EXTERNAL"),
                tags=spec.get("tags", []),
                description=spec.get("description", ""),
                author=spec.get("author", "external_import"),
            )

    @classmethod
    def register_builtins(cls) -> None:
        """自动注册内置的所有经典技术面、Alpha 101 与基本面因子"""
        from research.factors.base import (
            MomentumFactor,
            MeanReversionFactor,
            ROEValueFactor,
            RandomNoiseFactor,
        )
        from research.factors.library import (
            RSIFactor,
            MACDDiffFactor,
            BollingerBandsPctBFactor,
            ATRVolatilityFactor,
            Alpha006Factor,
            Alpha012Factor,
        )

        cls.register(
            name="momentum_20d",
            factor_class=MomentumFactor,
            category="MOMENTUM",
            tags=["momentum", "trend", "price", "classic"],
            description="20-day price momentum",
        )
        cls.register(
            name="reversal_5d",
            factor_class=MeanReversionFactor,
            category="REVERSAL",
            tags=["reversal", "short_term", "contrarian"],
            description="5-day short term mean reversion",
        )
        cls.register(
            name="rsi_14",
            factor_class=RSIFactor,
            category="TECHNICAL",
            tags=["rsi", "oscillator", "overbought_oversold", "technical"],
            description="14-day Relative Strength Index (RSI)",
        )
        cls.register(
            name="macd_dif",
            factor_class=MACDDiffFactor,
            category="TECHNICAL",
            tags=["macd", "trend", "moving_average", "technical"],
            description="MACD DIF line (12, 26)",
        )
        cls.register(
            name="boll_pct_b",
            factor_class=BollingerBandsPctBFactor,
            category="VOLATILITY",
            tags=["bollinger", "volatility", "bands", "reversion"],
            description="Bollinger Bands %B indicator",
        )
        cls.register(
            name="atr_volatility",
            factor_class=ATRVolatilityFactor,
            category="VOLATILITY",
            tags=["atr", "volatility", "risk", "low_vol"],
            description="Normalized Average True Range volatility",
        )
        cls.register(
            name="wq_alpha_006",
            factor_class=Alpha006Factor,
            category="ALPHA101",
            tags=["alpha101", "worldquant", "correlation", "volume_price"],
            description="WorldQuant Alpha #6: -1 * corr(open, volume, 10)",
        )
        cls.register(
            name="wq_alpha_012",
            factor_class=Alpha012Factor,
            category="ALPHA101",
            tags=["alpha101", "worldquant", "volume", "price_change"],
            description="WorldQuant Alpha #12: sign(delta(volume, 1)) * (-1 * delta(close, 1))",
        )
        cls.register(
            name="pit_roe",
            factor_class=ROEValueFactor,
            category="QUALITY",
            tags=["fundamental", "quality", "roe", "pit"],
            description="Point-In-Time Return on Equity (ROE)",
        )
        cls.register(
            name="random_noise",
            factor_class=RandomNoiseFactor,
            category="NOISE",
            tags=["noise", "baseline", "control"],
            description="Gaussian white noise factor for FDR control baseline",
        )


# 初始化自动加载内置因子库
FactorRegistry.register_builtins()
