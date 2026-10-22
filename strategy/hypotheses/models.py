"""
企业级量化投研平台 - 策略假设契约模型 (Strategy Hypothesis Models)
定义经济学假说结构，规范“假说驱动”而非“无脑数据挖掘”的严肃量化研发规范。
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class HypothesisTheme(str, Enum):
    """假说所属经济学主题风格"""
    MOMENTUM_TREND = "MOMENTUM_TREND"                 # 动量趋势追踪
    VALUE_REVERSION = "VALUE_REVERSION"               # 价值洼地与均值回归
    QUALITY_PROFITABILITY = "QUALITY_PROFITABILITY"   # 高盈利质量与护城河
    LOW_VOLATILITY = "LOW_VOLATILITY"                 # 低波动异常溢价
    LIQUIDITY_PREMIUM = "LIQUIDITY_PREMIUM"           # 流动性与换手率溢价
    COMPOSITE_MULTI_STYLE = "COMPOSITE_MULTI_STYLE"   # 复合多风格对冲


class MarketRegime(str, Enum):
    """假说目标适用的市场宏观状态"""
    BULL = "BULL"               # 趋势牛市
    BEAR = "BEAR"               # 熊市防守
    OSCILLATION = "OSCILLATION" # 震荡结构市
    ALL_WEATHER = "ALL_WEATHER" # 全天候自适应


class Hypothesis(BaseModel):
    """结构化量化经济假说契约"""
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = Field(..., description="唯一假说标识符")
    title: str = Field(..., description="假说名称")
    economic_rationale: str = Field(..., description="金融经济逻辑论证与因果机制说明")
    theme: HypothesisTheme = Field(..., description="主题风格")
    target_regime: MarketRegime = Field(default=MarketRegime.ALL_WEATHER, description="适用市场环境")
    expected_holding_days: int = Field(default=20, ge=1, description="预期调仓持有期 (天)")
    preferred_factor_tags: List[str] = Field(default_factory=list, description="建议关联的因子标签")
    tags: List[str] = Field(default_factory=list, description="分类检索标签")
    author: str = Field(default="Quant Researcher")
