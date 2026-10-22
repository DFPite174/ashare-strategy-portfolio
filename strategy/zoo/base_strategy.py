"""
企业级量化投研平台 - 策略基类与生命周期模型 (Base Strategy & Lifecycle)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class StrategyLifecycle(str, Enum):
    """策略生命周期状态机"""
    CANDIDATE = "CANDIDATE"       # 候选研发期 (初次提出假说与粗筛)
    AUDITED = "AUDITED"           # 已通过双引擎与过拟合严格核验
    INCUBATING = "INCUBATING"     # 模拟盘孵化观察期
    PRODUCTION = "PRODUCTION"     # 生产实盘执行期
    RETIRED = "RETIRED"           # Alpha 衰减退役


class StrategyMetadata(BaseModel):
    """策略元数据规范"""
    model_config = ConfigDict(frozen=True)

    strategy_id: str
    name: str
    version: str = "1.0.0"
    hypothesis_id: Optional[str] = None
    author: str = "Quant Researcher"
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    lifecycle_status: StrategyLifecycle = Field(default=StrategyLifecycle.CANDIDATE)


class BaseStrategy(ABC):
    """策略抽象基类"""

    def __init__(self, metadata: StrategyMetadata):
        self.metadata = metadata

    @abstractmethod
    def generate_target_weights(
        self,
        bars_df: pd.DataFrame,
        rebalance_interval: int = 10,
    ) -> pd.DataFrame:
        """
        基于输入行情数据与内置优化器生成全时序目标权重矩阵
        :return: DataFrame(index=trade_date, columns=symbol, values=weight 0.0~1.0)
        """
        pass
