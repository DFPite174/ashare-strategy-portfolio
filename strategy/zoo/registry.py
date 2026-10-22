"""
企业级量化投研平台 - 策略注册库与版本管理 (Strategy Registry)
"""

from __future__ import annotations
from typing import Dict, List, Optional
from strategy.zoo.base_strategy import BaseStrategy, StrategyMetadata, StrategyLifecycle


class StrategyRegistry:
    """策略库注册与生命周期管理中枢"""

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}

    def register_strategy(self, strategy: BaseStrategy) -> None:
        sid = strategy.metadata.strategy_id
        if sid in self._strategies:
            raise ValueError(f"Strategy {sid} already registered")
        self._strategies[sid] = strategy

    def get_strategy(self, strategy_id: str) -> BaseStrategy:
        if strategy_id not in self._strategies:
            raise KeyError(f"Strategy {strategy_id} not found in registry")
        return self._strategies[strategy_id]

    def list_strategies(self, status: Optional[StrategyLifecycle] = None) -> List[StrategyMetadata]:
        results = [s.metadata for s in self._strategies.values()]
        if status is not None:
            results = [m for m in results if m.lifecycle_status == status]
        return results

    def filter_by_tag(self, tag: str) -> List[BaseStrategy]:
        return [s for s in self._strategies.values() if tag in s.metadata.tags]
