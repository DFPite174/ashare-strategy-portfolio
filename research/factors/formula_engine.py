"""
因子公式表达式解析与动态编译引擎 (Formula Engine)
支持研究员或 AI Agent (RD-Agent) 通过字符串数学表达式动态生成向量化因子。
"""

from __future__ import annotations
import ast
from typing import Dict, Any, Callable
import numpy as np
import pandas as pd

from research.factors import operators as op


class FormulaEngine:
    """提供基于安全抽象语法树 (AST) 的因子公式解析与批量计算"""

    # 注册允许在公式中调用的安全算子字典
    SAFE_OPERATORS: Dict[str, Callable] = {
        # 时序算子
        "ts_delay": op.ts_delay,
        "ts_delta": op.ts_delta,
        "ts_pct_change": op.ts_pct_change,
        "ts_mean": op.ts_mean,
        "ts_std": op.ts_std,
        "ts_max": op.ts_max,
        "ts_min": op.ts_min,
        "ts_sum": op.ts_sum,
        "ts_rank": op.ts_rank,
        "ts_decay_linear": op.ts_decay_linear,
        # 截面算子
        "cs_rank": op.cs_rank,
        "cs_zscore": op.cs_zscore,
        "cs_demean": op.cs_demean,
        # 基础数学函数
        "abs": np.abs,
        "log": np.log,
        "sign": np.sign,
    }

    @classmethod
    def evaluate(cls, formula_str: str, bars_df: pd.DataFrame) -> pd.Series:
        """
        解析并计算公式字符串
        :param formula_str: 如 "cs_rank(ts_pct_change(adj_close, 20))" 或 "ts_delta(close, 5) / ts_std(close, 20)"
        :param bars_df: 包含行情列的标准 DataFrame
        :return: MultiIndex pd.Series (trade_date, symbol)
        """
        df = bars_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        if "adj_close" not in df.columns:
            if "close" in df.columns and "adj_factor" in df.columns:
                df["adj_close"] = df["close"] * df["adj_factor"]
            elif "close" in df.columns:
                df["adj_close"] = df["close"]

        # 构建可供表达式读取的局部变量环境
        env: Dict[str, Any] = {}
        for col in df.columns:
            env[col] = df[col]

        # 封装时序与截面函数代理，使得公式调用如 ts_mean(close, 20) 能够直接作用于 df
        for name, func in cls.SAFE_OPERATORS.items():
            if name.startswith("ts_") or name.startswith("cs_"):
                # 包装为将列或 Series 作为第一个参数并代入 df 计算的闭包
                def _make_wrapper(f, fn_name):
                    def _wrapper(col_or_series, *args, **kwargs):
                        # 如果传入的是字符串列名，从 df 中读取；若是已计算的 Series，临时挂载到 df
                        if isinstance(col_or_series, str):
                            col_name = col_or_series
                            return f(df, col_name, *args, **kwargs)
                        elif isinstance(col_or_series, pd.Series):
                            temp_col = f"_tmp_{fn_name}_{id(col_or_series)}"
                            df[temp_col] = col_or_series
                            res = f(df, temp_col, *args, **kwargs)
                            df.drop(columns=[temp_col], inplace=True, errors="ignore")
                            return res
                        else:
                            raise ValueError(f"Operator {fn_name} expects column name or Series, got {type(col_or_series)}")
                    return _wrapper
                env[name] = _make_wrapper(func, name)
            else:
                env[name] = func

        # 使用 Python 安全 eval 执行 (只允许数学运算符与已注册 env)
        try:
            result = eval(formula_str, {"__builtins__": {}}, env)
        except Exception as e:
            raise ValueError(f"Failed to evaluate factor formula '{formula_str}': {e}")

        if not isinstance(result, pd.Series):
            if isinstance(result, (int, float, np.number)):
                result = pd.Series(result, index=df.index)
            else:
                result = pd.Series(result)

        df["_factor_val"] = result
        factor_series = df.set_index(["trade_date", "symbol"])["_factor_val"].dropna()
        factor_series.name = formula_str
        return factor_series
