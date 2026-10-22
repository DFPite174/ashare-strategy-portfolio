"""
端到端因子研发与有效性批量评估流水线 (Factor Research Pipeline)
实现：因子库检索 ➔ 特征批量生成 ➔ Alphalens-cn 效能评估 ➔ HAC 纠偏与防未来函数审查 ➔ 产出全量对比看板。
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from data.pit.snapshot_manager import SnapshotManager
except ImportError:
    SnapshotManager = None
from research.factors.registry import FactorRegistry
from research.factors.alphalens_adapter import AlphalensAdapter
from research.factors.reporter import FactorReporter


class FactorPipeline:
    """提供一键式因子批处理评估与全景画像生成"""

    @classmethod
    def run_batch_evaluation(
        cls,
        snapshot_id: str,
        factor_names: List[str],
        output_dir: Optional[Path] = None,
        periods: tuple[int, ...] = (1, 5, 10, 20),
        quantiles: int = 5,
    ) -> Dict[str, Any]:
        """
        针对指定快照执行多个因子的自动化并行/批量评估
        """
        mgr = SnapshotManager()
        snapshot_data = mgr.load_snapshot(snapshot_id)
        bars_df = snapshot_data["bars"]
        fina_df = snapshot_data.get("fundamentals")
        prices = AlphalensAdapter.prepare_prices_matrix(bars_df)

        results = {}
        summary_rows = []

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        for name in factor_names:
            try:
                factor_obj = FactorRegistry.get(name)
                series = factor_obj.compute(bars_df, fina_df)
                if series.empty:
                    continue

                clean_data = AlphalensAdapter.clean_factor_data(
                    factor_series=series,
                    prices=prices,
                    periods=periods,
                    quantiles=quantiles,
                    max_loss=0.40,
                )
                if clean_data.empty:
                    continue

                # 计算统计画像
                metrics = AlphalensAdapter.evaluate_factor_metrics(clean_data, periods=periods)
                # 审查防未来函数
                audit = AlphalensAdapter.audit_leakage_and_traps(clean_data)

                # 生成单因子报告
                sub_dir = (out_path / name) if out_path else None
                report = FactorReporter.generate_report(factor_obj.definition, metrics, audit, output_dir=sub_dir)
                results[name] = report

                # 收集汇总行指标 (以 5D 观察期为准)
                p5 = metrics.get("periods", {}).get("5D", {})
                m5 = metrics.get("monotonicity_scores", {}).get("5D", 0.0)
                summary_rows.append({
                    "factor_name": name,
                    "category": factor_obj.definition.category,
                    "mean_ic_5d": p5.get("mean_ic", 0.0),
                    "icir_5d": p5.get("icir", 0.0),
                    "t_stat_hac_5d": p5.get("t_stat_hac", 0.0),
                    "p_val_hac_5d": p5.get("p_value_hac", 1.0),
                    "monotonicity_5d": m5,
                    "audit_status": audit.get("status", "UNKNOWN"),
                })
            except Exception as e:
                results[name] = {"error": str(e)}

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values("icir_5d", ascending=False).reset_index(drop=True)

        # 固化批量汇总报告
        if out_path and not summary_df.empty:
            summary_df.to_parquet(out_path / "batch_factor_summary.parquet", index=False)
            
            # 生成 Markdown 看板
            lines = [
                "# 批量因子全景效能排行榜 (Batch Factor Leaderboard)",
                f"- **快照 ID**: `{snapshot_id}`",
                f"- **评估因子总数**: {len(factor_names)}",
                "",
                "| 排名 | 因子名称 | 类别 | Mean IC (5D) | RankIC IR (5D) | HAC t-stat (5D) | 单调性得分 | 审查状态 |",
                "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
            for idx, r in summary_df.iterrows():
                lines.append(
                    f"| {idx+1} | `{r['factor_name']}` | {r['category']} | {r['mean_ic_5d']:.4f} | "
                    f"{r['icir_5d']:.4f} | {r['t_stat_hac_5d']:.2f} | {r['monotonicity_5d']} | **{r['audit_status']}** |"
                )
            (out_path / "batch_leaderboard.md").write_text("\n".join(lines), encoding="utf-8")

        return {
            "summary_table": summary_df.to_dict("records"),
            "individual_reports": results,
        }
