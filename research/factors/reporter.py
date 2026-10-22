"""
因子研究报告生成器 (Factor Reporter)
将因子评估结果固化为标准化的机器可读指标文件 (JSON/Parquet) 与 Markdown/HTML 摘要。
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from research.factors.base import FactorDefinition


class FactorReporter:
    """因子研究多维度报告固化中心"""

    @staticmethod
    def generate_report(
        factor_def: FactorDefinition,
        metrics: Dict[str, Any],
        audit_result: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        生成结构化报告数据并可选输出到指定产物目录
        """
        report_data = {
            "factor_name": factor_def.name,
            "category": factor_def.category,
            "description": factor_def.description,
            "lookback_days": factor_def.lookback_days,
            "direction": factor_def.direction,
            "evaluated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "leakage_audit": audit_result,
        }

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # 1. 固化 factor_metrics.json
            metrics_path = out_path / "factor_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            # 2. 固化 leakage_audit.json
            leakage_path = out_path / "leakage_audit.json"
            leakage_path.write_text(json.dumps(audit_result, indent=2), encoding="utf-8")

            # 3. 生成 Markdown 简报
            md_path = out_path / "factor_summary.md"
            md_lines = [
                f"# 因子效能评估简报: {factor_def.name}",
                f"- **因子类别**: {factor_def.category}",
                f"- **计算回溯期**: {factor_def.lookback_days} 天",
                f"- **防未来函数审查**: **{audit_result.get('status')}**",
                "",
                "## 核心 IC 与 RankIC 表现",
                "| 预测期 (Days) | Mean IC | RankIC IR | Raw t-stat | HAC t-stat (纠偏) | P-Value (HAC) | 单调性得分 |",
                "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
            
            p_data = metrics.get("periods", {})
            m_scores = metrics.get("monotonicity_scores", {})
            for p_key, row in p_data.items():
                m_score = m_scores.get(p_key, "N/A")
                md_lines.append(
                    f"| {p_key} | {row['mean_ic']:.4f} | {row['icir']:.4f} | {row['t_stat_raw']:.2f} | "
                    f"{row['t_stat_hac']:.2f} | {row['p_value_hac']:.6f} | {m_score} |"
                )

            md_path.write_text("\n".join(md_lines), encoding="utf-8")

        return report_data
