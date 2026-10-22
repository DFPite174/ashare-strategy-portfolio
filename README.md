# ashare-strategy-portfolio

> 企业级 A 股量化策略构建与组合优化器子系统 (A-Share Quantitative Strategy & Portfolio Optimization Subsystem)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-pep8-brightgreen.svg)](https://pep8.org/)

---

## 🌟 核心功能特性 (Key Features)

1. **结构化经济假说库 (`strategy.hypotheses`)**：
   - 杜绝纯数据挖掘与过拟合，以经济学因果逻辑（动量突破、价值均值回归、高成长质量、低波动异象）驱动策略设计。
   - 严格采用不可变 Pydantic v2 模型封装假说元数据与市场环境适用状态。

2. **双轨因子挑选与合成流水线 (`strategy.selection`)**：
   - **Track 1 (统计正交贪心挑选器)**：基于 Rank IC / ICIR 最大化目标，引入 $|corr| \le 0.35$ 严格截面低相关约束与施密特正交化降共线性。
   - **Track 2 (语义/Agent 假说挑选与公式合成器)**：支持自然语言量化意图分词、假说匹配，并自动合成安全 AST 抽象语法树公式动态注册入库。
   - **分层协同架构 (Hybrid Pipeline)**：阶段一由语义漏斗锚定主题因子池，阶段二由统计正交化剔除冗余噪音。

3. **四大组合优化器套件 (`strategy.optimizers`)**：
   - **等权与打分加权优化器 (`EqualWeightOptimizer`, `ScoreWeightedOptimizer`)**：基础基准优化器。
   - **均值-方差优化器 (`MeanVarianceOptimizer`)**：基于 `cvxpy` 凸优化求解器，支持二次规划风险厌恶、单票持仓上限与全投资约束。
   - **风险平价优化器 (`RiskParityOptimizer`)**：基于 `scipy.optimize` SLSQP 非线性规划，精确求解每个资产风险贡献相等（ERC）的权重配置。
   - **层次风险平价优化器 (`HierarchicalRiskParityOptimizer`, HRP)**：基于 Marcos López de Prado 的层次聚类树拟对角化与递归二分分配，彻底摆脱协方差矩阵求逆不稳定与病态条件数困扰。

4. **策略资产注册库与生命周期中台 (`strategy.zoo`)**：
   - 统一策略接口 `BaseStrategy` 与多因子轮动策略模型 `MultiFactorStrategy`。
   - 策略生命周期状态机：`CANDIDATE` 候选 $\rightarrow$ `AUDITED` 审核通过 $\rightarrow$ `INCUBATING` 模拟孵化 $\rightarrow$ `PRODUCTION` 实盘运行 $\rightarrow$ `RETIRED` 淘汰。

---

## 📐 核心算法与数学原理

### 1. 层次风险平价 (HRP)
层次风险平价 (Hierarchical Risk Parity, Marcos López de Prado, 2016) 避免了 Markowitz 均值-方差优化中由于协方差矩阵奇异造成的权重剧烈翻转：
1. **相关性距离矩阵**：
   $$d_{i,j} = \sqrt{\frac{1}{2} (1 - \rho_{i,j})}$$
2. **拟对角化树聚类**：使用单链接法 (`single linkage`) 聚类并重排叶节点索引。
3. **递归二分分配**：对左右子簇 $C_1, C_2$ 的方差按方差反比分配权重：
   $$\alpha_1 = 1 - \frac{V_1}{V_1 + V_2}, \quad \alpha_2 = 1 - \alpha_1$$

### 2. 施密特正交化因子融合 (Gram-Schmidt Orthogonalization)
对于选出的候选因子序列 $f_1, f_2, \dots, f_k$，依次剔除后续因子在先前因子空间上的投影：
$$f_k^* = f_k - \sum_{j=1}^{k-1} \frac{\langle f_k, f_j^* \rangle}{\langle f_j^*, f_j^* \rangle} f_j^*$$
从而生成各维度完全正交、低冗余度的纯净 Alpha 信号。

---

## 🚀 快速上手 (Quickstart)

```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示脚本
python examples/quickstart.py
```

### Python 代码调用示例：
```python
import numpy as np
from strategy.optimizers import HierarchicalRiskParityOptimizer, OptimizationConstraints

# 1. 准备资产预期收益与协方差矩阵
returns = np.array([0.15, 0.12, 0.08, 0.22])
cov_matrix = np.array([
    [0.0625, 0.0150, 0.0037, 0.0350],
    [0.0150, 0.0400, 0.0060, 0.0175],
    [0.0037, 0.0060, 0.0225, 0.0026],
    [0.0350, 0.0175, 0.0026, 0.1225]
])

# 2. 求解层次风险平价 (HRP) 权重
optimizer = HierarchicalRiskParityOptimizer()
weights = optimizer.optimize(
    expected_returns=returns,
    cov_matrix=cov_matrix,
    constraints=OptimizationConstraints(max_weight=0.40)
)
print("Optimal Weights:", weights)
```

---

## 🧪 单元测试

```bash
pytest tests/ -v
```

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 授权。
