# DeepResearcher 学习实践

> 基于 [GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) 的学习复现
> 原论文：[DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)

## 项目简介

DeepResearcher 是首个通过端到端强化学习（RL）在真实网络搜索环境中训练 LLM 深度研究 Agent 的框架。本仓库是个人学习复现，用于理解 Agentic RL 的训练范式。

**核心技术点：**
- GRPO（Group Relative Policy Optimization）强化学习训练
- 真实网络搜索交互环境
- 端到端 RL 训练涌现出的规划、反思、多源验证等认知行为
- 基于 veRL 框架的分布式训练

## 学习笔记

### 架构理解

```
┌─────────────────────────────────────────────────────┐
│                  训练循环 (GRPO)                      │
│                                                      │
│  Prompt → Rollout (多步推理+搜索) → Reward → Update   │
│     ↑                                    │            │
│     └────────────────────────────────────┘            │
├─────────────────────────────────────────────────────┤
│                  关键组件                             │
│  - scrl/handler: 搜索处理 + LLM 调用                 │
│  - scrl/trainer: GRPO 训练器                          │
│  - verl/: 分布式训练框架                              │
│  - evaluate/: 评估指标计算                            │
└─────────────────────────────────────────────────────┘
```

### 环境配置

```bash
conda create -n deepresearcher python=3.10
conda activate deepresearcher
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e .
pip3 install -r requirements.txt
```

### 训练流程

1. 启动 Ray 集群
2. 配置搜索 API（Serper/Bing）
3. 启动 Server Handler
4. 运行 `train_grpo.sh` 开始训练
5. 运行 `evaluate.sh` 评估模型

### 关键配置

- `scrl/handler/config.yaml` — 搜索 API 配置、Server URL
- `scrl/handler/server_handler.py` — LLM 调用配置
- `train_grpo.sh` — 训练超参数

## 性能参考

在开放域研究任务上，DeepResearcher 相比 prompt engineering 基线提升最高 28.9 分，相比 RAG-based RL Agent 提升最高 7.2 分。

## 致谢

- 原项目：[GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
- 原论文作者：Yuxiang Zheng, Dayuan Fu, Xiangkun Hu 等
- 基于 [veRL](https://github.com/volcengine/verl) 和 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
