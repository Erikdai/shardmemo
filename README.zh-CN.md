<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo-light.svg" alt="ShardMemo" width="520">
</picture>

<p><strong>面向分片智能体记忆的先作用域、后路由检索</strong></p>

<p>
  <a href="https://arxiv.org/abs/2601.21545"><img src="https://img.shields.io/badge/EMNLP%202026-Findings-1F4E79" alt="EMNLP 2026 Findings"></a>
  <a href="https://arxiv.org/abs/2601.21545"><img src="https://img.shields.io/badge/arXiv-2601.21545-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<p>Yang Zhao<sup>*</sup>, Chengxiao Dai<sup>*</sup>, Mengying Kou, Yue Xiu, Dusit Niyato<br>
<sub><sup>*</sup>共同第一作者</sub></p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

</div>

## 概述

本仓库是 *ShardMemo: Scope-Before-Routing for Agentic Memory Retrieval*（Findings of EMNLP 2026）的官方实现。ShardMemo 是面向大模型智能体的三层记忆服务：Tier A 维护有界的活动上下文，Tier B 把证据存放在带分片内近似最近邻索引的分片中，Tier C 保存带版本的技能库。在 Tier B 中，先由元数据谓词确定请求可以访问哪些分片，再由学习到的掩码路由器从可访问分片中选出少量分片进行检索，因此不可访问的分片不会占用探测预算。实验覆盖 LoCoMo、HotpotQA 和 ToolBench。

## 框架

<p align="center">
  <img src="assets/framework.png" alt="ShardMemo framework" width="100%">
</p>

## 安装

需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/Erikdai/shardmemo.git
cd shardmemo
pip install -r requirements.txt
cp .env.example .env    # 填写大模型 API
```

LoCoMo 和 HotpotQA distractor 验证集会在第一次运行时自动下载到 `data/raw/`。

## 运行

```bash
python experiments/run_locomo.py
python experiments/run_hotpotqa.py
python experiments/run_toolbench.py
```

| 脚本 | 内容 |
| --- | --- |
| `run_locomo.py` | 分类别结果、消融、探测预算扫描、分片数扩展、标签噪声鲁棒性和自适应 Top-P 敏感性；用 `--groups stats main ablations sweep scaling noise sensitivity` 选择要跑的部分 |
| `run_hotpotqa.py` | 56K、224K、448K 上下文长度下的结果，每个文档一个分片 |
| `run_toolbench.py` | Tier C 技能检索与消融、带在线归纳的技能族级 OOD，以及直接执行与 C 到 B 回退的案例 |

所有脚本都支持 `--seeds`。每次运行会写出 `results/<benchmark>/summary.json` 和 `results/<benchmark>/report.md`；嵌入和大模型回答缓存在 `results/cache/`。

全部设置在 `shardmemo/config.py`（`LocomoConfig`、`HotpotQAConfig`、`ToolBenchConfig`）。回答由任意兼容 OpenAI 接口的大模型生成，在 `.env` 中设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。

## 目录结构

```
.
├── assets/
│   ├── framework.png       框架图
│   ├── logo-light.svg      Logo（浅色）
│   └── logo-dark.svg       Logo（深色）
├── shardmemo/
│   ├── config.py           预算、路由器、训练、排序和索引设置；各基准的配置
│   ├── scope.py            记忆元数据、请求和作用域谓词
│   ├── embedding.py        带磁盘缓存的句向量编码器
│   ├── index.py            统计扫描向量数的分片内 HNSW / 暴力索引
│   ├── ranking.py          BM25 与融合证据分数
│   ├── memory.py           证据条目、分片和分片映射构建
│   ├── router.py           分片打分器、掩码 softmax、Top-B_probe 与自适应 Top-P 选择
│   ├── training.py         路由损失与路由器训练
│   ├── tier_a.py           Tier A 活动上下文
│   ├── tier_b.py           Tier B 先作用域后路由检索
│   ├── tier_c.py           Tier C 技能归纳、版本管理、检索和校验执行
│   ├── controller.py       层级门控与三层读取流程
│   ├── generator.py        提示词与 OpenAI 兼容的回答生成器
│   ├── metrics.py          F1、BLEU-1、ShardHit 和延迟统计
│   └── data/
│       ├── locomo.py       LoCoMo 证据、会话锚点和请求
│       ├── hotpotqa.py     MemAgent 长上下文协议下的 HotpotQA
│       └── toolbench.py    ToolBench 风格的工具环境、轨迹和任务
├── experiments/
│   ├── common.py           路由监督、评估和报告工具
│   ├── run_locomo.py       LoCoMo 实验
│   ├── run_hotpotqa.py     HotpotQA 实验
│   └── run_toolbench.py    ToolBench 实验
├── .env.example            大模型 API 配置模板
└── requirements.txt        固定版本的依赖
```

## 引用

```bibtex
@misc{zhao2026shardmemoscopebeforeroutingagenticmemory,
      title={ShardMemo: Scope-Before-Routing for Agentic Memory Retrieval}, 
      author={Yang Zhao and Chengxiao Dai and Mengying Kou and Yue Xiu and Dusit Niyato},
      year={2026},
      eprint={2601.21545},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.21545}, 
}
```

## 许可证

[MIT](LICENSE)
