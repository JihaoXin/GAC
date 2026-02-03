# AutoGAC v5 简化架构

## 设计原则

**核心理念**：信任 AI 的判断力，代码只做执行和保护。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Simplified Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ Reviewer │───▶│ Planner  │───▶│ Execute  │              │
│   │  评审     │    │  决策     │    │  执行     │              │
│   └──────────┘    └────┬─────┘    └──────────┘              │
│                        │                                     │
│                        ▼                                     │
│              Planner 输出 YAML:                              │
│              actions:                                        │
│                - agent: experimenter                         │
│                  task: "..."                                 │
│                - agent: writer                               │
│                  task: "..."                                 │
│                                                              │
│   ┌──────────────────────────────────────────┐              │
│   │           Memory (极简版)                 │              │
│   │  - scores: [7.0, 7.2, 7.5, ...]          │              │
│   │  - is_stagnating() → bool                │              │
│   │  - GOAL_ANCHOR (常量)                    │              │
│   └──────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Memory System (`memory.py`)

极简版，只追踪分数和停滞：

```python
class SimpleMemory:
    scores: List[float]       # 分数历史（最近20次）
    best_score: float         # 历史最高分
    stagnation_count: int     # 连续停滞计数

    def record_score(score)   # 记录分数
    def is_stagnating()       # 停滞检测
    def get_context()         # 获取上下文（Goal Anchor + 分数趋势）
```

### 2. Goal Anchor（目标锚定）

每次给 Agent 上下文时，始终包含原始目标：

```
论文标题: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs
目标会议: EuroMLSys 2026 (SIGPLAN format, 正文 6 pages, 引用和附录不限)

核心贡献（不可偏离）:
1. 发现并量化 Dimensional Collapse 现象
2. 分析 GPU 性能悬崖的根本原因
3. 提出 GAC 维度修复策略
4. 端到端验证修复效果
```

### 3. Planner Agent

**核心决策者**，输出结构化 action plan：

```yaml
actions:
  - agent: experimenter
    task: "运行 perplexity 实验验证"
    priority: 1
  - agent: writer
    task: "更新 Section 4.2"
    priority: 2
```

### 4. Orchestrator (`orchestrator.py`)

极简流程：

```python
def run_paper_iteration():
    # 1. 审稿
    review = run_agent("reviewer")
    score = parse_score(review)
    memory.record_score(score)

    # 2. 停滞检测
    if memory.is_stagnating():
        send_notification("需要人工介入")

    # 3. Planner 决策 + 执行
    run_planner_cycle(review)

    # 4. 验证 + 提交
    compile_latex()
    git_commit()
```

## Agent 列表

| Agent | 职责 |
|-------|------|
| reviewer | 审稿评分 |
| planner | 分析问题，生成 action plan |
| experimenter | 设计和提交 Slurm 实验 |
| researcher | 分析实验结果 |
| writer | 撰写/修改论文 |
| validator | 验证修改效果 |
| figure_fixer | 修复图表问题 |

## 已废弃

- `events.py` - 事件驱动系统（已废弃，改为 Planner 自主决策）
- 复杂的 Memory 追踪（issues, effective_actions, failed_attempts）

## 文件结构

```
auto_research/
├── orchestrator.py    # 主流程
├── memory.py          # 极简 Memory
├── events.py          # [DEPRECATED]
├── agents/            # Agent prompts
│   ├── reviewer.prompt
│   ├── planner.prompt
│   ├── experimenter.prompt
│   ├── researcher.prompt
│   ├── writer.prompt
│   ├── validator.prompt
│   └── figure_fixer.prompt
├── state/             # 状态文件
│   ├── action_plan.yaml
│   ├── latest_review.md
│   ├── findings.yaml
│   └── memory.yaml
└── logs/              # 运行日志
```
