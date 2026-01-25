# Research Workflow

这是 G-Compress 项目的科研工作流 skill。

## 状态文件

始终先读取 `research_state.yaml` 了解当前状态：
- 当前阶段 (C1-C5)
- 已完成实验
- 待办事项
- 历史发现

## 研究阶段

1. **C1_quantify**: 量化维度坍塌现象
2. **C2_probe**: 探测三层原因 (PyTorch → CUDA → Hardware)
3. **C3_formulate**: 形式化为约束优化问题
4. **C4_solver**: 求解器 + 修复 pass
5. **C5_validation**: Kernel 级别验证

## 工作流程

### 分析阶段
1. 读取 `research_state.yaml` 和 `report.md`
2. 分析当前阶段状态和待办事项
3. 查看已有实验结果 (`results/` 目录)
4. 总结发现

### 决策阶段
1. 根据假设和发现，决定下一步行动
2. 如果需要实验：
   - 设计实验参数
   - 使用 `/slurm` skill 创建并提交作业
3. 如果需要分析：
   - 读取结果文件
   - 生成图表
   - 更新 report.md

### 更新状态
每次操作后更新 `research_state.yaml`:
- 修改阶段 status
- 添加 findings
- 更新 todo
- 记录 experiments

## 输出格式

```markdown
## 当前状态
[阶段名称] - [状态]

## 分析
[基于已有数据的分析]

## 决策
[下一步行动及理由]

## 执行
[具体操作]

## 更新
[状态文件变更]
```

## 实验设计原则

1. 对照实验：aligned vs misaligned dimensions
2. 隔离变量：一次只改变一个因素
3. 多次重复：trials >= 3
4. 记录环境：GPU 型号、驱动版本、PyTorch 版本

## 报告更新

更新 `report.md` 对应章节:
- C1 → Section 1 (Dimensional Collapse)
- C2 → Section 2 (Possible Cause)
- C3-C5 → Section 3 (EuroMLSys contributions)
