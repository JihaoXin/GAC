# 论文修改讨论记录
**日期**: 2026-01-31
**论文**: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs

---

## Background 部分

### ✅ 压缩方法范围确认
**时间**: 2026-01-31 下午

**决策**:
- **主要聚焦**: SVD (低秩分解) 和 Structured Pruning (结构化剪枝)
- **次要提及**: NAS 只是一个额外的扩展，一句话带过即可

**代表工作**:
- SVD: PaLU, SVD-LLM, ASVD, FWSVD
- Pruning: Wanda, SparseGPT

**原因**:
- 这两类方法是主流，读者最熟悉
- NAS 是小众场景，不需要展开
- 保持论文聚焦，避免分散注意力

**待办**:
- [ ] 在 Background §2.3 重写压缩方法介绍
- [ ] 确保 SVD 和 Pruning 各有 1-2 个段落
- [ ] NAS 最多一句话："Other methods like NAS may also produce irregular dimensions..."

---

## 待讨论问题

(空)

---

## 下一步行动

1. 继续讨论其他章节的修改点
2. 根据讨论记录批量更新论文
3. 重新编译检查

---

**更新日志**:
- 2026-01-31: 创建讨论记录，记录 Background 部分范围确认
