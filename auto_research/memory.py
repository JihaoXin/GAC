#!/usr/bin/env python3
"""
AutoGAC Memory System - 增强版

功能：
1. 记录分数历史
2. 停滞检测
3. Goal Anchor
4. Issue 重复追踪（新增）
5. 修复验证（新增）
6. 元反思检查（新增）

设计原则：信任 AI 的判断力，代码只做执行和保护
"""

import yaml
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional

MEMORY_FILE = Path(__file__).parent / "state" / "memory.yaml"

# 配置
STAGNATION_THRESHOLD = 5
MIN_PROGRESS_DELTA = 0.3

# Issue 类型分类关键词
TECHNICAL_KEYWORDS = [
    "missing data", "no validation", "insufficient evidence", "benchmark",
    "experiment", "evaluation", "comparison", "e2e", "end-to-end",
    "perplexity", "accuracy", "latency measurement", "no proof"
]
PRESENTATION_KEYWORDS = [
    "font", "size", "layout", "overlap", "caption", "spacing", "margin",
    "related work", "citation", "figure", "table", "color", "width",
    "crowded", "dense", "readability", "visual", "formatting"
]

# Goal Anchor - 防止偏离大方向
GOAL_ANCHOR = """
## 论文核心目标 (Goal Anchor)

**论文标题**: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs
**目标会议**: EuroMLSys 2026 (SIGPLAN format, 正文 6 pages, 引用和附录不限)

**核心贡献** (不可偏离):
1. 发现并量化 Dimensional Collapse 现象
2. 分析 GPU 性能悬崖的根本原因 (TC, Vec, BW, L2)
3. 提出 GAC 维度修复策略
4. 端到端验证修复效果

**关键约束**:
- 6 页限制（不含引用）
- 保持技术深度，不泛泛而谈
- 每个论点必须有数据支撑
"""


class SimpleMemory:
    """增强版迭代记忆 - 追踪分数、停滞、Issue重复、修复验证"""

    def __init__(self):
        self.scores: List[float] = []
        self.best_score: float = 0.0
        self.stagnation_count: int = 0

        # 新增：Issue 重复追踪
        self.issue_history: Dict[str, int] = {}  # issue_id -> 出现次数
        self.last_issues: List[str] = []  # 上次 review 的 issues

        # 新增：Issue 修复方法历史（记录每个 issue 用过什么方法）
        self.issue_repair_methods: Dict[str, List[str]] = {}  # issue_id -> [方法列表]

        # 新增：修复验证
        self.expected_changes: Dict[str, str] = {}  # file_path -> change_type
        self.last_repair_iteration: int = 0  # 上次 self_repair 的迭代号
        self.repair_effective: Optional[bool] = None  # 上次修复是否有效

        # Meta-Debugger 支持
        self.experiment_empty_count: int = 0  # 实验空转计数

        self.load()

    def load(self):
        """从文件加载"""
        if MEMORY_FILE.exists():
            try:
                data = yaml.safe_load(MEMORY_FILE.read_text()) or {}
                self.scores = data.get("scores", [])
                self.best_score = data.get("best_score", 0.0)
                self.stagnation_count = data.get("stagnation_count", 0)
                # 新增字段
                self.issue_history = data.get("issue_history", {})
                self.last_issues = data.get("last_issues", [])
                self.issue_repair_methods = data.get("issue_repair_methods", {})
                self.expected_changes = data.get("expected_changes", {})
                self.last_repair_iteration = data.get("last_repair_iteration", 0)
                self.repair_effective = data.get("repair_effective")
                # Meta-Debugger 支持
                self.experiment_empty_count = data.get("experiment_empty_count", 0)
            except Exception:
                pass

    def save(self):
        """保存到文件"""
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(yaml.dump({
            "scores": self.scores[-20:],  # 只保留最近 20 个
            "best_score": self.best_score,
            "stagnation_count": self.stagnation_count,
            # Issue 追踪
            "issue_history": self.issue_history,
            "last_issues": self.last_issues,
            "issue_repair_methods": self.issue_repair_methods,
            "expected_changes": self.expected_changes,
            "last_repair_iteration": self.last_repair_iteration,
            "repair_effective": self.repair_effective,
            # Meta-Debugger 支持
            "experiment_empty_count": getattr(self, 'experiment_empty_count', 0),
            "last_updated": datetime.now().isoformat(),
        }, allow_unicode=True))

    def record_score(self, score: float):
        """记录新分数"""
        prev_score = self.scores[-1] if self.scores else 0.0
        self.scores.append(score)

        # 更新最高分
        if score > self.best_score:
            self.best_score = score

        # 停滞检测
        if (score - prev_score) >= MIN_PROGRESS_DELTA:
            self.stagnation_count = 0  # 有效进步，重置
        else:
            self.stagnation_count += 1

        self.save()

    def is_stagnating(self) -> Tuple[bool, str]:
        """检测是否停滞"""
        if self.stagnation_count >= STAGNATION_THRESHOLD:
            return True, f"连续 {self.stagnation_count} 次无有效进步 (delta < {MIN_PROGRESS_DELTA})"

        # 检查是否在原地打转
        if len(self.scores) >= 6:
            recent = self.scores[-6:]
            variance = max(recent) - min(recent)
            if variance < 0.5:
                return True, f"最近 6 次分数波动过小 ({variance:.2f})"

        return False, ""

    def get_context(self) -> str:
        """获取简单上下文（给 Agent 用）"""
        lines = [GOAL_ANCHOR, ""]

        # 停滞警告
        is_stuck, reason = self.is_stagnating()
        if is_stuck:
            lines.append(f"⚠️ **停滞警告**: {reason}")
            lines.append("建议：换一种完全不同的方法，或补充实验数据")
            lines.append("")

        # 分数趋势
        if self.scores:
            lines.append("## 分数趋势")
            lines.append(f"- 当前: **{self.scores[-1]}/10**")
            lines.append(f"- 最高: **{self.best_score}/10**")
            lines.append(f"- 历史: {' → '.join(f'{s:.1f}' for s in self.scores[-5:])}")
            lines.append("")

        # 添加自检报告（重复 Issue 警告等）
        self_check = self.get_self_check_report()
        if self_check:
            lines.append(self_check)

        return "\n".join(lines)

    def get_context_for_agent(self, agent_type: str) -> str:
        """兼容旧接口"""
        return self.get_context()

    # ==================== 新增：Issue 重复追踪 ====================

    def record_issues(self, issues: List[str], iteration: int):
        """记录本次 review 中出现的 issues

        Args:
            issues: Issue ID 列表，如 ["M1", "M2", "m1"]
            iteration: 当前迭代号
        """
        already_counted = set()

        # 检查上次修复是否有效（如果有的话）
        if self.last_repair_iteration > 0:
            # 检查上次修复后的 issues 是否还在
            repeat_issues = set(issues) & set(self.last_issues)
            if repeat_issues:
                self.repair_effective = False
                # 记录哪些 issue 修复无效
                for issue_id in repeat_issues:
                    self.issue_history[issue_id] = self.issue_history.get(issue_id, 0) + 1
                    already_counted.add(issue_id)
            else:
                self.repair_effective = True

        # 更新 issue 历史（跳过已在 repeat 分支中计数过的）
        for issue_id in issues:
            if issue_id not in already_counted:
                self.issue_history[issue_id] = self.issue_history.get(issue_id, 0) + 1

        self.last_issues = issues
        self.save()

    def get_repeat_issues(self, threshold: int = 3) -> List[Tuple[str, int]]:
        """获取重复出现的 issues

        Args:
            threshold: 出现次数阈值，默认 3

        Returns:
            List of (issue_id, count) tuples for issues appearing >= threshold times
        """
        return [(k, v) for k, v in self.issue_history.items() if v >= threshold]

    def get_issue_count(self, issue_id: str) -> int:
        """获取某个 issue 出现的次数"""
        return self.issue_history.get(issue_id, 0)

    # ==================== 新增：修复方法历史 ====================

    def record_repair_method(self, issue_id: str, method: str):
        """记录某个 issue 使用的修复方法（允许重复，记录尝试次数）

        Args:
            issue_id: Issue ID，如 "M1"
            method: 修复方法，如 "WRITING_ONLY", "FIGURE_CODE_REQUIRED"
        """
        if issue_id not in self.issue_repair_methods:
            self.issue_repair_methods[issue_id] = []
        # 允许重复添加，这样可以追踪同一方法被尝试了多少次
        self.issue_repair_methods[issue_id].append(method)
        self.save()

    def get_tried_methods(self, issue_id: str) -> List[str]:
        """获取某个 issue 已经尝试过的修复方法"""
        return self.issue_repair_methods.get(issue_id, [])

    def classify_issue_type(self, issue_description: str) -> str:
        """根据问题描述分类问题类型

        Returns:
            "technical" - 需要实验数据支撑的问题
            "presentation" - 排版/视觉/写作问题
        """
        desc_lower = issue_description.lower()

        # 先检查是否是技术问题
        for keyword in TECHNICAL_KEYWORDS:
            if keyword in desc_lower:
                return "technical"

        # 检查是否是展示问题
        for keyword in PRESENTATION_KEYWORDS:
            if keyword in desc_lower:
                return "presentation"

        # 默认为展示问题（更安全，不会乱跑实验）
        return "presentation"

    def get_banned_methods(self, issue_id: str, issue_description: str = "") -> List[str]:
        """获取某个 issue 的禁用方法列表

        新逻辑：根据问题类型区分处理
        - PRESENTATION 问题：永远不强制 EXPERIMENT_REQUIRED
        - TECHNICAL 问题：可以升级到 EXPERIMENT_REQUIRED

        Args:
            issue_id: Issue ID
            issue_description: Issue 描述（用于分类）
        """
        count = self.get_issue_count(issue_id)
        tried = self.get_tried_methods(issue_id)
        issue_type = self.classify_issue_type(issue_description)

        # PRESENTATION 问题：循环使用方法，不强制 EXPERIMENT
        if issue_type == "presentation":
            # 如果 WRITING_ONLY 试过 3+ 次但 FIGURE_CODE 没试过，建议换方法
            writing_tries = tried.count("WRITING_ONLY") if tried else 0
            figure_tries = tried.count("FIGURE_CODE_REQUIRED") if tried else 0

            if writing_tries >= 3 and figure_tries < 2:
                return ["WRITING_ONLY"]  # 禁用 WRITING_ONLY，建议 FIGURE_CODE
            elif figure_tries >= 3 and writing_tries < 2:
                return ["FIGURE_CODE_REQUIRED"]  # 反过来
            # 其他情况不禁用
            return []

        # TECHNICAL 问题：可以升级到 EXPERIMENT
        if count >= 10:
            return ["WRITING_ONLY", "FIGURE_CODE_REQUIRED", "LITERATURE_REQUIRED"]
        elif count >= 5:
            return ["WRITING_ONLY", "FIGURE_CODE_REQUIRED"]
        elif count >= 3:
            return ["WRITING_ONLY"]
        return []

    def get_strategy_escalation(self, issue_descriptions: Dict[str, str] = None) -> Dict[str, dict]:
        """获取需要策略升级的 issues

        Args:
            issue_descriptions: Optional dict of issue_id -> description for better classification

        Returns:
            Dict of issue_id -> {count, tried_methods, banned_methods, required_escalation, issue_type}
        """
        escalations = {}
        issue_descriptions = issue_descriptions or {}

        for issue_id, count in self.issue_history.items():
            if count >= 3:  # 重复 3+ 次需要关注
                tried = self.get_tried_methods(issue_id)
                desc = issue_descriptions.get(issue_id, "")
                issue_type = self.classify_issue_type(desc)
                banned = self.get_banned_methods(issue_id, desc)

                # 根据问题类型确定升级方向
                required = None

                if issue_type == "presentation":
                    # 展示问题：循环使用不同方法
                    if "WRITING_ONLY" in tried and "FIGURE_CODE_REQUIRED" not in tried:
                        required = "FIGURE_CODE_REQUIRED (修改 Python 绘图脚本)"
                    elif "FIGURE_CODE_REQUIRED" in tried and "LITERATURE_REQUIRED" not in tried:
                        required = "LITERATURE_REQUIRED (补充引用和 Related Work)"
                    elif all(m in tried for m in ["WRITING_ONLY", "FIGURE_CODE_REQUIRED"]):
                        required = "换一种完全不同的表述方式，或检查问题是否真的存在"
                else:
                    # 技术问题：可以升级到 EXPERIMENT
                    if count >= 7:
                        required = "EXPERIMENT_REQUIRED (需要新实验数据)"
                    elif count >= 5:
                        if "WRITING_ONLY" in tried:
                            required = "FIGURE_CODE_REQUIRED or EXPERIMENT_REQUIRED"
                    elif count >= 3:
                        if "WRITING_ONLY" in tried:
                            required = "Try FIGURE_CODE_REQUIRED"

                escalations[issue_id] = {
                    "count": count,
                    "tried_methods": tried,
                    "banned_methods": banned,
                    "required_escalation": required,
                    "issue_type": issue_type
                }
        return escalations

    def reset_issue_counts(self, reason: str = "manual reset"):
        """重置 issue 计数器（保留方法历史）

        用于清理因 bug 导致的错误累积

        Args:
            reason: 重置原因（记录用）
        """
        # 保留方法历史，只重置计数
        self.issue_history = {}
        self.last_issues = []
        self.repair_effective = None
        self.stagnation_count = 0
        # 保留 scores 和 best_score
        self.save()
        return f"Issue counts reset ({reason}). Method history preserved."

    def soft_reset_counts(self, max_count: int = 5):
        """软重置：将所有计数限制到 max_count

        用于修复计数过高但不完全清零的情况

        Args:
            max_count: 最大保留计数
        """
        for issue_id in self.issue_history:
            if self.issue_history[issue_id] > max_count:
                self.issue_history[issue_id] = max_count
        self.save()
        return f"Issue counts capped at {max_count}"

    # ==================== 新增：修复验证 ====================

    def record_expected_changes(self, changes: Dict[str, str]):
        """记录预期的修改

        Args:
            changes: Dict of file_path -> change_type
                     如 {"scripts/create_paper_figures.py": "FIGURE_CODE_REQUIRED"}
        """
        self.expected_changes = changes
        self.save()

    def verify_changes(self, modified_files: List[str]) -> Tuple[bool, List[str]]:
        """验证预期的修改是否发生

        Args:
            modified_files: 实际被修改的文件列表

        Returns:
            (all_verified, missing_files) tuple
        """
        if not self.expected_changes:
            return True, []

        missing = []
        for expected_file in self.expected_changes.keys():
            # 检查文件是否在修改列表中（支持部分匹配）
            found = any(expected_file in f or f in expected_file for f in modified_files)
            if not found:
                missing.append(expected_file)

        return len(missing) == 0, missing

    def mark_repair_attempt(self, iteration: int):
        """标记 self_repair 尝试

        Args:
            iteration: 当前迭代号
        """
        self.last_repair_iteration = iteration
        self.repair_effective = None  # 待验证
        self.save()

    def was_last_repair_effective(self) -> Tuple[bool, str]:
        """检查上次 self_repair 是否有效

        Returns:
            (effective, reason) tuple
        """
        if self.repair_effective is None:
            return True, "No repair attempted yet"
        elif self.repair_effective:
            return True, "Last repair was effective"
        else:
            repeat = self.get_repeat_issues(threshold=2)
            if repeat:
                return False, f"Issues still repeating: {[r[0] for r in repeat[:3]]}"
            return False, "Last repair was ineffective"

    # ==================== 新增：自检报告 ====================

    def get_self_check_report(self, issue_descriptions: Dict[str, str] = None) -> str:
        """生成自检报告

        Args:
            issue_descriptions: Optional dict of issue_id -> description
        """
        lines = ["## 自检报告\n"]

        # 1. 策略升级需求（按问题类型分组）
        escalations = self.get_strategy_escalation(issue_descriptions)
        if escalations:
            # 分组：展示问题 vs 技术问题
            presentation_issues = {k: v for k, v in escalations.items()
                                   if v.get("issue_type") == "presentation"}
            technical_issues = {k: v for k, v in escalations.items()
                               if v.get("issue_type") == "technical"}

            if presentation_issues:
                lines.append("### 📊 展示/排版问题（用 WRITING_ONLY 或 FIGURE_CODE）")
                lines.append("")
                for issue_id, info in sorted(presentation_issues.items(), key=lambda x: -x[1]["count"]):
                    count = info["count"]
                    tried = info["tried_methods"]
                    required = info["required_escalation"]

                    lines.append(f"**{issue_id}** (重复 {count} 次):")
                    if tried:
                        # 统计每种方法的尝试次数
                        method_counts = {}
                        for m in tried:
                            method_counts[m] = method_counts.get(m, 0) + 1
                        method_str = ", ".join(f"{m}×{c}" for m, c in method_counts.items())
                        lines.append(f"  - 已尝试: {method_str}")
                    if required:
                        lines.append(f"  - 💡 建议: **{required}**")
                    lines.append("")

            if technical_issues:
                lines.append("### 🔬 技术问题（可能需要 EXPERIMENT）")
                lines.append("")
                for issue_id, info in sorted(technical_issues.items(), key=lambda x: -x[1]["count"]):
                    count = info["count"]
                    tried = info["tried_methods"]
                    banned = info["banned_methods"]
                    required = info["required_escalation"]

                    lines.append(f"**{issue_id}** (重复 {count} 次):")
                    if tried:
                        method_counts = {}
                        for m in tried:
                            method_counts[m] = method_counts.get(m, 0) + 1
                        method_str = ", ".join(f"{m}×{c}" for m, c in method_counts.items())
                        lines.append(f"  - 已尝试: {method_str}")
                    if banned:
                        lines.append(f"  - ❌ 禁用: {', '.join(banned)}")
                    if required:
                        lines.append(f"  - ✅ 必须: **{required}**")
                    lines.append("")

        # 2. 重复 Issue 检测（如果没有 escalations）
        repeat_issues = self.get_repeat_issues(threshold=3)
        if repeat_issues and not escalations:
            lines.append("### ⚠️ 重复出现的 Issues（需要换方法！）")
            for issue_id, count in sorted(repeat_issues, key=lambda x: -x[1]):
                lines.append(f"- **{issue_id}**: 出现 {count} 次")
            lines.append("")

        # 3. 修复有效性
        effective, reason = self.was_last_repair_effective()
        if not effective:
            lines.append("### ⚠️ 上次修复无效")
            lines.append(f"原因: {reason}")
            lines.append("建议: 不要重复同样的方法，需要换一种完全不同的策略")
            lines.append("")

        # 4. 预期修改验证
        if self.expected_changes:
            lines.append("### 预期修改清单")
            for f, change_type in self.expected_changes.items():
                lines.append(f"- [ ] {f} ({change_type})")
            lines.append("")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ==================== Meta-Debugger 支持 ====================

    def get_diagnosis_context(self) -> Dict:
        """生成 Meta-Debugger 诊断所需的上下文

        Returns:
            包含所有诊断相关信息的字典
        """
        return {
            "scores": {
                "current": self.scores[-1] if self.scores else 0.0,
                "best": self.best_score,
                "recent": self.scores[-10:] if self.scores else [],
                "trend": self._calculate_trend(),
            },
            "stagnation": {
                "count": self.stagnation_count,
                "is_stagnating": self.is_stagnating()[0],
                "reason": self.is_stagnating()[1],
            },
            "issues": {
                "history": self.issue_history,
                "last_issues": self.last_issues,
                "repeat_issues": self.get_repeat_issues(threshold=3),
                "high_repeat": self.get_repeat_issues(threshold=7),
            },
            "repair": {
                "last_iteration": self.last_repair_iteration,
                "effective": self.repair_effective,
                "methods_used": self.issue_repair_methods,
                "expected_changes": self.expected_changes,
            },
            "experiment_empty_count": getattr(self, 'experiment_empty_count', 0),
        }

    def _calculate_trend(self) -> str:
        """计算分数趋势"""
        if len(self.scores) < 2:
            return "insufficient_data"
        recent = self.scores[-5:]
        if len(recent) < 2:
            return "insufficient_data"
        delta = recent[-1] - recent[0]
        if delta > 0.3:
            return "improving"
        elif delta < -0.3:
            return "declining"
        else:
            return "stagnant"

    def get_health_status(self) -> Tuple[str, List[str]]:
        """获取系统健康状态

        Returns:
            (status, reasons) 其中 status 为 HEALTHY, WARNING, 或 CRITICAL
        """
        reasons = []

        # 检查停滞
        is_stuck, reason = self.is_stagnating()
        if is_stuck:
            reasons.append(f"Stagnation: {reason}")

        # 检查高重复 issues
        high_repeat = self.get_repeat_issues(threshold=7)
        if high_repeat:
            issue_list = [f"{i[0]}({i[1]}x)" for i in high_repeat[:3]]
            reasons.append(f"High repeat issues: {', '.join(issue_list)}")

        # 检查分数下降
        if len(self.scores) >= 2 and self.scores[-1] < self.scores[-2] - 0.3:
            reasons.append(f"Score dropped: {self.scores[-2]:.2f} -> {self.scores[-1]:.2f}")

        # 检查修复有效性
        effective, repair_reason = self.was_last_repair_effective()
        if not effective and "repeating" in repair_reason.lower():
            reasons.append(f"Repair ineffective: {repair_reason}")

        # 检查实验空转
        empty_count = getattr(self, 'experiment_empty_count', 0)
        if empty_count >= 2:
            reasons.append(f"Experiments empty: {empty_count} times")

        # 判断状态
        if len(reasons) >= 3 or any("High repeat" in r for r in reasons):
            return "CRITICAL", reasons
        elif len(reasons) >= 1:
            return "WARNING", reasons
        else:
            return "HEALTHY", []

    def mark_experiment_empty(self):
        """标记实验产生空结果"""
        if not hasattr(self, 'experiment_empty_count'):
            self.experiment_empty_count = 0
        self.experiment_empty_count += 1
        self.save()

    def clear_experiment_empty(self):
        """清除实验空结果计数（当实验成功时）"""
        self.experiment_empty_count = 0
        self.save()

    def should_trigger_meta_debug(self) -> Tuple[bool, str]:
        """判断是否应该触发 Meta-Debugger

        Returns:
            (should_trigger, reason)
        """
        # 条件 1: 停滞
        if self.stagnation_count >= 3:
            return True, f"stagnation ({self.stagnation_count} iterations)"

        # 条件 2: Issue 高重复
        high_repeat = self.get_repeat_issues(threshold=7)
        if high_repeat:
            return True, f"issue_repeat ({high_repeat[0][0]}: {high_repeat[0][1]}x)"

        # 条件 3: 分数大幅下降
        if len(self.scores) >= 2:
            delta = self.scores[-1] - self.scores[-2]
            if delta <= -0.3:
                return True, f"score_drop ({delta:.2f})"

        # 条件 4: 实验空转
        empty_count = getattr(self, 'experiment_empty_count', 0)
        if empty_count >= 2:
            return True, f"experiment_empty ({empty_count}x)"

        return False, ""

    def reset(self):
        """重置"""
        self.scores = []
        self.best_score = 0.0
        self.stagnation_count = 0
        self.issue_history = {}
        self.last_issues = []
        self.issue_repair_methods = {}
        self.expected_changes = {}
        self.last_repair_iteration = 0
        self.repair_effective = None
        self.experiment_empty_count = 0
        self.save()


# 兼容旧接口的别名
IterationMemory = SimpleMemory

# 单例
_memory = None


def get_memory() -> SimpleMemory:
    """获取全局 Memory 实例"""
    global _memory
    if _memory is None:
        _memory = SimpleMemory()
    return _memory
