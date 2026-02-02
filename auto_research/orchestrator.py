#!/usr/bin/env python3
"""
GAC (GPU-Aligned Compression) 全自动科研主控系统

运行方式:
    # 研究模式（实验 + 分析）
    python auto_research/orchestrator.py --mode research --max-days 3

    # 论文模式（审稿 + 改进）
    python auto_research/orchestrator.py --mode paper --max-iterations 10

    # 后台运行
    nohup python auto_research/orchestrator.py --mode paper --max-iterations 20 > auto_research/logs/orchestrator.log 2>&1 &

功能:
    研究模式 (research):
        1. 读取研究状态，决定下一步行动
        2. 调用专业 Agent 执行任务
        3. 等待 Slurm 作业完成
        4. 分析结果，更新状态
        5. 循环直到完成或超时

    论文模式 (paper):
        1. Reviewer Agent 审稿，给出评分和改进建议
        2. Writer Agent 根据建议改进论文
        3. 编译 LaTeX 验证
        4. 循环直到评分达标 (>= 8/10) 或达到最大迭代次数
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import re

# 项目根目录（必须在导入自定义模块之前设置）
PROJECT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

# 极简版 Memory（不再使用 Event System）
from auto_research.memory import get_memory, SimpleMemory

# 配置
STATE_FILE = PROJECT_DIR / "auto_research" / "state" / "research_state.yaml"
FINDINGS_FILE = PROJECT_DIR / "auto_research" / "state" / "findings.yaml"
PAPER_STATE_FILE = PROJECT_DIR / "auto_research" / "state" / "paper_state.yaml"
PAPER_REQUIREMENTS_FILE = PROJECT_DIR / "auto_research" / "state" / "paper_requirements.yaml"
LOG_DIR = PROJECT_DIR / "auto_research" / "logs"
AGENTS_DIR = PROJECT_DIR / "auto_research" / "agents"
LATEX_DIR = PROJECT_DIR / "Latex"
FIGURES_DIR = LATEX_DIR / "figures"

# 确保目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 论文审稿通过阈值
PAPER_ACCEPT_THRESHOLD = 8  # 总体评分 >= 8/10 则通过

# Checkpoint 文件
CHECKPOINT_FILE = PROJECT_DIR / "auto_research" / "state" / "checkpoint.yaml"

# Action Plan 文件（Planner 生成）
ACTION_PLAN_FILE = PROJECT_DIR / "auto_research" / "state" / "action_plan.yaml"


class Orchestrator:
    """主控调度器"""

    def __init__(self, max_days: float = 3, max_iterations: int = 100, mode: str = "research"):
        self.max_end_time = datetime.now() + timedelta(days=max_days)
        self.max_iterations = max_iterations
        self.iteration = 0
        self.mode = mode  # "research" or "paper"

        # 统一日志文件: AutoGAC_<mode>_<date>.log
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"AutoGAC_{mode}_{self.run_id}.log"

        # 启动时清理旧日志
        self._cleanup_old_logs(keep=5)

        # Token 使用统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Rate limit 状态（防止重复通知）
        self._rate_limit_notified = False

        # 极简版 Memory（不再使用 EventQueue）
        self.memory = get_memory()
        self._last_score = 0.0  # 追踪评分变化

    def save_checkpoint(self):
        """保存运行状态检查点"""
        checkpoint = {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "mode": self.mode,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        with open(CHECKPOINT_FILE, "w") as f:
            yaml.dump(checkpoint, f, default_flow_style=False)
        self.log(f"Checkpoint saved: iteration={self.iteration}", "INFO")

    def load_checkpoint(self) -> dict:
        """加载检查点"""
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                return yaml.safe_load(f) or {}
        return {}

    def resume_from_checkpoint(self):
        """从检查点恢复"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.iteration = checkpoint.get("iteration", 0)
            self.total_input_tokens = checkpoint.get("total_input_tokens", 0)
            self.total_output_tokens = checkpoint.get("total_output_tokens", 0)
            self.log(f"Resumed from checkpoint: iteration={self.iteration}", "INFO")
            return True
        return False

    def _parse_rate_limit_wait(self, error_msg: str) -> int:
        """从 rate limit 错误信息中解析等待时间（秒）

        支持格式：
        - "retry after 60 seconds"
        - "retry after 2026-01-25T14:30:00"
        - "reset at 1706188200" (Unix timestamp)
        - "wait 5 minutes"

        Returns:
            等待秒数，解析失败返回 300（默认 5 分钟）
        """
        import re
        from datetime import datetime

        error_lower = error_msg.lower()

        # 格式1: "retry after X seconds" 或 "wait X seconds"
        match = re.search(r"(?:retry after|wait)\s+(\d+)\s*(?:seconds?|s)", error_lower)
        if match:
            return int(match.group(1))

        # 格式2: "X minutes"
        match = re.search(r"(?:retry after|wait)\s+(\d+)\s*(?:minutes?|m)", error_lower)
        if match:
            return int(match.group(1)) * 60

        # 格式3: ISO 时间戳 "2026-01-25T14:30:00"
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", error_msg)
        if match:
            try:
                reset_time = datetime.fromisoformat(match.group(1))
                wait_seconds = (reset_time - datetime.now()).total_seconds()
                return max(int(wait_seconds), 60)  # 至少等 60 秒
            except ValueError:
                pass

        # 格式4: Unix timestamp
        match = re.search(r"reset.*?(\d{10})", error_lower)
        if match:
            try:
                reset_time = datetime.fromtimestamp(int(match.group(1)))
                wait_seconds = (reset_time - datetime.now()).total_seconds()
                return max(int(wait_seconds), 60)
            except (ValueError, OSError):
                pass

        # 默认：5 分钟
        return 300

    def _cleanup_old_logs(self, keep: int = 5):
        """清理旧日志文件，保留最近的 N 个"""
        # 清理 AutoGAC_*.log
        autogac_logs = sorted(LOG_DIR.glob("AutoGAC_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        for old_log in autogac_logs[keep:]:
            old_log.unlink()

        # 清理旧的 agent_*.log 和 orchestrator_*.log（兼容旧格式）
        for pattern in ["agent_*.log", "orchestrator_*.log"]:
            old_logs = sorted(LOG_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
            for old_log in old_logs[keep:]:
                old_log.unlink()

    def _summarize_agent_output(self, agent_type: str, output: str) -> list:
        """生成 Agent 输出的摘要，返回列表格式"""
        if not output or len(output) < 50:
            return []

        summary_lines = []

        if agent_type == "reviewer":
            # 从 latest_review.md 读取关键信息
            review_file = PROJECT_DIR / "auto_research" / "state" / "latest_review.md"
            if review_file.exists():
                content = review_file.read_text()

                # 提取总评分
                score_match = re.search(r"\*\*Total\*\*.*?\*\*(\d+\.?\d*)/10\*\*", content, re.DOTALL)
                if score_match:
                    summary_lines.append(f"Score: {score_match.group(1)}/10")

                # 提取 Rating
                rating_match = re.search(r"\*\*Rating:\s*([^*\n]+)", content)
                if rating_match:
                    summary_lines.append(f"Rating: {rating_match.group(1).strip()}")

                # 提取各维度分数（紧凑格式）
                dimensions = [
                    ("Technical Quality", "Tech"),
                    ("Paper Presentation", "Pres"),
                    ("Innovation", "Innov"),
                    ("Writing Quality", "Write"),
                ]
                dim_scores = []
                for eng, chn in dimensions:
                    dim_match = re.search(rf"\|\s*{eng}\s*\|[^|]*\|\s*(\d+)/10", content)
                    if dim_match:
                        dim_scores.append(f"{chn}:{dim_match.group(1)}")
                if dim_scores:
                    summary_lines.append(" | ".join(dim_scores))

                # 提取 Major Issues
                major_issues = re.findall(r"### M\d+\.\s*([^\n]+)", content)
                if major_issues:
                    summary_lines.append(f"Major Issues ({len(major_issues)}):")
                    for issue in major_issues[:3]:
                        summary_lines.append(f"  • {issue.strip()[:40]}")

                # 提取 Minor Issues 数量
                minor_issues = re.findall(r"### m\d+\.\s*([^\n]+)", content)
                if minor_issues:
                    summary_lines.append(f"Minor Issues: {len(minor_issues)}")

        elif agent_type == "writer":
            if "main.tex" in output.lower():
                summary_lines.append("Modified: Latex/main.tex")
            fig_changes = re.findall(r"fig\d+|figure\s*\d+", output, re.IGNORECASE)
            if fig_changes:
                summary_lines.append(f"Figures touched: {len(set(fig_changes))}")

        elif agent_type == "validator":
            if "通过" in output or "pass" in output.lower():
                summary_lines.append("Result: PASS")
            elif "失败" in output or "fail" in output.lower():
                summary_lines.append("Result: FAIL")

        elif agent_type == "experimenter":
            slurm_jobs = re.findall(r"sbatch|srun|slurm", output, re.IGNORECASE)
            if slurm_jobs:
                summary_lines.append(f"Slurm jobs submitted: {len(slurm_jobs)}")

        elif agent_type == "planner":
            # 从 action_plan.yaml 读取
            if ACTION_PLAN_FILE.exists():
                try:
                    with open(ACTION_PLAN_FILE) as f:
                        plan = yaml.safe_load(f) or {}
                    issues = plan.get("issues", [])
                    if issues:
                        exp_count = sum(1 for i in issues if i.get("type") == "EXPERIMENT_REQUIRED")
                        write_count = sum(1 for i in issues if i.get("type") == "WRITING_ONLY")
                        summary_lines.append(f"Issues: {len(issues)} total")
                        summary_lines.append(f"  Experiments: {exp_count}, Writing: {write_count}")
                except Exception:
                    pass

        return summary_lines

    def cleanup_workspace(self):
        """清理工作区（LaTeX 临时文件、旧日志等）"""
        self.log("清理工作区...", "INFO")
        cleaned = 0

        # 1. 清理 LaTeX 临时文件
        latex_temp_exts = [".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz"]
        for ext in latex_temp_exts:
            for f in LATEX_DIR.glob(f"*{ext}"):
                try:
                    f.unlink()
                    cleaned += 1
                except Exception:
                    pass

        # 2. 清理旧的 page_*.png（保留最新一批）
        page_images = sorted(LATEX_DIR.glob("page_*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
        for img in page_images[10:]:  # 保留最多 10 张
            try:
                img.unlink()
                cleaned += 1
            except Exception:
                pass

        # 3. 清理 Python 缓存
        for cache_dir in PROJECT_DIR.rglob("__pycache__"):
            if cache_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(cache_dir)
                    cleaned += 1
                except Exception:
                    pass

        self.log(f"清理完成: 删除 {cleaned} 个临时文件", "INFO")

    def log(self, message: str, level: str = "INFO"):
        """记录日志，支持级别和实时刷新"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 简洁格式：只有时间戳，无级别标签（级别通过缩进和图标体现）
        if level == "RAW":
            log_message = message
        else:
            log_message = f"[{timestamp}] {message}"

        print(log_message, flush=True)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
            f.flush()

    def log_section(self, title: str, char: str = "═"):
        """打印大节标题"""
        line = char * 70
        self.log(line, "RAW")
        self.log(f"  {title}", "RAW")
        self.log(line, "RAW")

    def log_phase(self, phase_num: int, total_phases: int, name: str, status: str = "start"):
        """打印阶段标记

        Args:
            phase_num: 当前阶段编号
            total_phases: 总阶段数
            name: 阶段名称
            status: "start" 或 "end"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        if status == "start":
            self.log("", "RAW")
            self.log(f"┌─ PHASE {phase_num}/{total_phases}: {name} " + "─" * max(0, 50 - len(name)), "RAW")
            self.log(f"│ [{timestamp}] Starting...", "RAW")
        else:
            self.log(f"│ [{timestamp}] ✓ Completed", "RAW")
            self.log("└" + "─" * 69, "RAW")

    def log_step(self, message: str, status: str = "info"):
        """打印步骤信息（在阶段内部使用）

        Args:
            message: 步骤描述
            status: "info", "success", "warning", "error", "progress"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "info": " ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
            "progress": "→",
        }
        icon = icons.get(status, " ")
        self.log(f"│ [{timestamp}] {icon} {message}", "RAW")

    def log_summary_box(self, title: str, items: list, inside_phase: bool = True):
        """打印摘要框

        Args:
            title: 摘要标题
            items: 摘要项列表，每项为字符串
            inside_phase: 是否在阶段内部（带 │ 前缀）
        """
        prefix = "│   " if inside_phase else ""
        if inside_phase:
            self.log("│", "RAW")
        self.log(f"{prefix}┌─ {title} " + "─" * max(0, 50 - len(title)) + "┐", "RAW")
        for item in items:
            lines = item.split("\n") if "\n" in item else [item]
            for line in lines:
                if len(line) > 52:
                    line = line[:49] + "..."
                self.log(f"{prefix}│ {line:<52} │", "RAW")
        self.log(f"{prefix}└" + "─" * 54 + "┘", "RAW")

    def load_state(self) -> dict:
        """加载研究状态"""
        with open(STATE_FILE) as f:
            return yaml.safe_load(f)

    def save_state(self, state: dict):
        """保存研究状态"""
        with open(STATE_FILE, "w") as f:
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

    def get_current_phase(self, state: dict) -> str:
        """获取当前研究阶段"""
        phases = state.get("phases", {})
        for phase_name in ["C1_quantify", "C2_probe", "C3_formulate", "C4_solver", "C5_validation"]:
            phase = phases.get(phase_name, {})
            if phase.get("status") in ["pending", "in_progress"]:
                return phase_name
        return "completed"

    def run_agent(self, agent_type: str, task: str, timeout: int = 1800) -> str:
        """运行指定类型的 Agent，输出写入统一日志"""
        import json

        prompt_file = AGENTS_DIR / f"{agent_type}.prompt"
        if not prompt_file.exists():
            self.log(f"Agent prompt not found: {prompt_file}", "ERROR")
            return ""

        base_prompt = prompt_file.read_text()

        # 获取历史上下文（Memory System）
        history_context = self.memory.get_context_for_agent(agent_type)

        full_prompt = f"""{base_prompt}

---

## 当前任务

{task}

## 迭代历史（Memory）

{history_context if history_context else "这是第一次迭代，无历史记录。"}

## 上下文文件

请阅读以下文件获取上下文（如果存在）：
- auto_research/state/research_state.yaml - 当前研究状态
- auto_research/state/findings.yaml - 已有发现
- report.md - 研究报告
- results/ 目录 - 实验结果

执行任务并更新相应文件。
"""

        # 提取任务简短描述（取第一行或前50字符）
        task_brief = task.split("\n")[0][:50].strip()
        if len(task.split("\n")[0]) > 50:
            task_brief += "..."
        self.log_step(f"Agent [{agent_type}] → {task_brief}", "progress")

        try:
            # 使用 --no-session-persistence 避免 session 状态导致的 tool_use id 重复 bug
            process = subprocess.Popen(
                [
                    "claude", "-p", full_prompt,
                    "--dangerously-skip-permissions",
                    "--no-session-persistence",
                    "--output-format", "text",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PROJECT_DIR,
            )

            start_time = time.time()
            result = ""

            # 使用 communicate 并设置超时
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result = stdout

                # 检查 stderr 中的错误
                if stderr:
                    stderr_lower = stderr.lower()
                    # 检测 rate limit
                    if "rate limit" in stderr_lower or "rate_limit" in stderr_lower:
                        wait_seconds = self._parse_rate_limit_wait(stderr)
                        wait_minutes = wait_seconds / 60
                        self.log(f"检测到 Rate Limit: 需等待 {wait_minutes:.1f} 分钟", "WARN")

                        if not self._rate_limit_notified:
                            self._rate_limit_notified = True
                            self.save_checkpoint()
                            self.send_notification(
                                f"Rate Limit - 等待 {wait_minutes:.0f} 分钟",
                                f"Agent {agent_type} 触发 rate limit。\n"
                                f"等待时间: {wait_minutes:.1f} 分钟\n"
                                f"预计恢复: {(datetime.now() + timedelta(seconds=wait_seconds)).strftime('%H:%M:%S')}\n"
                                f"系统将自动等待后恢复。",
                                priority="critical"
                            )

                        self.log(f"等待 {wait_minutes:.1f} 分钟后自动恢复...", "INFO")
                        time.sleep(wait_seconds + 10)
                        self._rate_limit_notified = False
                    # 检测 API 错误
                    elif "error" in stderr_lower and "api" in stderr_lower:
                        self.log(f"  [{agent_type}] API Error: {stderr[:200]}", "AGENT")

            except subprocess.TimeoutExpired:
                process.kill()
                self.log(f"Agent {agent_type} 超时 ({timeout}s)", "WARN")
                stdout, _ = process.communicate()
                result = stdout

            elapsed = int(time.time() - start_time)

            # 空转检测：执行时间 < 15s 且输出太短 → Agent 可能因 API 问题返回空结果
            MIN_AGENT_TIME = 15  # 正常 agent 至少需要 15 秒
            MIN_RESULT_LEN = 100  # 正常输出至少 100 字符
            if elapsed < MIN_AGENT_TIME and len(result.strip()) < MIN_RESULT_LEN:
                self.log(f"⚠️ Agent [{agent_type}] 空转检测: 执行仅 {elapsed}s，输出仅 {len(result.strip())} 字符", "WARN")
                self.log(f"  可能原因: API rate limit / token 耗尽 / 连接失败", "WARN")
                # 记录空转事件，增加空转计数
                if not hasattr(self, '_agent_empty_count'):
                    self._agent_empty_count = 0
                self._agent_empty_count += 1
                # 连续 3 次空转则暂停等待
                if self._agent_empty_count >= 3:
                    wait_time = 300  # 5 分钟
                    self.log(f"🛑 连续 {self._agent_empty_count} 次 Agent 空转，暂停 {wait_time}s 等待 API 恢复", "ERROR")
                    self.send_notification(
                        "Agent 连续空转",
                        f"连续 {self._agent_empty_count} 次 Agent 在 <15s 内完成，可能 API 受限。\n暂停 5 分钟等待恢复。",
                        priority="critical"
                    )
                    time.sleep(wait_time)
                    self._agent_empty_count = 0
            else:
                # 正常执行，重置空转计数
                if hasattr(self, '_agent_empty_count'):
                    self._agent_empty_count = 0

            self.log_step(f"Agent [{agent_type}] completed ({elapsed}s)", "success")

            # 生成并打印 Agent 摘要
            summary_items = self._summarize_agent_output(agent_type, result)
            if summary_items:
                self.log_summary_box(f"{agent_type.upper()} Summary", summary_items)

            return result

        except Exception as e:
            self.log(f"Agent {agent_type} 错误: {e}", "ERROR")
            self.send_notification("Agent 错误", f"Agent {agent_type} 发生错误: {e}", priority="critical")
            return ""

    # ========== 智能调度：记忆系统 ==========

    def record_score_to_memory(self, score: float):
        """记录分数到 Memory（极简版）"""
        self.memory.record_score(score)
        self._last_score = score
        self.log(f"Memory: 记录评分 {score}/10", "MEMORY")

    def get_memory_context(self) -> str:
        """获取 Memory 上下文（包含 Goal Anchor、停滞警告、分数趋势）"""
        return self.memory.get_context()

    def wait_for_slurm(self, max_wait_hours: float = 4, job_prefix: str = "GAC_") -> bool:
        """等待 AutoGAC 提交的 Slurm 作业完成

        只等待作业名称以 job_prefix 开头的作业（默认 "GAC_"），
        忽略其他项目的作业。

        Args:
            max_wait_hours: 最大等待时间（小时）
            job_prefix: 作业名称前缀，只等待匹配的作业
        """
        self.log(f"Checking Slurm jobs (prefix: {job_prefix})...")
        max_wait = timedelta(hours=max_wait_hours)
        start_time = datetime.now()

        while datetime.now() - start_time < max_wait:
            try:
                # 使用 -o 格式获取作业名称: "%j" = job name
                result = subprocess.run(
                    ["squeue", "-u", os.environ.get("USER", "xinj"), "-h", "-o", "%j"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if not result.stdout.strip():
                    self.log("No Slurm jobs found")
                    return True

                # 只统计匹配前缀的作业
                all_jobs = result.stdout.strip().split("\n")
                gac_jobs = [j for j in all_jobs if j.startswith(job_prefix)]
                running_jobs = len(gac_jobs)

                if running_jobs == 0:
                    self.log(f"All {job_prefix}* jobs completed (other jobs: {len(all_jobs)})")
                    return True

                self.log(f"{running_jobs} {job_prefix}* jobs running, waiting 60s...")
                time.sleep(60)

            except Exception as e:
                self.log(f"Error checking Slurm: {e}")
                time.sleep(60)

        self.log(f"Slurm wait timeout after {max_wait_hours} hours")
        return False

    def git_commit(self, message: str, files: list = None):
        """在关键节点自动 git commit

        Args:
            message: commit 消息
            files: 要提交的文件列表，None 表示所有更改
        """
        try:
            # 先检查是否有更改
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=PROJECT_DIR, timeout=30
            )

            if not status_result.stdout.strip():
                self.log("Git: 没有更改需要提交", "INFO")
                return False

            # 添加文件
            if files:
                for f in files:
                    subprocess.run(["git", "add", f], cwd=PROJECT_DIR, timeout=30)
            else:
                # 添加关键文件（排除临时文件）
                key_files = [
                    "Latex/main.tex",
                    "report.md",
                    "auto_research/state/*.yaml",
                    "auto_research/state/*.md",
                ]
                for pattern in key_files:
                    subprocess.run(
                        ["git", "add", pattern],
                        cwd=PROJECT_DIR, timeout=30,
                        capture_output=True  # 忽略不存在的文件警告
                    )

            # 提交
            commit_msg = f"[AutoGAC] {message}\n\nIteration: {self.iteration}\nScore: {getattr(self, '_last_score', 'N/A')}"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, text=True, cwd=PROJECT_DIR, timeout=60
            )

            if result.returncode == 0:
                self.log(f"Git commit: {message}", "INFO")
                return True
            else:
                self.log(f"Git commit 失败: {result.stderr[:200]}", "WARN")
                return False

        except Exception as e:
            self.log(f"Git commit 错误: {e}", "ERROR")
            return False

    def send_notification(self, subject: str, message: str, priority: str = "normal"):
        """发送邮件通知

        Args:
            subject: 邮件主题
            message: 邮件内容
            priority: 优先级 ("normal", "critical")
        """
        # 关键词触发邮件通知
        critical_keywords = ["error", "failed", "token", "accepted", "completed", "timeout"]
        should_send = priority == "critical" or any(kw in subject.lower() for kw in critical_keywords)

        if not should_send:
            self.log(f"Notification skipped (non-critical): {subject}", "INFO")
            return

        try:
            email = "jihao.xin@kaust.edu.sa"
            # 添加状态摘要
            full_message = f"""{message}

---
AutoGAC Status:
- Run ID: {self.run_id}
- Mode: {self.mode}
- Iteration: {self.iteration}
- Total Tokens: in={self.total_input_tokens:,}, out={self.total_output_tokens:,}
- Log: {self.log_file}
"""
            subprocess.run(
                ["mail", "-s", f"[AutoGAC] {subject}", email],
                input=full_message,
                text=True,
                timeout=30,
            )
            self.log(f"Notification sent: {subject}", "INFO")
        except Exception as e:
            self.log(f"Failed to send notification: {e}", "WARN")

    def compile_latex(self) -> bool:
        """编译 LaTeX 论文"""
        self.log("Compiling LaTeX...")
        try:
            # 运行 pdflatex + bibtex + pdflatex x2
            for cmd in [
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ]:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=LATEX_DIR,
                )
                if result.returncode != 0 and "main.tex" in cmd:
                    self.log(f"LaTeX compilation warning: {result.stderr[:500]}")

            # 检查 PDF 是否生成
            pdf_path = LATEX_DIR / "main.pdf"
            if pdf_path.exists():
                self.log(f"LaTeX compiled successfully: {pdf_path}")
                return True
            else:
                self.log("LaTeX compilation failed: PDF not generated")
                return False
        except Exception as e:
            self.log(f"LaTeX compilation error: {e}")
            return False

    def load_paper_state(self) -> dict:
        """加载论文审稿状态"""
        if PAPER_STATE_FILE.exists():
            with open(PAPER_STATE_FILE) as f:
                return yaml.safe_load(f) or {}
        return {
            "reviews": [],
            "current_score": 0,
            "status": "in_progress",
        }

    def load_paper_requirements(self) -> dict:
        """加载论文需求配置"""
        if PAPER_REQUIREMENTS_FILE.exists():
            with open(PAPER_REQUIREMENTS_FILE) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _generate_figures_from_results(self) -> bool:
        """从最新实验结果生成图表并复制到论文目录"""
        self.log(">>> 从实验结果生成新图表...", "FIGURES")
        try:
            import glob
            import shutil

            # 找最新的实验结果目录
            result_dirs = []
            for exp_type in ["llm", "perplexity_validation", "gemm", "sdpa"]:
                pattern = str(PROJECT_DIR / f"results/{exp_type}/*")
                dirs = glob.glob(pattern)
                result_dirs.extend(dirs)

            if not result_dirs:
                self.log("  没有找到新实验结果", "FIGURES")
                self.memory.mark_experiment_empty()  # 标记实验空转
                return False

            # 按修改时间排序，取最新的
            result_dirs.sort(key=lambda x: Path(x).stat().st_mtime if Path(x).is_dir() else 0, reverse=True)
            latest_result = Path(result_dirs[0])

            self.log(f"  最新结果: {latest_result}", "FIGURES")

            # 尝试运行对应的图表生成脚本
            if "llm" in str(latest_result):
                cmd = f"python -m scripts.plot_llm_results {latest_result.parent}"
            else:
                cmd = f"python -m scripts.plot_results {latest_result}"

            self.log(f"  运行: {cmd}", "FIGURES")
            result = subprocess.run(
                ["bash", "-c", f"source ~/.bashrc && mamba activate gac && {cmd}"],
                capture_output=True, text=True, timeout=300, cwd=PROJECT_DIR
            )

            if result.returncode != 0:
                self.log(f"  图表生成警告: {result.stderr[:200]}", "WARN")

            # 复制新生成的图表到 Latex/figures/
            plots_dir = latest_result / "plots"
            if plots_dir.exists():
                for pdf in plots_dir.glob("*.pdf"):
                    dest = FIGURES_DIR / pdf.name
                    shutil.copy(pdf, dest)
                    self.log(f"  复制图表: {pdf.name} -> Latex/figures/", "FIGURES")
                self.memory.clear_experiment_empty()  # 实验成功产出结果

            return True

        except Exception as e:
            self.log(f"  图表生成错误: {e}", "ERROR")
            return False

    def generate_figures(self) -> bool:
        """运行图表生成脚本"""
        self.log("Generating paper figures...", "INFO")
        try:
            # 确保 figures 目录存在
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["bash", "-c", "source ~/.bashrc && mamba activate gac && python scripts/create_paper_figures.py"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=PROJECT_DIR,
            )

            if result.returncode == 0:
                self.log("Figure generation completed successfully", "INFO")
                # 生成后进行质量预检
                self._precheck_figure_quality()
                return True
            else:
                self.log(f"Figure generation warning: {result.stderr[:500]}", "WARN")
                return True  # 继续，即使有警告
        except Exception as e:
            self.log(f"Figure generation error: {e}", "ERROR")
            return False

    def _precheck_figure_quality(self):
        """图表质量预检 - 在 Reviewer 之前主动发现问题

        检查项：
        1. 图表比例是否过于扁平
        2. 分辨率是否足够
        3. PDF 版本是否存在
        """
        self.log(">>> 图表质量预检...", "PRECHECK")
        try:
            from PIL import Image

            issues = []
            for fig_path in sorted(FIGURES_DIR.glob("fig*.png")):
                img = Image.open(fig_path)
                width, height = img.size
                aspect_ratio = height / width

                # 检查 1: 比例是否过于扁平 (高度应 >= 宽度 * 0.35)
                if aspect_ratio < 0.35:
                    issues.append(f"  ⚠️ {fig_path.name}: 比例过扁 ({aspect_ratio:.2f})，建议高度 ≥ 宽度×0.4")

                # 检查 2: 分辨率是否足够
                if width < 600:
                    issues.append(f"  ⚠️ {fig_path.name}: 宽度过小 ({width}px)，建议 ≥ 800px")

                img.close()

            # 检查 PDF 版本是否存在
            for png_path in FIGURES_DIR.glob("fig*.png"):
                pdf_path = png_path.with_suffix('.pdf')
                if not pdf_path.exists():
                    issues.append(f"  ⚠️ {png_path.stem}: 缺少 PDF 版本")

            if issues:
                self.log("发现图表质量问题:", "PRECHECK")
                for issue in issues:
                    self.log(issue, "PRECHECK")
                self.log("建议修改 scripts/create_paper_figures.py 并重新生成", "PRECHECK")
            else:
                self.log("图表质量检查通过 ✓", "PRECHECK")

        except ImportError:
            self.log("Pillow 未安装，跳过图表预检", "WARN")
        except Exception as e:
            self.log(f"图表预检错误: {e}", "WARN")

    def pdf_to_images(self) -> list:
        """将 PDF 转换为图像用于视觉审核"""
        self.log("Converting PDF to images for visual review...", "INFO")
        try:
            result = subprocess.run(
                ["bash", "-c", "source ~/.bashrc && mamba activate gac && python scripts/pdf_to_images.py Latex/main.pdf --dpi 150"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=PROJECT_DIR,
            )

            if result.returncode == 0:
                # 获取生成的图像列表
                images = list(LATEX_DIR.glob("page_*.png"))
                self.log(f"Generated {len(images)} page images", "INFO")
                return [str(img) for img in sorted(images)]
            else:
                self.log(f"PDF to images warning: {result.stderr[:200]}", "WARN")
                return []
        except Exception as e:
            self.log(f"PDF to images error: {e}", "ERROR")
            return []

    def save_paper_state(self, state: dict):
        """保存论文审稿状态"""
        with open(PAPER_STATE_FILE, "w") as f:
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

    def _load_action_plan(self) -> dict:
        """加载 Planner 生成的 action plan

        包含容错处理：Planner 生成的 YAML 可能包含 LaTeX 命令（如 \\subsection），
        在双引号内会被 YAML 解析器当作非法 escape sequence。
        修复方法：将双引号值转换为单引号（单引号内反斜杠不会被转义）。
        """
        if ACTION_PLAN_FILE.exists():
            try:
                with open(ACTION_PLAN_FILE) as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                self.log(f"YAML 解析错误，尝试修复 LaTeX escape: {e}", "WARN")
                try:
                    raw = ACTION_PLAN_FILE.read_text()
                    import re
                    # 将包含反斜杠的双引号值转换为单引号
                    # 匹配: "...\...." → '...\....'
                    def fix_dquoted(match):
                        content = match.group(1)
                        if '\\' in content:
                            # 双引号内有反斜杠，转为单引号
                            # 先处理单引号转义（单引号内用 '' 表示 '）
                            content = content.replace("'", "''")
                            return "'" + content + "'"
                        return match.group(0)

                    # 匹配双引号字符串（非贪婪，不跨行）
                    fixed = re.sub(r'"([^"\n]*)"', fix_dquoted, raw)
                    ACTION_PLAN_FILE.write_text(fixed)
                    result = yaml.safe_load(fixed) or {}
                    self.log("YAML 修复成功（LaTeX escape → 单引号）", "INFO")
                    return result
                except Exception as e2:
                    self.log(f"YAML 修复失败: {e2}，返回空 plan", "ERROR")
                    return {"issues": []}
        return {"issues": []}

    def _save_action_plan(self, action_plan: dict):
        """保存 action plan"""
        with open(ACTION_PLAN_FILE, "w") as f:
            yaml.dump(action_plan, f, default_flow_style=False, allow_unicode=True)

    def _load_findings_summary(self) -> str:
        """加载 findings.yaml 的摘要"""
        if FINDINGS_FILE.exists():
            with open(FINDINGS_FILE) as f:
                findings = yaml.safe_load(f) or {}
            # 返回摘要（前 500 字符）
            return yaml.dump(findings, allow_unicode=True)[:500]
        return "暂无 findings"

    def _wait_for_slurm_jobs(self, max_wait_hours: float = 2) -> bool:
        """等待 Slurm 作业完成（内部使用的简化版本）"""
        return self.wait_for_slurm(max_wait_hours=max_wait_hours)

    def parse_review_score(self, review_output: str) -> float:
        """从审稿输出中解析总体评分"""
        # 支持多种格式：
        # 1. "总体评分: 7/10" 或 "Overall Score: 7/10"
        # 2. "| **Total** | 100% | - | **7.5/10** |" (表格格式)
        # 3. 任意 "X/10" 或 "X.X/10"
        patterns = [
            r"总体评分[：:]\s*(\d+\.?\d*)/10",
            r"Overall Score[：:]\s*(\d+\.?\d*)/10",
            r"总分[：:]\s*(\d+\.?\d*)/10",
            r"\*\*Total\*\*.*?\*\*(\d+\.?\d*)/10\*\*",  # 表格 Markdown 格式
            r"\|\s*Total\s*\|.*?(\d+\.?\d*)/10",       # 简化表格格式
        ]
        for pattern in patterns:
            match = re.search(pattern, review_output, re.IGNORECASE | re.DOTALL)
            if match:
                score = float(match.group(1))
                self.log(f"解析到评分: {score}/10")
                return score

        # 兜底：从 latest_review.md 文件读取
        review_file = PROJECT_DIR / "auto_research" / "state" / "latest_review.md"
        if review_file.exists():
            content = review_file.read_text()
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    score = float(match.group(1))
                    self.log(f"从 latest_review.md 解析到评分: {score}/10")
                    return score

        self.log("警告: 未能解析到评分，返回 0")
        return 0.0

    def extract_issue_ids(self) -> List[str]:
        """从 latest_review.md 提取所有 issue IDs

        Returns:
            List of issue IDs like ["M1", "M2", "m1", "m2"]
        """
        review_file = PROJECT_DIR / "auto_research" / "state" / "latest_review.md"
        if not review_file.exists():
            return []

        content = review_file.read_text()

        # 匹配 issue ID 模式：
        # - M1, M2, M3 (Major)
        # - m1, m2, m3 (minor)
        # - 也支持 "### M1." 或 "**M1:**" 等格式
        issue_pattern = r'\b([Mm]\d+)\b'
        matches = re.findall(issue_pattern, content)

        # 去重并标准化
        unique_issues = list(set(matches))
        self.log(f"提取到 {len(unique_issues)} 个 issues: {unique_issues}")
        return unique_issues

    def _check_needs_experiment(self, review_output: str) -> bool:
        """分析审稿意见，判断是否需要补充实验

        检测关键词：
        - "需要补充实验" / "add experiment"
        - "缺少数据" / "missing data"
        - "验证不足" / "insufficient validation"
        - "建议增加" / "suggest adding"
        """
        # 也检查 latest_review.md 文件
        review_file = PROJECT_DIR / "auto_research" / "state" / "latest_review.md"
        content = review_output
        if review_file.exists():
            content += "\n" + review_file.read_text()

        # 实验相关关键词
        experiment_keywords = [
            r"需要.*实验",
            r"补充.*实验",
            r"缺少.*数据",
            r"验证不足",
            r"建议.*增加.*实验",
            r"add.*experiment",
            r"missing.*data",
            r"insufficient.*validation",
            r"suggest.*adding.*experiment",
            r"need.*more.*evidence",
            r"require.*additional.*test",
        ]

        for pattern in experiment_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                self.log(f"检测到实验需求关键词: {pattern}", "INFO")
                return True

        return False

    def _check_needs_literature_search(self, review_output: str) -> tuple:
        """检查是否需要文献调研

        Returns:
            (needs_search: bool, search_topics: list)
        """
        # 读取 latest_review.md
        review_file = PROJECT_DIR / "auto_research" / "state" / "latest_review.md"
        content = review_output
        if review_file.exists():
            content += "\n" + review_file.read_text()

        search_topics = []

        # 检测 related work 相关问题
        related_work_keywords = [
            r"related work.*insufficient",
            r"related work.*missing",
            r"缺少.*相关工作",
            r"should cite",
            r"compare with.*other",
            r"missing.*comparison",
            r"prior work",
            r"existing.*method",
        ]
        for pattern in related_work_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                search_topics.append("related_work")
                break

        # 检测技术验证需求
        tech_keywords = [
            r"verify.*claim",
            r"documentation.*support",
            r"FlashAttention.*behavior",
            r"Tensor Core.*requirement",
            r"技术.*验证",
        ]
        for pattern in tech_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                search_topics.append("technical_verification")
                break

        # 检测竞争方法对比需求
        comparison_keywords = [
            r"compare.*baseline",
            r"other.*compression",
            r"alternative.*method",
            r"state.of.the.art",
            r"SOTA",
        ]
        for pattern in comparison_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                search_topics.append("competitive_analysis")
                break

        return len(search_topics) > 0, search_topics

    def run_literature_search(self, topics: list) -> str:
        """运行文献调研

        Args:
            topics: 需要调研的主题列表
        """
        self.log_step(f"Literature search: {topics}", "progress")

        topic_prompts = {
            "related_work": """
搜索与以下主题相关的论文和工作：
1. LLM 压缩方法 (SVD, low-rank, quantization)
2. FlashAttention 和 GPU attention 优化
3. Tensor Core 对齐和 CUDA 优化

搜索词建议：
- "LLM compression dimension alignment"
- "FlashAttention head dimension requirements"
- "Tensor Core alignment performance"
- "PaLU low-rank compression"

找到相关论文后，提取：
- 论文标题、作者、年份
- 核心贡献
- 与我们工作的关系
- 是否需要在 Related Work 中引用
""",
            "technical_verification": """
查找技术文档验证以下声明：
1. FlashAttention 对 head_dim 的要求
2. PyTorch SDPA backend 选择逻辑
3. Tensor Core mma 指令的对齐要求

查找资源：
- FlashAttention GitHub repo 和文档
- PyTorch 官方文档
- NVIDIA CUDA 编程指南

提取具体的技术规格和代码引用。
""",
            "competitive_analysis": """
搜索竞争方法的性能数据：
1. 其他 LLM 压缩方法的 latency/memory tradeoff
2. 推理框架（vLLM, TensorRT-LLM）的优化策略
3. 其他维度对齐解决方案

找到后提取：
- 方法名称
- 性能数据（latency, memory, accuracy）
- 与我们方法的对比点
"""
        }

        combined_prompt = "请进行以下文献调研：\n\n"
        for topic in topics:
            if topic in topic_prompts:
                combined_prompt += f"## {topic}\n{topic_prompts[topic]}\n\n"

        combined_prompt += """
完成后，将重要发现保存到 auto_research/state/literature.yaml
格式：
```yaml
searches:
  - date: "YYYY-MM-DD"
    topic: "topic_name"
    findings:
      - title: "论文/文档标题"
        source: "URL 或引用"
        relevance: "与我们工作的关系"
        key_points: ["要点1", "要点2"]
    action_items:
      - "需要引用的论文"
      - "需要对比的方法"
```
"""
        return self.run_agent("literature", combined_prompt, timeout=1800)

    def run_planner_cycle(self, review_output: str) -> bool:
        """Planner 驱动的迭代循环

        流程:
        1. 检查是否需要文献调研
        2. Planner 分析 review，生成 action_plan.yaml
        3. 执行所有实验类任务（inner loop）
        4. 所有实验完成后，执行写作任务

        Args:
            review_output: Reviewer 的输出

        Returns:
            True 如果成功完成
        """
        import json

        # Step 0: 检查是否需要文献调研
        needs_lit, lit_topics = self._check_needs_literature_search(review_output)
        if needs_lit:
            self.log_step(f"Literature search needed: {lit_topics}", "progress")
            self.run_literature_search(lit_topics)

        # Step 1: Planner 分析 review，生成 action plan
        self.log_step("Planner analyzing review...", "progress")
        findings_summary = self._load_findings_summary()

        # 获取策略升级要求（关键！）
        # 注意：这里没有 issue descriptions，所以用空 dict
        # 实际分类会在 plan 执行后基于 action_plan 的 title/description 进行
        escalations = self.memory.get_strategy_escalation()
        escalation_prompt = ""
        if escalations:
            escalation_prompt = "\n## 🚨 策略升级建议\n\n"
            escalation_prompt += "**注意**: 系统会根据问题类型智能判断是否需要升级\n"
            escalation_prompt += "- 展示/排版问题：优先用 WRITING_ONLY 或 FIGURE_CODE_REQUIRED\n"
            escalation_prompt += "- 技术/数据问题：可能需要 EXPERIMENT_REQUIRED\n\n"
            for issue_id, info in escalations.items():
                count = info["count"]
                banned = info["banned_methods"]
                required = info["required_escalation"]
                issue_type = info.get("issue_type", "unknown")
                escalation_prompt += f"**{issue_id}** (重复 {count} 次, 类型: {issue_type}):\n"
                if banned:
                    escalation_prompt += f"  - ⚠️ 已尝试多次: {', '.join(banned)}\n"
                if required:
                    escalation_prompt += f"  - 💡 建议: **{required}**\n"
                escalation_prompt += "\n"

        planner_output = self.run_agent("planner", f"""
分析以下审稿意见，生成 action_plan.yaml。
{escalation_prompt}
## 审稿意见摘要
请读取完整审稿报告：auto_research/state/latest_review.md

## 当前 Findings 摘要
{findings_summary}

## 任务
1. 识别所有 Major Issues (M1, M2, ...) 和 Minor Issues (m1, m2, ...)
2. 对每个 issue 分类：EXPERIMENT_REQUIRED, FIGURE_CODE_REQUIRED, 或 WRITING_ONLY
3. ⚠️ 检查策略升级要求，确保不使用禁用的方法
4. 为每个 issue 生成具体的 action 列表
5. 将结果写入 auto_research/state/action_plan.yaml

注意：
- EXPERIMENT_REQUIRED: 需要跑 GPU 实验（如添加 perplexity 测量、补充 benchmark）
- FIGURE_CODE_REQUIRED: 需要修改 Python 绘图脚本并重新运行
- WRITING_ONLY: 只需修改 LaTeX 文字
""", timeout=1200)

        # Step 2: 加载生成的 action plan
        action_plan = self._load_action_plan()
        issues = action_plan.get("issues", [])

        if not issues:
            self.log_step("Planner generated no issues, skipping", "warning")
            return False

        exp_count = sum(1 for i in issues if i.get("type") == "EXPERIMENT_REQUIRED")
        lit_count = sum(1 for i in issues if i.get("type") == "LITERATURE_REQUIRED")
        fig_count = sum(1 for i in issues if i.get("type") == "FIGURE_CODE_REQUIRED")
        write_count = sum(1 for i in issues if i.get("type") == "WRITING_ONLY")
        self.log_step(f"Plan: {len(issues)} issues (exp:{exp_count}, lit:{lit_count}, fig:{fig_count}, write:{write_count})", "success")

        # 记录每个 issue 使用的修复方法（用于自检和策略升级）
        violations = []
        for issue in issues:
            issue_id = issue.get("id", "")
            issue_type = issue.get("type", "")
            # 获取 issue 描述用于分类（优先 title，其次 description）
            issue_desc = issue.get("title", "") or issue.get("description", "")
            if issue_id and issue_type:
                self.memory.record_repair_method(issue_id, issue_type)
                # 检查是否违反了策略升级要求（传入描述以便正确分类）
                banned = self.memory.get_banned_methods(issue_id, issue_desc)
                if issue_type in banned:
                    violations.append((issue_id, issue_type, banned, issue_desc))
                    self.log(f"🚫 违规: {issue_id} 使用了被禁用的方法 {issue_type}！", "ERROR")

        # 如果有违规，提供建议但不强制重新规划（让系统自己判断）
        if violations:
            self.log("💡 检测到可能的策略改进建议", "WARNING")
            for v in violations:
                issue_id, issue_type, banned, issue_desc = v
                # 检查问题类型
                pres_type = self.memory.classify_issue_type(issue_desc)
                if pres_type == "presentation":
                    self.log(f"  {issue_id}: 展示问题，{issue_type} 可能合适，继续执行", "INFO")
                else:
                    self.log(f"  {issue_id}: 技术问题，建议换用 {[m for m in ['FIGURE_CODE_REQUIRED', 'EXPERIMENT_REQUIRED'] if m not in banned]}", "WARNING")

        # Step 2.5: 执行文献调研任务
        for issue in issues:
            if issue.get("type") == "LITERATURE_REQUIRED" and issue.get("status", "pending") == "pending":
                self.log_step(f"Literature: {issue.get('id')} - {issue.get('title', '')[:30]}", "progress")
                issue["status"] = "in_progress"
                self._save_action_plan(action_plan)

                # 提取搜索主题
                description = issue.get("description", "")
                search_task = f"""
文献调研任务: {issue.get('title')}

描述: {description}

请：
1. 使用 WebSearch 搜索相关论文和文档
2. 使用 WebFetch 获取具体内容
3. 提取关键信息
4. 更新 auto_research/state/literature.yaml
"""
                self.run_agent("literature", search_task, timeout=1800)
                issue["status"] = "completed"
                self._save_action_plan(action_plan)

        # Step 3: 执行实验类任务（inner loop）
        for issue in issues:
            if issue.get("type") == "EXPERIMENT_REQUIRED" and issue.get("status", "pending") == "pending":
                self.log_step(f"Experiment: {issue.get('id')} - {issue.get('title', '')[:30]}", "progress")
                self._run_experiment_inner_loop(issue, action_plan)

        # Step 4: 检查所有实验是否完成
        all_experiments_done = all(
            issue.get("status") in ["completed", "failed", "skipped"]
            for issue in issues
            if issue.get("type") == "EXPERIMENT_REQUIRED"
        )

        if all_experiments_done:
            self.log_step("Experiments done, starting writing phase...", "progress")
            self._run_writing_phase(action_plan)
            return True

        self.log_step("Some experiments incomplete", "warning")
        return False

    def _run_experiment_inner_loop(self, issue: dict, action_plan: dict):
        """执行单个实验类 issue 的内部迭代

        流程:
        1. Experimenter 设计并运行实验
        2. 等待 Slurm 作业完成
        3. Researcher 分析结果
        4. Planner 评估是否满意
        5. 如果不满意且未超过最大迭代次数，重复

        Args:
            issue: issue 字典
            action_plan: 完整的 action plan（用于保存状态）
        """
        import json

        max_loops = issue.get("max_inner_loops", 3)
        issue["inner_loop_count"] = issue.get("inner_loop_count", 0)
        issue["status"] = "in_progress"
        self._save_action_plan(action_plan)

        # 获取 experimenter 任务
        exp_task = issue.get("description", issue.get("title", "未知任务"))
        actions = issue.get("actions", [])
        if actions:
            for action in actions:
                if action.get("agent") == "experimenter":
                    exp_task = action.get("task", exp_task)
                    break

        while issue["inner_loop_count"] < max_loops:
            issue["inner_loop_count"] += 1
            loop_num = issue["inner_loop_count"]
            self.log_step(f"Inner Loop {loop_num}/{max_loops}: {issue.get('id')}", "progress")

            # 1. Experimenter 设计并运行实验
            exp_output = self.run_agent("experimenter", f"""
任务: {exp_task}

Issue 背景:
- ID: {issue.get('id')}
- 标题: {issue.get('title')}
- 描述: {issue.get('description', 'N/A')}

这是 Inner Loop 第 {loop_num} 次尝试（最多 {max_loops} 次）。
请设计并提交实验，使用 Slurm 运行 GPU 任务。
""", timeout=1800)

            # 2. 等待 Slurm 作业完成
            self.log_step("Waiting for Slurm jobs...", "info")
            self._wait_for_slurm_jobs(max_wait_hours=2)

            # 3. Researcher 分析结果
            research_output = self.run_agent("researcher", f"""
分析实验结果，判断是否满足审稿要求。

Issue: {issue.get('id')} - {issue.get('title')}
任务: {exp_task}

请检查:
1. 实验是否成功完成（无 OOM、无错误）
2. 结果文件是否生成
3. 数据是否支持论文论点
4. 更新 findings.yaml 记录新发现

给出明确结论：实验是否满足要求。
""", timeout=1200)

            # 4. Planner 评估是否满意
            eval_output = self.run_agent("planner", f"""
评估 Inner Loop 结果，判断是否可以进入下一阶段。

## Issue 信息
- ID: {issue.get('id')}
- 标题: {issue.get('title')}
- 当前尝试: {loop_num}/{max_loops}

## Experimenter 输出摘要
{exp_output[:800] if exp_output else '无输出'}

## Researcher 分析摘要
{research_output[:800] if research_output else '无输出'}

## 请判断
输出 JSON 格式的评估结果（只输出 JSON，不要其他内容）：

```json
{{
  "satisfied": true/false,
  "experiment_success": true/false,
  "supports_paper": true/false,
  "reason": "判断理由",
  "next_action": "proceed_to_writing" | "retry_experiment" | "mark_failed"
}}
```
""", timeout=600)

            # 解析 Planner 评估结果
            satisfied = False
            try:
                # 尝试从输出中提取 JSON
                json_match = re.search(r'\{[^{}]*"satisfied"[^{}]*\}', eval_output, re.DOTALL)
                if json_match:
                    eval_json = json.loads(json_match.group())
                    satisfied = eval_json.get("satisfied", False)
                else:
                    # 简单的关键词判断
                    satisfied = '"satisfied": true' in eval_output.lower() or '"satisfied":true' in eval_output.lower()
            except json.JSONDecodeError:
                satisfied = "satisfied.*true" in eval_output.lower() or "实验成功" in eval_output

            if satisfied:
                issue["status"] = "completed"
                self._save_action_plan(action_plan)
                self.log_step(f"Issue {issue.get('id')} completed", "success")
                self._generate_figures_from_results()
                return

            self.log_step(f"Loop {loop_num} not satisfied, retrying...", "warning")

        # 超过最大迭代次数
        issue["status"] = "failed"
        issue["failure_reason"] = f"Exceeded max loops {max_loops}"
        self._save_action_plan(action_plan)
        self.log_step(f"{issue.get('id')} failed after {max_loops} attempts", "error")

    def _run_writing_phase(self, action_plan: dict):
        """执行写作阶段

        将所有 WRITING_ONLY 任务和已完成实验的写作任务交给 Writer 处理。

        Args:
            action_plan: action plan 字典
        """
        issues = action_plan.get("issues", [])

        # 收集所有写作任务
        writing_tasks = []

        for issue in issues:
            if issue.get("type") == "WRITING_ONLY":
                writing_tasks.append({
                    "id": issue.get("id"),
                    "title": issue.get("title"),
                    "description": issue.get("description", ""),
                    "actions": issue.get("actions", []),
                })
                issue["status"] = "in_progress"
            elif issue.get("type") == "EXPERIMENT_REQUIRED" and issue.get("status") == "completed":
                # 实验完成后的写作任务
                for action in issue.get("actions", []):
                    if action.get("agent") == "writer":
                        writing_tasks.append({
                            "id": issue.get("id"),
                            "title": f"整合 {issue.get('title')} 实验结果",
                            "description": action.get("task", ""),
                            "actions": [action],
                        })
            elif issue.get("type") == "LITERATURE_REQUIRED" and issue.get("status") == "completed":
                # 文献调研完成后的写作任务（将文献内容写入 LaTeX）
                for action in issue.get("actions", []):
                    if action.get("agent") == "writer":
                        writing_tasks.append({
                            "id": issue.get("id"),
                            "title": f"根据文献调研更新 {issue.get('title')}",
                            "description": action.get("task", ""),
                            "actions": [action],
                        })
                        self.log(f"  📚 添加文献写作任务: {issue.get('id')} - 将文献内容写入 LaTeX", "INFO")

        if not writing_tasks:
            self.log_step("No writing tasks to execute", "info")
            return

        # 构建写作任务描述
        task_list = []
        for i, task in enumerate(writing_tasks, 1):
            task_list.append(f"""
### 任务 {i}: {task['id']} - {task['title']}
{task['description']}
""")

        # 分类任务：FIGURE_CODE_REQUIRED 和 WRITING_ONLY
        figure_tasks = [i for i in issues if i.get("type") == "FIGURE_CODE_REQUIRED"]
        pure_writing_tasks = [i for i in issues if i.get("type") == "WRITING_ONLY"]

        # ========== 阶段 1: 处理 FIGURE_CODE_REQUIRED 任务 ==========
        if figure_tasks:
            self.log_step(f"Processing {len(figure_tasks)} FIGURE_CODE_REQUIRED tasks (modifying Python code)", "progress")

            # 构建详细的 Python 代码修改指令
            figure_instructions = []
            for task in figure_tasks:
                task_id = task.get("id", "")
                task_title = task.get("title", "")
                task_desc = task.get("description", "")
                actions = task.get("actions", [])

                # 提取关键信息
                target_file = "scripts/create_paper_figures.py"
                target_function = ""
                modification = task_desc

                for action in actions:
                    if "target_file" in action:
                        target_file = action["target_file"]
                    if "target_function" in action:
                        target_function = action["target_function"]
                    if "modification" in action:
                        modification = action["modification"]

                figure_instructions.append(f"""
### {task_id}: {task_title}

**CRITICAL**: 你必须修改 Python 代码，不是 LaTeX！

**Target file**: {target_file}
**Target function**: {target_function if target_function else "见任务描述"}
**Modification needed**: {modification}

**Detailed task**:
{task_desc}

**Action steps**:
1. 使用 Read 工具阅读 {target_file}
2. 找到对应的函数（如 fig3_palu_distribution）
3. 使用 Edit 工具修改 matplotlib 参数（figsize, fontsize, tight_layout 等）
4. 确认修改后的代码语法正确
5. 不要运行脚本（系统会自动运行）
""")

            # 调用 writer agent 修改 Python 代码
            self.run_agent("writer", f"""
## CRITICAL TASK: 修改 Python 绘图脚本（不是 LaTeX！）

你收到了 {len(figure_tasks)} 个 FIGURE_CODE_REQUIRED 任务。
这些任务需要修改 **Python 代码**，而不是 LaTeX 文件！

{''.join(figure_instructions)}

## 工具使用要求
- 必须使用 Read 工具读取 scripts/create_paper_figures.py
- 必须使用 Edit 工具修改代码
- 不要使用 Bash 运行脚本（系统会自动运行）

## 函数名参考
- Figure 1: fig1_overview
- Figure 2: fig2_sdpa_latency
- Figure 3: fig3_palu_distribution
- Figure 4: fig4_root_cause
- Figure 5: fig5_repair_tradeoff
- Figure 6: fig6_e2e_performance

## 验证
修改完成后，确保：
1. Python 语法正确（没有缩进错误、括号匹配）
2. matplotlib 参数合理（figsize, fontsize, etc.）
3. 文件路径正确（Latex/figures/*.pdf）
""", timeout=1800)

            # 验证 Python 脚本是否真的被修改
            self.log_step("Verifying Python code changes...", "progress")
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "scripts/create_paper_figures.py"],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_DIR
                )
                if "scripts/create_paper_figures.py" in result.stdout:
                    self.log_step("Python code modified successfully", "success")
                else:
                    self.log_step("WARNING: FIGURE_CODE_REQUIRED task but Python not modified!", "error")
                    self.log("This indicates Writer agent did not correctly execute the task.", "WARN")
            except Exception as e:
                self.log(f"Verification failed: {e}", "WARN")

            # 运行修改后的脚本生成新图表
            self.log_step("Regenerating figures with modified code...", "progress")
            try:
                result = subprocess.run(
                    ["python", "scripts/create_paper_figures.py"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=PROJECT_DIR
                )
                if result.returncode == 0:
                    self.log_step("Figures regenerated successfully", "success")
                    # 重新编译 LaTeX 以更新 PDF
                    self.compile_latex()
                else:
                    self.log_step(f"Figure generation failed: {result.stderr[:200]}", "error")
            except Exception as e:
                self.log_step(f"Figure generation error: {e}", "error")

        # ========== 阶段 2: 处理写作任务（分优先级执行）==========
        # 将写作任务分为高优先级（逐个执行）和普通（可批量）
        high_priority_tasks = []
        batch_tasks = []

        for task in writing_tasks:
            task_id = task.get("id", "")
            title = (task.get("title", "") or "").lower()
            # Major issues 或 Related Work/Literature 任务 → 单独执行
            if (task_id.startswith("M") or
                "related work" in title or "文献" in title or
                "citation" in title or "引用" in title):
                high_priority_tasks.append(task)
            else:
                batch_tasks.append(task)

        self.log_step(f"Writing tasks: {len(high_priority_tasks)} high-priority (individual), {len(batch_tasks)} batch", "progress")

        # 阶段 2A: 逐个执行高优先级任务
        for task in high_priority_tasks:
            self.log_step(f"Writing (individual): {task['id']} - {task.get('title', '')[:40]}", "progress")

            # 获取文献上下文（如果有）
            literature_context = self._get_literature_context_for_task(task)

            self.run_agent("writer", f"""
你只有一个任务要完成。请认真、彻底地完成它。

## 任务: {task['id']} - {task.get('title', '')}
{task.get('description', '')}
{literature_context}

## 要求
1. 仔细阅读 Latex/main.tex 中的相关部分
2. 做出实质性修改（不是小调整）
3. 确保修改后 LaTeX 语法正确
4. 保持论文核心贡献不变
5. 参考 paper_example/ 目录的高质量论文风格

## 参考文件
- auto_research/state/latest_review.md - 审稿报告
- auto_research/state/literature.yaml - 文献调研结果
- Latex/references.bib - 参考文献库
""", timeout=1800)

            # 验证: 检查 main.tex 是否真的被修改
            try:
                diff_output = subprocess.run(
                    ["git", "diff", "--stat", "Latex/main.tex"],
                    capture_output=True, text=True, cwd=PROJECT_DIR
                ).stdout
                if diff_output.strip():
                    self.log_step(f"  ✓ {task['id']}: changes detected", "success")
                else:
                    self.log_step(f"  ✗ {task['id']}: No changes detected! Task may have failed.", "error")
            except Exception as e:
                self.log(f"  Verification error: {e}", "WARN")

        # 阶段 2B: 批量执行普通任务
        if batch_tasks:
            self.log_step(f"Processing {len(batch_tasks)} batch writing tasks", "progress")

            task_list = []
            for idx, task in enumerate(batch_tasks, 1):
                task_list.append(f"""
### Task {idx}: {task.get('id', '')} - {task.get('title', '')}
{task.get('description', '')}
""")

            self.run_agent("writer", f"""
请根据以下审稿修改任务，更新论文 Latex/main.tex。

## 修改任务列表

{''.join(task_list)}

## 注意事项
1. 每个修改后确保 LaTeX 语法正确
2. 保持论文核心贡献不变
3. 参考 paper_example/ 目录的高质量论文风格

## 参考文件
- auto_research/state/latest_review.md - 审稿报告
- auto_research/state/action_plan.yaml - 完整任务列表
""", timeout=3600)

        # 写作质量验证: 检查总修改量
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--numstat", "Latex/main.tex"],
                capture_output=True, text=True, cwd=PROJECT_DIR
            )
            total_added = 0
            if diff_result.stdout.strip():
                for line in diff_result.stdout.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        total_added += int(parts[0])

            if total_added < 5:
                self.log(f"⚠️ 写作阶段仅添加 {total_added} 行，修改可能不充分", "WARN")
            else:
                self.log(f"写作阶段共添加 {total_added} 行", "INFO")
        except Exception as e:
            self.log(f"写作质量验证失败: {e}", "WARN")

        # 标记写作任务完成
        for issue in issues:
            if issue.get("type") in ["WRITING_ONLY", "FIGURE_CODE_REQUIRED"]:
                issue["status"] = "completed"

        self._save_action_plan(action_plan)
        self.log_step("Writing phase completed", "success")

    def _get_literature_context_for_task(self, task: dict) -> str:
        """读取 literature.yaml，提取与当前任务相关的文献内容

        Args:
            task: 写作任务字典

        Returns:
            格式化的文献上下文字符串，如果无文献则返回空字符串
        """
        lit_file = PROJECT_DIR / "auto_research" / "state" / "literature.yaml"
        if not lit_file.exists():
            return ""

        try:
            lit_data = yaml.safe_load(lit_file.read_text()) or {}

            # 尝试多种可能的 key（literature agent 输出格式可能不同）
            entries = lit_data.get("entries", [])
            if not entries:
                entries = lit_data.get("papers", [])
            if not entries:
                entries = lit_data.get("references", [])
            if not entries and isinstance(lit_data, list):
                entries = lit_data

            if not entries:
                # 如果是嵌套结构，尝试提取所有文献
                for key, val in lit_data.items():
                    if isinstance(val, list) and len(val) > 0:
                        entries = val
                        break

            if not entries:
                return ""

            # 构建文献上下文
            lines = ["\n## 可用的文献资源（来自 Literature Agent）\n"]
            for entry in entries[:20]:  # 最多 20 条
                if isinstance(entry, dict):
                    title = entry.get("title", "")
                    bibtex_key = entry.get("bibtex_key", entry.get("key", entry.get("cite_key", "")))
                    summary = entry.get("summary", entry.get("abstract", ""))
                    if title:
                        cite_str = f" (cite: \\cite{{{bibtex_key}}})" if bibtex_key else ""
                        lines.append(f"- **{title}**{cite_str}")
                        if summary:
                            lines.append(f"  {summary[:200]}")
                elif isinstance(entry, str):
                    lines.append(f"- {entry}")

            if len(lines) > 1:
                lines.append("\n请在 Related Work 和其他适当位置引用这些文献。")
                lines.append("BibTeX 条目已在 Latex/references.bib 中，请使用 \\cite{} 引用。")
                return "\n".join(lines)
            return ""
        except Exception:
            return ""

    def _get_bottleneck(self) -> str:
        """从 latest_review.md 提取瓶颈维度

        Returns:
            瓶颈维度名称，如 "Technical Quality", "Presentation" 等
            如果无法提取，返回 "Unknown"
        """
        review_path = Path("auto_research/state/latest_review.md")
        if not review_path.exists():
            return "Unknown"

        try:
            content = review_path.read_text()
            # 匹配 "主要瓶颈维度: Technical Quality" 或 "主要瓶颈: Technical"
            import re
            patterns = [
                r"主要瓶颈维度[：:]\s*([^\n]+)",
                r"主要瓶颈[：:]\s*([^\n]+)",
                r"\*\*主要瓶颈\*\*[：:]\s*([^\n]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    bottleneck = match.group(1).strip()
                    # 清理可能的 markdown 格式
                    bottleneck = bottleneck.replace("*", "").strip()
                    return bottleneck
        except Exception as e:
            self.log(f"Failed to extract bottleneck: {e}", "WARN")

        return "Unknown"

    def decide_next_action(self) -> str:
        """根据当前状态动态决定下一步行动

        这是智能迭代的核心：不再是固定流程，而是根据状态选择。

        Returns:
            action: "RUN_EXPERIMENTS" | "TRIGGER_EXPERIMENT_PLANNING" |
                    "SELF_REPAIR" | "NORMAL_ITERATION"
        """
        # 读取状态
        score = self.memory.scores[-1] if self.memory.scores else 0
        action_plan = self._load_action_plan()
        bottleneck = self._get_bottleneck()

        self.log(f"Decision context: score={score}, bottleneck={bottleneck}, "
                 f"stagnation={self.memory.stagnation_count}", "DECISION")

        # 检查是否有待执行的实验
        issues = action_plan.get("issues", [])
        pending_experiments = [
            t for t in issues
            if t.get("type") == "EXPERIMENT_REQUIRED"
            and t.get("status") in ("pending", "in_progress")
        ]

        if pending_experiments:
            self.log(f"Found {len(pending_experiments)} pending experiments", "DECISION")
            return "RUN_EXPERIMENTS"

        # 检查瓶颈是否需要实验突破
        if "Technical" in bottleneck and score < 7.5:
            # 检查是否已有实验任务
            all_writing = all(
                t.get("type") == "WRITING_ONLY"
                for t in issues if t.get("status") != "completed"
            )
            if all_writing:
                self.log("Technical Quality bottleneck with no experiments - forcing experiment planning", "DECISION")
                return "TRIGGER_EXPERIMENT_PLANNING"

        # 检查是否停滞（使用 stagnation_count，不用 variance 条件）
        # stagnation_count >= 5: 触发 self_repair
        # stagnation_count >= 3: 由 Meta-Debugger 处理（在迭代末尾检查）
        if self.memory.stagnation_count >= 5:
            self.log(f"Stagnation detected ({self.memory.stagnation_count} iterations)", "DECISION")
            return "SELF_REPAIR"

        self.log("Normal iteration", "DECISION")
        return "NORMAL_ITERATION"

    def _force_experiment_planning(self):
        """强制 Planner 添加实验任务

        当 Technical Quality 是瓶颈但没有实验任务时调用。
        """
        self.log("Technical Quality 瓶颈，强制添加实验任务", "FORCE")

        bottleneck = self._get_bottleneck()
        current_score = self.memory.scores[-1] if self.memory.scores else 0

        prompt = f"""
## 强制实验规划模式

**触发原因**: Technical Quality 是瓶颈 (当前分数: {current_score}/10)，但当前没有实验任务。

**问题诊断**:
瓶颈维度: {bottleneck}
纯写作修改无法突破 Technical Quality 瓶颈。

**必须执行**:

1. 读取 auto_research/state/findings.yaml，检查有哪些实验可以做
2. 读取 auto_research/state/action_plan.yaml，了解当前任务状态
3. 检查以下选项：
   - 是否可以用 RAP SVD（不强制对齐）来验证 dimension repair？
   - 是否需要补充 perplexity/accuracy 数据？
   - 是否需要 E2E 验证？

4. **必须输出**：更新 action_plan.yaml，添加至少一个 EXPERIMENT_REQUIRED 任务

**注意**：
- 如果 PaLU 因为 100% 对齐无法验证 repair，使用 RAP SVD
- 实验任务必须是具体可执行的（包含命令、参数）
"""

        self.run_agent("planner", prompt)
        self.log("强制实验规划完成", "FORCE")

    # ==================== Meta-Debugger 集成 ====================

    def run_meta_debugger(self, trigger_reason: str) -> str:
        """运行 Meta-Debugger 进行系统诊断和修复

        Args:
            trigger_reason: 触发原因

        Returns:
            "CONTINUE" - 继续正常迭代
            "CONTINUE_WITH_FIX" - 已修复问题，继续迭代
            "PAUSE" - 发现严重问题，暂停迭代
        """
        self.log(f"🔍 触发 Meta-Debugger: {trigger_reason}", "META")

        # 获取诊断上下文
        diagnosis_ctx = self.memory.get_diagnosis_context()
        health_status, health_reasons = self.memory.get_health_status()

        # 构建诊断 prompt
        ctx_summary = f"""
## 触发原因
{trigger_reason}

## 系统健康状态
状态: **{health_status}**
{"原因: " + ", ".join(health_reasons) if health_reasons else "无异常"}

## 分数趋势
- 当前: {diagnosis_ctx['scores']['current']}/10
- 最高: {diagnosis_ctx['scores']['best']}/10
- 趋势: {diagnosis_ctx['scores']['trend']}
- 最近: {' → '.join(f"{s:.1f}" for s in diagnosis_ctx['scores']['recent'][-5:])}

## 停滞状态
- 停滞计数: {diagnosis_ctx['stagnation']['count']}
- 是否停滞: {diagnosis_ctx['stagnation']['is_stagnating']}
- 原因: {diagnosis_ctx['stagnation']['reason']}

## Issue 重复情况
- 高重复 (7+次): {diagnosis_ctx['issues']['high_repeat']}
- 中重复 (3+次): {diagnosis_ctx['issues']['repeat_issues'][:5]}

## 实验空转
- 空转次数: {diagnosis_ctx['experiment_empty_count']}
"""

        diagnosis_output = self.run_agent("meta_debugger", f"""
{ctx_summary}

请执行完整的系统诊断：

1. **读取关键状态文件**:
   - auto_research/state/memory.yaml
   - auto_research/state/action_plan.yaml
   - auto_research/state/latest_review.md

2. **分析最近的执行日志** (查看 auto_research/logs/ 目录)

3. **检查执行一致性**:
   - 运行 `git diff scripts/create_paper_figures.py` 检查 FIGURE_CODE 任务
   - 运行 `git status` 检查修改了哪些文件

4. **识别问题模式**:
   - 是否有"计划正确但执行失败"的情况？
   - 是否陷入了"方法循环"？
   - 是否有策略升级规则被违反？

5. **生成诊断报告** 到 auto_research/state/meta_diagnosis.md

6. **如果发现框架 bug，直接修复**:
   - 修改 auto_research/memory.py
   - 修改 auto_research/orchestrator.py
   - 修改 auto_research/agents/*.prompt
   - 重置 memory.yaml 中的错误累积

**重要**: 诊断要找到根因，不只是症状。修复要具体，不只是建议。
""", timeout=1800)  # 30 分钟超时

        # 读取诊断报告并解析状态
        # 注意: Meta-Debugger 永远不会导致 PAUSE。它的职责是诊断和修复，
        # 不是停止系统。即使报告中包含 "CRITICAL" 或 "需要人工" 等字样，
        # 这只是诊断描述，不应阻止迭代继续。
        diagnosis_file = PROJECT_DIR / "auto_research/state/meta_diagnosis.md"
        if diagnosis_file.exists():
            try:
                diagnosis_file.read_text()  # 确认文件可读
                self.log("⚠️ Meta-Debugger 已完成诊断，继续迭代", "WARNING")
                # 重新加载 memory 以获取可能的修复
                self.memory.load()
                return "CONTINUE_WITH_FIX"
            except Exception as e:
                self.log(f"读取诊断报告失败: {e}", "ERROR")

        return "CONTINUE"

    def check_and_trigger_meta_debug(self) -> str:
        """检查是否应该触发 Meta-Debugger，如果是则运行

        Returns:
            "CONTINUE" - 继续正常迭代
            "CONTINUE_WITH_FIX" - 已修复问题，继续迭代
            "PAUSE" - 发现严重问题，暂停迭代
        """
        should_trigger, reason = self.memory.should_trigger_meta_debug()
        if should_trigger:
            return self.run_meta_debugger(reason)
        return "CONTINUE"

    def self_repair(self, stagnation_reason: str) -> bool:
        """自我修复：停滞时自动让 Planner 重新规划策略

        增强版：包含元反思检查 - 如果上次修复无效，升级策略而不是重复同样的方法。

        Args:
            stagnation_reason: 停滞原因描述

        Returns:
            True if repair was attempted
        """
        self.log("启动自我修复流程...", "REPAIR")

        # 获取当前状态信息
        current_score = self.memory.scores[-1] if self.memory.scores else 0.0
        stagnation_count = self.memory.stagnation_count
        bottleneck = self._get_bottleneck()

        # 元反思检查：上次修复是否有效？
        last_repair_effective, ineffective_reason = self.memory.was_last_repair_effective()
        repeat_issues = self.memory.get_repeat_issues(threshold=3)

        self.log(f"瓶颈维度: {bottleneck}", "REPAIR")
        if not last_repair_effective:
            self.log(f"⚠️ 元反思警告: 上次修复无效 - {ineffective_reason}", "REPAIR")
        if repeat_issues:
            self.log(f"⚠️ 重复 Issues: {[r[0] for r in repeat_issues]}", "REPAIR")

        # 构建修复 prompt（带元反思和瓶颈感知）
        meta_reflection = ""
        if not last_repair_effective or repeat_issues:
            meta_reflection = f"""
### ⚠️ 元反思警告 - 必须换方法！

**上次修复无效**: {ineffective_reason}

**重复出现的 Issues**（这些问题已经尝试修复多次但仍未解决）:
"""
            for issue_id, count in repeat_issues:
                meta_reflection += f"- **{issue_id}**: 出现 {count} 次\n"

            meta_reflection += """
**强制要求**：
1. 不要使用与之前相同的修复方法
2. 分析为什么之前的修复无效
3. 采用完全不同的策略：
   - 如果之前是改文字 → 现在要跑实验
   - 如果之前是调格式 → 现在要改 Python 代码重新生成图表
   - 如果图表问题反复出现 → 必须使用 FIGURE_CODE_REQUIRED（修改绘图脚本并重新运行）
"""

        repair_prompt = f"""
## 自我修复模式 - 瓶颈突破

**停滞原因**: {stagnation_reason}
**当前分数**: {current_score}/10
**目标分数**: {PAPER_ACCEPT_THRESHOLD}/10
**连续停滞次数**: {stagnation_count}
**瓶颈维度**: {bottleneck}
{meta_reflection}

### 瓶颈分析

根据瓶颈维度决定修复策略：

| 瓶颈 | 策略 |
|------|------|
| Technical Quality | **必须添加实验任务**（纯写作无法突破） |
| Presentation | 重点改进图表（使用 FIGURE_CODE_REQUIRED）|
| Innovation | 重新 frame contribution |
| Writing Quality | 重写关键章节 |

### 必须执行的步骤

1. 读取 auto_research/state/action_plan.yaml，分析当前任务结构
2. 读取 auto_research/state/latest_review.md，了解 Reviewer 的具体建议
3. 检查重复 Issues，分析之前的修复为什么无效

### 强制要求

- 如果瓶颈是 **Technical Quality** 且 < 7.5：
  - 必须添加 EXPERIMENT_REQUIRED 任务

- 如果瓶颈是 **Presentation** 或图表相关 Issues 重复出现：
  - 必须使用 FIGURE_CODE_REQUIRED（不是 WRITING_ONLY！）
  - 必须修改 scripts/create_paper_figures.py
  - 必须运行脚本重新生成图表

- 如果所有任务都是 WRITING_ONLY：
  - 这是危险信号，必须添加实验或 FIGURE_CODE 任务

**重要**：这是自动触发的修复流程，你需要自主决策，不要等待人工指示。

**输出**：更新 auto_research/state/action_plan.yaml，确保有针对 {bottleneck} 瓶颈的突破任务。
"""

        # 调用 Planner 进行自我修复
        self.run_agent("planner", repair_prompt)

        # 标记这次修复尝试（用于下次验证）
        self.memory.mark_repair_attempt(self.iteration)

        # 修复后不重置停滞计数 — 让 Meta-Debugger 有机会被触发
        # self_repair 本身就是一种"尝试"，如果连续修复都无效，
        # stagnation_count 会继续累积，最终触发更强力的 Meta-Debugger
        if last_repair_effective:
            # 修复有效时才减少计数
            self.memory.stagnation_count = max(0, stagnation_count - 3)
            self.log(f"修复有效，停滞计数减少3: {stagnation_count} → {self.memory.stagnation_count}", "REPAIR")
        else:
            # 修复无效时不减少计数，让它继续累积
            self.log(f"上次修复无效，停滞计数保持不变: {stagnation_count}", "REPAIR")

        self.memory.save()

        self.log("自我修复完成", "REPAIR")
        return True

    def run_paper_iteration(self) -> bool:
        """执行论文审稿迭代，返回是否应该继续"""
        self.iteration += 1
        paper_state = self.load_paper_state()
        paper_requirements = self.load_paper_requirements()
        current_score = paper_state.get("current_score", 0)

        # 迭代标题
        self.log("", "RAW")
        self.log_section(f"ITERATION {self.iteration}/{self.max_iterations}  |  Score: {current_score}/10 → ?  |  Target: {PAPER_ACCEPT_THRESHOLD}/10")

        # 阶段计数（根据是否首次迭代动态调整）
        total_phases = 6 if self.iteration == 1 else 5
        phase_num = 0

        # 0. 首次迭代：生成图表并设置基本结构
        if self.iteration == 1:
            phase_num += 1
            self.log_phase(phase_num, total_phases, "Initialize (first run)")
            self.log_step("Generating paper figures...", "progress")
            self.generate_figures()

            # 让 writer 完成初始设置（作者信息、图表引用等）
            req_summary = yaml.dump(paper_requirements, allow_unicode=True) if paper_requirements else "无特殊需求"
            self.run_agent(
                "writer",
                f"""这是论文审稿的第一次迭代。请完成以下初始化工作：

1. 更新 Latex/main.tex 中的作者信息（根据 paper_requirements.yaml）
2. 确保所有生成的图表（Latex/figures/fig*.pdf）都正确引用到论文中
3. 检查论文结构是否完整
4. 确保符合 EuroMLSys SIGPLAN 双栏格式（正文六页，引用和附录不限）

论文需求配置：
```yaml
{req_summary}
```

参考 paper_example/ 目录中的高质量论文。""",
                timeout=3600,
            )
            self.log_phase(phase_num, total_phases, "Initialize (first run)", "end")

        # 1. 编译当前版本的 LaTeX
        phase_num += 1
        self.log_phase(phase_num, total_phases, "Compile LaTeX")
        self.log_step("Compiling LaTeX...", "progress")
        if not self.compile_latex():
            self.log_step("LaTeX compilation failed, fixing...", "warning")
            self.run_agent("writer", "LaTeX 编译失败，请修复语法错误并确保能正确编译")
            if not self.compile_latex():
                self.log_step("Still cannot compile, skipping iteration", "error")
                self.log_phase(phase_num, total_phases, "Compile LaTeX", "end")
                return True
        self.log_step("PDF generated successfully", "success")
        self.log_phase(phase_num, total_phases, "Compile LaTeX", "end")

        # 2. 转换 PDF 为图像用于视觉审核
        page_images = self.pdf_to_images()
        visual_review_section = ""
        if page_images:
            visual_review_section = f"""

## 视觉审核

请使用 Read 工具读取以下论文页面图像，进行视觉效果审核：
{chr(10).join(f'- {img}' for img in page_images)}

重点检查：
- 图表大小是否合适，字体是否清晰可读
- 排版是否专业（对齐、间距、边距）
- 信息密度是否适中
- 整体视觉效果是否达到 research publication 水平
"""

        # 3. Reviewer Agent 审稿
        phase_num += 1
        self.log_phase(phase_num, total_phases, "Review Paper")
        review_output = self.run_agent(
            "reviewer",
            f"""请审阅当前论文 Latex/main.tex 和生成的 Latex/main.pdf。

根据 EuroMLSys 标准进行评审：
- 技术质量 (40%)
- 论文呈现 (30%)
- 创新性 (20%)
- 写作质量 (10%)
{visual_review_section}
输出详细的审稿报告，包括：
1. 总体评分 (X/10)
2. 各项评分
3. Major Issues（必须修改）
4. Minor Issues（建议修改）
5. 具体改进建议

将审稿报告保存到 auto_research/state/latest_review.md""",
            timeout=2400,  # 40 分钟
        )

        # 解析评分
        score = self.parse_review_score(review_output)
        score_delta = score - current_score
        delta_str = f"+{score_delta:.1f}" if score_delta >= 0 else f"{score_delta:.1f}"
        self.log_step(f"Score: {score}/10 ({delta_str} from last)", "success" if score_delta >= 0 else "warning")

        # 记录 issues 用于重复追踪（自检机制）
        issue_ids = self.extract_issue_ids()
        self.memory.record_issues(issue_ids, self.iteration)

        # 检查是否有重复出现的问题（自检警告）
        repeat_issues = self.memory.get_repeat_issues(threshold=3)
        if repeat_issues:
            self.log("⚠️ 自检警告：以下 issues 重复出现 3+ 次，说明之前的修复无效！", "WARN")
            for issue_id, count in repeat_issues:
                self.log(f"  - {issue_id}: 出现 {count} 次", "WARN")
            self.log("建议：需要换一种完全不同的方法", "WARN")

        self.log_phase(phase_num, total_phases, "Review Paper", "end")

        # 更新论文状态
        paper_state["reviews"].append({
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "log": str(self.log_file),
        })
        paper_state["current_score"] = score

        # 检查是否达标
        if score >= PAPER_ACCEPT_THRESHOLD:
            self.log("", "RAW")
            self.log_section(f"PAPER ACCEPTED!  Score: {score}/10 >= {PAPER_ACCEPT_THRESHOLD}/10", "★")
            paper_state["status"] = "accepted"
            self.save_paper_state(paper_state)
            self._last_score = score
            self.git_commit(f"ACCEPTED: Final score {score}/10")
            self.send_notification(
                "Paper Accepted",
                f"Paper achieved score {score}/10, meeting the threshold of {PAPER_ACCEPT_THRESHOLD}/10!"
            )
            return False

        # 动态流程选择器：根据当前状态决定下一步
        phase_num += 1
        self.log_phase(phase_num, total_phases, "Plan & Execute Improvements")

        next_action = self.decide_next_action()
        self.log(f"Dynamic decision: {next_action}", "DECISION")

        if next_action == "TRIGGER_EXPERIMENT_PLANNING":
            # Technical Quality 是瓶颈但没有实验任务，强制添加
            self._force_experiment_planning()

        elif next_action == "SELF_REPAIR":
            # 检测到停滞，触发自我修复
            is_stagnating, stagnation_reason = self.memory.is_stagnating()
            self.self_repair(stagnation_reason)

        # 无论哪种情况，都运行 planner cycle
        planner_success = self.run_planner_cycle(review_output)

        if not planner_success:
            # Planner 循环失败，回退到简单的 Writer 修改
            self.log_step("Planner cycle incomplete, using fallback", "warning")
            req_str = ""
            if paper_requirements:
                quality_reqs = paper_requirements.get("quality_requirements", [])
                if quality_reqs:
                    req_str = "\n\n关键质量要求：\n" + "\n".join(f"- {r}" for r in quality_reqs)

            self.run_agent(
                "writer",
                f"""请阅读最新的审稿报告 auto_research/state/latest_review.md，
根据审稿意见改进论文：

1. 首先处理所有 Major Issues
2. 然后处理 Minor Issues
3. 改进图表质量和信息密度
4. 确保符合 EuroMLSys 双栏格式（正文六页，引用和附录不限）

注意：
- 保持论文的核心贡献不变
- 每次改进后确保 LaTeX 能编译通过
- 更新 report.md 保持同步{req_str}""",
                timeout=3600,
            )
        self.log_phase(phase_num, total_phases, "Plan & Execute Improvements", "end")

        # 验证改进
        phase_num += 1
        self.log_phase(phase_num, total_phases, "Validate Changes")
        self.run_agent("validator", "验证论文改进是否符合审稿要求")
        self.log_phase(phase_num, total_phases, "Validate Changes", "end")

        self.save_paper_state(paper_state)
        self._last_score = score

        # 迭代总结
        self.log("", "RAW")
        gap = PAPER_ACCEPT_THRESHOLD - score
        status = "CONTINUE" if gap > 0 else "ACCEPTED"
        self.log_summary_box(f"Iteration {self.iteration} Summary", [
            f"Score: {score}/10 (target: {PAPER_ACCEPT_THRESHOLD}/10)",
            f"Gap: {gap:.1f} points remaining" if gap > 0 else "Target reached!",
            f"Status: {status}",
        ], inside_phase=False)

        # 8.5 记录到 Memory 系统（极简版：只记录分数）
        self.record_score_to_memory(score)

        # 9. 清理工作区
        self.cleanup_workspace()

        # 10. Git Commit（每轮迭代都提交）
        self.git_commit(f"Iteration {self.iteration}: score {score}/10")

        # 保存检查点
        self.save_checkpoint()

        # 11. 停滞检测与自我修复
        # 注意：只在 stagnation_count >= 5 时触发 self_repair
        # stagnation_count < 5 时由 Meta-Debugger 在步骤 12 处理
        is_stagnating, stagnation_reason = self.memory.is_stagnating()
        if is_stagnating:
            self.log(f"⚠️ 停滞检测: {stagnation_reason} (stagnation_count={self.memory.stagnation_count})", "WARN")

            if self.memory.stagnation_count >= 5:
                # 高停滞：触发自我修复
                self.log("触发自我修复...", "REPAIR")
                self.self_repair(stagnation_reason)
            else:
                # 低停滞：仅记录，交给 Meta-Debugger 处理
                self.log("停滞计数较低，交给 Meta-Debugger 处理", "WARN")

            # 发送通知
            if self.memory.stagnation_count >= 10:
                self.send_notification(
                    "AutoGAC 自我修复",
                    f"检测到停滞并已自动重新规划策略。\n"
                    f"原因: {stagnation_reason}\n"
                    f"当前评分: {score}/10"
                )

        # 12. Meta-Debugger 检查（更深层次的系统诊断）
        # 注意: Meta-Debugger 不再能停止迭代，只能诊断和修复
        meta_result = self.check_and_trigger_meta_debug()
        if meta_result == "CONTINUE_WITH_FIX":
            self.log("✅ Meta-Debugger 已修复系统问题", "META")

        return True

    def run_iteration(self) -> bool:
        """执行一次迭代，返回是否应该继续"""
        self.iteration += 1
        self.log(f"\n{'='*60}")
        self.log(f"Iteration {self.iteration} started at {datetime.now()}")
        self.log(f"{'='*60}\n")

        # 1. 加载状态
        state = self.load_state()
        phase = self.get_current_phase(state)
        self.log(f"Current phase: {phase}")

        if phase == "completed":
            self.log("All phases completed!")
            self.send_notification("Research Completed", "All research phases have been completed!")
            return False

        # 2. 根据阶段执行不同的 Agent 流程
        if phase == "C1_quantify":
            # C1: 补充量化实验
            self.run_agent("experimenter", "检查 C1 阶段是否需要更多实验，如果需要，设计并提交")
            self.wait_for_slurm()
            self.run_agent("researcher", "分析 C1 实验结果，判断是否足够量化维度坍塌现象")

        elif phase == "C2_probe":
            # C2: 原因探究（重点）
            sub_task = self._get_c2_subtask(state)
            self.log(f"C2 sub-task: {sub_task}")

            self.run_agent("experimenter", f"针对 C2 的 {sub_task}，设计并提交实验")
            self.wait_for_slurm()
            self.run_agent("researcher", f"分析 {sub_task} 的实验结果，验证或推翻假设")

        elif phase == "C3_formulate":
            # C3: 形式化问题
            self.run_agent("researcher", "基于 C2 发现，形式化 Shape Contract 定义")

        elif phase == "C4_solver":
            # C4: 实现解决方案
            self.run_agent("experimenter", "实现 dimension repair 算法并验证")

        elif phase == "C5_validation":
            # C5: 端到端验证
            self.run_agent("experimenter", "设计并运行端到端 LLM 推理对比实验")
            self.wait_for_slurm()
            self.run_agent("researcher", "分析端到端验证结果")

        # 3. 更新报告和论文
        self.run_agent("writer", "根据最新发现更新 report.md 和 Latex")

        # 4. 验证完成度
        validation_output = self.run_agent("validator", "检查当前研究完成度")

        # 5. 检查是否阶段完成，发送通知
        new_state = self.load_state()
        new_phase = self.get_current_phase(new_state)
        if new_phase != phase:
            self.send_notification(
                f"Phase {phase} Completed",
                f"Phase {phase} has been completed. Moving to {new_phase}."
            )

        # 6. 更新迭代记录
        new_state["current_iteration"]["number"] = self.iteration
        new_state["history"].append({
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "log": str(self.log_file),
        })
        self.save_state(new_state)

        self.log(f"Iteration {self.iteration} completed\n")

        # 休息一段时间（避免 API 限流和额度消耗过快）
        self.log("Waiting 5 minutes before next iteration...")
        time.sleep(300)  # 5 分钟间隔，降低 API 调用频率

        return True

    def _get_c2_subtask(self, state: dict) -> str:
        """获取 C2 阶段当前应该做的子任务"""
        c2 = state.get("phases", {}).get("C2_probe", {})
        sub_tasks = c2.get("sub_tasks", [])

        for task in sub_tasks:
            if task.get("status") in ["pending", "partial"]:
                return task.get("name", "unknown")

        return "all sub-tasks completed"

    def run(self):
        """主循环"""
        self.log_section(f"AutoGAC Started  |  Mode: {self.mode.upper()}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.log(f"Max iterations: {self.max_iterations}  |  Max time: {self.max_end_time.strftime('%Y-%m-%d %H:%M')}", "RAW")
        self.log(f"Log: {self.log_file}", "RAW")
        self.log("", "RAW")

        self.send_notification(
            f"{self.mode.capitalize()} Mode Started",
            f"Automated {self.mode} system started.\nMax iterations: {self.max_iterations}\nLog: {self.log_file}"
        )

        try:
            while (
                datetime.now() < self.max_end_time
                and self.iteration < self.max_iterations
            ):
                if self.mode == "paper":
                    should_continue = self.run_paper_iteration()
                else:
                    should_continue = self.run_iteration()

                if not should_continue:
                    break

        except KeyboardInterrupt:
            self.log("", "RAW")
            self.log_section("INTERRUPTED BY USER", "!")
        except Exception as e:
            self.log("", "RAW")
            self.log_section(f"ERROR: {str(e)[:50]}", "!")
            self.send_notification("Error", f"{self.mode.capitalize()} system encountered an error: {e}")
            raise

        # 结束摘要
        self.log("", "RAW")
        if self.mode == "paper":
            paper_state = self.load_paper_state()
            final_score = paper_state.get('current_score', 0)
            status = paper_state.get('status', 'unknown')
            self.log_section(f"AutoGAC Finished  |  Score: {final_score}/10  |  Status: {status.upper()}")
        else:
            self.log_section(f"AutoGAC Finished  |  Iterations: {self.iteration}")
        self.log(f"Total iterations: {self.iteration}", "RAW")


def main():
    parser = argparse.ArgumentParser(description="GAC Automated Research Orchestrator")
    parser.add_argument("--mode", type=str, default="research", choices=["research", "paper"],
                        help="Mode: 'research' for experiments, 'paper' for review iterations")
    parser.add_argument("--max-days", type=float, default=3, help="Maximum runtime in days")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations")
    args = parser.parse_args()

    orchestrator = Orchestrator(
        max_days=args.max_days,
        max_iterations=args.max_iterations,
        mode=args.mode,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
