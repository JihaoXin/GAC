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
from typing import Optional
import re

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_DIR)

# 配置
STATE_FILE = PROJECT_DIR / "auto_research" / "state" / "research_state.yaml"
FINDINGS_FILE = PROJECT_DIR / "auto_research" / "state" / "findings.yaml"
PAPER_STATE_FILE = PROJECT_DIR / "auto_research" / "state" / "paper_state.yaml"
LOG_DIR = PROJECT_DIR / "auto_research" / "logs"
AGENTS_DIR = PROJECT_DIR / "auto_research" / "agents"
LATEX_DIR = PROJECT_DIR / "Latex"

# 确保目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 论文审稿通过阈值
PAPER_ACCEPT_THRESHOLD = 8  # 总体评分 >= 8/10 则通过


class Orchestrator:
    """主控调度器"""

    def __init__(self, max_days: float = 3, max_iterations: int = 100, mode: str = "research"):
        self.max_end_time = datetime.now() + timedelta(days=max_days)
        self.max_iterations = max_iterations
        self.iteration = 0
        self.mode = mode  # "research" or "paper"
        self.log_file = LOG_DIR / f"orchestrator_{mode}_{datetime.now():%Y%m%d_%H%M%S}.log"

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

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
        """运行指定类型的 Agent"""
        prompt_file = AGENTS_DIR / f"{agent_type}.prompt"
        if not prompt_file.exists():
            self.log(f"Agent prompt not found: {prompt_file}")
            return ""

        base_prompt = prompt_file.read_text()

        full_prompt = f"""{base_prompt}

---

## 当前任务

{task}

## 上下文文件

请阅读以下文件获取上下文（如果存在）：
- auto_research/state/research_state.yaml - 当前研究状态
- auto_research/state/findings.yaml - 已有发现
- report.md - 研究报告
- results/ 目录 - 实验结果

执行任务并更新相应文件。
"""

        self.log(f"Running {agent_type} agent: {task[:50]}...")

        # 记录 agent 调用日志
        agent_log = LOG_DIR / f"agent_{agent_type}_{datetime.now():%Y%m%d_%H%M%S}.log"

        try:
            result = subprocess.run(
                [
                    "claude", "-p", full_prompt,
                    "--dangerously-skip-permissions",  # 跳过权限确认，自动化必需
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_DIR,
            )

            output = result.stdout + "\n" + result.stderr

            # 保存 agent 输出
            with open(agent_log, "w") as f:
                f.write(f"Task: {task}\n\n")
                f.write(f"Output:\n{output}\n")

            self.log(f"Agent {agent_type} completed. Log: {agent_log}")
            return output

        except subprocess.TimeoutExpired:
            self.log(f"Agent {agent_type} timed out after {timeout}s")
            return ""
        except Exception as e:
            self.log(f"Agent {agent_type} error: {e}")
            return ""

    def wait_for_slurm(self, max_wait_hours: float = 4) -> bool:
        """等待所有 Slurm 作业完成"""
        self.log("Checking Slurm jobs...")
        max_wait = timedelta(hours=max_wait_hours)
        start_time = datetime.now()

        while datetime.now() - start_time < max_wait:
            try:
                result = subprocess.run(
                    ["squeue", "-u", os.environ.get("USER", "xinj"), "-h"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                running_jobs = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

                if running_jobs == 0:
                    self.log("All Slurm jobs completed")
                    return True

                self.log(f"{running_jobs} jobs still running, waiting 60s...")
                time.sleep(60)

            except Exception as e:
                self.log(f"Error checking Slurm: {e}")
                time.sleep(60)

        self.log(f"Slurm wait timeout after {max_wait_hours} hours")
        return False

    def send_notification(self, subject: str, message: str):
        """发送邮件通知"""
        try:
            email = "jihao.xin@kaust.edu.sa"
            subprocess.run(
                ["mail", "-s", f"GAC: {subject}", email],
                input=message,
                text=True,
                timeout=30,
            )
            self.log(f"Notification sent: {subject}")
        except Exception as e:
            self.log(f"Failed to send notification: {e}")

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

    def save_paper_state(self, state: dict):
        """保存论文审稿状态"""
        with open(PAPER_STATE_FILE, "w") as f:
            yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

    def parse_review_score(self, review_output: str) -> int:
        """从审稿输出中解析总体评分"""
        # 寻找 "总体评分: X/10" 或 "Overall Score: X/10" 格式
        patterns = [
            r"总体评分[：:]\s*(\d+)/10",
            r"Overall Score[：:]\s*(\d+)/10",
            r"总分[：:]\s*(\d+)/10",
        ]
        for pattern in patterns:
            match = re.search(pattern, review_output)
            if match:
                return int(match.group(1))
        return 0

    def run_paper_iteration(self) -> bool:
        """执行论文审稿迭代，返回是否应该继续"""
        self.iteration += 1
        self.log(f"\n{'='*60}")
        self.log(f"Paper Review Iteration {self.iteration} started at {datetime.now()}")
        self.log(f"{'='*60}\n")

        paper_state = self.load_paper_state()

        # 1. 编译当前版本的 LaTeX
        if not self.compile_latex():
            self.log("LaTeX compilation failed, asking writer to fix...")
            self.run_agent("writer", "LaTeX 编译失败，请修复语法错误并确保能正确编译")
            if not self.compile_latex():
                self.log("Still cannot compile, skipping this iteration...")
                return True

        # 2. Reviewer Agent 审稿
        self.log("Running reviewer agent...")
        review_output = self.run_agent(
            "reviewer",
            """请审阅当前论文 Latex/main.tex 和生成的 Latex/main.pdf。

根据 EuroMLSys 标准进行评审：
- 技术质量 (40%)
- 论文呈现 (30%)
- 创新性 (20%)
- 写作质量 (10%)

输出详细的审稿报告，包括：
1. 总体评分 (X/10)
2. 各项评分
3. Major Issues（必须修改）
4. Minor Issues（建议修改）
5. 具体改进建议

将审稿报告保存到 auto_research/state/latest_review.md""",
            timeout=2400,  # 40 分钟
        )

        # 3. 解析评分
        score = self.parse_review_score(review_output)
        self.log(f"Review score: {score}/10")

        # 4. 更新论文状态
        paper_state["reviews"].append({
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "log": str(self.log_file),
        })
        paper_state["current_score"] = score

        # 5. 检查是否达标
        if score >= PAPER_ACCEPT_THRESHOLD:
            self.log(f"Paper accepted! Score {score}/10 >= threshold {PAPER_ACCEPT_THRESHOLD}/10")
            paper_state["status"] = "accepted"
            self.save_paper_state(paper_state)
            self.send_notification(
                "Paper Accepted",
                f"Paper achieved score {score}/10, meeting the threshold of {PAPER_ACCEPT_THRESHOLD}/10!"
            )
            return False

        # 6. Writer Agent 根据审稿意见改进
        self.log("Running writer agent to improve paper...")
        self.run_agent(
            "writer",
            """请阅读最新的审稿报告 auto_research/state/latest_review.md，
根据审稿意见改进论文：

1. 首先处理所有 Major Issues
2. 然后处理 Minor Issues
3. 改进图表质量和信息密度
4. 确保符合 EuroMLSys 双栏六页格式

注意：
- 保持论文的核心贡献不变
- 每次改进后确保 LaTeX 能编译通过
- 更新 report.md 保持同步""",
            timeout=3600,  # 60 分钟
        )

        # 7. 验证改进
        self.run_agent("validator", "验证论文改进是否符合审稿要求")

        self.save_paper_state(paper_state)
        self.log(f"Paper iteration {self.iteration} completed, score: {score}/10\n")

        # 休息
        self.log("Waiting 3 minutes before next iteration...")
        time.sleep(180)

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
        self.log("="*60)
        self.log(f"GAC Automated System Started - Mode: {self.mode.upper()}")
        self.log(f"Max end time: {self.max_end_time}")
        self.log(f"Max iterations: {self.max_iterations}")
        self.log(f"Log file: {self.log_file}")
        self.log("="*60 + "\n")

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
            self.log("\nInterrupted by user")
        except Exception as e:
            self.log(f"\nError: {e}")
            self.send_notification("Error", f"{self.mode.capitalize()} system encountered an error: {e}")
            raise

        self.log("\n" + "="*60)
        self.log(f"GAC Automated System Finished - Mode: {self.mode.upper()}")
        self.log(f"Total iterations: {self.iteration}")
        if self.mode == "paper":
            paper_state = self.load_paper_state()
            self.log(f"Final paper score: {paper_state.get('current_score', 0)}/10")
            self.log(f"Paper status: {paper_state.get('status', 'unknown')}")
        self.log("="*60)


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
