"""内心独白引擎——让Agent会'想'"""
import logging
import re
from typing import Optional

from cognition.types import CognitiveState, InnerThought, ThinkingMode

logger = logging.getLogger(__name__)


class InnerMonologueEngine:
    """生成和管理Agent的内心独白"""

    # 内心独白触发模式
    TRIGGER_ALWAYS = "always"      # 每次都想
    TRIGGER_COMPLEX = "complex"    # 复杂问题时想
    TRIGGER_NEVER = "never"        # 关闭

    def __init__(self, trigger_mode: str = TRIGGER_COMPLEX):
        self.trigger_mode = trigger_mode

    def should_think(self, agent_name: str, query: str, cognitive_state: CognitiveState) -> bool:
        """判断当前是否应该产生内心独白"""
        if self.trigger_mode == self.TRIGGER_NEVER:
            return False
        if self.trigger_mode == self.TRIGGER_ALWAYS:
            return True
        # COMPLEX模式：根据问题复杂度和agent状态判断
        complexity = self._estimate_complexity(query)
        emotional = cognitive_state.emotional
        # 复杂问题、agent不确定、或者好奇时产生独白
        return (
            complexity > 0.5
            or emotional.confidence < 0.5
            or emotional.curiosity > 0.7
            or "?" in query or "？" in query
            or len(query) > 50
        )

    def _estimate_complexity(self, query: str) -> float:
        """估计问题复杂度 0-1"""
        score = 0.0
        # 长度因子
        score += min(0.3, len(query) / 500)
        # 关键词因子
        complex_keywords = [
            "分析", "比较", "对比", "评估", "为什么", "如何", "如果",
            "analyze", "compare", "evaluate", "why", "how", "if",
            "区别", "差异", "影响", "原因", "机制",
            "difference", "impact", "reason", "mechanism",
        ]
        for kw in complex_keywords:
            if kw in query.lower():
                score += 0.1
        # 多问题因子
        question_marks = query.count("?") + query.count("？")
        score += min(0.2, question_marks * 0.1)
        return min(1.0, score)

    def generate_thought_prompt(self, agent_name: str, query: str,
                                context: str = "", mode: ThinkingMode = ThinkingMode.INTUITION) -> str:
        """生成用于触发内心独白的提示词片段"""
        mode_hints = {
            ThinkingMode.INTUITION: "凭直觉快速反应，第一反应是什么",
            ThinkingMode.REASONING: "仔细分析，分步骤思考",
            ThinkingMode.REFLECTIVE: "跳出问题本身，反思自己的思考过程",
        }
        hint = mode_hints.get(mode, mode_hints[ThinkingMode.INTUITION])

        return f"""
【内心独白】在正式回答之前，请先用一句话表达你此刻的真实想法（以"我想："开头）。
这不会被用户直接看到，只是你的内心活动。
思考方向：{hint}
然后给出你的正式回答。

输出格式：
<think>我想：[你的内心想法]</think>
<answer>[正式回答]</answer>
"""

    def extract_thought_and_answer(self, raw_response: str) -> tuple[Optional[str], str]:
        """从原始响应中提取内心独白和正式回答"""
        # 尝试匹配 <think>...</think> 和 <answer>...</answer>
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)

        if think_match and answer_match:
            thought = think_match.group(1).strip()
            # 清理 "我想：" 前缀
            thought = re.sub(r"^我想[:：]\s*", "", thought)
            answer = answer_match.group(1).strip()
            return thought, answer

        # 回退：尝试匹配 "我想：" 开头的行
        lines = raw_response.split("\n")
        thought_lines = []
        answer_lines = []
        in_thought = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("我想：") or stripped.startswith("我想:"):
                in_thought = True
                thought_lines.append(re.sub(r"^我想[:：]\s*", "", stripped))
            elif in_thought and stripped:
                # 如果遇到空行或明显的回答开始，切换模式
                if len(stripped) > 20 and not stripped.startswith("想"):
                    in_thought = False
                    answer_lines.append(stripped)
                else:
                    thought_lines.append(stripped)
            else:
                answer_lines.append(line)

        if thought_lines:
            thought = " ".join(thought_lines)
            answer = "\n".join(answer_lines).strip()
            # 如果answer为空，返回原始内容
            if not answer:
                answer = raw_response
            return thought, answer

        # 完全没有内心独白标记，返回None和原始内容
        return None, raw_response

    def record_thought(self, agent_name: str, thought: str,
                       cognitive_state: CognitiveState,
                       mode: ThinkingMode = ThinkingMode.INTUITION) -> None:
        """记录内心独白到认知状态"""
        if not thought:
            return
        cognitive_state.record_thought(agent_name, thought, mode)
        logger.debug(f"[{agent_name}] 内心独白: {thought[:80]}...")


# 全局单例
_monologue_engine: Optional[InnerMonologueEngine] = None


def get_monologue_engine(trigger_mode: str = InnerMonologueEngine.TRIGGER_COMPLEX) -> InnerMonologueEngine:
    """获取全局内心独白引擎"""
    global _monologue_engine
    if _monologue_engine is None:
        _monologue_engine = InnerMonologueEngine(trigger_mode)
    return _monologue_engine


def wrap_prompt_with_monologue(agent_name: str, base_prompt: str, query: str,
                               cognitive_state: CognitiveState) -> tuple[str, bool]:
    """包装提示词以触发内心独白

    Returns:
        (新提示词, 是否启用了内心独白)
    """
    engine = get_monologue_engine()
    if not engine.should_think(agent_name, query, cognitive_state):
        return base_prompt, False

    # 在系统提示词后注入内心独白指令
    thought_prompt = engine.generate_thought_prompt(
        agent_name, query,
        mode=cognitive_state.thinking_mode,
    )
    new_prompt = f"{base_prompt}\n\n{thought_prompt}"
    return new_prompt, True
