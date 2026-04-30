"""元认知系统——让Agent能"思考自己的思考"

模仿人类的元认知能力：
- 知道自己知道什么、不知道什么
- 能评估自己答案的质量
- 能发现错误并自我修正
- 能识别自己的偏见和盲点
"""
import logging
import re
from typing import Optional

from cognition.types import MetacognitionResult, ThinkingMode

logger = logging.getLogger(__name__)


class MetacognitionEngine:
    """元认知引擎——在输出前进行自我反思"""

    # 元认知提示词模板
    METACOGNITION_PROMPT = """在给出最终回答之前，请对自己的思考进行一次快速自检：

【元认知自检清单】
1. 我对这个答案有多大把握？(0-100%)
2. 我的知识中有什么盲区可能导致错误？
3. 我是否被某种偏见影响了？（如过度自信、确认偏误）
4. 如果我的答案有误，最可能错在哪里？

请用以下格式输出自检结果（这不会被用户直接看到）：

<metacognition>
把握度: [0-100]%
知识盲区: [列出可能不知道或不确定的点]
潜在偏见: [你意识到的思维偏见]
可能的错误: [如果错了，最可能的原因]
</metacognition>

然后给出你的正式回答。"""

    UNCERTAINTY_MARKERS = [
        "不确定", "可能", "也许", "大概", "我不太清楚",
        "据我所知", "我不确定", "可能不准确",
        "not sure", "might be", "probably", "possibly",
        "i'm not certain", "as far as i know", "i think",
    ]

    OVERCONFIDENCE_MARKERS = [
        "绝对", "一定", "毫无疑问", "100%", "肯定是",
        "absolutely", "definitely", "100%", "without a doubt",
        "certainly", "always", "never",
    ]

    def __init__(self, enabled: bool = True, certainty_threshold: float = 0.6):
        self.enabled = enabled
        self.certainty_threshold = certainty_threshold

    def analyze_response(self, query: str, response: str,
                         agent_name: str = "responder") -> MetacognitionResult:
        """分析回答的质量和自我一致性

        不需要额外的LLM调用，基于规则快速分析。
        """
        result = MetacognitionResult(
            certainty=0.5,
            knowledge_gaps=[],
            potential_biases=[],
            self_correction=None,
            should_rethink=False,
        )

        # 1. 评估确定性
        result.certainty = self._assess_certainty(response)

        # 2. 检测知识盲区
        result.knowledge_gaps = self._detect_knowledge_gaps(query, response)

        # 3. 检测潜在偏见
        result.potential_biases = self._detect_biases(response)

        # 4. 判断是否需要重新思考
        result.should_rethink = (
            result.certainty < self.certainty_threshold
            or len(result.knowledge_gaps) > 2
            or len(result.potential_biases) > 1
        )

        if result.should_rethink:
            result.self_correction = self._generate_correction_hint(result)

        logger.debug(f"[{agent_name}] 元认知: 把握度{result.certainty:.1f}, "
                    f"盲区{len(result.knowledge_gaps)}, 偏见{len(result.potential_biases)}, "
                    f"重想={result.should_rethink}")
        return result

    def _assess_certainty(self, response: str) -> float:
        """评估回答的确定性水平"""
        response_lower = response.lower()

        uncertainty_count = sum(1 for m in self.UNCERTAINTY_MARKERS if m in response_lower)
        overconfidence_count = sum(1 for m in self.OVERCONFIDENCE_MARKERS if m in response_lower)

        # 基础确定性
        certainty = 0.7

        # 不确定性标记降低确定性
        certainty -= uncertainty_count * 0.1

        # 过度自信标记——其实是心虚的表现（人类常这样）
        if overconfidence_count > 2:
            certainty -= 0.15

        # 回答长度因子：太短的回答可能准备不足
        if len(response) < 50:
            certainty -= 0.1

        # 包含"我不知道"类表达——诚实但确定性低
        if any(p in response_lower for p in ["不知道", "不清楚", "无法确定",
                                               "i don't know", "not sure"]):
            certainty = max(0.2, certainty - 0.3)

        return max(0.0, min(1.0, certainty))

    def _detect_knowledge_gaps(self, query: str, response: str) -> list[str]:
        """检测潜在的知识盲区"""
        gaps = []
        response_lower = response.lower()

        # 检测模糊/回避回答
        vague_patterns = [
            r"这(个|件)?事.{0,5}(很复杂|不好说|很难讲)",
            r"具体.*(不清楚|不知道|不确定)",
            r"可能.*(要看|取决于|视情况而定)",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, response):
                gaps.append("回答中存在模糊地带，可能知识不够精确")
                break

        # 检测时间敏感信息但没有时效性说明
        time_keywords = ["最新", "目前", "现在", "202", "今年"]
        if any(kw in query for kw in time_keywords):
            if "截至" not in response and "as of" not in response_lower:
                gaps.append("涉及时效性信息但未标注知识截止时间")

        # 检测专业领域但没有引用来源
        technical_keywords = ["研究表明", "数据显示", "统计", "论文", "research shows"]
        if any(kw in response for kw in technical_keywords):
            if "来源" not in response and "reference" not in response_lower:
                gaps.append("引用研究/数据但未提供来源")

        return gaps

    def _detect_biases(self, response: str) -> list[str]:
        """检测潜在的认知偏见"""
        biases = []
        response_lower = response.lower()

        # 确认偏误：只支持一种观点
        if response.count("但是") + response.count("然而") + response.count("不过") < 1:
            if len(response) > 200 and "?" not in response[:100]:
                biases.append("可能忽略了反方观点（确认偏误）")

        # 锚定效应：过早下结论
        if re.search(r"^(显然|明显|毫无疑问|当然)", response):
            biases.append("过早下结论，可能有锚定效应")

        # 过度概括
        generalization_markers = ["所有", "总是", "从不", "everyone", "always", "never", "all"]
        if any(m in response_lower for m in generalization_markers):
            biases.append("使用绝对化表述，可能有过度概括倾向")

        return biases

    def _generate_correction_hint(self, result: MetacognitionResult) -> str:
        """生成自我修正建议"""
        hints = []
        if result.certainty < 0.5:
            hints.append("建议明确表达不确定性，不要假装确定")
        if result.knowledge_gaps:
            hints.append("建议补充说明知识边界")
        if result.potential_biases:
            hints.append("建议考虑不同角度，避免片面")
        return "；".join(hints) if hints else None

    def get_metacognition_prompt(self) -> str:
        """获取用于注入到系统提示词中的元认知指令"""
        return """
【元认知提醒】
在回答时，请注意：
1. 如果你不确定，请诚实说"我不太确定"而不是编造
2. 考虑是否有你忽略的角度或反方观点
3. 涉及时效性信息时，标注你的知识截止时间
4. 避免绝对化表述（"永远"、"绝对"），保持谦逊
"""

    def inject_uncertainty_expression(self, response: str,
                                       certainty: float) -> str:
        """在回答中自然注入不确定性表达（如果确定性低）"""
        if certainty > 0.7:
            return response

        # 在开头添加适度的不确定性表达
        if certainty < 0.4:
            prefix = "嗯...这个问题我不太确定，但我可以分享一下我的理解："
        elif certainty < 0.6:
            prefix = "关于这个问题，我的看法是："
        else:
            return response

        # 避免重复添加
        if response.startswith("嗯") or response.startswith("关于"):
            return response

        return f"{prefix}\n\n{response}"


# 全局单例
_metacognition_engine: Optional[MetacognitionEngine] = None


def get_metacognition_engine(enabled: bool = True) -> MetacognitionEngine:
    """获取全局元认知引擎"""
    global _metacognition_engine
    if _metacognition_engine is None:
        _metacognition_engine = MetacognitionEngine(enabled=enabled)
    return _metacognition_engine
