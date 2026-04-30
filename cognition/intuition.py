"""直觉引擎——系统1：快速、自动、经验驱动"""
import logging
import re
from typing import Optional
from dataclasses import dataclass

from cognition.types import IntuitionResult, ThinkingMode

logger = logging.getLogger(__name__)


@dataclass
class IntuitionPattern:
    """直觉模式——经验规则"""
    pattern: str           # 匹配正则
    intent: str            # 意图分类
    gut_feeling: str       # 直觉感受
    action: str            # 建议行动
    confidence_boost: float = 0.0  # 信心调整


# 预定义的直觉模式库——像人类的"经验"
INTUITION_PATTERNS = [
    # 问候类
    IntuitionPattern(r"^(你好|嗨|hello|hi|hey|在吗|在么)",
                     "greeting", "用户在打招呼", "友好回应，简短温暖", 0.9),
    IntuitionPattern(r"^(谢谢|感谢|thank)",
                     "gratitude", "用户在表达感谢", "谦逊回应，表示乐意帮助", 0.8),
    IntuitionPattern(r"^(再见|拜拜|bye|goodbye)",
                     "farewell", "用户要结束对话", "友好告别", 0.9),

    # 简单问题
    IntuitionPattern(r"^(什么|what).*(时间|天气|名字)",
                     "simple_fact", "简单事实查询", "直接回答", 0.8),
    IntuitionPattern(r"^(怎么|how).*(做|用|安装|开始)",
                     "how_to", "操作指南类问题", "给出步骤说明", 0.7),

    # 情绪类
    IntuitionPattern(r".*(烦|生气|讨厌|垃圾|差劲|stupid|hate|angry)",
                     "user_frustrated", "用户情绪负面", "先安抚情绪，再解决问题", 0.8),
    IntuitionPattern(r".*(开心|棒|厉害| awesome|great|love)",
                     "user_happy", "用户情绪正面", "分享喜悦，适度回应", 0.8),

    # 搜索类
    IntuitionPattern(r".*(搜索|查找|找一下|search|look up|find)",
                     "search", "需要搜索信息", "触发搜索工具", 0.7),
    IntuitionPattern(r".*(最新|新闻|趋势|news|latest|trend)",
                     "news", "时效性信息需求", "必须联网搜索", 0.8),

    # 编码类
    IntuitionPattern(r".*(代码|编程|bug|error|python|javascript|写个程序)",
                     "coding", "编程相关问题", "给出代码示例和解释", 0.7),

    # 比较类
    IntuitionPattern(r".*(区别|对比|比较|vs|versus|difference|compare)",
                     "comparison", "比较类问题", "结构化对比", 0.6),

    # 情感/建议类
    IntuitionPattern(r".*(建议|推荐|应该|怎么办|advice|recommend|should)",
                     "advice", "寻求建议", "给出考虑周全的建议", 0.6),
    IntuitionPattern(r".*(难过|伤心|失恋|压力|焦虑|sad|depressed|anxious)",
                     "emotional_support", "情感支持需求", "温柔倾听，共情回应", 0.7),

    # 创意类
    IntuitionPattern(r".*(写|创作|故事| poem|story|creative|imagine)",
                     "creative", "创意生成需求", "发挥想象力", 0.7),
]


class IntuitionEngine:
    """直觉引擎——模仿人类的系统1思维"""

    def __init__(self):
        self.patterns = INTUITION_PATTERNS
        self._experience_cache: dict[str, IntuitionResult] = {}  # 缓存常见判断

    def classify(self, query: str, history_length: int = 0) -> IntuitionResult:
        """对查询进行直觉分类

        特点：
        - 快速（< 1ms）
        - 基于模式匹配而非深度分析
        - 给出" gut feeling"——直觉感受
        - 判断是否需要理性验证
        """
        # 检查缓存
        cache_key = query.lower().strip()[:50]
        if cache_key in self._experience_cache:
            return self._experience_cache[cache_key]

        best_match: Optional[IntuitionPattern] = None
        best_score = 0.0

        for pattern in self.patterns:
            if re.search(pattern.pattern, query, re.IGNORECASE):
                # 计算匹配质量（简单长度因子）
                score = len(query) / 100 + pattern.confidence_boost
                if score > best_score:
                    best_score = score
                    best_match = pattern

        if best_match:
            confidence = min(0.95, 0.5 + best_match.confidence_boost)
            result = IntuitionResult(
                intent=best_match.intent,
                confidence=confidence,
                gut_feeling=best_match.gut_feeling,
                suggested_action=best_match.action,
                should_verify=confidence < 0.8,  # 信心不足时需要验证
            )
        else:
            # 没有匹配到任何模式——像人类遇到不熟悉的情况
            result = IntuitionResult(
                intent="unknown",
                confidence=0.3,
                gut_feeling="这个情况不太熟悉，需要仔细看看",
                suggested_action="谨慎处理，可能需要搜索或深入分析",
                should_verify=True,
            )

        # 对话历史因子：对话越长，直觉越不准
        if history_length > 10:
            result.confidence *= 0.9
            result.gut_feeling += "（对话已经很长，上下文可能很复杂）"

        # 缓存结果
        self._experience_cache[cache_key] = result
        # 限制缓存大小
        if len(self._experience_cache) > 1000:
            self._experience_cache.clear()

        logger.debug(f"直觉判断: {result.intent}, 信心{result.confidence:.1f}, "
                    f"验证={result.should_verify}")
        return result

    def route_decision(self, query: str, history_length: int = 0) -> dict:
        """基于直觉的路由决策

        Returns:
            {"route": str, "skip_search": bool, "skip_memory": bool,
             "reasoning": str, "thinking_mode": ThinkingMode}
        """
        intuition = self.classify(query, history_length)

        # 路由映射
        route_map = {
            "greeting": ("responder", True, True),
            "gratitude": ("responder", True, True),
            "farewell": ("responder", True, True),
            "simple_fact": ("responder", True, False),
            "how_to": ("researcher", False, False),
            "user_frustrated": ("responder", True, True),
            "user_happy": ("responder", True, True),
            "search": ("researcher", False, False),
            "news": ("researcher", False, True),
            "coding": ("researcher", False, False),
            "comparison": ("researcher", False, False),
            "advice": ("responder", True, False),
            "emotional_support": ("responder", True, False),
            "creative": ("responder", True, False),
            "unknown": ("coordinator", False, False),
        }

        route, skip_search, skip_memory = route_map.get(
            intuition.intent, ("coordinator", False, False)
        )

        # 直觉+理性混合：信心低时升级到coordinator
        if intuition.should_verify and route != "coordinator":
            route = "coordinator"
            reasoning = f"直觉觉得是{intuition.intent}，但信心不足，需要coordinator确认"
        else:
            reasoning = f"直觉: {intuition.gut_feeling} → 直接路由到 {route}"

        return {
            "route": route,
            "skip_search": skip_search,
            "skip_memory": skip_memory,
            "skip_knowledge": skip_search,  # 和搜索同步
            "reasoning": reasoning,
            "thinking_mode": ThinkingMode.INTUITION if intuition.confidence > 0.7 else ThinkingMode.REASONING,
            "intuition_confidence": intuition.confidence,
        }

    def get_intuition_hint_for_prompt(self, query: str) -> str:
        """获取直觉提示，注入到agent的prompt中"""
        result = self.classify(query)
        if result.confidence < 0.5:
            return ""  # 直觉不明确，不干扰
        return f"【你的直觉】{result.gut_feeling}（信心{result.confidence:.0%}）"


# 全局单例
_intuition_engine: Optional[IntuitionEngine] = None


def get_intuition_engine() -> IntuitionEngine:
    """获取全局直觉引擎"""
    global _intuition_engine
    if _intuition_engine is None:
        _intuition_engine = IntuitionEngine()
    return _intuition_engine
