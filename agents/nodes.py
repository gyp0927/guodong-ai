"""Agent 节点函数 - Coordinator、Researcher、Responder、Reviewer、Planner。"""

import json
import logging
import re
from dataclasses import asdict
from typing import Optional, Callable

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from core.cache import get_cache
from core.plugin_system import get_plugins_prompt
from agents.prompts import COORDINATOR_PROMPT, get_reviewer_prompt, build_responder_prompt, PLANNER_PROMPT
from agents.llm import get_llm, get_streaming_callback
from agents.search import run_parallel_search
from state.stop_flag import is_stopped

from cognition.human_mind import HumanMind
from cognition.types import CognitiveState
from cognition.utils import get_cognitive_state_from_dict, save_cognitive_state_to_dict
from cognition.tool_engine import run_tool_loop

logger = logging.getLogger(__name__)


def _build_result_dict(
    response: str,
    agent_name: str,
    cognitive_state: CognitiveState | None = None,
) -> dict:
    """构建标准返回字典，自动包含认知状态（如果提供）。"""
    result = {"messages": [AIMessage(content=response, name=agent_name)]}
    if cognitive_state is not None:
        result["cognitive_state"] = asdict(cognitive_state)
    return result


async def _run_agent(
    state: dict,
    system_prompt: str,
    agent_name: str,
    sid: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
    enable_cognition: bool = True,
    enable_monologue: bool = True,
    tools: Optional[list[BaseTool]] = None,
) -> dict:
    """通用 Agent 执行函数（接入认知系统 + 工具调用）。"""
    cognitive_state = get_cognitive_state_from_dict(state)
    is_responder = agent_name == "responder"
    mind = HumanMind(
        enable_monologue=enable_monologue and is_responder,
        enable_emotion=is_responder,
        enable_intuition=True,
        enable_metacognition=is_responder,
        enable_persona=is_responder,
    ) if enable_cognition else None
    query = state["messages"][-1].content if state["messages"] else ""

    enhanced_prompt = system_prompt
    had_monologue = False
    if mind and enable_cognition:
        enhanced_prompt, had_monologue = mind.enhance_prompt(
            agent_name, system_prompt, query, cognitive_state
        )

    llm = get_llm(sid or "")
    messages = [SystemMessage(content=enhanced_prompt)] + list(state["messages"])

    # 缓存
    cache_enabled = agent_name in ("responder", "coordinator")
    cache = get_cache() if cache_enabled else None
    from core.config import get_provider, get_model_name
    cache_key = (get_provider(), get_model_name())

    if cache and cache_enabled:
        try:
            cached = cache.get(messages, cache_key[0], cache_key[1])
            if cached is not None:
                logger.info(f"[{agent_name}] Cache hit")
                if agent_name == "coordinator":
                    return {"messages": [AIMessage(content=cached, name=agent_name)]}
                stream_cb = on_token if on_token else get_streaming_callback(sid)
                if stream_cb:
                    for i in range(0, len(cached), 20):
                        stream_cb(cached[i:i+20])
                return {"messages": [AIMessage(content=cached, name=agent_name)]}
        except (OSError, ValueError) as e:
            logger.warning(f"Cache lookup failed: {e}")

    if agent_name == "coordinator":
        llm = llm.bind(max_tokens=80)

    if tools:
        stream_cb = on_token if on_token else get_streaming_callback(sid)
        response = await run_tool_loop(llm, messages, tools, max_iterations=3, sid=sid or "", on_token=stream_cb)
    else:
        response = ""
        stream_cb = on_token if on_token else (get_streaming_callback(sid) if agent_name == "responder" else None)
        async for chunk in llm.astream(messages):
            if is_stopped(sid):
                break
            if chunk.content:
                response += chunk.content
                if stream_cb:
                    stream_cb(chunk.content)

    if cache and cache_enabled and response:
        try:
            cache.set(messages, cache_key[0], cache_key[1], response)
        except (OSError, ValueError) as e:
            logger.warning(f"Cache write failed: {e}")

    if mind and enable_cognition:
        response = mind.process_response(agent_name, query, response, cognitive_state, had_monologue)
        save_cognitive_state_to_dict(state, cognitive_state)

    return _build_result_dict(response, agent_name, cognitive_state if enable_cognition else None)


async def coordinator_node(state: dict, sid: str | None = None) -> dict:
    """协调者Agent - 分析需求并决定路由。"""
    return await _run_agent(state, COORDINATOR_PROMPT, "coordinator", sid, enable_cognition=True)


async def researcher_node(state: dict, sid: str | None = None) -> dict:
    """研究员 Agent - 仅并行搜索并整合结果。"""
    search_context = await run_parallel_search(state)
    if search_context:
        return {"messages": [SystemMessage(
            content=f"【搜索结果】\n\n{search_context}\n\n请基于以上搜索结果生成最终回答。",
            name="researcher",
        )]}
    return {"messages": []}


async def responder_node(state: dict, sid: str | None = None) -> dict:
    """响应者Agent - 生成最终回答。"""
    plugin_prompt = get_plugins_prompt()
    from core.i18n import LANG_INSTRUCTIONS
    detected_lang = state.get("task_context", {}).get("detected_language", "zh")
    lang_instr = LANG_INSTRUCTIONS.get(detected_lang, "")
    responder_prompt = build_responder_prompt(plugin_prompt, lang_instr)
    return await _run_agent(state, responder_prompt, "responder", sid, enable_cognition=True, tools=None)


async def reviewer_node(state: dict, language: str = "zh", sid: str | None = None) -> dict:
    """检查者Agent - 审查回答质量。"""
    reviewer_prompt = get_reviewer_prompt(language)
    return await _run_agent(state, reviewer_prompt, "reviewer", sid, enable_cognition=True)


async def planner_node(state: dict, sid: str | None = None) -> dict:
    """计划者Agent - 分析需求并生成结构化计划。"""
    return await _run_agent(state, PLANNER_PROMPT, "planner", sid)


def parse_plan_from_response(text: str) -> dict | None:
    """从 Agent 输出中提取 JSON 计划。"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\"title\"[\s\S]*\"steps\"[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def create_agents(language: str = "zh", fast_mode: bool = False):
    """创建所有 Agent 节点函数的便捷函数。"""
    async def _reviewer_node(state: dict, sid: str | None = None) -> dict:
        return await reviewer_node(state, language=language, sid=sid)

    return (coordinator_node, researcher_node, responder_node, _reviewer_node)
