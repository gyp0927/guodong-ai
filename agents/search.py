"""搜索子 Agent - 联网搜索、记忆搜索、知识库搜索。"""

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# 子 Agent 的搜索超时（秒）。比单源搜索的内部超时大,留出 fallback 时间。
WEB_SEARCH_TIMEOUT_S = 12.0
MEMORY_SEARCH_TIMEOUT_S = 3.0


async def _safe_search(
    search_fn: Callable,
    label: str,
    *args,
    success_check: Callable = None,
    **kwargs,
) -> str:
    """通用搜索包装器——统一异常处理和结果格式化。

    异常一律打 warning 日志，避免静默失败掩盖根因。
    """
    try:
        result = await search_fn(*args, **kwargs)
        if success_check:
            if success_check(result):
                return f"[{label}]\n\n{result}"
            logger.debug(f"{label}: 结果未通过 success_check，丢弃")
        elif result:
            return f"[{label}]\n\n{result}"
        else:
            logger.debug(f"{label}: 空结果")
    except asyncio.TimeoutError:
        logger.warning(f"{label} 超时")
    except (TimeoutError, ConnectionError) as e:
        logger.warning(f"{label} 网络错误: {e}")
    except ImportError as e:
        logger.warning(f"{label} 依赖缺失: {e}")
    except Exception as e:
        logger.warning(f"{label} failed: {e}")
    return ""


async def web_searcher_agent(query: str, user_id: str = "", session_id: str = "") -> str:
    """联网搜索子 Agent - 执行 DuckDuckGo 搜索并总结结果。"""
    from tools.search import search_and_summarize

    def _check(result: str) -> bool:
        return result and "未找到" not in result

    async def _search_with_timeout(q: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(search_and_summarize, q, max_results=2),
            timeout=WEB_SEARCH_TIMEOUT_S,
        )

    return await _safe_search(_search_with_timeout, "联网搜索结果", query, success_check=_check)


async def memory_searcher_agent(query: str, user_id: str = "", session_id: str = "") -> str:
    """记忆搜索子 Agent - 从自适应记忆系统检索相关记忆。

    session_id 非空时按会话隔离,只检索本会话写入的记忆;空串=不过滤(全局检索)。
    """
    from core.memory_client import get_memory_store, _MEMORY_SYSTEM_AVAILABLE

    async def _do_search(q: str, uid: str, sess: str) -> str:
        if not _MEMORY_SYSTEM_AVAILABLE:
            return ""
        store = get_memory_store()
        memories = await asyncio.wait_for(
            store.retrieve(q, top_k=5, user_id=uid, source=sess),
            timeout=MEMORY_SEARCH_TIMEOUT_S,
        )
        if memories:
            return store.format_memories_for_prompt(memories) or ""
        return ""

    return await _safe_search(_do_search, "记忆检索结果", query, user_id, session_id)


async def knowledge_searcher_agent(query: str, user_id: str = "") -> str:
    """知识库搜索子 Agent - 从 RAG 向量库检索相关文档。"""
    from core.rag import search_knowledge

    def _check(result: str) -> bool:
        return result and "知识库为空" not in result and "未启用" not in result

    return await _safe_search(
        lambda q: asyncio.to_thread(search_knowledge, q, top_k=3),
        "知识库检索结果",
        query,
        success_check=_check,
    )


async def run_parallel_search(state: dict) -> str:
    """根据 state 中的 mode 并行运行搜索子 Agent，返回整合后的搜索上下文。"""
    user_message = state["messages"][-1].content
    mode = state.get("task_context", {}).get("mode", "coordination")
    user_id = state.get("task_context", {}).get("user_id", "")
    session_id = state.get("task_context", {}).get("session_id", "")

    tasks = []
    if mode in ("fast", "planning"):
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id, session_id))
    else:
        tasks.append(web_searcher_agent(user_message))
        tasks.append(memory_searcher_agent(user_message, user_id, session_id))
        tasks.append(knowledge_searcher_agent(user_message))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    search_parts = [r for r in results if isinstance(r, str) and r]
    return "\n\n".join(search_parts) if search_parts else ""
