"""向量存储后端工厂。"""

import logging
import os

from .numpy_backend import NumpyBackend, HAS_NUMPY
from .chroma_backend import ChromaBackend, HAS_CHROMA

logger = logging.getLogger(__name__)

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "state", "model_configs.json")


def _get_backend_from_config() -> str:
    """从全局配置中读取向量存储后端设置。"""
    try:
        import json
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "state", "rag_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("backend", "numpy")
    except Exception:
        pass
    return "numpy"


def _save_backend_config(backend: str, **kwargs):
    """保存向量存储后端配置。"""
    try:
        import json
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "state", "rag_config.json")
        data = {"backend": backend}
        data.update(kwargs)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save RAG backend config: {e}")


def get_vector_store(backend: str = None, persist_path: str = None):
    """获取向量存储后端实例。

    Args:
        backend: 后端类型 "numpy" 或 "chroma"，None 则读取配置
        persist_path: Chroma 持久化路径（仅 chroma 有效）

    Returns:
        向量存储后端实例
    """
    if backend is None:
        backend = _get_backend_from_config()

    backend = backend.lower()

    if backend == "chroma":
        if not HAS_CHROMA:
            logger.warning("ChromaDB 未安装，回退到 numpy 后端")
            return NumpyBackend()
        try:
            return ChromaBackend(persist_path=persist_path)
        except Exception as e:
            logger.warning(f"Chroma backend failed: {e}, falling back to numpy")
            return NumpyBackend()

    # 默认使用 numpy
    if not HAS_NUMPY:
        logger.warning("numpy 未安装，RAG 功能不可用")
    return NumpyBackend()


def set_backend(backend: str, persist_path: str = None) -> bool:
    """切换向量存储后端并保存配置。

    Returns:
        是否切换成功
    """
    backend = backend.lower()
    if backend == "chroma" and not HAS_CHROMA:
        return False
    _save_backend_config(backend, persist_path=persist_path)
    return True


def list_backends() -> list[dict]:
    """列出可用的后端及其状态。"""
    return [
        {"name": "numpy", "available": HAS_NUMPY, "description": "基于 numpy 的轻量级实现（默认）"},
        {"name": "chroma", "available": HAS_CHROMA, "description": "基于 ChromaDB 的持久化实现"},
    ]


def list_documents() -> list[dict]:
    """列出知识库中的所有文档（按来源分组）。"""
    store = get_vector_store()
    if hasattr(store, "list_documents"):
        return store.list_documents()
    return []


def delete_by_source(source: str) -> int:
    """删除指定来源的所有文档块。

    Returns:
        删除的块数
    """
    store = get_vector_store()
    if hasattr(store, "delete_by_source"):
        return store.delete_by_source(source)
    return 0
