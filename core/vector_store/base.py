"""向量存储后端抽象基类。"""

import abc
from typing import Any, Optional


class VectorStoreBackend(abc.ABC):
    """向量存储后端抽象基类。所有后端实现必须继承此类。"""

    @abc.abstractmethod
    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        """添加向量到存储。"""
        pass

    @abc.abstractmethod
    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        """搜索最相似的向量。

        Returns:
            结果列表，每项包含 text, metadata, score
        """
        pass

    @abc.abstractmethod
    def clear(self):
        """清空所有数据。"""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """返回存储的向量数量。"""
        pass

    @abc.abstractmethod
    def save(self):
        """保存数据到磁盘（如果后端支持持久化）。"""
        pass
