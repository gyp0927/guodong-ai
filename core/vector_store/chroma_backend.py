"""Chroma 向量存储后端 - 使用 ChromaDB 的持久化实现。"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Chroma 是可选依赖
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    chromadb = None  # type: ignore

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chroma_db")


class ChromaBackend:
    """基于 ChromaDB 的向量存储后端"""

    def __init__(self, persist_path: str = None):
        if not HAS_CHROMA:
            raise ImportError(
                "ChromaDB 未安装。请运行: pip install chromadb\n"
                "或继续使用 numpy 后端（默认）。"
            )
        self.persist_path = persist_path or _DEFAULT_PATH
        os.makedirs(self.persist_path, exist_ok=True)

        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_path,
            anonymized_telemetry=False,
        ))
        self.collection = self.client.get_or_create_collection("rag_documents")
        logger.info(f"Chroma backend initialized at {self.persist_path}")

    def add(self, vector: Any, text: str, metadata: dict = None, auto_save: bool = True):
        import uuid
        meta = metadata or {}
        self.collection.add(
            embeddings=[vector.tolist() if hasattr(vector, "tolist") else list(vector)],
            documents=[text],
            metadatas=[meta],
            ids=[meta.get("chunk_id", str(uuid.uuid4()))],
        )
        # Chroma 自动持久化，无需手动 save

    def search(self, query_vector: Any, top_k: int = 3) -> list[dict]:
        vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        results = self.collection.query(
            query_embeddings=[vector_list],
            n_results=top_k,
        )
        output = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                output.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": float(results["distances"][0][i]) if results["distances"] else 0.0,
                })
        return output

    def clear(self):
        self.client.delete_collection("rag_documents")
        self.collection = self.client.get_or_create_collection("rag_documents")

    def count(self) -> int:
        return self.collection.count()

    def save(self):
        # Chroma 自动持久化
        pass

    def list_documents(self) -> list[dict]:
        """按 source 分组列出所有文档。"""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.get(
            limit=count,
            include=["metadatas"],
        )
        sources = {}
        for meta in results.get("metadatas", []):
            if meta:
                source = meta.get("source", "未知来源")
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
        return [{"source": s, "chunks": c} for s, c in sorted(sources.items())]

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的所有文档块。

        Returns:
            删除的块数
        """
        count = self.collection.count()
        if count == 0:
            return 0

        results = self.collection.get(
            limit=count,
            include=["metadatas"],
        )
        ids_to_remove = []
        for i, meta in enumerate(results.get("metadatas", [])):
            if meta and meta.get("source") == source:
                doc_id = results.get("ids", [])[i] if i < len(results.get("ids", [])) else None
                if doc_id:
                    ids_to_remove.append(doc_id)

        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
        return len(ids_to_remove)
