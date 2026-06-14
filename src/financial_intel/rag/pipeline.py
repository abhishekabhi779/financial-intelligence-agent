"""
Channel Intelligence Agent — RAG Pipeline

Document ingestion, hybrid retrieval (BM25 + vector), and reranking.
"""

import os
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
import numpy as np

from financial_intel.config import get_settings
from financial_intel.state import VendorInfo


class RAGPipeline:
    """
    Complete RAG pipeline with:
    - Document ingestion (PDF, text, markdown)
    - Semantic chunking
    - Hybrid retrieval (BM25 + vector)
    - Cross-encoder reranking
    - Metadata filtering
    """

    def __init__(self):
        self.settings = get_settings()
        self.rag_config = self.settings.rag
        self.vector_config = self.settings.vector_db

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embeddings.model,
            dimensions=self.settings.embeddings.dimensions,
        )

        # Initialize vector DB
        self._init_vector_db()

        # BM25 retriever (built on demand)
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._documents: List[Document] = []

        # Reranker (lazy load)
        self._reranker = None

    def _init_vector_db(self):
        """Initialize ChromaDB client and collection."""
        persist_dir = self.settings.paths.chromadb
        collection_name = self.vector_config.chromadb.get("collection_name", "financial_intel")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.vector_config.chromadb.get("distance_metric", "cosine")},
        )

    def _get_reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.rag_config.rerank_model)
            except Exception:
                self._reranker = None
        return self._reranker

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    async def ingest_vendor_docs(self, vendor_name: str, doc_paths: List[str]) -> Dict[str, Any]:
        """Ingest vendor documentation (PDFs, specs, etc.)."""
        results = {"vendor": vendor_name, "chunks": 0, "errors": []}

        for path in doc_paths:
            try:
                docs = await self._load_document(path, vendor_name)
                chunks = self._chunk_documents(docs)
                await self._add_to_vector_db(chunks, vendor_name)
                self._documents.extend(chunks)
                results["chunks"] += len(chunks)
            except Exception as e:
                results["errors"].append({"file": path, "error": str(e)})

        # Rebuild BM25 index
        if self._documents:
            self._build_bm25_index()

        return results

    async def ingest_partner_data(self, partner_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest partner profile data."""
        # Create document from structured data
        content = self._format_partner_data(partner_name, data)
        doc = Document(
            page_content=content,
            metadata={
                "source": "partner_profile",
                "partner_name": partner_name,
                "type": "partner",
                "ingested_at": datetime.now().isoformat(),
            },
        )

        chunks = self._chunk_documents([doc])
        await self._add_to_vector_db(chunks, partner_name)
        self._documents.extend(chunks)

        if self._documents:
            self._build_bm25_index()

        return {"partner": partner_name, "chunks": len(chunks)}

    async def ingest_market_report(self, report_name: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest market research report."""
        doc = Document(
            page_content=content,
            metadata={
                "source": "market_report",
                "report_name": report_name,
                "type": "market",
                **metadata,
                "ingested_at": datetime.now().isoformat(),
            },
        )

        chunks = self._chunk_documents([doc])
        await self._add_to_vector_db(chunks, report_name)
        self._documents.extend(chunks)

        if self._documents:
            self._build_bm25_index()

        return {"report": report_name, "chunks": len(chunks)}

    async def _load_document(self, path: str, vendor_name: str) -> List[Document]:
        """Load document from file path."""
        ext = Path(path).suffix.lower()
        docs = []

        if ext == ".pdf":
            import fitz  # pymupdf
            pdf = fitz.open(path)
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "vendor_doc",
                            "vendor_name": vendor_name,
                            "file_path": path,
                            "page": page_num + 1,
                            "type": "vendor",
                        },
                    ))
            pdf.close()

        elif ext in [".txt", ".md", ".markdown"]:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "vendor_doc",
                    "vendor_name": vendor_name,
                    "file_path": path,
                    "type": "vendor",
                },
            ))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return docs

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks using configured strategy."""
        strategy = self.rag_config.chunking_strategy

        if strategy == "semantic":
            splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.rag_config.semantic_chunker_threshold * 100,
            )
        elif strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.rag_config.chunk_size,
                chunk_overlap=self.rag_config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:  # fixed
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.rag_config.chunk_size,
                chunk_overlap=self.rag_config.chunk_overlap,
            )

        return splitter.split_documents(docs)

    def _format_partner_data(self, partner_name: str, data: Dict[str, Any]) -> str:
        """Format partner data as searchable text."""
        parts = [f"Partner: {partner_name}"]

        for key, value in data.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif value:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    async def _add_to_vector_db(self, chunks: List[Document], source_id: str):
        """Add chunks to ChromaDB."""
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [self._generate_chunk_id(c, source_id, i) for i, c in enumerate(chunks)]

        # Get embeddings
        embeddings = await self.embeddings.aembed_documents(texts)

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def _generate_chunk_id(self, chunk: Document, source_id: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        return f"{source_id}_{index}_{content_hash}"

    def _build_bm25_index(self):
        """Build BM25 index from all documents."""
        if not self._documents:
            return

        corpus = [doc.page_content for doc in self._documents]
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25_retriever = BM25Retriever.from_texts(
            texts=corpus,
            metadatas=[doc.metadata for doc in self._documents],
            bm25=BM25Okapi(tokenized),
            k=self.rag_config.top_k,
        )

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results (default from config)
            filter_metadata: Metadata filters
            search_type: "vector", "bm25", or "hybrid"

        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or self.rag_config.top_k
        rerank_k = self.rag_config.rerank_top_k

        # Vector search
        vector_results = await self._vector_search(query, top_k * 2, filter_metadata)

        # BM25 search
        bm25_results = self._bm25_search(query, top_k * 2, filter_metadata) if self._bm25_retriever else []

        # Hybrid fusion
        if search_type == "hybrid":
            fused = self._reciprocal_rank_fusion(vector_results, bm25_results)
        elif search_type == "vector":
            fused = vector_results
        else:
            fused = bm25_results

        # Rerank
        if rerank_k and len(fused) > rerank_k:
            fused = await self._rerank(query, fused, rerank_k)

        return fused[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        query_embedding = await self.embeddings.aembed_query(query)

        where = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
                "source": "vector",
            })
        return formatted

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 keyword search."""
        if not self._bm25_retriever:
            return []

        # BM25Retriever doesn't support metadata filtering natively
        # We filter post-retrieval
        docs = self._bm25_retriever.invoke(query)

        results = []
        for doc in docs[:top_k]:
            if filter_metadata:
                match = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            results.append({
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.5),
                "source": "bm25",
            })
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion for hybrid search."""
        # Score by reciprocal rank
        scores = {}

        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in scores or "content" not in scores[doc_id]:
                scores[doc_id] = {"content": result["content"], "metadata": result["metadata"], "score": 0}

        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: x[1] if isinstance(x[1], float) else x[1].get("score", 0), reverse=True)

        # Reconstruct full results
        all_results = {r["id"]: r for r in vector_results + bm25_results}
        return [all_results[doc_id] for doc_id, _ in fused if doc_id in all_results]

    async def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Cross-encoder reranking."""
        reranker = self._get_reranker()
        if not reranker or not results:
            return results

        pairs = [(query, r["content"]) for r in results]
        scores = reranker.predict(pairs)

        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
            result["score"] = float(scores[i])  # Override with rerank score

        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "embedding_model": self.settings.embeddings.model,
        }

    def delete_vendor_data(self, vendor_name: str) -> int:
        """Delete all data for a vendor."""
        # ChromaDB doesn't support delete by metadata directly
        # Would need to query and delete by IDs
        return 0


# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get global RAG pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline