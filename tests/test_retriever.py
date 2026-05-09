"""
Unit tests for the HybridRetriever's internal logic (RRF, gating).
These tests use mock objects — no FAISS, BM25, or API keys needed.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import HybridRetriever


def _make_doc(content, source="test.pdf", category="ML"):
    """Helper to create a mock LangChain Document."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source_file": source, "category": category}
    return doc


# ============================================================
# 1. RECIPROCAL RANK FUSION TESTS
# ============================================================

class TestReciprocalRankFusion:
    """Tests the RRF merging logic in isolation."""

    def _build_retriever(self):
        """Create a HybridRetriever with mocked dependencies."""
        return HybridRetriever(
            vectorstore=MagicMock(),
            bm25_index=MagicMock(),
            documents=[],
            cross_encoder=MagicMock(),
        )

    def test_rrf_empty_inputs(self):
        """RRF with no results from either retriever returns empty list."""
        ret = self._build_retriever()
        result = ret._reciprocal_rank_fusion([], [])
        assert result == []

    def test_rrf_faiss_only(self):
        """RRF with only FAISS results should return those documents."""
        ret = self._build_retriever()
        doc = _make_doc("FAISS doc")
        result = ret._reciprocal_rank_fusion([(doc, 0.5)], [])
        assert len(result) == 1
        assert result[0].page_content == "FAISS doc"

    def test_rrf_bm25_only(self):
        """RRF with only BM25 results should return those documents."""
        ret = self._build_retriever()
        doc = _make_doc("BM25 doc")
        result = ret._reciprocal_rank_fusion([], [(doc, 15.0)])
        assert len(result) == 1
        assert result[0].page_content == "BM25 doc"

    def test_rrf_deduplication(self):
        """Docs with matching content prefixes should be deduplicated and scores merged."""
        ret = self._build_retriever()
        shared_content = "X" * 250  # > 200 chars to match the prefix dedup logic
        doc_faiss = _make_doc(shared_content)
        doc_bm25 = _make_doc(shared_content)
        result = ret._reciprocal_rank_fusion(
            [(doc_faiss, 0.3)], [(doc_bm25, 18.0)]
        )
        # Should be deduplicated into a single entry
        assert len(result) == 1

    def test_rrf_unique_docs_preserved(self):
        """Distinct documents from FAISS and BM25 should all appear in results."""
        ret = self._build_retriever()
        doc_a = _make_doc("Document Alpha about LoRA")
        doc_b = _make_doc("Document Beta about DPO")
        result = ret._reciprocal_rank_fusion(
            [(doc_a, 0.4)], [(doc_b, 12.0)]
        )
        assert len(result) == 2

    def test_rrf_ranking_order(self):
        """Higher-ranked documents should appear first after fusion."""
        ret = self._build_retriever()
        doc1 = _make_doc("Top ranked doc")
        doc2 = _make_doc("Second ranked doc")
        doc3 = _make_doc("Third ranked doc")
        # doc1 appears first in both lists, should have highest RRF score
        faiss_results = [(doc1, 0.1), (doc2, 0.5), (doc3, 0.9)]
        bm25_results = [(doc1, 20.0), (doc3, 15.0)]
        result = ret._reciprocal_rank_fusion(faiss_results, bm25_results)
        # doc1 should be first (appears in both lists at rank 0)
        assert result[0].page_content == "Top ranked doc"


# ============================================================
# 2. THRESHOLD GATING TESTS
# ============================================================

class TestThresholdGating:
    """Tests the score-based filtering logic."""

    def test_faiss_gating_filters_high_distance(self):
        """FAISS docs with L2 distance above threshold should be removed."""
        threshold = 1.5
        results = [(_make_doc("good"), 0.8), (_make_doc("bad"), 2.0)]
        gated = [(doc, s) for doc, s in results if s <= threshold]
        assert len(gated) == 1
        assert gated[0][0].page_content == "good"

    def test_bm25_gating_filters_low_scores(self):
        """BM25 docs with relevance below threshold should be removed."""
        threshold = 1.0
        results = [(_make_doc("relevant"), 15.0), (_make_doc("noise"), 0.3)]
        gated = [(doc, s) for doc, s in results if s >= threshold]
        assert len(gated) == 1
        assert gated[0][0].page_content == "relevant"

    def test_cross_encoder_negative_filter(self):
        """Documents scoring below -4.0 from cross-encoder should be dropped."""
        scored = [
            (_make_doc("good match"), 3.5),
            (_make_doc("borderline"), -3.9),
            (_make_doc("irrelevant"), -7.2),
        ]
        final = [doc for doc, score in scored if score >= -4.0]
        assert len(final) == 2
        assert all(d.page_content != "irrelevant" for d in final)
