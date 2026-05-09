"""
Unit tests for utility functions used across the RAG pipeline.
These tests do NOT require API keys or external services.
"""
import pytest
import sys
import os

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. TESTS FOR extract_context_chunks
# ============================================================

from evaluate_rag import extract_context_chunks


class TestExtractContextChunks:
    """Validates the context string parser used by the DeepEval pipeline."""

    def test_empty_string_returns_default(self):
        """Empty context should return a single-element list with empty string."""
        result = extract_context_chunks("")
        assert result == [""]

    def test_whitespace_only_returns_default(self):
        """Whitespace-only input should be treated as empty."""
        result = extract_context_chunks("   \n\t  ")
        assert result == [""]

    def test_single_source_extracted(self):
        """A single [Source: ...] block should produce exactly one chunk."""
        context = "--- INTERNAL DATA ---\n[Source: LoRA.pdf] LoRA freezes weights."
        result = extract_context_chunks(context)
        assert len(result) == 1
        assert result[0].startswith("[Source: ")
        assert "LoRA freezes weights" in result[0]

    def test_multiple_sources_extracted(self):
        """Multiple [Source: ...] blocks should be split into separate chunks."""
        context = (
            "--- INTERNAL DATA ---\n"
            "[Source: LoRA.pdf] LoRA freezes weights.\n\n"
            "[Source: DPO.pdf] DPO optimizes preferences.\n\n"
            "[Source: mml-book.pdf] Eigenvalues decompose matrices."
        )
        result = extract_context_chunks(context)
        assert len(result) == 3
        assert any("LoRA" in c for c in result)
        assert any("DPO" in c for c in result)
        assert any("Eigenvalues" in c for c in result)

    def test_internal_data_header_filtered(self):
        """The '--- INTERNAL DATA ---' header should never appear in chunks."""
        context = "--- INTERNAL DATA ---\n[Source: test.pdf] Some content."
        result = extract_context_chunks(context)
        for chunk in result:
            assert "--- INTERNAL DATA" not in chunk

    def test_web_search_fallback_as_single_chunk(self):
        """Emergency web results (no [Source: ] tags) become one chunk."""
        context = "[EMERGENCY SEARCH RESULTS FOR: 'test query']:\nSome web data..."
        result = extract_context_chunks(context)
        # Since there's no [Source: tag, the whole string is returned as fallback
        assert len(result) == 1
        assert "EMERGENCY SEARCH" in result[0]

    def test_source_prefix_preserved(self):
        """Each chunk must start with '[Source: ' after splitting."""
        context = (
            "--- INTERNAL DATA ---\n"
            "[Source: A.pdf] Content A.\n"
            "[Source: B.pdf] Content B."
        )
        result = extract_context_chunks(context)
        for chunk in result:
            assert chunk.startswith("[Source: ")


# ============================================================
# 2. TESTS FOR AgentState SCHEMA
# ============================================================

class TestAgentStateSchema:
    """Validates the LangGraph state dictionary contract."""

    def test_state_has_required_keys(self):
        """AgentState should include all required fields for the pipeline."""
        from main import AgentState
        required_keys = [
            "user_query", "topics", "context_data",
            "final_report", "loop_count", "needs_retry",
            "new_search_query"
        ]
        annotations = AgentState.__annotations__
        for key in required_keys:
            assert key in annotations, f"Missing required state key: {key}"


# ============================================================
# 3. TESTS FOR get_env UTILITY
# ============================================================

class TestGetEnv:
    """Validates the environment variable loader."""

    def test_get_env_returns_value(self, monkeypatch):
        """Should return the value when the env var exists."""
        monkeypatch.setenv("TEST_KEY_12345", "test_value")
        from main import get_env
        assert get_env("TEST_KEY_12345") == "test_value"

    def test_get_env_raises_on_missing(self):
        """Should raise ValueError when the env var is not set."""
        from main import get_env
        with pytest.raises(ValueError, match="not set"):
            get_env("COMPLETELY_NONEXISTENT_KEY_XYZ_999")


# ============================================================
# 4. TESTS FOR EVAL DATASET INTEGRITY
# ============================================================

class TestEvalDataset:
    """Validates the evaluation dataset structure."""

    def test_dataset_not_empty(self):
        """Dataset should have at least one test case."""
        from evaluate_rag import EVAL_DATASET
        assert len(EVAL_DATASET) > 0

    def test_all_entries_have_required_fields(self):
        """Every dataset entry must have 'query' and 'ground_truth'."""
        from evaluate_rag import EVAL_DATASET
        for i, item in enumerate(EVAL_DATASET):
            assert "query" in item, f"Entry {i} missing 'query'"
            assert "ground_truth" in item, f"Entry {i} missing 'ground_truth'"

    def test_no_empty_queries(self):
        """No query should be an empty string."""
        from evaluate_rag import EVAL_DATASET
        for i, item in enumerate(EVAL_DATASET):
            assert item["query"].strip(), f"Entry {i} has empty query"

    def test_no_empty_ground_truths(self):
        """No ground truth should be an empty string."""
        from evaluate_rag import EVAL_DATASET
        for i, item in enumerate(EVAL_DATASET):
            assert item["ground_truth"].strip(), f"Entry {i} has empty ground_truth"
