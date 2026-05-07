import os
import sys
import glob
import json
import hashlib
import pickle
import warnings
import numpy as np
from typing import List, TypedDict

# Fix Windows console encoding for emoji/unicode output
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Suppress annoying HuggingFace and LangChain FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()


def get_env(key):
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable '{key}' not set.")
    return value


# ============================================================
# 1. HYBRID RETRIEVER — BM25 + FAISS + Cross-Encoder Reranker
# ============================================================

class HybridRetriever:
    """
    Three-stage hybrid retrieval pipeline:
    
      Stage 1 — Dual Retrieval:
          FAISS (dense / semantic similarity) + BM25 (sparse / keyword matching)
          Each retriever pulls `initial_k` candidates independently.
    
      Stage 2 — Threshold Gating:
          FAISS results gated by max L2 distance  (lower  = more similar).
          BM25  results gated by min relevance score (higher = more relevant).
          Results below threshold are discarded before fusion.
    
      Stage 3 — Reciprocal Rank Fusion + Cross-Encoder Reranking:
          Surviving candidates are merged via RRF scoring, then a cross-encoder
          re-scores each (query, passage) pair for final ranking.
    """

    def __init__(self, vectorstore, bm25_index, documents, cross_encoder,
                 vector_threshold=1.5, bm25_threshold=1.0,
                 initial_k=20, final_k=6):
        self.vectorstore = vectorstore
        self.bm25 = bm25_index
        self.documents = documents          # full list of LangChain Document chunks
        self.cross_encoder = cross_encoder
        self.vector_threshold = vector_threshold
        self.bm25_threshold = bm25_threshold
        self.initial_k = initial_k
        self.final_k = final_k

    def invoke(self, query: str):
        """Run the full hybrid retrieval pipeline and return top-k documents."""

        # ── Stage 1a: FAISS dense retrieval (with L2 scores) ──
        faiss_results = self.vectorstore.similarity_search_with_score(
            query, k=self.initial_k
        )

        # ── Stage 1b: BM25 sparse retrieval (with BM25 scores) ──
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:self.initial_k]
        bm25_results = [(self.documents[i], float(bm25_scores[i]))
                        for i in top_bm25_idx]

        # ── Stage 2: Threshold Gating ──
        # FAISS: keep only if L2 distance ≤ threshold (lower = better)
        faiss_gated = [(doc, score) for doc, score in faiss_results
                       if score <= self.vector_threshold]
        # BM25: keep only if relevance score ≥ threshold (higher = better)
        bm25_gated = [(doc, score) for doc, score in bm25_results
                      if score >= self.bm25_threshold]

        # ── Diagnostic logging ──
        print(f"      [FAISS]  {len(faiss_results)} candidates → "
              f"{len(faiss_gated)} passed gate (threshold ≤ {self.vector_threshold})")
        if faiss_results:
            scores = [s for _, s in faiss_results]
            print(f"               score range: {min(scores):.3f} – {max(scores):.3f}")
        print(f"      [BM25]   {len(bm25_results)} candidates → "
              f"{len(bm25_gated)} passed gate (threshold ≥ {self.bm25_threshold})")
        if bm25_gated:
            scores = [s for _, s in bm25_gated]
            print(f"               score range: {min(scores):.3f} – {max(scores):.3f}")

        # ── Stage 3a: Reciprocal Rank Fusion ──
        combined = self._reciprocal_rank_fusion(faiss_gated, bm25_gated)

        if not combined:
            # Fallback: return raw FAISS top-k if both gates killed everything
            print("      [GATE]   ⚠️ Both gates empty — falling back to raw FAISS top-k")
            return [doc for doc, _ in faiss_results[:self.final_k]]

        print(f"      [RRF]    {len(combined)} unique candidates after fusion")

        # ── Stage 3b: Cross-Encoder Reranking ──
        pairs = [(query, doc.page_content) for doc in combined]
        ce_scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(combined, ce_scores),
                        key=lambda x: x[1], reverse=True)

        print(f"      [RERANK] Cross-encoder top score: {ranked[0][1]:.4f} | "
              f"bottom: {ranked[-1][1]:.4f}")

        # Drop any documents that score worse than -4 (allow mildly negative matches)
        final_docs = [doc for doc, score in ranked if score >= -4.0]
        
        if not final_docs:
            print("      [RERANK] ⚠️ All documents scored worse than -4. Local data is irrelevant!")

        return final_docs[:self.final_k]

    def _reciprocal_rank_fusion(self, faiss_results, bm25_results, rrf_k=60):
        """Merge results from both retrievers using RRF scoring."""
        doc_scores = {}

        for rank, (doc, _) in enumerate(faiss_results):
            key = id(doc)
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if key not in doc_scores:
                doc_scores[key] = (doc, rrf_score)
            else:
                doc_scores[key] = (doc, doc_scores[key][1] + rrf_score)

        for rank, (doc, _) in enumerate(bm25_results):
            # Deduplicate by matching page_content prefix
            matched_key = None
            for existing_key, (existing_doc, _) in doc_scores.items():
                if existing_doc.page_content[:200] == doc.page_content[:200]:
                    matched_key = existing_key
                    break

            rrf_score = 1.0 / (rrf_k + rank + 1)
            if matched_key is not None:
                old = doc_scores[matched_key]
                doc_scores[matched_key] = (old[0], old[1] + rrf_score)
            else:
                doc_scores[id(doc)] = (doc, rrf_score)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


# ============================================================
# 2. HASH-BASED SMART CACHING + KNOWLEDGE BASE BUILDER
# ============================================================

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def build_knowledge_base():
    """
    Build the full hybrid retrieval stack:
      1. FAISS vector store  (with hash-based incremental caching)
      2. BM25 keyword index  (rebuilt from cached document chunks)
      3. Cross-encoder model (loaded once)
    
    Returns a HybridRetriever instance with a single .invoke() interface.
    """
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    CACHE_DIR = "faiss_index_cache"
    HASH_FILE = os.path.join(CACHE_DIR, "file_hashes.json")
    INDEX_FILE = os.path.join(CACHE_DIR, "index.faiss")
    CHUNKS_FILE = os.path.join(CACHE_DIR, "document_chunks.pkl")

    # --- Scan all PDFs across both data categories ---
    DATA_DIRS = [os.path.join("Data", "math"), os.path.join("Data", "ML")]
    pdf_files = []
    for d in DATA_DIRS:
        if os.path.isdir(d):
            pdf_files.extend(glob.glob(os.path.join(d, "*.pdf")))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in Data/math or Data/ML.")

    # --- Compute current file hashes ---
    current_hashes = {}
    for f in pdf_files:
        norm = os.path.normpath(f)
        current_hashes[norm] = compute_file_hash(f)

    print(f"📂 Found {len(pdf_files)} PDF(s) across data categories.")

    # --- Load cached hashes (if any) ---
    cached_hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as fh:
            cached_hashes = json.load(fh)

    # --- Diff: classify every file ---
    new_files = [f for f in current_hashes if f not in cached_hashes]
    modified_files = [f for f in current_hashes
                      if f in cached_hashes and current_hashes[f] != cached_hashes[f]]
    deleted_files = [f for f in cached_hashes if f not in current_hashes]
    unchanged_files = [f for f in current_hashes
                       if f in cached_hashes and current_hashes[f] == cached_hashes[f]]

    # --- Status report ---
    if new_files:
        print(f"   🆕  New files:      {[os.path.basename(f) for f in new_files]}")
    if modified_files:
        print(f"   ✏️  Modified files: {[os.path.basename(f) for f in modified_files]}")
    if deleted_files:
        print(f"   🗑️  Deleted files:  {[os.path.basename(f) for f in deleted_files]}")
    if unchanged_files:
        print(f"   ✅ Unchanged:       {[os.path.basename(f) for f in unchanged_files]}")

    cache_exists = os.path.exists(INDEX_FILE)
    chunks_cached = os.path.exists(CHUNKS_FILE)
    files_to_embed = new_files + modified_files

    # ---- FAST PATH: nothing changed ----
    if not files_to_embed and not deleted_files and cache_exists and chunks_cached:
        print("⚡ All files unchanged — loading from cache...")
        vectorstore = FAISS.load_local(CACHE_DIR, embeddings,
                                       allow_dangerous_deserialization=True)
        with open(CHUNKS_FILE, "rb") as f:
            all_documents = pickle.load(f)
        print(f"   Loaded {len(all_documents)} cached chunks for BM25.")

    # ---- FULL REBUILD: modifications or deletions detected ----
    elif deleted_files or modified_files:
        print("🔄 Modifications/deletions detected — full rebuild required...")
        files_to_embed = list(current_hashes.keys())
        vectorstore, all_documents = _process_and_build(
            files_to_embed, embeddings, None)
        _save_cache(CACHE_DIR, vectorstore, all_documents,
                    current_hashes, HASH_FILE, CHUNKS_FILE)

    # ---- MERGE PATH: only brand-new files ----
    elif new_files and cache_exists and chunks_cached:
        print(f"📥 Embedding {len(new_files)} new file(s) only — merging...")
        existing_store = FAISS.load_local(CACHE_DIR, embeddings,
                                          allow_dangerous_deserialization=True)
        with open(CHUNKS_FILE, "rb") as f:
            existing_docs = pickle.load(f)
        vectorstore, new_docs = _process_and_build(
            new_files, embeddings, existing_store)
        all_documents = existing_docs + new_docs
        _save_cache(CACHE_DIR, vectorstore, all_documents,
                    current_hashes, HASH_FILE, CHUNKS_FILE)

    # ---- FIRST-TIME BUILD ----
    else:
        print("🏗️  Building index from scratch...")
        files_to_embed = list(current_hashes.keys())
        vectorstore, all_documents = _process_and_build(
            files_to_embed, embeddings, None)
        _save_cache(CACHE_DIR, vectorstore, all_documents,
                    current_hashes, HASH_FILE, CHUNKS_FILE)

    # --- Build BM25 from document chunks (always fast, no embedding needed) ---
    print("🔤 Building BM25 keyword index...")
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"   BM25 index built over {len(all_documents)} chunks.")

    # --- Load Cross-Encoder reranker ---
    print("🎯 Loading Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("   Cross-Encoder ready.\n")

    # --- Assemble the Hybrid Retriever ---
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        documents=all_documents,
        cross_encoder=cross_encoder,
        vector_threshold=1.5,   # Max FAISS L2 distance (lower = more similar)
        bm25_threshold=1.0,     # Min BM25 score (higher = more relevant)
        initial_k=20,           # Candidates pulled from each retriever
        final_k=6               # Final docs returned after reranking
    )

    return retriever


def _process_and_build(files_to_embed, embeddings, existing_store):
    """Load PDFs, chunk them, build/merge FAISS index."""
    docs = []
    for file_path in files_to_embed:
        try:
            category = os.path.basename(os.path.dirname(file_path))
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["category"] = category
                doc.metadata["source_file"] = os.path.basename(file_path)
            docs.extend(file_docs)
        except Exception as e:
            print(f"   ⚠️ Failed to load {file_path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(docs)
    print(f"   📄 Processed {len(documents)} chunks from {len(files_to_embed)} file(s).")

    if existing_store is not None:
        new_store = FAISS.from_documents(documents, embeddings)
        existing_store.merge_from(new_store)
        vectorstore = existing_store
        print("✅ Merged new embeddings into existing FAISS index.")
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("✅ FAISS index built successfully.")

    return vectorstore, documents


def _save_cache(cache_dir, vectorstore, documents, hashes, hash_file, chunks_file):
    """Persist FAISS index + document chunks + hash manifest."""
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(cache_dir)
    with open(chunks_file, "wb") as f:
        pickle.dump(documents, f)
    with open(hash_file, "w") as fh:
        json.dump(hashes, fh, indent=2)
    print("💾 FAISS index + document chunks + file hashes cached locally.")


# --- Boot ---
print("Booting up AI Knowledge Engine...\n")
retriever = build_knowledge_base()


# ============================================================
# 3. MULTI-AGENT STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    user_query: str
    topics: List[str]
    context_data: str
    final_report: str
    loop_count: int
    needs_retry: bool
    new_search_query: str


llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2,
               api_key=get_env("grog"))


# ============================================================
# 4. SPECIALISED AGENT NODES
# ============================================================

def router_agent_node(state: AgentState):
    """Agent 1: Extracts key AI / Math topics from the user query."""
    print("=> 🧠 Routing Agent: Analysing the query...")
    prompt = ChatPromptTemplate.from_template(
        "You are a routing assistant for an AI & Mathematics knowledge base.\n"
        "The knowledge base covers two categories:\n"
        "  • math  — Linear Algebra, Calculus, Probability, Optimization\n"
        "  • ML    — DPO, GRPO, LoRA, Fine-tuning, Reinforcement Learning\n"
        "Extract the key technical topics from the user's query.\n"
        "Output ONLY a valid JSON list of strings. Example: [\"LoRA\", \"low-rank\"]\n\n"
        "Query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["user_query"]}).content
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        topics = json.loads(clean)
        if not isinstance(topics, list):
            topics = [state["user_query"]]
    except Exception:
        topics = [state["user_query"]]

    return {"topics": topics, "context_data": "", "loop_count": 0}


def internal_analyst_node(state: AgentState):
    """Agent 2: Searches the hybrid knowledge base (BM25 + FAISS + Reranker)."""
    print("=> 📚 Internal Analyst: Hybrid retrieval in progress...")
    current_context = state.get("context_data", "")
    
    # Do exactly ONE targeted retrieval using the raw user query for best Cross-Encoder accuracy
    docs = retriever.invoke(state["user_query"])
    
    internal_results = ""
    if docs:
        info = "\n\n".join([
            f"[Source: {d.metadata.get('source_file','?')} | "
            f"Category: {d.metadata.get('category','?')}]\n{d.page_content}"
            for d in docs
        ])
        internal_results = f"\n--- INTERNAL DATA ---\n{info}\n\n"

    return {"context_data": current_context + internal_results}




def synthesizer_node(state: AgentState):
    """Agent 4: Combines all context into a coherent technical explanation."""
    print("=> 📝 Synthesizer Agent: Crafting the response...")
    
    # HARD SHORT-CIRCUIT: Prevent LLM hallucination entirely if no data was found
    if not state.get("context_data", "").strip():
        print("   [Synthesizer] ⚠️ Context is empty. Bypassing LLM to enforce strict fallback.")
        return {"final_report": f"INSUFFICIENT_DATA: {state['user_query']}"}
        
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI & Mathematics educator.\n"
        "Based ONLY on the Context Data below, provide a clear, detailed, "
        "and technically accurate explanation.\n"
        "Guidelines:\n"
        "1. Prioritise 'INTERNAL DATA' (textbooks/papers) for core concepts "
        "and mathematical formulations.\n"
        "2. If 'EMERGENCY SEARCH' results are provided in the context, integrate them smoothly to answer the query.\n"
        "3. Include mathematical notation where relevant.\n"
        "4. Structure your response with clear headings and logical flow.\n"
        "5. CRITICAL: You are STRICTLY FORBIDDEN from answering using outside knowledge. If the exact answer is not in the Context Data, you MUST reply with EXACTLY 'INSUFFICIENT_DATA: <a short 6-word Google search query to find the answer>' and nothing else. Do not apologize or explain.\n"
        "6. Attribute information to its source when possible.\n\n"
        "Context Data:\n{context}\n\n"
        "User Query:\n{query}"
    )
    chain = prompt | llm
    report = chain.invoke({
        "context": state["context_data"],
        "query": state["user_query"]
    }).content

    return {"final_report": report}


def reviewer_node(state: AgentState):
    """Agent 5: Reviews the response for completeness and accuracy."""
    print("=> 🔍 Reviewer Agent: Auditing the response...")
    loop_count = state.get("loop_count", 0)

    if loop_count >= 2:
        print("   [Reviewer] Max depth reached (2 loops). Accepting.")
        return {"needs_retry": False}

    report_text = state.get("final_report", "")
    
    # Catch the explicit missing data flag from the Synthesizer
    if "INSUFFICIENT_DATA:" in report_text:
        query_part = report_text.split("INSUFFICIENT_DATA:")[-1].strip()
        print(f"   [Reviewer] ❌ Missing data detected → Emergency Search: '{query_part}'")
        return {"needs_retry": True, "new_search_query": query_part, "loop_count": loop_count + 1}

    prompt = ChatPromptTemplate.from_template(
        "Did the following response answer the user's question with real data?\n\n"
        "User Question: {query}\n"
        "Response: {report}\n\n"
        "If YES: respond with exactly one word: PASS\n"
        "If NO: respond with a search query (MAX 6 words) to find missing info.\n"
        "Output ONLY 'PASS' or the search query, nothing else."
    )
    chain = prompt | llm
    response = chain.invoke({
        "query": state["user_query"],
        "report": report_text
    }).content.strip()

    if "\n" in response:
        response = response.strip().split("\n")[-1].strip()
    words = response.split()
    if len(words) > 8:
        response = " ".join(words[:8])

    if "PASS" in response.upper():
        print("   [Reviewer] Response is complete. PASSED.")
        return {"needs_retry": False, "loop_count": loop_count + 1}
    else:
        print(f"   [Reviewer] ❌ Gaps detected → Emergency Search: '{response}'")
        return {"needs_retry": True, "new_search_query": response,
                "loop_count": loop_count + 1}


def emergency_web_node(state: AgentState):
    """Agent 6: Surgical web search for missing information."""
    print("=> 🚑 Emergency Web Agent: Fetching missing data...")
    client = TavilyClient(api_key=get_env("TAVILY_API_KEY"))
    try:
        search = client.search(query=state["new_search_query"],
                               search_depth="advanced")
        text = ""
        for res in search.get("results", [])[:5]:
            text += res.get("content", "")[:800] + "...\n"
        new_data = (f"\n\n[EMERGENCY SEARCH RESULTS FOR: "
                    f"'{state['new_search_query']}']:\n{text}\n")
    except Exception:
        new_data = "\n\n[EMERGENCY SEARCH FAILED]\n"

    return {"context_data": state.get("context_data", "") + new_data}


def should_loop(state: AgentState):
    """Conditional edge router for the review loop."""
    if state.get("needs_retry"):
        return "retry"
    return "end"


# ============================================================
# 5. BUILD THE CYCLIC MULTI-AGENT PIPELINE
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("Router", router_agent_node)
workflow.add_node("Internal_Analyst", internal_analyst_node)
workflow.add_node("Synthesizer", synthesizer_node)
workflow.add_node("Reviewer", reviewer_node)
workflow.add_node("Emergency_Web", emergency_web_node)

workflow.set_entry_point("Router")
workflow.add_edge("Router", "Internal_Analyst")
workflow.add_edge("Internal_Analyst", "Synthesizer")
workflow.add_edge("Synthesizer", "Reviewer")
workflow.add_conditional_edges("Reviewer", should_loop,
                               {"retry": "Emergency_Web", "end": END})
workflow.add_edge("Emergency_Web", "Synthesizer")

app_multi_agent = workflow.compile()

# Generate visual diagram
try:
    with open("langgraph_multi_agent_pipeline.png", "wb") as f:
        f.write(app_multi_agent.get_graph().draw_mermaid_png())
except Exception:
    pass


# ============================================================
# 6. INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 AI & MATHEMATICS KNOWLEDGE ENGINE")
    print("=" * 60)
    print("📚 Math  : Linear Algebra, Calculus, Probability, Optimization")
    print("🧠 ML    : DPO, GRPO, LoRA — research papers")
    print("🔀 Hybrid: BM25 + FAISS + Cross-Encoder Reranker")
    print("🔄 Self-Correcting Reviewer Loop Active")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower().strip() in ["exit", "quit", "q", "e"]:
                break
            if not user_input.strip():
                continue

            state = app_multi_agent.invoke({"user_query": user_input})

            print("\n📝 Final Response:")
            print(state["final_report"])
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"\n⚠️ Pipeline Error: {e}")