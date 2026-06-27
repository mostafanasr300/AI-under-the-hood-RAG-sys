import signal

# Monkey-patch signal.signal to catch ValueError in non-main threads (e.g. under Streamlit)
_original_signal = signal.signal
def _safe_signal(signalnum, handler):
    try:
        return _original_signal(signalnum, handler)
    except ValueError:
        # Ignore: signal only works in main thread of the main interpreter
        return None
signal.signal = _safe_signal

import streamlit as st
import time
import os
from main import app_multi_agent, get_env, incremental_add_document
from document_validator import validate_file, validate_topic
from evaluate_rag import extract_context_chunks, custom_eval_model
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric
)

import logging
import warnings

# Suppress all library warnings in terminal
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(
    page_title="Agentic Hybrid RAG Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .agent-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid #4a90e2;
        background-color: #161b22;
    }
    /* Category card styles */
    .cat-card {
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid transparent;
        margin-bottom: 6px;
    }
    .cat-card-ml {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f3460 100%);
        border-color: #4a90e2;
    }
    .cat-card-math {
        background: linear-gradient(135deg, #1a2d1a 0%, #1b4332 100%);
        border-color: #2ecc71;
    }
    .cat-card-selected-ml {
        background: linear-gradient(135deg, #1e3a6e 0%, #2563eb 100%);
        border-color: #60a5fa;
        box-shadow: 0 0 16px rgba(96,165,250,0.35);
    }
    .cat-card-selected-math {
        background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
        border-color: #4ade80;
        box-shadow: 0 0 16px rgba(74,222,128,0.35);
    }
    .cat-icon { font-size: 2.2rem; margin-bottom: 6px; }
    .cat-label { font-weight: 700; font-size: 0.95rem; color: #e2e8f0; }
    .cat-sub   { font-size: 0.72rem; color: #94a3b8; margin-top: 3px; }
    .val-step  { font-size: 0.82rem; padding: 4px 0; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.info("Hybrid Engine: FAISS + BM25 + Cross-Encoder")
    
    st.divider()
    st.subheader("Evaluation Mode")
    eval_enabled = st.toggle("Enable DeepEval Real-time Metrics", value=False)
    
    if eval_enabled:
        st.warning("⚠️ Evaluation adds ~2-3 minutes of delay due to sequential rate-limiting (Groq TPM).")
        st.write("Judge Model: Llama-3.3-70b")

    st.divider()
    st.markdown("### Predefined Test Cases")
    predefined_queries = {
        "What is LoRA?": "LoRA (Low-Rank Adaptation) freezes pre-trained weights and injects trainable matrices into Transformer layers.",
        "DPO Loss Function": "DPO loss is a maximum likelihood objective optimizing policy model to satisfy Bradley-Terry preferences.",
        "Eigenvalue Decomposition": "Eigenvalue decomposition breaks a square matrix into eigenvalues and eigenvectors, used in PCA.",
        "2026 MoE Updates": "GRPO 2026 updates include Ultra long context RL (380K window) and FP8 precision in Unsloth.",
        "GRPO vs PPO": "GRPO eliminates the critic/value model by averaging scores from a group of outputs to estimate the baseline.",
        "Reversal Curse": "The Reversal Curse is a failure of LLMs to generalize 'A is B' to 'B is A' after training.",
        "Matrix Rank in LoRA": "The rank 'r' in LoRA determines the dimensionality of the update matrices A and B, balancing parameter count vs complexity."
    }
    
    selected_predef = st.selectbox("Quick Select Query", options=["None"] + list(predefined_queries.keys()))

    # ── ── ── Upload Section ── ── ──
    st.divider()
    with st.expander("📤 Upload New Document", expanded=False):
        st.markdown(
            "<p style='font-size:0.82rem;color:#94a3b8;margin-bottom:10px;'>"
            "Expand the knowledge base at runtime. The document will be validated "
            "and indexed immediately — no restart needed."
            "</p>",
            unsafe_allow_html=True,
        )

        # ── Category selection ──
        st.markdown(
            "<p style='font-size:0.88rem;font-weight:600;color:#cbd5e1;margin-bottom:6px;'>"
            "Select document category:"
            "</p>",
            unsafe_allow_html=True,
        )

        cat_col1, cat_col2 = st.columns(2)
        if "upload_category" not in st.session_state:
            st.session_state.upload_category = None

        with cat_col1:
            ml_selected = st.session_state.upload_category == "ML"
            card_cls = "cat-card cat-card-selected-ml" if ml_selected else "cat-card cat-card-ml"
            st.markdown(
                f"""<div class="{card_cls}">
                    <div class="cat-icon">🧠</div>
                    <div class="cat-label">Machine Learning</div>
                    <div class="cat-sub">DPO · LoRA · GRPO · LLMs</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Select ML", key="btn_cat_ml", use_container_width=True):
                st.session_state.upload_category = "ML"
                st.rerun()

        with cat_col2:
            math_selected = st.session_state.upload_category == "math"
            card_cls = "cat-card cat-card-selected-math" if math_selected else "cat-card cat-card-math"
            st.markdown(
                f"""<div class="{card_cls}">
                    <div class="cat-icon">📐</div>
                    <div class="cat-label">Absolute Math</div>
                    <div class="cat-sub">Algebra · Calculus · Stats</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Select Math", key="btn_cat_math", use_container_width=True):
                st.session_state.upload_category = "math"
                st.rerun()

        # Show which category is active
        chosen_cat = st.session_state.upload_category
        if chosen_cat == "ML":
            st.success("🧠 **Machine Learning** selected")
        elif chosen_cat == "math":
            st.success("📐 **Absolute Math** selected")
        else:
            st.info("☝️ Please select a category above.")

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        # ── File uploader ──
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )

        upload_btn = st.button(
            "🚀 Validate & Index Document",
            key="upload_btn",
            use_container_width=True,
            disabled=(uploaded_file is None or chosen_cat is None),
        )

        # ── Upload pipeline ──
        if upload_btn and uploaded_file and chosen_cat:
            file_bytes = uploaded_file.read()
            filename   = uploaded_file.name

            with st.status("🔍 Processing document...", expanded=True) as upload_status:

                # Step 1 — File Validation
                st.write("**Step 1/3** — File validation...")
                file_ok, file_err = validate_file(file_bytes, filename)

                if not file_ok:
                    upload_status.update(
                        label="❌ Validation failed", state="error", expanded=True
                    )
                    st.error(f"🚫 **File rejected:** {file_err}")
                    st.stop()

                st.write("✅ File checks passed (format · size · readability · dedup)")

                # Step 2 — Topic Validation
                st.write(f"**Step 2/3** — Topic classification (selected: **{chosen_cat}**)...")
                try:
                    groq_key = get_env("grog")
                    topic_result = validate_topic(
                        file_bytes=file_bytes,
                        filename=filename,
                        selected_category=chosen_cat,
                        groq_api_key=groq_key,
                    )
                except Exception as exc:
                    upload_status.update(
                        label="❌ Topic check failed", state="error", expanded=True
                    )
                    st.error(f"🚫 **Topic validation error:** {exc}")
                    st.stop()

                if not topic_result["accepted"]:
                    upload_status.update(
                        label="❌ Document rejected", state="error", expanded=True
                    )
                    st.error(
                        f"🚫 **Document rejected** — the content does not belong to ML or Math.\n\n"
                        f"**Confidence:** {topic_result['confidence']:.0%}\n\n"
                        f"**Reason:** {topic_result['reason']}"
                    )
                    st.stop()

                # Detected category is valid — may have been redirected
                actual_cat = topic_result["actual_category"]
                if topic_result["redirected"]:
                    st.warning(
                        f"⚠️ **Category mismatch detected!**\n\n"
                        f"You selected **{chosen_cat}** but the document was identified as "
                        f"**{actual_cat}** "
                        f"(confidence: {topic_result['confidence']:.0%}).\n\n"
                        f"The file will be indexed under **{actual_cat}** instead. "
                        f"Reason: {topic_result['reason']}"
                    )
                else:
                    st.write(
                        f"✅ Topic verified as **{actual_cat}** "
                        f"(confidence: {topic_result['confidence']:.0%})"
                    )

                # Step 3 — Save & Incremental Index
                st.write("**Step 3/3** — Saving and indexing...")

                # Save PDF to the correct Data/ subfolder
                save_dir = os.path.join("Data", actual_cat)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)

                with open(save_path, "wb") as out:
                    out.write(file_bytes)

                # Incremental indexing (no rebuild)
                try:
                    result = incremental_add_document(
                        file_path=save_path,
                        category=actual_cat,
                    )
                except Exception as exc:
                    # Clean up saved file if indexing fails
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    upload_status.update(
                        label="❌ Indexing failed", state="error", expanded=True
                    )
                    st.error(f"🚫 **Indexing error:** {exc}")
                    st.stop()

                upload_status.update(
                    label="✅ Document indexed successfully!",
                    state="complete",
                    expanded=False,
                )

            # ── Success summary ──
            icon = "🧠" if actual_cat == "ML" else "📐"
            redirect_note = (
                f"\n> ⚠️ Redirected from **{chosen_cat}** → **{actual_cat}** "
                "based on document content."
                if topic_result["redirected"] else ""
            )
            st.success(
                f"{icon} **{filename}** has been added to the **{actual_cat}** knowledge base!\n\n"
                f"- **New chunks indexed:** {result['chunks_added']}\n"
                f"- **Total knowledge base size:** {result['total_chunks']} chunks\n"
                f"- **Immediately available** for search & retrieval."
                f"{redirect_note}"
            )
            # Reset uploader state
            st.session_state.upload_category = None


# --- Main Interface ---
st.title("🤖 Agentic Hybrid RAG Engine")
st.markdown("Query your private PDF knowledge base with autonomous self-correction.")

# Input Row
col1, col2 = st.columns([4, 1])

with col1:
    default_input = "" if selected_predef == "None" else selected_predef
    user_query = st.text_input("Enter your technical question:", value=default_input, placeholder="e.g. How does eigenvalue decomposition work?")

with col2:
    ground_truth = ""
    if eval_enabled:
        if selected_predef != "None":
            ground_truth = predefined_queries[selected_predef]
        ground_truth = st.text_area("Ground Truth (for Eval)", value=ground_truth, placeholder="Required for metrics...")

run_btn = st.button("🚀 Execute Pipeline", use_container_width=True)

if run_btn and user_query:
    # 1. RUN PIPELINE
    with st.status("🧠 Agentic Journey in Progress...", expanded=True) as status:
        st.write("➡️ **Router Agent** analyzing query...")
        start_time = time.time()
        
        # Invoke LangGraph
        state = app_multi_agent.invoke({"user_query": user_query})
        
        st.write("➡️ **Internal Analyst** performed hybrid retrieval...")
        if state.get("needs_retry"):
            st.write(f"➡️ **Reviewer Agent** detected gaps. Triggering **Emergency Web Search** for: '{state['new_search_query']}'")
        else:
            st.write("➡️ **Reviewer Agent** validated the internal response.")
            
        st.write("➡️ **Synthesizer Agent** finalized the report.")
        status.update(label="✅ Pipeline Completed!", state="complete", expanded=False)

    # 2. DISPLAY RESULTS
    st.divider()
    
    res_col, debug_col = st.columns([1.5, 1])
    
    with res_col:
        st.markdown("""
            <div style="background-color: #4a90e2; color: white; padding: 5px 15px; border-radius: 5px 5px 0 0; font-weight: bold; width: fit-content; font-size: 0.8rem;">
                FINAL AI RESPONSE
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background-color: #f0f2f6; color: #1f2937; padding: 25px; border-radius: 0 10px 10px 10px; border: 1px solid #e5e7eb; line-height: 1.6; font-size: 1.05rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                {state["final_report"]}
            </div>
            """, unsafe_allow_html=True)
        st.write("") # Spacer
    
    with debug_col:
        st.subheader("📂 Retrieved Evidence")
        # Use a scrollable container for better reading
        with st.container(height=450, border=True):
            st.markdown(state["context_data"] if state["context_data"] else "*No internal data found.*")
        
        st.subheader("💡 Topic Tags")
        st.write(", ".join([f"`{t}`" for t in state["topics"]]))

    # 3. REAL-TIME EVALUATION
    if eval_enabled:
        if not ground_truth:
            st.error("❌ Please provide a Ground Truth answer to run evaluation metrics.")
        else:
            st.divider()
            st.subheader("📊 DeepEval Metric Dashboard")
            st.caption("Judge: Llama-3.3-70b | Throttled for Groq API limits")
            
            # Extract chunks for metrics
            context_chunks = extract_context_chunks(state["context_data"])
            test_case = LLMTestCase(
                input=user_query,
                actual_output=state["final_report"],
                expected_output=ground_truth,
                retrieval_context=context_chunks
            )
            
            m_relevance = ContextualRelevancyMetric(threshold=0.5, model=custom_eval_model, include_reason=True)
            m_precision = ContextualPrecisionMetric(threshold=0.5, model=custom_eval_model, include_reason=True)
            m_recall = ContextualRecallMetric(threshold=0.5, model=custom_eval_model, include_reason=True)
            m_faithfulness = FaithfulnessMetric(threshold=0.5, model=custom_eval_model, include_reason=True)
            
            metrics = [m_relevance, m_precision, m_recall, m_faithfulness]
            icons = ["🎯", "🔍", "🧠", "🛡️"]
            
            met_cols = st.columns(4)
            
            for i, metric in enumerate(metrics):
                with met_cols[i]:
                    metric_name = metric.__class__.__name__.replace("Metric", "").replace("Contextual", "")
                    with st.spinner(f"Measuring {metric_name}..."):
                        try:
                            metric.measure(test_case)
                            score = metric.score
                            
                            # Dynamic color coding
                            color = "#2ecc71" if score >= 0.7 else "#f1c40f" if score >= 0.4 else "#e74c3c"
                            
                            st.markdown(f"""
                            <div style="background-color: #1e2130; padding: 15px; border-radius: 10px; border-top: 5px solid {color}; text-align: center;">
                                <div style="font-size: 24px;">{icons[i]}</div>
                                <div style="font-weight: bold; color: #a3a8b8;">{metric_name}</div>
                                <div style="font-size: 32px; font-weight: bold; color: {color};">{score:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("Show Reasoning"):
                                st.caption(getattr(metric, 'reason', 'N/A'))
                        except Exception as e:
                            st.error(f"Error: {e}")
                        
                        # Sequential throttling for Groq
                        if i < len(metrics) - 1:
                            time.sleep(38)

st.markdown("---")
st.caption("Built with LangGraph + DeepEval + Streamlit | Judge: Llama-3.3-70b")
