import streamlit as st
import time
import os
from main import app_multi_agent, get_env
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
