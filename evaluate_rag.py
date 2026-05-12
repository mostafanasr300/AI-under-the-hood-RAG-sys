import os
import sys
import warnings
import asyncio
import time
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric
)
from deepeval import evaluate

# Import our RAG pipeline
from main import app_multi_agent, get_env

load_dotenv()

# ============================================================
# 1. DEEPEVAL CUSTOM MODEL WRAPPER
# ============================================================
class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "llama-3.3-70b-versatile"

# Instantiate our LangChain Groq model
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, api_key=get_env("grog"))
# Wrap it for DeepEval
custom_eval_model = GroqDeepEvalModel(llm)

# ============================================================
# 2. TEST DATASET (Queries + Ground Truth Answers)
# ============================================================
EVAL_DATASET = [
    {
        "query": "What is LoRA and how does it reduce the number of trainable parameters?",
        "ground_truth": "LoRA (Low-Rank Adaptation) freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture. It reduces parameters by only training these small A and B matrices instead of the entire weight matrix W."
    },
    {
        "query": "What is the DPO loss function?",
        "ground_truth": "The DPO (Direct Preference Optimization) loss function is a maximum likelihood objective that optimizes a policy model to satisfy Bradley-Terry preferences. It uses the log-ratio of the policy model and a reference model to ensure the preferred completion has a higher probability than the rejected one without needing a separate reward model."
    },
    {
        "query": "Explain eigenvalue decomposition and its role in PCA.",
        "ground_truth": "Eigenvalue decomposition breaks a square matrix into eigenvalues and eigenvectors. In PCA, the covariance matrix of the data is decomposed. The eigenvectors corresponding to the largest eigenvalues become the principal components, which are used to project the data into a lower-dimensional space while maximizing variance."
    },
    {
        "query": "What are the latest 2026 improvements to Transformer with Mix of Experts?",
        "ground_truth": "According to the GRPO 2026 updates, improvements include Ultra long context RL allowing for a 380K context window to train gpt-oss, as well as the introduction of FP8 precision RL and GRPO in Unsloth."
    },
    {
        "query": "How does GRPO differ from PPO in reinforcement learning?",
        "ground_truth": "GRPO (Group Relative Policy Optimization) differs from PPO by eliminating the need for a critic/value model. Instead, it estimates the baseline by averaging the scores of a group of outputs generated for the same prompt, significantly reducing memory usage during RLHF."
    },
    {
        "query": "What is the rank of a matrix and why is it important in LoRA?",
        "ground_truth": "The rank of a matrix is the dimension of the vector space spanned by its columns or rows. In LoRA, the 'rank' (r) is a hyperparameter that determines the size of the low-rank matrices A and B. A lower rank reduces parameters further but may capture less complexity from the fine-tuning data."
    },
    {
        "query": "What is the Reversal Curse in LLMs?",
        "ground_truth": "The Reversal Curse is a phenomenon where LLMs trained on 'A is B' fail to generalize to 'B is A'. For example, if a model knows Tom Cruise's mother is Mary Lee Pfeiffer, it may fail to answer who Mary Lee Pfeiffer's son is."
    }
]

# Helper to split the concatenated context string back into chunks
def extract_context_chunks(context_string):
    if not context_string.strip():
        return [""]
    # The pipeline outputs "\n--- INTERNAL DATA ---\n[Source: ..."
    # Let's split by "[Source: " to get distinct chunks
    parts = context_string.split("[Source: ")
    chunks = []
    for p in parts:
        if not p.strip() or "--- INTERNAL DATA" in p:
            continue
        chunks.append("[Source: " + p.strip())
    return chunks if chunks else [context_string]

# ============================================================
# 3. RUN DEEPEVAL PIPELINE
# ============================================================

def run_evaluation():
    print("="*60)
    print("RUNNING RAG EVALUATION (DEEPEVAL FRAMEWORK)")
    print("="*60 + "\n")
    
    test_cases = []
    
    for i, item in enumerate(EVAL_DATASET):
        query = item["query"]
        truth = item["ground_truth"]
        
        print(f"Processing Query {i+1}/{len(EVAL_DATASET)}: '{query}'")
        
        # 1. Run our Hybrid RAG Pipeline
        try:
            state = app_multi_agent.invoke({"user_query": query})
            context_string = state.get("context_data", "")
            answer = state.get("final_report", "")
        except Exception as e:
            print(f" Pipeline failed: {e}")
            continue
            
        # 2. Extract list of chunks
        context_chunks = extract_context_chunks(context_string)
        
        # 3. Create DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=truth,
            retrieval_context=context_chunks
        )
        test_cases.append(test_case)
        print(" Fetched pipeline response & context.\n")

    # Instantiate Metrics
    print("Initializing DeepEval Metrics using Groq LLM...")
    metrics = [
        ContextualRelevancyMetric(threshold=0.5, model=custom_eval_model, include_reason=True),
        ContextualPrecisionMetric(threshold=0.5, model=custom_eval_model, include_reason=True),
        ContextualRecallMetric(threshold=0.5, model=custom_eval_model, include_reason=True),
        FaithfulnessMetric(threshold=0.5, model=custom_eval_model, include_reason=True)
    ]
    
    # 4. RUN SEQUENTIAL EVALUATION
    print("\n Starting Sequential Evaluation (Throttling 40s per metric to respect Groq limit)...")
    
    final_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n Evaluating Test Case #{i+1}...")
        case_scores = {"query": test_case.input}
        
        for metric in metrics:
            metric_name = metric.__class__.__name__
            print(f"Measuring {metric_name}...")
            
            try:
                # Measure metric
                metric.measure(test_case)
                case_scores[metric_name] = metric.score
                case_scores[f"{metric_name}_reason"] = getattr(metric, 'reason', 'N/A')
                print(f"Score: {metric.score}")
            except Exception as e:
                print(f"Error: {e}")
                case_scores[metric_name] = 0.0
                
            # Wait 40 seconds between EVERY metric call to stay under 6000 TPM
            time.sleep(40)
            
        final_results.append(case_scores)

    # 5. FINAL REPORT
    print("\n" + "="*60)
    print("  FINAL EVALUATION REPORT")
    print("="*60)
    
    for res in final_results:
        print(f"\nQuery: {res['query']}")
        print(f"|- Relevance:    {res.get('ContextualRelevancyMetric', 0.0)}")
        print(f"|- Precision:    {res.get('ContextualPrecisionMetric', 0.0)}")
        print(f"|- Recall:       {res.get('ContextualRecallMetric', 0.0)}")
        print(f"|- Faithfulness: {res.get('FaithfulnessMetric', 0.0)}")
        #print(f"|- Faithfulness_Reason: {res.get('FaithfulnessMetric_reason', 'N/A')}")
        #print(f"|-ContextualRelevancy_Reason: {res.get('ContextualRelevancyMetric_reason', 'N/A')}")
        print(f"|-ContextualPrecision_Reason: {res.get('ContextualPrecisionMetric_reason', 'N/A')}")
        print(f"|-ContextualRecall_Reason: {res.get('ContextualRecallMetric_reason', 'N/A')}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    run_evaluation()
