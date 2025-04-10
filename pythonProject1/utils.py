import sys

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from llama_parse import LlamaParse


CHROMA_PATH = "chroma"
DATA_PATH = "data"
BACKUP_PATH = "backup"
OUTPUT_FILE = "data/output.md"

sys.modules["torch.classes"] = None
huggingface_api_key = st.secrets["hf_token"]
PROMPT_TEMPLATE_GENERAL = """
You are an expert on Swami Vivekananda’s works and philosophy. Your task is to summarize his teachings on the given topic while capturing his key insights accurately.

Context:
{context}

Question:
{question}

Instructions:

Provide a concise and faithful summary of Swami Vivekananda’s views on the topic.
Focus on his main ideas without adding personal interpretations or deviations from the context.
Maintain clarity and coherence while staying true to his philosophy.
"""

PROMPT_TEMPLATE_QUOTE = """
You are an expert on Swami Vivekananda’s works. Your task is to extract direct quotes from the provided context that are relevant to the given question.

Context:
{context}

Question:
{question}

Instructions:

Extract only the most relevant quotes from the context.
Do not provide explanations, summaries, or reworded text—only exact quotes.
Present each quote as a separate bullet point for clarity.
Output Format:

"First relevant quote."
"Second relevant quote."
"Third relevant quote."
"""


@st.cache_resource
def load_embedding_function():
    local_model_path = "sentence-transformers/all-mpnet-base-v2"  # "models/embedding/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"trust_remote_code": True}
    )


@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.6,
        huggingfacehub_api_token=huggingface_api_key,
        task="text-generation"
    )

def get_parser():
    return LlamaParse(
        api_key="llx-VfpCxb1VIaZSs4Mhyr0tFFgg2vc4kCt8MkntJ3wUki8xBfkK",
        # "llx-rN27IDyz5rvBEzwQ8FWT8AatIAsN7QiaJs7JMBXK4joDmM7g",#"llx-1nuYutoPQP68zW05zrInDDq4j6YNM2VH4DJdzQcd01ttbK3D",
        result_type="text",
        verbose=False,
    )