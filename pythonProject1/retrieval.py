import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import re
import sys

sys.modules["torch.classes"] = None
CHROMA_PATH = "chroma"
huggingface_api_key = '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'
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
def get_embedding_function():
    local_model_path = "models/embedding/all-MiniLM-L6-v2"
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

def classify_query(query):
    specific_keywords = ["quote", "exact phrase", "specific", "mention of", "quotes", "instance", "instances"]
    if any(keyword in query.lower() for keyword in specific_keywords):
        return "specific"
    return "general"

from langchain_community.llms import HuggingFaceEndpoint  # Import this if not already

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    query_type = classify_query(query_text)
    k = 25 if query_type == "general" else 10
    PROMPT_TEMPLATE = PROMPT_TEMPLATE_GENERAL if query_type == "general" else PROMPT_TEMPLATE_QUOTE

    results = db.similarity_search_with_score(query_text, k=k)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 🔄 Replace Ollama with HuggingFaceEndpoint

    response_text = get_llm().invoke(prompt)

    if query_type == "specific":
        response_text = response_text.strip()
        response_lines = response_text.splitlines()
        response_text = "\n".join([f"• {line.strip()}" for line in response_lines if line.strip()])

    sources = [doc.page_content for doc, _score in results]
    cleaned_sources = [re.sub(r'\s+', ' ', source.strip()) for source in sources]
    formatted_sources = "\n".join([f"{i+1}. {source}" for i, source in enumerate(cleaned_sources)])

    return response_text, formatted_sources