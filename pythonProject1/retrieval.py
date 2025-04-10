import re

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from utils import load_embedding_function, CHROMA_PATH, PROMPT_TEMPLATE_GENERAL, PROMPT_TEMPLATE_QUOTE, get_llm


def classify_query(query):
    specific_keywords = ["quote", "exact phrase", "specific", "mention of", "quotes", "instance", "instances"]
    if any(keyword in query.lower() for keyword in specific_keywords):
        return "specific"
    return "general"


def query_rag(query_text: str):
    embedding_function = load_embedding_function()
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
    formatted_sources = "\n".join([f"{i + 1}. {source}" for i, source in enumerate(cleaned_sources)])

    return response_text, formatted_sources
