import streamlit as st
import os
import shutil
import re
import nest_asyncio
from llama_index.core.schema import Document
from llama_parse import LlamaParse
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import sys
from langchain_huggingface import HuggingFaceEmbeddings
# This prevents Streamlit from trying to inspect `torch.classes`
sys.modules["torch.classes"] = None
nest_asyncio.apply()

CHROMA_PATH = "chroma"
DATA_PATH = "data"
BACKUP_PATH = "backup"
OUTPUT_FILE = "data/output.md"





def clean_text(text):
    # ... (Keep your full clean_text function here)
    # truncated for brevity
    return text.strip()


def parse_document(path):
    print(f"parse_document: {path}")
    parser = LlamaParse(
        api_key="llx-VfpCxb1VIaZSs4Mhyr0tFFgg2vc4kCt8MkntJ3wUki8xBfkK", #"llx-rN27IDyz5rvBEzwQ8FWT8AatIAsN7QiaJs7JMBXK4joDmM7g",#"llx-1nuYutoPQP68zW05zrInDDq4j6YNM2VH4DJdzQcd01ttbK3D",
        result_type="text",
        verbose=False,
    )
    parsed_documents = parser.load_data(path)
    cleaned_documents = [
        Document(text=clean_text(clean_text(doc.get_content())), metadata=doc.metadata)
        for doc in parsed_documents
    ]
    return cleaned_documents


def load_documents_llama_parser(file_path):
    parsed_document = parse_document(file_path)
    with open(OUTPUT_FILE, 'w') as f:
        for doc in parsed_document:
            f.write(doc.text + '\n')
    loader = UnstructuredMarkdownLoader(OUTPUT_FILE)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks, existing_ids=None):
    if not existing_ids:
        last_page_id = None
        current_chunk_index = 0
    else:
        last_page_id = existing_ids[-1].rsplit(":", 1)[0]
        current_chunk_index = int(existing_ids[-1].split(":")[-1])
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    return chunks



@st.cache_resource
def load_embedding_function():
    local_model_path = "models/embedding/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=local_model_path,
        model_kwargs={"trust_remote_code": True}
    )
@st.cache_resource
def add_to_chroma(_chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=load_embedding_function())
    existing_ids = list(db.get(include=[])["ids"])
    chunks_with_ids = calculate_chunk_ids(_chunks, existing_ids)

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        return len(new_chunks)
    return 0


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return "Database cleared."


def process_pdfs():
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)

    processed_files = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file_name)
            st.write(f"📄 Processing **{file_name}**")
            documents = load_documents_llama_parser(file_path)
            chunks = split_documents(documents)
            added_count = add_to_chroma(chunks)

            shutil.move(file_path, os.path.join(BACKUP_PATH, file_name))
            processed_files.append((file_name, added_count))
            st.success(f"✅ Moved {file_name} to {BACKUP_PATH}, Added {added_count} new chunks")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        st.info(f"🧹 Removed intermediate file: {OUTPUT_FILE}")

    return processed_files