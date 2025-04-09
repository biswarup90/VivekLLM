import os
from pathlib import Path
import streamlit as st

from populate_database import clear_database, process_pdfs, BACKUP_PATH, DATA_PATH
from retrieval import query_rag

# --- Constants ---

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(BACKUP_PATH, exist_ok=True)

# --- Page config ---
st.set_page_config(page_title="Vivekananda PDF RAG", layout="wide")
st.title("🧘‍♂️ Swami Vivekananda RAG System")

# --- Tabs ---
tab1, tab2 = st.tabs(["📤 Upload & Process PDFs", "📜 Ask Questions"])

# --- Tab 1: Upload & Process PDFs ---
with tab1:
    st.header("📤 Upload & Process PDFs into Chroma Vector DB")

    if st.button("🧹 Reset Chroma DB"):
        st.warning(clear_database())

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            file_path = Path(DATA_PATH) / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s) to `{DATA_PATH}/`")

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if pdf_files:
        st.subheader("📂 PDFs Ready for Processing")
        st.write(f"Found **{len(pdf_files)}** file(s) in `{DATA_PATH}/`")
        if st.button("🚀 Process All PDFs"):
            processed = process_pdfs()
            st.success(f"🎉 Done! Processed {len(processed)} PDF(s).")
    else:
        st.info("📁 No PDFs ready. Upload some above.")

# --- Tab 2: Ask Questions ---
with tab2:
    st.header("📜 Ask Swami Vivekananda")

    # Show processed/available PDF files
    st.markdown("#### 🗂️ Available Documents for Querying")
    pdf_files = sorted([f for f in os.listdir(BACKUP_PATH) if f.endswith(".pdf")])

    if pdf_files:
        col1, col2 = st.columns(2)
        for i, file in enumerate(pdf_files):
            with (col1 if i % 2 == 0 else col2):
                st.success(f"📄 {file}")
    else:
        st.info("No processed PDFs available yet.")

    st.markdown("---")

    # Query input
    query_text = st.text_area("Enter your question here:", height=100)

    if st.button("🤔 Ask"):
        if query_text.strip():
            with st.spinner("Thinking..."):
                response, sources = query_rag(query_text)
                st.subheader("🧠 Response")
                st.markdown(response)
                st.subheader("📚 Sources")
                st.markdown(sources)
        else:
            st.warning("Please enter a question.")
