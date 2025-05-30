import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# 📄 Function to extract text from supported files
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(file)
        return '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == 'docx':
        doc = Document(file)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == 'txt':
        return file.read().decode('utf-8')
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
        if 'Question' in df.columns and 'Answer' in df.columns:
            return '\n'.join([f"{i+1}. {q.strip()}?\n{a.strip()}" for i, (q, a) in enumerate(zip(df['Question'], df['Answer']))])
        else:
            # Concatenate all rows if no proper columns found
            return '\n'.join(df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist())
    else:
        st.error("Unsupported file type.")
        return ""

# 🔍 Function to extract Q&A pairs
def extract_qa_pairs(text):
    return re.findall(r"\d+\.\s*(.*?)\?\s*\n\s*(.*?)(?=\n\d+\.|\Z)", text, re.DOTALL)

# 🚀 Streamlit UI
st.title("📄 Document Q&A Chatbot (PDF, DOCX, TXT, Excel Supported)")

uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, XLSX, XLS)", type=['pdf', 'docx', 'txt', 'xls', 'xlsx'])

if uploaded_file:
    text = extract_text(uploaded_file)
    qa_pairs = extract_qa_pairs(text)

    if not qa_pairs:
        st.warning("⚠️ No Q&A pairs found. Ensure your file is formatted like:\n\n1. Question?\nAnswer...")
    else:
        docs = [q.strip() + "\n" + a.strip() for q, a in qa_pairs]

        # ✅ Create embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(docs)
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        # 💬 Q&A Chat Interface
        st.success(f"✅ Found {len(docs)} Q&A pairs.")
        question = st.text_input("❓ Ask a question:")

        if question:
            q_embed = model.encode([question])
            D, I = index.search(np.array(q_embed), k=1)
            st.markdown(f"💬 **Best Match Answer:**\n\n{docs[I[0][0]]}")
