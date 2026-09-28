try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import re
import json
import streamlit as st

# 1. Basic Page Config
st.set_page_config(
    page_title="Ritam Medical AI",
    page_icon="💊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
EMERGENCY_CONFIG_PATH = os.path.join(BASE_DIR, "config", "emergency.json")

# 2. Lazy Loaded Resource Caching (Prevents Cloud OOM Crashes)
@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Connecting to Vector DB...")
def get_chroma_collection():
    import chromadb
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        return client.get_collection("drug_labels")
    except Exception as e:
        st.error(f"Vector Database Notice: {e}")
        return None

# 3. Emergency Helper
def check_emergency(query: str) -> bool:
    keywords = ["overdose", "chest pain", "cannot breathe", "anaphylaxis", "poison", "dying", "fainting"]
    return any(k in query.lower() for k in keywords)

# 4. Header UI
st.title("💊 Ritam AI — FDA Clinical Intelligence")
st.caption("Safety-Gated Medical Question Answering Engine")

# 5. Sidebar Controls
st.sidebar.header("Configuration")
drug_options = {
    "Metformin": "metformin",
    "Acetaminophen": "acetaminophen",
    "Omeprazole": "omeprazole",
    "Amoxicillin": "amoxicillin",
    "Cetirizine": "cetirizine",
    "Levothyroxine": "levothyroxine"
}
selected_label = st.sidebar.selectbox("Select FDA Drug Label", list(drug_options.keys()))
selected_drug = drug_options[selected_label]
audience_mode = st.sidebar.radio("Target Audience", ["Patient", "Doctor"])
audience = "patient" if audience_mode == "Patient" else "clinician"

# 6. Main Query Form
user_query = st.text_input(f"Ask a question about {selected_label}:")

if st.button("Analyze & Verify", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a medical question.")
    elif check_emergency(user_query):
        st.error("🚨 **CRITICAL EMERGENCY DETECTED**")
        st.write("If you are experiencing severe symptoms, call Emergency Services (911) or Poison Control (1-800-222-1222) immediately.")
    else:
        with st.spinner("Processing through safety pipeline..."):
            embed_model = get_embedding_model()
            collection = get_chroma_collection()

            if not collection:
                st.error("ChromaDB vector store not found. Ensure `data/chroma_db` is committed to GitHub.")
            else:
                # Query Vector Database
                query_vec = embed_model.encode([user_query]).tolist()
                results = collection.query(
                    query_embeddings=query_vec,
                    n_results=3,
                    where={"drug": selected_drug}
                )

                chunks = []
                if results and results.get("documents") and results["documents"][0]:
                    for idx, text in enumerate(results["documents"][0]):
                        meta = results["metadatas"][0][idx]
                        chunks.append({"text": text, "section": meta.get("section", "General"), "page": meta.get("page", 1)})

                if not chunks:
                    st.warning(f"No specific label references found for '{selected_label}'.")
                else:
                    st.subheader("📋 Context Chunks Retrieved")
                    for i, c in enumerate(chunks, 1):
                        with st.expander(f"Chunk #{i} — Section: {c['section']} (Page {c['page']})"):
                            st.write(c["text"])