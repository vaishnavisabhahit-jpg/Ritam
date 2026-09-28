import os
import sys
import json
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from backend.agents.safety import (
    check_emergency_intent,
    verify_grounded_quotes,
    verify_numerical_consistency,
    calculate_readability
)
from backend.agents.llm import generate_medical_answer

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Ritam - Verified Medical AI",
    page_icon="💊",
    layout="wide"
)

# Load Embedding Model and ChromaDB (Cached for speed)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    chroma_dir = os.path.join(BASE_DIR, "data", "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("drug_labels")
    return model, collection

try:
    embedding_model, collection = load_resources()
    db_loaded = True
except Exception as e:
    db_loaded = False

# Sidebar Controls
st.sidebar.title("💊 Ritam Configuration")
st.sidebar.markdown("---")

drug_map = {
    "Metformin": "metformin",
    "Acetaminophen (Paracetamol)": "acetaminophen",
    "Omeprazole": "omeprazole",
    "Amoxicillin": "amoxicillin",
    "Cetirizine": "cetirizine",
    "Levothyroxine Sodium": "levothyroxine"
}

selected_drug_name = st.sidebar.selectbox("Select Drug Context", list(drug_map.keys()))
selected_drug = drug_map[selected_drug_name]

audience = st.sidebar.radio(
    "Target Audience Mode",
    ["patient", "clinician"],
    format_func=lambda x: "👤 Patient Mode" if x == "patient" else "🩺 Clinician Mode"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Guardrails Active")
st.sidebar.caption("• Emergency Keyword Intercept")
st.sidebar.caption("• Direct Vector Quote Matching")
st.sidebar.caption("• Numeric Consistency Audit")
st.sidebar.caption("• Flesch-Kincaid Readability Check")

# Main Interface Header
st.title("Ritam Medical AI Engine")
st.caption("Verified FDA Medical Intelligence powered by RAG and Grounded Evaluation")

if not db_loaded:
    st.error("⚠️ ChromaDB vector store not found. Please run `python backend/ingest.py` locally before deploying.")
    st.stop()

# Query Input
query = st.text_input(
    f"Ask a question regarding {selected_drug_name}:",
    placeholder="e.g., What are common side effects or dosage warnings?"
)

if st.button("Ask Ritam", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        # 1. Emergency Gate Check
        emergency_contact = check_emergency_intent(query)
        if emergency_contact:
            st.error("🚨 **CRITICAL HEALTH EMERGENCY DETECTED**")
            st.write("Your query indicates a potential medical emergency. Seek immediate help.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Emergency Services", emergency_contact["primary"]["tel"])
            with col2:
                st.metric("Poison Control Center", emergency_contact["poison"]["tel"])
        else:
            with st.spinner("Retrieving FDA label context and running safety verification..."):
                # 2. Retrieval
                query_vector = embedding_model.encode([query]).tolist()
                results = collection.query(
                    query_embeddings=query_vector,
                    n_results=3,
                    where={"drug": selected_drug}
                )

                retrieved_chunks = []
                if results and results.get("documents") and results["documents"][0]:
                    for idx, doc_text in enumerate(results["documents"][0]):
                        meta = results["metadatas"][0][idx]
                        retrieved_chunks.append({
                            "chunk_id": results["ids"][0][idx],
                            "text": doc_text,
                            "section": meta.get("section", "General"),
                            "page": meta.get("page", 1)
                        })

                if not retrieved_chunks:
                    st.info(f"No specific matching context found in the official label for {selected_drug_name}.")
                else:
                    # 3. LLM Synthesis
                    answer = generate_medical_answer(
                        query=query,
                        drug=selected_drug,
                        context_chunks=retrieved_chunks,
                        audience=audience
                    )

                    # 4. Guardrails Verification
                    quotes_valid = verify_grounded_quotes(answer, retrieved_chunks)
                    numbers_valid = verify_numerical_consistency(answer, retrieved_chunks)
                    readability = calculate_readability(answer)

                    # Output Dashboard Display
                    st.subheader("Synthesized Medical Guidance")
                    st.info(answer)

                    # Audit Metrics Metrics Bar
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Quote Grounding", "Verified ✅" if quotes_valid else "Unverified ⚠️")
                    m2.metric("Numeric Audit", "Consistent ✅" if numbers_valid else "Mismatch ⚠️")
                    m3.metric("Readability Score", f"Grade {readability.get('flesch_kincaid_grade', 'N/A')}")

                    # Retrieved Sources Expanders
                    with st.expander("📄 View Retrieved FDA Context Chunks"):
                        for c in retrieved_chunks:
                            st.markdown(f"**Section:** {c['section']} *(Page {c['page']})*")
                            st.caption(f'"{c["text"]}"')
                            st.markdown("---")