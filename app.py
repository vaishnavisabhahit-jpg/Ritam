try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import json
import numpy as np
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Ritam AI — Medical Intelligence",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Get API Key
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@st.cache_data
def get_gemini_embedding(text: str):
    if not client:
        return None
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embedding.values

@st.cache_data
def load_drug_chunks_with_embeddings(drug_id: str):
    json_path = os.path.join(PROCESSED_DIR, f"{drug_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        for chunk in chunks:
            if "embedding" not in chunk:
                chunk["embedding"] = get_gemini_embedding(chunk["text"])
        return chunks
    return []

def check_emergency(query: str) -> bool:
    keywords = ["overdose", "chest pain", "cannot breathe", "anaphylaxis", "poison", "dying", "fainting"]
    return any(k in query.lower() for k in keywords)

# --- UI Setup ---
st.title("⚡ Ritam AI — FDA Medical Intelligence")
st.caption("Powered by Gemini `text-embedding-004` & `gemini-2.0-flash`")

st.sidebar.header("Configuration")
drug_options = {
    "Metformin": "metformin",
    "Acetaminophen": "acetaminophen",
    "Omeprazole": "omeprazole",
    "Amoxicillin": "amoxicillin",
    "Cetirizine": "cetirizine",
    "Levothyroxine": "levothyroxine"
}
selected_label = st.sidebar.selectbox("Select Target Drug", list(drug_options.keys()))
selected_drug = drug_options[selected_label]
audience_mode = st.sidebar.radio("Target Audience", ["Patient", "Doctor"])
audience = "patient" if audience_mode == "Patient" else "clinician"

user_query = st.text_input(f"Ask a question about {selected_label}:", placeholder="e.g. What are common side effects?")

if st.button("🚀 Analyze & Verify", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif check_emergency(user_query):
        st.error("🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**")
        st.write("If you are experiencing severe acute distress, contact Emergency Services (911) or Poison Control (1-800-222-1222) immediately.")
    elif not client:
        st.error("⚠️ GEMINI_API_KEY is missing from Secrets.")
    else:
        with st.spinner("Executing High-Accuracy Semantic Search..."):
            chunks = load_drug_chunks_with_embeddings(selected_drug)
            query_embedding = get_gemini_embedding(user_query)

            if query_embedding and chunks:
                scored_chunks = []
                for c in chunks:
                    if c.get("embedding"):
                        sim = cosine_similarity(query_embedding, c["embedding"])
                        scored_chunks.append((sim, c))
                
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_chunks = [c for sim, c in scored_chunks[:3]]

                context_text = "\n\n".join([
                    f"--- Section: {c.get('section', 'General')} (Page {c.get('page', 1)}) ---\n{c.get('text', '')}"
                    for c in top_chunks
                ])

                prompt = f"Target Drug: {selected_label}\nTarget Audience: {audience}\nUser Question: {user_query}\n\nFDA Context:\n{context_text}"
                
                system_instruction = (
                    "You are Ritam AI, a safety-critical medical assistant. Answer using ONLY the provided FDA label context.\n"
                    "Do not guess or assume. If the info is missing, state that clearly."
                )

                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.1
                        )
                    )
                    
                    st.subheader("📋 Grounded Answer")
                    st.write(response.text)

                    st.markdown("---")
                    st.subheader("📄 Top Retained Source Context")
                    for i, c in enumerate(top_chunks, 1):
                        with st.expander(f"Source #{i} — Section: {c.get('section', 'General')} (Page {c.get('page', 1)})"):
                            st.write(f'"{c.get("text", "")}"')

                except Exception as e:
                    st.error(f"API Error: {e}")