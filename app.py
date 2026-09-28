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

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Ritam AI — Medical Intelligence",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# --- Initialize Gemini API Client ---
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@st.cache_data
def get_gemini_embedding(text: str):
    """Generate vector embedding using text-embedding-004."""
    if not client:
        return None
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values
    except Exception:
        return None


@st.cache_data
def load_drug_chunks(drug_id: str):
    """Load pre-processed text chunks from JSON store."""
    json_path = os.path.join(PROCESSED_DIR, f"{drug_id}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def check_emergency(query: str) -> bool:
    """Interception guardrail for acute life-threatening emergencies."""
    keywords = [
        "overdose", "chest pain", "cannot breathe", "anaphylaxis",
        "poison", "dying", "fainting", "severe allergy"
    ]
    return any(k in query.lower() for k in keywords)


# --- UI Header & Sidebar Setup ---
st.title("⚡ Ritam AI — FDA Medical Intelligence Engine")
st.caption("Grounded FDA Intelligence Powered by `gemini-2.5-flash`")

st.sidebar.header("System Controls")
drug_options = {
    "Metformin": "metformin",
    "Acetaminophen": "acetaminophen",
    "Omeprazole": "omeprazole",
    "Amoxicillin": "amoxicillin",
    "Cetirizine": "cetirizine",
    "Levothyroxine": "levothyroxine"
}
selected_label = st.sidebar.selectbox("Target Medication", list(drug_options.keys()))
selected_drug = drug_options[selected_label]

audience_mode = st.sidebar.radio("Response Tone", ["Patient View", "Clinical/Doctor View"])
audience = "patient" if audience_mode == "Patient View" else "clinician"

user_query = st.text_input(
    f"Enter inquiry regarding {selected_label}:",
    placeholder="e.g. What are common side effects or dosing instructions?"
)

# --- Execution Core ---
if st.button("🚀 Analyze & Verify", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif check_emergency(user_query):
        st.error("🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**")
        st.markdown(
            "If you or someone else is experiencing severe symptoms or an immediate emergency, "
            "contact emergency services (**911**) or Poison Control (**1-800-222-1222**) immediately."
        )
    elif not client:
        st.error("⚠️ `GEMINI_API_KEY` is missing from Streamlit Secrets or Environment Variables.")
    else:
        with st.spinner("Executing Semantic Search & Grounded Analysis..."):
            chunks = load_drug_chunks(selected_drug)

            if not chunks:
                st.error(f"No processed data found for **{selected_label}** in `data/processed/{selected_drug}.json`.")
            else:
                # 1. Keyword Matching Score
                query_words = set(user_query.lower().split())
                for c in chunks:
                    c_text = c.get("text", "").lower()
                    c["score"] = sum(1 for w in query_words if w in c_text)

                # 2. Vector Search Scoring
                query_emb = get_gemini_embedding(user_query)
                if query_emb:
                    top_candidates = sorted(chunks, key=lambda x: x["score"], reverse=True)[:5]
                    for c in top_candidates:
                        snippet = c.get("text", "")[:500]
                        c_emb = get_gemini_embedding(snippet)
                        if c_emb:
                            sim = cosine_similarity(query_emb, c_emb)
                            c["score"] += sim * 10.0

                # Sort and select top context chunks
                chunks.sort(key=lambda x: x["score"], reverse=True)
                top_chunks = chunks[:3]

                # 3. Context & Prompt Assembly
                context_chunks = []
                for c in top_chunks:
                    sec = c.get("section_title") or c.get("section") or "General"
                    pg = c.get("page", 1)
                    txt = c.get("text", "")
                    context_chunks.append(f"--- Section: {sec} (Page {pg}) ---\n{txt}")
                
                context_text = "\n\n".join(context_chunks)

                prompt = (
                    f"Target Drug: {selected_label}\n"
                    f"Target Audience Profile: {audience}\n"
                    f"User Inquiry: {user_query}\n\n"
                    f"FDA Official Context:\n{context_text}"
                )

                system_instruction = (
                    "You are Ritam AI, a safety-critical medical assistant. Answer using ONLY the provided FDA label context.\n"
                    "Do not guess, assume, or fabricate medical advice. If information is not in the context, explicitly state that.\n"
                    "Adjust complexity to match the requested audience profile."
                )

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
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
                        sec = c.get("section_title") or c.get("section") or "General"
                        pg = c.get("page", 1)
                        with st.expander(f"Source #{i} — Section: {sec} (Page {pg})"):
                            st.write(f'"{c.get("text", "")}"')

                except Exception as e:
                    st.error(f"API Generation Failure: {e}")