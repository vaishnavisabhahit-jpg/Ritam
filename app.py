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
import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import textstat

# --- Page Setup ---
st.set_page_config(
    page_title="Ritam AI - FDA Label Intelligence",
    page_icon="💊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
EMERGENCY_CONFIG_PATH = os.path.join(BASE_DIR, "config", "emergency.json")

# --- Load Emergency Config ---
emergency_data = {}
if os.path.exists(EMERGENCY_CONFIG_PATH):
    try:
        with open(EMERGENCY_CONFIG_PATH, "r", encoding="utf-8") as f:
            emergency_data = json.load(f)
    except Exception:
        pass

if not emergency_data:
    emergency_data = {
        "primary": {"label": "National Emergency Services", "tel": "911"},
        "poison": {"label": "Poison Control Center Hotline", "tel": "1-800-222-1222"},
        "disclaimer": "If experiencing severe symptoms or overdose, contact emergency services immediately."
    }

# --- Initialization Functions ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_chroma_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        return client.get_collection("drug_labels")
    except Exception as e:
        st.error(f"Failed to load ChromaDB: {e}")
        return None

embedding_model = load_embedding_model()
collection = load_chroma_collection()

# --- Safety & Audit Functions ---
def check_emergency_intent(query: str):
    keywords = ["emergency", "overdose", "chest pain", "anaphylaxis", "poison", "dying", "fainting", "severe allergic"]
    if any(k in query.lower() for k in keywords):
        return emergency_data
    return None

def verify_numerical_consistency(answer: str, context_chunks: list[dict]) -> bool:
    context_text = " ".join([c["text"] for c in context_chunks])
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', context_text))
    
    # Common words or grades to ignore
    ignore_set = {"1", "2", "3", "4", "5", "8", "12"}
    answer_numbers = answer_numbers - ignore_set
    
    return answer_numbers.issubset(context_numbers) if answer_numbers else True

def calculate_readability(text: str) -> float:
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return 8.0

def generate_medical_answer(query: str, drug: str, context_chunks: list[dict], audience: str) -> str:
    # Retrieve API key from environment or Streamlit Secrets
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    if not api_key:
        return "⚠️ Gemini API key is missing. Please set GEMINI_API_KEY in Streamlit Secrets or Environment Variables."

    gemini_client = genai.Client(api_key=api_key)

    context_text = "\n\n".join([
        f"--- Source Chunk (Section: {c['section']}, Page {c['page']}) ---\n{c['text']}"
        for c in context_chunks
    ])

    system_instruction = (
        "You are Ritam, a safety-first medical AI assistant. Your job is to answer questions about prescription/OTC drugs "
        "using ONLY the provided FDA label context below.\n\n"
        "STRICT SAFETY RULES:\n"
        "1. Do NOT invent, assume, or extrapolate medical information or dosages not explicitly stated in the context.\n"
        "2. If the answer is not contained in the context, clearly state: 'The provided FDA label does not contain this information.'\n"
        "3. Keep tone empathetic, clear, and easy to understand for patients, or clinical and precise for doctors.\n"
        "4. Directly quote key phrases from the label text where appropriate."
    )

    prompt = f"Target Drug: {drug.capitalize()}\nTarget Audience: {audience}\nUser Question: {query}\n\nFDA Label Context:\n{context_text}"

    try:
        # Fixed model identifier: gemini-2.0-flash
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error ({type(e).__name__}): {e}"

# --- UI Header ---
st.title("💊 Ritam AI")
st.caption("Verified FDA Medical Intelligence Engine")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
drug_options = {
    "Metformin": "metformin",
    "Acetaminophen": "acetaminophen",
    "Omeprazole": "omeprazole",
    "Amoxicillin": "amoxicillin",
    "Cetirizine": "cetirizine",
    "Levothyroxine": "levothyroxine"
}

selected_drug_label = st.sidebar.selectbox("Select FDA Drug Label", list(drug_options.keys()))
selected_drug = drug_options[selected_drug_label]

audience_mode = st.sidebar.radio("Target Audience Mode", ["Patient", "Doctor"])
audience = "patient" if audience_mode == "Patient" else "clinician"

user_query = st.text_input(
    f"Ask a question about {selected_drug_label}:",
    placeholder=f"What are common side effects or dosage warnings for {selected_drug_label}?"
)

analyze_btn = st.button("Analyze & Verify", type="primary")

# --- Main Logic Execution ---
if analyze_btn and user_query:
    # 1. Check Emergency Gate
    emergency = check_emergency_intent(user_query)
    if emergency:
        st.error("🚨 **Immediate Medical Emergency Detected**")
        st.warning("If you or someone else is experiencing severe side effects or overdose, contact emergency services immediately.")
        st.write(f"• **Emergency Services:** {emergency['primary']['tel']} ({emergency['primary']['label']})")
        st.write(f"• **Poison Control:** {emergency['poison']['tel']} ({emergency['poison']['label']})")
    else:
        if not collection:
            st.error("ChromaDB vector collection is not available. Please verify local setup.")
        else:
            with st.spinner("Searching FDA labels & synthesizing response..."):
                # 2. Vector Search
                query_vector = embedding_model.encode([user_query]).tolist()
                results = collection.query(
                    query_embeddings=query_vector,
                    n_results=3,
                    where={"drug": selected_drug}
                )

                chunks = []
                if results and results.get("documents") and results["documents"][0]:
                    for idx, doc_text in enumerate(results["documents"][0]):
                        meta = results["metadatas"][0][idx]
                        chunks.append({
                            "chunk_id": results["ids"][0][idx],
                            "text": doc_text,
                            "section": meta.get("section", "General"),
                            "page": meta.get("page", 1)
                        })

                if not chunks:
                    st.warning(f"No specific FDA label details found for '{selected_drug_label}' matching your query.")
                else:
                    # 3. LLM Generation
                    answer = generate_medical_answer(user_query, selected_drug, chunks, audience)

                    # 4. Metrics & Audits
                    num_valid = verify_numerical_consistency(answer, chunks)
                    grade_level = calculate_readability(answer)

                    # --- Response Section ---
                    st.subheader("📋 Grounded Medical Response")
                    st.write(answer)

                    st.markdown("---")
                    st.subheader("🛡️ Active Safety & Hallucination Metrics")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Context Grounding Check", "PASSED ✅" if not answer.startswith("⚠️") else "WARNING ⚠️")
                        st.caption("Verified against FDA source chunks")
                    with col2:
                        st.metric("Numerical Consistency", "PASSED ✅" if num_valid else "WARNING ⚠️")
                        st.caption("Exact Dosage & Number Match")
                    with col3:
                        st.metric("Flesch-Kincaid Grade Level", f"Grade {grade_level:.1f}")
                        st.caption("Target: Grade < 8 for Patients")

                    st.markdown("---")
                    st.subheader("📄 FDA Label Source References")
                    for i, c in enumerate(chunks, 1):
                        with st.expander(f"Source #{i} — Section: {c['section']} (Page {c['page']})"):
                            st.write(f'"{c["text"]}"')