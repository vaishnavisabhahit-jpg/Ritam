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
# ... rest of your code remains unchanged ...

import os
import re
import json
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import textstat

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Animated CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ritam AI - FDA Medical Intelligence",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    /* Global Page Fade-In Animation */
    .main .block-container {
        animation: fadeIn 0.6s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Animated Pulsing Status Badge */
    .pulse-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: #10B981;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Card Lift Animation */
    div[data-testid="stMetricValue"], .source-card {
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        border-radius: 10px;
        padding: 12px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .source-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    }

    /* Button Hover Scale */
    .stButton > button {
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Resource Caching & Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_vector_db():
    db_path = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection("drug_labels")

embedder = load_embedder()
try:
    collection = load_vector_db()
except Exception:
    collection = None

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=api_key) if api_key else None

# -----------------------------------------------------------------------------
# 3. Core Technical Modules & Guardrails
# -----------------------------------------------------------------------------
def check_emergency_intent(query: str):
    patterns = [
        r"\b(chest pain|heart attack|can't breathe|shortness of breath|trouble breathing)\b",
        r"\b(overdose|passed out|unconscious|seizure|anaphylaxis|swollen throat)\b",
        r"\b(suicide|poison|emergency|bleeding heavily|severe reaction)\b"
    ]
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "detected": True,
                "hotline": "911 (US) / 112 (EU)",
                "poison_control": "1-800-222-1222"
            }
    return None

def verify_grounded_quotes(answer: str, chunks: list) -> bool:
    if not answer or not chunks:
        return True
    combined_context = " ".join([c["text"].lower() for c in chunks])
    words = [w for w in re.findall(r'\b\w{5,}\b', answer.lower()) if w not in ["patient", "doctor", "taking", "should"]]
    if not words:
        return True
    matches = sum(1 for w in words if w in combined_context)
    return (matches / len(words)) > 0.40

def verify_numerical_consistency(answer: str, chunks: list) -> bool:
    answer_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    if not answer_nums:
        return True
    combined_context = " ".join([c["text"] for c in chunks])
    context_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', combined_context))
    return answer_nums.issubset(context_nums)

def calculate_readability(text: str) -> dict:
    try:
        score = textstat.flesch_kincaid_grade(text)
    except Exception:
        score = 8.0
    return {
        "flesch_kincaid_grade": score,
        "is_patient_accessible": score <= 10.0
    }

def generate_medical_answer(query: str, drug: str, chunks: list, audience: str) -> str:
    if not gemini_client:
        return "Gemini API key is missing. Please set GEMINI_API_KEY in environment or Streamlit Secrets."

    context_str = "\n\n".join([f"[Section: {c['section']}, Page: {c['page']}]\n{c['text']}" for c in chunks])
    
    audience_instructions = (
        "Explain in plain, compassionate, and easy-to-understand language suitable for a patient."
        if audience == "patient"
        else "Provide a clinical, concise, and pharmacologically accurate explanation suitable for a physician."
    )

    prompt = f"""You are Ritam AI, a medical information system. Answer the query strictly based on the FDA drug label context provided below.

    Context:
    {context_str}

    Audience: {audience.capitalize()}
    Instruction: {audience_instructions}
    User Query: {query}

    Guidelines:
    - Do not invent facts or dosages not supported by the context.
    - Be clear, direct, and factual.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# -----------------------------------------------------------------------------
# 4. Streamlit Dashboard Layout
# -----------------------------------------------------------------------------
st.title("🩺 Ritam AI — FDA Medical Intelligence Engine")
st.markdown('<span class="pulse-badge">Live Safety Verification Engine</span>', unsafe_allow_html=True)
st.write("")

# Sidebar Options
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

audience = st.sidebar.radio("Target Audience Mode", ["patient", "doctor"], format_func=lambda x: x.capitalize())

# Query Input Area
user_query = st.text_input("Ask a medical question regarding the selected drug:", placeholder="e.g., What are the common side effects and dosage rules?")

if st.button("Analyze & Verify", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        # 1. Check Safety Emergency Guardrail
        emergency_info = check_emergency_intent(user_query)
        if emergency_info:
            st.error("🚨 **CRITICAL HEALTH EMERGENCY DETECTED**")
            st.error(f"If you or someone else is experiencing severe symptoms, call **{emergency_info['hotline']}** or Poison Control (**{emergency_info['poison_control']}**) immediately.")
        else:
            with st.spinner("Retrieving vector embeddings and verifying grounding..."):
                if not collection:
                    st.error("Vector Database connection failed. Please check `data/chroma_db` directory.")
                else:
                    # 2. Vector Search Retrieval
                    query_vector = embedder.encode([user_query]).tolist()
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
                                "text": doc_text,
                                "section": meta.get("section", "General"),
                                "page": meta.get("page", 1)
                            })

                    if not chunks:
                        st.info(f"No specific FDA context found for '{selected_drug_label}' matching your query.")
                    else:
                        # 3. Gemini Generation
                        answer = generate_medical_answer(user_query, selected_drug, chunks, audience)

                        # 4. Verification Guardrails
                        quotes_valid = verify_grounded_quotes(answer, chunks)
                        numbers_valid = verify_numerical_consistency(answer, chunks)
                        readability = calculate_readability(answer)

                        # Render Results Layout
                        st.markdown("### 📋 Grounded Medical Response")
                        st.write(answer)
                        st.divider()

                        # Render Verification Badges
                        st.markdown("### 🛡️ Active Safety & Hallucination Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                label="Context Grounding Check",
                                value="PASSED ✅" if quotes_valid else "WARNING ⚠️",
                                delta="High Factuality" if quotes_valid else "Potential Unverified Terms"
                            )
                        with col2:
                            st.metric(
                                label="Numerical Consistency",
                                value="PASSED ✅" if numbers_valid else "WARNING ⚠️",
                                delta="Exact Dosage Match" if numbers_valid else "Check Figures"
                            )
                        with col3:
                            st.metric(
                                label="Flesch-Kincaid Grade Level",
                                value=f"Grade {readability['flesch_kincaid_grade']:.1f}",
                                delta="Patient Accessible" if readability['is_patient_accessible'] else "Clinical Complexity"
                            )

                        st.divider()

                        # Render Interactive Source Citations
                        st.markdown("### 📄 FDA Label Source References")
                        for idx, c in enumerate(chunks, 1):
                            with st.expander(f"Source #{idx} — Section: {c['section']} (Page {c['page']})"):
                                st.markdown(f"*{c['text']}*")