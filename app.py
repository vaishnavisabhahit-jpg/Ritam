import os
import json
import re
import streamlit as st
import textstat
from google import genai
from google.genai import types

# Page Config
st.set_page_config(
    page_title="RITAM — Truth Grounded Medical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .emergency-banner {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Utilities & Data Loaders
@st.cache_data
def load_drug_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    drugs_dir = os.path.join(base_dir, "data", "processed")
    drugs_data = {}
    
    if os.path.exists(drugs_dir):
        for fname in os.listdir(drugs_dir):
            if fname.endswith(".json"):
                drug_key = fname.replace(".json", "").lower()
                with open(os.path.join(drugs_dir, fname), "r", encoding="utf-8") as f:
                    drugs_data[drug_key] = json.load(f)
    return drugs_data

drugs_database = load_drug_data()

def keyword_retrieval(query: str, drug_data: list, top_k: int = 3):
    """Retrieve top-k relevant chunks based on keyword matching."""
    words = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    
    for chunk in drug_data:
        text = chunk.get("text", "").lower()
        score = sum(1 for w in words if w in text and len(w) > 2)
        scored_chunks.append((score, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

def check_emergency(query: str):
    triggers = ["overdose", "suicide", "poison", "swallowed bottle", "dying", "unresponsive", "emergency"]
    if any(t in query.lower() for t in triggers):
        return {
            "is_emergency": True,
            "contacts": [
                {"label": "National Emergency Helpline (India)", "number": "112"},
                {"label": "AIIMS Poison Information Centre", "number": "1800-111-6117"},
                {"label": "US Poison Control Center", "number": "1-800-222-1222"}
            ]
        }
    return {"is_emergency": False}

def run_verifications(response_text: str, source_chunks: list):
    source_blob = " ".join([c.get("text", "").lower() for c in source_chunks])
    
    # 1. Quote Verification
    quotes = re.findall(r'"([^"]*)"', response_text)
    quotes_valid = True
    for q in quotes:
        if len(q) > 8 and q.lower() not in source_blob:
            quotes_valid = False
            break
            
    # 2. Number Consistency
    resp_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', response_text))
    src_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', source_blob))
    allowed = {"1", "2", "3", "4", "5", "10"}
    numbers_valid = len(resp_numbers - src_numbers - allowed) == 0
    
    # 3. Readability Score
    grade = textstat.flesch_kincaid_grade(response_text) if response_text else 0.0
    
    return {
        "quotes_valid": quotes_valid,
        "numbers_valid": numbers_valid,
        "grade_level": grade,
        "patient_accessible": grade <= 9.0
    }

# Header Section
st.markdown('<div class="main-header">RITAM AI 🩺</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Truth-Grounded Medical Intelligence & Hallucination Guardrails</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Query Settings")
    
    available_drugs = list(drugs_database.keys()) if drugs_database else ["metformin", "acetaminophen", "amoxicillin"]
    selected_drug = st.selectbox("Select Medication", [d.capitalize() for d in available_drugs]).lower()
    
    audience = st.radio("Target Audience", ["Patient (Plain Language)", "Clinician (Technical)"])
    audience_key = "patient" if "Patient" in audience else "clinician"
    
    st.divider()
    st.markdown("### 🛡️ Guardrails Active")
    st.markdown("✅ Grounded Context Retrieval")
    st.markdown("✅ Direct Quote Verification")
    st.markdown("✅ Numerical Dosage Guard")
    st.markdown("✅ Immediate Poison/Emergency Intercept")

# Main Interface Layout
col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input(
        "Ask a question about this medication:",
        placeholder="e.g., What are the common side effects and recommended daily dosage limit?"
    )
    submit_btn = st.button("Generate Verified Answer 🚀", use_container_width=True)

if submit_btn and user_query:
    # 1. Emergency Safety Check
    emergency_status = check_emergency(user_query)
    if emergency_status["is_emergency"]:
        st.markdown("""
        <div class="emergency-banner">
            <h3 style="color: #991B1B; margin:0 0 10px 0;">🚨 CRITICAL EMERGENCY DETECTED</h3>
            <p style="color: #7F1D1D; margin-bottom: 10px;">If you or someone else is experiencing an emergency, immediate medical assistance is required:</p>
        </div>
        """, unsafe_allow_html=True)
        
        for contact in emergency_status["contacts"]:
            st.warning(f"📞 **{contact['label']}:** `{contact['number']}`")
            
    else:
        # 2. Context Retrieval
        drug_chunks = drugs_database.get(selected_drug, [])
        top_chunks = keyword_retrieval(user_query, drug_chunks, top_k=3)
        
        if not top_chunks:
            st.warning("No context available for this medication file.")
        else:
            # Construct Prompt
            context_str = "\n\n".join([f"Source [{i+1}]: {c.get('text', '')}" for i, c in enumerate(top_chunks)])
            
            system_instruction = (
                "You are Ritam AI, a clinical assistant. Answer strictly based on the provided sources below. "
                "If information is missing, state that it is not covered in the document. Do not invent numbers or dosages."
                f" Format answer appropriately for a {audience_key}."
            )
            
            prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}"
            
            # API Generation
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("Missing `GEMINI_API_KEY`. Please set it in Streamlit Cloud Secrets or `.env` file.")
            else:
                try:
                    client = genai.Client(api_key=api_key)
                    with st.spinner("Synthesizing grounded response..."):
                        response = client.models.generate_content(
                            model="models/gemini-1.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.1
                            )
                        )
                        
                    answer_text = response.text
                    
                    # 3. Verification Metrics
                    verif = run_verifications(answer_text, top_chunks)
                    
                    # Display Side Metrics
                    with col2:
                        st.subheader("📊 Safety Audit")
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            q_badge = '<span class="badge-pass">Verified</span>' if verif["quotes_valid"] else '<span class="badge-fail">Unverified</span>'
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Quotes</div><div>{q_badge}</div></div>', unsafe_allow_html=True)
                        with m2:
                            n_badge = '<span class="badge-pass">Accurate</span>' if verif["numbers_valid"] else '<span class="badge-fail">Flagged</span>'
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Numbers</div><div>{n_badge}</div></div>', unsafe_allow_html=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{verif["grade_level"]}</div><div class="metric-label">Flesch Grade Score</div></div>', unsafe_allow_html=True)
                    
                    # Main Response Box
                    st.subheader("📋 Response")
                    st.success(answer_text)
                    
                    # Top Retained Context Accordion
                    st.subheader("📄 Retained Source Context")
                    for idx, chunk in enumerate(top_chunks, 1):
                        with st.expander(f"Source #{idx} — Page {chunk.get('page', 1)}"):
                            st.write(f'"{chunk.get("text", "")}"')
                            
                except Exception as e:
                    st.error(f"Generation Error: {e}")