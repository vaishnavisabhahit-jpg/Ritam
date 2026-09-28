import re
import json
import os
import textstat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMERGENCY_CONFIG_PATH = os.path.join(BASE_DIR, "config", "emergency.json")

def check_emergency_intent(user_query: str) -> dict | None:
    """Detects acute emergency triggers and returns hardcoded emergency contacts."""
    emergency_keywords = ["overdose", "suicide", "poison", "swallowed whole bottle", "dying", "emergency", "unresponsive"]
    query_lower = user_query.lower()
    
    if any(kw in query_lower for kw in emergency_keywords):
        try:
            with open(EMERGENCY_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {
                "country": "IN",
                "primary": {"label": "Emergency (India)", "tel": "112"},
                "poison": {"label": "Poison Helpline (AIIMS NPIC)", "tel": "18001116117"}
            }
    return None

def verify_grounded_quotes(llm_response: str, source_chunks: list[dict]) -> bool:
    """Ensures quoted text in the response exists inside source documents."""
    quotes = re.findall(r'"([^"]*)"', llm_response)
    if not quotes:
        return True

    source_text_blob = " ".join([c["text"].lower() for c in source_chunks])
    for quote in quotes:
        clean_quote = quote.strip().lower()
        if len(clean_quote) > 10 and clean_quote not in source_text_blob:
            return False
    return True

def verify_numerical_consistency(llm_response: str, source_chunks: list[dict]) -> bool:
    """Validates that dosage and numerical values exist in retrieved chunks."""
    response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', llm_response))
    if not response_numbers:
        return True

    source_text_blob = " ".join([c["text"] for c in source_chunks])
    source_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', source_text_blob))

    allowed_structural = {"1", "2", "3", "4", "5", "10"}
    unverified_numbers = response_numbers - source_numbers - allowed_structural

    return len(unverified_numbers) == 0

def calculate_readability(text: str) -> dict:
    """Computes Flesch-Kincaid Grade Level."""
    grade_level = textstat.flesch_kincaid_grade(text)
    return {
        "flesch_kincaid_grade": grade_level,
        "is_patient_accessible": grade_level <= 8.0
    }