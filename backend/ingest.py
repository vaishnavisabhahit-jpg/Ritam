import os
import json
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Directory Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 2. Known PDFs Map (Matched to exact filenames in data/raw)
DRUG_FILES = {
    "metformin": "METFORMIN.pdf",
    "acetaminophen": "ACETAMINOPHEN 325 MG (acetaminophen) tablet.pdf",
    "omeprazole": "OMEPRAZOLE.pdf",
    "amoxicillin": "AMOXICILLIN.pdf",
    "cetirizine": "CETIRIZINE.pdf",
    "levothyroxine": "LEVOTHYROXINE SODIUM capsule.pdf"
}

# 3. Text Cleaning Map
TEXT_REPLACEMENTS = {
    "1.73 m²": "1.73 m2",
    "vitamin B12": "vitamin B12",
    "cetiirizine": "cetirizine",
    "Ceitirizine": "Cetirizine"
}

def clean_text(text: str) -> str:
    for bad, good in TEXT_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return " ".join(text.split())

def extract_chunks_from_pdf(drug_id: str, filename: str) -> list[dict]:
    pdf_path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(pdf_path):
        print(f"⚠️ Warning: File not found {pdf_path}")
        return []

    doc = pymupdf.open(pdf_path)
    chunks = []
    chunk_counter = 0
    current_section = "General Information"
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = page.get_text("blocks")
        
        for block in text_blocks:
            block_text = block[4].strip()
            if not block_text:
                continue

            cleaned_block = clean_text(block_text)
            
            if len(cleaned_block) < 60 and any(cleaned_block.lower().startswith(p) for p in [
                "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10 ",
                "uses", "warnings", "directions", "do not use", "patient information", "boxed warning"
            ]):
                current_section = cleaned_block

            if len(cleaned_block) > 40:
                chunk_counter += 1
                is_patient_leaflet = "PATIENT INFORMATION" in current_section.upper()
                
                chunk = {
                    "chunk_id": f"{drug_id}_{chunk_counter:03d}",
                    "drug": drug_id,
                    "section_title": current_section,
                    "audience": "patient" if is_patient_leaflet else "clinician",
                    "page": page_num + 1,
                    "text": cleaned_block,
                    "source_file": filename
                }
                chunks.append(chunk)

    doc.close()
    return chunks

def build_vector_store(all_chunks: list[dict]):
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        client.delete_collection("drug_labels")
    except Exception:
        pass

    collection = client.create_collection(name="drug_labels")

    documents = [c["text"] for c in all_chunks]
    ids = [c["chunk_id"] for c in all_chunks]
    metadatas = [
        {
            "drug": c["drug"],
            "section": c["section_title"],
            "page": c["page"],
            "audience": c["audience"]
        }
        for c in all_chunks
    ]

    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ ChromaDB vector database saved successfully to {CHROMA_DB_DIR}")

def run_ingestion():
    all_chunks = []
    
    for drug_id, filename in DRUG_FILES.items():
        print(f"Processing {drug_id} ({filename})...")
        chunks = extract_chunks_from_pdf(drug_id, filename)
        all_chunks.extend(chunks)
        
        drug_json_path = os.path.join(PROCESSED_DATA_DIR, f"{drug_id}.json")
        with open(drug_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

    print(f"Extracted {len(all_chunks)} total chunks across all drugs.")
    
    if all_chunks:
        build_vector_store(all_chunks)

if __name__ == "__main__":
    run_ingestion()