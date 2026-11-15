import os
import re
import json

# Tüm AI modellerinin C diskine değil, doğrudan D diskine inmesini zorunlu kılıyoruz
os.environ["HF_HOME"] = r"D:\inertia-quant\data\hf_cache"
os.environ["TORCH_HOME"] = r"D:\inertia-quant\data\hf_cache"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_DIR = r"D:\inertia-quant\data\chroma_db"
FILINGS_DIR = r"D:\inertia-quant\data\sec_filings"

def detect_sec_section(text_chunk: str, current_section: str) -> str:
    """Metin parçası içinde SEC 10-K bölüm başlığı geçiyorsa aktif bölümü günceller."""
    patterns = [
        (r"ITEM\s+1A\.?\s+RISK\s+FACTORS", "Item 1A - Risk Factors"),
        (r"ITEM\s+1\.?\s+BUSINESS", "Item 1 - Business"),
        (r"ITEM\s+7\.?\s+MANAGEMENT['’]S\s+DISCUSSION", "Item 7 - MD&A (Finansal Analiz)"),
        (r"ITEM\s+7A\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE", "Item 7A - Piyasa Riskleri"),
        (r"ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS", "Item 8 - Financial Statements (Bilanço/Tablolar)"),
    ]
    upper_text = text_chunk.upper()
    for pattern, label in patterns:
        if re.search(pattern, upper_text):
            return label
    return current_section

def index_filing_to_chroma(json_file_path: str):
    """
    İndirilen 10-K JSON dosyasını okur, kaynak meta verileriyle birlikte parçalar
    ve D diskindeki ChromaDB vektör veritabanına kaydeder.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ticker = data["ticker"]
    company_name = data["company_name"]
    filing_date = data["filing_date"]
    source_url = data["source_url"]
    full_text = data["content"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", "|", ". ", " "]
    )

    raw_chunks = splitter.split_text(full_text)
    documents = []
    active_section = "Genel / Kapak"

    for idx, chunk in enumerate(raw_chunks):
        active_section = detect_sec_section(chunk, active_section)
        
        doc = Document(
            page_content=chunk,
            metadata={
                "ticker": ticker,
                "company_name": company_name,
                "form_type": "10-K",
                "filing_date": filing_date,
                "section": active_section,
                "chunk_id": idx + 1,
                "total_chunks": len(raw_chunks),
                "source_url": source_url
            }
        )
        documents.append(doc)

    print(f"[BİLGİ] {ticker} raporu {len(documents)} adet kaynaklı parçaya bölündü.")
    print("[BİLGİ] Embedding modeli D:\\inertia-quant\\data\\hf_cache üzerine yükleniyor...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=r"D:\inertia-quant\data\hf_cache"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="sec_filings_rag"
    )
    
    print(f"[BAŞARILI] {ticker} verileri D:\\inertia-quant\\data\\chroma_db veritabanına işlendi.")
    return vectorstore

if __name__ == "__main__":
    files = [f for f in os.listdir(FILINGS_DIR) if f.endswith(".json")]
    if not files:
        print("Önce sec_fetcher.py çalıştırarak en az bir 10-K raporu indirin.")
    else:
        target_path = os.path.join(FILINGS_DIR, files[0])
        index_filing_to_chroma(target_path)