import os
import time
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Dinamik dizin tanımları (Hem Windows yerel hem Linux/Docker bulut uyumlu)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
HF_CACHE_DIR = os.path.join(DATA_DIR, "hf_cache")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Yerelde varsa .env dosyasını yükle, bulutta sistem ortam değişkenlerini (Secrets) kullan
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TORCH_HOME"] = HF_CACHE_DIR

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

SYSTEM_PROMPT = """Sen kurumsal düzeyde bir Kantitatif Finans Analistisin (Inertia Quant AI).
Görevin, kullanıcıların sorularını YALNIZCA aşağıda sana verilen resmi SEC 10-K / KAP raporu parçalarına (CONTEXT) dayanarak yanıtlamaktır.

KESİN KURALLAR:
1. Sadece sana verilen CONTEXT içindeki verileri ve rakamları kullan. Kendi genel kültüründen veya dışarıdan hiçbir bilgi, yorum veya tahmin ekleme.
2. Eğer sorunun cevabı verilen CONTEXT içinde açıkça yer almıyorsa, kesinlikle uydurma yapma ve tam olarak şu yanıtı ver: "Bu bilgi seçili SEC raporunda yer almamaktadır."
3. Yanıtını Türkçe ver ancak rapordaki finansal terimleri (örn. Gross Margin, Data Center Revenue, Free Cash Flow) parantez içinde orijinal İngilizce haliyle koru.
4. Yaptığın her sayısal veya stratejik çıkarımın hemen sonuna mutlaka kaynak etiketini ekle: [{{ticker}} 10-K | {{filing_date}} | {{section}} | Parça #{{chunk_id}}]
5. Yanıtın en altında "Doğrudan Rapor Alıntıları (Citations)" başlığı açarak kullandığın en kritik 1 veya 2 cümleyi rapordaki orijinal İngilizce haliyle tırnak içinde göster.
"""

_EMBEDDINGS_CACHE = None
_VECTORSTORE_CACHE = None


def get_vectorstore():
    """Embedding modelini ve ChromaDB bağlantısını bellekte önbelleğe alarak yükler."""
    global _EMBEDDINGS_CACHE, _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is None:
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            cache_folder=HF_CACHE_DIR
        )
        _VECTORSTORE_CACHE = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_EMBEDDINGS_CACHE,
            collection_name="sec_filings_rag"
        )
    return _VECTORSTORE_CACHE


def calculate_telemetry(model_used: str, input_text: str, output_text: str, latency_ms: float) -> dict:
    """Token tüketimini, maliyeti ve gecikme süresini hesaplar."""
    enc = tiktoken.get_encoding("cl100k_base")
    in_tokens = len(enc.encode(str(input_text)))
    out_tokens = len(enc.encode(str(output_text)))
    return {
        "model": f"{model_used} (Free Tier)",
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "cost_usd": 0.0,
        "latency_ms": latency_ms
    }


def ask_sec_filing(question: str, ticker: str = "NVDA", k: int = 5) -> dict:
    """Seçili şirket raporunda hibrit vektör araması yapar ve kaynaklı AI yanıtı üretir."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasına veya bulut Secrets ayarlarına ekleyin.")

    start_time = time.time()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"ticker": ticker.upper()}}
    )
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": f"{ticker.upper()} için veritabanında indekslenmiş rapor bulunamadı.",
            "sources": [],
            "telemetry": {
                "model": "N/A",
                "latency_ms": round((time.time() - start_time) * 1000, 1),
                "cost_usd": 0.0
            }
        }

    context_blocks = []
    sources_metadata = []
    for doc in retrieved_docs:
        meta = doc.metadata
        t_code = meta.get("ticker", ticker.upper())
        f_date = meta.get("filing_date", "N/A")
        sec_name = meta.get("section", "General")
        c_id = meta.get("chunk_id", 0)
        s_url = meta.get("source_url", "")

        header = f"--- KAYNAK: [{t_code} 10-K | {f_date} | {sec_name} | Parça #{c_id}] ---"
        context_blocks.append(f"{header}\n{doc.page_content}")
        sources_metadata.append({
            "section": sec_name,
            "chunk_id": c_id,
            "filing_date": f_date,
            "source_url": s_url,
            "snippet": doc.page_content[:300] + "..."
        })

    full_context = "\n\n".join(context_blocks)
    user_prompt = f"CONTEXT:\n{full_context}\n\nSORU:\n{question}"

    client = genai.Client(api_key=api_key)
    models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
    answer_text = ""
    used_model = models_to_try[0]

    for attempt in range(4):
        model_name = models_to_try[attempt % len(models_to_try)]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0
                )
            )
            answer_text = response.text or ""
            used_model = model_name
            if answer_text.strip():
                break
        except Exception as e:
            if attempt == 3:
                raise e
            wait_sec = 5 * (attempt + 1)
            print(f"[Uyarı] {model_name} geçici yoğun ({e}). {wait_sec} sn bekleyip yeniden deneniyor...")
            time.sleep(wait_sec)

    latency_ms = round((time.time() - start_time) * 1000, 1)
    telemetry = calculate_telemetry(used_model, SYSTEM_PROMPT + user_prompt, answer_text, latency_ms)

    return {
        "answer": answer_text,
        "sources": sources_metadata,
        "telemetry": telemetry
    }


if __name__ == "__main__":
    # Hızlı doğrulama testi
    test_res = ask_sec_filing("What caused the gross margin decrease in fiscal year 2026?", ticker="NVDA")
    print(test_res["answer"])
    print("\nTelemetri:", test_res["telemetry"])