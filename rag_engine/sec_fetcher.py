import os
import json
from edgar import Company, set_identity

set_identity("Muhammed Sefa Sut inertia.quant@research.local")

DEFAULT_SAVE_DIR = r"D:\inertia-quant\data\sec_filings"

def fetch_latest_10k(ticker: str, save_dir: str = DEFAULT_SAVE_DIR) -> dict:
    """
    Verilen hisse kodu (örn. 'NVDA') için en güncel 10-K raporunu SEC EDGAR'dan çeker
    ve meta verileriyle birlikte D diskindeki hedef dizine JSON olarak kaydeder.
    """
    os.makedirs(save_dir, exist_ok=True)
    ticker = ticker.upper().strip()
    
    company = Company(ticker)
    # Güncel edgartools sürümünde latest(1) doğrudan tek bir EntityFiling nesnesi döndürür
    filing = company.get_filings(form="10-K").latest(1)
    
    if not filing:
        raise ValueError(f"{ticker} için 10-K raporu bulunamadı.")
    
    filing_date = str(filing.filing_date)
    accession_no = filing.accession_no
    
    # Raporu tabloları koruyarak Markdown formatında metne çevir
    markdown_text = filing.markdown()
    
    document_data = {
        "ticker": ticker,
        "company_name": company.name,
        "form_type": "10-K",
        "filing_date": filing_date,
        "accession_no": accession_no,
        "source_url": filing.filing_url,
        "content": markdown_text
    }
    
    file_path = os.path.join(save_dir, f"{ticker}_10K_{filing_date}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(document_data, f, ensure_ascii=False, indent=2)
        
    print(f"[BAŞARILI] {company.name} ({ticker}) - {filing_date} tarihli 10-K indirildi.")
    print(f"Karakter uzunluğu: {len(markdown_text):,} | Kayıt yeri: {file_path}")
    
    return document_data

if __name__ == "__main__":
    fetch_latest_10k("NVDA")