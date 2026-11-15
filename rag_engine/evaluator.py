import os
import sys
import json
import time

sys.path.append(r"D:\inertia-quant\rag_engine")
from qa_chain import ask_sec_filing

BENCHMARK_DATASET = [
    # --- IN-CONTEXT: FİNANSAL & STRATEJİK SORULAR (20 SORU) ---
    {"id": "Q01", "type": "In-Context", "ticker": "NVDA", "question": "How much was the charge associated with H20 excess inventory and purchase obligations in fiscal year 2026?", "expected_keywords": ["4.5 billion", "H20"]},
    {"id": "Q02", "type": "In-Context", "ticker": "NVDA", "question": "By what percentage did revenue from Data Center networking grow?", "expected_keywords": ["142%", "NVLink"]},
    {"id": "Q03", "type": "In-Context", "ticker": "NVDA", "question": "By what percentage did Data Center computing revenue grow?", "expected_keywords": ["59%", "Blackwell"]},
    {"id": "Q04", "type": "In-Context", "ticker": "NVDA", "question": "What caused the gross margin decrease in fiscal year 2026?", "expected_keywords": ["Blackwell", "Hopper"]},
    {"id": "Q05", "type": "In-Context", "ticker": "NVDA", "question": "Which platforms drove Data Center networking growth alongside NVLink?", "expected_keywords": ["Ethernet", "InfiniBand"]},
    {"id": "Q06", "type": "In-Context", "ticker": "NVDA", "question": "Which specific GPU systems used the NVLink compute fabric mentioned in MD&A?", "expected_keywords": ["GB200", "GB300"]},
    {"id": "Q07", "type": "In-Context", "ticker": "NVDA", "question": "What are the major supply chain and manufacturing risks disclosed in Item 1A?", "expected_keywords": ["supply", "10-K"]},
    {"id": "Q08", "type": "In-Context", "ticker": "NVDA", "question": "How do export controls and geopolitical restrictions affect NVIDIA's business?", "expected_keywords": ["export", "10-K"]},
    {"id": "Q09", "type": "In-Context", "ticker": "NVDA", "question": "What role does Research and Development play in NVIDIA's competitive strategy?", "expected_keywords": ["Research", "10-K"]},
    {"id": "Q10", "type": "In-Context", "ticker": "NVDA", "question": "How does NVIDIA describe its transition from Hopper HGX systems?", "expected_keywords": ["Hopper", "Blackwell"]},
    {"id": "Q11", "type": "In-Context", "ticker": "NVDA", "question": "What are the primary market segments served by NVIDIA according to Item 1?", "expected_keywords": ["Data Center", "Gaming"]},
    {"id": "Q12", "type": "In-Context", "ticker": "NVDA", "question": "What risks does NVIDIA face regarding third-party foundries and packaging?", "expected_keywords": ["10-K"]},
    {"id": "Q13", "type": "In-Context", "ticker": "NVDA", "question": "How does macroeconomic volatility impact demand for NVIDIA products?", "expected_keywords": ["10-K"]},
    {"id": "Q14", "type": "In-Context", "ticker": "NVDA", "question": "What intellectual property and cybersecurity risks are outlined in the 10-K?", "expected_keywords": ["10-K"]},
    {"id": "Q15", "type": "In-Context", "ticker": "NVDA", "question": "What is NVIDIA's liquidity and cash flow management overview in Item 7?", "expected_keywords": ["cash", "10-K"]},
    {"id": "Q16", "type": "In-Context", "ticker": "NVDA", "question": "In which quarter of fiscal year 2026 did the H20 inventory charge occur?", "expected_keywords": ["first", "H20"]},
    {"id": "Q17", "type": "In-Context", "ticker": "NVDA", "question": "What software platforms complement NVIDIA's accelerated computing hardware?", "expected_keywords": ["CUDA", "10-K"]},
    {"id": "Q18", "type": "In-Context", "ticker": "NVDA", "question": "How does competition in the AI semiconductor market affect pricing and margins?", "expected_keywords": ["10-K"]},
    {"id": "Q19", "type": "In-Context", "ticker": "NVDA", "question": "What regulatory compliance challenges are disclosed in Item 1A Risk Factors?", "expected_keywords": ["10-K"]},
    {"id": "Q20", "type": "In-Context", "ticker": "NVDA", "question": "What are the key drivers of operating expenses in fiscal year 2026?", "expected_keywords": ["10-K"]},

    # --- OUT-OF-CONTEXT: HALÜSİNASYON TUZAK SORULARI (10 SORU) ---
    {"id": "Q21_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What is NVIDIA's projected revenue target for the year 2035 according to the 10-K?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q22_TRAP", "type": "Trap", "ticker": "NVDA", "question": "How many shares of Apple stock did Jensen Huang personally purchase in 2026?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q23_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What is the exact budget allocated for NVIDIA's Mars colony data center?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q24_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What was the closing price of Bitcoin on the day the 10-K was signed?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q25_TRAP", "type": "Trap", "ticker": "NVDA", "question": "Which Formula 1 team did NVIDIA sponsor in 2026 according to Item 7?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q26_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What is the secret recipe used in the Santa Clara headquarters cafeteria?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q27_TRAP", "type": "Trap", "ticker": "NVDA", "question": "How many employees own electric scooters according to the 10-K filing?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q28_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What is NVIDIA's target dividend yield for fiscal year 2040?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q29_TRAP", "type": "Trap", "ticker": "NVDA", "question": "Who won the UEFA Champions League final mentioned in Item 1A?", "expected_keywords": ["yer almamaktadır"]},
    {"id": "Q30_TRAP", "type": "Trap", "ticker": "NVDA", "question": "What is the exact serial number of the first Blackwell chip shipped?", "expected_keywords": ["yer almamaktadır"]}
]

def run_benchmark():
    results = []
    passed_count = 0
    hallucination_count = 0
    trap_total = 0
    latencies = []

    print("=== INERTIA QUANT // 30-SORULUK GERÇEK BENCHMARK BAŞLATILIYOR ===\n")

    for item in BENCHMARK_DATASET:
        print(f"[{item['id']}] Test ediliyor: {item['question']}")
        res = ask_sec_filing(item["question"], ticker=item["ticker"], k=5)
        answer = res["answer"]
        lat = res["telemetry"].get("latency_ms", 0)
        latencies.append(lat)

        matched = all(kw.lower() in answer.lower() for kw in item["expected_keywords"])

        if item["type"] == "Trap":
            trap_total += 1
            if not matched:
                hallucination_count += 1

        status = "PASS" if matched else "FAIL"
        if matched:
            passed_count += 1

        print(f" -> Durum: {status} | Süre: {lat} ms")
        results.append({
            "id": item["id"],
            "type": item["type"],
            "ticker": item["ticker"],
            "question": item["question"],
            "status": status,
            "latency_ms": lat,
            "answer_snippet": answer[:220].replace("\n", " ") + "..."
        })
        # Ücretsiz Gemini API dakika başı istek sınırına (RPM) takılmamak için 4.5 sn bekleme
        time.sleep(4.5)

    total = len(BENCHMARK_DATASET)
    accuracy = round((passed_count / total) * 100, 1)
    hallucination_rate = round((hallucination_count / trap_total) * 100, 1) if trap_total else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    report = {
        "summary": {
            "total_questions": total,
            "passed": passed_count,
            "accuracy_percent": accuracy,
            "hallucination_rate_percent": hallucination_rate,
            "avg_latency_ms": avg_latency,
            "cost_per_query_usd": 0.0
        },
        "details": results
    }

    report_path = r"D:\inertia-quant\data\benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== BENCHMARK TAMAMLANDI ===")
    print(f"Doğruluk: %{accuracy} ({passed_count}/{total}) | Halüsinasyon Oranı: %{hallucination_rate} | Ort. Süre: {avg_latency} ms")
    print(f"Rapor kaydedildi: {report_path}")

if __name__ == "__main__":
    run_benchmark()