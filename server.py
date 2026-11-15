import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag_engine")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

sys.path.append(RAG_DIR)
from qa_chain import ask_sec_filing
from sec_fetcher import fetch_latest_10k
from parser_chunker import index_filing_to_chroma

app = FastAPI(title="Inertia Quant Enterprise Quantitative Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIQueryRequest(BaseModel):
    ticker: str
    question: str
    top_k: int = 5

class BacktestRequest(BaseModel):
    weights: Dict[str, float]
    benchmark: str = "SPY"
    period: str = "3y"
    initial_capital: float = 10000.0
    rebalance_monthly: bool = True

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/health")
def health_check():
    return {"status": "online", "engine": "Inertia Quant v4.0 Cloud"}

@app.get("/api/markets/overview")
def get_global_markets():
    watchlist = {
        "S&P 500": "^GSPC",
        "NASDAQ 100": "^NDX",
        "BIST 100": "XU100.IS",
        "USD / TRY": "TRY=X",
        "EUR / USD": "EURUSD=X",
        "GOLD (OZ)": "GC=F",
        "BITCOIN": "BTC-USD"
    }
    results = []
    for name, sym in watchlist.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if len(hist) >= 2:
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg = ((last / prev) - 1) * 100
                results.append({"name": name, "symbol": sym, "price": round(last, 4), "change_pct": round(chg, 2)})
        except Exception:
            continue
    return {"markets": results}

@app.post("/api/backtest")
def run_portfolio_backtest(req: BacktestRequest):
    tickers = [t.upper().strip() for t in req.weights.keys() if req.weights[t] > 0]
    raw_w = np.array([req.weights[t] for t in req.weights.keys() if req.weights[t] > 0], dtype=float)
    if len(tickers) == 0 or raw_w.sum() <= 0:
        raise HTTPException(status_code=400, detail="Geçerli varlık ağırlıkları girilmelidir.")
    weights = raw_w / raw_w.sum()

    all_symbols = list(set(tickers + [req.benchmark.upper()]))
    data = yf.download(all_symbols, period=req.period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=all_symbols[0])
    data = data.dropna()

    if len(data) < 30:
        raise HTTPException(status_code=404, detail="Yeterli tarihsel veri bulunamadı.")

    daily_ret = data[tickers].pct_change().fillna(0.0)
    bench_ret = data[req.benchmark.upper()].pct_change().fillna(0.0)

    port_values = [req.initial_capital]
    current_holdings = weights * req.initial_capital
    prev_month = daily_ret.index[0].month

    for i in range(1, len(daily_ret)):
        r_vec = daily_ret.iloc[i].values
        current_holdings = current_holdings * (1.0 + r_vec)
        total_val = float(current_holdings.sum())
        if req.rebalance_monthly and daily_ret.index[i].month != prev_month:
            current_holdings = weights * total_val
            prev_month = daily_ret.index[i].month
        port_values.append(total_val)

    port_series = pd.Series(port_values, index=daily_ret.index)
    bench_series = req.initial_capital * (1.0 + bench_ret).cumprod()

    p_daily = port_series.pct_change().dropna()
    years = len(port_series) / 252.0
    cagr = ((port_series.iloc[-1] / req.initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100
    bench_cagr = ((bench_series.iloc[-1] / req.initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100

    ann_vol = float(p_daily.std() * np.sqrt(252) * 100)
    rf = 4.25
    sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else 0.0

    downside = p_daily[p_daily < 0]
    down_vol = float(downside.std() * np.sqrt(252) * 100) if len(downside) > 0 else ann_vol
    sortino = (cagr - rf) / down_vol if down_vol > 0 else 0.0

    cum_max = port_series.cummax()
    dd_series = ((port_series - cum_max) / cum_max) * 100
    max_dd = float(dd_series.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    cov_pb = np.cov(p_daily, bench_ret.iloc[1:])[0, 1]
    var_b = np.var(bench_ret.iloc[1:])
    beta = float(cov_pb / var_b) if var_b > 0 else 1.0

    return {
        "metrics": {
            "final_value": round(float(port_series.iloc[-1]), 2),
            "bench_final_value": round(float(bench_series.iloc[-1]), 2),
            "cagr": round(cagr, 2),
            "bench_cagr": round(bench_cagr, 2),
            "ann_vol": round(ann_vol, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "calmar": round(calmar, 2),
            "max_dd": round(max_dd, 2),
            "beta": round(beta, 2)
        },
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in port_series.index],
            "portfolio": [round(float(x), 2) for x in port_series.values],
            "benchmark": [round(float(x), 2) for x in bench_series.values],
            "drawdown": [round(float(x), 2) for x in dd_series.values]
        }
    }

@app.get("/api/benchmark/report")
def get_benchmark_report():
    report_path = os.path.join(DATA_DIR, "benchmark_report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Benchmark raporu bulunamadı.")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/sec-ai/ask")
def query_sec_rag(req: AIQueryRequest):
    return ask_sec_filing(req.question, ticker=req.ticker, k=req.top_k)

@app.post("/api/sec-ai/ingest/{ticker}")
def ingest_sec_filing(ticker: str):
    doc_data = fetch_latest_10k(ticker.upper())
    file_path = os.path.join(DATA_DIR, "sec_filings", f"{ticker.upper()}_10K_{doc_data['filing_date']}.json")
    index_filing_to_chroma(file_path)
    return {"status": "indexed", "ticker": ticker.upper(), "filing_date": doc_data["filing_date"]}