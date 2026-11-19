import os
import sys
import time
import json
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag_engine")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

sys.path.append(RAG_DIR)
try:
    from qa_chain import ask_sec_filing
    from sec_fetcher import fetch_latest_10k
    from parser_chunker import index_filing_to_chroma
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

app = FastAPI(title="Inertia Quant // Institutional Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

class OptimizeRequest(BaseModel):
    tickers: List[str]
    period: str = "2y"
    risk_free_rate: float = 4.25
    simulations: int = 4000

class ScreenerRequest(BaseModel):
    tickers: List[str]

class PortfolioItem(BaseModel):
    ticker: str
    qty: float
    avg_cost: float
    currency: str = "USD"

class PortfolioEvalRequest(BaseModel):
    items: List[PortfolioItem]

class BotSimRequest(BaseModel):
    ticker: str = "NVDA"
    period: str = "1y"
    initial_cash: float = 10000.0
    rsi_buy: float = 35.0
    rsi_sell: float = 70.0

# ==============================================================================
# ULTRA-FAST BATCH MARKET ENGINE WITH 30-SEC MEMORY CACHE
# ==============================================================================
MARKET_CACHE = {"timestamp": 0, "data": None}

def fetch_tr_global_markets(force_refresh: bool = False):
    now = time.time()
    if not force_refresh and MARKET_CACHE["data"] and (now - MARKET_CACHE["timestamp"] < 30):
        return MARKET_CACHE["data"]

    watchlist = [
        ("USD/TRY", "TRY=X", "₺", "FX"),
        ("EUR/TRY", "EURTRY=X", "₺", "FX"),
        ("GBP/TRY", "GBPTRY=X", "₺", "FX"),
        ("BIST 100", "XU100.IS", "", "INDEX"),
        ("ONS ALTIN", "GC=F", "$", "COMMODITY"),
        ("GÜMÜŞ (OZ)", "SI=F", "$", "COMMODITY"),
        ("BRENT PETROL", "BZ=F", "$", "COMMODITY"),
        ("S&P 500", "^GSPC", "", "INDEX"),
        ("NASDAQ 100", "^NDX", "", "INDEX"),
        ("BITCOIN", "BTC-USD", "$", "CRYPTO"),
        ("ETHEREUM", "ETH-USD", "$", "CRYPTO"),
        ("VIX KORKU", "^VIX", "", "VOL")
    ]
    symbols = [w[1] for w in watchlist]
    raw_df = yf.download(symbols, period="5d", progress=False, threads=True)["Close"]

    results = []
    usd_try, usd_try_prev = 0.0, 0.0
    ons_gold, ons_gold_prev = 0.0, 0.0
    vix_val = 15.0

    for name, sym, prefix, cat in watchlist:
        try:
            if sym not in raw_df.columns:
                continue
            series = raw_df[sym].dropna()
            if len(series) >= 2:
                last = float(series.iloc[-1])
                prev = float(series.iloc[-2])
                chg = ((last / prev) - 1) * 100
                if sym == "TRY=X":
                    usd_try, usd_try_prev = last, prev
                elif sym == "GC=F":
                    ons_gold, ons_gold_prev = last, prev
                elif sym == "^VIX":
                    vix_val = last

                decimals = 4 if cat == "FX" else 2
                results.append({
                    "name": name,
                    "symbol": sym,
                    "prefix": prefix,
                    "category": cat,
                    "price": round(last, decimals),
                    "change_pct": round(chg, 2)
                })
        except Exception:
            continue

    if usd_try > 0 and ons_gold > 0:
        gram_try = (ons_gold / 31.1034768) * usd_try
        gram_prev = (ons_gold_prev / 31.1034768) * usd_try_prev
        gram_chg = ((gram_try / gram_prev) - 1) * 100 if gram_prev > 0 else 0.0
        results.insert(3, {
            "name": "GRAM ALTIN",
            "symbol": "GAUTRY",
            "prefix": "₺",
            "category": "COMMODITY",
            "price": round(gram_try, 2),
            "change_pct": round(gram_chg, 2)
        })

    regime = "DÜŞÜK VOLATİLİTE // RİSK İŞTAHI AÇIK" if vix_val < 18 else (
        "ORTA VOLATİLİTE // DENGELİ" if vix_val < 26 else "YÜKSEK STRES // DEFANSİF MOD"
    )
    payload = {
        "timestamp": datetime.utcnow().strftime("%H:%M:%S UTC"),
        "usd_try": round(usd_try, 4) if usd_try > 0 else 34.50,
        "vix": round(vix_val, 2),
        "regime": regime,
        "markets": results
    }
    MARKET_CACHE["timestamp"] = now
    MARKET_CACHE["data"] = payload
    return payload

@app.get("/api/markets/tr-live")
def get_tr_live_markets(refresh: bool = False):
    return fetch_tr_global_markets(force_refresh=refresh)

@app.websocket("/ws/live")
async def live_telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await asyncio.to_thread(fetch_tr_global_markets)
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(25)
    except WebSocketDisconnect:
        pass

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/health")
def health_check():
    return {"status": "online", "engine": "Inertia Quant Core", "rag_ready": RAG_AVAILABLE}

# ==============================================================================
# PERSONAL PORTFOLIO LIVE VALUATION (USD + TRY)
# ==============================================================================
@app.post("/api/portfolio/evaluate")
def evaluate_personal_portfolio(req: PortfolioEvalRequest):
    if not req.items:
        return {"usd_try": 34.5, "total_usd": 0, "total_try": 0, "pnl_usd": 0, "pnl_pct": 0, "rows": []}

    mkt = fetch_tr_global_markets()
    usd_try = mkt.get("usd_try", 34.50)

    tickers = list(set([it.ticker.upper().strip() for it in req.items if it.ticker.strip()]))
    raw_df = yf.download(tickers, period="5d", progress=False, threads=True)["Close"]
    if isinstance(raw_df, pd.Series):
        raw_df = raw_df.to_frame(name=tickers[0])

    rows = []
    tot_val_usd = 0.0
    tot_cost_usd = 0.0

    for it in req.items:
        sym = it.ticker.upper().strip()
        try:
            spot = float(raw_df[sym].dropna().iloc[-1]) if sym in raw_df.columns else it.avg_cost
        except Exception:
            spot = it.avg_cost

        is_try = sym.endswith(".IS") or it.currency.upper() == "TRY"
        spot_usd = (spot / usd_try) if is_try else spot
        cost_usd = (it.avg_cost / usd_try) if is_try else it.avg_cost

        val_usd = spot_usd * it.qty
        cst_usd = cost_usd * it.qty
        pnl_usd = val_usd - cst_usd
        pnl_pct = ((spot / it.avg_cost) - 1) * 100 if it.avg_cost > 0 else 0.0

        tot_val_usd += val_usd
        tot_cost_usd += cst_usd

        rows.append({
            "ticker": sym,
            "qty": it.qty,
            "avg_cost": round(it.avg_cost, 2),
            "spot": round(spot, 2),
            "currency": "TRY" if is_try else "USD",
            "val_usd": round(val_usd, 2),
            "val_try": round(val_usd * usd_try, 2),
            "pnl_pct": round(pnl_pct, 2)
        })

    tot_pnl_usd = tot_val_usd - tot_cost_usd
    tot_pnl_pct = ((tot_val_usd / tot_cost_usd) - 1) * 100 if tot_cost_usd > 0 else 0.0

    return {
        "usd_try": round(usd_try, 4),
        "total_usd": round(tot_val_usd, 2),
        "total_try": round(tot_val_usd * usd_try, 2),
        "pnl_usd": round(tot_pnl_usd, 2),
        "pnl_try": round(tot_pnl_usd * usd_try, 2),
        "pnl_pct": round(tot_pnl_pct, 2),
        "rows": rows
    }

# ==============================================================================
# PAPER TRADING ALGORITHMIC BOT SIMULATOR (RSI + EMA CROSSOVER)
# ==============================================================================
@app.post("/api/bot/simulate")
def simulate_trading_bot(req: BotSimRequest):
    sym = req.ticker.upper().strip()
    df = yf.Ticker(sym).history(period=req.period)
    if len(df) < 30:
        raise HTTPException(status_code=404, detail="Simülasyon için yeterli veri yok.")

    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    ema20 = close.ewm(span=20, adjust=False).mean()

    cash = req.initial_cash
    shares = 0.0
    equity_curve = []
    trades = []

    for i in range(len(df)):
        dt_str = df.index[i].strftime("%Y-%m-%d")
        px = float(close.iloc[i])
        r_val = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else 50.0
        e_val = float(ema20.iloc[i])

        # Alım Sinyali: RSI aşırı satım altında veya fiyat EMA20 üstünde momentum
        if cash > 0 and (r_val <= req.rsi_buy or (px > e_val and r_val < 52)):
            shares = cash / px
            trades.append({"date": dt_str, "side": "AL (BUY)", "price": round(px, 2), "rsi": round(r_val, 1), "value": round(cash, 2)})
            cash = 0.0
        # Satım Sinyali: RSI aşırı alım üstünde
        elif shares > 0 and r_val >= req.rsi_sell:
            cash = shares * px
            trades.append({"date": dt_str, "side": "SAT (SELL)", "price": round(px, 2), "rsi": round(r_val, 1), "value": round(cash, 2)})
            shares = 0.0

        tot_eq = cash + (shares * px)
        equity_curve.append(round(tot_eq, 2))

    final_eq = equity_curve[-1]
    bot_ret = ((final_eq / req.initial_cash) - 1) * 100
    bh_ret = ((float(close.iloc[-1]) / float(close.iloc[0])) - 1) * 100

    return {
        "ticker": sym,
        "initial_cash": req.initial_cash,
        "final_equity": round(final_eq, 2),
        "bot_return_pct": round(bot_ret, 2),
        "buy_hold_pct": round(bh_ret, 2),
        "trade_count": len(trades),
        "trades": trades[-12:][::-1],
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "equity": equity_curve
    }

# ==============================================================================
# TECHNICAL CHART & BATCH SCREENER ENDPOINTS
# ==============================================================================
@app.get("/api/chart/{ticker}")
def get_technical_chart(ticker: str, period: str = "6mo"):
    sym = ticker.upper().strip()
    df = yf.Ticker(sym).history(period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail="Veri bulunamadı.")
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

    return {
        "ticker": sym,
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "close": [round(float(x), 4 if "TRY" in sym else 2) for x in close],
        "sma20": [round(float(x), 4 if "TRY" in sym else 2) if pd.notna(x) else None for x in sma20],
        "bb_upper": [round(float(x), 4 if "TRY" in sym else 2) if pd.notna(x) else None for x in bb_upper],
        "bb_lower": [round(float(x), 4 if "TRY" in sym else 2) if pd.notna(x) else None for x in bb_lower],
        "rsi": [round(float(x), 1) if pd.notna(x) else 50.0 for x in rsi]
    }

@app.post("/api/screener")
def run_quant_screener(req: ScreenerRequest):
    symbols = [s.upper().strip() for s in req.tickers if s.strip()]
    if not symbols:
        return {"signals": []}
    raw_df = yf.download(symbols, period="6mo", progress=False, threads=True)["Close"]
    if isinstance(raw_df, pd.Series):
        raw_df = raw_df.to_frame(name=symbols[0])

    results = []
    for sym in symbols:
        try:
            if sym not in raw_df.columns:
                continue
            close = raw_df[sym].dropna()
            if len(close) < 35:
                continue
            last_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            chg_1d = ((last_price / prev_price) - 1) * 100

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rsi = float((100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).iloc[-1])

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_hist = float((macd_line - macd_line.ewm(span=9, adjust=False).mean()).iloc[-1])

            ret_20d = ((last_price / float(close.iloc[-21])) - 1) * 100
            ann_vol = float(close.pct_change().dropna().std() * np.sqrt(252) * 100)

            score = (2 if rsi < 35 else (-2 if rsi > 70 else 0)) + (1.5 if macd_hist > 0 else -1.5) + (1 if ret_20d > 0 else 0)
            signal = "GÜÇLÜ AL" if score >= 2.5 else ("BİRİKTİR" if score > 0.5 else ("AZALT" if score < -1.5 else "NÖTR"))

            results.append({
                "ticker": sym,
                "price": round(last_price, 2),
                "change_1d": round(chg_1d, 2),
                "ret_20d": round(ret_20d, 2),
                "rsi": round(rsi, 1),
                "macd_hist": round(macd_hist, 3),
                "volatility": round(ann_vol, 1),
                "signal": signal
            })
        except Exception:
            continue
    return {"signals": results}

# ==============================================================================
# MARKOWITZ & BACKTEST ENGINES
# ==============================================================================
@app.post("/api/optimize")
def optimize_portfolio(req: OptimizeRequest):
    tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="En az 2 varlık girilmelidir.")
    data = yf.download(tickers, period=req.period, progress=False, threads=True)["Close"]
    if isinstance(data, pd.Series):
        raise HTTPException(status_code=400, detail="Çoklu varlık verisi çekilemedi.")
    data = data.dropna()
    if len(data) < 40:
        raise HTTPException(status_code=404, detail="Yeterli tarihsel veri yok.")

    valid_tickers = list(data.columns)
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()

    rf = req.risk_free_rate / 100.0
    n = len(valid_tickers)
    sims = min(max(req.simulations, 1000), 8000)

    weights_record = np.random.dirichlet(np.ones(n), size=sims)
    port_returns = weights_record @ mean_returns.values
    port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights_record, cov_matrix.values, weights_record))
    sharpe_ratios = (port_returns - rf) / np.where(port_vols > 0, port_vols, 1e-6)

    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    min_vol_idx = int(np.argmin(port_vols))

    step = max(1, sims // 550)
    scatter = [{"x": round(float(port_vols[i]*100), 2), "y": round(float(port_returns[i]*100), 2)} for i in range(0, sims, step)]

    return {
        "tickers": valid_tickers,
        "scatter": scatter,
        "max_sharpe": {
            "return_pct": round(float(port_returns[max_sharpe_idx] * 100), 2),
            "vol_pct": round(float(port_vols[max_sharpe_idx] * 100), 2),
            "sharpe": round(float(sharpe_ratios[max_sharpe_idx]), 2),
            "weights": {valid_tickers[i]: round(float(weights_record[max_sharpe_idx, i] * 100), 2) for i in range(n)}
        },
        "min_vol": {
            "return_pct": round(float(port_returns[min_vol_idx] * 100), 2),
            "vol_pct": round(float(port_vols[min_vol_idx] * 100), 2),
            "sharpe": round(float(sharpe_ratios[min_vol_idx]), 2),
            "weights": {valid_tickers[i]: round(float(weights_record[min_vol_idx, i] * 100), 2) for i in range(n)}
        },
        "correlation": corr_matrix.round(2).values.tolist()
    }

@app.post("/api/backtest")
def run_portfolio_backtest(req: BacktestRequest):
    tickers = [t.upper().strip() for t in req.weights.keys() if req.weights[t] > 0]
    raw_w = np.array([req.weights[t] for t in req.weights.keys() if req.weights[t] > 0], dtype=float)
    if len(tickers) == 0 or raw_w.sum() <= 0:
        raise HTTPException(status_code=400, detail="Geçerli varlık ağırlıkları girilmelidir.")
    weights = raw_w / raw_w.sum()

    all_symbols = list(set(tickers + [req.benchmark.upper()]))
    data = yf.download(all_symbols, period=req.period, progress=False, threads=True)["Close"]
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

    var_95 = float(np.percentile(p_daily, 5) * 100)
    cvar_95 = float(p_daily[p_daily <= np.percentile(p_daily, 5)].mean() * 100)
    win_rate = float((p_daily > 0).mean() * 100)

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
            "beta": round(beta, 2),
            "var_95": round(var_95, 2),
            "cvar_95": round(cvar_95, 2),
            "win_rate": round(win_rate, 1)
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
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG modülü aktif değil.")
    return ask_sec_filing(req.question, ticker=req.ticker, k=req.top_k)

@app.post("/api/sec-ai/ingest/{ticker}")
def ingest_sec_filing(ticker: str):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG modülü aktif değil.")
    doc_data = fetch_latest_10k(ticker.upper())
    file_path = os.path.join(DATA_DIR, "sec_filings", f"{ticker.upper()}_10K_{doc_data['filing_date']}.json")
    index_filing_to_chroma(file_path)
    return {"status": "indexed", "ticker": ticker.upper(), "filing_date": doc_data["filing_date"]}