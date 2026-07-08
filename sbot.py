import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# 150 liquid NSE stocks (TATAMOTORS removed)
SYMBOLS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
    "BAJFINANCE.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","NTPC.NS","ONGC.NS",
    "POWERGRID.NS","WIPRO.NS","HCLTECH.NS","ULTRACEMCO.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","ADANIPORTS.NS","ADANIENT.NS","DIVISLAB.NS","DRREDDY.NS",
    "CIPLA.NS","BRITANNIA.NS","HDFCLIFE.NS","SBILIFE.NS","EICHERMOT.NS","M&M.NS",
    "HINDZINC.NS","VEDL.NS","DLF.NS","INDIGO.NS","HAVELLS.NS","VOLTAS.NS",
    "DABUR.NS","PIDILITIND.NS","BERGEPAINT.NS","LUPIN.NS","AUROPHARMA.NS",
    "BIOCON.NS","TORNTPHARM.NS","ALKEM.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "BAJAJFINSV.NS","BAJAJHLDNG.NS","BALKRISIND.NS","BANDHANBNK.NS","BEL.NS",
    "BHARATFORG.NS","BOSCHLTD.NS","BPCL.NS","CANBK.NS","CHOLAFIN.NS","COALINDIA.NS",
    "COLPAL.NS","CONCOR.NS","CUMMINSIND.NS","DEEPAKNTR.NS","ESCORTS.NS",
    "GAIL.NS","GODREJCP.NS","GODREJPROP.NS","GRASIM.NS","HAL.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDPETRO.NS","ICICIPRULI.NS",
    "IDFCFIRSTB.NS","INDUSINDBK.NS","INDUSTOWER.NS","IOC.NS","IRCTC.NS",
    "JINDALSTEL.NS","JUBLFOOD.NS","LICHSGFIN.NS","LUPIN.NS","M&MFIN.NS",
    "MARICO.NS","MFSL.NS","MOTHERSON.NS","MPHASIS.NS","MRF.NS","MUTHOOTFIN.NS",
    "NAUKRI.NS","NAVINFLUOR.NS","NESTLEIND.NS","OBEROIRLTY.NS","OFSS.NS",
    "PAGEIND.NS","PERSISTENT.NS","PETRONET.NS","PFC.NS",
    "PIIND.NS","PNB.NS","POLYCAB.NS","POONAWALLA.NS",
    "PRESTIGE.NS","RAMCOCEM.NS","RBLBANK.NS","RECLTD.NS","SAIL.NS",
    "SBICARD.NS","SHREECEM.NS","SIEMENS.NS","SRF.NS","SUNTV.NS",
    "SYNGENE.NS","TATACHEM.NS","TATACOMM.NS","TATACONSUM.NS","TECHM.NS",
    "TIINDIA.NS","TRENT.NS","TVSMOTOR.NS","UPL.NS","YESBANK.NS",
    "ZEEL.NS","ZOMATO.NS","PAYTM.NS","POLICYBZR.NS","NYKAA.NS","DELHIVERY.NS"
]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def flatten_columns(df):
    """Ensure columns are simple strings, not MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        # Join multi-level columns with underscore, then strip trailing underscore
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    return df

def compute_technicals(df):
    df = flatten_columns(df)
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, 1)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['vol_avg20'] = vol.rolling(20).mean()
    df['high20'] = high.rolling(20).max()
    df['low20'] = low.rolling(20).min()
    return df

def get_basic_fundamentals(symbol):
    try:
        info = yf.Ticker(symbol).info
        pe = info.get('trailingPE')
        de = info.get('debtToEquity')
        if pe and pe > 30: return False
        if de and de > 2.0: return False
        return True
    except:
        return True   # allow if data not available

def generate_signals():
    calls = []
    puts = []
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period="6mo", interval="1d", progress=False)
            if df.empty or len(df) < 100:
                continue
            df = compute_technicals(df)
            latest = df.iloc[-1]

            # Fundamental filter (skip if not passed)
            if not get_basic_fundamentals(sym):
                continue

            # Lenient volume condition
            vol_ok = latest['Volume'] > 0.9 * latest['vol_avg20']

            # --- CALL criteria ---
            uptrend = (latest['Close'] > latest['ema50'])
            rsi_call_ok = 40 < latest['rsi'] < 70
            near_high = latest['Close'] >= latest['high20'] * 0.92
            if uptrend and rsi_call_ok and vol_ok and near_high:
                entry = round(latest['Close'], 2)
                atr = latest['atr']
                target = round(entry + 4 * atr, 2)
                sl = round(entry - 2 * atr, 2)
                calls.append((sym.replace('.NS',''), entry, target, sl))

            # --- PUT criteria ---
            downtrend = (latest['Close'] < latest['ema50'])
            rsi_put_ok = 30 < latest['rsi'] < 60
            near_low = latest['Close'] <= latest['low20'] * 1.08
            if downtrend and rsi_put_ok and vol_ok and near_low:
                entry = round(latest['Close'], 2)
                atr = latest['atr']
                target = round(entry - 4 * atr, 2)
                sl = round(entry + 2 * atr, 2)
                puts.append((sym.replace('.NS',''), entry, target, sl))
        except Exception as e:
            print(f"Error {sym}: {e}")
            continue

    calls.sort(key=lambda x: x[1], reverse=True)
    puts.sort(key=lambda x: x[1], reverse=True)
    return calls[:5], puts[:5]

def send_report():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    date_str = now.strftime("%d %B %Y, %I:%M %p")

    calls, puts = generate_signals()

    message = f"🤖 NSE TRADING SIGNALS\n📅 {date_str}\n━━━━━━━━━━━━━━━\n\n"

    if calls:
        message += "📈 BUY (CALL) Signals:\n"
        for name, price, target, sl in calls:
            message += f"• {name}: ₹{price}\n  → Target: ₹{target} | SL: ₹{sl}\n\n"
    else:
        message += "📈 No BUY signals today\n\n"

    if puts:
        message += "📉 SELL (PUT) Signals:\n"
        for name, price, target, sl in puts:
            message += f"• {name}: ₹{price}\n  → Target: ₹{target} | SL: ₹{sl}\n\n"
    else:
        message += "📉 No SELL signals today\n\n"

    message += "━━━━━━━━━━━━━━━\n⚠️ SL mandatory | Targets based on volatility (ATR)"
    send_telegram(message)

if __name__ == "__main__":
    send_report()
