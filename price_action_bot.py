"""
Price-action alert bot  (Nifty / Sensex)  ->  Telegram
Data: Angel One SmartAPI.   Runs 24x7 on a cloud server (see deploy.sh).

Every closed 5-minute candle is analysed for:
  * Trend (bullish / bearish / sideways) and market structure (HH, HL, LH, LL)
  * Support & resistance zones (from swing points)
  * Breakout / breakdown, BOS (break of structure), CHoCH (change of character)
  * Candlestick patterns (Hammer, Shooting Star, Doji, Engulfing, Marubozu,
    Morning/Evening Star)
  * Volume confirmation (vs 20-bar average)
  * EMA 20 and RSI 14 analysis
A CE/PE alert is sent only when a trigger (break or reversal pattern at S/R)
fires AND enough confirmations agree (MIN_SCORE). The alert includes entry,
stop-loss, two targets and the full analysis above.

Volume: index spot has no volume, so the bot analyses the NEAR-MONTH FUTURES
candles (price + volume). If futures cannot be found it falls back to spot
and volume analysis is shown as n/a.

ALERTS ONLY - it never places orders. Not financial advice.
Run `python price_action_bot.py --test` to send a Telegram test message.
"""

import json
import os
import sys
import time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyotp
import requests
from SmartApi import SmartConnect

IST = ZoneInfo("Asia/Kolkata")

# ----------------------------- CONFIG ---------------------------------------
INTERVAL = "FIVE_MINUTE"
CANDLE_MIN = 5
POLL_SECONDS = 30
SWING_N = 3                      # bars each side to confirm a swing high/low
EMA_LEN, RSI_LEN, ATR_LEN, VOL_LEN = 20, 14, 14, 20
MIN_SCORE = 5                    # confirmations needed (out of 6, or 5 w/o volume)
MIN_BREAK_BODY_ATR = 0.6         # break candle body must be at least this many ATR
MAX_ALERTS_PER_DAY = 4           # per index, to avoid alert spam
COOLDOWN_CANDLES = 3             # no repeat alert, same side, within this many candles
SL_BUFFER_ATR = 0.1
MIN_RISK_ATR, MAX_RISK_ATR = 0.8, 2.5
T1_R, T2_R = 1.5, 2.5
DELTA = 0.5                      # rough ATM option delta for premium estimates
TRADE_START, TRADE_END = dt.time(9, 25), dt.time(15, 10)
MARKET_OPEN, MARKET_CLOSE = dt.time(9, 15), dt.time(15, 31)
SUMMARY_EVERY_MIN = 60           # hourly market summary to Telegram (0 = off)
STATE_FILE = os.environ.get("STATE_FILE", "state.json")

INDICES = {
    "NIFTY":  {"exchange": "NSE", "token": "99926000", "spot_symbol": "Nifty 50",
               "deriv_exch": "NFO", "step": 50},
    "SENSEX": {"exchange": "BSE", "token": "99919000", "spot_symbol": "SENSEX",
               "deriv_exch": "BFO", "step": 100},
}
SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST).replace(tzinfo=None)


def f1(x) -> str:
    return f"{x:,.1f}"


# ----------------------------- INDICATORS -----------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------- TELEGRAM / STATE -----------------------------
def send_telegram(text: str) -> None:
    print("\n" + text + "\n", flush=True)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(text), 3800):
        try:
            r = requests.post(url, json={"chat_id": chat, "text": text[i:i + 3800],
                                         "disable_web_page_preview": True}, timeout=15)
            if not r.ok:
                print("Telegram error:", r.text)
        except Exception as e:
            print("Telegram send failed:", e)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("last_candle", {})
    s.setdefault("last_signal", {})
    s.setdefault("last_summary", None)
    return s


def save_state(s: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f)
    except Exception as e:
        print("Could not save state:", e)


# ----------------------------- ANGEL ONE ------------------------------------
class Angel:
    def __init__(self):
        self.api = None
        self.day = None
        self.last_login = 0.0

    def login(self):
        api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
        totp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
        res = api.generateSession(os.environ["ANGEL_CLIENT_ID"], os.environ["ANGEL_PIN"], totp)
        self.last_login = time.time()
        if not res.get("status"):
            raise RuntimeError(f"Angel login failed: {res}")
        self.api, self.day = api, now_ist().date()
        print("Logged in to Angel One", flush=True)

    def ensure(self):
        if self.api is None or self.day != now_ist().date():
            if time.time() - self.last_login < 60:
                raise RuntimeError("waiting before re-login")
            self.login()

    def candles(self, exch, token) -> pd.DataFrame:
        self.ensure()
        now = now_ist()
        res = self.api.getCandleData({
            "exchange": exch, "symboltoken": token, "interval": INTERVAL,
            "fromdate": (now - dt.timedelta(days=6)).strftime("%Y-%m-%d %H:%M"),
            "todate": now.strftime("%Y-%m-%d %H:%M"),
        })
        if not res.get("status"):
            self.api = None            # force a fresh login on the next call
            raise RuntimeError(f"candle error: {res}")
        df = pd.DataFrame(res.get("data") or [],
                          columns=["time", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        if getattr(df["time"].dt, "tz", None) is not None:
            df["time"] = df["time"].dt.tz_localize(None)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    def ltp(self, exch, symbol, token):
        try:
            self.ensure()
            return float(self.api.ltpData(exch, symbol, token)["data"]["ltp"])
        except Exception:
            return None


def load_master() -> pd.DataFrame:
    print("Downloading instrument master...", flush=True)
    raw = requests.get(SCRIP_MASTER_URL, timeout=180).json()
    rows = [r for r in raw if r.get("instrumenttype") in ("OPTIDX", "FUTIDX")
            and r.get("name") in INDICES]
    del raw
    if not rows:
        raise RuntimeError("Instrument master had no index derivatives")
    df = pd.DataFrame(rows)
    df["strike"] = df["strike"].astype(float) / 100.0
    df["expiry_dt"] = pd.to_datetime(df["expiry"], format="%d%b%Y", errors="coerce")
    return df


def pick_future(master, name, exch):
    today = pd.Timestamp(now_ist().date())
    m = master[(master["name"] == name) & (master["instrumenttype"] == "FUTIDX")
               & (master["exch_seg"] == exch) & (master["expiry_dt"] > today)]
    return None if m.empty else m.sort_values("expiry_dt").iloc[0]


def pick_option(master, name, exch, spot, step, side):
    today = pd.Timestamp(now_ist().date())
    atm = round(spot / step) * step
    m = master[(master["name"] == name) & (master["instrumenttype"] == "OPTIDX")
               & (master["exch_seg"] == exch) & (master["expiry_dt"] >= today)
               & (master["symbol"].str.endswith(side))]
    if m.empty:
        return None
    m = m[m["expiry_dt"] == m["expiry_dt"].min()]
    m = m[m["strike"] == atm]
    return None if m.empty else m.iloc[0]


# ----------------------------- PRICE ACTION ---------------------------------
def swing_points(df, n=SWING_N):
    """Confirmed swing highs/lows as lists of (index, price)."""
    h, l = df["high"].values, df["low"].values
    hs, ls = [], []
    for i in range(n, len(df) - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            hs.append((i, float(h[i])))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            ls.append((i, float(l[i])))
    return hs, ls


def market_structure(hs, ls, tol):
    if len(hs) < 2 or len(ls) < 2:
        return "Sideways", "not enough swings"

    def d(a, b):
        return 1 if a > b + tol else (-1 if a < b - tol else 0)

    dh, dl = d(hs[-1][1], hs[-2][1]), d(ls[-1][1], ls[-2][1])
    lab_h = {1: "HH", -1: "LH", 0: "EQH"}[dh]
    lab_l = {1: "HL", -1: "LL", 0: "EQL"}[dl]
    trend = "Bullish" if (dh == 1 and dl == 1) else "Bearish" if (dh == -1 and dl == -1) \
        else "Sideways"
    return trend, f"{lab_h} + {lab_l}"


def support_resistance(df, hs, ls):
    close = float(df["close"].iloc[-1])
    levels = [p for _, p in hs[-8:]] + [p for _, p in ls[-8:]]
    above = [x for x in levels if x > close]
    below = [x for x in levels if x < close]
    res = min(above) if above else float(df["high"].tail(60).max())
    sup = max(below) if below else float(df["low"].tail(60).min())
    return sup, res


def detect_break(df, hs, ls, trend, atr_v):
    """Did the LAST closed candle break a recent swing high/low?"""
    c, p = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
    if abs(c - float(df["open"].iloc[-1])) < MIN_BREAK_BODY_ATR * atr_v:
        return None                       # weak candle, not a decisive break
    buf = 0.05 * atr_v
    for _, lvl in reversed(hs[-3:]):
        if p <= lvl and c > lvl + buf:
            kind = "BOS" if trend == "Bullish" else "CHoCH" if trend == "Bearish" else "Breakout"
            return {"dir": "up", "level": lvl, "kind": kind}
    for _, lvl in reversed(ls[-3:]):
        if p >= lvl and c < lvl - buf:
            kind = "BOS" if trend == "Bearish" else "CHoCH" if trend == "Bullish" else "Breakdown"
            return {"dir": "down", "level": lvl, "kind": kind}
    return None


def candle_patterns(df, atr_v):
    """Patterns on the last closed candle -> list of (name, 'bull'|'bear'|'neutral')."""
    c, p, pp = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    rng = h - l
    if rng <= 0:
        return []
    body = abs(cl - o)
    upper, lower = h - max(o, cl), min(o, cl) - l
    green = cl > o
    closes = df["close"]
    down_ctx = closes.iloc[-2] < closes.iloc[-5]
    up_ctx = closes.iloc[-2] > closes.iloc[-5]
    out = []
    if body <= 0.1 * rng:
        out.append(("Doji", "neutral"))
    if body > 0 and lower >= 2 * body and upper <= max(0.5 * body, 0.1 * rng) and down_ctx:
        out.append(("Hammer", "bull"))
    if body > 0 and upper >= 2 * body and lower <= max(0.5 * body, 0.1 * rng) and up_ctx:
        out.append(("Shooting Star", "bear"))
    p_o, p_c = float(p["open"]), float(p["close"])
    p_body = abs(p_c - p_o)
    if p_c < p_o and green and cl >= p_o and o <= p_c and body > p_body and down_ctx:
        out.append(("Bullish Engulfing", "bull"))
    if p_c > p_o and (not green) and cl <= p_o and o >= p_c and body > p_body and up_ctx:
        out.append(("Bearish Engulfing", "bear"))
    if body >= 0.85 * rng and body >= 0.8 * atr_v:
        out.append(("Bullish Marubozu", "bull") if green else ("Bearish Marubozu", "bear"))
    pp_o, pp_c = float(pp["open"]), float(pp["close"])
    pp_body = abs(pp_c - pp_o)
    if (pp_c < pp_o and pp_body >= 0.6 * atr_v and p_body <= 0.3 * pp_body
            and green and cl > (pp_o + pp_c) / 2):
        out.append(("Morning Star", "bull"))
    if (pp_c > pp_o and pp_body >= 0.6 * atr_v and p_body <= 0.3 * pp_body
            and (not green) and cl < (pp_o + pp_c) / 2):
        out.append(("Evening Star", "bear"))
    return out


def rsi_zone(v):
    if v >= 70:
        return "overbought"
    if v >= 60:
        return "strong bullish momentum"
    if v >= 50:
        return "mild bullish"
    if v >= 40:
        return "mild bearish"
    if v > 30:
        return "strong bearish momentum"
    return "oversold"


def analyze(df, has_volume):
    if len(df) < 60:
        return None
    d = df.reset_index(drop=True).copy()
    d["ema"] = ema(d["close"], EMA_LEN)
    d["rsi"] = rsi(d["close"], RSI_LEN)
    d["atr"] = atr(d, ATR_LEN)
    k, prev = d.iloc[-1], d.iloc[-2]
    atr_v = float(k["atr"])
    if not atr_v > 0:
        return None

    hs, ls = swing_points(d)
    trend, struct = market_structure(hs, ls, 0.1 * atr_v)
    sup, res = support_resistance(d, hs, ls)

    relvol = None
    if has_volume:
        vma = d["volume"].rolling(VOL_LEN).mean().iloc[-1]
        if vma and vma > 0:
            relvol = float(k["volume"] / vma)

    ema_now = float(k["ema"])
    slope = ema_now - float(d["ema"].iloc[-4])
    ema_slope = "rising" if slope > 0.05 * atr_v else "falling" if slope < -0.05 * atr_v else "flat"
    cross = None
    if prev["close"] <= prev["ema"] and k["close"] > k["ema"]:
        cross = "up"
    elif prev["close"] >= prev["ema"] and k["close"] < k["ema"]:
        cross = "down"

    rsi_v, rsi_prev = float(k["rsi"]), float(d["rsi"].iloc[-4])
    rsi_dir = "rising" if rsi_v > rsi_prev + 1 else "falling" if rsi_v < rsi_prev - 1 else "flat"
    rng = float(k["high"] - k["low"])

    return {
        "time": k["time"].to_pydatetime(), "open": float(k["open"]), "high": float(k["high"]),
        "low": float(k["low"]), "close": float(k["close"]), "atr": atr_v,
        "close_pos": (float(k["close"]) - float(k["low"])) / rng if rng > 0 else 0.5,
        "hs": hs, "ls": ls, "trend": trend, "struct": struct,
        "support": sup, "resistance": res,
        "break": detect_break(d, hs, ls, trend, atr_v),
        "patterns": candle_patterns(d, atr_v),
        "relvol": relvol,
        "ema": ema_now, "ema_slope": ema_slope, "cross": cross,
        "dist": (float(k["close"]) - ema_now) / atr_v,
        "rsi": rsi_v, "rsi_dir": rsi_dir,
    }


def decide(a):
    """Returns (side, score, total, checks, trigger) or None."""
    atr_v, best = a["atr"], None
    for side in ("CE", "PE"):
        bull = side == "CE"
        brk = a["break"] if (a["break"] and a["break"]["dir"] == ("up" if bull else "down")) else None
        pats = [n for n, b in a["patterns"] if b == ("bull" if bull else "bear")]
        near = (a["low"] <= a["support"] + 0.5 * atr_v) if bull \
            else (a["high"] >= a["resistance"] - 0.5 * atr_v)
        if brk:
            trigger = brk["kind"]
        elif pats and near:
            trigger = f"{pats[0]} at {'support' if bull else 'resistance'}"
        else:
            continue
        checks = {
            "EMA": a["dist"] > 0 if bull else a["dist"] < 0,
            "Slope": a["ema_slope"] == ("rising" if bull else "falling"),
            "RSI": (55 <= a["rsi"] <= 75) if bull else (25 <= a["rsi"] <= 45),
        }
        if a["relvol"] is not None:
            checks["Volume"] = a["relvol"] >= 1.2
        checks["Structure"] = bool(brk) or a["trend"] == ("Bullish" if bull else "Bearish")
        strong = ((a["close"] > a["open"]) == bull) and \
                 (a["close_pos"] >= 0.6 if bull else a["close_pos"] <= 0.4)
        checks["Candle"] = bool(pats) or strong
        score = sum(checks.values())
        if score >= MIN_SCORE and (best is None or score > best[1]):
            best = (side, score, len(checks), checks, trigger)
    return best


def trade_levels(a, side):
    entry, atr_v = a["close"], a["atr"]
    if side == "CE":
        below = [p for _, p in a["ls"] if p < entry]
        raw = entry - (below[-1] - SL_BUFFER_ATR * atr_v) if below else 1.5 * atr_v
    else:
        above = [p for _, p in a["hs"] if p > entry]
        raw = (above[-1] + SL_BUFFER_ATR * atr_v) - entry if above else 1.5 * atr_v
    risk = min(max(raw, MIN_RISK_ATR * atr_v), MAX_RISK_ATR * atr_v)
    s = 1 if side == "CE" else -1
    return {"entry": entry, "sl": entry - s * risk, "t1": entry + s * T1_R * risk,
            "t2": entry + s * T2_R * risk, "risk": risk}


# ----------------------------- MESSAGES -------------------------------------
def event_text(a):
    b = a["break"]
    if not b:
        return "None on this candle"
    lvl = f1(b["level"])
    if b["dir"] == "up":
        return {"BOS": f"Bullish BOS - closed above swing high {lvl} (breakout above resistance)",
                "CHoCH": f"Bullish CHoCH - closed above lower-high {lvl} (possible reversal up)",
                "Breakout": f"Breakout above range high {lvl}"}[b["kind"]]
    return {"BOS": f"Bearish BOS - closed below swing low {lvl} (breakdown below support)",
            "CHoCH": f"Bearish CHoCH - closed below higher-low {lvl} (possible reversal down)",
            "Breakdown": f"Breakdown below range low {lvl}"}[b["kind"]]


def volume_text(a):
    rv = a["relvol"]
    if rv is None:
        return "n/a (no volume data)"
    tag = "High" if rv >= 1.5 else "Above average" if rv >= 1.2 else "Normal" if rv >= 0.8 else "Low"
    return f"{rv:.1f}x 20-bar average ({tag})"


def analysis_block(a):
    side = "above" if a["dist"] >= 0 else "below"
    ema_line = f"{f1(a['ema'])} - price {side} by {abs(a['dist']):.1f} ATR, EMA {a['ema_slope']}"
    if a["cross"]:
        ema_line += f", just crossed {a['cross']}"
    pats = ", ".join(n for n, _ in a["patterns"]) or "No clear pattern"
    return "\n".join([
        f"Trend: {a['trend']} ({a['struct']})",
        f"Event: {event_text(a)}",
        f"Support {f1(a['support'])} | Resistance {f1(a['resistance'])}",
        f"Candle: {pats}",
        f"Volume: {volume_text(a)}",
        f"EMA20: {ema_line}",
        f"RSI14: {a['rsi']:.0f} - {rsi_zone(a['rsi'])}, {a['rsi_dir']}",
    ])


def signal_message(name, src, a, sig, lv, opt, prem, basis):
    side, score, total, checks, trigger = sig
    icon = "🟢" if side == "CE" else "🔴"
    ct = a["time"] + dt.timedelta(minutes=CANDLE_MIN)
    lines = [
        f"{icon} {name} - BUY {side}   (score {score}/{total})",
        f"Setup: {trigger} | 5m candle closed {ct:%H:%M}",
        f"Source: {src['label']} @ {f1(a['close'])}",
        "",
        "TRADE (on source price)",
        f"Entry ~{f1(lv['entry'])}",
        f"Stop loss {f1(lv['sl'])}",
        f"Target 1 {f1(lv['t1'])} ({T1_R}R) | Target 2 {f1(lv['t2'])} ({T2_R}R)",
    ]
    if basis is not None:
        lines.append(f"Spot equiv: SL {f1(lv['sl'] - basis)} | T1 {f1(lv['t1'] - basis)} "
                     f"| T2 {f1(lv['t2'] - basis)}")
    if opt is not None:
        lines += ["", f"OPTION: {opt['symbol']} (lot {opt['lotsize']})"]
        if prem:
            lines.append(f"Premium ~{prem:.1f} | SL ~{max(prem - DELTA * lv['risk'], 0.05):.1f} | "
                         f"T1 ~{prem + DELTA * T1_R * lv['risk']:.1f} | "
                         f"T2 ~{prem + DELTA * T2_R * lv['risk']:.1f}")
            lines.append("(rough estimate, delta 0.5, ignores theta)")
    lines += ["", "ANALYSIS", analysis_block(a), "",
              "Checks: " + " ".join(f"{k} {'✓' if v else '✗'}" for k, v in checks.items()),
              "Alert only - manage your own risk."]
    return "\n".join(lines)


def summary_message(latest):
    parts = []
    for name, a in latest.items():
        ct = a["time"] + dt.timedelta(minutes=CANDLE_MIN)
        parts.append(f"📊 {name} ({a['label']}) {ct:%H:%M} | {f1(a['close'])}\n{analysis_block(a)}")
    return "\n\n".join(parts)


# ----------------------------- MAIN LOOP ------------------------------------
def handle_index(name, cfg, src, angel, master, state, latest):
    df = angel.candles(src["exch"], src["token"])
    now = now_ist()
    df = df[df["time"] + dt.timedelta(minutes=CANDLE_MIN) <= now].reset_index(drop=True)
    if len(df) < 60:
        return
    last_t = df["time"].iloc[-1].to_pydatetime()
    if now - last_t > dt.timedelta(minutes=20):      # stale data / holiday
        return
    key = last_t.isoformat()
    if state["last_candle"].get(name) == key:        # already processed
        return
    state["last_candle"][name] = key

    a = analyze(df, src["volume"])
    if a is None:
        return
    a["label"] = src["label"]
    latest[name] = a
    close_t = last_t + dt.timedelta(minutes=CANDLE_MIN)
    if TRADE_START <= close_t.time() <= TRADE_END:
        sig = decide(a)
        if sig:
            side = sig[0]
            prev = state["last_signal"].get(name)
            recent = prev and prev["side"] == side and \
                (last_t - dt.datetime.fromisoformat(prev["time"])) < \
                dt.timedelta(minutes=CANDLE_MIN * COOLDOWN_CANDLES)
            today_key = last_t.date().isoformat()
            cnt = state.setdefault("alert_count", {}).get(name, {})
            used = cnt.get("n", 0) if cnt.get("day") == today_key else 0
            if not recent and used < MAX_ALERTS_PER_DAY:
                state["alert_count"][name] = {"day": today_key, "n": used + 1}
                lv = trade_levels(a, side)
                spot = angel.ltp(cfg["exchange"], cfg["spot_symbol"], cfg["token"])
                basis = (a["close"] - spot) if (spot and src["volume"]) else None
                ref = spot if spot else a["close"]
                opt = pick_option(master, name, cfg["deriv_exch"], ref, cfg["step"], side)
                prem = angel.ltp(cfg["deriv_exch"], opt["symbol"], opt["token"]) \
                    if opt is not None else None
                send_telegram(signal_message(name, src, a, sig, lv, opt, prem, basis))
                state["last_signal"][name] = {"side": side, "time": key}
    save_state(state)


def main():
    if "--test" in sys.argv:
        send_telegram("✅ Test message from your price-action bot. Telegram is working.")
        return
    for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET",
              "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if not os.environ.get(k):
            raise SystemExit(f"Missing environment variable: {k}")

    state, angel = load_state(), Angel()
    master, sources, day, latest = None, {}, None, {}
    print("Bot started.", flush=True)

    while True:
        now = dt.datetime.now(IST)
        t = now.time()
        if now.weekday() >= 5 or not (dt.time(9, 0) <= t <= dt.time(15, 35)):
            time.sleep(60)
            continue

        try:
            if day != now.date():                      # once per trading day
                master = load_master()
                sources = {}
                for name, cfg in INDICES.items():
                    fut = pick_future(master, name, cfg["deriv_exch"])
                    if fut is not None:
                        sources[name] = {"exch": cfg["deriv_exch"], "token": fut["token"],
                                         "label": fut["symbol"], "volume": True}
                    else:
                        sources[name] = {"exch": cfg["exchange"], "token": cfg["token"],
                                         "label": f"{name} spot", "volume": False}
                day = now.date()
                latest = {}
                send_telegram("🤖 Bot online. Analysing: " + "; ".join(
                    f"{n} on {s['label']} (volume {'on' if s['volume'] else 'off'})"
                    for n, s in sources.items()))
        except Exception as e:
            print("Init error:", e, flush=True)
            time.sleep(60)
            continue

        if t < MARKET_OPEN or t > MARKET_CLOSE:
            time.sleep(30)
            continue

        for name, cfg in INDICES.items():
            try:
                handle_index(name, cfg, sources[name], angel, master, state, latest)
            except Exception as e:
                print(f"[{name}] error: {e}", flush=True)
                time.sleep(5)

        if SUMMARY_EVERY_MIN and latest:
            last = state.get("last_summary")
            due = (not last) or (now_ist() - dt.datetime.fromisoformat(last) >=
                                 dt.timedelta(minutes=SUMMARY_EVERY_MIN))
            if due:
                send_telegram(summary_message(latest))
                state["last_summary"] = now_ist().isoformat()
                save_state(state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
