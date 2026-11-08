#!/usr/bin/env python3
"""
Angel One SmartAPI price-action alert bot: NIFTY / SENSEX -> Telegram.

AWS EC2 / Ubuntu / Python 3.12 friendly.
- Polls Angel One five-minute candles (REST API; this version does NOT use WebSocket).
- Uses near-month index futures for price + volume where available; otherwise spot.
- Analyses closed candles only.
- Sends CE/PE-style directional alerts only; NEVER places orders.
- Uses a JSON state file to avoid repeat processing and limit daily alerts.

Required environment variables:
  ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN, ANGEL_TOTP_SECRET
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Optional environment variables:
  STATE_FILE=state.json
  LOG_FILE=price_action_bot.log
  POLL_SECONDS=30
  SUMMARY_EVERY_MIN=60       (0 disables hourly summaries)
  MIN_SCORE=5
  MAX_ALERTS_PER_DAY=4
  TELEGRAM_ENABLED=true

Install dependencies:
  pip install smartapi-python pyotp requests pandas numpy python-dotenv

If using a .env file, install python-dotenv and the script loads it automatically.
Keep .env and all API credentials private; never commit them to Git.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import sys
import tempfile
import time
import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyotp
import requests
from SmartApi import SmartConnect

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------- CONFIG --------------------------------------

IST = ZoneInfo("Asia/Kolkata")

INTERVAL = "FIVE_MINUTE"
CANDLE_MIN = 5
POLL_SECONDS = max(15, int(os.getenv("POLL_SECONDS", "30")))
SWING_N = 3
EMA_LEN, RSI_LEN, ATR_LEN, VOL_LEN = 20, 14, 14, 20

MIN_SCORE = int(os.getenv("MIN_SCORE", "5"))
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "4"))
COOLDOWN_CANDLES = 3
MIN_BREAK_BODY_ATR = 0.6
SL_BUFFER_ATR = 0.1
MIN_RISK_ATR, MAX_RISK_ATR = 0.8, 2.5
T1_R, T2_R = 1.5, 2.5

# Approximate option premium sensitivity only. Not a live option-pricing model.
DELTA = 0.5

TRADE_START = dt.time(9, 25)
TRADE_END = dt.time(15, 10)
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 31)

SUMMARY_EVERY_MIN = int(os.getenv("SUMMARY_EVERY_MIN", "60"))
STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))
LOG_FILE = Path(os.getenv("LOG_FILE", "price_action_bot.log"))
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("1", "true", "yes")

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/"
    "OpenAPIScripMaster.json"
)

INDICES = {
    "NIFTY": {
        "exchange": "NSE",
        "token": "99926000",
        "spot_symbol": "Nifty 50",
        "deriv_exch": "NFO",
        "step": 50,
    },
    "SENSEX": {
        "exchange": "BSE",
        "token": "99919000",
        "spot_symbol": "SENSEX",
        "deriv_exch": "BFO",
        "step": 100,
    },
}

REQUIRED_ENV = (
    "ANGEL_API_KEY",
    "ANGEL_CLIENT_ID",
    "ANGEL_PIN",
    "ANGEL_TOTP_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)

STOP_REQUESTED = False


# -------------------------------- LOGGING ------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("price_action_bot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging unavailable: %s", exc)

    return logger


log = setup_logging()


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log.info("Shutdown signal %s received; finishing current cycle.", signum)


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


# ------------------------------- UTILITIES ----------------------------------

def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)


def f1(value: float) -> str:
    return f"{value:,.1f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def atomic_save_json(path: Path, data: dict) -> None:
    """Write state atomically so a restart cannot leave a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(data, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise


# ------------------------------- INDICATORS ---------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI with explicit handling of zero average gains/losses."""
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return result.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()


# ----------------------------- TELEGRAM / STATE -----------------------------

def send_telegram(message: str) -> bool:
    """Return True only when Telegram confirms every message chunk."""
    print("\n" + message + "\n", flush=True)

    if not TELEGRAM_ENABLED:
        log.warning("Telegram disabled by TELEGRAM_ENABLED.")
        return False

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("Telegram credentials are missing.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Keep below Telegram's 4096-character message limit.
    chunks = []
    remaining = message
    while remaining:
        if len(remaining) <= 3800:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, 3800)
        if cut < 1000:
            cut = 3800
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    for chunk in chunks:
        sent = False
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "disable_web_page_preview": True,
                    },
                    timeout=15,
                )
                if response.ok:
                    payload = response.json()
                    if payload.get("ok") is True:
                        sent = True
                        break
                log.error(
                    "Telegram response (attempt %s): HTTP %s %s",
                    attempt + 1, response.status_code, response.text[:500],
                )
            except requests.RequestException as exc:
                log.error("Telegram send failed (attempt %s): %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
        if not sent:
            return False
    return True


def load_state() -> dict:
    try:
        with STATE_FILE.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError("State root is not an object")
    except FileNotFoundError:
        state = {}
    except Exception as exc:
        log.exception("Could not read state file; starting with empty state: %s", exc)
        state = {}

    state.setdefault("last_candle", {})
    state.setdefault("last_signal", {})
    state.setdefault("alert_count", {})
    state.setdefault("last_summary", None)
    return state


def save_state(state: dict) -> None:
    try:
        atomic_save_json(STATE_FILE, state)
    except Exception as exc:
        log.exception("Could not save state: %s", exc)


# ------------------------------- ANGEL ONE ----------------------------------

class Angel:
    def __init__(self):
        self.api = None
        self.session_day = None
        self.last_login_attempt = 0.0

    def login(self, force: bool = False) -> None:
        if not force and time.time() - self.last_login_attempt < 15:
            raise RuntimeError("Login retry throttled; waiting before another attempt")

        self.last_login_attempt = time.time()
        api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
        otp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()

        response = api.generateSession(
            os.environ["ANGEL_CLIENT_ID"],
            os.environ["ANGEL_PIN"],
            otp,
        )
        if not isinstance(response, dict) or response.get("status") is not True:
            raise RuntimeError(f"Angel One login failed: {response}")

        self.api = api
        self.session_day = now_ist().date()
        log.info("Successfully logged in to Angel One.")

    def ensure(self) -> None:
        if self.api is None or self.session_day != now_ist().date():
            self.login()

    def invalidate(self) -> None:
        self.api = None
        self.session_day = None

    def candles(self, exchange: str, token: str) -> pd.DataFrame:
        self.ensure()
        current = now_ist()
        params = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": INTERVAL,
            "fromdate": (current - dt.timedelta(days=7)).strftime("%Y-%m-%d %H:%M"),
            "todate": current.strftime("%Y-%m-%d %H:%M"),
        }

        try:
            response = self.api.getCandleData(params)
        except Exception:
            # Do not assume every exception is an expired session. Let caller retry.
            raise

        if not isinstance(response, dict) or response.get("status") is not True:
            message = str(response)
            lower = message.lower()
            if any(word in lower for word in ("token", "session", "unauthorized", "jwt")):
                self.invalidate()
            raise RuntimeError(f"Angel candle API returned failure: {message[:700]}")

        rows = response.get("data") or []
        if not rows:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            rows, columns=["time", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # SmartAPI candle timestamps are normally exchange-local/naive.
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert(IST).dt.tz_localize(None)

        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["time", "open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0)
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["low"] <= df[["open", "close"]].min(axis=1))
        ]
        return (
            df.drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True)
        )

    def ltp(self, exchange: str, symbol: str, token: str) -> float | None:
        try:
            self.ensure()
            response = self.api.ltpData(exchange, symbol, str(token))
            if response.get("status") is not True:
                log.warning("LTP request failed for %s: %s", symbol, response)
                return None
            value = safe_float(response.get("data", {}).get("ltp"), default=np.nan)
            return value if np.isfinite(value) and value > 0 else None
        except Exception as exc:
            log.warning("LTP request failed for %s: %s", symbol, exc)
            return None


# ------------------------------ INSTRUMENTS ---------------------------------

def load_master() -> pd.DataFrame:
    log.info("Downloading Angel One instrument master.")
    response = requests.get(SCRIP_MASTER_URL, timeout=(15, 90))
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list):
        raise RuntimeError("Instrument master response was not a JSON list.")

    rows = [
        row for row in raw
        if row.get("instrumenttype") in ("OPTIDX", "FUTIDX")
        and row.get("name") in INDICES
    ]
    if not rows:
        raise RuntimeError("Instrument master contains no NIFTY/SENSEX derivatives.")

    master = pd.DataFrame(rows)
    master["strike"] = pd.to_numeric(master.get("strike"), errors="coerce") / 100.0
    master["expiry_dt"] = pd.to_datetime(
        master["expiry"].astype(str).str.upper(),
        format="%d%b%Y",
        errors="coerce",
    )
    master["token"] = master["token"].astype(str)
    master["symbol"] = master["symbol"].astype(str)
    return master


def pick_future(master: pd.DataFrame, name: str, exchange: str):
    today = pd.Timestamp(now_ist().date())
    candidates = master[
        (master["name"] == name)
        & (master["instrumenttype"] == "FUTIDX")
        & (master["exch_seg"] == exchange)
        & (master["expiry_dt"] >= today)
    ]
    if candidates.empty:
        return None
    return candidates.sort_values("expiry_dt").iloc[0]


def pick_option(
    master: pd.DataFrame, name: str, exchange: str,
    spot: float, step: int, side: str
):
    today = pd.Timestamp(now_ist().date())
    atm = int(np.floor((spot / step) + 0.5) * step)

    candidates = master[
        (master["name"] == name)
        & (master["instrumenttype"] == "OPTIDX")
        & (master["exch_seg"] == exchange)
        & (master["expiry_dt"] >= today)
        & (master["symbol"].str.endswith(side))
    ]
    if candidates.empty:
        return None

    nearest_expiry = candidates["expiry_dt"].min()
    candidates = candidates[
        (candidates["expiry_dt"] == nearest_expiry)
        & np.isclose(candidates["strike"], atm)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0]


# ----------------------------- PRICE ACTION ---------------------------------

def swing_points(df: pd.DataFrame, n: int = SWING_N):
    """Confirmed swing highs/lows; last n bars are intentionally unconfirmed."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    swing_highs, swing_lows = [], []

    for i in range(n, len(df) - n):
        if (
            highs[i] > highs[i - n:i].max()
            and highs[i] >= highs[i + 1:i + n + 1].max()
        ):
            swing_highs.append((i, float(highs[i])))

        if (
            lows[i] < lows[i - n:i].min()
            and lows[i] <= lows[i + 1:i + n + 1].min()
        ):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def market_structure(swing_highs, swing_lows, tolerance: float):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Sideways", "not enough confirmed swings"

    def direction(new: float, old: float) -> int:
        if new > old + tolerance:
            return 1
        if new < old - tolerance:
            return -1
        return 0

    high_direction = direction(swing_highs[-1][1], swing_highs[-2][1])
    low_direction = direction(swing_lows[-1][1], swing_lows[-2][1])

    high_label = {1: "HH", -1: "LH", 0: "EQH"}[high_direction]
    low_label = {1: "HL", -1: "LL", 0: "EQL"}[low_direction]

    trend = (
        "Bullish" if high_direction == 1 and low_direction == 1
        else "Bearish" if high_direction == -1 and low_direction == -1
        else "Sideways"
    )
    return trend, f"{high_label} + {low_label}"


def support_resistance(df: pd.DataFrame, swing_highs, swing_lows):
    """Nearest confirmed swing low below and swing high above current close."""
    close = float(df["close"].iloc[-1])

    supports = [price for _, price in swing_lows[-8:] if price < close]
    resistances = [price for _, price in swing_highs[-8:] if price > close]

    support = max(supports) if supports else float(df["low"].tail(60).min())
    resistance = (
        min(resistances) if resistances
        else float(df["high"].tail(60).max())
    )
    return support, resistance


def detect_break(df, swing_highs, swing_lows, trend, atr_value):
    """Detect a close through a confirmed swing level on the latest candle."""
    if len(df) < 2:
        return None

    current = df.iloc[-1]
    previous = df.iloc[-2]
    close = float(current["close"])
    previous_close = float(previous["close"])
    body = abs(close - float(current["open"]))

    if body < MIN_BREAK_BODY_ATR * atr_value:
        return None

    buffer = 0.05 * atr_value

    for _, level in reversed(swing_highs[-3:]):
        if previous_close <= level and close > level + buffer:
            kind = (
                "BOS" if trend == "Bullish"
                else "CHoCH" if trend == "Bearish"
                else "Breakout"
            )
            return {"dir": "up", "level": level, "kind": kind}

    for _, level in reversed(swing_lows[-3:]):
        if previous_close >= level and close < level - buffer:
            kind = (
                "BOS" if trend == "Bearish"
                else "CHoCH" if trend == "Bullish"
                else "Breakdown"
            )
            return {"dir": "down", "level": level, "kind": kind}

    return None


def candle_patterns(df: pd.DataFrame, atr_value: float):
    """Return patterns detected on the last closed candle."""
    if len(df) < 5:
        return []

    current, previous, two_back = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    open_, high = float(current["open"]), float(current["high"])
    low, close = float(current["low"]), float(current["close"])

    candle_range = high - low
    if candle_range <= 0:
        return []

    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    bullish = close > open_
    bearish = close < open_

    closes = df["close"]
    down_context = float(closes.iloc[-2]) < float(closes.iloc[-5])
    up_context = float(closes.iloc[-2]) > float(closes.iloc[-5])

    patterns = []

    if body <= 0.1 * candle_range:
        patterns.append(("Doji", "neutral"))

    if (
        body > 0 and lower_wick >= 2 * body
        and upper_wick <= max(0.5 * body, 0.1 * candle_range)
        and down_context
    ):
        patterns.append(("Hammer", "bull"))

    if (
        body > 0 and upper_wick >= 2 * body
        and lower_wick <= max(0.5 * body, 0.1 * candle_range)
        and up_context
    ):
        patterns.append(("Shooting Star", "bear"))

    previous_open = float(previous["open"])
    previous_close = float(previous["close"])
    previous_body = abs(previous_close - previous_open)

    if (
        previous_close < previous_open and bullish
        and close >= previous_open and open_ <= previous_close
        and body > previous_body and down_context
    ):
        patterns.append(("Bullish Engulfing", "bull"))

    if (
        previous_close > previous_open and bearish
        and close <= previous_open and open_ >= previous_close
        and body > previous_body and up_context
    ):
        patterns.append(("Bearish Engulfing", "bear"))

    if body >= 0.85 * candle_range and body >= 0.8 * atr_value:
        patterns.append(
            ("Bullish Marubozu", "bull")
            if bullish else ("Bearish Marubozu", "bear")
        )

    two_open = float(two_back["open"])
    two_close = float(two_back["close"])
    two_body = abs(two_close - two_open)

    if (
        two_close < two_open and two_body >= 0.6 * atr_value
        and previous_body <= 0.3 * two_body
        and bullish and close > (two_open + two_close) / 2
    ):
        patterns.append(("Morning Star", "bull"))

    if (
        two_close > two_open and two_body >= 0.6 * atr_value
        and previous_body <= 0.3 * two_body
        and bearish and close < (two_open + two_close) / 2
    ):
        patterns.append(("Evening Star", "bear"))

    return patterns


def rsi_zone(value: float) -> str:
    if value >= 70:
        return "overbought"
    if value >= 60:
        return "strong bullish momentum"
    if value >= 50:
        return "mild bullish"
    if value >= 40:
        return "mild bearish"
    if value > 30:
        return "strong bearish momentum"
    return "oversold"


def analyze(df: pd.DataFrame, has_volume: bool):
    if len(df) < 60:
        return None

    data = df.reset_index(drop=True).copy()
    data["ema"] = ema(data["close"], EMA_LEN)
    data["rsi"] = rsi(data["close"], RSI_LEN)
    data["atr"] = atr(data, ATR_LEN)

    current, previous = data.iloc[-1], data.iloc[-2]
    atr_value = safe_float(current["atr"])
    if atr_value <= 0:
        return None

    swing_highs, swing_lows = swing_points(data)
    trend, structure = market_structure(
        swing_highs, swing_lows, 0.1 * atr_value
    )
    support, resistance = support_resistance(
        data, swing_highs, swing_lows
    )

    relative_volume = None
    if has_volume:
        # Compare the just-closed candle with the preceding 20 bars, not a
        # moving average that includes the candle being evaluated.
        volume_average = data["volume"].shift(1).rolling(VOL_LEN).mean().iloc[-1]
        if pd.notna(volume_average) and volume_average > 0:
            relative_volume = float(current["volume"] / volume_average)

    ema_now = safe_float(current["ema"])
    ema_reference = safe_float(data["ema"].iloc[-4])
    ema_change = ema_now - ema_reference
    ema_slope = (
        "rising" if ema_change > 0.05 * atr_value
        else "falling" if ema_change < -0.05 * atr_value
        else "flat"
    )

    cross = None
    if previous["close"] <= previous["ema"] and current["close"] > current["ema"]:
        cross = "up"
    elif previous["close"] >= previous["ema"] and current["close"] < current["ema"]:
        cross = "down"

    rsi_value = safe_float(current["rsi"], 50)
    rsi_reference = safe_float(data["rsi"].iloc[-4], 50)
    rsi_direction = (
        "rising" if rsi_value > rsi_reference + 1
        else "falling" if rsi_value < rsi_reference - 1
        else "flat"
    )

    candle_range = float(current["high"] - current["low"])
    close_position = (
        (float(current["close"]) - float(current["low"])) / candle_range
        if candle_range > 0 else 0.5
    )

    return {
        "time": pd.Timestamp(current["time"]).to_pydatetime(),
        "open": float(current["open"]),
        "high": float(current["high"]),
        "low": float(current["low"]),
        "close": float(current["close"]),
        "atr": atr_value,
        "close_pos": close_position,
        "hs": swing_highs,
        "ls": swing_lows,
        "trend": trend,
        "struct": structure,
        "support": support,
        "resistance": resistance,
        "break": detect_break(
            data, swing_highs, swing_lows, trend, atr_value
        ),
        "patterns": candle_patterns(data, atr_value),
        "relvol": relative_volume,
        "ema": ema_now,
        "ema_slope": ema_slope,
        "cross": cross,
        "dist": (float(current["close"]) - ema_now) / atr_value,
        "rsi": rsi_value,
        "rsi_dir": rsi_direction,
    }


def decide(analysis: dict):
    """Return (side, score, total, checks, trigger), or None."""
    best = None
    atr_value = analysis["atr"]

    for side in ("CE", "PE"):
        bullish = side == "CE"
        desired_direction = "up" if bullish else "down"

        breakout = analysis["break"]
        matching_break = (
            breakout if breakout and breakout["dir"] == desired_direction
            else None
        )
        desired_pattern = "bull" if bullish else "bear"
        matching_patterns = [
            name for name, direction in analysis["patterns"]
            if direction == desired_pattern
        ]

        if bullish:
            near_level = (
                analysis["low"] <= analysis["support"] + 0.5 * atr_value
                and analysis["close"] >= analysis["support"]
            )
        else:
            near_level = (
                analysis["high"] >= analysis["resistance"] - 0.5 * atr_value
                and analysis["close"] <= analysis["resistance"]
            )

        if matching_break:
            trigger = matching_break["kind"]
        elif matching_patterns and near_level:
            trigger = (
                f"{matching_patterns[0]} at "
                f"{'support' if bullish else 'resistance'}"
            )
        else:
            continue

        checks = {
            "EMA": analysis["dist"] > 0 if bullish else analysis["dist"] < 0,
            "Slope": analysis["ema_slope"] == (
                "rising" if bullish else "falling"
            ),
            "RSI": (
                55 <= analysis["rsi"] <= 75
                if bullish else 25 <= analysis["rsi"] <= 45
            ),
        }

        if analysis["relvol"] is not None:
            checks["Volume"] = analysis["relvol"] >= 1.2

        checks["Structure"] = (
            bool(matching_break)
            or analysis["trend"] == ("Bullish" if bullish else "Bearish")
        )

        candle_body = analysis["close"] - analysis["open"]
        strong_candle = (
            candle_body > 0 and analysis["close_pos"] >= 0.6
            if bullish
            else candle_body < 0 and analysis["close_pos"] <= 0.4
        )
        checks["Candle"] = bool(matching_patterns) or strong_candle

        score = sum(bool(value) for value in checks.values())
        total = len(checks)

        if score >= MIN_SCORE and (
            best is None or score > best[1]
        ):
            best = (side, score, total, checks, trigger)

    return best


def trade_levels(analysis: dict, side: str):
    """Build underlying-price levels using the nearest relevant confirmed swing."""
    entry = analysis["close"]
    atr_value = analysis["atr"]

    if side == "CE":
        swing_levels = [
            price for _, price in analysis["ls"] if price < entry
        ]
        nearest = max(swing_levels) if swing_levels else None
        raw_risk = (
            entry - (nearest - SL_BUFFER_ATR * atr_value)
            if nearest is not None else 1.5 * atr_value
        )
    else:
        swing_levels = [
            price for _, price in analysis["hs"] if price > entry
        ]
        nearest = min(swing_levels) if swing_levels else None
        raw_risk = (
            (nearest + SL_BUFFER_ATR * atr_value) - entry
            if nearest is not None else 1.5 * atr_value
        )

    risk = min(
        max(raw_risk, MIN_RISK_ATR * atr_value),
        MAX_RISK_ATR * atr_value,
    )
    direction = 1 if side == "CE" else -1

    return {
        "entry": entry,
        "sl": entry - direction * risk,
        "t1": entry + direction * T1_R * risk,
        "t2": entry + direction * T2_R * risk,
        "risk": risk,
    }


# -------------------------------- MESSAGES -----------------------------------

def event_text(analysis: dict) -> str:
    event = analysis["break"]
    if not event:
        return "None on this candle"

    level = f1(event["level"])
    if event["dir"] == "up":
        descriptions = {
            "BOS": f"Bullish BOS - close above swing high {level}",
            "CHoCH": f"Bullish CHoCH - close above swing high {level}",
            "Breakout": f"Breakout above range high {level}",
        }
    else:
        descriptions = {
            "BOS": f"Bearish BOS - close below swing low {level}",
            "CHoCH": f"Bearish CHoCH - close below swing low {level}",
            "Breakdown": f"Breakdown below range low {level}",
        }
    return descriptions.get(event["kind"], event["kind"])


def volume_text(analysis: dict) -> str:
    relative = analysis["relvol"]
    if relative is None:
        return "n/a (no usable volume data)"
    label = (
        "High" if relative >= 1.5
        else "Above average" if relative >= 1.2
        else "Normal" if relative >= 0.8
        else "Low"
    )
    return f"{relative:.1f}x preceding 20-bar average ({label})"


def analysis_block(analysis: dict) -> str:
    side = "above" if analysis["dist"] >= 0 else "below"
    ema_line = (
        f"{f1(analysis['ema'])} - price {side} by "
        f"{abs(analysis['dist']):.1f} ATR, EMA {analysis['ema_slope']}"
    )
    if analysis["cross"]:
        ema_line += f", just crossed {analysis['cross']}"

    patterns = (
        ", ".join(name for name, _ in analysis["patterns"])
        or "No clear pattern"
    )

    return "\n".join(
        [
            f"Trend: {analysis['trend']} ({analysis['struct']})",
            f"Event: {event_text(analysis)}",
            f"Support {f1(analysis['support'])} | "
            f"Resistance {f1(analysis['resistance'])}",
            f"Candle: {patterns}",
            f"Volume: {volume_text(analysis)}",
            f"EMA20: {ema_line}",
            f"RSI14: {analysis['rsi']:.0f} - "
            f"{rsi_zone(analysis['rsi'])}, {analysis['rsi_dir']}",
        ]
    )


def signal_message(
    name: str, source: dict, analysis: dict, signal_result,
    levels: dict, option, premium, spot_basis
) -> str:
    side, score, total, checks, trigger = signal_result
    icon = "🟢" if side == "CE" else "🔴"
    candle_close_time = analysis["time"] + dt.timedelta(minutes=CANDLE_MIN)

    lines = [
        f"{icon} {name} - DIRECTIONAL BUY {side} (score {score}/{total})",
        f"Setup: {trigger} | 5m candle closed {candle_close_time:%H:%M} IST",
        f"Analysis source: {source['label']} @ {f1(analysis['close'])}",
        "",
        "UNDERLYING-PRICE LEVELS (not option premium levels)",
        f"Reference entry ~{f1(levels['entry'])}",
        f"Invalidation / SL {f1(levels['sl'])}",
        f"Target 1 {f1(levels['t1'])} ({T1_R}R) | "
        f"Target 2 {f1(levels['t2'])} ({T2_R}R)",
    ]

    if spot_basis is not None:
        lines.append(
            f"Approx. spot-equivalent levels: "
            f"SL {f1(levels['sl'] - spot_basis)} | "
            f"T1 {f1(levels['t1'] - spot_basis)} | "
            f"T2 {f1(levels['t2'] - spot_basis)}"
        )

    if option is not None:
        lines += [
            "",
            f"ATM OPTION REFERENCE: {option['symbol']} "
            f"(lot size {option.get('lotsize', 'n/a')})",
        ]
        if premium is not None and premium > 0:
            lines.append(
                f"Observed option LTP ~{premium:.2f}; this is a quote, "
                "not a recommended entry."
            )
            lines.append(
                f"Illustrative delta-{DELTA:.1f} premium movement only: "
                f"risk ~{DELTA * levels['risk']:.1f} points; "
                f"T1 ~{DELTA * T1_R * levels['risk']:.1f}; "
                f"T2 ~{DELTA * T2_R * levels['risk']:.1f}."
            )
            lines.append(
                "Premium estimates ignore changing delta, gamma, theta, "
                "IV, spread and slippage; do not use as option SL/targets."
            )
    else:
        lines += ["", "ATM option contract could not be resolved from master."]

    lines += [
        "",
        "ANALYSIS",
        analysis_block(analysis),
        "",
        "Checks: " + " ".join(
            f"{key} {'✓' if value else '✗'}"
            for key, value in checks.items()
        ),
        "ALERT ONLY - no order has been placed.",
        "Validate liquidity, spread, contract expiry and risk before any trade.",
    ]
    return "\n".join(lines)


def summary_message(latest: dict) -> str:
    sections = []
    for name, analysis in latest.items():
        close_time = analysis["time"] + dt.timedelta(minutes=CANDLE_MIN)
        sections.append(
            f"📊 {name} ({analysis['label']}) "
            f"{close_time:%H:%M} IST | {f1(analysis['close'])}\n"
            f"{analysis_block(analysis)}"
        )
    return "HOURLY MARKET SUMMARY\n\n" + "\n\n".join(sections)


# ------------------------------- MAIN LOGIC ---------------------------------

def is_closed_candle_frame(df: pd.DataFrame, current_time: dt.datetime):
    """Remove the in-progress candle; candle timestamps are treated as start times."""
    if df.empty:
        return df
    now_naive = current_time.replace(tzinfo=None)
    return df[
        df["time"] + dt.timedelta(minutes=CANDLE_MIN) <= now_naive
    ].reset_index(drop=True)


def handle_index(
    name: str, config: dict, source: dict, angel: Angel,
    master: pd.DataFrame, state: dict, latest: dict
) -> None:
    frame = angel.candles(source["exchange"], source["token"])
    current_time = now_ist()
    frame = is_closed_candle_frame(frame, current_time)

    if len(frame) < 60:
        log.info("[%s] Not enough candles yet (%s).", name, len(frame))
        return

    last_candle_time = pd.Timestamp(frame["time"].iloc[-1]).to_pydatetime()
    candle_close_time = last_candle_time + dt.timedelta(minutes=CANDLE_MIN)

    if current_time.replace(tzinfo=None) - candle_close_time > dt.timedelta(minutes=20):
        log.warning("[%s] Candle data is stale; skipping.", name)
        return

    candle_key = last_candle_time.isoformat()
    if state["last_candle"].get(name) == candle_key:
        return

    analysis = analyze(frame, source["volume"])
    if analysis is None:
        log.warning("[%s] Analysis unavailable for candle %s.", name, candle_key)
        return

    analysis["label"] = source["label"]
    latest[name] = analysis

    # Outside the alert window, mark the candle processed and move on.
    if not (TRADE_START <= candle_close_time.time() <= TRADE_END):
        state["last_candle"][name] = candle_key
        save_state(state)
        return

    signal_result = decide(analysis)
    if not signal_result:
        state["last_candle"][name] = candle_key
        save_state(state)
        return

    side = signal_result[0]
    previous_signal = state["last_signal"].get(name)
    cooldown_active = False

    if previous_signal and previous_signal.get("side") == side:
        try:
            previous_time = dt.datetime.fromisoformat(previous_signal["time"])
            cooldown_active = (
                last_candle_time - previous_time
                < dt.timedelta(minutes=CANDLE_MIN * COOLDOWN_CANDLES)
            )
        except (ValueError, TypeError, KeyError):
            cooldown_active = False

    day_key = last_candle_time.date().isoformat()
    counts = state["alert_count"].get(name, {})
    alerts_used = counts.get("n", 0) if counts.get("day") == day_key else 0

    if cooldown_active or alerts_used >= MAX_ALERTS_PER_DAY:
        log.info(
            "[%s] Signal suppressed (cooldown=%s, daily_count=%s).",
            name, cooldown_active, alerts_used,
        )
        state["last_candle"][name] = candle_key
        save_state(state)
        return

    levels = trade_levels(analysis, side)

    spot_ltp = angel.ltp(
        config["exchange"], config["spot_symbol"], config["token"]
    )
    spot_basis = (
        analysis["close"] - spot_ltp
        if spot_ltp and source["volume"] else None
    )
    option_reference = spot_ltp if spot_ltp else analysis["close"]

    option = pick_option(
        master, name, config["deriv_exch"],
        option_reference, config["step"], side,
    )
    premium = None
    if option is not None:
        premium = angel.ltp(
            config["deriv_exch"], option["symbol"], option["token"]
        )

    message = signal_message(
        name, source, analysis, signal_result,
        levels, option, premium, spot_basis,
    )

    # Do not persist the candle/signal as handled unless Telegram confirms delivery.
    if send_telegram(message):
        state["last_candle"][name] = candle_key
        state["last_signal"][name] = {
            "side": side,
            "time": candle_key,
        }
        state["alert_count"][name] = {
            "day": day_key,
            "n": alerts_used + 1,
        }
        save_state(state)
        log.info("[%s] Alert delivered: %s score %s/%s.", name, side,
                 signal_result[1], signal_result[2])
    else:
        log.error("[%s] Telegram delivery failed; candle will be retried.", name)


def build_sources(master: pd.DataFrame) -> dict:
    sources = {}
    for name, config in INDICES.items():
        future = pick_future(master, name, config["deriv_exch"])
        if future is not None:
            sources[name] = {
                "exchange": config["deriv_exch"],
                "token": str(future["token"]),
                "label": str(future["symbol"]),
                "volume": True,
            }
        else:
            sources[name] = {
                "exchange": config["exchange"],
                "token": config["token"],
                "label": f"{name} spot",
                "volume": False,
            }
            log.warning(
                "[%s] No current futures contract found; falling back to spot. "
                "Volume confirmation will be omitted.", name,
            )
    return sources


def validate_environment() -> None:
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        raise SystemExit(
            "Missing environment variables: " + ", ".join(missing)
        )

    if not 1 <= MIN_SCORE <= 6:
        raise SystemExit("MIN_SCORE must be between 1 and 6.")


def run_bot(once: bool = False) -> None:
    validate_environment()

    state = load_state()
    angel = Angel()
    master = None
    sources = {}
    loaded_day = None
    latest = {}

    log.info("Bot started. Mode: %s. Alert-only; no order placement.",
             "single cycle" if once else "continuous")
    log.info("Configured indices: %s", ", ".join(INDICES))

    while not STOP_REQUESTED:
        now = now_ist()
        current_time = now.time()

        if now.weekday() >= 5:
            if once:
                log.info("Weekend; no market processing.")
                return
            time.sleep(60)
            continue

        if not (dt.time(9, 0) <= current_time <= dt.time(15, 35)):
            if once:
                log.info("Outside market operating window.")
                return
            time.sleep(60)
            continue

        if loaded_day != now.date():
            try:
                master = load_master()
                sources = build_sources(master)
                loaded_day = now.date()
                latest = {}
                online_text = "🤖 Angel One price-action bot online.\n" + "\n".join(
                    f"{index}: {source['label']} "
                    f"(volume {'on' if source['volume'] else 'off'})"
                    for index, source in sources.items()
                )
                if not send_telegram(online_text):
                    log.warning("Could not deliver startup Telegram message.")
            except Exception as exc:
                log.exception("Daily initialization failed: %s", exc)
                if once:
                    raise
                time.sleep(60)
                continue

        if not (MARKET_OPEN <= current_time <= MARKET_CLOSE):
            if once:
                log.info("Before/after candle processing window.")
                return
            time.sleep(30)
            continue

        for name, config in INDICES.items():
            if STOP_REQUESTED:
                break
            try:
                handle_index(
                    name, config, sources[name], angel,
                    master, state, latest,
                )
            except Exception as exc:
                log.exception("[%s] Processing error: %s", name, exc)

        if SUMMARY_EVERY_MIN > 0 and latest:
            previous_summary = state.get("last_summary")
            summary_due = False
            if not previous_summary:
                summary_due = True
            else:
                try:
                    last_summary_time = dt.datetime.fromisoformat(previous_summary)
                    summary_due = (
                        now_ist().replace(tzinfo=None) - last_summary_time
                        >= dt.timedelta(minutes=SUMMARY_EVERY_MIN)
                    )
                except (ValueError, TypeError):
                    summary_due = True

            if summary_due:
                if send_telegram(summary_message(latest)):
                    state["last_summary"] = now_ist().replace(
                        tzinfo=None
                    ).isoformat()
                    save_state(state)

        if once:
            break
        time.sleep(POLL_SECONDS)

    log.info("Bot stopped cleanly.")


def main():
    if "--test" in sys.argv:
        if not TELEGRAM_ENABLED:
            raise SystemExit("Set TELEGRAM_ENABLED=true to run the Telegram test.")
        if not send_telegram(
            "✅ Test message from your Angel One price-action bot. "
            "Telegram delivery is working."
        ):
            raise SystemExit("Telegram test failed; check token, chat ID and logs.")
        log.info("Telegram test succeeded.")
        return

    run_bot(once="--once" in sys.argv)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        raise
