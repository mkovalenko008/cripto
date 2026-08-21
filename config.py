"""
Конфигурация бота. Все значения читаются из переменных окружения (.env файл).
Никогда не хардкодь ключи прямо в коде.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


# --- Bitget API credentials (создать ключ ТОЛЬКО с правами Spot Trade,
#     БЕЗ права Withdraw, и желательно с привязкой к IP) ---
API_KEY = os.getenv("BITGET_API_KEY", "")
SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

BASE_URL = "https://api.bitget.com"

# --- Торговые параметры ---
SYMBOL = os.getenv("SYMBOL", "ETHUSDT")          # без слэша, как на Bitget
TRADE_SIZE_USDT = _env_float("TRADE_SIZE_USDT", 1.0)  # размер сделки в USDT
CANDLE_GRANULARITY = os.getenv("CANDLE_GRANULARITY", "15min")
POLL_INTERVAL_SECONDS = _env_int("POLL_INTERVAL_SECONDS", 60)

# --- Пример стратегии (RSI) — см. strategy.py, замени на свою логику ---
RSI_PERIOD = _env_int("RSI_PERIOD", 14)
RSI_OVERSOLD = _env_float("RSI_OVERSOLD", 30.0)
RSI_OVERBOUGHT = _env_float("RSI_OVERBOUGHT", 70.0)

# --- Риск-менеджмент (это не опционально, это защита от бага) ---
MAX_DAILY_LOSS_USDT = _env_float("MAX_DAILY_LOSS_USDT", 5.0)
MAX_TRADES_PER_DAY = _env_int("MAX_TRADES_PER_DAY", 10)
KILL_SWITCH_FILE = os.getenv("KILL_SWITCH_FILE", "STOP")

# --- Главный предохранитель. По умолчанию TRUE.
#     Пока не увидишь в логах адекватные решения бота — не трогай. ---
DRY_RUN = _env_bool("DRY_RUN", True)

TRADES_LOG_FILE = os.getenv("TRADES_LOG_FILE", "trades_log.jsonl")
BOT_LOG_FILE = os.getenv("BOT_LOG_FILE", "bot.log")
