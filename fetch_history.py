"""
Скачивает историю за N дней на нескольких таймфреймах через history-candles
(с пагинацией по endTime) и сохраняет в data/*.json, чтобы бэктест не дёргал
API при каждом прогоне.

Публичные эндпоинты, ключи не нужны.

Запуск: python3 fetch_history.py [SYMBOL ...]
Без аргументов — качает ETHUSDT (поведение по умолчанию, как раньше).
"""
import json
import os
import sys
import time

from bitget_client import BitgetClient

DAYS = int(os.environ.get("FETCH_DAYS", "180"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GRANULARITIES = {
    "15min": ("15min", 15 * 60 * 1000),
    "1H": ("1h", 60 * 60 * 1000),
    "4H": ("4h", 4 * 60 * 60 * 1000),
}


def fetch_all(client: BitgetClient, symbol: str, api_granularity: str, bar_ms: int, days: int):
    end_time = int(time.time() * 1000)
    cutoff = end_time - days * 24 * 60 * 60 * 1000
    all_candles = {}

    while end_time > cutoff:
        batch = client.get_history_candles(
            symbol=symbol, granularity=api_granularity,
            end_time_ms=end_time, limit=200,
        )
        if not batch:
            break
        for c in batch:
            all_candles[c["ts"]] = c

        oldest_ts = min(c["ts"] for c in batch)
        if oldest_ts >= end_time:
            break
        end_time = oldest_ts - 1
        time.sleep(0.3)

    candles = sorted(all_candles.values(), key=lambda c: c["ts"])
    candles = [c for c in candles if c["ts"] >= cutoff]
    return candles


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    client = BitgetClient(api_key="", secret_key="", passphrase="")

    symbols = sys.argv[1:] or ["ETHUSDT"]

    for symbol in symbols:
        for label, (api_gran, bar_ms) in GRANULARITIES.items():
            print(f"Скачиваю {symbol} {label} ({api_gran}) за {DAYS} дней...")
            candles = fetch_all(client, symbol, api_gran, bar_ms, DAYS)
            path = os.path.join(DATA_DIR, f"{symbol}_{label}.json")
            with open(path, "w") as f:
                json.dump(candles, f)
            if candles:
                from datetime import datetime, timezone
                first = datetime.fromtimestamp(candles[0]["ts"] / 1000, tz=timezone.utc)
                last = datetime.fromtimestamp(candles[-1]["ts"] / 1000, tz=timezone.utc)
                print(f"  {len(candles)} свечей, {first.date()} .. {last.date()} -> {path}")
            else:
                print("  ПУСТО — проверь granularity/API")


if __name__ == "__main__":
    main()
