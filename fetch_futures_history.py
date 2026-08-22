"""
Качает историю USDT-M бессрочных фьючерсов (свечи с объёмом + funding rate)
за 180 дней и сохраняет в data/*.json.

Запуск: python3 fetch_futures_history.py [SYMBOL ...]
Без аргументов — ETHUSDT.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from bitget_client import BitgetClient

DAYS = 180
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRANULARITY = "1H"


def fetch_candles(client: BitgetClient, symbol: str, days: int):
    end_time = int(time.time() * 1000)
    cutoff = end_time - days * 24 * 60 * 60 * 1000
    all_candles = {}
    while end_time > cutoff:
        batch = client.get_futures_history_candles(symbol, GRANULARITY, end_time_ms=end_time, limit=200)
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
    return [c for c in candles if c["ts"] >= cutoff]


def fetch_funding(client: BitgetClient, symbol: str, days: int):
    cutoff = int(time.time() * 1000) - days * 24 * 60 * 60 * 1000
    all_rates = {}
    page = 1
    while True:
        batch = client.get_funding_rate_history(symbol, page_no=page, page_size=100)
        if not batch:
            break
        for r in batch:
            all_rates[r["ts"]] = r
        if min(r["ts"] for r in batch) < cutoff:
            break
        page += 1
        time.sleep(0.3)
    rates = sorted(all_rates.values(), key=lambda r: r["ts"])
    return [r for r in rates if r["ts"] >= cutoff]


def attach_funding(candles: list[dict], funding: list[dict]) -> list[dict]:
    """Приклеивает к каждой свече последнюю известную на тот момент ставку funding."""
    result = []
    fi = 0
    current_rate = funding[0]["rate"] if funding else 0.0
    for c in candles:
        while fi < len(funding) and funding[fi]["ts"] <= c["ts"]:
            current_rate = funding[fi]["rate"]
            fi += 1
        cc = dict(c)
        cc["funding_rate"] = current_rate
        result.append(cc)
    return result


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    symbols = sys.argv[1:] or ["ETHUSDT"]

    for symbol in symbols:
        print(f"Качаю фьючерс {symbol} {GRANULARITY} за {DAYS} дней...")
        candles = fetch_candles(client, symbol, DAYS)
        print(f"  свечей: {len(candles)}")
        funding = fetch_funding(client, symbol, DAYS)
        print(f"  funding-событий: {len(funding)}")
        merged = attach_funding(candles, funding)

        path = os.path.join(DATA_DIR, f"{symbol}_FUT_1H.json")
        with open(path, "w") as f:
            json.dump(merged, f)
        if merged:
            first = datetime.fromtimestamp(merged[0]["ts"] / 1000, tz=timezone.utc)
            last = datetime.fromtimestamp(merged[-1]["ts"] / 1000, tz=timezone.utc)
            print(f"  сохранено: {len(merged)} свечей, {first.date()} .. {last.date()} -> {path}")


if __name__ == "__main__":
    main()
