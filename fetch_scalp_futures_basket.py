"""
Качает 5-минутные свечи USDT-M бессрочных фьючерсов + funding rate для
отобранных "спокойных" монет (см. calm_coins.json) — фьючерсный аналог
fetch_scalp_basket.py, для проверки идеи "пара тиков" на фьючерсах вместо
спота (ниже комиссия — 0.06%/сторону вместо 0.1%, см. futures_backtest.py —
и доступен шорт).

Та же пагинация по endTime, что и в fetch_futures_history.py, но на 5m вместо
1H и по списку монет, а не одной. funding_rate приклеивается к каждой свече
той же логикой (attach_funding), что и там.

Запуск: python3 fetch_scalp_futures_basket.py SYMBOL [SYMBOL ...]
Глубина истории — FETCH_DAYS (по умолчанию 60), как у спотового скрипта —
те же 60 дней, чтобы сравнение спот/фьючерс было на одном и том же окне.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from bitget_client import BitgetClient
from fetch_futures_history import fetch_funding, attach_funding

DAYS = int(os.environ.get("FETCH_DAYS", "60"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRANULARITY = "5m"


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


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    symbols = sys.argv[1:]

    for symbol in symbols:
        print(f"Качаю фьючерс {symbol} {GRANULARITY} за {DAYS} дней...", flush=True)
        candles = fetch_candles(client, symbol, DAYS)
        funding = fetch_funding(client, symbol, DAYS)
        merged = attach_funding(candles, funding)

        path = os.path.join(DATA_DIR, f"{symbol}_FUT_5min.json")
        with open(path, "w") as f:
            json.dump(merged, f)
        if merged:
            first = datetime.fromtimestamp(merged[0]["ts"] / 1000, tz=timezone.utc)
            last = datetime.fromtimestamp(merged[-1]["ts"] / 1000, tz=timezone.utc)
            print(f"  {len(merged)} свечей, funding-событий: {len(funding)}, "
                  f"{first.date()} .. {last.date()} -> {path}", flush=True)
        else:
            print(f"  ПУСТО для {symbol}", flush=True)


if __name__ == "__main__":
    main()
