"""
Быстрая версия fetch_history.py — качает только 1H (для широкого прогона по
корзине монет; полный набор таймфреймов для 20 монет занял бы слишком
долго). Использует ту же fetch_all() логику.
"""
import json
import os
import sys
import time

from bitget_client import BitgetClient
from fetch_history import fetch_all, DAYS, DATA_DIR

SYMBOLS = sys.argv[1:]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    client = BitgetClient(api_key="", secret_key="", passphrase="")

    for symbol in SYMBOLS:
        print(f"Скачиваю {symbol} 1H за {DAYS} дней...")
        candles = fetch_all(client, symbol, "1h", 60 * 60 * 1000, DAYS)
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        with open(path, "w") as f:
            json.dump(candles, f)
        if candles:
            from datetime import datetime, timezone
            first = datetime.fromtimestamp(candles[0]["ts"] / 1000, tz=timezone.utc)
            last = datetime.fromtimestamp(candles[-1]["ts"] / 1000, tz=timezone.utc)
            print(f"  {len(candles)} свечей, {first.date()} .. {last.date()}")
        else:
            print(f"  ПУСТО для {symbol}")


if __name__ == "__main__":
    main()
