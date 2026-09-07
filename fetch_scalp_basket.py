"""
Качает 5-минутные свечи для отобранных "спокойных" монет (см.
select_calm_coins.py) — таймфрейм для скальп-бэктеста (scalp_backtest.py).
Пятиминутки, а не минутки: на минутках цель в доли процента почти всегда
решается шумом и комиссия съедает вообще всё почти при любом target_pct (см.
разбор в bb_config_search_v2.py про арифметику комиссии от таймфрейма), а
качать полную минутную историю на десяток монет — это на порядок больше
запросов к API без явной причины на этом, первом проходе идеи.

Та же fetch_all() логика, что и в fetch_history.py/fetch_1h_basket.py.

Запуск: python3 fetch_scalp_basket.py SYMBOL [SYMBOL ...]
Глубина истории — переменная окружения FETCH_DAYS (по умолчанию 60).
"""
import json
import os
import sys

from bitget_client import BitgetClient
from fetch_history import fetch_all, DATA_DIR

DAYS = int(os.environ.get("FETCH_DAYS", "60"))
SYMBOLS = sys.argv[1:]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    client = BitgetClient(api_key="", secret_key="", passphrase="")

    for symbol in SYMBOLS:
        print(f"Скачиваю {symbol} 5min за {DAYS} дней...", flush=True)
        candles = fetch_all(client, symbol, "5min", 5 * 60 * 1000, DAYS)
        path = os.path.join(DATA_DIR, f"{symbol}_5min.json")
        with open(path, "w") as f:
            json.dump(candles, f)
        if candles:
            from datetime import datetime, timezone
            first = datetime.fromtimestamp(candles[0]["ts"] / 1000, tz=timezone.utc)
            last = datetime.fromtimestamp(candles[-1]["ts"] / 1000, tz=timezone.utc)
            print(f"  {len(candles)} свечей, {first.date()} .. {last.date()} -> {path}", flush=True)
        else:
            print(f"  ПУСТО для {symbol}", flush=True)


if __name__ == "__main__":
    main()
