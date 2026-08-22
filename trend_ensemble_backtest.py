"""
Ансамбль из топ-3 конфигураций по широте (см. trend_config_search_report.txt):
  1. ADX>=30, ATRx3.0, std=2.5  (69.6% широты на train)
  2. ADX>=25, ATRx3.0, std=2.5  (65.2% широты на train)
  3. ADX>=25, ATRx2.0, std=2.5  (65.2% широты на train)

Комбинирование — не повторный подбор по test (эти три уже выбраны по train),
а равновзвешенное усреднение их результата на каждой монете: как если бы
треть капитала торговалась по каждой конфигурации параллельно. Это снижает
риск того, что весь результат держится на одной удачной точке сетки
параметров, а не на самой идее пробоя+ADX+ATR-трейлинг.
"""
import json
import os

from trend_backtest import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FRAC = 0.7
MIN_TRADES = 20

TOP3_CONFIGS = [
    dict(period=20, num_std=2.5, adx_period=14, adx_threshold=30.0, atr_period=14,
         atr_stop_mult=3.0, max_holding_bars=100, fee_pct_per_side=0.1),
    dict(period=20, num_std=2.5, adx_period=14, adx_threshold=25.0, atr_period=14,
         atr_stop_mult=3.0, max_holding_bars=100, fee_pct_per_side=0.1),
    dict(period=20, num_std=2.5, adx_period=14, adx_threshold=25.0, atr_period=14,
         atr_stop_mult=2.0, max_holding_bars=100, fee_pct_per_side=0.1),
]

SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "HYPEUSDT",
    "BTCUSDT", "XRPUSDT", "DOGEUSDT", "ZECUSDT", "SUIUSDT", "PEPEUSDT",
    "ENAUSDT", "ONDOUSDT", "TRUMPUSDT", "LINKUSDT", "BNBUSDT", "UNIUSDT",
    "ADAUSDT", "LTCUSDT", "NEARUSDT", "XLMUSDT", "AAVEUSDT", "AVAXUSDT",
    "TRXUSDT", "BCHUSDT",
]


def load_all_candles():
    data = {}
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            candles = json.load(f)
        split_i = int(len(candles) * TRAIN_FRAC)
        data[symbol] = (candles[:split_i], candles[split_i:])
    return data


def evaluate_ensemble(candles_by_symbol: dict, part: str) -> dict:
    idx = 0 if part == "train" else 1
    per_symbol = {}
    for symbol, halves in candles_by_symbol.items():
        candles = halves[idx]
        sub_returns = []
        total_trades = 0
        for config in TOP3_CONFIGS:
            r = run_backtest(candles, **config)
            sub_returns.append(r.total_return_pct)
            total_trades += r.total_trades
        blended_return = sum(sub_returns) / len(sub_returns)
        per_symbol[symbol] = {
            "sub_returns": sub_returns,
            "blended_return": blended_return,
            "total_trades": total_trades,
            "reliable": total_trades >= MIN_TRADES,
        }
        print(f"  [{part}] {symbol}: ансамбль={blended_return:+.2f}%", flush=True)
    reliable = [v for v in per_symbol.values() if v["reliable"]]
    positive = [v for v in reliable if v["blended_return"] > 0]
    breadth = len(positive) / len(reliable) * 100 if reliable else 0.0
    median_return = sorted(v["blended_return"] for v in reliable)[len(reliable) // 2] if reliable else 0.0
    avg_return = sum(v["blended_return"] for v in reliable) / len(reliable) if reliable else 0.0
    return {"per_symbol": per_symbol, "n_reliable": len(reliable), "n_positive": len(positive),
            "breadth_pct": breadth, "median_return": median_return, "avg_return": avg_return}


def main():
    data = load_all_candles()
    print(f"Загружено монет: {len(data)}\n")

    for part in ("train", "test"):
        res = evaluate_ensemble(data, part)
        print(f"{'='*90}\nАНСАМБЛЬ (топ-3, равные доли) — {part.upper()}\n{'='*90}")
        print(f"{'Символ':<12}{'Сделок':>8}{'#1%':>9}{'#2%':>9}{'#3%':>9}{'Ансамбль%':>11}  Надёжн.")
        print("-" * 70)
        for symbol, v in sorted(res["per_symbol"].items(), key=lambda kv: kv[1]["blended_return"], reverse=True):
            flag = "OK" if v["reliable"] else "мало"
            s1, s2, s3 = v["sub_returns"]
            print(f"{symbol:<12}{v['total_trades']:>8}{s1:>+9.2f}{s2:>+9.2f}{s3:>+9.2f}"
                  f"{v['blended_return']:>+11.2f}  {flag}")
        print(f"\n{res['n_positive']}/{res['n_reliable']} монет в плюсе ({res['breadth_pct']:.1f}%), "
              f"медиана {res['median_return']:+.2f}%, среднее {res['avg_return']:+.2f}%\n")


if __name__ == "__main__":
    main()
