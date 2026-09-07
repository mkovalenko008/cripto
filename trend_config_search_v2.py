"""
Поиск конфигурации защиты от отдачи прибыли для трендовой стратегии v2.

Дисциплина та же, что в trend_config_search.py: сетка гоняется по TRAIN,
победитель выбирается ТОЛЬКО по train, и уже потом один раз прогоняется по
TEST без какой-либо дальнейшей подгонки. Базовая конфигурация входа
(ADX>=30, ATR×3.0, std=2.5) зафиксирована — она уже прошла train/test в
прошлом поиске, и переподбирать её заново на тех же данных значило бы
использовать test дважды.

Меняется только управление выходом:
  breakeven_at — перевод стопа в безубыток при прибыли N×R
  tighten_at/tighten_mult — сжатие трейлинга при прибыли M×R

Базовая строка сетки (None/None) — это v1 с LONG-only, то есть контроль:
если ни один вариант защиты её не обходит, значит защита не нужна.
"""
from __future__ import annotations

import json
import os

from trend_backtest_v2 import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FRAC = 0.7
MIN_TRADES = 5        # монета считается надёжной, если дала хотя бы столько сделок
MIN_COVERAGE = 15     # конфигурация ранжируется, только если надёжна на стольки монетах

# Базовая конфигурация входа — уже валидирована, не трогаем.
BASE = dict(period=20, num_std=2.5, adx_threshold=30.0, atr_stop_mult=3.0,
            max_holding_bars=100, fee_pct_per_side=0.1, long_only=True)

# (breakeven_at, tighten_at, tighten_mult)
VARIANTS = [
    (None, None, None),      # контроль: v1 + long_only, без защиты
    (1.0,  None, None),
    (1.5,  None, None),
    (2.0,  None, None),
    (1.0,  2.0,  1.5),
    (1.0,  2.0,  2.0),
    (1.5,  2.5,  1.5),
    (1.0,  1.5,  1.5),
    (None, 2.0,  1.5),
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


def evaluate(data: dict, part: str, variant: tuple) -> dict:
    be, ta, tm = variant
    idx = 0 if part == "train" else 1
    per_symbol, returns, givebacks = {}, [], []
    n_reliable = n_positive = 0

    for symbol, parts in data.items():
        candles = parts[idx]
        res = run_backtest(candles, breakeven_at=be, tighten_at=ta, tighten_mult=tm, **BASE)
        total = res.total_return_pct
        per_symbol[symbol] = (res.total_trades, total)
        if res.total_trades >= MIN_TRADES:
            n_reliable += 1
            returns.append(total)
            givebacks.append(res.avg_giveback_pct)
            if total > 0:
                n_positive += 1

    breadth = (n_positive / n_reliable * 100) if n_reliable else 0.0
    median = sorted(returns)[len(returns) // 2] if returns else 0.0
    return {
        "variant": variant, "n_reliable": n_reliable, "n_positive": n_positive,
        "breadth": breadth, "median": median,
        "mean": sum(returns) / len(returns) if returns else 0.0,
        "giveback": sum(givebacks) / len(givebacks) if givebacks else 0.0,
        "per_symbol": per_symbol,
    }


def label(v):
    be, ta, tm = v
    if be is None and ta is None:
        return "без защиты (контроль)"
    bits = []
    if be is not None:
        bits.append(f"безубыток при {be}R")
    if ta is not None:
        bits.append(f"сжатие трейлинга до ATR×{tm} при {ta}R")
    return ", ".join(bits)


def main():
    data = load_all_candles()
    if not data:
        print("Нет данных в data/ — сначала выкачай историю (fetch_1h_basket.py)")
        return
    print(f"Загружено монет: {len(data)}\n")

    train_results = []
    for v in VARIANTS:
        r = evaluate(data, "train", v)
        train_results.append(r)
        print(f"  train {label(v):55s} широта={r['breadth']:5.1f}%  медиана={r['median']:+7.2f}%  "
              f"отдача={r['giveback']:5.2f} п.п.  монет={r['n_reliable']}")

    ranked = [r for r in train_results if r["n_reliable"] >= MIN_COVERAGE]
    if not ranked:
        print("\nНи одна конфигурация не набрала покрытия — выборка слишком мала")
        return
    ranked.sort(key=lambda r: (-r["breadth"], -r["median"]))
    best = ranked[0]

    lines = []
    lines.append("=" * 92)
    lines.append("ПОИСК ПО TRAIN — защита от отдачи прибыли (вход зафиксирован: ADX>=30, ATR×3.0, std=2.5, LONG-only)")
    lines.append("=" * 92)
    header = f"{'Конфигурация':<56}{'Широта%':>9}{'Медиана%':>11}{'Отдача п.п.':>13}{'Монет':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(train_results, key=lambda r: -r["breadth"]):
        lines.append(f"{label(r['variant']):<56}{r['breadth']:>9.1f}{r['median']:>11.2f}"
                     f"{r['giveback']:>13.2f}{r['n_reliable']:>7}")

    lines.append("")
    lines.append(f"Лучшая по TRAIN: {label(best['variant'])} — "
                 f"{best['n_positive']}/{best['n_reliable']} монет в плюсе ({best['breadth']:.1f}%), "
                 f"медиана {best['median']:+.2f}%")

    control_train = next(r for r in train_results if r["variant"] == (None, None, None))
    lines.append(f"Контроль (без защиты) на TRAIN: широта {control_train['breadth']:.1f}%, "
                 f"медиана {control_train['median']:+.2f}%, отдача {control_train['giveback']:.2f} п.п.")

    lines.append("")
    lines.append("=" * 92)
    lines.append("ТЕ ЖЕ КОНФИГУРАЦИИ НА TEST (без дальнейшей подгонки)")
    lines.append("=" * 92)
    test_best = evaluate(data, "test", best["variant"])
    test_control = evaluate(data, "test", (None, None, None))

    lines.append(f"{'Символ':<14}{'Сделок':>8}{'Итог% (v2)':>13}{'Итог% (контроль)':>19}")
    lines.append("-" * 54)
    for symbol in sorted(test_best["per_symbol"], key=lambda s: -test_best["per_symbol"][s][1]):
        n, tot = test_best["per_symbol"][symbol]
        cn, ctot = test_control["per_symbol"].get(symbol, (0, 0.0))
        mark = "" if n >= MIN_TRADES else "  (мало сделок)"
        lines.append(f"{symbol:<14}{n:>8}{tot:>13.2f}{ctot:>19.2f}{mark}")

    lines.append("")
    lines.append(f"TEST победитель train: {test_best['n_positive']}/{test_best['n_reliable']} в плюсе "
                 f"({test_best['breadth']:.1f}%), медиана {test_best['median']:+.2f}%, "
                 f"среднее {test_best['mean']:+.2f}%, отдача {test_best['giveback']:.2f} п.п.")
    lines.append(f"TEST контроль (без защиты): {test_control['n_positive']}/{test_control['n_reliable']} в плюсе "
                 f"({test_control['breadth']:.1f}%), медиана {test_control['median']:+.2f}%, "
                 f"среднее {test_control['mean']:+.2f}%, отдача {test_control['giveback']:.2f} п.п.")
    lines.append("")
    lines.append("Вывод считать честным только если победитель обходит контроль И на train, И на test.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(os.path.dirname(__file__), "trend_config_search_v2_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
