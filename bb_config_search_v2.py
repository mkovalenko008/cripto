"""
Поиск конфигурации mean-reversion v2 на часовом таймфрейме.

Зачем именно час: на живых данных v1 (минутки) стратегия дала +25.71 п.п.
ДО комиссии и -130.89 п.п. ПОСЛЕ. Комиссия 0.2% за круг — фиксированная, а
типичное движение растёт с таймфреймом: на минутках комиссия составляет 82%
от среднего движения, на 15 минутах 25%, на часе 16%. То есть проблема v1
арифметическая, а не рыночная, и первый кандидат на исправление — таймфрейм.

Дополнительно проверяются два параметра выхода, потому что разбор живых
сделок показал перекос не в ту сторону: средний выигрыш по цели +0.187%,
средний убыток по стопу -0.751% (1:4), и 46% сделок закрывались по таймауту
со средним -0.227%.
  stop_mult — множитель ширины полос для стопа (сейчас 1.0, стоп далеко)
  max_holding_bars — сколько баров держим до таймаута (сейчас 20)

Дисциплина: сетка гоняется по TRAIN, победитель выбирается только по train,
TEST прогоняется один раз без подгонки. Контроль — текущая конфигурация.
"""
from __future__ import annotations

import json
import os

from bb_strategy import decide, Side
from indicators import bollinger_bands

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FRAC = 0.7
MIN_TRADES = 5
MIN_COVERAGE = 15
LOOKBACK = 150      # сглаживание Уайлдера сходится за ~100-150 баров — так O(n) вместо O(n^2)
FEE_PCT_PER_SIDE = 0.1

STOP_MULT_VARIANTS = [0.5, 1.0, 1.5]
MAX_BARS_VARIANTS = [10, 20, 40]
CONTROL = (1.0, 20)   # то, что крутится в v1 сейчас

SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "HYPEUSDT",
    "BTCUSDT", "XRPUSDT", "DOGEUSDT", "ZECUSDT", "SUIUSDT", "PEPEUSDT",
    "ENAUSDT", "ONDOUSDT", "TRUMPUSDT", "LINKUSDT", "BNBUSDT", "UNIUSDT",
    "ADAUSDT", "LTCUSDT", "NEARUSDT", "XLMUSDT", "AAVEUSDT", "AVAXUSDT",
    "TRXUSDT", "BCHUSDT",
]


def run_backtest(candles, period=20, num_std=2.0, stop_mult=1.0,
                 max_holding_bars=20, fee_pct_per_side=FEE_PCT_PER_SIDE):
    """Тот же алгоритм, что в paper_trader: вход по сигналу, выход цель/стоп/таймаут.
    Отличие от bb_backtest.py только в ограниченном окне индикаторов."""
    closes = [c["close"] for c in candles]
    n = len(candles)
    i = period
    trades = []

    while i < n:
        window = candles[max(0, i + 1 - LOOKBACK):i + 1]
        d = decide(window, period, num_std)
        if not d.take_trade or d.side == Side.SHORT:   # спот: только LONG
            i += 1
            continue

        entry = closes[i]
        w_closes = [c["close"] for c in window]
        basis, upper, lower = bollinger_bands(w_closes, period, num_std)
        if basis is None:
            i += 1
            continue
        target = basis
        stop = entry - stop_mult * (upper - lower)

        exit_price, exit_i, reason = None, None, "таймаут"
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            p = closes[j]
            if p >= target:
                exit_price, exit_i, reason = p, j, "цель"
                break
            if p <= stop:
                exit_price, exit_i, reason = p, j, "стоп"
                break
        if exit_price is None:
            exit_i = min(i + max_holding_bars, n - 1)
            exit_price = closes[exit_i]

        pnl = (exit_price - entry) / entry * 100 - 2 * fee_pct_per_side
        trades.append((pnl, reason))
        i = exit_i + 1

    return trades


def load_all_candles():
    data = {}
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            candles = json.load(f)
        split = int(len(candles) * TRAIN_FRAC)
        data[symbol] = (candles[:split], candles[split:])
    return data


def evaluate(data, part, cfg):
    stop_mult, max_bars = cfg
    idx = 0 if part == "train" else 1
    per_symbol, returns = {}, []
    n_reliable = n_positive = 0
    for symbol, parts in data.items():
        trades = run_backtest(parts[idx], stop_mult=stop_mult, max_holding_bars=max_bars)
        total = sum(t[0] for t in trades)
        per_symbol[symbol] = (len(trades), total)
        if len(trades) >= MIN_TRADES:
            n_reliable += 1
            returns.append(total)
            if total > 0:
                n_positive += 1
    return {
        "cfg": cfg, "n_reliable": n_reliable, "n_positive": n_positive,
        "breadth": (n_positive / n_reliable * 100) if n_reliable else 0.0,
        "median": sorted(returns)[len(returns) // 2] if returns else 0.0,
        "mean": sum(returns) / len(returns) if returns else 0.0,
        "per_symbol": per_symbol,
    }


def main():
    data = load_all_candles()
    if not data:
        print("Нет данных в data/ — сначала выкачай историю")
        return
    print(f"Монет загружено: {len(data)}, таймфрейм 1H\n")

    results = []
    for sm in STOP_MULT_VARIANTS:
        for mb in MAX_BARS_VARIANTS:
            r = evaluate(data, "train", (sm, mb))
            results.append(r)
            print(f"  train стоп×{sm} таймаут={mb:2d} баров: широта={r['breadth']:5.1f}%  "
                  f"медиана={r['median']:+7.2f}%  среднее={r['mean']:+7.2f}%  монет={r['n_reliable']}")

    ranked = [r for r in results if r["n_reliable"] >= MIN_COVERAGE]
    if not ranked:
        print("\nНи одна конфигурация не набрала покрытия")
        return
    ranked.sort(key=lambda r: (-r["breadth"], -r["median"]))
    best = ranked[0]

    lines = ["=" * 88,
             "MEAN-REVERSION v2 — ПОИСК ПО TRAIN (таймфрейм 1H, только LONG, комиссия 0.2% за круг)",
             "=" * 88]
    header = f"{'Конфигурация':<32}{'Широта%':>10}{'Медиана%':>12}{'Среднее%':>12}{'Монет':>8}"
    lines += [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: -r["breadth"]):
        sm, mb = r["cfg"]
        mark = "  <- контроль (как в v1)" if r["cfg"] == CONTROL else ""
        lines.append(f"{'стоп×' + str(sm) + ', таймаут ' + str(mb):<32}{r['breadth']:>10.1f}"
                     f"{r['median']:>12.2f}{r['mean']:>12.2f}{r['n_reliable']:>8}{mark}")

    lines.append("")
    lines.append(f"Лучшая по TRAIN: стоп×{best['cfg'][0]}, таймаут {best['cfg'][1]} — "
                 f"{best['n_positive']}/{best['n_reliable']} монет в плюсе ({best['breadth']:.1f}%)")

    lines += ["", "=" * 88, "TEST (без дальнейшей подгонки)", "=" * 88]
    t_best = evaluate(data, "test", best["cfg"])
    t_ctrl = evaluate(data, "test", CONTROL)
    lines.append(f"{'Символ':<14}{'Сделок':>8}{'Итог% (v2)':>13}{'Итог% (контроль)':>19}")
    lines.append("-" * 54)
    for s in sorted(t_best["per_symbol"], key=lambda s: -t_best["per_symbol"][s][1]):
        n, tot = t_best["per_symbol"][s]
        cn, ctot = t_ctrl["per_symbol"].get(s, (0, 0.0))
        lines.append(f"{s:<14}{n:>8}{tot:>13.2f}{ctot:>19.2f}")
    lines.append("")
    lines.append(f"TEST победитель: {t_best['n_positive']}/{t_best['n_reliable']} в плюсе "
                 f"({t_best['breadth']:.1f}%), медиана {t_best['median']:+.2f}%, среднее {t_best['mean']:+.2f}%")
    lines.append(f"TEST контроль:   {t_ctrl['n_positive']}/{t_ctrl['n_reliable']} в плюсе "
                 f"({t_ctrl['breadth']:.1f}%), медиана {t_ctrl['median']:+.2f}%, среднее {t_ctrl['mean']:+.2f}%")
    lines.append("")
    lines.append("Честный вывод возможен только если победитель обходит контроль и на train, и на test.")
    lines.append("Если обе колонки в минусе — стратегия не проходит, и это тот же вывод, что дал")
    lines.append("исходный бэктест: отскок от полос не отбивает комиссию спота.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(os.path.dirname(__file__), "bb_config_search_v2_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
