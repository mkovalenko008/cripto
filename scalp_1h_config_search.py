"""
Тот же движок (scalp_backtest.py — фиксированная цель/стоп, high/low каждого
следующего бара, консервативный тай-брейк), но на часовых свечах и на ВСЕЙ
корзине из 99 монет (не только 15 "спокойных" — на масштабе часа отбор по
ATR% часовых баров менее осмыслен: то, что было "спокойным" по внутричасовой
волатильности, не обязательно даёт лучшую часовую реверсию).

Цели/стопы пересчитаны под масштаб часа: у "спокойных" монет медианный
часовой ATR% был 0.3-0.7% (см. select_calm_coins.py), у волатильных — 1-2%,
поэтому цели 0.15-0.9%, откалиброванные под 5-минутный шум, здесь бы почти
всегда били таймаут. Взяты цели 0.5-4%, соразмерные часовому движению.

Данные уже есть в кэше (data/*_1H.json, 45 дней, см. select_calm_coins.py) —
новая закачка не нужна.

Дисциплина та же: сетка на train, победитель по train, test — один раз.
"""
from __future__ import annotations

import json
import os

from scalp_backtest import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FRAC = 0.7
MIN_TRADES = 15
MIN_COVERAGE = 20
FEE_PCT_PER_SIDE = 0.1

PERIOD_VARIANTS = [10, 20]
NUM_STD_VARIANTS = [1.5, 2.0, 2.5]
TARGET_PCT_VARIANTS = [0.5, 1.0, 1.5, 2.5, 4.0]
STOP_PCT_VARIANTS = [0.8, 1.5, 2.5]
MAX_BARS_VARIANTS = [6, 12, 24]
CONTROL = (20, 2.0, 0.5, 1.5, 12)


def load_all_candles(symbols: list[str]):
    data = {}
    for symbol in symbols:
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            candles = json.load(f)
        split = int(len(candles) * TRAIN_FRAC)
        data[symbol] = (candles[:split], candles[split:])
    return data


def evaluate(data, part, cfg):
    period, num_std, target_pct, stop_pct, max_bars = cfg
    idx = 0 if part == "train" else 1
    per_symbol, returns = {}, []
    n_reliable = n_positive = 0
    for symbol, parts in data.items():
        result = run_backtest(parts[idx], period=period, num_std=num_std,
                               target_pct=target_pct, stop_pct=stop_pct,
                               max_holding_bars=max_bars,
                               fee_pct_per_side=FEE_PCT_PER_SIDE)
        total = result.total_return_pct
        per_symbol[symbol] = (result.total_trades, total, result.win_rate)
        if result.total_trades >= MIN_TRADES:
            n_reliable += 1
            returns.append(total)
            if total > 0:
                n_positive += 1
    return {
        "cfg": cfg, "n_reliable": n_reliable, "n_positive": n_positive,
        "breadth": (n_positive / n_reliable * 100) if n_reliable else 0.0,
        "median": sorted(returns)[len(returns) // 2] if returns else 0.0,
        "mean": sum(returns) / len(returns) if returns else 0.0,
        "max": max(returns) if returns else 0.0,
        "per_symbol": per_symbol,
    }


def cfg_label(cfg):
    period, num_std, target_pct, stop_pct, max_bars = cfg
    return f"пер={period} std={num_std} цель={target_pct}% стоп={stop_pct}% барч={max_bars}"


def main():
    basket = json.load(open(os.path.join(os.path.dirname(__file__), "basket_100.json")))
    data = load_all_candles(basket)
    if not data:
        print("Нет данных в data/*_1H.json")
        return
    print(f"Монет с данными: {len(data)} из {len(basket)}, таймфрейм 1H\n")

    grid = [(p, s, t, sp, mb)
            for p in PERIOD_VARIANTS
            for s in NUM_STD_VARIANTS
            for t in TARGET_PCT_VARIANTS
            for sp in STOP_PCT_VARIANTS
            for mb in MAX_BARS_VARIANTS]
    print(f"Конфигураций в сетке: {len(grid)}\n")

    results = [evaluate(data, "train", cfg) for cfg in grid]
    ranked = [r for r in results if r["n_reliable"] >= MIN_COVERAGE]
    if not ranked:
        print("Ни одна конфигурация не набрала покрытия")
        return
    ranked.sort(key=lambda r: (-r["breadth"], -r["median"]))
    best = ranked[0]
    ranked_by_breadth_desc = sorted(results, key=lambda r: (-r["breadth"], -r["median"]))

    lines = ["=" * 100,
             "SCALP 1H, ВСЯ КОРЗИНА — поиск по TRAIN (только LONG на споте, комиссия 0.2% за круг)",
             "=" * 100]
    header = f"{'Конфигурация':<46}{'Широта%':>10}{'Медиана%':>12}{'Среднее%':>12}{'Макс%':>10}{'Монет':>8}"
    lines += [header, "-" * len(header)]
    for r in ranked_by_breadth_desc[:20]:
        mark = "  <- контроль" if r["cfg"] == CONTROL else ""
        lines.append(f"{cfg_label(r['cfg']):<46}{r['breadth']:>10.1f}{r['median']:>12.2f}"
                     f"{r['mean']:>12.2f}{r['max']:>10.2f}{r['n_reliable']:>8}{mark}")
    lines.append(f"... показаны топ-20 из {len(results)} конфигураций")

    lines.append("")
    lines.append(f"Лучшая по TRAIN: {cfg_label(best['cfg'])} — "
                 f"{best['n_positive']}/{best['n_reliable']} монет в плюсе ({best['breadth']:.1f}%)")

    lines += ["", "=" * 100, "TEST (без дальнейшей подгонки)", "=" * 100]
    t_best = evaluate(data, "test", best["cfg"])
    t_ctrl = evaluate(data, "test", CONTROL)
    lines.append(f"TEST best:    {t_best['n_positive']}/{t_best['n_reliable']} в плюсе "
                 f"({t_best['breadth']:.1f}%), медиана {t_best['median']:+.2f}%, "
                 f"среднее {t_best['mean']:+.2f}%, макс {t_best['max']:+.2f}%")
    lines.append(f"TEST контроль: {t_ctrl['n_positive']}/{t_ctrl['n_reliable']} в плюсе "
                 f"({t_ctrl['breadth']:.1f}%), медиана {t_ctrl['median']:+.2f}%, среднее {t_ctrl['mean']:+.2f}%")
    lines.append("")
    lines.append("Топ-10 монет на TEST по лучшей конфигурации:")
    for s in sorted(t_best["per_symbol"], key=lambda s: -t_best["per_symbol"][s][1])[:10]:
        n, tot, wr = t_best["per_symbol"][s]
        lines.append(f"  {s:<12} сделок={n:3d} winrate={wr*100:5.1f}% итог={tot:+7.2f}%")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(os.path.dirname(__file__), "scalp_1h_config_search_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
