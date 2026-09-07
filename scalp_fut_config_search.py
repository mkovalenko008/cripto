"""
То же самое, что scalp_config_search.py, но на фьючерсных данных
(scalp_fut_backtest.py: комиссия 0.06%/сторону вместо 0.1%, шорт разрешён,
учтён funding). Сетка и дисциплина train/test/control идентичны спотовому
поиску — сравнение спот vs фьючерс должно быть честным, не "две разные
методики дали разные числа".

CONTROL здесь ниже, чем на споте (0.08% вместо 0.15%): круглая комиссия
фьючерса — 0.12%, и старый спотовый контроль 0.15% уже был бы выше неё,
то есть больше не гарантированно убыточен по построению. 0.08% меньше
0.12% с запасом — тот же смысл контроля, что и раньше.
"""
from __future__ import annotations

import json
import os

from scalp_fut_backtest import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FRAC = 0.7
MIN_TRADES = 20
MIN_COVERAGE = 6
FEE_PCT_PER_SIDE = 0.06

PERIOD_VARIANTS = [10, 20, 40]
NUM_STD_VARIANTS = [1.5, 2.0, 2.5]
TARGET_PCT_VARIANTS = [0.08, 0.15, 0.25, 0.4, 0.6, 0.9]
STOP_PCT_VARIANTS = [0.4, 0.6, 0.9]
MAX_BARS_VARIANTS = [12, 24]
CONTROL = (20, 2.0, 0.08, 0.6, 12)   # цель 0.08% < комиссии за круг (0.12%) — заведомо убыточный ориентир


def load_all_candles(symbols: list[str]):
    data = {}
    for symbol in symbols:
        path = os.path.join(DATA_DIR, f"{symbol}_FUT_5min.json")
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
        "per_symbol": per_symbol,
    }


def cfg_label(cfg):
    period, num_std, target_pct, stop_pct, max_bars = cfg
    return f"пер={period} std={num_std} цель={target_pct}% стоп={stop_pct}% барm={max_bars}"


def main():
    with open(os.path.join(os.path.dirname(__file__), "calm_coins.json")) as f:
        symbols = json.load(f)

    data = load_all_candles(symbols)
    if not data:
        print("Нет данных в data/*_FUT_5min.json — сначала fetch_scalp_futures_basket.py")
        return
    print(f"Спокойных монет с фьючерсными данными: {len(data)} из {len(symbols)}, таймфрейм 5min\n")

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
        print("Ни одна конфигурация не набрала покрытия (мало сделок/монет)")
        return
    ranked.sort(key=lambda r: (-r["breadth"], -r["median"]))
    best = ranked[0]
    ranked_by_breadth_desc = sorted(results, key=lambda r: (-r["breadth"], -r["median"]))

    lines = ["=" * 96,
             "SCALP FUTURES — поиск по TRAIN (5min, LONG+SHORT, комиссия 0.12% за круг, funding учтён)",
             "=" * 96]
    header = f"{'Конфигурация':<46}{'Широта%':>10}{'Медиана%':>12}{'Среднее%':>12}{'Монет':>8}"
    lines += [header, "-" * len(header)]
    for r in ranked_by_breadth_desc[:25]:
        mark = "  <- контроль" if r["cfg"] == CONTROL else ""
        lines.append(f"{cfg_label(r['cfg']):<46}{r['breadth']:>10.1f}"
                     f"{r['median']:>12.2f}{r['mean']:>12.2f}{r['n_reliable']:>8}{mark}")
    lines.append(f"... показаны топ-25 из {len(results)} конфигураций")

    lines.append("")
    lines.append(f"Лучшая по TRAIN: {cfg_label(best['cfg'])} — "
                 f"{best['n_positive']}/{best['n_reliable']} монет в плюсе ({best['breadth']:.1f}%)")

    lines += ["", "=" * 96, "TEST (без дальнейшей подгонки)", "=" * 96]
    t_best = evaluate(data, "test", best["cfg"])
    t_ctrl = evaluate(data, "test", CONTROL)
    lines.append(f"{'Символ':<14}{'Сделок':>8}{'WinRate%':>10}{'Итог% (best)':>14}{'Итог% (контроль)':>19}")
    lines.append("-" * 65)
    for s in sorted(t_best["per_symbol"], key=lambda s: -t_best["per_symbol"][s][1]):
        n, tot, wr = t_best["per_symbol"][s]
        cn, ctot, cwr = t_ctrl["per_symbol"].get(s, (0, 0.0, 0.0))
        lines.append(f"{s:<14}{n:>8}{wr*100:>9.1f}%{tot:>14.2f}{ctot:>19.2f}")
    lines.append("")
    lines.append(f"TEST best:    {t_best['n_positive']}/{t_best['n_reliable']} в плюсе "
                 f"({t_best['breadth']:.1f}%), медиана {t_best['median']:+.2f}%, среднее {t_best['mean']:+.2f}%")
    lines.append(f"TEST контроль: {t_ctrl['n_positive']}/{t_ctrl['n_reliable']} в плюсе "
                 f"({t_ctrl['breadth']:.1f}%), медиана {t_ctrl['median']:+.2f}%, среднее {t_ctrl['mean']:+.2f}%")
    lines.append("")
    lines.append(f"Цель лучшей конфигурации: {best['cfg'][2]}%. Комиссия за круг: "
                 f"{2*FEE_PCT_PER_SIDE:.2f}%. Отношение цель/комиссия: "
                 f"{best['cfg'][2] / (2*FEE_PCT_PER_SIDE):.2f}x.")
    lines.append("")
    lines.append("Честный вывод возможен только если победитель обходит контроль и на train, и на test,")
    lines.append("и удерживает широту/медиану, сопоставимые с train.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(os.path.dirname(__file__), "scalp_fut_config_search_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
