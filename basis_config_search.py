"""
Та же дисциплина train/test/широта, что и в trend_config_search.py и
futures_config_search.py, применённая к basis_arb_backtest.py.

Важная оговорка (в отличие от предыдущих стратегий): здесь сделки держатся
днями-неделями, поэтому даже за 180 дней на монету выходит немного сделок
физически, особенно на test-половине (~54 дня). Число сделок будет
маленьким почти везде — это не баг метода, а следствие природы стратегии.
Порог "надёжности" здесь ниже, чем раньше, и результаты в любом случае
стоит читать как разведочные, а не статистически строгие.
"""
import os

from basis_arb_backtest import load_merged, run_backtest

MIN_TRADES = 5
MIN_COVERAGE = 6  # из скольки монет минимум должно набраться MIN_TRADES
TRAIN_FRAC = 0.7

ENTRY_VARIANTS = [0.00002, 0.00003, 0.00005]
EXIT_VARIANTS = [-0.00002, -0.00005, -0.0001]
MAX_HOLD_VARIANTS = [480, 720]

SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT", "DOGEUSDT",
           "LINKUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "ZECUSDT", "TRUMPUSDT"]


def load_all():
    data = {}
    for symbol in SYMBOLS:
        bars = load_merged(symbol)
        if not bars:
            continue
        split_i = int(len(bars) * TRAIN_FRAC)
        data[symbol] = (bars[:split_i], bars[split_i:])
    return data


def evaluate(data: dict, part: str, config: dict) -> dict:
    idx = 0 if part == "train" else 1
    per_symbol = {}
    for symbol, halves in data.items():
        r = run_backtest(halves[idx], **config)
        per_symbol[symbol] = {"trades": r.total_trades, "return_pct": r.total_return_pct,
                              "reliable": r.total_trades >= MIN_TRADES}
    reliable = [v for v in per_symbol.values() if v["reliable"]]
    positive = [v for v in reliable if v["return_pct"] > 0]
    breadth = len(positive) / len(reliable) * 100 if reliable else 0.0
    median_return = sorted(v["return_pct"] for v in reliable)[len(reliable) // 2] if reliable else 0.0
    avg_return = sum(v["return_pct"] for v in reliable) / len(reliable) if reliable else 0.0
    return {"per_symbol": per_symbol, "n_reliable": len(reliable), "n_positive": len(positive),
            "breadth_pct": breadth, "median_return": median_return, "avg_return": avg_return}


def main():
    data = load_all()
    print(f"Загружено монет: {len(data)}\n")

    grid = []
    for entry_th in ENTRY_VARIANTS:
        for exit_th in EXIT_VARIANTS:
            for max_hold in MAX_HOLD_VARIANTS:
                config = dict(entry_threshold=entry_th, exit_threshold=exit_th, max_holding_bars=max_hold)
                res = evaluate(data, "train", config)
                grid.append((config, res))

    grid.sort(key=lambda cr: (cr[1]["breadth_pct"], cr[1]["median_return"]), reverse=True)

    header = f"{'entry':>10}{'exit':>10}{'maxHold':>9}{'Монет':>7}{'ВПлюсе':>8}{'Широта%':>9}{'Медиана%':>10}{'Среднее%':>10}"
    lines = ["=" * 90, "ПОИСК ПО TRAIN (basis arb)", "=" * 90, header, "-" * len(header)]
    for config, res in grid:
        flag = "" if res["n_reliable"] >= MIN_COVERAGE else "  <- МАЛО ПОКРЫТИЕ"
        lines.append(
            f"{config['entry_threshold']:>10.5f}{config['exit_threshold']:>10.5f}{config['max_holding_bars']:>9}"
            f"{res['n_reliable']:>7}{res['n_positive']:>8}{res['breadth_pct']:>9.1f}"
            f"{res['median_return']:>10.2f}{res['avg_return']:>10.2f}{flag}"
        )

    eligible = [cr for cr in grid if cr[1]["n_reliable"] >= MIN_COVERAGE]
    if not eligible:
        lines.append(f"\nНи одна конфигурация не набрала покрытие >= {MIN_COVERAGE} монет.")
        print("\n".join(lines))
        return

    best_config, best_res = eligible[0]
    lines.append(f"\n(Из {len(grid)} конфигураций {len(eligible)} с покрытием >= {MIN_COVERAGE})")
    lines.append(f"\nЛучшая по TRAIN: entry={best_config['entry_threshold']}, exit={best_config['exit_threshold']}, "
                 f"maxHold={best_config['max_holding_bars']} — {best_res['n_positive']}/{best_res['n_reliable']} "
                 f"монет в плюсе ({best_res['breadth_pct']:.1f}%), медиана {best_res['median_return']:+.2f}%")

    test_res = evaluate(data, "test", best_config)
    lines.append("\n" + "=" * 90)
    lines.append("ТА ЖЕ КОНФИГУРАЦИЯ НА TEST")
    lines.append("=" * 90)
    lines.append(f"{'Символ':<12}{'Сделок':>8}{'Итог%':>10}  Надёжн.")
    lines.append("-" * 40)
    for symbol, v in sorted(test_res["per_symbol"].items(), key=lambda kv: kv[1]["return_pct"], reverse=True):
        flag = "OK" if v["reliable"] else "мало"
        lines.append(f"{symbol:<12}{v['trades']:>8}{v['return_pct']:>+10.2f}  {flag}")
    lines.append(f"\nНа TEST: {test_res['n_positive']}/{test_res['n_reliable']} монет в плюсе "
                 f"({test_res['breadth_pct']:.1f}%), медиана {test_res['median_return']:+.2f}%, "
                 f"среднее {test_res['avg_return']:+.2f}%")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "basis_config_search_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
