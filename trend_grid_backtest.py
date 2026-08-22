"""
Сетка бэктестов для trend_strategy.py на реальных данных из data/*.json.
Та же дисциплина: train/test по времени, подбор лучшей конфигурации только
по train, fee_pct_per_side=0.1 обязательно.

Запуск: python3 trend_grid_backtest.py [SYMBOL]
Без аргумента — ETHUSDT (поведение по умолчанию, как раньше).
"""
import json
import os
import sys

from trend_backtest import run_backtest, BacktestResult

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TIMEFRAMES = ["15min", "1H", "4H"]
TRAIN_FRAC = 0.7
MIN_TRADES = 30

ADX_THRESHOLD_VARIANTS = [20.0, 25.0, 30.0]
ATR_STOP_MULT_VARIANTS = [1.5, 2.0, 3.0]
NUM_STD_VARIANTS = [2.0, 2.5]


def load_candles(label: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{SYMBOL}_{label}.json")
    with open(path) as f:
        return json.load(f)


def split_train_test(candles: list[dict], train_frac: float):
    n = len(candles)
    split_i = int(n * train_frac)
    return candles[:split_i], candles[split_i:]


def max_drawdown_pct(result: BacktestResult) -> float:
    if not result.trades:
        return 0.0
    cum = 0.0
    peak = 0.0
    worst_dd = 0.0
    for t in result.trades:
        cum += t.pnl_pct
        peak = max(peak, cum)
        worst_dd = min(worst_dd, cum - peak)
    return worst_dd


def run_grid(candles: list[dict]) -> list[dict]:
    rows = []
    for adx_threshold in ADX_THRESHOLD_VARIANTS:
        for atr_stop_mult in ATR_STOP_MULT_VARIANTS:
            for num_std in NUM_STD_VARIANTS:
                result = run_backtest(candles, num_std=num_std, fee_pct_per_side=0.1,
                                       adx_threshold=adx_threshold, atr_stop_mult=atr_stop_mult)
                rows.append({
                    "adx_threshold": adx_threshold,
                    "atr_stop_mult": atr_stop_mult,
                    "num_std": num_std,
                    "trades": result.total_trades,
                    "win_rate": result.win_rate * 100,
                    "total_return_pct": result.total_return_pct,
                    "max_losing_streak": result.max_losing_streak,
                    "max_drawdown_pct": max_drawdown_pct(result),
                    "reliable": result.total_trades >= MIN_TRADES,
                })
    return rows


def fmt_table(rows: list[dict]) -> str:
    header = (f"{'ADX':>5}{'ATRx':>7}{'std':>6}{'Сделок':>8}"
              f"{'WinRate%':>10}{'Итог%':>10}{'МаксСерия':>11}{'МаксDD%':>10}  Надёжность")
    lines = [header, "-" * len(header)]
    for r in rows:
        flag = "OK" if r["reliable"] else "МАЛО СДЕЛОК"
        lines.append(
            f"{r['adx_threshold']:>5.0f}{r['atr_stop_mult']:>7.1f}{r['num_std']:>6.1f}"
            f"{r['trades']:>8}{r['win_rate']:>10.1f}{r['total_return_pct']:>+10.2f}"
            f"{r['max_losing_streak']:>11}{r['max_drawdown_pct']:>10.2f}  {flag}"
        )
    return "\n".join(lines)


def main():
    report_lines = []
    best_by_tf = {}

    for tf in TIMEFRAMES:
        candles = load_candles(tf)
        train, test = split_train_test(candles, TRAIN_FRAC)

        report_lines.append(f"\n{'=' * 90}")
        report_lines.append(f"ТАЙМФРЕЙМ {tf}  (всего свечей: {len(candles)}, "
                             f"train: {len(train)}, test: {len(test)})")
        report_lines.append('=' * 90)

        train_rows = run_grid(train)
        test_rows = run_grid(test)

        report_lines.append(f"\n-- TRAIN ({tf}) --")
        report_lines.append(fmt_table(train_rows))
        report_lines.append(f"\n-- TEST ({tf}) --")
        report_lines.append(fmt_table(test_rows))

        reliable_train = [r for r in train_rows if r["reliable"]]
        pool = reliable_train if reliable_train else train_rows
        best_train = max(pool, key=lambda r: r["total_return_pct"])
        idx = train_rows.index(best_train)
        matching_test = test_rows[idx]
        best_by_tf[tf] = (best_train, matching_test)

        report_lines.append(
            f"\nЛучшая по TRAIN для {tf}: ADX>={best_train['adx_threshold']:.0f}, "
            f"ATRx{best_train['atr_stop_mult']}, num_std={best_train['num_std']} "
            f"(train: {best_train['trades']} сделок, {best_train['total_return_pct']:+.2f}%)"
        )
        report_lines.append(
            f"Та же конфигурация на TEST: {matching_test['trades']} сделок, "
            f"winrate {matching_test['win_rate']:.1f}%, итог {matching_test['total_return_pct']:+.2f}%, "
            f"макс.просадка {matching_test['max_drawdown_pct']:.2f}%"
        )

    report_lines.append(f"\n{'=' * 90}")
    report_lines.append("СВОДКА: лучшая по train конфигурация -> результат на test")
    report_lines.append('=' * 90)
    for tf, (bt, mt) in best_by_tf.items():
        report_lines.append(
            f"{tf:<6} ADX>={bt['adx_threshold']:<3.0f} ATRx{bt['atr_stop_mult']:<4} std={bt['num_std']:<4} | "
            f"train: {bt['trades']:>4} сделок, {bt['total_return_pct']:+7.2f}% | "
            f"test: {mt['trades']:>4} сделок, {mt['total_return_pct']:+7.2f}%"
            f"{'  [МАЛО СДЕЛОК]' if not mt['reliable'] else ''}"
        )

    text = f"СИМВОЛ: {SYMBOL}\n" + "\n".join(report_lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), f"trend_backtest_report_{SYMBOL}.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
