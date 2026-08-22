"""
Расширенная сетка поверх grid_backtest.py: добавляет порог ADX (15/20/25) и
множитель стопа (0.5/1.0/1.5) как отдельные измерения поиска, помимо
num_std. Только 1H и 4H — на 15min уже исчерпывающе показали (см.
grid_backtest.py/backtest_report.txt), что комиссия там съедает всё, а
полный расчёт по этой сетке на 17к свечей занял бы ~35 минут впустую.

Подбор лучшей конфигурации — СТРОГО по train, как и раньше. Test смотрим
один раз для отобранной по train конфигурации, не для того чтобы дальше
докручивать по нему (иначе overfitting просто переезжает с train на test).
"""
import json
import os

from bb_backtest import run_backtest, BacktestResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TIMEFRAMES = ["1H", "4H"]
TRAIN_FRAC = 0.7
MIN_TRADES = 30

FILTER_VARIANTS = [
    ("v1: без фильтров", dict(use_adx_filter=False, use_rsi_confirmation=False)),
    ("v2a: ADX-фильтр", dict(use_adx_filter=True, use_rsi_confirmation=False)),
    ("v2b: ADX+RSI", dict(use_adx_filter=True, use_rsi_confirmation=True)),
    ("v2c: только RSI", dict(use_adx_filter=False, use_rsi_confirmation=True)),
]
NUM_STD_VARIANTS = [2.0, 2.5, 3.0]
ADX_THRESHOLD_VARIANTS = [15.0, 20.0, 25.0]
STOP_MULT_VARIANTS = [0.5, 1.0, 1.5]


def load_candles(label: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"ETHUSDT_{label}.json")
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
    for filt_label, filt_kwargs in FILTER_VARIANTS:
        uses_adx = filt_kwargs["use_adx_filter"]
        adx_variants = ADX_THRESHOLD_VARIANTS if uses_adx else [25.0]  # не используется, но нужно 1 значение для цикла
        for adx_threshold in adx_variants:
            for num_std in NUM_STD_VARIANTS:
                for stop_mult in STOP_MULT_VARIANTS:
                    result = run_backtest(candles, num_std=num_std, fee_pct_per_side=0.1,
                                           adx_threshold=adx_threshold, stop_mult=stop_mult,
                                           **filt_kwargs)
                    rows.append({
                        "filter": filt_label,
                        "adx_threshold": adx_threshold if uses_adx else None,
                        "num_std": num_std,
                        "stop_mult": stop_mult,
                        "trades": result.total_trades,
                        "win_rate": result.win_rate * 100,
                        "total_return_pct": result.total_return_pct,
                        "max_losing_streak": result.max_losing_streak,
                        "max_drawdown_pct": max_drawdown_pct(result),
                        "reliable": result.total_trades >= MIN_TRADES,
                    })
    return rows


def fmt_row(r: dict) -> str:
    adx_str = f"{r['adx_threshold']:.0f}" if r["adx_threshold"] is not None else "—"
    flag = "OK" if r["reliable"] else "МАЛО СДЕЛОК"
    return (f"{r['filter']:<20}{adx_str:>5}{r['num_std']:>8.1f}{r['stop_mult']:>8.1f}"
            f"{r['trades']:>8}{r['win_rate']:>10.1f}{r['total_return_pct']:>+10.2f}"
            f"{r['max_losing_streak']:>11}{r['max_drawdown_pct']:>10.2f}  {flag}")


def fmt_table(rows: list[dict]) -> str:
    header = (f"{'Фильтр':<20}{'ADX':>5}{'std':>8}{'stopX':>8}{'Сделок':>8}"
              f"{'WinRate%':>10}{'Итог%':>10}{'МаксСерия':>11}{'МаксDD%':>10}  Надёжность")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main():
    report_lines = []

    for tf in TIMEFRAMES:
        candles = load_candles(tf)
        train, test = split_train_test(candles, TRAIN_FRAC)

        report_lines.append(f"\n{'=' * 95}")
        report_lines.append(f"ТАЙМФРЕЙМ {tf}  (всего свечей: {len(candles)}, "
                             f"train: {len(train)}, test: {len(test)})")
        report_lines.append(f"Комбинаций в сетке: {len(FILTER_VARIANTS)} фильтра x "
                             f"{len(ADX_THRESHOLD_VARIANTS)} порога ADX (только там где применим) x "
                             f"{len(NUM_STD_VARIANTS)} num_std x {len(STOP_MULT_VARIANTS)} стоп")
        report_lines.append('=' * 95)

        train_rows = run_grid(train)
        test_rows = run_grid(test)

        # топ-10 по train (среди надёжных по числу сделок), сортировка по итогу
        reliable_train = [r for r in train_rows if r["reliable"]]
        pool = reliable_train if reliable_train else train_rows
        top10 = sorted(pool, key=lambda r: r["total_return_pct"], reverse=True)[:10]

        report_lines.append(f"\n-- ТОП-10 по TRAIN ({tf}), среди конфигураций с >= {MIN_TRADES} сделок --")
        report_lines.append(fmt_table(top10))

        report_lines.append(f"\n-- Те же конфигурации на TEST ({tf}) --")
        test_matched = []
        for best_train in top10:
            idx = train_rows.index(best_train)
            test_matched.append(test_rows[idx])
        report_lines.append(fmt_table(test_matched))

    text = "\n".join(report_lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "backtest_report_v2.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
