"""
Фьючерсный аналог scalp_fee_diagnostic.py: та же сетка на фьючерсных данных,
"без комиссии" здесь означает без комиссии И без funding (funding_period_bars
выставлен настолько большим, что ни одна сделка внутри max_holding_bars через
него не пройдёт) — чтобы порог безубыточности отражал чистую цену, как и в
спотовой версии, и сравнение спот/фьючерс было на одинаковых основаниях.

Запуск: python3 scalp_fut_fee_diagnostic.py
"""
from __future__ import annotations

import json
import os

import scalp_fut_config_search as scs
from scalp_fut_backtest import run_backtest

TOP_N = 10
REAL_FEE = scs.FEE_PCT_PER_SIDE
NO_FUNDING = 999_999   # funding_period_bars настолько большой, что не сработает ни разу


def evaluate_zero_fee(data, cfg):
    period, num_std, target_pct, stop_pct, max_bars = cfg
    n_reliable = n_positive = 0
    returns = []
    total_trades = 0
    total_pnl = 0.0
    for symbol, (train, _test) in data.items():
        r = run_backtest(train, period=period, num_std=num_std, target_pct=target_pct,
                          stop_pct=stop_pct, max_holding_bars=max_bars, fee_pct_per_side=0.0,
                          funding_period_bars=NO_FUNDING)
        total_trades += r.total_trades
        total_pnl += r.total_return_pct
        if r.total_trades >= scs.MIN_TRADES:
            n_reliable += 1
            returns.append(r.total_return_pct)
            if r.total_return_pct > 0:
                n_positive += 1
    breadth = n_positive / n_reliable * 100 if n_reliable else 0.0
    median = sorted(returns)[len(returns) // 2] if returns else 0.0
    avg_pnl_per_trade = total_pnl / total_trades if total_trades else 0.0
    return {"cfg": cfg, "breadth": breadth, "median": median,
            "n_reliable": n_reliable, "total_trades": total_trades,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "breakeven_fee_per_side": avg_pnl_per_trade / 2}


def main():
    with open(os.path.join(os.path.dirname(__file__), "calm_coins.json")) as f:
        symbols = json.load(f)
    data = scs.load_all_candles(symbols)
    if not data:
        print("Нет данных в data/*_FUT_5min.json")
        return

    grid = [(p, s, t, sp, mb)
            for p in scs.PERIOD_VARIANTS
            for s in scs.NUM_STD_VARIANTS
            for t in scs.TARGET_PCT_VARIANTS
            for sp in scs.STOP_PCT_VARIANTS
            for mb in scs.MAX_BARS_VARIANTS]

    zero_fee = [evaluate_zero_fee(data, cfg) for cfg in grid]

    n60 = sum(1 for r in zero_fee if r["breadth"] >= 60)
    n50 = sum(1 for r in zero_fee if r["breadth"] >= 50)
    medians_of_medians = sorted(r["median"] for r in zero_fee)
    mm = medians_of_medians[len(medians_of_medians) // 2]

    zero_fee.sort(key=lambda r: (-r["breadth"], -r["median"]))
    top = zero_fee[:TOP_N]

    real_fee_by_cfg = {r["cfg"]: scs.evaluate(data, "train", r["cfg"]) for r in top}

    lines = ["=" * 100,
             "ДИАГНОСТИКА (ФЬЮЧЕРС): сигнал против комиссии+funding (5min, спокойные монеты, train)",
             "=" * 100,
             "",
             f"Без комиссии и funding: {n60}/{len(grid)} конфигураций дают широту >=60% "
             f"({n60/len(grid)*100:.0f}%), {n50}/{len(grid)} дают >=50%.",
             f"Медиана медианных доходностей по всей сетке: {mm:+.2f}%.",
             "",
             f"Топ-{TOP_N} конфигураций по широте без комиссии/funding, и что с ними при реальной "
             f"комиссии фьючерса (0.06%/сторону, {2*REAL_FEE:.2f}% за круг, funding учтён):",
             ""]
    header = f"{'Конфигурация':<46}{'Широта% (0 fee)':>16}{'Широта% (real)':>16}{'Порог/сторону%':>16}"
    lines += [header, "-" * len(header)]
    for r in top:
        rf = real_fee_by_cfg[r["cfg"]]
        lines.append(f"{scs.cfg_label(r['cfg']):<46}{r['breadth']:>16.1f}{rf['breadth']:>16.1f}"
                     f"{r['breakeven_fee_per_side']:>16.4f}")

    avg_breakeven = sum(r["breakeven_fee_per_side"] for r in top) / len(top)
    lines += ["",
             f"Средний порог безубыточности по комиссии за сторону в топ-{TOP_N}: {avg_breakeven:.4f}%.",
             f"Реальная комиссия фьючерса Bitget: {REAL_FEE:.2f}%/сторону — "
             f"в {REAL_FEE/avg_breakeven:.1f} раза выше порога "
             f"(для сравнения на споте было в 4.4 раза при пороге 0.1%/сторону)."]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(os.path.dirname(__file__), "scalp_fut_fee_diagnostic_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
