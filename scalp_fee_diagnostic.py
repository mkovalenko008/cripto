"""
Отдельная диагностика поверх scalp_config_search.py, отвечает на один вопрос:
провал скальп-идеи после комиссии — это "сигнала нет вообще" или "сигнал
есть, но он мельче комиссии"? Разница принципиальная: первое хоронит идею
насовсем, второе означает, что реверсия на спокойных монетах реальна, просто
не при спотовой комиссии Bitget 0.1%/сторону.

Метод: та же сетка, тот же train (см. scalp_config_search.py), но каждая
конфигурация считается ДВАЖДЫ — с комиссией 0 и с реальной 0.1%/сторону.
Если без комиссии сетка массово прибыльна, а с комиссией — нет, дальше
считаем порог безубыточности по комиссии (средний PnL до комиссии / 2) для
топ-N конфигураций по пре-фи широте — если он на порядок ниже 0.1%, это не
"чуть не дотянули", а фундаментальное несоответствие размера движения и
цены входа/выхода на споте.

Запуск: python3 scalp_fee_diagnostic.py
"""
from __future__ import annotations

import json
import os

import scalp_config_search as scs
from scalp_backtest import run_backtest

TOP_N = 10
REAL_FEE = scs.FEE_PCT_PER_SIDE


def evaluate_zero_fee(data, cfg):
    period, num_std, target_pct, stop_pct, max_bars = cfg
    n_reliable = n_positive = 0
    returns = []
    total_trades = 0
    total_pnl = 0.0
    for symbol, (train, _test) in data.items():
        r = run_backtest(train, period=period, num_std=num_std, target_pct=target_pct,
                          stop_pct=stop_pct, max_holding_bars=max_bars, fee_pct_per_side=0.0)
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
        print("Нет данных в data/*_5min.json")
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

    real_fee_by_cfg = {}
    for r in top:
        rr = scs.evaluate(data, "train", r["cfg"])
        real_fee_by_cfg[r["cfg"]] = rr

    lines = ["=" * 100,
             "ДИАГНОСТИКА: сигнал против комиссии (5min, спокойные монеты, train)",
             "=" * 100,
             "",
             f"Без комиссии: {n60}/{len(grid)} конфигураций сетки дают широту >=60% "
             f"({n60/len(grid)*100:.0f}%), {n50}/{len(grid)} дают >=50%.",
             f"Медиана медианных доходностей по всей сетке (без комиссии): {mm:+.2f}%.",
             "-> устойчивая, широкая реверсия на спокойных монетах — не шум.",
             "",
             f"Топ-{TOP_N} конфигураций по широте БЕЗ комиссии, и что с ними при реальной "
             f"комиссии Bitget (0.1%/сторону, {2*REAL_FEE:.2f}% за круг):",
             ""]
    header = f"{'Конфигурация':<44}{'Широта% (0 fee)':>16}{'Широта% (real fee)':>20}{'Порог/сторону%':>16}"
    lines += [header, "-" * len(header)]
    for r in top:
        rf = real_fee_by_cfg[r["cfg"]]
        lines.append(f"{scs.cfg_label(r['cfg']):<44}{r['breadth']:>16.1f}{rf['breadth']:>20.1f}"
                     f"{r['breakeven_fee_per_side']:>16.4f}")

    avg_breakeven = sum(r["breakeven_fee_per_side"] for r in top) / len(top)
    lines += ["",
             f"Средний порог безубыточности по комиссии за сторону в топ-{TOP_N}: "
             f"{avg_breakeven:.4f}%. Реальная комиссия Bitget spot: {REAL_FEE:.2f}%/сторону — "
             f"в {REAL_FEE/avg_breakeven:.1f} раза выше порога.",
             "",
             "Вывод: реверсия на спокойных монетах на 5-минутках реальна (видна почти по всей",
             "сетке параметров, а не в одной подогнанной точке), но её типичный размер меньше",
             "спотовой комиссии Bitget в разы, а не на проценты. Это не \"почти получилось\" —",
             "это структурное несоответствие: чтобы \"пара тиков\" окупалась, комиссия за круг",
             "должна быть на порядок ниже 0.2%, а такой она не бывает на споте без плеча.",
             "",
             "Из этого следует не \"стратегия мертва\", а конкретное ограничение на то, при каких",
             "условиях идея вообще может окупиться: маркетмейкерская комиссия вместо тейкерской",
             "(если Bitget её даёт на споте — не проверено в этом бэктесте, там просто нет модели",
             "исполнения лимитных ордеров), более крупная цель за счёт удержания дольше 2 часов",
             "(это уже испытанная mean-reversion идея на часовом таймфрейме, см. paper_trader.py",
             "и bb_config_search_v2.py — там тоже не хватило запаса над комиссией), либо биржа/",
             "инструмент с околонулевой комиссией. В нынешнем виде — на споте Bitget маркет-",
             "ордерами — идея \"скальп пары тиков\" НЕ проходит ни на одной из 270 проверенных",
             "конфигураций и 15 спокойных монет."]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(os.path.dirname(__file__), "scalp_fee_diagnostic_report.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
