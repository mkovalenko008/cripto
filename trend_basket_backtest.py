"""
Проверка одной ЗАФИКСИРОВАННОЙ (не подобранной под конкретную монету)
конфигурации trend_strategy на широкой корзине монет — 1H, 180 дней,
комиссия 0.1%. Параметры выбраны ДО просмотра результатов, по стандартным
учебным значениям (ADX>=25 — классический порог Уайлдера, ATRx2.0 —
типичный трейлинг-стоп, num_std=2.5 — средняя точка того, что тестировали
раньше), а не как "победитель" grid-поиска по какой-то одной монете.

Цель: если конфигурация даёт устойчивый плюс на большинстве монет —
это структура, а не подгонка под конкретный исторический путь одной монеты.
Если плюс только там, где мы её уже видели (HYPE) — это, скорее всего,
совпадение с одним конкретным трендом.

Для каждой монеты также считаем train/test (первая половина / вторая
половина периода) — если знак результата не совпадает между половинами,
это дополнительный сигнал нестабильности.
"""
import json
import os

from trend_backtest import run_backtest, BacktestResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_TRADES = 30

# Зафиксированная конфигурация — одна на все монеты, не подбирается заново.
CONFIG = dict(period=20, num_std=2.5, adx_period=14, adx_threshold=25.0,
              atr_period=14, atr_stop_mult=2.0, max_holding_bars=100,
              fee_pct_per_side=0.1)

SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "HYPEUSDT",
    "BTCUSDT", "XRPUSDT", "DOGEUSDT", "ZECUSDT", "SUIUSDT", "PEPEUSDT",
    "ENAUSDT", "ONDOUSDT", "TRUMPUSDT", "LINKUSDT", "BNBUSDT", "UNIUSDT",
    "ADAUSDT", "LTCUSDT", "NEARUSDT", "XLMUSDT", "AAVEUSDT", "AVAXUSDT",
    "TRXUSDT", "BCHUSDT",
]


def load_candles(symbol: str):
    path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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


def stats(result: BacktestResult) -> dict:
    return {
        "trades": result.total_trades,
        "win_rate": result.win_rate * 100,
        "total_return_pct": result.total_return_pct,
        "max_dd": max_drawdown_pct(result),
        "reliable": result.total_trades >= MIN_TRADES,
    }


def main():
    rows = []
    for symbol in SYMBOLS:
        candles = load_candles(symbol)
        if not candles:
            print(f"{symbol}: нет данных, пропуск")
            continue

        split_i = int(len(candles) * 0.5)
        first_half, second_half = candles[:split_i], candles[split_i:]

        full = stats(run_backtest(candles, **CONFIG))
        h1 = stats(run_backtest(first_half, **CONFIG))
        h2 = stats(run_backtest(second_half, **CONFIG))

        same_sign = (h1["total_return_pct"] >= 0) == (h2["total_return_pct"] >= 0)

        rows.append({"symbol": symbol, "full": full, "h1": h1, "h2": h2, "same_sign": same_sign})

    header = (f"{'Символ':<12}{'Сделок':>8}{'WinRate%':>10}{'Итог%':>10}{'МаксDD%':>10}"
              f"  Надёжн.  {'1яполовина%':>12}{'2яполовина%':>12}  Совпадает?")
    lines = [header, "-" * len(header)]
    for r in rows:
        f = r["full"]
        flag = "OK" if f["reliable"] else "МАЛО"
        lines.append(
            f"{r['symbol']:<12}{f['trades']:>8}{f['win_rate']:>10.1f}{f['total_return_pct']:>+10.2f}"
            f"{f['max_dd']:>10.2f}  {flag:<7}  {r['h1']['total_return_pct']:>+12.2f}"
            f"{r['h2']['total_return_pct']:>+12.2f}  {'да' if r['same_sign'] else 'НЕТ'}"
        )

    reliable_rows = [r for r in rows if r["full"]["reliable"]]
    positive = [r for r in reliable_rows if r["full"]["total_return_pct"] > 0]
    consistent_positive = [r for r in positive if r["same_sign"]]

    summary = [
        "",
        "=" * len(header),
        f"Всего монет проверено: {len(rows)}",
        f"С надёжной выборкой (>= {MIN_TRADES} сделок): {len(reliable_rows)}",
        f"Из них в плюсе за весь период: {len(positive)} ({len(positive) / len(reliable_rows) * 100:.0f}%)"
        if reliable_rows else "Из них в плюсе: н/д",
        f"Из них в плюсе И с совпадающим знаком на обеих половинах периода: {len(consistent_positive)}",
    ]
    if reliable_rows:
        avg_return = sum(r["full"]["total_return_pct"] for r in reliable_rows) / len(reliable_rows)
        median_return = sorted(r["full"]["total_return_pct"] for r in reliable_rows)[len(reliable_rows) // 2]
        summary.append(f"Средний результат по надёжным монетам: {avg_return:+.2f}%")
        summary.append(f"Медианный результат по надёжным монетам: {median_return:+.2f}%")

    text = "\n".join(lines + summary)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "trend_basket_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
