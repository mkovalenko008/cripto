"""
Кросс-секционный моментум: каждые rebalance_bars часов ранжируем ВСЕ монеты
корзины по доходности за последние lookback_bars часов, держим топ-K
(равными долями) до следующей ребалансировки. Не про "эта монета имеет
сигнал", а про "какие монеты СЕЙЧАС сильнее остальных" — принципиально
другой, отдельно задокументированный в литературе класс стратегий (Jegadeesh
& Titman и десятки последующих работ, включая крипто-специфичные).

Комиссия: на каждой ребалансировке считаем ПОЛНОЕ перевложение (продать всё,
купить новый топ-K) — это консервативная (переоценивающая издержки) модель,
в реальности часть позиций может остаться, но так честнее.

Для контроля считаем также худший-K (боттом) и равновзвешенный бенчмарк по
всей корзине — если топ-K не обгоняет боттом-K, моментума в данных нет.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEE_PCT_PER_SIDE = 0.1  # спот Bitget taker

SYMBOLS = ["ETHUSDT", "SOLUSDT", "HYPEUSDT", "BTCUSDT", "XRPUSDT", "DOGEUSDT",
           "SUIUSDT", "PEPEUSDT", "ENAUSDT", "ONDOUSDT",
           "LINKUSDT", "BNBUSDT", "UNIUSDT", "ADAUSDT", "LTCUSDT", "NEARUSDT",
           "XLMUSDT", "AAVEUSDT", "AVAXUSDT", "TRXUSDT", "BCHUSDT"]
# ZECUSDT и TRUMPUSDT исключены — слишком молодые (полгода-год истории),
# резали бы общее пересечение времени для всей корзины до ~7-8 месяцев.


def load_all():
    series = {}
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            candles = json.load(f)
        series[symbol] = {c["ts"]: c["close"] for c in candles}

    common_ts = None
    for prices in series.values():
        ts_set = set(prices.keys())
        common_ts = ts_set if common_ts is None else (common_ts & ts_set)
    common_ts = sorted(common_ts)

    aligned = {symbol: [prices[ts] for ts in common_ts] for symbol, prices in series.items()}
    return aligned, common_ts


def run_rotation(aligned: dict, start: int, end: int, lookback_bars: int, rebalance_bars: int,
                  top_k: int, direction: str = "top") -> float:
    """direction: 'top' (моментум) или 'bottom' (антимоментум, для контроля)."""
    symbols = list(aligned.keys())
    equity = 100.0
    t = start + lookback_bars
    while t + rebalance_bars <= end:
        returns = {}
        for s in symbols:
            p0, p1 = aligned[s][t - lookback_bars], aligned[s][t]
            if p0 > 0:
                returns[s] = (p1 - p0) / p0
        ranked = sorted(returns.items(), key=lambda kv: kv[1], reverse=(direction == "top"))
        selected = [s for s, _ in ranked[:top_k]]

        period_returns = []
        for s in selected:
            p0, p1 = aligned[s][t], aligned[s][t + rebalance_bars]
            period_returns.append((p1 - p0) / p0)
        avg_return = sum(period_returns) / len(period_returns) if period_returns else 0.0

        equity *= (1 + avg_return)
        equity *= (1 - 2 * FEE_PCT_PER_SIDE / 100)  # полное перевложение каждый период
        t += rebalance_bars

    return equity


def run_benchmark(aligned: dict, start: int, end: int) -> float:
    """Равновзвешенный buy&hold по всей корзине, без ребалансировок/комиссий после входа."""
    symbols = list(aligned.keys())
    rets = []
    for s in symbols:
        p0, p1 = aligned[s][start], aligned[s][end]
        rets.append((p1 - p0) / p0)
    avg = sum(rets) / len(rets)
    return 100.0 * (1 + avg) * (1 - 2 * FEE_PCT_PER_SIDE / 100)


def main():
    aligned, common_ts = load_all()
    n = len(common_ts)
    print(f"Монет: {len(aligned)}, общих баров (1H): {n}\n")

    split = int(n * 0.7)

    # Диапазон по академическим работам (crypto cross-sectional momentum):
    # формирование 2-4 недели, ребалансировка раз в неделю — не дневная.
    lookback_variants = [336, 504, 720]   # 14, 21, 30 дней
    rebalance_variants = [168]             # раз в неделю
    top_k_variants = [3, 5]

    print(f"{'lookback':>9}{'rebal':>7}{'topK':>6}{'TRAIN(top)':>12}{'TRAIN(bottom)':>15}{'TRAIN(bench)':>14}")
    print("-" * 65)
    results = []
    for lb in lookback_variants:
        for rb in rebalance_variants:
            for k in top_k_variants:
                eq_top = run_rotation(aligned, 0, split, lb, rb, k, "top")
                eq_bottom = run_rotation(aligned, 0, split, lb, rb, k, "bottom")
                results.append((lb, rb, k, eq_top, eq_bottom))
                print(f"{lb:>9}{rb:>7}{k:>6}{eq_top - 100:>+11.2f}%{eq_bottom - 100:>+14.2f}%")

    bench_train = run_benchmark(aligned, 0, split)
    print(f"\nБенчмарк (равновзвешенно всё, без ротации), TRAIN: {bench_train - 100:+.2f}%")

    # лучшая по TRAIN конфигурация топ-K (не боттом, не бенчмарк)
    best = max(results, key=lambda r: r[3])
    lb, rb, k, eq_top_train, eq_bottom_train = best
    print(f"\nЛучшая по TRAIN (top-K): lookback={lb}ч, rebalance={rb}ч, K={k} -> {eq_top_train - 100:+.2f}%")
    print(f"(тот же период, bottom-K для сравнения: {eq_bottom_train - 100:+.2f}%)")

    eq_top_test = run_rotation(aligned, split, n - 1, lb, rb, k, "top")
    eq_bottom_test = run_rotation(aligned, split, n - 1, lb, rb, k, "bottom")
    bench_test = run_benchmark(aligned, split, n - 1)
    print(f"\nТА ЖЕ КОНФИГУРАЦИЯ НА TEST: top-K={eq_top_test - 100:+.2f}%, "
          f"bottom-K={eq_bottom_test - 100:+.2f}%, бенчмарк={bench_test - 100:+.2f}%")


if __name__ == "__main__":
    main()
