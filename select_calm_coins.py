"""
Ранжирует корзину монет по волатильности и отбирает "спокойный" хвост —
кандидатов для скальп-стратегии на пару тиков (см. scalp_strategy.py).

Метрика — медианный ATR% (ATR/цена закрытия * 100) на часовых свечах: медиана
устойчивее к одному выбросу (например, к делистинг-панике или разовому
пампу), чем среднее, а нам нужен именно "обычный" характер движения монеты,
а не хвостовые события.

Данные берутся из уже закэшированных data/{SYMBOL}_1H.json (см.
fetch_1h_basket.py). Монеты без кэша или с совсем короткой историей
пропускаются — по ним нельзя честно посчитать волатильность.

Запуск: python3 select_calm_coins.py [сколько монет отобрать, по умолчанию 15]
"""
from __future__ import annotations

import json
import os
import sys

from indicators import _true_range, _wilder_smooth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_CANDLES = 200   # меньше — история слишком короткая для честной оценки
ATR_PERIOD = 14


def load_candles(symbol: str) -> list[dict] | None:
    path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        candles = json.load(f)
    return candles if len(candles) >= MIN_CANDLES else None


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def atr_pct_series(candles: list[dict], period: int = ATR_PERIOD) -> list[float]:
    """ATR% на каждом баре, начиная с period-го — не одно число на весь ряд,
    а серия, чтобы медиана отражала типичный день, а не последнюю точку.
    Считает True Range один раз и сглаживает по Уайлдеру напрямую (как в
    indicators.atr), а не пересчитывает ATR заново на каждом баре — тот же
    результат за O(n) вместо O(n^2)."""
    closes = [c["close"] for c in candles]
    n = len(candles)
    if n < period + 1:
        return []
    trs = [_true_range(candles[i]["high"], candles[i]["low"], closes[i - 1])
           for i in range(1, n)]
    smooth = _wilder_smooth(trs, period)
    # smooth[k] — сглаженный ATR*period на баре index = period + k (тот же
    # сдвиг, что и в _wilder_smooth: первое значение — сумма первых period TR).
    out = []
    for k, s in enumerate(smooth):
        close_i = period + k
        if close_i >= n or not closes[close_i]:
            continue
        out.append((s / period) / closes[close_i] * 100)
    return out


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    with open(os.path.join(os.path.dirname(__file__), "basket_100.json")) as f:
        basket = json.load(f)

    rows = []
    skipped = []
    for sym in basket:
        candles = load_candles(sym)
        if candles is None:
            skipped.append(sym)
            continue
        series = atr_pct_series(candles)
        if not series:
            skipped.append(sym)
            continue
        rows.append((sym, median(series), len(candles)))

    rows.sort(key=lambda r: r[1])

    print(f"Оценено монет: {len(rows)} (пропущено — нет/мало данных: {len(skipped)})")
    if skipped:
        print(f"  пропущены: {', '.join(skipped)}")
    print(f"\nСамые СПОКОЙНЫЕ {top_n} по медианному ATR% (1H):")
    for sym, m, n in rows[:top_n]:
        print(f"  {sym:12s} медианный ATR% = {m:.3f}   ({n} свечей)")

    print(f"\nСамые ВОЛАТИЛЬНЫЕ 5 — для контраста:")
    for sym, m, n in rows[-5:]:
        print(f"  {sym:12s} медианный ATR% = {m:.3f}   ({n} свечей)")

    calm = [sym for sym, _, _ in rows[:top_n]]
    out_path = os.path.join(os.path.dirname(__file__), "calm_coins.json")
    with open(out_path, "w") as f:
        json.dump(calm, f, indent=1)
    print(f"\nСписок сохранён в {out_path}")


if __name__ == "__main__":
    main()
