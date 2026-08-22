"""
Трендовая (breakout) стратегия — по духу противоположность bb_strategy.py.
Там ставка была на отскок ОТ полосы обратно к середине, здесь — ставка на
продолжение движения ПОСЛЕ пробоя полосы: цена ушла за границу диапазона —
значит, начинается тренд, и мы в нём остаёмся, пока он не развернётся.

ADX используется НАОБОРОТ по сравнению с bb_strategy: там высокий ADX был
поводом пропустить сделку ("сильный тренд — не наш случай для отскока"),
здесь высокий ADX — обязательное условие входа ("нужен настоящий тренд, а
не шум/боковик").

Выход — не тесная цель у средней линии (как в bb_strategy), а трейлинг-стоп
по ATR, который подтягивается за ценой и никогда не двигается против
прибыли. Так тренд можно "прокатать" значительно дальше 0.2% комиссии за
круг, а не только до середины полосы, как раньше.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from indicators import bollinger_bands, adx, atr


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Decision:
    take_trade: bool
    side: Side | None = None
    reason: str = ""
    adx_value: float | None = None
    atr_value: float | None = None


def decide(candles: list[dict], period: int = 20, num_std: float = 2.0,
           adx_period: int = 14, adx_threshold: float = 25.0,
           atr_period: int = 14, lookback: int = 150) -> Decision:
    """
    candles: список {"open","high","low","close"} от старой к новой. Решение
    принимается по последней свече.

    lookback ограничивает окно для расчёта индикаторов последними N барами
    вместо всей истории целиком. Сглаживание Уайлдера сходится к тому же
    значению за ~100-150 баров независимо от точки старта окна — а на
    бэктесте с тысячами баров это превращает O(n^2) в O(n).
    """
    window = candles[-lookback:] if len(candles) > lookback else candles
    closes = [c["close"] for c in window]
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]

    basis, upper, lower = bollinger_bands(closes, period, num_std)
    if basis is None:
        return Decision(take_trade=False, reason="недостаточно данных для полос")

    adx_val = adx(highs, lows, closes, adx_period)
    if adx_val is None:
        return Decision(take_trade=False, reason="недостаточно данных для ADX")

    atr_val = atr(highs, lows, closes, atr_period)
    price = closes[-1]

    if adx_val < adx_threshold:
        return Decision(take_trade=False,
                         reason=f"ADX={adx_val:.1f} < {adx_threshold} — тренда нет, пропускаем",
                         adx_value=adx_val, atr_value=atr_val)

    if price > upper:
        return Decision(take_trade=True, side=Side.LONG,
                         reason=f"пробой верхней полосы, ADX={adx_val:.1f} подтверждает тренд",
                         adx_value=adx_val, atr_value=atr_val)
    if price < lower:
        return Decision(take_trade=True, side=Side.SHORT,
                         reason=f"пробой нижней полосы, ADX={adx_val:.1f} подтверждает тренд",
                         adx_value=adx_val, atr_value=atr_val)

    return Decision(take_trade=False, reason="цена внутри полос, пробоя нет",
                     adx_value=adx_val, atr_value=atr_val)
