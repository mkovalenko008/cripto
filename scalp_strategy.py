"""
Сигнал для скальп-идеи: "касание микро-полосы Боллинджера" на коротком окне
и мелком таймфрейме — то же ядро, что у bb_strategy (касание -> ставка на
отскок), но период короче на порядок (это и делает полосу "микро"), а выход
считается не здесь, а в scalp_backtest.py фиксированным % от цены входа
("пара тиков"), а не возвратом к базовой линии — в этом всё отличие от
bb_strategy, не в самом сигнале входа.

ADX-фильтр входа опционален и по умолчанию выключен: на споте монете и так
разрешён только LONG, а мы отбираем "спокойные" монеты заранее (см.
select_calm_coins.py) именно чтобы избежать затяжных трендов — фильтр по
силе тренда может оказаться избыточным ограничением здесь, поэтому это
явный параметр для config search, а не решение по умолчанию.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from indicators import bollinger_bands, percent_b, adx


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Decision:
    take_trade: bool
    side: Side | None = None
    reason: str = ""
    percent_b: float | None = None
    adx_value: float | None = None


def decide(candles: list[dict], period: int = 20, num_std: float = 2.0,
           use_adx_filter: bool = False, adx_period: int = 14,
           adx_threshold: float = 30.0) -> Decision:
    """candles: {"open","high","low","close"} от старой к новой. Решение — по
    последней свече."""
    closes = [c["close"] for c in candles]

    basis, upper, lower = bollinger_bands(closes, period, num_std)
    if basis is None:
        return Decision(take_trade=False, reason="недостаточно данных для полосы")

    price = closes[-1]
    pb = percent_b(price, upper, lower)

    if pb <= 0.0:
        side = Side.LONG
    elif pb >= 1.0:
        side = Side.SHORT
    else:
        return Decision(take_trade=False, reason=f"%b={pb:.2f}, касания нет", percent_b=pb)

    adx_val = None
    if use_adx_filter:
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        adx_val = adx(highs, lows, closes, adx_period)
        if adx_val is not None and adx_val >= adx_threshold:
            return Decision(take_trade=False,
                             reason=f"ADX={adx_val:.1f} >= {adx_threshold} — трендовый режим, пропуск",
                             percent_b=pb, adx_value=adx_val)

    return Decision(take_trade=True, side=side, reason=f"%b={pb:.2f}",
                     percent_b=pb, adx_value=adx_val)
