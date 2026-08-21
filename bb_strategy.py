"""
v2: то же самое ядро идеи (касание полосы -> ставка на отскок к середине),
но с доработками, которые реально используют квант-трейдеры (см. разбор в
чате): ADX вместо наклона SMA как фильтр режима рынка, RSI как подтверждение
силы сигнала, %b вместо прямого сравнения цены с полосой.

- ADX >= adx_threshold -> рынок трендовый ("хождение по полосе") -> пропуск.
- Подтверждение RSI: касание нижней полосы должно сопровождаться
  перепроданностью по RSI, а не просто самим касанием — паттерн "двойное
  подтверждение", который в источниках даёт win rate заметно выше, чем
  голый touch.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from indicators import bollinger_bands, percent_b, adx, rsi


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
    rsi_value: float | None = None


def decide(candles: list[dict], period: int = 20, num_std: float = 2.0,
           use_adx_filter: bool = True, adx_period: int = 14, adx_threshold: float = 25.0,
           use_rsi_confirmation: bool = True, rsi_period: int = 14,
           rsi_oversold: float = 35.0, rsi_overbought: float = 65.0) -> Decision:
    """
    candles: список {"open","high","low","close"} от старой к новой (как
    возвращает bitget_client.get_candles). Решение принимается по последней
    свече в списке.
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    basis, upper, lower = bollinger_bands(closes, period, num_std)
    if basis is None:
        return Decision(take_trade=False, reason="недостаточно данных для полос")

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
        adx_val = adx(highs, lows, closes, adx_period)
        if adx_val is not None and adx_val >= adx_threshold:
            return Decision(take_trade=False,
                             reason=f"ADX={adx_val:.1f} >= {adx_threshold} — сильный тренд, "
                                    f"пропускаем (это и есть 'хождение по полосе')",
                             percent_b=pb, adx_value=adx_val)

    rsi_val = None
    if use_rsi_confirmation:
        rsi_val = rsi(closes, rsi_period)
        if rsi_val is not None:
            if side == Side.LONG and rsi_val > rsi_oversold:
                return Decision(take_trade=False,
                                 reason=f"RSI={rsi_val:.1f} не подтверждает перепроданность "
                                        f"(нужно <= {rsi_oversold})",
                                 percent_b=pb, adx_value=adx_val, rsi_value=rsi_val)
            if side == Side.SHORT and rsi_val < rsi_overbought:
                return Decision(take_trade=False,
                                 reason=f"RSI={rsi_val:.1f} не подтверждает перекупленность "
                                        f"(нужно >= {rsi_overbought})",
                                 percent_b=pb, adx_value=adx_val, rsi_value=rsi_val)

    return Decision(take_trade=True, side=side,
                     reason=f"%b={pb:.2f}, ADX и RSI подтверждают вход",
                     percent_b=pb, adx_value=adx_val, rsi_value=rsi_val)
