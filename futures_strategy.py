"""
Трендовая стратегия для фьючерсов: тот же пробой полосы + подтверждение ADX,
что и в trend_strategy.py, но добавлены два принципиально новых фильтра,
которых не было ни в одном из предыдущих тестов на споте:

- Объём: пробой должен сопровождаться объёмом выше среднего — иначе это
  может быть шум/ложный прокол на тонком стакане, а не реальный интерес.
- Funding rate (только фьючерсы, на споте этого не существует): если ставка
  финансирования уже сильно перекошена в сторону сделки (например, сильно
  положительная перед входом в лонг), это значит рынок уже "перегружен"
  такими позициями — вход по тренду в этот момент статистически более
  рискован (риск сквиза/разворота при перегруженной стороне). Фильтр не
  даёт открывать сделку В ТУ ЖЕ сторону, что и перекошенный funding.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from indicators import bollinger_bands, adx, atr, sma


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
    funding_rate: float | None = None


def decide(candles: list[dict], period: int = 20, num_std: float = 2.5,
           adx_period: int = 14, adx_threshold: float = 25.0,
           atr_period: int = 14, lookback: int = 150,
           use_volume_filter: bool = True, volume_period: int = 20, volume_mult: float = 1.5,
           use_funding_filter: bool = True, funding_extreme: float = 0.0005) -> Decision:
    """
    candles: список {"open","high","low","close","volume","funding_rate"}
    от старой к новой. funding_rate — ставка, действовавшая на момент бара
    (см. fetch_futures_history.attach_funding).
    """
    window = candles[-lookback:] if len(candles) > lookback else candles
    closes = [c["close"] for c in window]
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]
    volumes = [c.get("volume", 0.0) for c in window]

    basis, upper, lower = bollinger_bands(closes, period, num_std)
    if basis is None:
        return Decision(take_trade=False, reason="недостаточно данных для полос")

    adx_val = adx(highs, lows, closes, adx_period)
    if adx_val is None:
        return Decision(take_trade=False, reason="недостаточно данных для ADX")

    atr_val = atr(highs, lows, closes, atr_period)
    price = closes[-1]
    funding_rate = window[-1].get("funding_rate", 0.0)

    if adx_val < adx_threshold:
        return Decision(take_trade=False,
                         reason=f"ADX={adx_val:.1f} < {adx_threshold} — тренда нет",
                         adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)

    if price > upper:
        side = Side.LONG
    elif price < lower:
        side = Side.SHORT
    else:
        return Decision(take_trade=False, reason="цена внутри полос, пробоя нет",
                         adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)

    if use_volume_filter:
        avg_vol = sma(volumes, volume_period)
        if avg_vol is None or volumes[-1] < volume_mult * avg_vol:
            return Decision(take_trade=False,
                             reason=f"объём не подтверждает пробой (нужно >= {volume_mult}x среднего)",
                             adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)

    if use_funding_filter:
        if side == Side.LONG and funding_rate > funding_extreme:
            return Decision(take_trade=False,
                             reason=f"funding={funding_rate:.4%} уже перекошен в лонг — пропускаем",
                             adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)
        if side == Side.SHORT and funding_rate < -funding_extreme:
            return Decision(take_trade=False,
                             reason=f"funding={funding_rate:.4%} уже перекошен в шорт — пропускаем",
                             adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)

    return Decision(take_trade=True, side=side,
                     reason=f"пробой + ADX={adx_val:.1f} + объём подтверждён + funding не перекошен",
                     adx_value=adx_val, atr_value=atr_val, funding_rate=funding_rate)
