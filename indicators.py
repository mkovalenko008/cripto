"""Простые технические индикаторы без внешних зависимостей."""
from __future__ import annotations


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Классический RSI (Wilder). Возвращает None, если данных недостаточно."""
    if len(closes) <= period:
        return None

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def bollinger_bands(closes: list[float], period: int = 20, num_std: float = 2.0):
    """
    Возвращает (basis, upper, lower) для последней точки, как на твоём графике
    (BB 20 SMA close 2): basis — синяя SMA, upper — красная, lower — зелёная.
    None, если данных меньше period.
    """
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    basis = sum(window) / period
    variance = sum((c - basis) ** 2 for c in window) / period
    std = variance ** 0.5
    return basis, basis + num_std * std, basis - num_std * std


def bollinger_series(closes: list[float], period: int = 20, num_std: float = 2.0):
    """То же самое, но для каждой точки ряда — нужно для бэктеста и тренд-фильтра."""
    basis_series, upper_series, lower_series = [], [], []
    for i in range(len(closes)):
        b, u, l = bollinger_bands(closes[:i + 1], period, num_std)
        basis_series.append(b)
        upper_series.append(u)
        lower_series.append(l)
    return basis_series, upper_series, lower_series


def percent_b(price: float, upper: float, lower: float) -> float:
    """0 = цена на нижней полосе, 1 = на верхней, 0.5 = точно на средней."""
    band_range = upper - lower
    if band_range == 0:
        return 0.5
    return (price - lower) / band_range


def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _wilder_smooth(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    smoothed = [sum(values[:period])]
    for v in values[period:]:
        smoothed.append(smoothed[-1] - smoothed[-1] / period + v)
    return smoothed


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> float | None:
    """Average True Range по Уайлдеру. None, если данных меньше period+1."""
    n = len(closes)
    if n < period + 1:
        return None
    trs = [_true_range(highs[i], lows[i], closes[i - 1]) for i in range(1, n)]
    smooth = _wilder_smooth(trs, period)
    if not smooth:
        return None
    return smooth[-1] / period


def adx(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> float | None:
    """
    Классический ADX (Уайлдер). >25 обычно считается признаком тренда,
    <20 — боковика/слабого тренда. Нужны настоящие high/low, не только close.
    """
    n = len(closes)
    if n < period * 2 + 1:
        return None

    trs, plus_dms, minus_dms = [], [], []
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        trs.append(_true_range(highs[i], lows[i], closes[i - 1]))
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    tr_smooth = _wilder_smooth(trs, period)
    plus_dm_smooth = _wilder_smooth(plus_dms, period)
    minus_dm_smooth = _wilder_smooth(minus_dms, period)
    if not tr_smooth:
        return None

    plus_di = [100 * pd / tr if tr else 0.0 for pd, tr in zip(plus_dm_smooth, tr_smooth)]
    minus_di = [100 * md / tr if tr else 0.0 for md, tr in zip(minus_dm_smooth, tr_smooth)]
    dx = [100 * abs(p - m) / (p + m) if (p + m) else 0.0 for p, m in zip(plus_di, minus_di)]

    if len(dx) < period:
        return None
    adx_val = sum(dx[:period]) / period
    for d in dx[period:]:
        adx_val = (adx_val * (period - 1) + d) / period
    return adx_val

