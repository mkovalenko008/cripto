"""
ПРИМЕР стратегии, а не рекомендация и не гарантия результата.
Простейшая логика на RSI: перекупленность/перепроданность.
Прежде чем пускать на реальные деньги — прогони на истории (backtest)
и посмотри логи в DRY_RUN хотя бы несколько дней.

Замени эту функцию на свою логику: другой индикатор, комбинацию сигналов,
подключение к тому же MACD/Bollinger Bands, что ты смотришь в TradingView, и т.д.
"""
from enum import Enum

import config
from indicators import rsi


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


def decide(closes: list[float], has_position: bool) -> Signal:
    value = rsi(closes, config.RSI_PERIOD)
    if value is None:
        return Signal.HOLD

    if not has_position and value <= config.RSI_OVERSOLD:
        return Signal.BUY
    if has_position and value >= config.RSI_OVERBOUGHT:
        return Signal.SELL
    return Signal.HOLD
