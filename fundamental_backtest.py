"""
Движок бэктеста для fundamental_strategy.py. Та же дисциплина выхода, что у
fvg_backtest.py/trend_backtest.py: одна сделка одновременно, вход по сигналу
decide(), стоп за линией (с буфером ATR — чтобы не выбивало ровно на касании),
цель = reward_r * риск, выход проверяется по закрытиям свечей.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fundamental_strategy import decide, FearGreedIndex
from indicators import atr


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    pnl_pct: float
    exit_reason: str
    fear_greed: int | None


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)

    @property
    def total_trades(self):
        return len(self.trades)

    @property
    def win_rate(self):
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct > 0 for t in self.trades) / len(self.trades)

    @property
    def total_return_pct(self):
        return sum(t.pnl_pct for t in self.trades)

    @property
    def max_losing_streak(self):
        streak = worst = 0
        for t in self.trades:
            if t.pnl_pct <= 0:
                streak += 1
                worst = max(worst, streak)
            else:
                streak = 0
        return worst

    def summary(self) -> str:
        return (
            f"Сделок: {self.total_trades}\n"
            f"Win rate: {self.win_rate * 100:.1f}%\n"
            f"Суммарный результат: {self.total_return_pct:+.2f}%\n"
            f"Худшая серия подряд убыточных сделок: {self.max_losing_streak}"
        )


def run_backtest(candles: list[dict], fgi: FearGreedIndex,
                  pivot_lookback: int = 5, swing_count: int = 4, lookback_bars: int = 200,
                  fear_threshold: int = 25, greed_threshold: int = 75, touch_pct: float = 1.0,
                  atr_period: int = 14, atr_stop_buffer_mult: float = 0.5, reward_r: float = 2.0,
                  max_holding_bars: int = 150, fee_pct_per_side: float = 0.1,
                  long_only: bool = False) -> BacktestResult:
    result = BacktestResult()
    n = len(candles)
    closes = [c["close"] for c in candles]

    min_start = pivot_lookback * 2 + swing_count + 5
    i = min_start
    while i < n:
        window = candles[max(0, i + 1 - lookback_bars):i + 1]
        d = decide(window, fgi, candles[i]["ts"], pivot_lookback, swing_count, lookback_bars,
                   fear_threshold, greed_threshold, touch_pct)

        if not d.take_trade or (long_only and d.side == "SHORT"):
            i += 1
            continue

        highs = [c["high"] for c in window]
        lows = [c["low"] for c in window]
        a = atr(highs, lows, [c["close"] for c in window], atr_period) or 0.0
        buffer = atr_stop_buffer_mult * a

        entry_price = closes[i]
        side_long = d.side == "LONG"
        stop_price = (d.line_value - buffer) if side_long else (d.line_value + buffer)
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            i += 1
            continue
        target_price = entry_price + reward_r * risk if side_long else entry_price - reward_r * risk

        exit_price, exit_index, reason = None, None, "таймаут"
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            p = closes[j]
            if side_long:
                if p <= stop_price:
                    exit_price, exit_index, reason = p, j, "стоп"
                    break
                if p >= target_price:
                    exit_price, exit_index, reason = p, j, "цель"
                    break
            else:
                if p >= stop_price:
                    exit_price, exit_index, reason = p, j, "стоп"
                    break
                if p <= target_price:
                    exit_price, exit_index, reason = p, j, "цель"
                    break

        if exit_price is None:
            exit_index = min(i + max_holding_bars, n - 1)
            exit_price = closes[exit_index]

        pnl_pct = ((exit_price - entry_price) / entry_price * 100 if side_long
                   else (entry_price - exit_price) / entry_price * 100)
        pnl_pct -= 2 * fee_pct_per_side

        result.trades.append(Trade(
            side=d.side, entry_price=entry_price, exit_price=exit_price,
            entry_index=i, exit_index=exit_index, pnl_pct=pnl_pct,
            exit_reason=reason, fear_greed=d.fear_greed,
        ))
        i = exit_index + 1

    return result
