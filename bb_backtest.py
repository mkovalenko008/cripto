"""
Движок бэктеста для bb_strategy v2. Работает на свечах OHLC (список словарей
{"open","high","low","close"}) — том же формате, что отдаёт
bitget_client.get_candles, так что бэктест и живой бот используют одну и ту
же структуру данных.

Логика сделки: вход по касанию полосы (после фильтров ADX/RSI), цель —
базовая линия (SMA) на момент входа, стоп — на расстоянии stop_mult от
полосы входа, таймаут — max_holding_bars баров.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from indicators import bollinger_bands
from bb_strategy import decide, Side


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    pnl_pct: float
    exit_reason: str
    adx_at_entry: float | None = None
    rsi_at_entry: float | None = None


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    skipped_by_adx: int = 0
    skipped_by_rsi: int = 0

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
            f"Сделок: {self.total_trades} "
            f"(отфильтровано по ADX: {self.skipped_by_adx}, по RSI: {self.skipped_by_rsi})\n"
            f"Win rate: {self.win_rate * 100:.1f}%\n"
            f"Суммарный результат: {self.total_return_pct:+.2f}% "
            f"(сумма % по сделкам равного размера)\n"
            f"Худшая серия подряд убыточных сделок: {self.max_losing_streak}"
        )


def run_backtest(candles: list[dict], period: int = 20, num_std: float = 2.0,
                  use_adx_filter: bool = True, adx_threshold: float = 25.0,
                  use_rsi_confirmation: bool = True,
                  stop_mult: float = 1.0, max_holding_bars: int = 20,
                  fee_pct_per_side: float = 0.1) -> BacktestResult:
    """
    fee_pct_per_side: комиссия биржи в % за одну сторону сделки (вход ИЛИ
    выход). На споте Bitget это 0.1% и на вход, и на выход, то есть 0.2%
    "съедает" каждую сделку туда-обратно. По умолчанию включена — чтобы
    увидеть цифры без комиссии, передай fee_pct_per_side=0.
    """
    result = BacktestResult()
    closes = [c["close"] for c in candles]
    i = period
    n = len(candles)

    while i < n:
        window = candles[:i + 1]
        d = decide(window, period, num_std,
                    use_adx_filter=use_adx_filter, adx_threshold=adx_threshold,
                    use_rsi_confirmation=use_rsi_confirmation)

        if not d.take_trade:
            if d.adx_value is not None and d.adx_value >= adx_threshold:
                result.skipped_by_adx += 1
            elif d.percent_b is not None and (d.percent_b <= 0.0 or d.percent_b >= 1.0):
                result.skipped_by_rsi += 1
            i += 1
            continue

        basis, upper, lower = bollinger_bands(closes[:i + 1], period, num_std)
        band_width = upper - lower
        entry_price = closes[i]
        entry_index = i

        if d.side == Side.LONG:
            target = basis
            stop = entry_price - stop_mult * band_width
        else:
            target = basis
            stop = entry_price + stop_mult * band_width

        exit_price, exit_index, reason = None, None, "timeout"
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            p = closes[j]
            if d.side == Side.LONG:
                if p >= target:
                    exit_price, exit_index, reason = p, j, "цель"
                    break
                if p <= stop:
                    exit_price, exit_index, reason = p, j, "стоп"
                    break
            else:
                if p <= target:
                    exit_price, exit_index, reason = p, j, "цель"
                    break
                if p >= stop:
                    exit_price, exit_index, reason = p, j, "стоп"
                    break

        if exit_price is None:
            exit_index = min(i + max_holding_bars, n - 1)
            exit_price = closes[exit_index]

        pnl_pct = ((exit_price - entry_price) / entry_price * 100
                   if d.side == Side.LONG
                   else (entry_price - exit_price) / entry_price * 100)
        pnl_pct -= 2 * fee_pct_per_side  # комиссия на вход и на выход

        result.trades.append(Trade(
            side=d.side.value, entry_price=entry_price, exit_price=exit_price,
            entry_index=entry_index, exit_index=exit_index, pnl_pct=pnl_pct,
            exit_reason=reason, adx_at_entry=d.adx_value, rsi_at_entry=d.rsi_value,
        ))
        i = exit_index + 1

    return result
