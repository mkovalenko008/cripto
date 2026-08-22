"""
Движок бэктеста для futures_strategy.py. Механика выхода та же, что в
trend_backtest.py (трейлинг-стоп по ATR), но:
- комиссия по умолчанию 0.06%/сторону (Bitget USDT-M taker), а не 0.1% как
  на споте — 0.12% за круг вместо 0.2%;
- дополнительно учитывается funding: если позиция удерживается через момент
  выплаты funding (каждые 8 часов), баланс корректируется на funding_rate *
  знак_позиции (лонг платит при положительном funding, шорт платит при
  отрицательном) — это реальные периодические списания на фьючерсах,
  которых нет на споте.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from futures_strategy import decide, Side


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    pnl_pct: float
    funding_pnl_pct: float
    exit_reason: str
    adx_at_entry: float | None = None


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


def run_backtest(candles: list[dict], period: int = 20, num_std: float = 2.5,
                  adx_period: int = 14, adx_threshold: float = 25.0,
                  atr_period: int = 14, atr_stop_mult: float = 2.0,
                  max_holding_bars: int = 100,
                  use_volume_filter: bool = True, volume_period: int = 20, volume_mult: float = 1.5,
                  use_funding_filter: bool = True, funding_extreme: float = 0.0005,
                  fee_pct_per_side: float = 0.06, funding_period_bars: int = 8,
                  long_only: bool = False) -> BacktestResult:
    """funding_period_bars: раз в сколько 1H-баров списывается funding (на
    Bitget — каждые 8 часов, т.е. 8 баров на таймфрейме 1H)."""
    result = BacktestResult()
    closes = [c["close"] for c in candles]
    i = max(period, adx_period * 2 + 1, atr_period + 1)
    n = len(candles)

    while i < n:
        window = candles[:i + 1]
        d = decide(window, period, num_std, adx_period, adx_threshold, atr_period,
                    use_volume_filter=use_volume_filter, volume_period=volume_period,
                    volume_mult=volume_mult, use_funding_filter=use_funding_filter,
                    funding_extreme=funding_extreme)

        if not d.take_trade or (long_only and d.side == Side.SHORT):
            i += 1
            continue

        entry_price = closes[i]
        entry_index = i
        side = d.side
        stop_distance = atr_stop_mult * (d.atr_value or 0.0)
        if stop_distance <= 0:
            i += 1
            continue

        extreme = entry_price
        trailing_stop = (entry_price - stop_distance if side == Side.LONG
                          else entry_price + stop_distance)

        exit_price, exit_index, reason = None, None, "таймаут"
        funding_pnl_pct = 0.0
        bars_held = 0
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            p = closes[j]
            bars_held += 1
            if bars_held % funding_period_bars == 0:
                rate = candles[j].get("funding_rate", 0.0)
                funding_pnl_pct += (-rate if side == Side.LONG else rate) * 100

            if side == Side.LONG:
                if p > extreme:
                    extreme = p
                    trailing_stop = extreme - stop_distance
                if p <= trailing_stop:
                    exit_price, exit_index, reason = p, j, "трейлинг-стоп"
                    break
            else:
                if p < extreme:
                    extreme = p
                    trailing_stop = extreme + stop_distance
                if p >= trailing_stop:
                    exit_price, exit_index, reason = p, j, "трейлинг-стоп"
                    break

        if exit_price is None:
            exit_index = min(i + max_holding_bars, n - 1)
            exit_price = closes[exit_index]

        price_pnl_pct = ((exit_price - entry_price) / entry_price * 100
                          if side == Side.LONG
                          else (entry_price - exit_price) / entry_price * 100)
        pnl_pct = price_pnl_pct - 2 * fee_pct_per_side + funding_pnl_pct

        result.trades.append(Trade(
            side=side.value, entry_price=entry_price, exit_price=exit_price,
            entry_index=entry_index, exit_index=exit_index, pnl_pct=pnl_pct,
            funding_pnl_pct=funding_pnl_pct, exit_reason=reason, adx_at_entry=d.adx_value,
        ))
        i = exit_index + 1

    return result
