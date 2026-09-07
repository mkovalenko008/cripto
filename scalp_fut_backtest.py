"""
Фьючерсный вариант scalp_backtest.py: тот же сигнал и тот же принцип выхода
(фиксированная цель/стоп в % от входа, high/low каждого следующего бара,
консервативный тай-брейк при задетых цели и стопе в одном баре), но:

- комиссия по умолчанию 0.06%/сторону (Bitget USDT-M taker), как в
  futures_backtest.py, а не 0.1% на споте;
- шорт не пропускается: на фьючерсах, в отличие от спота, шорт без плеча —
  штатная операция, не требует отдельного обоснования;
- добавлен funding: если позиция удерживается через момент выплаты (каждые
  8 часов = funding_period_bars баров на этом таймфрейме), баланс
  корректируется на funding_rate следующего бара, знак — как в
  futures_backtest.py (лонг платит при положительном funding, шорт получает).
  На короткой скальп-позиции (десятки минут – пара часов) funding обычно не
  успевает набежать ни разу, но при max_holding_bars, покрывающем 8+ часов,
  это уже не пренебрежимо, поэтому считается честно, а не игнорируется.

Плечо (10x, о котором просил пользователь) НЕ параметр этого бэктеста: PnL
здесь везде в % от цены (то же самое поле, что и в спотовом движке), а не
в % от маржи. Плечо линейно масштабирует и профит, и убыток, и комиссию
относительно маржи ОДИНАКОВО — оно не меняет, окупает ли % цель комиссию.
Оно лишь определяет, сколько маржи нужно, чтобы то же самое процентное
движение дало нужный доход в долларах, и сокращает соответствующую этому
плечу дистанцию до ликвидации (см. дисклеймер в отчёте, не в этом файле).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scalp_strategy import decide, Side


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    price_pnl_pct: float
    funding_pnl_pct: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


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


def run_backtest(candles: list[dict], period: int = 20, num_std: float = 2.0,
                  use_adx_filter: bool = False, adx_period: int = 14, adx_threshold: float = 30.0,
                  target_pct: float = 0.4, stop_pct: float = 0.5,
                  max_holding_bars: int = 12, fee_pct_per_side: float = 0.06,
                  funding_period_bars: int = 96) -> BacktestResult:
    """funding_period_bars=96 -> 8 часов на 5-минутном таймфрейме (96*5мин=8ч),
    как в futures_backtest.py на 1H (там funding_period_bars=8)."""
    result = BacktestResult()
    n = len(candles)
    i = period
    lookback = max(period, adx_period + 100 if use_adx_filter else 0) + 5

    while i < n:
        window = candles[max(0, i + 1 - lookback):i + 1]
        d = decide(window, period, num_std, use_adx_filter=use_adx_filter,
                   adx_period=adx_period, adx_threshold=adx_threshold)

        if not d.take_trade:
            i += 1
            continue

        entry_price = candles[i]["close"]
        entry_index = i
        if d.side == Side.LONG:
            target = entry_price * (1 + target_pct / 100)
            stop = entry_price * (1 - stop_pct / 100)
        else:
            target = entry_price * (1 - target_pct / 100)
            stop = entry_price * (1 + stop_pct / 100)

        exit_price, exit_index, reason = None, None, "таймаут"
        funding_pnl_pct = 0.0
        bars_held = 0
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            bar = candles[j]
            bars_held += 1
            if bars_held % funding_period_bars == 0:
                rate = bar.get("funding_rate", 0.0)
                funding_pnl_pct += (-rate if d.side == Side.LONG else rate) * 100

            if d.side == Side.LONG:
                hit_stop = bar["low"] <= stop
                hit_target = bar["high"] >= target
            else:
                hit_stop = bar["high"] >= stop
                hit_target = bar["low"] <= target

            if hit_stop:
                exit_price, exit_index = stop, j
                reason = "стоп+цель в одном баре (взят стоп)" if hit_target else "стоп"
                break
            if hit_target:
                exit_price, exit_index, reason = target, j, "цель"
                break

        if exit_price is None:
            exit_index = min(i + max_holding_bars, n - 1)
            exit_price = candles[exit_index]["close"]

        price_pnl_pct = ((exit_price - entry_price) / entry_price * 100
                          if d.side == Side.LONG
                          else (entry_price - exit_price) / entry_price * 100)
        pnl_pct = price_pnl_pct - 2 * fee_pct_per_side + funding_pnl_pct

        result.trades.append(Trade(
            side=d.side.value, entry_price=entry_price, exit_price=exit_price,
            entry_index=entry_index, exit_index=exit_index, price_pnl_pct=price_pnl_pct,
            funding_pnl_pct=funding_pnl_pct, pnl_pct=pnl_pct,
            exit_reason=reason, bars_held=exit_index - entry_index,
        ))
        i = exit_index + 1

    return result
