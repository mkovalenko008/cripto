"""
Движок бэктеста для трендовой стратегии v2 — то же ядро, что в
trend_backtest.py, плюс защита от отдачи прибыли обратно трейлинг-стопу.

Зачем: на живых данных v1 обнаружилась характерная проблема — сделки
заходили в плюс и полностью разворачивались до срабатывания трейлинга.
12 из 20 убыточных сделок были в плюсе больше 1%, одна доходила до +11%
и закрылась в ноль; победители тоже отдавали по 1.3-7 п.п. от максимума.
ATR×3.0 достаточно широкий, чтобы позиция успела развернуться целиком.

Два механизма, оба выражены в единицах начального риска R
(R = atr_stop_mult × ATR на входе — то есть расстояние до первого стопа):

  breakeven_at — при достижении прибыли >= N×R стоп переносится в точку
                 входа (плюс комиссия), дальше сделка не может стать
                 убыточной. Классический "перевод в безубыток".
  tighten_at / tighten_mult — при прибыли >= M×R трейлинг сжимается с
                 atr_stop_mult до tighten_mult, чтобы забирать больше от
                 уже состоявшегося движения.

None в любом из параметров = механизм выключен (тогда это ровно v1).
long_only здесь по умолчанию True: на споте без плеча шорт неисполним.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trend_strategy import decide, Side


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    pnl_pct: float
    exit_reason: str
    mfe_pct: float = 0.0          # максимум в плюс за время сделки
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
    def avg_giveback_pct(self):
        """Сколько в среднем отдано обратно от максимума позиции."""
        if not self.trades:
            return 0.0
        return sum(t.mfe_pct - t.pnl_pct for t in self.trades) / len(self.trades)

    def summary(self) -> str:
        return (
            f"Сделок: {self.total_trades}\n"
            f"Win rate: {self.win_rate * 100:.1f}%\n"
            f"Суммарный результат: {self.total_return_pct:+.2f}%\n"
            f"Средняя отдача от максимума: {self.avg_giveback_pct:.2f} п.п."
        )


def run_backtest(candles: list[dict], period: int = 20, num_std: float = 2.5,
                  adx_period: int = 14, adx_threshold: float = 30.0,
                  atr_period: int = 14, atr_stop_mult: float = 3.0,
                  max_holding_bars: int = 100,
                  fee_pct_per_side: float = 0.1, long_only: bool = True,
                  breakeven_at: float | None = None,
                  tighten_at: float | None = None,
                  tighten_mult: float | None = None,
                  lookback: int = 150) -> BacktestResult:
    result = BacktestResult()
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    i = max(period, adx_period * 2 + 1, atr_period + 1)
    n = len(candles)
    fee_round_trip = 2 * fee_pct_per_side

    while i < n:
        window = candles[max(0, i + 1 - lookback):i + 1]
        d = decide(window, period, num_std, adx_period, adx_threshold, atr_period, lookback=lookback)

        if not d.take_trade or (long_only and d.side == Side.SHORT):
            i += 1
            continue

        entry_price = closes[i]
        entry_index = i
        side = d.side
        atr_val = d.atr_value or 0.0
        risk = atr_stop_mult * atr_val          # 1R — начальное расстояние до стопа

        if risk <= 0:
            i += 1
            continue

        # Безубыток ставим чуть выше входа, чтобы комиссия круга была покрыта.
        fee_cushion = entry_price * fee_round_trip / 100
        breakeven_price = (entry_price + fee_cushion if side == Side.LONG
                           else entry_price - fee_cushion)

        extreme = entry_price
        trail_distance = risk
        trailing_stop = (entry_price - risk if side == Side.LONG else entry_price + risk)

        exit_price, exit_index, reason = None, None, "таймаут"
        mfe = 0.0

        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            p = closes[j]
            hi, lo = highs[j], lows[j]

            # MFE считаем по экстремуму бара, а не по закрытию — это то,
            # сколько прибыли реально показывала позиция.
            fav = ((hi - entry_price) / entry_price * 100 if side == Side.LONG
                   else (entry_price - lo) / entry_price * 100)
            mfe = max(mfe, fav)

            if side == Side.LONG:
                if p > extreme:
                    extreme = p
                profit_r = (extreme - entry_price) / risk
                if tighten_at is not None and tighten_mult is not None and profit_r >= tighten_at:
                    trail_distance = tighten_mult * atr_val
                candidate = extreme - trail_distance
                if breakeven_at is not None and profit_r >= breakeven_at:
                    candidate = max(candidate, breakeven_price)
                trailing_stop = max(trailing_stop, candidate)
                if p <= trailing_stop:
                    exit_price, exit_index, reason = p, j, "трейлинг-стоп"
                    break
            else:
                if p < extreme:
                    extreme = p
                profit_r = (entry_price - extreme) / risk
                if tighten_at is not None and tighten_mult is not None and profit_r >= tighten_at:
                    trail_distance = tighten_mult * atr_val
                candidate = extreme + trail_distance
                if breakeven_at is not None and profit_r >= breakeven_at:
                    candidate = min(candidate, breakeven_price)
                trailing_stop = min(trailing_stop, candidate)
                if p >= trailing_stop:
                    exit_price, exit_index, reason = p, j, "трейлинг-стоп"
                    break

        if exit_price is None:
            exit_index = min(i + max_holding_bars, n - 1)
            exit_price = closes[exit_index]

        pnl_pct = ((exit_price - entry_price) / entry_price * 100
                   if side == Side.LONG
                   else (entry_price - exit_price) / entry_price * 100)
        pnl_pct -= fee_round_trip

        result.trades.append(Trade(
            side=side.value, entry_price=entry_price, exit_price=exit_price,
            entry_index=entry_index, exit_index=exit_index, pnl_pct=pnl_pct,
            exit_reason=reason, mfe_pct=mfe - fee_round_trip, adx_at_entry=d.adx_value,
        ))
        i = exit_index + 1

    return result
