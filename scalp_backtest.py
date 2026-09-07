"""
Движок бэктеста скальп-идеи: фиксированная цель и стоп в % от цены входа
("пара тиков"), а не возврат к базовой линии, как в bb_backtest. Именно
поэтому это отдельный файл, а не параметр к bb_backtest.

Ключевое отличие от bb_backtest в самом способе проверки выхода: там цель —
базовая линия, до которой обычно далеко, и проверки по close свечи достаточно.
Здесь цель в 0.2-0.9% часто решается тем, что было ВНУТРИ бара, а не на его
закрытии — close мог не дойти ни до цели, ни до стопа, а high/low того же
бара оба их задеть. Поэтому здесь выход проверяется по high/low каждого
следующего бара. Если в одном баре задеты и цель, и стоп одновременно —
считаем, что стоп сработал первым (мы не знаем реальный порядок событий
внутри бара без тиковых данных, и предполагать лучшее для стратегии — это
путь к переоптимистичному бэктесту).
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
    pnl_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    skipped_short: int = 0   # LONG-only: сигналы SHORT на споте нереализуемы

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
            f"Сделок: {self.total_trades} (пропущено SHORT-сигналов: {self.skipped_short})\n"
            f"Win rate: {self.win_rate * 100:.1f}%\n"
            f"Суммарный результат: {self.total_return_pct:+.2f}% "
            f"(сумма % по сделкам равного размера)\n"
            f"Худшая серия подряд убыточных сделок: {self.max_losing_streak}"
        )


def run_backtest(candles: list[dict], period: int = 20, num_std: float = 2.0,
                  use_adx_filter: bool = False, adx_period: int = 14, adx_threshold: float = 30.0,
                  target_pct: float = 0.4, stop_pct: float = 0.5,
                  max_holding_bars: int = 12, fee_pct_per_side: float = 0.1,
                  long_only: bool = True) -> BacktestResult:
    """
    target_pct / stop_pct: расстояние до цели/стопа в % от цены входа —
    "пара тиков", откалиброванная под волатильность конкретной монеты (или
    подобранная поиском по сетке, см. scalp_config_search.py).
    long_only: на споте Bitget шорт без плеча невозможен (см. paper_trader.py)
    — при True сигналы SHORT считаются и пропускаются, не берутся в бэктест,
    чтобы суммарная доходность отражала то, что бот реально может исполнить.
    """
    result = BacktestResult()
    n = len(candles)
    i = period
    # На каждом баре decide() пересчитывает индикаторы, а bollinger_bands сам
    # смотрит только на последние period точек — но candles[:i+1] без
    # ограничения копировал бы всё более длинный срез на каждом шаге,
    # O(n^2) по памяти и времени на десятках тысяч баров и сотнях
    # конфигураций сетки. LOOKBACK ограничивает срез тем, что реально нужно
    # (плюс запас на ADX, если он включён; Wilder-сглаживание сходится за
    # ~100-150 баров, см. bb_config_search_v2.py).
    lookback = max(period, adx_period + 100 if use_adx_filter else 0) + 5

    while i < n:
        window = candles[max(0, i + 1 - lookback):i + 1]
        d = decide(window, period, num_std, use_adx_filter=use_adx_filter,
                   adx_period=adx_period, adx_threshold=adx_threshold)

        if not d.take_trade:
            i += 1
            continue
        if long_only and d.side == Side.SHORT:
            result.skipped_short += 1
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
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            bar = candles[j]
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

        pnl_pct = ((exit_price - entry_price) / entry_price * 100
                   if d.side == Side.LONG
                   else (entry_price - exit_price) / entry_price * 100)
        pnl_pct -= 2 * fee_pct_per_side

        result.trades.append(Trade(
            side=d.side.value, entry_price=entry_price, exit_price=exit_price,
            entry_index=entry_index, exit_index=exit_index, pnl_pct=pnl_pct,
            exit_reason=reason, bars_held=exit_index - entry_index,
        ))
        i = exit_index + 1

    return result
