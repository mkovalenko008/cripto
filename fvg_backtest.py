"""
Движок бэктеста торговой гипотезы вокруг FVG-зон (см. fvg_strategy.py):
"цена, вернувшаяся в непроверенную FVG-зону, оттуда отскакивает в сторону
исходного импульса" — именно так эти зоны используют в ICT/SMC-чартах вроде
того, что нашёл пользователь. Проверяем это честно, а не на веру.

Правила (простые и фиксированные, без дискреционных "подтверждений"):
  - Зона активна max_zone_age_bars баров с момента формирования — не
    оживляем полугодовалые нетронутые зоны, это неправдоподобно.
  - Вход — на первом баре, чей диапазон (low/high) касается зоны, входим по
    ближнему краю зоны. Одна сделка в моменте, как и в trend_backtest.py.
  - Стоп — на дальней границе зоны (без буфера). Цель — reward_r * риск.
  - Выход по закрытиям свечей (как в bb_backtest.py / trend_backtest.py),
    вход — по касанию диапазоном (low/high), иначе саму идею "ретеста" не
    проверить.
  - Как только зона тронута (сделка была или почти была) — она выбывает из
    активных: повторно её не торгуем.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fvg_strategy import find_fvg_zones


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    entry_index: int
    exit_index: int
    pnl_pct: float
    exit_reason: str
    zone_width_pct: float


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


def run_backtest(candles: list[dict], max_zone_age_bars: int = 200,
                  reward_r: float = 2.0, max_holding_bars: int = 100,
                  fee_pct_per_side: float = 0.1, long_only: bool = False,
                  min_zone_width_pct: float = 0.0) -> BacktestResult:
    result = BacktestResult()
    n = len(candles)
    closes = [c["close"] for c in candles]

    zones = find_fvg_zones(candles)
    zones_by_formed = zones  # find_fvg_zones уже отдаёт их по возрастанию formed_index
    zone_ptr = 0
    active: list = []

    i = 3
    while i < n:
        bar = candles[i]

        # активируем зоны, сформированные к этому бару
        while zone_ptr < len(zones_by_formed) and zones_by_formed[zone_ptr].formed_index <= i:
            active.append(zones_by_formed[zone_ptr])
            zone_ptr += 1

        # ищем касание среди активных зон (предпочитаем недавно сформированную)
        touched = None
        still_active = []
        for z in reversed(active):
            age = i - z.formed_index
            width_pct = z.width / bar["close"] * 100
            expired = age > max_zone_age_bars
            too_narrow = width_pct < min_zone_width_pct
            if expired or too_narrow:
                continue  # выбывает молча — не тронута, просто устарела/слишком узкая
            # Зона по построению касается САМА СЕБЯ на баре формирования
            # (z.top/z.bottom буквально равны low/high этого бара) — вход
            # разрешаем только СТРОГО ПОЗЖЕ, иначе это не ретест, а
            # заглядывание вперёд по лучшей цене бара, который сам же
            # подтвердил зону.
            if touched is None and age > 0:
                if z.kind == "bull" and bar["low"] <= z.top:
                    touched = z
                    continue
                if z.kind == "bear" and bar["high"] >= z.bottom:
                    touched = z
                    continue
            still_active.append(z)
        active = list(reversed(still_active))

        if touched is None:
            i += 1
            continue

        side_long = touched.kind == "bull"
        if long_only and not side_long:
            i += 1
            continue

        entry_price = touched.top if side_long else touched.bottom
        stop_price = touched.bottom if side_long else touched.top
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            i += 1
            continue
        target_price = (entry_price + reward_r * risk) if side_long else (entry_price - reward_r * risk)

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
            side="LONG" if side_long else "SHORT",
            entry_price=entry_price, exit_price=exit_price,
            entry_index=i, exit_index=exit_index, pnl_pct=pnl_pct,
            exit_reason=reason, zone_width_pct=touched.width / bar["close"] * 100,
        ))
        i = exit_index + 1

    return result
