"""
Третья стратегия сессии — намеренно другой природы, чем bb_strategy (mean-
reversion) и trend_strategy (пробой полосы): здесь ГЛАВНЫЙ триггер —
фундаментальный (Индекс страха и жадности, дальше в живом боте — новости
Coinbase/Investing.com), а из технического анализа используются только
трендовые линии и линии сопротивления/поддержки — никаких осцилляторов,
ADX, RSI, полос Боллинджера.

Как формализованы "линии", которые обычно рисуют рукой:
  - находим точки разворота (свинги) — локальный хай/лоу за pivot_lookback
    баров в обе стороны;
  - через последние swing_count лоёв строим линию тренда/поддержки методом
    наименьших квадратов (наклон>=0 — растущая, ещё не пробита);
  - через последние swing_count хаёв — линию сопротивления симметрично.
Это единственная механическая замена ручным линиям — сделано явно и
воспроизводимо, а не "на глаз".

Логика входа (фундаментал — главный, техника — фильтр/точка входа):
  LONG  — индекс страха <= fear_threshold (классическое "покупай на страхе")
          И цена вернулась к растущей линии поддержки (в пределах touch_pct)
  SHORT — индекс жадности >= greed_threshold ("продавай на жадности")
          И цена вернулась к падающей линии сопротивления

Стоп — за линией (с буфером ATR), цель — reward_r * риск. Максимум одна
позиция одновременно, выход по закрытиям — та же дисциплина, что у
trend_backtest.py / fvg_backtest.py.

Индекс страха/жадности — из data/fear_greed_index.json (alternative.me,
дневные значения, история с 2018 года, никакого API-ключа не нужно).
Значение на конкретный час свечи ищем по последнему дню <= её времени
(вперёд не заглядываем).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass
class SwingLine:
    slope: float
    intercept: float  # цена в индексе 0 окна (не глобальном индексе свечи)
    window_start_index: int

    def value_at(self, index: int) -> float:
        return self.intercept + self.slope * (index - self.window_start_index)


def find_swing_points(candles: list[dict], pivot_lookback: int, kind: str) -> list[tuple[int, float]]:
    """kind: 'high' или 'low'. Возвращает [(индекс_свечи, цена), ...] по возрастанию индекса."""
    key = "high" if kind == "high" else "low"
    better = (lambda a, b: a > b) if kind == "high" else (lambda a, b: a < b)
    points = []
    n = len(candles)
    for i in range(pivot_lookback, n - pivot_lookback):
        v = candles[i][key]
        is_pivot = True
        for j in range(i - pivot_lookback, i + pivot_lookback + 1):
            if j == i:
                continue
            if not better(v, candles[j][key]) and candles[j][key] != v:
                is_pivot = False
                break
            if candles[j][key] == v and j < i:
                is_pivot = False  # равные хаи/лои — оставляем самый ранний как pivot
                break
        if is_pivot:
            points.append((i, v))
    return points


def fit_line(points: list[tuple[int, float]]) -> SwingLine | None:
    """МНК по последним точкам (index, price). Возвращает None, если точек < 2."""
    if len(points) < 2:
        return None
    x0 = points[0][0]
    xs = [p[0] - x0 for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x
    return SwingLine(slope=slope, intercept=intercept, window_start_index=x0)


class FearGreedIndex:
    """Дневной индекс, без забегания вперёд — на любой момент времени отдаёт
    последнее известное К ЭТОМУ МОМЕНТУ значение (закрытие предыдущего дня)."""

    def __init__(self, records: list[dict]):
        self.records = sorted(records, key=lambda r: r["ts"])
        self.timestamps = [r["ts"] for r in self.records]

    def value_at(self, ts_ms: int) -> int | None:
        idx = bisect_right(self.timestamps, ts_ms) - 1
        if idx < 0:
            return None
        return self.records[idx]["value"]


@dataclass
class Decision:
    take_trade: bool
    side: str | None = None  # "LONG" | "SHORT"
    reason: str = ""
    fear_greed: int | None = None
    line_value: float | None = None
    line_slope: float | None = None


def decide(candles: list[dict], fgi: FearGreedIndex, current_ts_ms: int,
           pivot_lookback: int = 5, swing_count: int = 4, lookback_bars: int = 200,
           fear_threshold: int = 25, greed_threshold: int = 75,
           touch_pct: float = 1.0) -> Decision:
    """candles — окно свечей до текущего бара включительно (последняя = текущая)."""
    fg = fgi.value_at(current_ts_ms)
    if fg is None:
        return Decision(take_trade=False, reason="нет данных индекса страха/жадности на эту дату")

    price = candles[-1]["close"]
    window = candles[-lookback_bars:] if len(candles) > lookback_bars else candles
    offset = len(candles) - len(window)

    if fg <= fear_threshold:
        lows = find_swing_points(window, pivot_lookback, "low")[-swing_count:]
        line = fit_line(lows)
        if line is None:
            return Decision(take_trade=False, reason="страх есть, но мало точек для линии поддержки", fear_greed=fg)
        support_now = line.value_at(len(window) - 1)
        touch = abs(price - support_now) / price * 100 <= touch_pct
        if line.slope >= 0 and touch:
            return Decision(take_trade=True, side="LONG",
                             reason=f"страх={fg}<= {fear_threshold}, цена у растущей линии поддержки",
                             fear_greed=fg, line_value=support_now, line_slope=line.slope)
        return Decision(take_trade=False, reason="страх есть, но цена не у линии поддержки (или линия падает)",
                         fear_greed=fg, line_value=support_now, line_slope=line.slope)

    if fg >= greed_threshold:
        highs = find_swing_points(window, pivot_lookback, "high")[-swing_count:]
        line = fit_line(highs)
        if line is None:
            return Decision(take_trade=False, reason="жадность есть, но мало точек для линии сопротивления", fear_greed=fg)
        resistance_now = line.value_at(len(window) - 1)
        touch = abs(price - resistance_now) / price * 100 <= touch_pct
        if line.slope <= 0 and touch:
            return Decision(take_trade=True, side="SHORT",
                             reason=f"жадность={fg}>= {greed_threshold}, цена у падающей линии сопротивления",
                             fear_greed=fg, line_value=resistance_now, line_slope=line.slope)
        return Decision(take_trade=False, reason="жадность есть, но цена не у линии сопротивления (или линия растёт)",
                         fear_greed=fg, line_value=resistance_now, line_slope=line.slope)

    return Decision(take_trade=False, reason=f"индекс {fg} — нейтральная зона, вне страха/жадности", fear_greed=fg)
