"""
Поиск зон Fair Value Gap (FVG) — идея из ICT/Smart Money Concepts, которую
пользователь нашёл на чужом скрине в TradingView. Модная в трейдинг-блогах,
но без строгих доказательств edge — этот модуль только НАХОДИТ зоны по OHLC,
торговую гипотезу (вход на ретесте зоны) реализует fvg_backtest.py.

Определение (3-свечной паттерн, свечи k-2, k-1, k):
  бычий FVG  — low свечи k строго выше high свечи k-2 (гэп вверх, пропущенный
               "средней" свечой k-1) -> зона поддержки [high(k-2), low(k)]
  медвежий FVG — high свечи k строго ниже low свечи k-2 -> зона сопротивления
               [high(k), low(k-2)]

Не требует объёма — считается по high/low/close, которые уже есть в кэше
data/{SYMBOL}_1H.json.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FVGZone:
    kind: str  # "bull" (зона снизу, ждём LONG на ретесте) | "bear" (зона сверху, ждём SHORT)
    formed_index: int  # индекс свечи k, на которой зона подтвердилась
    top: float
    bottom: float

    @property
    def width(self) -> float:
        return self.top - self.bottom


def find_fvg_zones(candles: list[dict]) -> list[FVGZone]:
    """Один проход по всей серии свечей — возвращает зоны в порядке формирования."""
    zones = []
    for k in range(2, len(candles)):
        c_left, c_right = candles[k - 2], candles[k]
        if c_right["low"] > c_left["high"]:
            zones.append(FVGZone("bull", k, top=c_right["low"], bottom=c_left["high"]))
        elif c_right["high"] < c_left["low"]:
            zones.append(FVGZone("bear", k, top=c_left["low"], bottom=c_right["high"]))
    return zones
