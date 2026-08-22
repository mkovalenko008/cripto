"""
Арбитраж базиса spot/perp через funding rate — принципиально другой класс
стратегии по сравнению со всем, что тестировали раньше: не прогноз цены, а
сбор периодических выплат funding на ДЕЛЬТА-НЕЙТРАЛЬНОЙ позиции.

Механика: лонг спот + шорт бессрочный фьючерс на ту же монету, в равном
номинале. Ценовой риск взаимно гасится (если цена растёт — спот в плюсе,
шорт-фьючерс в минусе на ту же величину, и наоборот). Источник прибыли —
funding rate: когда он положительный, шорты получают выплату от лонгов
каждые 8 часов, а мы именно в шорте по фьючерсу.

PnL сделки = изменение базиса (futures_price - spot_price) между входом и
выходом [+ если базис сузился, - если расширился] + сумма funding-выплат за
время удержания - комиссии по обеим ногам (спот 0.1%/сторону + фьючерс
0.06%/сторону = 0.32% за полный цикл вход+выход по обеим ногам).

Вход: funding >= entry_threshold. Выход: funding упал ниже exit_threshold,
либо стал отрицательным (не в нашу пользу), либо истёк max_holding_bars.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class BasisTrade:
    entry_index: int
    exit_index: int
    bars_held: int
    basis_pnl_pct: float
    funding_pnl_pct: float
    fee_pct: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BasisResult:
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

    def summary(self) -> str:
        return (f"Сделок: {self.total_trades}\n"
                f"Win rate: {self.win_rate * 100:.1f}%\n"
                f"Суммарный результат: {self.total_return_pct:+.2f}%")


def load_merged(symbol: str):
    """Совмещает спот (data/{SYMBOL}_1H.json) и фьючерс с funding
    (data/{SYMBOL}_FUT_1H.json) по временной метке."""
    spot_path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
    fut_path = os.path.join(DATA_DIR, f"{symbol}_FUT_1H.json")
    if not (os.path.exists(spot_path) and os.path.exists(fut_path)):
        return None

    with open(spot_path) as f:
        spot = {c["ts"]: c["close"] for c in json.load(f)}
    with open(fut_path) as f:
        fut = json.load(f)

    merged = []
    for c in fut:
        if c["ts"] in spot:
            merged.append({
                "ts": c["ts"],
                "spot_close": spot[c["ts"]],
                "fut_close": c["close"],
                "funding_rate": c.get("funding_rate", 0.0),
            })
    return merged


def run_backtest(bars: list[dict], entry_threshold: float = 0.0003,
                  exit_threshold: float = 0.0001, max_holding_bars: int = 200,
                  funding_period_bars: int = 8,
                  spot_fee_pct_per_side: float = 0.1, fut_fee_pct_per_side: float = 0.06) -> BasisResult:
    result = BasisResult()
    n = len(bars)
    round_trip_fee = 2 * spot_fee_pct_per_side + 2 * fut_fee_pct_per_side
    i = 0

    while i < n:
        funding = bars[i]["funding_rate"]
        if funding < entry_threshold:
            i += 1
            continue

        entry_index = i
        spot_entry = bars[i]["spot_close"]
        fut_entry = bars[i]["fut_close"]
        basis_entry = fut_entry - spot_entry

        exit_index, reason = None, "таймаут"
        funding_pnl_pct = 0.0
        bars_held = 0
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            bars_held += 1
            if bars_held % funding_period_bars == 0:
                funding_pnl_pct += bars[j]["funding_rate"] * 100  # мы в шорте фьючерса — собираем funding

            current_funding = bars[j]["funding_rate"]
            if current_funding < exit_threshold:
                exit_index, reason = j, "funding упал ниже порога"
                break

        if exit_index is None:
            exit_index = min(i + max_holding_bars, n - 1)

        spot_exit = bars[exit_index]["spot_close"]
        fut_exit = bars[exit_index]["fut_close"]
        basis_exit = fut_exit - spot_exit

        # мы шорт фьючерса + лонг спот: прибыль от сужения базиса (в % от спот-цены на входе)
        basis_pnl_pct = (basis_entry - basis_exit) / spot_entry * 100

        pnl_pct = basis_pnl_pct + funding_pnl_pct - round_trip_fee

        result.trades.append(BasisTrade(
            entry_index=entry_index, exit_index=exit_index, bars_held=bars_held,
            basis_pnl_pct=basis_pnl_pct, funding_pnl_pct=funding_pnl_pct,
            fee_pct=round_trip_fee, pnl_pct=pnl_pct, exit_reason=reason,
        ))
        i = exit_index + 1

    return result
