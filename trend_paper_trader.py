"""
Живой paper-trading на конфигурации, отобранной в trend_config_search.py:
ADX>=30, ATRx3.0, num_std=2.5, таймфрейм 1H — та самая, что дала 63.6%
широты (14/22 монет в плюсе) на честном test после train/test-разбивки на
2 годах истории. Полный отчёт: trend_config_search_report.txt.

Торгует ОДНОВРЕМЕННО корзину из 23 монет, каждая — независимым виртуальным
суб-балансом, теми же правилами входа/выхода без подгонки под конкретную
монету — так же, как считался бэктест. Депозит 300 USDT, на монету — не
больше 5% депозита (сейчас это капитал/23, автоматически ниже лимита; см.
load_state).

Реальные ордера НИКОГДА не отправляются — скрипт не импортирует функции
размещения ордеров. Ключи API не нужны (только публичные свечи).

Режимы: --once (одна проверка, для GitHub Actions), без него — цикл с
опросом (для локального запуска).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
from bitget_client import BitgetClient
from trend_strategy import decide, Side
from indicators import bollinger_bands

BASE_DIR = os.path.dirname(__file__)

# BOT_VARIANT выбирает профиль бота. v1 — исходная версия, поведение которой
# не меняется (файлы и логика ровно те же, что были). v2 — версия с правками
# по итогам разбора живых сделок, живёт в отдельных файлах и торгует
# параллельно, чтобы сравнение шло вперёд, а не подгонкой задним числом.
VARIANT = os.getenv("BOT_VARIANT", "v1").lower()
_SUF = "" if VARIANT == "v1" else f"_{VARIANT}"

STATE_FILE = os.path.join(BASE_DIR, f"trend_paper_state{_SUF}.json")
TRADES_LOG_FILE = os.path.join(BASE_DIR, f"trend_paper_trades_log{_SUF}.jsonl")
STATUS_FILE = os.path.join(BASE_DIR, f"TREND_PAPER_STATUS{_SUF.upper()}.md")
LOG_FILE = os.path.join(BASE_DIR, f"trend_paper_bot{_SUF}.log")
KILL_SWITCH_FILE = os.path.join(BASE_DIR, config.KILL_SWITCH_FILE)

SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "HYPEUSDT", "BTCUSDT", "XRPUSDT", "DOGEUSDT",
    "ZECUSDT", "SUIUSDT", "PEPEUSDT", "ENAUSDT", "ONDOUSDT", "TRUMPUSDT",
    "LINKUSDT", "BNBUSDT", "UNIUSDT", "ADAUSDT", "LTCUSDT", "NEARUSDT",
    "XLMUSDT", "AAVEUSDT", "AVAXUSDT", "TRXUSDT", "BCHUSDT",
]

STRAT = dict(period=20, num_std=2.5, adx_period=14, adx_threshold=30.0, atr_period=14)
STOP_MULT = 3.0
MAX_HOLDING_BARS = 100
FEE_PCT_PER_SIDE = 0.1
GRANULARITY = "1h"

# --- профиль версии -------------------------------------------------------
# v1: как было. Шорты берутся (хотя на споте они неисполнимы — это и есть
#     одна из найденных проблем), защиты от отдачи прибыли нет, вход
#     ищется только по самому свежему бару.
# v2: только LONG (реальность спота — шорт без плеча неисполним) и проверка
#     входа по всем пропущенным барам (иначе разрывы в расписании GitHub
#     Actions съедают точки входа).
#
# Защиты прибыли (перевод в безубыток / поджатие трейлинга) здесь НЕТ, и это
# результат проверки, а не недосмотр. По живым сделкам казалось, что проблема
# в отдаче прибыли: 12 из 20 убыточных заходили в плюс больше 1%, одна была
# +11% и закрылась в ноль. Прогон train/test на 2 годах по 23 монетам
# (trend_config_search_v2.py) показал обратное:
#   - перевод в безубыток при 1R/1.5R/2R дал ровно те же цифры, что и без него.
#     Так и должно быть: при трейлинге ATR×3.0 прибыль в 1R = 3×ATR означает,
#     что стоп уже подтянут ровно ко входу — двигать нечего;
#   - поджатие трейлинга до ATR×1.5 ухудшило медиану с +15.62% до +1.69%:
#     оно режет тех самых редких больших победителей, на которых у трендовой
#     стратегии и держится весь результат.
# Отдача прибыли обратно — это цена входа в трендследование, а не поломка.
if VARIANT == "v2":
    LONG_ONLY = True
    BREAKEVEN_AT = TIGHTEN_AT = TIGHTEN_MULT = None
    BACKFILL_ENTRIES = True
else:
    LONG_ONLY = False
    BREAKEVEN_AT = TIGHTEN_AT = TIGHTEN_MULT = None
    BACKFILL_ENTRIES = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("trend_paper_trader")


@dataclass
class CoinState:
    balance_usdt: float
    starting_capital: float
    last_processed_ts: int | None = None
    position: dict | None = None
    trades: list = field(default_factory=list)

    @classmethod
    def fresh(cls, capital: float):
        return cls(balance_usdt=capital, starting_capital=capital)

    def to_dict(self):
        return {"balance_usdt": self.balance_usdt, "starting_capital": self.starting_capital,
                "last_processed_ts": self.last_processed_ts, "position": self.position,
                "trades": self.trades}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


MAX_POSITION_PCT = 0.05  # не больше 5% депозита в одной монете


def load_state(total_capital: float, reset: bool) -> dict:
    equal_split = total_capital / len(SYMBOLS)
    max_per_coin = total_capital * MAX_POSITION_PCT
    per_coin = min(equal_split, max_per_coin)
    if not reset and os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            raw = json.load(f)
        return {s: CoinState.from_dict(raw[s]) for s in SYMBOLS if s in raw}
    log.info("Стартую с чистого листа: %.2f USDT на монету (лимит 5%% = %.2f) x %d монет = %.2f USDT задействовано из %.2f USDT депозита",
              per_coin, max_per_coin, len(SYMBOLS), per_coin * len(SYMBOLS), total_capital)
    return {s: CoinState.fresh(per_coin) for s in SYMBOLS}


def save_state(states: dict):
    with open(STATE_FILE, "w") as f:
        json.dump({s: st.to_dict() for s, st in states.items()}, f, indent=2)


def append_trade_log(row: dict):
    with open(TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def close_position(symbol: str, st: CoinState, exit_price: float, exit_ts: int, reason: str):
    pos = st.position
    if pos["side"] == "LONG":
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    else:
        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * 100
    pnl_pct -= 2 * FEE_PCT_PER_SIDE

    balance_before = st.balance_usdt
    st.balance_usdt = balance_before * (1 + pnl_pct / 100)
    row = {
        "symbol": symbol, "side": pos["side"], "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "entry_time": datetime.fromtimestamp(pos["entry_ts"] / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).isoformat(),
        "bars_held": pos["bars_held"], "pnl_pct": pnl_pct,
        "pnl_usdt": st.balance_usdt - balance_before, "balance_after": st.balance_usdt,
        "exit_reason": reason, "adx_at_entry": pos.get("adx_at_entry"), "paper": True,
    }
    append_trade_log(row)
    log.info("[%s] ЗАКРЫЛ %s по %.6f (%s), PnL=%+.2f%%, баланс=%.4f USDT",
              symbol, pos["side"], exit_price, reason, pnl_pct, st.balance_usdt)
    st.trades.append(row)
    st.position = None


def process_symbol_tick(client: BitgetClient, symbol: str, st: CoinState) -> float | None:
    try:
        candles = client.get_candles(symbol=symbol, granularity=GRANULARITY, limit=200)
    except Exception as e:
        log.error("[%s] ошибка получения свечей: %s", symbol, e)
        return None
    if len(candles) < 2:
        return None

    closed = candles[:-1]
    last_price = candles[-1]["close"]
    if not closed:
        return last_price

    if st.last_processed_ts is None:
        new_bars = [(len(closed) - 1, closed[-1])]
    else:
        new_bars = [(i, c) for i, c in enumerate(closed) if c["ts"] > st.last_processed_ts]
    if not new_bars:
        return last_price

    for idx, bar in new_bars:
        is_latest = bar["ts"] == closed[-1]["ts"]
        st.last_processed_ts = bar["ts"]
        price = bar["close"]
        window = closed[:idx + 1]

        if st.position is not None:
            st.position["bars_held"] += 1
            pos = st.position
            # risk — начальное расстояние до стопа (1R). У позиций, открытых
            # до появления v2, поля risk нет — берём stop_distance, который
            # тогда ещё не поджимался и равен исходному.
            risk = pos.get("risk") or pos["stop_distance"]
            hit_target_stop = False

            if pos["side"] == "LONG":
                if price > pos["extreme"]:
                    pos["extreme"] = price
                profit_r = (pos["extreme"] - pos["entry_price"]) / risk if risk else 0.0
                if TIGHTEN_AT is not None and profit_r >= TIGHTEN_AT:
                    atr_at_entry = pos.get("atr_at_entry") or (risk / STOP_MULT)
                    pos["stop_distance"] = TIGHTEN_MULT * atr_at_entry
                candidate = pos["extreme"] - pos["stop_distance"]
                if BREAKEVEN_AT is not None and profit_r >= BREAKEVEN_AT:
                    candidate = max(candidate, pos["entry_price"] * (1 + 2 * FEE_PCT_PER_SIDE / 100))
                pos["trailing_stop"] = max(pos["trailing_stop"], candidate)
                if price <= pos["trailing_stop"]:
                    hit_target_stop = True
            else:
                if price < pos["extreme"]:
                    pos["extreme"] = price
                profit_r = (pos["entry_price"] - pos["extreme"]) / risk if risk else 0.0
                if TIGHTEN_AT is not None and profit_r >= TIGHTEN_AT:
                    atr_at_entry = pos.get("atr_at_entry") or (risk / STOP_MULT)
                    pos["stop_distance"] = TIGHTEN_MULT * atr_at_entry
                candidate = pos["extreme"] + pos["stop_distance"]
                if BREAKEVEN_AT is not None and profit_r >= BREAKEVEN_AT:
                    candidate = min(candidate, pos["entry_price"] * (1 - 2 * FEE_PCT_PER_SIDE / 100))
                pos["trailing_stop"] = min(pos["trailing_stop"], candidate)
                if price >= pos["trailing_stop"]:
                    hit_target_stop = True

            if hit_target_stop:
                close_position(symbol, st, price, bar["ts"], "трейлинг-стоп")
            elif pos["bars_held"] >= MAX_HOLDING_BARS:
                close_position(symbol, st, price, bar["ts"], "таймаут")

        elif is_latest or BACKFILL_ENTRIES:
            # v1 смотрит вход только по самому свежему бару; v2 разбирает и
            # пропущенные — при разрывах в расписании GitHub Actions иначе
            # теряются точки входа. decide() получает window (бары строго по
            # состоянию на этот момент), чтобы не заглядывать вперёд.
            d = decide(window, **STRAT)
            if d.take_trade and not (LONG_ONLY and d.side == Side.SHORT):
                atr_val = d.atr_value or 0.0
                stop_distance = STOP_MULT * atr_val
                if stop_distance > 0:
                    st.position = {
                        "side": d.side.value, "entry_price": price, "entry_ts": bar["ts"],
                        "extreme": price,
                        "trailing_stop": (price - stop_distance if d.side == Side.LONG
                                          else price + stop_distance),
                        "stop_distance": stop_distance, "risk": stop_distance,
                        "atr_at_entry": atr_val, "bars_held": 0,
                        "adx_at_entry": d.adx_value,
                    }
                    log.info("[%s] ОТКРЫЛ %s по %.6f (%s)", symbol, d.side.value, price, d.reason)
            elif d.take_trade and LONG_ONLY and d.side == Side.SHORT:
                log.info("[%s] сигнал SHORT пропущен — на споте без плеча шорт неисполним", symbol)

    return last_price


def write_status(states: dict, last_prices: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_start = sum(st.starting_capital for st in states.values())
    total_now = sum(st.balance_usdt for st in states.values())
    total_return = (total_now / total_start - 1) * 100

    rows = []
    for s, st in sorted(states.items(), key=lambda kv: kv[1].balance_usdt / kv[1].starting_capital, reverse=True):
        ret = (st.balance_usdt / st.starting_capital - 1) * 100
        pos = f"{st.position['side']} @ {st.position['entry_price']:.4f}" if st.position else "—"
        rows.append(f"| {s} | {ret:+.2f}% | {len(st.trades)} | {pos} |")

    text = f"""# Trend paper-trading статус (живая корзина, вымышленные деньги)

**Это НЕ реальная торговля.** Конфигурация: ADX>=30, ATRx3.0, num_std=2.5, 1H,
отобрана по train/test на 2 годах истории (см. trend_config_search_report.txt,
63.6% монет в плюсе на test). Капитал поровну разбит на {len(states)} монет,
каждая торгуется независимо одинаковыми правилами.

Последняя проверка: **{now}**

## Портфель

| Стартовый капитал | Текущий | Результат |
|---|---|---|
| {total_start:.2f} USDT | {total_now:.4f} USDT | {total_return:+.2f}% |

## По монетам

| Монета | Результат | Сделок | Позиция |
|---|---|---|---|
{chr(10).join(rows)}

Лог сделок — [trend_paper_trades_log.jsonl](trend_paper_trades_log.jsonl).
"""
    with open(STATUS_FILE, "w") as f:
        f.write(text)


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


def run_once(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    states = load_state(args.capital, args.reset)

    if kill_switch_active():
        log.warning("Kill switch активен. Пропускаю проверку.")
        save_state(states)
        write_status(states, {})
        return

    last_prices = {}
    for symbol in SYMBOLS:
        last_prices[symbol] = process_symbol_tick(client, symbol, states[symbol])

    save_state(states)
    write_status(states, last_prices)
    total_start = sum(st.starting_capital for st in states.values())
    total_now = sum(st.balance_usdt for st in states.values())
    log.info("Портфель: %.4f -> %.4f USDT (%+.2f%%)", total_start, total_now,
              (total_now / total_start - 1) * 100)


def run_loop(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    states = load_state(args.capital, args.reset)
    deadline = time.time() + args.duration_hours * 3600
    last_heartbeat = 0.0

    try:
        while time.time() < deadline:
            if kill_switch_active():
                log.warning("Kill switch активен. Останавливаюсь.")
                break
            last_prices = {}
            for symbol in SYMBOLS:
                last_prices[symbol] = process_symbol_tick(client, symbol, states[symbol])
            save_state(states)
            write_status(states, last_prices)

            now = time.time()
            if now - last_heartbeat > 300:
                total_start = sum(st.starting_capital for st in states.values())
                total_now = sum(st.balance_usdt for st in states.values())
                log.info("Портфель: %.4f -> %.4f USDT (%+.2f%%)", total_start, total_now,
                          (total_now / total_start - 1) * 100)
                last_heartbeat = now
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        log.info("Остановлено пользователем.")

    save_state(states)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--capital", type=float, default=300.0)
    p.add_argument("--duration-hours", type=float, default=24.0)
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.once:
        run_once(_args)
    else:
        run_loop(_args)
