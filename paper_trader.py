"""
Бумажная (paper) торговля стратегией bb_strategy на РЕАЛЬНЫХ живых свечах
Bitget, но с вымышленным балансом. Реальные ордера НИКОГДА не отправляются —
скрипт даже не импортирует place_market_buy/place_market_sell, физически не
может их вызвать. Работает только на публичных данных (get_candles), ключи
не нужны.

Правила входа/выхода — те же, что в bb_backtest.py (чтобы этот прогон был
сравним с уже сделанным бэктестом): цель = базовая линия (SMA), стоп = на
расстоянии stop_mult*ширина_полосы, таймаут = max_holding_bars новых баров,
комиссия fee_pct_per_side на вход и на выход.

Торгует ОДНОВРЕМЕННО корзину из 30 монет — топ-30 по капитализации среди
листингов Coinbase (см. SYMBOLS), у каждой реально есть USDT-пара на Bitget
Spot. Раньше бот торговал только ETHUSDT — расширено по прямой просьбе
пользователя, стратегия не показала edge ни на одной монете даже в
бэктесте на 23-монетной корзине, так что это не попытка найти прибыльную
монету, а просто более честное сравнение с трендовым ботом (у которого тоже
корзина, а не одна монета).

Условный депозит — 300 USDT, тот же, что у trend_paper_trader.py, поровну
между монетами (капитал/30), с явным лимитом 5% на монету — сейчас капитал/30
уже меньше 5%, лимит ничего не режет, но зафиксирован на случай изменения
списка монет (см. load_state).

Спот не поддерживает шорт без плеча — сигналы SHORT пропускаются и
логируются, реально исполняются только LONG.

Два режима запуска:
  --once            одна проверка и выход — под это заточен GitHub Actions
                     (cron дёргает скрипт раз в N минут, состояние живёт в
                     paper_state.json, который коммитится обратно в репо).
  без --once         бесконечный цикл с опросом раз в poll-seconds — для
                     локального запуска (например, --duration-hours 24).

Остановка (в обоих режимах): файл STOP в этой же папке (тот же kill switch,
что и у main.py — risk_manager.py тут не используется, это своя, отдельная
проверка). В цикле локального режима — ещё и Ctrl+C.
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
from bb_strategy import decide, Side
from indicators import bollinger_bands

BASE_DIR = os.path.dirname(__file__)
STATE_FILE = os.path.join(BASE_DIR, "paper_state.json")
TRADES_LOG_FILE = os.path.join(BASE_DIR, "paper_trades_log.jsonl")
STATUS_FILE = os.path.join(BASE_DIR, "PAPER_STATUS.md")
LOG_FILE = os.path.join(BASE_DIR, "paper_bot.log")
KILL_SWITCH_FILE = os.path.join(BASE_DIR, config.KILL_SWITCH_FILE)

# Топ-30 по капитализации среди монет, реально листингованных на Coinbase
# (проверено через публичный Coinbase Exchange API, без стейблкоинов и
# "обёрнутых"/пегованных активов), пересечённое с наличием USDT-пары на
# Bitget Spot (проверено вручную через get_candles на каждую монету,
# 2026-08-23) — данные всё равно берутся с Bitget, Coinbase тут только
# источник для отбора списка монет.
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "HYPEUSDT",
    "DOGEUSDT", "ZECUSDT", "LINKUSDT", "ADAUSDT", "XLMUSDT", "BCHUSDT",
    "LTCUSDT", "HBARUSDT", "SUIUSDT", "AVAXUSDT", "SHIBUSDT", "CROUSDT",
    "UNIUSDT", "NEARUSDT", "TAOUSDT", "PUMPUSDT", "AAVEUSDT", "WLFIUSDT",
    "ONDOUSDT", "ASTERUSDT", "PEPEUSDT", "MORPHOUSDT", "DOTUSDT", "SKYUSDT",
]

DEPOSIT = 300.0
MAX_POSITION_PCT = 0.05  # не больше 5% депозита в одной монете

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("paper_trader")


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
        return {
            "balance_usdt": self.balance_usdt,
            "starting_capital": self.starting_capital,
            "last_processed_ts": self.last_processed_ts,
            "position": self.position,
            "trades": self.trades,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


def load_state(total_capital: float, reset: bool) -> dict:
    equal_split = total_capital / len(SYMBOLS)
    max_per_coin = total_capital * MAX_POSITION_PCT
    per_coin = min(equal_split, max_per_coin)
    if not reset and os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            raw = json.load(f)
        return {s: CoinState.from_dict(raw[s]) for s in SYMBOLS if s in raw}
    log.info("Стартую с чистого листа: %.2f USDT на монету (лимит 5%% = %.2f) x %d монет = "
              "%.2f USDT задействовано из %.2f USDT депозита",
              per_coin, max_per_coin, len(SYMBOLS), per_coin * len(SYMBOLS), total_capital)
    return {s: CoinState.fresh(per_coin) for s in SYMBOLS}


def save_state(states: dict):
    with open(STATE_FILE, "w") as f:
        json.dump({s: st.to_dict() for s, st in states.items()}, f, indent=2)


def append_trade_log(row: dict):
    with open(TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def close_position(symbol: str, st: CoinState, exit_price: float, exit_ts: int, reason: str,
                    fee_pct_per_side: float):
    pos = st.position
    if pos["side"] == "LONG":
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    else:
        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * 100
    pnl_pct -= 2 * fee_pct_per_side

    balance_before = st.balance_usdt
    st.balance_usdt = balance_before * (1 + pnl_pct / 100)
    pnl_usdt = st.balance_usdt - balance_before

    row = {
        "symbol": symbol,
        "side": pos["side"],
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts,
        "entry_time": datetime.fromtimestamp(pos["entry_ts"] / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).isoformat(),
        "bars_held": pos["bars_held"],
        "pnl_pct": pnl_pct,
        "pnl_usdt": pnl_usdt,
        "balance_after": st.balance_usdt,
        "exit_reason": reason,
        "adx_at_entry": pos.get("adx_at_entry"),
        "rsi_at_entry": pos.get("rsi_at_entry"),
        "paper": True,
    }
    append_trade_log(row)
    log.info("[%s] ЗАКРЫЛ %s по %.6f (%s), PnL=%+.2f%% (%+.4f USDT), баланс=%.4f USDT",
              symbol, pos["side"], exit_price, reason, pnl_pct, pnl_usdt, st.balance_usdt)
    st.trades.append(row)
    st.position = None


def process_symbol_tick(client: BitgetClient, symbol: str, st: CoinState, args) -> float | None:
    """
    Одна проверка по одной монете: тянет свечи, если появилась новая
    закрытая свеча — управляет позицией или ищет сигнал на вход. Возвращает
    последнюю цену (для статуса), либо None при ошибке получения свечей.
    """
    try:
        candles = client.get_candles(symbol=symbol, granularity=args.granularity, limit=200)
    except Exception as e:
        log.error("[%s] ошибка получения свечей: %s", symbol, e)
        return None

    if len(candles) < 2:
        return None

    closed = candles[:-1]  # последняя свеча может быть ещё не закрыта
    last_price = candles[-1]["close"]

    if not closed:
        return last_price

    # Свечи короче интервала проверки означают, что между двумя запусками
    # может закрыться сразу несколько баров — разбираем все новые закрытые
    # свечи по порядку, а не только последнюю, чтобы не пропустить момент
    # касания цели/стопа на промежуточном баре.
    if st.last_processed_ts is None:
        new_bars = [closed[-1]]
    else:
        new_bars = [c for c in closed if c["ts"] > st.last_processed_ts]

    if not new_bars:
        return last_price

    for bar in new_bars:
        is_latest = bar["ts"] == closed[-1]["ts"]
        st.last_processed_ts = bar["ts"]
        price = bar["close"]

        if st.position is not None:
            st.position["bars_held"] += 1
            pos = st.position
            hit_target = (price >= pos["target"]) if pos["side"] == "LONG" else (price <= pos["target"])
            hit_stop = (price <= pos["stop"]) if pos["side"] == "LONG" else (price >= pos["stop"])

            if hit_target:
                close_position(symbol, st, price, bar["ts"], "цель", args.fee_pct_per_side)
            elif hit_stop:
                close_position(symbol, st, price, bar["ts"], "стоп", args.fee_pct_per_side)
            elif pos["bars_held"] >= args.max_holding_bars:
                close_position(symbol, st, price, bar["ts"], "таймаут", args.fee_pct_per_side)
            elif is_latest:
                log.info("[%s] позиция %s открыта: цена=%.6f, цель=%.6f, стоп=%.6f, бар %d/%d",
                          symbol, pos["side"], price, pos["target"], pos["stop"],
                          pos["bars_held"], args.max_holding_bars)

        elif is_latest:
            # Новую позицию открываем только по самому свежему бару — входить
            # по устаревшему сигналу (цена уже ушла) смысла нет.
            d = decide(closed, period=args.period, num_std=args.num_std,
                       use_adx_filter=args.use_adx, adx_threshold=args.adx_threshold,
                       use_rsi_confirmation=args.use_rsi,
                       rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought)

            if not d.take_trade:
                pass  # тихо — на 30 монетах логировать "сигнала нет" на каждой было бы шумом
            elif d.side == Side.SHORT:
                log.info("[%s] сигнал SHORT пропущен — на споте без плеча шорт невозможен (%s)",
                          symbol, d.reason)
            else:
                closes = [c["close"] for c in closed]
                basis, upper, lower = bollinger_bands(closes, args.period, args.num_std)
                band_width = upper - lower
                target = basis
                stop = price - args.stop_mult * band_width

                st.position = {
                    "side": "LONG",
                    "entry_price": price,
                    "entry_ts": bar["ts"],
                    "target": target,
                    "stop": stop,
                    "bars_held": 0,
                    "adx_at_entry": d.adx_value,
                    "rsi_at_entry": d.rsi_value,
                }
                log.info("[%s] ОТКРЫЛ LONG по %.6f, цель=%.6f, стоп=%.6f (%s)",
                          symbol, price, target, stop, d.reason)

    return last_price


def write_status(states: dict, args, last_prices: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_start = sum(st.starting_capital for st in states.values())
    total_now = sum(st.balance_usdt for st in states.values())
    total_trades = sum(len(st.trades) for st in states.values())
    total_return = (total_now / total_start - 1) * 100 if total_start else 0.0

    rows = []
    for s, st in sorted(states.items(), key=lambda kv: kv[1].balance_usdt / kv[1].starting_capital, reverse=True):
        ret = (st.balance_usdt / st.starting_capital - 1) * 100
        pos = f"{st.position['side']} @ {st.position['entry_price']:.6f}" if st.position else "—"
        rows.append(f"| {s} | {ret:+.2f}% | {len(st.trades)} | {pos} |")

    text = f"""# Paper-trading статус (bb_strategy, вымышленные деньги)

**Это НЕ реальная торговля.** Ордера никогда не отправляются на биржу — это
симуляция стратегии из `bb_strategy.py` на живых ценах корзины из
{len(states)} монет (топ по капитализации среди листингов Coinbase,
пересечённое с наличием на Bitget Spot), чтобы посмотреть, как она вела бы
себя, без риска для реальных денег. Стратегия не показала устойчивого edge
после комиссии ни на одной монете при бэктесте — это сравнение с трендовым
ботом, а не рекомендация.

Последняя проверка: **{now}**

## Портфель

| Стартовый капитал | Текущий баланс | Результат | Сделок всего |
|---|---|---|---|
| {total_start:.2f} USDT | {total_now:.4f} USDT | {total_return:+.2f}% | {total_trades} |

## По монетам

| Монета | Результат | Сделок | Позиция |
|---|---|---|---|
{chr(10).join(rows)}

## Конфигурация

num_std={args.num_std}, ADX-фильтр={args.use_adx}, RSI-подтверждение={args.use_rsi}, стоп={args.stop_mult}x ширины полосы, таймаут={args.max_holding_bars} баров, комиссия={args.fee_pct_per_side}%/сторону. Только LONG (спот без плеча шорт не поддерживает).

Полный лог сделок — [paper_trades_log.jsonl](paper_trades_log.jsonl).
"""
    with open(STATUS_FILE, "w") as f:
        f.write(text)


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


def log_start(args):
    log.info("Paper-trading: %s, %d монет, num_std=%.1f, ADX-фильтр=%s, RSI-подтв=%s, "
              "стоп=%.1fx ширины полосы, таймаут=%d баров, комиссия=%.2f%%/сторону",
              args.granularity, len(SYMBOLS), args.num_std, args.use_adx, args.use_rsi,
              args.stop_mult, args.max_holding_bars, args.fee_pct_per_side)
    log.info("НАПОМИНАНИЕ: это симуляция на вымышленные деньги. Реальные ордера "
              "не отправляются ни при каких условиях.")


def run_once(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    states = load_state(args.capital, args.reset)
    log_start(args)

    if kill_switch_active():
        log.warning("Kill switch активен (файл %s). Пропускаю проверку.", KILL_SWITCH_FILE)
        save_state(states)
        write_status(states, args, {})
        return

    last_prices = {}
    for symbol in SYMBOLS:
        last_prices[symbol] = process_symbol_tick(client, symbol, states[symbol], args)

    save_state(states)
    write_status(states, args, last_prices)
    total_start = sum(st.starting_capital for st in states.values())
    total_now = sum(st.balance_usdt for st in states.values())
    log.info("Портфель: %.4f -> %.4f USDT (%+.2f%%)", total_start, total_now,
              (total_now / total_start - 1) * 100)


def run_loop(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    states = load_state(args.capital, args.reset)
    deadline = time.time() + args.duration_hours * 3600
    last_heartbeat = 0.0
    log_start(args)
    log.info("Длительность цикла: %.1fч", args.duration_hours)

    try:
        while time.time() < deadline:
            if kill_switch_active():
                log.warning("Kill switch активен (файл %s). Останавливаюсь.", KILL_SWITCH_FILE)
                break

            last_prices = {}
            for symbol in SYMBOLS:
                last_prices[symbol] = process_symbol_tick(client, symbol, states[symbol], args)
            save_state(states)
            write_status(states, args, last_prices)

            now = time.time()
            if now - last_heartbeat > 300:
                total_start = sum(st.starting_capital for st in states.values())
                total_now = sum(st.balance_usdt for st in states.values())
                log.info("Портфель: %.4f -> %.4f USDT (%+.2f%%)", total_start, total_now,
                          (total_now / total_start - 1) * 100)
                last_heartbeat = now

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        log.info("Остановлено пользователем (Ctrl+C).")

    save_state(states)


def parse_args():
    p = argparse.ArgumentParser(description="Paper-trading bb_strategy на живых данных Bitget (без реальных денег)")
    p.add_argument("--once", action="store_true",
                    help="Одна проверка и выход (режим для GitHub Actions / cron)")
    p.add_argument("--capital", type=float, default=DEPOSIT,
                    help="Вымышленный стартовый депозит в USDT (делится поровну между монетами, лимит 5% на монету)")
    p.add_argument("--duration-hours", type=float, default=24.0, help="Сколько часов крутить бота (без --once)")
    p.add_argument("--granularity", default=config.CANDLE_GRANULARITY, help="Таймфрейм свечей (15min, 1h, 4h, ...)")
    p.add_argument("--poll-seconds", type=int, default=30, help="Как часто опрашивать API (без --once)")
    p.add_argument("--period", type=int, default=20, help="Период полос Боллинджера")
    p.add_argument("--num-std", type=float, default=2.0, help="Ширина полос в std")
    p.add_argument("--use-adx", action="store_true", default=True)
    p.add_argument("--no-adx", dest="use_adx", action="store_false")
    p.add_argument("--use-rsi", action="store_true", default=True)
    p.add_argument("--no-rsi", dest="use_rsi", action="store_false")
    p.add_argument("--adx-threshold", type=float, default=25.0)
    p.add_argument("--rsi-oversold", type=float, default=35.0)
    p.add_argument("--rsi-overbought", type=float, default=65.0)
    p.add_argument("--stop-mult", type=float, default=1.0)
    p.add_argument("--max-holding-bars", type=int, default=20)
    p.add_argument("--fee-pct-per-side", type=float, default=0.1)
    p.add_argument("--reset", action="store_true", help="Начать с чистого состояния, игнорируя paper_state.json")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.once:
        run_once(_args)
    else:
        run_loop(_args)
