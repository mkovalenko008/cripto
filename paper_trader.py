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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("paper_trader")


@dataclass
class PaperState:
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


def load_state(capital: float, reset: bool) -> PaperState:
    if not reset and os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            st = PaperState.from_dict(json.load(f))
        log.info("Продолжаю с сохранённого состояния: баланс=%.4f USDT, сделок=%d",
                  st.balance_usdt, len(st.trades))
        return st
    log.info("Стартую с чистого листа: виртуальный капитал=%.2f USDT", capital)
    return PaperState.fresh(capital)


def save_state(st: PaperState):
    with open(STATE_FILE, "w") as f:
        json.dump(st.to_dict(), f, indent=2)


def append_trade_log(row: dict):
    with open(TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def close_position(st: PaperState, exit_price: float, exit_ts: int, reason: str,
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
    log.info("[PAPER] ЗАКРЫЛ %s по %.4f (%s), PnL=%+.2f%% (%+.4f USDT), баланс=%.4f USDT",
              pos["side"], exit_price, reason, pnl_pct, pnl_usdt, st.balance_usdt)
    st.trades.append(row)
    st.position = None


def compute_stats(st: PaperState) -> dict:
    trades = st.trades
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate = (wins / n * 100) if n else 0.0
    total_return = (st.balance_usdt / st.starting_capital - 1) * 100

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t["pnl_pct"]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "n": n, "win_rate": win_rate, "total_return": total_return, "max_dd": max_dd,
    }


def summarize(st: PaperState) -> str:
    s = compute_stats(st)
    lines = [
        "\n" + "=" * 60,
        "ИТОГ ПО ВИРТУАЛЬНОЙ ТОРГОВЛЕ",
        "=" * 60,
        f"Стартовый капитал: {st.starting_capital:.2f} USDT",
        f"Текущий баланс:    {st.balance_usdt:.4f} USDT",
        f"Результат:         {s['total_return']:+.2f}%",
        f"Сделок закрыто:    {s['n']}",
        f"Win rate:          {s['win_rate']:.1f}%",
        f"Макс. просадка:    {s['max_dd']:.2f}%",
    ]
    if st.position:
        lines.append(f"Есть открытая позиция: {st.position['side']} по {st.position['entry_price']:.4f} "
                      f"(не включена в итог выше)")
    return "\n".join(lines)


def write_status_markdown(st: PaperState, args, last_price: float | None):
    s = compute_stats(st)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    pos_line = "нет открытой позиции"
    if st.position:
        pos = st.position
        pos_line = (f"{pos['side']} по {pos['entry_price']:.4f}, "
                     f"цель={pos['target']:.4f}, стоп={pos['stop']:.4f}, "
                     f"держим {pos['bars_held']}/{args.max_holding_bars} баров")

    recent = st.trades[-10:][::-1]
    recent_lines = ["_сделок пока не было_"] if not recent else [
        f"| {t['exit_time'][:16].replace('T',' ')} | {t['side']} | {t['entry_price']:.4f} | "
        f"{t['exit_price']:.4f} | {t['pnl_pct']:+.2f}% | {t['exit_reason']} |"
        for t in recent
    ]
    recent_table = "\n".join(recent_lines) if not st.trades else (
        "| Закрыта (UTC) | Сторона | Вход | Выход | PnL% | Причина |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(recent_lines)
    )

    text = f"""# Paper-trading статус (bb_strategy, вымышленные деньги)

**Это НЕ реальная торговля.** Ордера никогда не отправляются на биржу — это
симуляция стратегии из `bb_strategy.py` на живых ценах ETHUSDT, чтобы
посмотреть, как она вела бы себя, без риска для реальных денег.

Последняя проверка: **{now}**
Текущая цена ETHUSDT: **{last_price if last_price is not None else "н/д"}**

## Баланс

| Стартовый капитал | Текущий баланс | Результат | Сделок | Win rate | Макс. просадка |
|---|---|---|---|---|---|
| {st.starting_capital:.2f} USDT | {st.balance_usdt:.4f} USDT | {s['total_return']:+.2f}% | {s['n']} | {s['win_rate']:.1f}% | {s['max_dd']:.2f}% |

## Позиция

{pos_line}

## Последние сделки

{recent_table}

## Конфигурация

Таймфрейм={args.granularity}, num_std={args.num_std}, ADX-фильтр={args.use_adx}, RSI-подтверждение={args.use_rsi}, стоп={args.stop_mult}x ширины полосы, таймаут={args.max_holding_bars} баров, комиссия={args.fee_pct_per_side}%/сторону. Только LONG (спот без плеча шорт не поддерживает).

Полный лог сделок — [paper_trades_log.jsonl](paper_trades_log.jsonl). Бэктест на истории, из которого взяты эти параметры — [backtest_report.txt](backtest_report.txt).
"""
    with open(STATUS_FILE, "w") as f:
        f.write(text)


def process_tick(client: BitgetClient, st: PaperState, args) -> float | None:
    """
    Одна проверка: тянет свечи, если появилась новая закрытая свеча —
    управляет позицией или ищет сигнал на вход. Возвращает последнюю цену
    (для статуса), либо None при ошибке получения свечей.
    """
    try:
        candles = client.get_candles(granularity=args.granularity, limit=200)
    except Exception as e:
        log.error("Ошибка при получении свечей: %s", e)
        return None

    if len(candles) < 2:
        return None

    closed = candles[:-1]  # последняя свеча может быть ещё не закрыта
    last_closed = closed[-1]
    last_price = candles[-1]["close"]

    if st.last_processed_ts == last_closed["ts"]:
        return last_price

    st.last_processed_ts = last_closed["ts"]
    price = last_closed["close"]

    if st.position is not None:
        st.position["bars_held"] += 1
        pos = st.position
        hit_target = (price >= pos["target"]) if pos["side"] == "LONG" else (price <= pos["target"])
        hit_stop = (price <= pos["stop"]) if pos["side"] == "LONG" else (price >= pos["stop"])

        if hit_target:
            close_position(st, price, last_closed["ts"], "цель", args.fee_pct_per_side)
        elif hit_stop:
            close_position(st, price, last_closed["ts"], "стоп", args.fee_pct_per_side)
        elif pos["bars_held"] >= args.max_holding_bars:
            close_position(st, price, last_closed["ts"], "таймаут", args.fee_pct_per_side)
        else:
            log.info("Позиция %s открыта: цена=%.4f, цель=%.4f, стоп=%.4f, бар %d/%d",
                      pos["side"], price, pos["target"], pos["stop"],
                      pos["bars_held"], args.max_holding_bars)

    else:
        d = decide(closed, period=args.period, num_std=args.num_std,
                   use_adx_filter=args.use_adx, adx_threshold=args.adx_threshold,
                   use_rsi_confirmation=args.use_rsi,
                   rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought)

        if not d.take_trade:
            log.info("Сигнала нет: %s", d.reason)
        elif d.side == Side.SHORT:
            log.info("Сигнал SHORT пропущен — на споте без плеча шорт невозможен (%s)", d.reason)
        else:
            closes = [c["close"] for c in closed]
            basis, upper, lower = bollinger_bands(closes, args.period, args.num_std)
            band_width = upper - lower
            target = basis
            stop = price - args.stop_mult * band_width

            st.position = {
                "side": "LONG",
                "entry_price": price,
                "entry_ts": last_closed["ts"],
                "target": target,
                "stop": stop,
                "bars_held": 0,
                "adx_at_entry": d.adx_value,
                "rsi_at_entry": d.rsi_value,
            }
            log.info("[PAPER] ОТКРЫЛ LONG по %.4f, цель=%.4f, стоп=%.4f (%s)",
                      price, target, stop, d.reason)

    return last_price


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


def log_start(args):
    log.info("Paper-trading: %s, num_std=%.1f, ADX-фильтр=%s, RSI-подтв=%s, "
              "стоп=%.1fx ширины полосы, таймаут=%d баров, комиссия=%.2f%%/сторону",
              args.granularity, args.num_std, args.use_adx, args.use_rsi,
              args.stop_mult, args.max_holding_bars, args.fee_pct_per_side)
    log.info("НАПОМИНАНИЕ: это симуляция на вымышленные деньги. Реальные ордера "
              "не отправляются ни при каких условиях.")


def run_once(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    st = load_state(args.capital, args.reset)
    log_start(args)

    if kill_switch_active():
        log.warning("Kill switch активен (файл %s). Пропускаю проверку.", KILL_SWITCH_FILE)
        write_status_markdown(st, args, last_price=None)
        return

    last_price = process_tick(client, st, args)
    save_state(st)
    write_status_markdown(st, args, last_price)
    log.info(summarize(st))


def run_loop(args):
    client = BitgetClient(api_key="", secret_key="", passphrase="")
    st = load_state(args.capital, args.reset)
    start_time = time.time()
    deadline = start_time + args.duration_hours * 3600
    last_heartbeat = 0.0
    log_start(args)
    log.info("Длительность цикла: %.1fч", args.duration_hours)

    try:
        while time.time() < deadline:
            if kill_switch_active():
                log.warning("Kill switch активен (файл %s). Останавливаюсь.", KILL_SWITCH_FILE)
                break

            last_price = process_tick(client, st, args)
            save_state(st)
            if last_price is not None:
                write_status_markdown(st, args, last_price)
                now = time.time()
                if now - last_heartbeat > 300:
                    log.info("Текущая цена=%.4f, баланс=%.4f USDT%s", last_price, st.balance_usdt,
                              f", позиция={st.position['side']}" if st.position else "")
                    last_heartbeat = now

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        log.info("Остановлено пользователем (Ctrl+C).")

    save_state(st)
    log.info(summarize(st))


def parse_args():
    p = argparse.ArgumentParser(description="Paper-trading bb_strategy на живых данных Bitget (без реальных денег)")
    p.add_argument("--once", action="store_true",
                    help="Одна проверка и выход (режим для GitHub Actions / cron)")
    p.add_argument("--capital", type=float, default=10.0, help="Вымышленный стартовый капитал в USDT")
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
