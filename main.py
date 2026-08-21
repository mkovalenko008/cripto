"""
Точка входа. Запуск: python main.py

Цикл:
  1. проверить kill switch и дневные лимиты (risk_manager)
  2. получить свечи с Bitget
  3. получить сигнал от strategy.decide()
  4. если есть сигнал и он проходит риск-проверку — выполнить (или залогировать
     как dry-run) и сохранить в trades_log.jsonl
  5. уснуть POLL_INTERVAL_SECONDS и повторить

Остановить: Ctrl+C, или создать файл STOP в этой же папке (kill switch),
бот подхватит его в начале следующей итерации.
"""
import json
import logging
import os
import time

import config
from bitget_client import BitgetClient, BitgetError
import risk_manager
from strategy import decide, Signal

STATE_FILE = "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(config.BOT_LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("main")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"has_position": False, "entry_price": None, "qty_base": None}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def run_once(client: BitgetClient, state: dict):
    can, reason = risk_manager.can_trade()
    if not can:
        log.info("Торговля приостановлена: %s", reason)
        return state

    candles = client.get_candles()
    closes = [c["close"] for c in candles]
    last_price = closes[-1]

    signal = decide(closes, has_position=state["has_position"])
    log.info("Цена=%.4f сигнал=%s позиция=%s", last_price, signal.value,
              state["has_position"])

    if signal == Signal.BUY:
        qty_base = config.TRADE_SIZE_USDT / last_price  # приближённо, без учёта комиссии
        if config.DRY_RUN:
            log.info("[DRY_RUN] купил бы %.6f %s на %.2f USDT по цене %.4f",
                      qty_base, config.SYMBOL, config.TRADE_SIZE_USDT, last_price)
            resp = None
        else:
            resp = client.place_market_buy(config.TRADE_SIZE_USDT)
            log.info("Ордер BUY отправлен: %s", resp)

        state = {"has_position": True, "entry_price": last_price, "qty_base": qty_base}
        risk_manager.log_trade("buy", config.SYMBOL, config.TRADE_SIZE_USDT,
                                last_price, pnl_usdt=0.0, dry_run=config.DRY_RUN,
                                raw_response=resp)
        save_state(state)

    elif signal == Signal.SELL:
        qty_base = state["qty_base"]
        entry_price = state["entry_price"]
        pnl_usdt = qty_base * (last_price - entry_price)

        if config.DRY_RUN:
            log.info("[DRY_RUN] продал бы %.6f %s по цене %.4f, PnL=%.4f USDT",
                      qty_base, config.SYMBOL, last_price, pnl_usdt)
            resp = None
        else:
            resp = client.place_market_sell(qty_base)
            log.info("Ордер SELL отправлен: %s", resp)

        risk_manager.log_trade("sell", config.SYMBOL, qty_base * last_price,
                                last_price, pnl_usdt=pnl_usdt, dry_run=config.DRY_RUN,
                                raw_response=resp)
        state = {"has_position": False, "entry_price": None, "qty_base": None}
        save_state(state)

    return state


def main():
    if not config.DRY_RUN and not (config.API_KEY and config.SECRET_KEY and config.PASSPHRASE):
        raise SystemExit("DRY_RUN=False, но API-ключи не заданы в .env. Останавливаюсь.")

    log.info("Старт бота. Символ=%s DRY_RUN=%s размер_сделки=%.2f USDT",
              config.SYMBOL, config.DRY_RUN, config.TRADE_SIZE_USDT)
    if config.DRY_RUN:
        log.info("Режим DRY_RUN: реальные ордера НЕ отправляются.")

    client = BitgetClient(config.API_KEY, config.SECRET_KEY, config.PASSPHRASE)
    state = load_state()

    while True:
        if risk_manager.kill_switch_active():
            log.warning("Kill switch активен (файл %s). Останавливаюсь.",
                        config.KILL_SWITCH_FILE)
            break
        try:
            state = run_once(client, state)
        except BitgetError as e:
            log.error("Ошибка Bitget API: %s", e)
        except Exception:
            log.exception("Неожиданная ошибка в цикле")
        time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
