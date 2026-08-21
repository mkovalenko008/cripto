"""
Всё, что не даёт боту сжечь депозит из-за бага или зависшего цикла.
Работает поверх файла-лога сделок (trades_log.jsonl), так что лимиты
переживают перезапуск бота в течение дня.
"""
import json
import os
from datetime import datetime, timezone

import config


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def kill_switch_active() -> bool:
    """Если в папке бота лежит файл STOP — бот немедленно останавливается."""
    return os.path.exists(config.KILL_SWITCH_FILE)


def _read_today_trades() -> list[dict]:
    if not os.path.exists(config.TRADES_LOG_FILE):
        return []
    today = _today_str()
    trades = []
    with open(config.TRADES_LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("date") == today:
                trades.append(row)
    return trades


def can_trade() -> tuple[bool, str]:
    """Возвращает (можно_ли_торговать, причина_если_нет)."""
    if kill_switch_active():
        return False, f"kill switch активен (найден файл {config.KILL_SWITCH_FILE})"

    today_trades = _read_today_trades()

    if len(today_trades) >= config.MAX_TRADES_PER_DAY:
        return False, f"дневной лимит сделок исчерпан ({config.MAX_TRADES_PER_DAY})"

    realized_pnl = sum(t.get("pnl_usdt", 0.0) for t in today_trades)
    if realized_pnl <= -abs(config.MAX_DAILY_LOSS_USDT):
        return False, f"дневной лимит убытка исчерпан ({realized_pnl:.4f} USDT)"

    return True, ""


def log_trade(side: str, symbol: str, size_usdt: float, price: float,
              pnl_usdt: float = 0.0, dry_run: bool = True, raw_response=None):
    row = {
        "date": _today_str(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "side": side,
        "symbol": symbol,
        "size_usdt": size_usdt,
        "price": price,
        "pnl_usdt": pnl_usdt,
        "dry_run": dry_run,
        "raw_response": raw_response,
    }
    with open(config.TRADES_LOG_FILE, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
