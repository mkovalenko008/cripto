"""
Собирает dashboard/index.html из локальных файлов состояния
paper_state.json / trend_paper_state.json. Полные данные по сделкам
"запекаются" в HTML как JSON на момент сборки (страница ничего не
запрашивает извне — CSP артефактов не пускает произвольный fetch с
клиента); рендер карточек/графика/модалок — на JS в браузере, чтобы
работал переключатель периода без пересборки.

Обновление раз в день — отдельный плановый прогон: git pull -> этот
скрипт -> публикация.

Запуск: python3 dashboard/build_dashboard.py
"""
import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_json(name):
    with open(os.path.join(BASE_DIR, name)) as f:
        return json.load(f)


def coin_payload(state: dict) -> dict:
    return {
        "starting_capital": state["starting_capital"],
        "balance": state["balance_usdt"],
        "position": state.get("position"),
        "trades": [
            {
                "side": t["side"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "entry_time": t.get("entry_time") or t.get("entry_ts"),
                "exit_time": t.get("exit_time") or t.get("exit_ts"),
                "bars_held": t.get("bars_held"),
                "pnl_pct": t["pnl_pct"],
                "exit_reason": t.get("exit_reason", ""),
            }
            for t in state.get("trades", [])
        ],
    }


def main():
    trend_state = load_json("trend_paper_state.json")
    bb_state_raw = load_json("paper_state.json")

    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%d.%m.%Y, %H:%M UTC"),
        "bots": {
            "trend": {
                "label": "Трендовый бот",
                "subtitle": "Пробой полосы Боллинджера + ADX + трейлинг-стоп по ATR · 23 монеты · 1H",
                "accent": "trend",
                "validated": True,
                "method_note": (
                    "Конфигурация ADX≥30 / ATR×3.0 / std=2.5 отобрана честным train/test на 2 годах "
                    "истории: 63.6% монет дали плюс на test-выборке, которую не видели при подборе "
                    "параметров. Депозит 300 USDT, не больше 5% на монету (сейчас — поровну между 23 "
                    "монетами, каждая торгуется независимо)."
                ),
                "coins": {sym: coin_payload(st) for sym, st in trend_state.items()},
            },
            "meanrev": {
                "label": "Mean-reversion бот",
                "subtitle": "Отскок от полос Боллинджера + ADX + RSI · ETHUSDT · 1min",
                "accent": "meanrev",
                "validated": False,
                "method_note": (
                    "Первая протестированная стратегия сессии. На бэктесте устойчивого плюса после "
                    "комиссии не нашлось ни на одном таймфрейме — бот оставлен работать для сравнения "
                    "с трендовым, а не как рекомендация."
                ),
                "coins": {"ETHUSDT": coin_payload(bb_state_raw)},
            },
        },
    }

    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    out = template.replace("/*__DASHBOARD_DATA__*/", json.dumps(data, ensure_ascii=False))

    # dashboard/index.html — источник для публикации через инструмент Artifact
    # (плановая облачная задача раз в сутки).
    out_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)

    # docs/index.html — источник для GitHub Pages, обновляется чаще (вместе
    # с прогонами trend_paper_trader.yml, см. .github/workflows).
    docs_dir = os.path.join(BASE_DIR, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    docs_path = os.path.join(docs_dir, "index.html")
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(out)

    trend_trades = sum(len(c["trades"]) for c in data["bots"]["trend"]["coins"].values())
    mr_trades = sum(len(c["trades"]) for c in data["bots"]["meanrev"]["coins"].values())
    print(f"Собрано: {out_path}")
    print(f"Собрано: {docs_path}")
    print(f"Trend: {trend_trades} сделок всего")
    print(f"MeanRev: {mr_trades} сделок всего")


if __name__ == "__main__":
    main()
