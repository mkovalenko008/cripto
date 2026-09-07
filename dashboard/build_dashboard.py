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

TREND_V2_NOTE = (
    "Ядро входа то же, что у v1 (ADX≥30 / ATR×3.0 / std=2.5, 1H). Отличий два, и оба — "
    "исправления, а не подбор параметров. Первое: только LONG. У v1 фильтра шортов не было "
    "вовсе, и 9 из 23 его живых сделок — шорты на споте, которые без плеча неисполнимы; на них "
    "пришлось 75% всего убытка v1. Второе: вход проверяется по всем пропущенным барам, а не "
    "только по последнему — у GitHub Actions бывают многочасовые разрывы в расписании, и v1 "
    "молча терял в них точки входа. Защиту прибыли (перевод в безубыток, поджатие трейлинга) "
    "проверили отдельным train/test-прогоном на 2 годах по 23 монетам и ОТКЛОНИЛИ: безубыток "
    "дал ровно те же цифры (при трейлинге ATR×3.0 стоп к прибыли в 1R и так стоит на входе), "
    "а поджатие ухудшило медиану с +15.62% до +1.69%, срезая редких крупных победителей. "
    "На test-выборке конфигурация LONG-only дала 15/23 монет в плюсе (65.2%), медиана +6.03%."
)

MEANREV_V2_NOTE = (
    "Тот же отскок от полос, но час вместо минуты, стоп ×1.5 ширины полос и таймаут 10 баров — "
    "параметры отобраны train/test-поиском на 2 годах по 23 монетам, а не подогнаны под живые "
    "сделки v1. Причина перехода на час арифметическая: комиссия 0.2% за круг фиксированная, а "
    "типичное движение растёт с таймфреймом — на минутках комиссия составляла 82% среднего "
    "движения, на часе 16%. У v1 это и был приговор: до комиссии он давал +25.71 п.п., после — "
    "минус 130.89 п.п. Честно о результате v2: он заметно лучше текущей конфигурации и на train "
    "(широта 47.8% против 34.8%), и на test (47.8% против 17.4%, медиана -1.47% против -13.29%), "
    "но устойчивого плюса всё равно нет — медиана отрицательная на обеих выборках, в плюсе 11 из "
    "23 монет. Это по-прежнему бенчмарк для сравнения с трендовым, а не рекомендация."
)


def load_json(name, default=None):
    """default возвращается, если файла ещё нет — так дашборд собирается и
    до первого прогона новой версии бота, не падая на отсутствующем состоянии."""
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
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
                "pnl_usdt": t.get("pnl_usdt"),
                "balance_after": t.get("balance_after"),
                "exit_reason": t.get("exit_reason", ""),
            }
            for t in state.get("trades", [])
        ],
    }


def main():
    trend_state = load_json("trend_paper_state.json")
    bb_state_raw = load_json("paper_state.json")
    trend_v2_state = load_json("trend_paper_state_v2.json", default={})
    bb_v2_state = load_json("paper_state_v2.json", default={})

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
                "subtitle": f"Отскок от полос Боллинджера + ADX + RSI · {len(bb_state_raw)} монет · 1min",
                "accent": "meanrev",
                "validated": False,
                "method_note": (
                    "Первая протестированная стратегия сессии. На бэктесте устойчивого плюса после "
                    "комиссии не нашлось ни на одной монете — бот оставлен работать для сравнения с "
                    "трендовым, а не как рекомендация. Корзина — топ по капитализации среди листингов "
                    "Coinbase, пересечённый с наличием на Bitget Spot. Депозит тот же условный 300 USDT, "
                    "что у трендового, не больше 5% на монету."
                ),
                "coins": {sym: coin_payload(st) for sym, st in bb_state_raw.items()},
            },
        },
    }

    # v2-версии добавляются, только когда их состояние уже появилось — до
    # первого прогона нового бота на дашборде просто нет лишней пустой карточки.
    if trend_v2_state:
        data["bots"]["trendv2"] = {
            "label": "Трендовый бот v2.0",
            "subtitle": (f"То же ядро входа + только LONG + защита прибыли · "
                         f"{len(trend_v2_state)} монет · 1H"),
            "accent": "trendv2",
            "validated": True,
            "method_note": TREND_V2_NOTE,
            "coins": {sym: coin_payload(st) for sym, st in trend_v2_state.items()},
        }
    if bb_v2_state:
        data["bots"]["meanrevv2"] = {
            "label": "Mean-reversion бот v2.0",
            "subtitle": (f"Тот же отскок от полос, но часовой таймфрейм · "
                         f"{len(bb_v2_state)} монет · 1H"),
            "accent": "meanrevv2",
            "validated": False,
            "method_note": MEANREV_V2_NOTE,
            "coins": {sym: coin_payload(st) for sym, st in bb_v2_state.items()},
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
