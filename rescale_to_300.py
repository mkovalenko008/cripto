"""
Одноразовая миграция: возвращает суммарный стартовый капитал каждого бота к
$300 после того, как расширение корзины до 100 монет раздуло его (новые
монеты заводились по доле СТАРЫХ монет, а не по $300/100 — из-за этого total
рос вместе с числом монет: $1317 у trend v1, $1010 у meanrev v1 и т.д.).

Метод — чистое масштабирование, не сброс: starting_capital и balance_usdt
КАЖДОЙ активной монеты умножаются на один и тот же коэффициент k = 300/total,
и то же k применяется к pnl_usdt/balance_after в истории её сделок. Все
балансы в этом боте — произведение starting_capital на цепочку множителей
(1+pnl_pct/100) по сделкам (см. close_position() в paper_trader.py и
trend_paper_trader.py: balance = balance_before*(1+pnl_pct/100)), а
pnl_pct — чистое отношение цен входа/выхода, от размера капитала не зависит.
Значит масштабирование НЕ меняет ни один % доходности, ни одну сделку, ни
рейтинг монет между собой — просто пересчитывает то же самое в других
единицах. position (открытые позиции) не трогается вообще: там только цены
(entry_price, trailing_stop, stop_distance), не суммы.

Выбывшие из корзины монеты (TRXUSDT, CROUSDT — оставлены только чтобы
закрыть историю/позицию, см. load_state) масштабируются ТЕМ ЖЕ общим
коэффициентом, что и активные — иначе итоговая сумма была бы не ровно $300,
а $300 + их небольшой остаток (у trend v1 — $313.04, у meanrev v1 — $310.00),
а пользователь просил именно "везде по 300, не больше". В load_state (код
бота) они по-прежнему не участвуют в перерасчёте при появлении новых
активных монет — это чисто разовая правка сегодняшних файлов, не меняет
дальнейшую логику работы бота с выбывшими монетами.

Запуск: python3 rescale_to_300.py [--dry-run]
"""
from __future__ import annotations

import json
import sys

TARGET_TOTAL = 300.0

FILES = [
    "trend_paper_state.json",
    "trend_paper_state_v2.json",
    "paper_state.json",
    "paper_state_v2.json",
]


def main():
    dry_run = "--dry-run" in sys.argv
    basket = set(json.load(open("basket_100.json")))

    for path in FILES:
        with open(path) as f:
            data = json.load(f)

        active_syms = [s for s in data if s in basket]
        retired_syms = [s for s in data if s not in basket]
        current_total = sum(data[s]["starting_capital"] for s in data)
        k = TARGET_TOTAL / current_total if current_total else 1.0

        for s in data:
            c = data[s]
            c["starting_capital"] *= k
            c["balance_usdt"] *= k
            for t in c.get("trades", []):
                if "pnl_usdt" in t:
                    t["pnl_usdt"] *= k
                if "balance_after" in t:
                    t["balance_after"] *= k

        new_total = sum(data[s]["starting_capital"] for s in data)
        retired_total = sum(data[s]["starting_capital"] for s in retired_syms)
        print(f"{path}: {current_total:.2f} -> {new_total:.2f} "
              f"(k={k:.6f}, активных={len(active_syms)}, "
              f"выбывшие тоже смасштабированы: {retired_syms or '—'} -> ${retired_total:.2f})")

        if not dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    if dry_run:
        print("\n--dry-run: файлы не изменены")
    else:
        print("\nФайлы обновлены")


if __name__ == "__main__":
    main()
