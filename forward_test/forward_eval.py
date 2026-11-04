"""
Прямая (forward) проверка гипотезы H1 на данных, которых не было на момент заморозки правил.
Скрипт БЕЗ состояния: каждый запуск заново скачивает свечи Binance (публичные данные, без ключей) и пересчитывает всё
с момента заморозки (FREEZE), поэтому пропущенные запуски GitHub Actions ничего не ломают.
Правила - в PREREGISTRATION.md. Никаких ордеров, никаких ключей.

Запуск: python3 forward_eval.py
Переменные окружения (для отладки): FREEZE_TS_MS - другое время заморозки; OUT_DIR - куда писать отчёт.
"""
import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import absorb_lib as L
import absorb2_lib as L2
import absorb3_lib as L3
import absorb4_lib as L4

HOUR, DAY = 3600_000, 86_400_000
FREEZE_DEFAULT = 1793836800000          # 2026-11-05T00:00:00Z
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"]
ALPHA = 0.025                           # 2 основные гипотезы (P1, P2): одностор. p < 0.025
MAX_SE_BPS = 25.0                       # шире этого - решение не принимаем ("недостаточно данных")
CHECK_DAYS = (180, 365)                 # даты формальной проверки


def fetch_symbol(sym, start_ms, now_ms, hosts_state):
    rows, start, delay = [], start_ms, 2
    while True:
        r = None
        for _ in range(4):
            host = hosts_state["host"]
            try:
                r = requests.get(host + "/api/v3/klines", params={"symbol": sym, "interval": "1h", "startTime": start,
                                                                   "limit": 1000}, timeout=30)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                break
            if r is not None and r.status_code in (418, 429):
                time.sleep(min(int(r.headers.get("Retry-After", delay)), 60))
            elif r is None or r.status_code in (403, 451):
                # хост недоступен с этого сервера - пробуем следующий
                i = HOSTS.index(host)
                hosts_state["host"] = HOSTS[(i + 1) % len(HOSTS)]
            time.sleep(delay)
            r = None
        if r is None:
            return None
        data = r.json()
        if not data:
            break
        rows.extend(data)
        start = data[-1][0] + HOUR
        if len(data) < 1000 or start + HOUR > now_ms:
            break
    rows = [x for x in rows if x[0] + HOUR <= now_ms]
    return rows


def save_npz(path, rows):
    a = np.array(rows, dtype=object)
    f = lambda i: a[:, i].astype(np.float64)
    np.savez_compressed(path, ts=a[:, 0].astype(np.int64), open=f(1), high=f(2), low=f(3), close=f(4),
                        qv=f(7), tbq=f(10), ntr=f(8))


def md_row(name, r):
    if not r["ok"]:
        return f"| {name} | пока мало событий (N={r['n_ev']}) | | | | | |"
    return (f"| {name} | {r['n_ev']:,} | {r['diff_bps']:+.1f} | {r['se']:.1f} | {r['p_one']:.3f} | "
            f"{r['hit_ev']*100:.0f}% / {r['hit_ct']*100:.0f}% | {r['gross_raw_bps']:+.1f} |")


def main():
    now_ms = int(time.time() * 1000)
    freeze = int(os.environ.get("FREEZE_TS_MS", FREEZE_DEFAULT))
    out_dir = os.environ.get("OUT_DIR", HERE)
    os.makedirs(out_dir, exist_ok=True)
    days = (now_ms - freeze) / DAY
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now_ms / 1000))
    fz_txt = time.strftime("%Y-%m-%d", time.gmtime(freeze / 1000))

    if days < 1.5:
        probe_state = {"host": HOSTS[0]}
        probe = fetch_symbol("BTCUSDT", now_ms - 6 * HOUR, now_ms, probe_state)
        conn = (f"Связь с биржей с этого сервера: ОК (хост {probe_state['host']}, получено {len(probe)} свечей)."
                if probe else "Связь с биржей с этого сервера: НЕТ — биржа недоступна, проверка работать не будет.")
        txt = (f"# Прямая проверка H1: отчёт от {stamp}\n\nСбор ещё не начался или данных слишком мало: правила заморожены на "
               f"{fz_txt} 00:00 UTC, прошло {max(days, 0):.1f} суток. Первые числа появятся через пару недель.\n\n{conn}\n")
        open(os.path.join(out_dir, "FORWARD_REPORT.md"), "w").write(txt)
        print(txt)
        return 0 if probe else 1

    uni = json.load(open(os.path.join(HERE, "universe_203.json")))
    tmp = tempfile.mkdtemp()
    hosts_state = {"host": HOSTS[0]}
    start_ms = freeze - 10 * DAY

    def work(c):
        rows = fetch_symbol(c["binance_symbol"], start_ms, now_ms, hosts_state)
        if rows is None or len(rows) < 200:
            return c["binance_symbol"], 0
        save_npz(os.path.join(tmp, c["binance_symbol"] + ".npz"), rows)
        return c["binance_symbol"], len(rows)

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(work, uni))
    got = [s for s, n in res if n]
    print(f"скачано монет: {len(got)} из {len(uni)}; хост: {hosts_state['host']}", flush=True)
    if len(got) < 0.7 * len(uni):
        txt = (f"# Прямая проверка H1: отчёт от {stamp}\n\nНЕ УДАЛОСЬ скачать данные: получено {len(got)} монет из {len(uni)} "
               f"(возможно, биржа блокирует сервер GitHub). Числа не обновлены.\n")
        open(os.path.join(out_dir, "FORWARD_REPORT.md"), "w").write(txt)
        print(txt)
        return 1

    names, tms, O, C, QV, TBQ = L.load_matrices(tmp)
    feat = L.build_features(O, C, QV, TBQ, processes=1)
    pm = tms >= freeze
    H = L4.load_high(tmp, names, tms)
    sig = L4.hourly_sigma(O, C)
    le, _ = L4.short_with_stop(O, H, C, sig, 24, 2.0)
    L4.add_stop_variant(feat, ("S", 24, 2.0), le)

    ev1, ct1, d1 = L2.hyp_masks2(feat, "H1", 8.0, 0.02)
    P1 = L3.evaluate3(feat, tms, ev1, ct1, d1, 24, pm, L3.BLOCK_H)
    P2 = L3.evaluate3(feat, tms, ev1, ct1, d1, ("S", 24, 2.0), pm, L3.BLOCK_H)
    ev4, ct4, d4 = L2.hyp_masks2(feat, "H4", 2.0, 0.03)
    D1 = L3.evaluate3(feat, tms, ev4, ct4, d4, 2, pm, L3.BLOCK_H)

    def verdict(r):
        if not r["ok"]:
            return "данных пока недостаточно"
        if days < CHECK_DAYS[0]:
            return "промежуточно (формальная проверка - позже)"
        if r["se"] > MAX_SE_BPS:
            return "ПРОВЕРКА ПРОВЕДЕНА: данных недостаточно, чтобы что-либо утверждать (SE слишком велика)"
        return ("ПОДТВЕРЖДЕНО (diff>0, p<%.3f)" % ALPHA) if (r["diff_bps"] > 0 and r["p_one"] < ALPHA) else "НЕ подтверждено"

    lines = [f"# Прямая проверка H1: отчёт от {stamp}", "",
             f"Правила заморожены {fz_txt} 00:00 UTC (см. PREREGISTRATION.md). Прошло {days:.0f} суток. Данные: Binance spot 1H, {len(got)} монет.",
             "Это ПРОМЕЖУТОЧНЫЕ числа, пока не наступит формальная проверка (через %s суток после заморозки)." % " и ".join(map(str, CHECK_DAYS)),
             "", "| Проверка | Событий | Разница с контролем, bps | SE, bps | p (одностор.) | Успех: события / контроль | Валовая сделка, bps |",
             "|---|---|---|---|---|---|---|",
             md_row("P1: H1 шорт, удержание 24 ч, без стопа", P1),
             md_row("P2: H1 шорт, удержание 24 ч, стоп k=2", P2),
             md_row("справочно: H4 (R=2, U=3%), удержание 2 ч", D1), "",
             f"- P1: {verdict(P1)}", f"- P2: {verdict(P2)}", ""]
    for nm, r in (("P1", P1), ("P2", P2)):
        if r["ok"]:
            lines.append(f"- {nm}: минимальный обнаружимый эффект сейчас около {2.5*r['se']:.0f} bps (2.5 x SE). Всё, что меньше, "
                         f"эта проверка отличить от шума не может; издержки сделки 12-20 bps.")
    lines += ["", "Положительная «Разница» = цена пошла в сторону гипотезы (вниз после всплеска покупок) сильнее, чем после обычного объёма. "
              "«Валовая сделка» - средний результат шорта до комиссий без вычета рыночного движения.", ""]
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "FORWARD_REPORT.md"), "w").write(txt)
    hist = os.path.join(out_dir, "history.csv")
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "days", "P1_n", "P1_diff", "P1_se", "P1_p", "P2_n", "P2_diff", "P2_se", "P2_p", "D1_n", "D1_diff", "D1_se", "D1_p"])
        row = [stamp, f"{days:.1f}"]
        for r in (P1, P2, D1):
            row += [r["n_ev"]] + ([f"{r['diff_bps']:.2f}", f"{r['se']:.2f}", f"{r['p_one']:.4f}"] if r["ok"] else ["", "", ""])
        w.writerow(row)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
