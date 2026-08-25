"""
Честная проверка "фундаментальной" стратегии (индекс страха/жадности как
главный триггер + механические линии тренда/поддержки-сопротивления как
единственный технический фильтр) — той же дисциплиной train/test, что и
trend_config_search.py/fvg_config_search.py. Живого бота не трогает.

Новостной компонент (Coinbase/Investing.com) сюда НЕ включён — честно
проверить его на 2 годах истории нельзя (нет архива заголовков с
таймстемпами), он идёт только в живой бот отдельным фильтром, без claim'а
о валидации на истории.
"""
import json
import os

from fundamental_backtest import run_backtest
from fundamental_strategy import FearGreedIndex

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_TRADES = 8  # у "терпеливой" стратегии сделок заметно меньше, чем у trend/fvg
TRAIN_FRAC = 0.7
MIN_COVERAGE = 12

# Раунд 2 — расширяем не столько пороги, сколько структуру: чувствительность
# линий (pivot_lookback) и срок удержания, которые в раунде 1 были
# зафиксированы произвольно. touch_pct=1.0% зафиксирован по чистому TRAIN-
# наблюдению раунда 1 (1.0% стабильно обгонял 2.0% на каждом пороге страха) —
# это не подглядывание в test, тот вывод был виден только по train-таблице.
FEAR_VARIANTS = [15, 20, 25]
PIVOT_LOOKBACK_VARIANTS = [3, 5, 8]
MAX_HOLDING_VARIANTS = [75, 250]
REWARD_R_VARIANTS = [2.5, 3.5]
TOUCH_PCT = 1.0

SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "HYPEUSDT",
    "BTCUSDT", "XRPUSDT", "DOGEUSDT", "ZECUSDT", "SUIUSDT", "PEPEUSDT",
    "ENAUSDT", "ONDOUSDT", "TRUMPUSDT", "LINKUSDT", "BNBUSDT", "UNIUSDT",
    "ADAUSDT", "LTCUSDT", "NEARUSDT", "XLMUSDT", "AAVEUSDT", "AVAXUSDT",
    "TRXUSDT", "BCHUSDT",
]


def load_all_candles():
    data = {}
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_1H.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            candles = json.load(f)
        split_i = int(len(candles) * TRAIN_FRAC)
        data[symbol] = (candles[:split_i], candles[split_i:])
    return data


def evaluate_config(candles_by_symbol: dict, fgi: FearGreedIndex, part: str, config: dict) -> dict:
    idx = 0 if part == "train" else 1
    per_symbol = {}
    for symbol, halves in candles_by_symbol.items():
        candles = halves[idx]
        result = run_backtest(candles, fgi, **config)
        per_symbol[symbol] = {
            "trades": result.total_trades,
            "return_pct": result.total_return_pct,
            "win_rate": result.win_rate * 100,
            "reliable": result.total_trades >= MIN_TRADES,
        }

    reliable = [v for v in per_symbol.values() if v["reliable"]]
    positive = [v for v in reliable if v["return_pct"] > 0]
    breadth = len(positive) / len(reliable) * 100 if reliable else 0.0
    median_return = sorted(v["return_pct"] for v in reliable)[len(reliable) // 2] if reliable else 0.0
    avg_return = sum(v["return_pct"] for v in reliable) / len(reliable) if reliable else 0.0

    return {
        "per_symbol": per_symbol, "n_reliable": len(reliable), "n_positive": len(positive),
        "breadth_pct": breadth, "median_return": median_return, "avg_return": avg_return,
    }


def main():
    import time
    fgi = FearGreedIndex(json.load(open(os.path.join(DATA_DIR, "fear_greed_index.json"))))
    candles_by_symbol = load_all_candles()
    print(f"Загружено монет: {len(candles_by_symbol)}\n", flush=True)

    total = len(FEAR_VARIANTS) * len(PIVOT_LOOKBACK_VARIANTS) * len(MAX_HOLDING_VARIANTS) * len(REWARD_R_VARIANTS)
    t_start = time.time()
    grid_results = []
    done = 0
    for fear in FEAR_VARIANTS:
        greed = 100 - fear  # симметрично: страх<=20 <-> жадность>=80, и т.п.
        for pivot_lookback in PIVOT_LOOKBACK_VARIANTS:
            for max_holding in MAX_HOLDING_VARIANTS:
                for reward_r in REWARD_R_VARIANTS:
                    config = dict(pivot_lookback=pivot_lookback, swing_count=4, lookback_bars=200,
                                  fear_threshold=fear, greed_threshold=greed, touch_pct=TOUCH_PCT,
                                  atr_period=14, atr_stop_buffer_mult=0.5, reward_r=reward_r,
                                  max_holding_bars=max_holding, fee_pct_per_side=0.1, long_only=False)
                    res = evaluate_config(candles_by_symbol, fgi, "train", config)
                    grid_results.append((config, res))
                    done += 1
                    elapsed = time.time() - t_start
                    print(f"[{done}/{total}] страх<={fear} pivot={pivot_lookback} hold<={max_holding} R={reward_r} -> "
                          f"широта={res['breadth_pct']:.0f}% медиана={res['median_return']:+.2f}% "
                          f"(прошло {elapsed:.0f}с)", flush=True)

    grid_results.sort(key=lambda cr: (cr[1]["breadth_pct"], cr[1]["median_return"]), reverse=True)

    header = f"{'Страх':>6}{'Pivot':>6}{'Hold':>6}{'R':>5}{'Монет':>7}{'ВПлюсе':>8}{'Широта%':>9}{'Медиана%':>10}{'Среднее%':>10}"
    lines = ["=" * 95, "ПОИСК ПО TRAIN, РАУНД 2 (фундаментал: страх/жадность + линии, сортировка по широте)", "=" * 95,
              header, "-" * len(header)]
    for config, res in grid_results:
        flag = "" if res["n_reliable"] >= MIN_COVERAGE else "  <- ПОКРЫТИЕ СЛИШКОМ МАЛО, НЕ УЧАСТВУЕТ В ВЫБОРЕ"
        lines.append(
            f"{config['fear_threshold']:>6}{config['pivot_lookback']:>6}{config['max_holding_bars']:>6}"
            f"{config['reward_r']:>5.1f}{res['n_reliable']:>7}{res['n_positive']:>8}{res['breadth_pct']:>9.1f}"
            f"{res['median_return']:>10.2f}{res['avg_return']:>10.2f}{flag}"
        )

    eligible = [cr for cr in grid_results if cr[1]["n_reliable"] >= MIN_COVERAGE]
    if not eligible:
        lines.append(f"\nНи одна конфигурация не набрала надёжную выборку на >= {MIN_COVERAGE} монетах.")
        text = "\n".join(lines)
        print(text)
        out_path = os.path.join(os.path.dirname(__file__), "fundamental_config_search_report_round2.txt")
        with open(out_path, "w") as f:
            f.write(text)
        return
    best_config, best_train_res = eligible[0]
    lines.append(f"\n(Из {len(grid_results)} конфигураций только {len(eligible)} набрали покрытие "
                 f">= {MIN_COVERAGE} монет — выбор идёт только среди них)")
    lines.append("")
    lines.append(f"Лучшая по TRAIN (широта): страх<={best_config['fear_threshold']}, "
                 f"жадность>={best_config['greed_threshold']}, pivot_lookback={best_config['pivot_lookback']}, "
                 f"hold<={best_config['max_holding_bars']}бар, touch={best_config['touch_pct']}%, "
                 f"R={best_config['reward_r']} — "
                 f"{best_train_res['n_positive']}/{best_train_res['n_reliable']} монет в плюсе "
                 f"({best_train_res['breadth_pct']:.1f}%), медиана {best_train_res['median_return']:+.2f}%")

    test_res = evaluate_config(candles_by_symbol, fgi, "test", best_config)
    lines.append("")
    lines.append("=" * 95)
    lines.append("ТА ЖЕ КОНФИГУРАЦИЯ НА TEST (без дальнейшей подгонки)")
    lines.append("=" * 95)
    lines.append(f"{'Символ':<12}{'Сделок':>8}{'Win%':>7}{'Итог%':>10}{'Надёжн.'}")
    lines.append("-" * 48)
    for symbol, v in sorted(test_res["per_symbol"].items(), key=lambda kv: kv[1]["return_pct"], reverse=True):
        flag = "OK" if v["reliable"] else "мало"
        lines.append(f"{symbol:<12}{v['trades']:>8}{v['win_rate']:>6.0f}%{v['return_pct']:>+10.2f}  {flag}")
    lines.append("")
    lines.append(f"На TEST: {test_res['n_positive']}/{test_res['n_reliable']} монет в плюсе "
                 f"({test_res['breadth_pct']:.1f}%), медиана {test_res['median_return']:+.2f}%, "
                 f"среднее {test_res['avg_return']:+.2f}%")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "fundamental_config_search_report_round2.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
