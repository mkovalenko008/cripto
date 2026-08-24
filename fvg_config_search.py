"""
Честная проверка гипотезы "вход на ретесте FVG-зоны" — той же дисциплиной,
что и trend_config_search.py: подбор конфигурации СТРОГО по train (первые
70% истории каждой монеты), критерий — широта (% монет в плюсе), одна
валидация на test без донастройки. Отдельный скрипт-исследование, живого
бота не трогает и не меняет.

Данные — тот же локальный кэш data/{SYMBOL}_1H.json (23 монеты, ~2 года
1H-свечей с Bitget Spot), которым уже пользовался trend_config_search.py.
"""
import json
import os

from fvg_backtest import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_TRADES = 15  # у FVG сделок заметно меньше, чем у trend/bb — порог ниже, но не нулевой
TRAIN_FRAC = 0.7
MIN_COVERAGE = 12  # нужно набрать надёжную выборку минимум на стольки монетах из 23

REWARD_R_VARIANTS = [1.5, 2.0, 3.0]
MAX_AGE_VARIANTS = [100, 200]  # баров (1H) — ~4 и ~8 суток
MIN_WIDTH_VARIANTS = [0.0, 0.15, 0.3]  # % от цены — 0.0 без фильтра, иначе отсекаем совсем
                                        # крохотные гэпы-шум (в пилотном прогоне на ETH было
                                        # много зон шириной 0.03-0.05% — это не "имбаланс", а шум)

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


def evaluate_config(candles_by_symbol: dict, part: str, config: dict) -> dict:
    idx = 0 if part == "train" else 1
    per_symbol = {}
    for symbol, halves in candles_by_symbol.items():
        candles = halves[idx]
        result = run_backtest(candles, **config)
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
        "per_symbol": per_symbol,
        "n_reliable": len(reliable),
        "n_positive": len(positive),
        "breadth_pct": breadth,
        "median_return": median_return,
        "avg_return": avg_return,
    }


def main():
    import time
    candles_by_symbol = load_all_candles()
    print(f"Загружено монет: {len(candles_by_symbol)}\n", flush=True)

    total = len(REWARD_R_VARIANTS) * len(MAX_AGE_VARIANTS) * len(MIN_WIDTH_VARIANTS)
    t_start = time.time()
    grid_results = []
    done = 0
    for reward_r in REWARD_R_VARIANTS:
        for max_age in MAX_AGE_VARIANTS:
            for min_width in MIN_WIDTH_VARIANTS:
                config = dict(max_zone_age_bars=max_age, reward_r=reward_r,
                              max_holding_bars=100, fee_pct_per_side=0.1,
                              long_only=False, min_zone_width_pct=min_width)
                res = evaluate_config(candles_by_symbol, "train", config)
                grid_results.append((config, res))
                done += 1
                elapsed = time.time() - t_start
                print(f"[{done}/{total}] R={reward_r} возраст<={max_age}бар -> "
                      f"широта={res['breadth_pct']:.0f}% медиана={res['median_return']:+.2f}% "
                      f"(прошло {elapsed:.0f}с)", flush=True)

    grid_results.sort(key=lambda cr: (cr[1]["breadth_pct"], cr[1]["median_return"]), reverse=True)

    header = f"{'R':>5}{'Возраст':>9}{'Монет':>7}{'ВПлюсе':>8}{'Широта%':>9}{'Медиана%':>10}{'Среднее%':>10}"
    lines = ["=" * 90, "ПОИСК ПО TRAIN (FVG-ретест, сортировка по широте)", "=" * 90, header, "-" * len(header)]
    for config, res in grid_results:
        flag = "" if res["n_reliable"] >= MIN_COVERAGE else "  <- ПОКРЫТИЕ СЛИШКОМ МАЛО, НЕ УЧАСТВУЕТ В ВЫБОРЕ"
        lines.append(
            f"{config['reward_r']:>5.1f}{config['max_zone_age_bars']:>9}"
            f"{res['n_reliable']:>7}{res['n_positive']:>8}{res['breadth_pct']:>9.1f}"
            f"{res['median_return']:>10.2f}{res['avg_return']:>10.2f}{flag}"
        )

    eligible = [cr for cr in grid_results if cr[1]["n_reliable"] >= MIN_COVERAGE]
    if not eligible:
        lines.append(f"\nНи одна конфигурация не набрала надёжную выборку на >= {MIN_COVERAGE} монетах — "
                     f"сделок у FVG-ретеста в принципе слишком мало для честного вывода на этих данных.")
        text = "\n".join(lines)
        print(text)
        out_path = os.path.join(os.path.dirname(__file__), "fvg_config_search_report.txt")
        with open(out_path, "w") as f:
            f.write(text)
        return
    best_config, best_train_res = eligible[0]
    lines.append(f"\n(Из {len(grid_results)} конфигураций только {len(eligible)} набрали покрытие "
                 f">= {MIN_COVERAGE} монет — выбор идёт только среди них)")
    lines.append("")
    lines.append(f"Лучшая по TRAIN (широта): R={best_config['reward_r']}, "
                 f"возраст зоны<={best_config['max_zone_age_bars']} бар — "
                 f"{best_train_res['n_positive']}/{best_train_res['n_reliable']} монет в плюсе "
                 f"({best_train_res['breadth_pct']:.1f}%), медиана {best_train_res['median_return']:+.2f}%")

    test_res = evaluate_config(candles_by_symbol, "test", best_config)
    lines.append("")
    lines.append("=" * 90)
    lines.append("ТА ЖЕ КОНФИГУРАЦИЯ НА TEST (без дальнейшей подгонки)")
    lines.append("=" * 90)
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
    out_path = os.path.join(os.path.dirname(__file__), "fvg_config_search_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
