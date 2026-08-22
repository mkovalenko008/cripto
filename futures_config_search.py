"""
Та же дисциплина, что в trend_config_search.py (подбор по train, широта =
% монет в плюсе, минимальное покрытие для честности, одна проверка на test
без дальнейшей подгонки), но на фьючерсах и с новыми осями: включён ли
фильтр по объёму, включён ли фильтр по funding rate. Сравниваем базовую
версию (без новых фильтров) с версиями, где они добавлены — чтобы увидеть,
дают ли объём/funding реальный прирост широты, а не просто красивую цифру
на паре монет.
"""
import json
import os

from futures_backtest import run_backtest, BacktestResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_TRADES = 15
MIN_COVERAGE = 8  # из 12 монет
TRAIN_FRAC = 0.7

ATR_VARIANTS = [1.5, 2.0, 3.0]
STD_VARIANTS = [2.0, 2.5]
FILTER_VARIANTS = [
    ("база (без объёма/funding)", dict(use_volume_filter=False, use_funding_filter=False)),
    ("+ объём", dict(use_volume_filter=True, use_funding_filter=False)),
    ("+ funding", dict(use_volume_filter=False, use_funding_filter=True)),
    ("+ объём + funding", dict(use_volume_filter=True, use_funding_filter=True)),
]

SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT", "DOGEUSDT",
           "LINKUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "ZECUSDT", "TRUMPUSDT"]


def load_all_candles():
    data = {}
    for symbol in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{symbol}_FUT_1H.json")
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
        result = run_backtest(halves[idx], **config)
        per_symbol[symbol] = {
            "trades": result.total_trades,
            "return_pct": result.total_return_pct,
            "reliable": result.total_trades >= MIN_TRADES,
        }
    reliable = [v for v in per_symbol.values() if v["reliable"]]
    positive = [v for v in reliable if v["return_pct"] > 0]
    breadth = len(positive) / len(reliable) * 100 if reliable else 0.0
    median_return = sorted(v["return_pct"] for v in reliable)[len(reliable) // 2] if reliable else 0.0
    avg_return = sum(v["return_pct"] for v in reliable) / len(reliable) if reliable else 0.0
    return {"per_symbol": per_symbol, "n_reliable": len(reliable), "n_positive": len(positive),
            "breadth_pct": breadth, "median_return": median_return, "avg_return": avg_return}


def main():
    candles_by_symbol = load_all_candles()
    print(f"Загружено монет: {len(candles_by_symbol)}\n")

    grid_results = []
    for filt_label, filt_kwargs in FILTER_VARIANTS:
        for atr_mult in ATR_VARIANTS:
            for std in STD_VARIANTS:
                config = dict(period=20, num_std=std, adx_period=14, adx_threshold=25.0,
                              atr_period=14, atr_stop_mult=atr_mult, max_holding_bars=100,
                              fee_pct_per_side=0.06, **filt_kwargs)
                res = evaluate_config(candles_by_symbol, "train", config)
                grid_results.append((filt_label, config, res))

    grid_results.sort(key=lambda cr: (cr[2]["breadth_pct"], cr[2]["median_return"]), reverse=True)

    header = f"{'Фильтр':<24}{'ATRx':>6}{'std':>6}{'Монет':>7}{'ВПлюсе':>8}{'Широта%':>9}{'Медиана%':>10}{'Среднее%':>10}"
    lines = ["=" * 100, "ПОИСК ПО TRAIN (фьючерсы, сортировка по широте)", "=" * 100, header, "-" * len(header)]
    for filt_label, config, res in grid_results:
        flag = "" if res["n_reliable"] >= MIN_COVERAGE else "  <- МАЛО ПОКРЫТИЕ"
        lines.append(
            f"{filt_label:<24}{config['atr_stop_mult']:>6.1f}{config['num_std']:>6.1f}"
            f"{res['n_reliable']:>7}{res['n_positive']:>8}{res['breadth_pct']:>9.1f}"
            f"{res['median_return']:>10.2f}{res['avg_return']:>10.2f}{flag}"
        )

    eligible = [cr for cr in grid_results if cr[2]["n_reliable"] >= MIN_COVERAGE]
    if not eligible:
        lines.append(f"\nНи одна конфигурация не набрала покрытие >= {MIN_COVERAGE} монет.")
        print("\n".join(lines))
        return

    best_label, best_config, best_train_res = eligible[0]
    lines.append(f"\n(Из {len(grid_results)} конфигураций {len(eligible)} с покрытием >= {MIN_COVERAGE})")
    lines.append(f"\nЛучшая по TRAIN: {best_label}, ATRx{best_config['atr_stop_mult']}, "
                 f"num_std={best_config['num_std']} — {best_train_res['n_positive']}/{best_train_res['n_reliable']} "
                 f"монет в плюсе ({best_train_res['breadth_pct']:.1f}%), медиана {best_train_res['median_return']:+.2f}%")

    test_res = evaluate_config(candles_by_symbol, "test", best_config)
    lines.append("\n" + "=" * 100)
    lines.append("ТА ЖЕ КОНФИГУРАЦИЯ НА TEST")
    lines.append("=" * 100)
    lines.append(f"{'Символ':<12}{'Сделок':>8}{'Итог%':>10}  Надёжн.")
    lines.append("-" * 40)
    for symbol, v in sorted(test_res["per_symbol"].items(), key=lambda kv: kv[1]["return_pct"], reverse=True):
        flag = "OK" if v["reliable"] else "мало"
        lines.append(f"{symbol:<12}{v['trades']:>8}{v['return_pct']:>+10.2f}  {flag}")
    lines.append(f"\nНа TEST: {test_res['n_positive']}/{test_res['n_reliable']} монет в плюсе "
                 f"({test_res['breadth_pct']:.1f}%), медиана {test_res['median_return']:+.2f}%, "
                 f"среднее {test_res['avg_return']:+.2f}%")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "futures_config_search_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
