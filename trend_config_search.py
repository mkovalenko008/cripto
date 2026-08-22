"""
Поиск конфигурации trend_strategy, которая держится не на одной монете, а
на БОЛЬШИНСТВЕ монет в корзине. В отличие от trend_basket_backtest.py (там
была одна зафиксированная конфигурация), здесь перебираем сетку конфигураций
— но подбор лучшей идёт СТРОГО по train (первые 70% периода каждой монеты),
а на test (последние 30%) валидируем один раз, без дальнейшей донастройки.

Критерий "лучшей" конфигурации — не средний результат (его легко утащить
вверх парой счастливых выбросов, как мы уже видели), а ШИРОТА: на какой доле
монет из корзины конфигурация даёт плюс. Это прямой ответ на вопрос "работает
почти на всём" против "работает только там, где повезло".
"""
import json
import os

from trend_backtest import run_backtest, BacktestResult

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIN_TRADES = 20  # на train-половине сделок меньше, чем на полном периоде — порог снижен пропорционально
TRAIN_FRAC = 0.7
MIN_COVERAGE = 15  # конфигурация участвует в ранжировании, только если набрала
                    # надёжную (>=MIN_TRADES) выборку минимум на стольки монетах из 23 —
                    # иначе "широта" на 2 монетах из 23 может случайно выиграть у честных 8/23

ADX_VARIANTS = [20.0, 25.0, 30.0]
ATR_VARIANTS = [1.5, 2.0, 3.0]
STD_VARIANTS = [2.5]  # 2.5 стабильно был лучшим/средним по всей сессии — фиксируем, чтобы не утроять сетку впустую

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
    """part: 'train' или 'test' — индекс 0 или 1 в кортеже (train, test)."""
    idx = 0 if part == "train" else 1
    per_symbol = {}
    for symbol, halves in candles_by_symbol.items():
        candles = halves[idx]
        result = run_backtest(candles, **config)
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

    total = len(ADX_VARIANTS) * len(ATR_VARIANTS) * len(STD_VARIANTS)
    t_start = time.time()
    grid_results = []
    done = 0
    for adx in ADX_VARIANTS:
        for atr_mult in ATR_VARIANTS:
            for std in STD_VARIANTS:
                config = dict(period=20, num_std=std, adx_period=14, adx_threshold=adx,
                              atr_period=14, atr_stop_mult=atr_mult, max_holding_bars=100,
                              fee_pct_per_side=0.1)
                res = evaluate_config(candles_by_symbol, "train", config)
                grid_results.append((config, res))
                done += 1
                elapsed = time.time() - t_start
                print(f"[{done}/{total}] ADX>={adx} ATRx{atr_mult} std={std} -> "
                      f"широта={res['breadth_pct']:.0f}% медиана={res['median_return']:+.2f}% "
                      f"(прошло {elapsed:.0f}с)", flush=True)

    grid_results.sort(key=lambda cr: (cr[1]["breadth_pct"], cr[1]["median_return"]), reverse=True)

    header = f"{'ADX':>5}{'ATRx':>7}{'std':>6}{'Монет':>7}{'ВПлюсе':>8}{'Широта%':>9}{'Медиана%':>10}{'Среднее%':>10}"
    lines = ["=" * 90, "ПОИСК ПО TRAIN (сортировка по широте — % монет в плюсе)", "=" * 90, header, "-" * len(header)]
    for config, res in grid_results:
        flag = "" if res["n_reliable"] >= MIN_COVERAGE else "  <- ПОКРЫТИЕ СЛИШКОМ МАЛО, НЕ УЧАСТВУЕТ В ВЫБОРЕ"
        lines.append(
            f"{config['adx_threshold']:>5.0f}{config['atr_stop_mult']:>7.1f}{config['num_std']:>6.1f}"
            f"{res['n_reliable']:>7}{res['n_positive']:>8}{res['breadth_pct']:>9.1f}"
            f"{res['median_return']:>10.2f}{res['avg_return']:>10.2f}{flag}"
        )

    eligible = [cr for cr in grid_results if cr[1]["n_reliable"] >= MIN_COVERAGE]
    if not eligible:
        lines.append(f"\nНи одна конфигурация не набрала надёжную выборку на >= {MIN_COVERAGE} монетах — "
                     f"дальше смотреть нечего, сетку нужно менять (или порог надёжности).")
        text = "\n".join(lines)
        print(text)
        return
    best_config, best_train_res = eligible[0]
    lines.append(f"\n(Из {len(grid_results)} конфигураций только {len(eligible)} набрали покрытие "
                 f">= {MIN_COVERAGE} монет — выбор идёт только среди них)")
    lines.append("")
    lines.append(f"Лучшая по TRAIN (широта): ADX>={best_config['adx_threshold']:.0f}, "
                 f"ATRx{best_config['atr_stop_mult']}, num_std={best_config['num_std']} — "
                 f"{best_train_res['n_positive']}/{best_train_res['n_reliable']} монет в плюсе "
                 f"({best_train_res['breadth_pct']:.1f}%), медиана {best_train_res['median_return']:+.2f}%")

    test_res = evaluate_config(candles_by_symbol, "test", best_config)
    lines.append("")
    lines.append("=" * 90)
    lines.append("ТА ЖЕ КОНФИГУРАЦИЯ НА TEST (без дальнейшей подгонки)")
    lines.append("=" * 90)
    lines.append(f"{'Символ':<12}{'Сделок':>8}{'Итог%':>10}{'Надёжн.'}")
    lines.append("-" * 40)
    for symbol, v in sorted(test_res["per_symbol"].items(), key=lambda kv: kv[1]["return_pct"], reverse=True):
        flag = "OK" if v["reliable"] else "мало"
        lines.append(f"{symbol:<12}{v['trades']:>8}{v['return_pct']:>+10.2f}  {flag}")
    lines.append("")
    lines.append(f"На TEST: {test_res['n_positive']}/{test_res['n_reliable']} монет в плюсе "
                 f"({test_res['breadth_pct']:.1f}%), медиана {test_res['median_return']:+.2f}%, "
                 f"среднее {test_res['avg_return']:+.2f}%")

    text = "\n".join(lines)
    print(text)
    out_path = os.path.join(os.path.dirname(__file__), "trend_config_search_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n\nОтчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
