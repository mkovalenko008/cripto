"""
Синтетическая проверка v2: боковик vs тренд, три варианта стратегии —
без фильтров (наивная версия из первого теста), с ADX-фильтром режима,
с ADX + RSI-подтверждением (полная v2).
"""
import random

from bb_backtest import run_backtest

random.seed(7)
N = 1500


def make_candles(closes: list[float], intrabar_sigma: float = 8.0) -> list[dict]:
    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        hi = max(o, c) + abs(random.gauss(0, intrabar_sigma))
        lo = min(o, c) - abs(random.gauss(0, intrabar_sigma))
        candles.append({"open": o, "high": hi, "low": lo, "close": c})
    return candles


def simulate_ranging(start=77000.0, mean=77000.0, theta=0.05, sigma=25.0):
    price = start
    closes = [price]
    for _ in range(N):
        price += theta * (mean - price) + random.gauss(0, sigma)
        closes.append(price)
    return closes


def simulate_trending(start=77000.0, drift=6.0, sigma=25.0):
    price = start
    closes = [price]
    for _ in range(N):
        price += drift + random.gauss(0, sigma)
        closes.append(price)
    return closes


def main():
    for market_label, closes in [("БОКОВИК", simulate_ranging()), ("ТРЕНД", simulate_trending())]:
        candles = make_candles(closes)
        print(f"\n########## Рынок: {market_label} ##########")

        variants = [
            ("v1: без фильтров (голое касание полосы)", dict(use_adx_filter=False, use_rsi_confirmation=False)),
            ("v2a: + ADX-фильтр режима", dict(use_adx_filter=True, use_rsi_confirmation=False)),
            ("v2b: + ADX-фильтр + RSI-подтверждение", dict(use_adx_filter=True, use_rsi_confirmation=True)),
        ]
        for label, kwargs in variants:
            result = run_backtest(candles, **kwargs)
            print(f"\n-- {label} --")
            print(result.summary())


if __name__ == "__main__":
    main()
