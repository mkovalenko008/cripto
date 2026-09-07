"""
Собирает корзину монет для paper-ботов: топ по капитализации среди тех, что
одновременно листингованы на Coinbase и имеют USDT-пару в спот-секции Bitget.

Раньше списки SYMBOLS собирались руками и в коде оставались просто как
константа — воспроизвести или обновить их было нечем. Этот скрипт делает то
же самое, но повторяемо: три публичных источника, никаких ключей.

  CoinGecko   — рейтинг по рыночной капитализации
  Coinbase    — что реально листингуется на бирже (публичный Exchange API)
  Bitget      — что реально торгуется парой к USDT на споте (данные берём отсюда)

Стейблкоины и обёрнутые/ставочные производные (WBTC, stETH и т.п.) исключаются:
у первых нет движения, ради которого работают обе стратегии, вторые дублируют
базовый актив и ломают диверсификацию корзины.

Запуск:  python3 build_basket.py [сколько монет, по умолчанию 100]
"""
from __future__ import annotations

import json
import sys
import time

import requests   # тот же клиент, что и у ботов: несёт свои корневые сертификаты

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Стейблы и обёртки/производные — по тикеру. Держим отдельным списком, чтобы
# было видно, что именно отсеиваем и почему.
STABLE = {
    "USDT", "USDC", "DAI", "FDUSD", "USDE", "USDS", "PYUSD", "TUSD", "USDP",
    "USDD", "GUSD", "EURC", "RLUSD", "BUSD", "USD1", "USDG", "USDF", "LUSD",
}
WRAPPED = {
    "WBTC", "WETH", "WBETH", "STETH", "WSTETH", "RETH", "CBBTC", "CBETH",
    "WEETH", "EZETH", "METH", "RSETH", "SOLVBTC", "LBTC", "BSC-USD", "WBNB",
    "JITOSOL", "MSOL", "JUPSOL", "BNSOL", "SUSDE", "SUSDS", "WHYPE", "WSOL",
}


def get_json(url: str, tries: int = 3):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": "basket-builder"}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  повтор после ошибки: {e}")
            time.sleep(5)


def coingecko_ranking(pages: int = 3) -> list[dict]:
    """Топ монет по капитализации, по 250 на страницу."""
    out = []
    for page in range(1, pages + 1):
        url = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
               f"&order=market_cap_desc&per_page=250&page={page}&sparkline=false")
        out += get_json(url)
        time.sleep(2)   # бесплатный тариф CoinGecko не любит частых запросов
    return out


def coinbase_symbols() -> set[str]:
    """Базовые активы, торгуемые на Coinbase (любая котируемая валюта)."""
    products = get_json("https://api.exchange.coinbase.com/products")
    return {
        p["base_currency"].upper()
        for p in products
        if not p.get("trading_disabled") and p.get("status") == "online"
    }


def bitget_usdt_symbols() -> set[str]:
    """Базовые активы, у которых на споте Bitget есть живая пара к USDT."""
    data = get_json("https://api.bitget.com/api/v2/spot/public/symbols")
    out = set()
    for s in data.get("data", []):
        if s.get("quoteCoin", "").upper() == "USDT" and s.get("status") == "online":
            out.add(s["baseCoin"].upper())
    return out


def main():
    print("Тяну рейтинг CoinGecko...")
    ranking = coingecko_ranking()
    print(f"  монет в рейтинге: {len(ranking)}")

    print("Тяну листинги Coinbase...")
    cb = coinbase_symbols()
    print(f"  активов на Coinbase: {len(cb)}")

    print("Тяну спотовые пары Bitget...")
    bg = bitget_usdt_symbols()
    print(f"  активов с USDT-парой на Bitget: {len(bg)}")

    picked, skipped_reason = [], {}
    for coin in ranking:
        sym = coin["symbol"].upper()
        if len(picked) >= TARGET:
            break
        if sym in STABLE:
            skipped_reason.setdefault("стейблкоин", []).append(sym); continue
        if sym in WRAPPED:
            skipped_reason.setdefault("обёртка/производная", []).append(sym); continue
        if sym not in cb:
            skipped_reason.setdefault("нет на Coinbase", []).append(sym); continue
        if sym not in bg:
            skipped_reason.setdefault("нет пары USDT на Bitget", []).append(sym); continue
        if any(p == sym for p, _ in picked):
            continue
        picked.append((sym, coin.get("market_cap") or 0))

    print(f"\nОтобрано монет: {len(picked)} (цель {TARGET})")
    for reason, syms in skipped_reason.items():
        print(f"  отсеяно «{reason}»: {len(syms)}")

    symbols = [f"{s}USDT" for s, _ in picked]
    print("\nSYMBOLS = [")
    for i in range(0, len(symbols), 6):
        print("    " + ", ".join(f'"{s}"' for s in symbols[i:i + 6]) + ",")
    print("]")

    with open("basket_100.json", "w") as f:
        json.dump(symbols, f, indent=1)
    print(f"\nСписок сохранён в basket_100.json ({len(symbols)} шт.)")


if __name__ == "__main__":
    main()
