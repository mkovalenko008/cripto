"""
Тонкий клиент для Bitget API v2 (Spot).
Схема подписи: HMAC-SHA256(secret, timestamp+method+requestPath+body) -> base64.
Документация: https://www.bitget.com/api-doc/spot/intro
ВАЖНО: биржи периодически меняют мелкие детали API (названия полей, поведение
market-ордеров и т.п.). Перед боевым запуском сверься с актуальной документацией
и сначала прогони бота в DRY_RUN, чтобы увидеть реальные ответы биржи в логах.
"""
import base64
import hashlib
import hmac
import json
import time
import logging

import requests

import config

log = logging.getLogger("bitget_client")


class BitgetError(Exception):
    pass


class BitgetClient:
    def __init__(self, api_key: str, secret_key: str, passphrase: str,
                 base_url: str = config.BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = base_url
        self.session = requests.Session()

    # ---------- подпись ----------

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(self.secret_key.encode(), prehash.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, request_path: str, body: str) -> dict:
        ts = self._timestamp()
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._sign(ts, method, request_path, body),
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP": ts,
            "locale": "en-US",
            "Content-Type": "application/json",
        }

    # ---------- низкоуровневые запросы ----------

    def _request(self, method: str, path: str, params: dict = None, body: dict = None,
                 signed: bool = True):
        params = params or {}
        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        request_path = path + query
        body_str = json.dumps(body) if body else ""
        url = self.base_url + request_path

        headers = {"Content-Type": "application/json"}
        if signed:
            headers = self._headers(method, request_path, body_str)

        resp = self.session.request(method, url, headers=headers,
                                     data=body_str if body else None, timeout=15)
        data = resp.json()
        if data.get("code") != "00000":
            raise BitgetError(f"{path} -> {data}")
        return data["data"]

    # ---------- публичные методы ----------

    def get_candles(self, symbol: str = None, granularity: str = None, limit: int = 100):
        """Возвращает список свечей [{open, high, low, close, ts}, ...] от старой к новой."""
        symbol = symbol or config.SYMBOL
        granularity = granularity or config.CANDLE_GRANULARITY
        raw = self._request("GET", "/api/v2/spot/market/candles", params={
            "symbol": symbol, "granularity": granularity, "limit": limit,
        }, signed=False)
        candles = []
        for row in raw:
            # формат ответа: [ts, open, high, low, close, baseVol, quoteVol, ...]
            candles.append({
                "ts": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            })
        return candles

    def get_history_candles(self, symbol: str = None, granularity: str = None,
                             end_time_ms: int = None, limit: int = 200):
        """
        Постраничная выгрузка истории свечей глубже, чем отдаёт /candles.
        endTime задаёт правую границу окна (мс, UTC), отдаёт до `limit` свечей
        от старой к новой, заканчивая на endTime. Чтобы уйти глубже в историю,
        вызови повторно с endTime = ts самой старой полученной свечи.
        """
        symbol = symbol or config.SYMBOL
        granularity = granularity or config.CANDLE_GRANULARITY
        params = {"symbol": symbol, "granularity": granularity, "limit": limit}
        if end_time_ms is not None:
            params["endTime"] = str(end_time_ms)
        raw = self._request("GET", "/api/v2/spot/market/history-candles",
                             params=params, signed=False)
        candles = []
        for row in raw:
            candles.append({
                "ts": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            })
        return candles

    def get_futures_history_candles(self, symbol: str, granularity: str = "1H",
                                     end_time_ms: int = None, limit: int = 200,
                                     product_type: str = "usdt-futures"):
        """
        История свечей USDT-M бессрочных фьючерсов (с объёмом). Пагинация
        так же, как get_history_candles — вызывай повторно с endTime = ts
        самой старой полученной свечи.
        """
        params = {"symbol": symbol, "productType": product_type,
                  "granularity": granularity, "limit": limit}
        if end_time_ms is not None:
            params["endTime"] = str(end_time_ms)
        raw = self._request("GET", "/api/v2/mix/market/history-candles",
                             params=params, signed=False)
        candles = []
        for row in raw:
            candles.append({
                "ts": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return candles

    def get_funding_rate_history(self, symbol: str, page_no: int = 1, page_size: int = 100,
                                  product_type: str = "usdt-futures"):
        """История ставок финансирования (funding rate), от новых к старым."""
        raw = self._request("GET", "/api/v2/mix/market/history-fund-rate", params={
            "symbol": symbol, "productType": product_type,
            "pageSize": page_size, "pageNo": page_no,
        }, signed=False)
        return [{"ts": int(r["fundingTime"]), "rate": float(r["fundingRate"])} for r in raw]

    def get_balance(self, coin: str = "USDT") -> float:
        data = self._request("GET", "/api/v2/spot/account/assets", params={"coin": coin})
        for item in data:
            if item.get("coin") == coin:
                return float(item.get("available", 0))
        return 0.0

    def place_market_buy(self, size_usdt: float, symbol: str = None):
        """
        Рыночная покупка. size указывается в котируемой валюте (USDT).
        ПРОВЕРЬ на маленькой тестовой сделке перед тем, как включать DRY_RUN=False —
        биржи иногда меняют это поведение.
        """
        symbol = symbol or config.SYMBOL
        body = {
            "symbol": symbol,
            "side": "buy",
            "orderType": "market",
            "force": "gtc",
            "size": str(size_usdt),
            "clientOid": f"bot_{int(time.time() * 1000)}",
        }
        return self._request("POST", "/api/v2/spot/trade/place-order", body=body)

    def place_market_sell(self, size_base: float, symbol: str = None):
        """
        Рыночная продажа. size указывается в базовой валюте (например ETH), а не в USDT.
        ПРОВЕРЬ на маленькой тестовой сделке перед тем, как включать DRY_RUN=False.
        """
        symbol = symbol or config.SYMBOL
        body = {
            "symbol": symbol,
            "side": "sell",
            "orderType": "market",
            "force": "gtc",
            "size": str(size_base),
            "clientOid": f"bot_{int(time.time() * 1000)}",
        }
        return self._request("POST", "/api/v2/spot/trade/place-order", body=body)
