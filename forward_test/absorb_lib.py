"""
Библиотека для проверки гипотезы "поглощения" (absorption) на почасовых свечах.

ПРЕ-РЕГИСТРАЦИЯ (зафиксировано ДО просмотра результатов):

Бар t (час). Признаки считаются только по барам <= t (базовые уровни — медиана
по ПРЕДЫДУЩИМ 168 часам, текущий бар в них не входит).
  r_t   = close/open - 1                         (чистое движение цены внутри часа)
  rb_t  = taker_buy_quote_vol_t  / median(taker_buy_quote_vol[t-168..t-1])
  rs_t  = taker_sell_quote_vol_t / median(taker_sell_quote_vol[t-168..t-1])
  rv_t  = quote_vol_t / median(quote_vol[t-168..t-1])
Фильтр ликвидности: median(quote_vol, 168ч) >= 50 000 USDT/час.

Гипотезы (направление задано пользователем, а не выбрано по данным):
  H1  покупки-поглощение:  rb >= R  и  -0.5% <= r <= +U   -> дальше ВНИЗ
  H2  продажи-поглощение:  rs >= R  и  -U <= r <= +0.5%   -> дальше ВВЕРХ  (зеркало H1)
  H3  двусторонний всплеск: rv >= R и 0.2% <= |r| <= U    -> дальше ПРОТИВ движения бара
Сетка (единственная, без расширений): R in {3,5,8}, U in {2%,3%}.
Вход: открытие бара t+1. Выход: закрытие бара t+h. Основной горизонт h=1
("в следующий час"); h=2,4,24 — только описательно.
Рыночная нейтрализация: из доходности вычитается среднее по всем ликвидным
монетам за то же окно (иначе общерыночное движение часа маскируется под сигнал).
КОНТРОЛЬ: те же условия по цене, но НОРМАЛЬНЫЙ объём (ratio в [0.5, 1.5]).
Сигнал = (событие) - (контроль): так отделяется вклад ОБЪЁМА от вклада самой
формы цены.
Статистика: bootstrap по календарным дням (бары одного дня и разных монет
коррелированы), B=2000.
Время: train = до 2025-04-30, test = с 2025-05-01 (UTC); сутки у границы выкинуты.
Выбор конфигурации — только по train; test смотрим один раз.
"""
import glob
import os
import warnings
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HOUR_MS = 3_600_000
W = 168
MINV = 120
LIQ_MIN = 50_000.0
SPLIT_MS = 1746057600000  # 2025-05-01T00:00:00Z
HORIZONS = (1, 2, 4, 24)
R_GRID = (3.0, 5.0, 8.0)
U_GRID = (0.02, 0.03)
HYPS = ("H1", "H2", "H3")
MIN_MKT_COINS = 20


# --------------------------------------------------------------------------- данные
def load_matrices(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    loaded, t_min, t_max = [], None, None
    for f in files:
        z = np.load(f, allow_pickle=False)
        ts = z["ts"]
        loaded.append((os.path.basename(f)[:-4], ts, z["open"], z["close"], z["qv"], z["tbq"]))
        t_min = ts[0] if t_min is None else min(t_min, ts[0])
        t_max = ts[-1] if t_max is None else max(t_max, ts[-1])
    T = int((t_max - t_min) // HOUR_MS) + 1
    n = len(loaded)
    O = np.full((n, T), np.nan)
    C = np.full((n, T), np.nan)
    QV = np.full((n, T), np.nan, dtype=np.float32)
    TBQ = np.full((n, T), np.nan, dtype=np.float32)
    names = []
    for i, (name, ts, o, c, qv, tbq) in enumerate(loaded):
        idx = ((ts - t_min) // HOUR_MS).astype(int)
        O[i, idx], C[i, idx], QV[i, idx], TBQ[i, idx] = o, c, qv, tbq
        names.append(name)
    tms = t_min + np.arange(T, dtype=np.int64) * HOUR_MS
    return names, tms, O, C, QV, TBQ


# --------------------------------------------------------------------------- признаки
def rolling_median_prev(x, w=W, minv=MINV):
    """median(x[t-w..t-1]) для каждого t; текущий бар НЕ входит. NaN, если валидных < minv."""
    T = len(x)
    out = np.full(T, np.nan, dtype=np.float32)
    if T <= w:
        return out
    win = sliding_window_view(x.astype(np.float32), w)[: T - w]      # окна, заканчивающиеся на t-1
    s = np.sort(win, axis=1)                                          # NaN уходят в конец
    nvalid = (~np.isnan(win)).sum(axis=1)
    k = np.maximum((nvalid - 1) // 2, 0)
    med = s[np.arange(len(s)), k]
    med[nvalid < minv] = np.nan
    out[w:] = med
    return out


def _base_one(args):
    i, tb, ts_, qv = args
    return i, rolling_median_prev(tb), rolling_median_prev(ts_), rolling_median_prev(qv)


def compute_bases(TBQ, TS, QV, processes=8):
    n, T = QV.shape
    b_tb = np.full((n, T), np.nan, np.float32)
    b_ts = np.full((n, T), np.nan, np.float32)
    b_qv = np.full((n, T), np.nan, np.float32)
    jobs = [(i, TBQ[i], TS[i], QV[i]) for i in range(n)]
    if processes > 1:
        with Pool(processes) as p:
            for i, a, b, c in p.imap_unordered(_base_one, jobs, chunksize=4):
                b_tb[i], b_ts[i], b_qv[i] = a, b, c
    else:
        for j in jobs:
            i, a, b, c = _base_one(j)
            b_tb[i], b_ts[i], b_qv[i] = a, b, c
    return b_tb, b_ts, b_qv


def fwd_returns(O, C, h):
    """fwd_h[t] = close[t+h] / open[t+1] - 1  (вход по открытию СЛЕДУЮЩЕГО бара)."""
    n, T = C.shape
    out = np.full((n, T), np.nan)
    out[:, : T - h] = C[:, h:] / O[:, 1: T - h + 1] - 1.0
    return out


def build_features(O, C, QV, TBQ, bases=None, processes=8):
    TS = np.clip(QV - TBQ, 0, None)
    if bases is None:
        bases = compute_bases(TBQ, TS, QV, processes)
    b_tb, b_ts, b_qv = bases
    with np.errstate(divide="ignore", invalid="ignore"):
        r = C / O - 1.0
        rb = np.where(b_tb > 0, TBQ / b_tb, np.nan)
        rs = np.where(b_ts > 0, TS / b_ts, np.nan)
        rv = np.where(b_qv > 0, QV / b_qv, np.nan)
    liq = np.isfinite(b_qv) & (b_qv >= LIQ_MIN) & np.isfinite(r)
    feat = dict(r=r, rb=rb.astype(np.float32), rs=rs.astype(np.float32), rv=rv.astype(np.float32), liq=liq,
                raw={}, ex={})
    for h in HORIZONS:
        f = fwd_returns(O, C, h)
        A = np.where(liq & np.isfinite(f), f, np.nan)
        cnt = np.isfinite(A).sum(axis=0)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            mkt = np.nanmean(A, axis=0)
        mkt[cnt < MIN_MKT_COINS] = np.nan
        feat["raw"][h] = f
        feat["ex"][h] = f - mkt[None, :]
    return feat


# --------------------------------------------------------------------------- события
def hyp_masks(feat, hyp, R, U):
    """Возвращает (маска события, маска контроля, матрица направления dir: score = dir * excess)."""
    r, liq = feat["r"], feat["liq"]
    if hyp == "H1":
        band = liq & (r >= -0.005) & (r <= U)
        ev = band & (feat["rb"] >= R)
        ct = band & (feat["rb"] >= 0.5) & (feat["rb"] <= 1.5)
        dirm = -1.0
    elif hyp == "H2":
        band = liq & (r <= 0.005) & (r >= -U)
        ev = band & (feat["rs"] >= R)
        ct = band & (feat["rs"] >= 0.5) & (feat["rs"] <= 1.5)
        dirm = 1.0
    elif hyp == "H3":
        band = liq & (np.abs(r) >= 0.002) & (np.abs(r) <= U)
        ev = band & (feat["rv"] >= R)
        ct = band & (feat["rv"] >= 0.5) & (feat["rv"] <= 1.5)
        dirm = -np.sign(r)
    else:
        raise ValueError(hyp)
    return ev, ct, dirm


def period_mask(tms, which):
    day = 24 * HOUR_MS
    if which == "train":
        return tms < SPLIT_MS - day
    if which == "test":
        return tms >= SPLIT_MS
    return np.ones_like(tms, dtype=bool)


def day_sums(mask, val, dayid, ndays):
    m = mask & np.isfinite(val)
    rows, cols = np.nonzero(m)
    d = dayid[cols]
    S = np.bincount(d, weights=val[rows, cols].astype(np.float64), minlength=ndays)
    N = np.bincount(d, minlength=ndays).astype(np.float64)
    return S, N


def evaluate(feat, tms, hyp, R, U, h, which, B=2000, seed=1):
    """Основная статистика: score = dir*excess (bps). Положительный score = гипотеза подтверждается."""
    T = len(tms)
    dayid = (np.arange(T) // 24).astype(int)
    ndays = dayid.max() + 1
    pm = period_mask(tms, which)
    ev, ct, dirm = hyp_masks(feat, hyp, R, U)
    ev = ev & pm[None, :]
    ct = ct & pm[None, :]
    score = dirm * feat["ex"][h] * 1e4
    Se, Ne = day_sums(ev, score, dayid, ndays)
    Sc, Nc = day_sums(ct, score, dayid, ndays)
    days = np.unique(dayid[pm])
    ne, nc = Ne[days].sum(), Nc[days].sum()
    if ne < 30 or nc < 30:
        return dict(n_ev=int(ne), n_ct=int(nc), ok=False)
    mean_e = Se[days].sum() / ne
    mean_c = Sc[days].sum() / nc
    rng = np.random.default_rng(seed)
    idx = rng.choice(days, size=(B, len(days)), replace=True)
    me = Se[idx].sum(1) / np.maximum(Ne[idx].sum(1), 1)
    mc = Sc[idx].sum(1) / np.maximum(Nc[idx].sum(1), 1)
    d = me - mc
    se = float(d.std(ddof=1))
    diff = mean_e - mean_c
    # сырой (не нейтрализованный) валовый PnL сделки в направлении гипотезы
    rawscore = dirm * feat["raw"][h] * 1e4
    Sr, Nr = day_sums(ev, rawscore, dayid, ndays)
    return dict(ok=True, n_ev=int(ne), n_ct=int(nc), ev_bps=float(mean_e), ct_bps=float(mean_c),
                diff_bps=float(diff), se=se, t=float(diff / se) if se > 0 else float("nan"),
                p_one=float(np.mean(d <= 0)), gross_raw_bps=float(Sr[days].sum() / max(Nr[days].sum(), 1)),
                days_with_events=int((Ne[days] > 0).sum()), ndays=int(len(days)))


def per_coin_breadth(feat, tms, hyp, R, U, h, which, min_ev=10):
    pm = period_mask(tms, which)
    ev, ct, dirm = hyp_masks(feat, hyp, R, U)
    ev, ct = ev & pm[None, :], ct & pm[None, :]
    score = dirm * feat["ex"][h] * 1e4
    diffs = []
    for i in range(score.shape[0]):
        a = score[i][ev[i] & np.isfinite(score[i])]
        b = score[i][ct[i] & np.isfinite(score[i])]
        if len(a) >= min_ev and len(b) >= min_ev:
            diffs.append(a.mean() - b.mean())
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return dict(n_coins=0)
    return dict(n_coins=int(len(diffs)), share_pos=float((diffs > 0).mean()), median=float(np.median(diffs)))


BUCKETS = [(0.5, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 7.0), (7.0, 15.0), (15.0, 1e9)]


def dose_response(feat, tms, hyp, U, h, which):
    """Средний score по корзинам объёмного отношения при фиксированной ценовой полосе."""
    pm = period_mask(tms, which)
    r, liq = feat["r"], feat["liq"]
    if hyp == "H1":
        band, ratio, dirm = liq & (r >= -0.005) & (r <= U), feat["rb"], -1.0
    elif hyp == "H2":
        band, ratio, dirm = liq & (r <= 0.005) & (r >= -U), feat["rs"], 1.0
    else:
        band, ratio, dirm = liq & (np.abs(r) >= 0.002) & (np.abs(r) <= U), feat["rv"], -np.sign(r)
    score = dirm * feat["ex"][h] * 1e4
    out = []
    for lo, hi in BUCKETS:
        m = band & (ratio >= lo) & (ratio < hi) & pm[None, :] & np.isfinite(score)
        n = int(m.sum())
        out.append((lo, hi, n, float(score[m].mean()) if n else float("nan")))
    return out
