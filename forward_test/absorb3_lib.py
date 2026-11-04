"""
Раунд 3: (1) длинный горизонт удержания, (2) режимы рынка. Использует absorb_lib.py и absorb2_lib.py.

ПРЕ-РЕГИСТРАЦИЯ (зафиксировано ДО запуска на данных; ни горизонт 72ч, ни дневные события, ни режимы
на реальных данных ещё не считались; суточный горизонт для H1 на train/test видели в раунде 1 -
поэтому конфигурации тут ВСЕГДА заново выбираются на train, а проверка идёт на блоках A и B):

ЧАСТЬ 1a. Часовые события, длинное удержание.
  Гипотезы H1, H2, H3 (сетки R in {3,5,8} x U in {2%,3%}) и H4 (R in {2,3,5} x U in {2%,3%}) - определения как в
  absorb_lib / absorb2_lib. Вход - открытие следующего бара, выход - закрытие через h часов, h in {24, 72}.
  Это 4 гипотезы x 2 горизонта = 8 тестов.
ЧАСТЬ 1b. Дневные события (свечи дня UTC из часовых; дни с пропущенными часами отбрасываются).
  r = close/open-1 за день; rb, rs, rv - дневные taker-покупки/продажи/общий оборот, делённые на медиану
  ПРЕДЫДУЩИХ 30 дней (текущий день не входит, минимум 20 валидных); ликвидность: медиана дневного оборота
  >= 24 x 50 000 USDT. Вход - открытие следующего дня, выход - закрытие дня t+h, h in {1, 3} дня.
  Сетка для всех четырёх: R in {2,3,5} x U in {3%,5%}.
    H1d: rb>=R,  -1% <= r <= U            -> вниз
    H2d: rs>=R,  -U <= r <= +1%           -> вверх
    H3d: rv>=R,  0.5% <= |r| <= U         -> против движения дня
    H4d: rb>=R и rs>=R, 0.5% <= |r| <= U  -> против движения дня
  Контроль всегда: те же условия по цене, но нормальный объём (отношение в [0.5, 1.5]; для H4d оба в [0.5,1.5]).
  Это ещё 4 x 2 = 8 тестов.
ЧАСТЬ 2. Режимы рынка (часовые события, h=1).
  Режим по BTCUSDT на момент закрытия бара t: доходность BTC за предыдущие 336 ч (14 суток):
    range: |ret| < 8%;  up: ret >= +8%;  down: ret <= -8%.
  Для каждой из H1..H4 берётся ФИКСИРОВАННАЯ конфигурация, выбранная на train в раундах 1-2
  (H1: R=8,U=2%; H2: R=3,U=3%; H3: R=8,U=2%; H4: R=2,U=3%). Для гипотезы на train выбирается режим с максимальным
  t (N>=300); контроль берётся из того же режима. Это 4 теста.

Общий протокол: выбор (конфигурации и/или режима) ТОЛЬКО на train (2023-11..2025-04); дальше одна проверка на
блоке A (2020-11..2023-10, 116 монет) и блоке B (2025-05..). Рыночно-нейтральная избыточная доходность, событие минус
контроль, бутстрэп по БЛОКАМ из 7 суток (бары и сделки с удержанием 1-3 суток перекрываются), B=2000.
Критерий "статистически подтверждено": на ОБОИХ блоках diff > 0 и одностор. p < 0.05
(совместно ~0.0025 на тест; 20 тестов => вероятность хотя бы одного ложного срабатывания ~5%).
Критерий "торгуемо" (дополнительно, на обоих блоках): для частей 1a/1b рыночно-нейтральная разница > 25 bps и
валовая сделка > 25 bps (издержки 12-20 bps + фондирование/проскальзывание); для части 2 (h=1): разница > 12 bps и
валовая > 12 bps. Ничего больше не подбирается.
"""
import numpy as np

import absorb_lib as L
import absorb2_lib as L2

DAY_MS = 86_400_000
BLOCK_H = 24 * 7          # бутстрэп-блок для часовых массивов: 7 суток
BLOCK_D = 7               # для дневных массивов: 7 суток
D_W, D_MINV = 30, 20
D_R_GRID = (2.0, 3.0, 5.0)
D_U_GRID = (0.03, 0.05)
REGIME_THR = 0.08
REGIME_H = 336
REGIME_NAMES = {0: "боковик", 1: "рост BTC", 2: "падение BTC"}
FIXED_CFG = {"H1": (8.0, 0.02), "H2": (3.0, 0.03), "H3": (8.0, 0.02), "H4": (2.0, 0.03)}
STAT_ALPHA = 0.05


# --------------------------------------------------------------------------- горизонты / режимы
def add_horizon(feat, O, C, h):
    """Добавляет raw[h] и ex[h] (как в L.build_features) для нового горизонта h."""
    import warnings
    f = L.fwd_returns(O, C, h)
    A = np.where(feat["liq"] & np.isfinite(f), f, np.nan)
    cnt = np.isfinite(A).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mkt = np.nanmean(A, axis=0)
    mkt[cnt < L.MIN_MKT_COINS] = np.nan
    feat["raw"][h] = f
    feat["ex"][h] = f - mkt[None, :]


def btc_regime(C_btc):
    """-1 неизвестно; 0 боковик; 1 рост; 2 падение. Только прошлое: close[t]/close[t-336]-1."""
    T = len(C_btc)
    ret = np.full(T, np.nan)
    ret[REGIME_H:] = C_btc[REGIME_H:] / C_btc[:-REGIME_H] - 1.0
    reg = np.full(T, -1, dtype=np.int8)
    ok = np.isfinite(ret)
    reg[ok] = 0
    reg[ok & (ret >= REGIME_THR)] = 1
    reg[ok & (ret <= -REGIME_THR)] = 2
    return reg


# --------------------------------------------------------------------------- дневные свечи
def aggregate_daily(tms, O, C, QV, TBQ):
    """Часовые -> дневные (UTC). День валиден, только если все 24 часа на месте."""
    off = int(((-int(tms[0])) % DAY_MS) // L.HOUR_MS)
    nd = (len(tms) - off) // 24
    sl = slice(off, off + nd * 24)
    n = O.shape[0]
    Oh = O[:, sl].reshape(n, nd, 24)
    Ch = C[:, sl].reshape(n, nd, 24)
    Q = QV[:, sl].astype(np.float64).reshape(n, nd, 24)
    Tb = TBQ[:, sl].astype(np.float64).reshape(n, nd, 24)
    full = np.isfinite(Q).all(2) & np.isfinite(Tb).all(2) & np.isfinite(Oh[:, :, 0]) & np.isfinite(Ch[:, :, 23])
    O_d = np.where(full, Oh[:, :, 0], np.nan)
    C_d = np.where(full, Ch[:, :, 23], np.nan)
    QV_d = np.where(full, Q.sum(2), np.nan)
    TB_d = np.where(full, Tb.sum(2), np.nan)
    tms_d = tms[sl][::24]
    return tms_d, O_d, C_d, QV_d, TB_d


def daily_features(O, C, QV, TBQ, horizons=(1, 3)):
    import warnings
    n, T = C.shape
    TS = np.clip(QV - TBQ, 0, None)
    b_tb, b_ts, b_qv = (np.full((n, T), np.nan, np.float32) for _ in range(3))
    for i in range(n):
        b_tb[i] = L.rolling_median_prev(TBQ[i], D_W, D_MINV)
        b_ts[i] = L.rolling_median_prev(TS[i], D_W, D_MINV)
        b_qv[i] = L.rolling_median_prev(QV[i], D_W, D_MINV)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = C / O - 1.0
        rb = np.where(b_tb > 0, TBQ / b_tb, np.nan).astype(np.float32)
        rs = np.where(b_ts > 0, TS / b_ts, np.nan).astype(np.float32)
        rv = np.where(b_qv > 0, QV / b_qv, np.nan).astype(np.float32)
    liq = np.isfinite(b_qv) & (b_qv >= 24 * L.LIQ_MIN) & np.isfinite(r)
    feat = dict(r=r, rb=rb, rs=rs, rv=rv, liq=liq, raw={}, ex={})
    for h in horizons:
        f = L.fwd_returns(O, C, h)
        A = np.where(liq & np.isfinite(f), f, np.nan)
        cnt = np.isfinite(A).sum(axis=0)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            mkt = np.nanmean(A, axis=0)
        mkt[cnt < L.MIN_MKT_COINS] = np.nan
        feat["raw"][h], feat["ex"][h] = f, f - mkt[None, :]
    return feat


def hyp_masks_daily(feat, hyp, R, U):
    r, liq = feat["r"], feat["liq"]
    rb, rs, rv = feat["rb"], feat["rs"], feat["rv"]
    norm = lambda x: (x >= 0.5) & (x <= 1.5)
    if hyp == "H1":
        band = liq & (r >= -0.01) & (r <= U)
        return band & (rb >= R), band & norm(rb), -1.0
    if hyp == "H2":
        band = liq & (r <= 0.01) & (r >= -U)
        return band & (rs >= R), band & norm(rs), 1.0
    band = liq & (np.abs(r) >= 0.005) & (np.abs(r) <= U)
    if hyp == "H3":
        return band & (rv >= R), band & norm(rv), -np.sign(r)
    if hyp == "H4":
        return band & (rb >= R) & (rs >= R), band & norm(rb) & norm(rs), -np.sign(r)
    raise ValueError(hyp)


def grid_daily():
    return [dict(R=R, U=U, Z=None) for R in D_R_GRID for U in D_U_GRID]


# --------------------------------------------------------------------------- статистика (бутстрэп по блокам)
def evaluate3(feat, tms, ev, ct, dirm, h, pm, block, B=2000, seed=1):
    """Событие минус контроль; нейтральный score (bps), доля успеха (сырая цена), бутстрэп по блокам из `block` баров."""
    T = len(tms)
    bid = (np.arange(T) // block).astype(int)
    nb = int(bid.max()) + 1
    score = dirm * feat["ex"][h] * 1e4
    rawsc = dirm * feat["raw"][h] * 1e4
    okm = np.isfinite(score) & np.isfinite(rawsc) & pm[None, :]
    ev, ct = ev & okm, ct & okm
    hit = (rawsc > 0).astype(np.float64)
    Se, Ne = L.day_sums(ev, score, bid, nb)
    Sc, Nc = L.day_sums(ct, score, bid, nb)
    He, _ = L.day_sums(ev, hit, bid, nb)
    Hc, _ = L.day_sums(ct, hit, bid, nb)
    Re, _ = L.day_sums(ev, rawsc, bid, nb)
    blocks = np.unique(bid[pm])
    ne, nc = Ne[blocks].sum(), Nc[blocks].sum()
    if ne < 30 or nc < 30:
        return dict(ok=False, n_ev=int(ne), n_ct=int(nc))
    rng = np.random.default_rng(seed)
    idx = rng.choice(blocks, size=(B, len(blocks)), replace=True)
    ne_b, nc_b = np.maximum(Ne[idx].sum(1), 1), np.maximum(Nc[idx].sum(1), 1)
    d_sc = Se[idx].sum(1) / ne_b - Sc[idx].sum(1) / nc_b
    d_hit = He[idx].sum(1) / ne_b - Hc[idx].sum(1) / nc_b
    diff = Se[blocks].sum() / ne - Sc[blocks].sum() / nc
    hdiff = He[blocks].sum() / ne - Hc[blocks].sum() / nc
    se, hse = float(d_sc.std(ddof=1)), float(d_hit.std(ddof=1))
    return dict(ok=True, n_ev=int(ne), n_ct=int(nc), n_blocks=int(len(blocks)),
                ev_bps=float(Se[blocks].sum() / ne), ct_bps=float(Sc[blocks].sum() / nc),
                diff_bps=float(diff), se=se, t=float(diff / se) if se > 0 else float("nan"),
                p_one=float(np.mean(d_sc <= 0)),
                hit_ev=float(He[blocks].sum() / ne), hit_ct=float(Hc[blocks].sum() / nc),
                hit_diff=float(hdiff), hit_t=float(hdiff / hse) if hse > 0 else float("nan"),
                gross_raw_bps=float(Re[blocks].sum() / ne))
