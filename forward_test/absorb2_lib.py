"""
Раунд 2: две НОВЫЕ гипотезы поглощения (H4, H5) + проверка на новом нетронутом блоке истории.

ПРЕ-РЕГИСТРАЦИЯ (зафиксировано ДО запуска на данных; признаки/нормировки/ликвидность/нейтрализация —
те же, что в absorb_lib.py):

H4  "бой с двух сторон" (дословно из задачи: объёмы и покупки, и продажи выше обычного, цена почти на месте)
    событие:  rb >= R  И  rs >= R  И  0.2% <= |r| <= U      направление: ПРОТИВ движения бара (-sign(r))
    контроль: rb и rs оба в [0.5, 1.5], та же ценовая полоса.
    сетка:    R in {2, 3, 5}, U in {2%, 3%}   (6 конфигураций)

H5  "цена не сдвинулась относительно силы потока" (price-impact failure)
    s_t   = (taker_buy - taker_sell) / (taker_buy + taker_sell)  - дисбаланс потока бара, в [-1, 1]
    По ПРЕДЫДУЩИМ 168 ч (текущий бар не входит), отдельно по каждой монете:
        регрессия r на s -> ожидаемое движение  e_t = mean_r + beta * (s_t - mean_s)
        z_t = (r_t - e_t) / std(остатка)      ("насколько цена отстала/опередила силу потока")
    событие (покупки не двинули цену):  s >= 0.20 и z <= -Z и rv >= R и -0.5% <= r <= U   -> дальше ВНИЗ
    событие (продажи не двинули цену):  s <= -0.20 и z >= +Z и rv >= R и -U <= r <= +0.5% -> дальше ВВЕРХ
    контроль: те же условия по s, z и цене, но НОРМАЛЬНЫЙ объём (rv в [0.5, 1.5]).
    порог 0.20 по |s| фиксирован (не подбирается)
    сетка:    R in {2, 3}, Z in {0.5, 1}, U = 3%        (4 конфигурации)
    ИЗМЕНЕНИЕ ДИЗАЙНА ДО ПРОСМОТРА ИСХОДОВ: первая сетка (R in {3,5}, Z in {1,2}, U in {2%,3%}) дала на train
    всего 4..563 события (проверялось ТОЛЬКО число событий, доходности не читались); U почти не влияет, т.к. z<=-Z
    уже ограничивает ход цены. Поэтому сетка заменена на более мягкую и короткую (4 конфигурации).

Протокол: выбор конфигурации ТОЛЬКО на train (2023-11 .. 2025-04; максимум t при n_событий >= 300),
затем одна проверка на двух независимых блоках:
   блок A - НОВАЯ история 2020-11-01 .. 2023-10-31 (раньше не использовалась),
   блок B - test 2025-05-01 .. (для H1-H3 уже смотрели, но H4/H5 на нём не выбирались).
Горизонт h=1 (вход по открытию следующего бара, выход по закрытию).
Статистика: разница событие-контроль в bps (рыночно-нейтрально), бутстрэп по календарным дням (B=2000),
дополнительно доля прибыльных сделок (сырая цена, в направлении гипотезы) и её разница с контролем.
Критерии (поправка Бонферрони на 2 новые гипотезы: одностор. p < 0.025):
   СТАТИСТИЧЕСКИ ПОДТВЕРЖДЕНО: на блоках A и B одновременно diff > 0 и p < 0.025;
   ТОРГУЕМО: дополнительно на обоих блоках средняя валовая сделка > 12 bps (издержки перпов за круг)
             и доля прибыльных сделок >= 54%.
Больше ничего не подбирается; любые дальнейшие варианты - уже новый раунд и новые данные.
"""
import numpy as np

import absorb_lib as L

S_MIN = 0.20
R4_GRID = (2.0, 3.0, 5.0)
R5_GRID = (2.0, 3.0)
Z_GRID = (0.5, 1.0)
U_GRID = (0.02, 0.03)
U5_GRID = (0.03,)
ALPHA = 0.05 / 2
COST_PERP, COST_SPOT = 12.0, 20.0
HIT_MIN = 0.54


# --------------------------------------------------------------------------- признаки потока/импакта
def _rsum_prev(x, w=L.W):
    """sum(x[t-w..t-1]) вдоль оси 1 (текущий бар не входит); NaN при t < w."""
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    out[:, w] = cs[:, w - 1]
    out[:, w + 1:] = cs[:, w:-1] - cs[:, : -w - 1]
    return out


def impact_features(r, QV, TBQ):
    """s_t (дисбаланс потока) и z_t (нормированный остаток доходности после поправки на s), без look-ahead."""
    TS = np.clip(QV - TBQ, 0, None)
    tot = TBQ + TS
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(tot > 0, (TBQ - TS) / tot, np.nan)
        valid = np.isfinite(r) & np.isfinite(s)
        r0 = np.where(valid, r, 0.0)
        s0 = np.where(valid, s, 0.0)
        N = _rsum_prev(valid.astype(np.float64))
        Sr, Ss = _rsum_prev(r0), _rsum_prev(s0)
        Srs, Sss, Srr = _rsum_prev(r0 * s0), _rsum_prev(s0 * s0), _rsum_prev(r0 * r0)
        mr, ms = Sr / N, Ss / N
        cov = Srs / N - mr * ms
        vs = Sss / N - ms ** 2
        vr = Srr / N - mr ** 2
        beta = cov / vs
        vres = vr - cov ** 2 / vs
        sd = np.sqrt(np.maximum(vres, 0.0))
        u = r - (mr + beta * (s - ms))
        z = u / sd
        ok = (N >= L.MINV) & (vs > 1e-8) & (sd > 0) & np.isfinite(z)
        z = np.where(ok, z, np.nan)
    return s.astype(np.float32), z.astype(np.float32)


def build_features2(O, C, QV, TBQ, bases=None, processes=8):
    feat = L.build_features(O, C, QV, TBQ, bases=bases, processes=processes)
    feat["s"], feat["z"] = impact_features(feat["r"], QV, TBQ)
    return feat


# --------------------------------------------------------------------------- события
def hyp_masks2(feat, hyp, R, U, Z=None):
    r, liq = feat["r"], feat["liq"]
    if hyp == "H4":
        band = liq & (np.abs(r) >= 0.002) & (np.abs(r) <= U)
        ev = band & (feat["rb"] >= R) & (feat["rs"] >= R)
        ct = band & (feat["rb"] >= 0.5) & (feat["rb"] <= 1.5) & (feat["rs"] >= 0.5) & (feat["rs"] <= 1.5)
        dirm = -np.sign(r)
    elif hyp == "H5":
        s, z = feat["s"], feat["z"]
        buy = liq & (s >= S_MIN) & (z <= -Z) & (r >= -0.005) & (r <= U)
        sell = liq & (s <= -S_MIN) & (z >= Z) & (r <= 0.005) & (r >= -U)
        side = buy | sell
        ev = side & (feat["rv"] >= R)
        ct = side & (feat["rv"] >= 0.5) & (feat["rv"] <= 1.5)
        dirm = np.where(buy, -1.0, 1.0)
    else:
        return L.hyp_masks(feat, hyp, R, U)
    return ev, ct, dirm


def grid(hyp):
    if hyp == "H4":
        return [dict(R=R, U=U, Z=None) for R in R4_GRID for U in U_GRID]
    if hyp == "H5":
        return [dict(R=R, U=U, Z=Z) for R in R5_GRID for Z in Z_GRID for U in U5_GRID]
    return [dict(R=R, U=U, Z=None) for R in L.R_GRID for U in L.U_GRID]


def cfg_str(cfg):
    z = f" Z={cfg['Z']:g}" if cfg.get("Z") else ""
    return f"R={cfg['R']:g}{z} U={cfg['U']*100:.0f}%"


# --------------------------------------------------------------------------- статистика
def evaluate2(feat, tms, hyp, cfg, h, pm, B=2000, seed=1):
    """Событие минус контроль: нейтральный score (bps), доля прибыльных сделок (сырая цена), бутстрэп по дням."""
    T = len(tms)
    dayid = (np.arange(T) // 24).astype(int)
    ndays = dayid.max() + 1
    ev, ct, dirm = hyp_masks2(feat, hyp, cfg["R"], cfg["U"], cfg.get("Z"))
    score = dirm * feat["ex"][h] * 1e4
    rawsc = dirm * feat["raw"][h] * 1e4
    okm = np.isfinite(score) & np.isfinite(rawsc) & pm[None, :]
    ev, ct = ev & okm, ct & okm
    hit = (rawsc > 0).astype(np.float64)
    Se, Ne = L.day_sums(ev, score, dayid, ndays)
    Sc, Nc = L.day_sums(ct, score, dayid, ndays)
    He, _ = L.day_sums(ev, hit, dayid, ndays)
    Hc, _ = L.day_sums(ct, hit, dayid, ndays)
    Re, _ = L.day_sums(ev, rawsc, dayid, ndays)
    days = np.unique(dayid[pm])
    ne, nc = Ne[days].sum(), Nc[days].sum()
    if ne < 30 or nc < 30:
        return dict(ok=False, n_ev=int(ne), n_ct=int(nc))
    rng = np.random.default_rng(seed)
    idx = rng.choice(days, size=(B, len(days)), replace=True)
    ne_b, nc_b = np.maximum(Ne[idx].sum(1), 1), np.maximum(Nc[idx].sum(1), 1)
    d_sc = Se[idx].sum(1) / ne_b - Sc[idx].sum(1) / nc_b
    d_hit = He[idx].sum(1) / ne_b - Hc[idx].sum(1) / nc_b
    diff = Se[days].sum() / ne - Sc[days].sum() / nc
    hdiff = He[days].sum() / ne - Hc[days].sum() / nc
    se, hse = float(d_sc.std(ddof=1)), float(d_hit.std(ddof=1))
    return dict(ok=True, n_ev=int(ne), n_ct=int(nc),
                ev_bps=float(Se[days].sum() / ne), ct_bps=float(Sc[days].sum() / nc),
                diff_bps=float(diff), se=se, t=float(diff / se) if se > 0 else float("nan"),
                p_one=float(np.mean(d_sc <= 0)),
                hit_ev=float(He[days].sum() / ne), hit_ct=float(Hc[days].sum() / nc),
                hit_diff=float(hdiff), hit_t=float(hdiff / hse) if hse > 0 else float("nan"),
                gross_raw_bps=float(Re[days].sum() / ne))
