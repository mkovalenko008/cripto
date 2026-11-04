"""
Раунд 4: H1-шорт с защитным стопом (единственный вариант, нацеленный на "процент успеха").
Использует absorb_lib / absorb2_lib / absorb3_lib.

ПРЕ-РЕГИСТРАЦИЯ (зафиксировано ДО запуска на данных):

Событие: H1 с ФИКСИРОВАННОЙ конфигурацией R=8, U=2% (rb>=8 и -0.5% <= r <= +2%; выбрана на train в раундах 1 и 3).
Сделка: ТОЛЬКО шорт. Вход - открытие следующего бара. Удержание до закрытия бара t+h, h in {24, 72} часов.
Стоп: уровень = вход * (1 + k * sigma * sqrt(24)), где sigma = стандартное отклонение часовых доходностей бара
  (close/open-1) монеты за ПРЕДЫДУЩИЕ 168 ч (текущий бар не входит, минимум 120 валидных);
  k in {1.0, 1.5, 2.0}. Срабатывает по максимуму (high) любого бара удержания, включая бар входа;
  исполнение по уровню стопа, а если бар ОТКРЫЛСЯ выше уровня (гэп) - по цене открытия бара. Тейк-профита нет
  (иначе процент успеха надувается искусственно). Все прочие допущения консервативны: проскальзывания сверх гэпа нет,
  поэтому дополнительно показываются результаты с издержками 12 и 20 bps за круг.
Контроль: те же условия по цене, нормальный объём покупок (rb в [0.5, 1.5]), то же правило стопа.
Нейтрализация: из результата сделки (в долях цены входа) вычитается среднее по всем ликвидным монетам (>=20)
  результата ТОГО ЖЕ правила шорта-со-стопом в тот же час (чтобы общерыночное движение не выдавалось за сигнал).
Выбор: для каждого h выбирается k с максимальным t на train (N>=300). Затем одна проверка на блоках
  A (2020-11..2023-10) и B (2025-05..); бутстрэп по блокам из 7 суток, B=2000. Это 2 теста.
Статистически подтверждено: на ОБОИХ блоках diff(событие-контроль) > 0 и одностор. p < 0.05
  (совместно ~0.0025 на тест, на 2 теста ~0.005).
Торгуемо (дополнительно, на обоих блоках): средний результат шорта-со-стопом БЕЗ нейтрализации > 25 bps
  и доля прибыльных сделок >= 54%.
ЗАРАНЕЕ СФОРМУЛИРОВАННОЕ ОЖИДАНИЕ (чтобы не подгонять объяснение под результат): стоп на ценовом пути без
  собственного преимущества не создаёт матожидание, он меняет форму распределения (срезает хвосты, снижает долю
  успеха). Поэтому основная гипотеза - что средняя сделка на блоке B останется около нуля; тест проверяет,
  верно ли это, а не подтверждает надежду.
"""
import os

import numpy as np

import absorb_lib as L
import absorb2_lib as L2

K_GRID = (1.0, 1.5, 2.0)
H_GRID = (24, 72)
H1_CFG = (8.0, 0.02)
SQ24 = np.sqrt(24.0)


def load_high(data_dir, names, tms):
    """Матрица high в той же сетке и порядке монет, что и load_matrices."""
    T = len(tms)
    H = np.full((len(names), T), np.nan)
    for i, nm in enumerate(names):
        z = np.load(os.path.join(data_dir, nm + ".npz"), allow_pickle=False)
        idx = ((z["ts"] - tms[0]) // L.HOUR_MS).astype(int)
        H[i, idx] = z["high"]
    return H


def hourly_sigma(O, C):
    """std часовых доходностей бара за предыдущие 168 ч (текущий бар не входит)."""
    r = C / O - 1.0
    valid = np.isfinite(r)
    r0 = np.where(valid, r, 0.0)
    N = L2._rsum_prev(valid.astype(np.float64))
    S1, S2 = L2._rsum_prev(r0), L2._rsum_prev(r0 * r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = S1 / N
        var = S2 / N - m * m
        sig = np.sqrt(np.maximum(var, 0.0))
    sig[~(N >= L.MINV)] = np.nan
    return sig


def short_with_stop(O, H, C, sig, h, k):
    """
    Для события на баре t: вход O[t+1], стоп = вход*(1+k*sig[t]*sqrt(24)), удержание баров t+1..t+h, выход по закрытию t+h.
    Возвращает (long_equiv, stopped): long_equiv = exit/entry - 1 (то, что "заплатит" шорт; шорт-доходность = -long_equiv),
    stopped - был ли выход по стопу. Если внутри окна есть пропущенный бар - результат NaN.
    """
    n, T = C.shape
    Tm = T - h
    long_eq = np.full((n, T), np.nan)
    stopped = np.zeros((n, T), dtype=bool)
    if Tm <= 0:
        return long_eq, stopped
    e = O[:, 1:Tm + 1]
    s = e * (1.0 + k * sig[:, :Tm] * SQ24)
    alive = np.isfinite(e) & np.isfinite(s)
    bad = ~alive
    exit_p = np.full((n, Tm), np.nan)
    stp = np.zeros((n, Tm), dtype=bool)
    for j in range(1, h + 1):
        Ob, Hb = O[:, j:j + Tm], H[:, j:j + Tm]
        bad |= ~(np.isfinite(Ob) & np.isfinite(Hb))
        gap = alive & (Ob >= s)
        hit = alive & ~gap & (Hb >= s)
        exit_p[gap] = Ob[gap]
        exit_p[hit] = s[hit]
        stp |= gap | hit
        alive = alive & ~gap & ~hit
    Cend = C[:, h:h + Tm]
    exit_p[alive] = Cend[alive]
    res = exit_p / e - 1.0
    res[bad | ~np.isfinite(exit_p)] = np.nan
    long_eq[:, :Tm] = res
    stopped[:, :Tm] = stp
    return long_eq, stopped


def add_stop_variant(feat, key, long_eq):
    """Кладёт результат в feat['raw'][key] и нейтрализованный feat['ex'][key] (вычитая среднее по ликвидным монетам)."""
    import warnings
    A = np.where(feat["liq"] & np.isfinite(long_eq), long_eq, np.nan)
    cnt = np.isfinite(A).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mkt = np.nanmean(A, axis=0)
    mkt[cnt < L.MIN_MKT_COINS] = np.nan
    feat["raw"][key] = long_eq
    feat["ex"][key] = long_eq - mkt[None, :]
