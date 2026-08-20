"""
Что будет, если животное не стоит на месте: отклик уплывает или его нет вовсе.

ЗАЧЕМ. validate_stim_optimizer.py проверяет оптимизатор на НЕПОДВИЖНОЙ
поверхности. Это допущение, а не факт. Гауссов процесс по построению считает,
что отклик за сессию не меняется, и старые наблюдения не забывает. Живое
животное устаёт: к пятидесятой пробе оптимум может быть не там, где был на
десятой, и тогда модель уверенно держится за точку, которой уже нет.

Второй сценарий - отклик плоский. Стимуляция почти ничего не даёт. Правильное
поведение здесь не "найти оптимум", а честно сказать, что эффекта нет. Проверить
это надо ДО эксперимента, потому что отчитаться несуществующим эффектом легко:
максимум шести десятков зашумлённых нулей заметно больше нуля.

ЧЕТЫРЕ ПОВЕРХНОСТИ.
    stationary - контроль, отклик не меняется;
    fatigue    - польза затухает к концу сессии, оптимум на месте;
    drift      - оптимум по амплитуде уезжает (порог животного растёт);
    flat       - пользы нет вообще, только штраф за дозу и шум.

ЧТО МЕРЯЕМ. Не "лучшее за сессию", а промах ТОЙ точки, которую система выдаёт
как рекомендацию, на поверхности, какой она стала К КОНЦУ. Это и есть вопрос
эксперимента: чем стимулировать дальше.

Сравниваются два правила рекомендации:
    best()      - набор с наибольшим наблюдением (наивное правило);
    recommend() - набор с наибольшим сглаженным предсказанием.

И отдельно - завышение: насколько число, которым система отчиталась, больше
правды. На плоской поверхности любое завышение это выдуманный эффект.

Запуск:
    python validate_stim_drift.py --trials 60 --repeats 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stim_optimizer import StimOptimizer  # noqa: E402
from stim_params import (Contact, Montage, PulseShape, SafetyLimits,  # noqa: E402
                         SearchSpace, StimParams, from_vector, normalized_dose)

FATIGUE_DEPTH = 0.50      # к концу сессии польза слабее вдвое
DRIFT_SHIFT = 0.25        # оптимум по амплитуде уезжает на четверть диапазона


def make_setup():
    shape = PulseShape(phase_width_us=200.0)
    montage = Montage("L2", cathode=Contact("c3", "L2", area_mm2=0.30))
    limits = SafetyLimits(max_amplitude_ua=700.0, max_charge_density_uc_cm2=60.0)
    return SearchSpace(), shape, montage, limits


def truth(p: StimParams, limits: SafetyLimits, mode: str, u: float) -> float:
    """
    Истинный счёт. u - доля сессии, от 0 в начале до 1 в конце.

    Форма пользы та же, что в validate_stim_optimizer, чтобы результаты двух
    стендов были сравнимы: меняется только то, что делает время.
    """
    a = (p.amplitude_ua - 100.0) / 500.0
    f = (p.frequency_hz - 20.0) / 60.0
    t = (p.train_ms - 50.0) / 150.0
    c = p.interchannel_ms / 60.0

    if mode == "flat":
        return -0.20 * normalized_dose(p, limits)
    gain = 1.0 - FATIGUE_DEPTH * u if mode == "fatigue" else 1.0
    a_opt = 0.55 + DRIFT_SHIFT * u if mode == "drift" else 0.55

    eff = (np.exp(-((a - a_opt) ** 2) / 0.055)
           * np.exp(-((f - 0.40) ** 2) / 0.10)
           * (1.0 - np.exp(-3.0 * t))
           * np.exp(-((c - 0.35) ** 2) / 0.20))
    return float(0.95 * gain * eff - 0.20 * normalized_dose(p, limits))


def ceiling(space, shape, montage, limits, mode: str, u: float,
            n: int = 20_000) -> float:
    """Лучшее достижимое на поверхности в момент u."""
    rng = np.random.default_rng(0)
    b = np.asarray(space.as_bounds(), float)
    best = -np.inf
    for v in b[:, 0] + rng.random((n, 4)) * (b[:, 1] - b[:, 0]):
        p = from_vector(v, space, shape, montage)
        if p.is_safe(limits):
            best = max(best, truth(p, limits, mode, u))
    return best


def run(mode, seed, trials, noise, space, shape, montage, limits, **kw):
    opt = StimOptimizer(space, shape, montage, limits, noise=noise, seed=seed,
                        n_candidates=1024, **kw)
    rng = np.random.default_rng(seed + 10_000)
    for k in range(trials):
        u = k / max(1, trials - 1)
        p = opt.suggest()
        opt.observe(p, truth(p, limits, mode, u) + rng.normal(0.0, noise))

    naive, naive_score = opt.best()
    smooth, smooth_mu, _ = opt.recommend()
    return {
        # промах на поверхности, какой она стала к концу
        "naive_true": truth(naive, limits, mode, 1.0),
        "smooth_true": truth(smooth, limits, mode, 1.0),
        # чем система отчиталась
        "naive_claim": naive_score,
        "smooth_claim": smooth_mu,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.06)
    ap.add_argument("--half-life", type=float, default=15.0,
                    help="период полузабывания своих проб")
    a = ap.parse_args(argv)

    space, shape, montage, limits = make_setup()
    modes = ("stationary", "fatigue", "drift", "flat")
    variants = (("помнит всё", {}),
                (f"забывает, T½={a.half_life:g}", {"forget_half_life": a.half_life}))

    print(f"проб {a.trials}, повторов {a.repeats}, шум {a.noise}")
    print(f"затухание пользы к концу: {100 * FATIGUE_DEPTH:.0f}%, "
          f"уезд оптимума: {100 * DRIFT_SHIFT:.0f}% диапазона амплитуды\n")

    for mode in modes:
        top = ceiling(space, shape, montage, limits, mode, 1.0)
        print(f"=== {mode} === потолок в конце сессии {top:.3f}", flush=True)
        print(f"{'':<22} {'промах рекомендации':>21} {'завышение отчёта':>19}")
        print(f"{'вариант':<22} {'наивно':>10} {'сглажено':>10} "
              f"{'наивно':>9} {'сглажено':>9}")
        for name, kw in variants:
            res = [run(mode, s, a.trials, a.noise, space, shape, montage, limits, **kw)
                   for s in range(a.repeats)]
            nt = np.array([r["naive_true"] for r in res])
            st = np.array([r["smooth_true"] for r in res])
            nc = np.array([r["naive_claim"] for r in res])
            sc = np.array([r["smooth_claim"] for r in res])
            print(f"{name:<22} {top - nt.mean():>10.3f} {top - st.mean():>10.3f} "
                  f"{(nc - nt).mean():>9.3f} {(sc - st).mean():>9.3f}", flush=True)
        print()

    print("промах  = потолок минус истинный счёт рекомендованной точки "
          "(меньше лучше)")
    print("завышение = чем отчиталась система минус правда (ноль лучше; "
          "на flat это выдуманный эффект)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
