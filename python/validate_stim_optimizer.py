"""
Проверка медленной петли: находит ли оптимизатор хорошие параметры за то число
проб, которое реально помещается в сессию.

ЗАЧЕМ. Собрать GP-BO несложно; вопрос в том, сойдётся ли он при НАШЕМ уровне
шума и за НАШЕ число проб. Шум счёта походки измерен: блок из 10 шагов даёт
разброс около 0.06 по шкале, где интакт ~0.94 (см. validate_gait_score.py).
Если при таком шуме за 30 проб оптимизатор не обгоняет случайный перебор, вся
затея с адаптивной стимуляцией на живом животном бессмысленна.

МОДЕЛЬ ЖИВОТНОГО. Настоящего отклика у нас нет, поэтому берётся правдоподобная
поверхность: эффект растёт с амплитудой до оптимума и падает за ним
(перестимуляция), есть предпочтительная частота, длительность серии выходит на
насыщение, межканальный сдвиг имеет свой оптимум. Сверху вычитается доза.
Это НЕ предсказание того, как поведёт себя крыса; это стенд, на котором
проверяется сам оптимизатор.

ЧТО СРАВНИВАЕМ. GP-BO против случайного перебора при одинаковом бюджете проб и
одинаковых ограничениях безопасности.

Запуск:
    python validate_stim_optimizer.py --trials 30 --repeats 40
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


def make_setup():
    """Электрод с площадью, при которой пределы не режут весь диапазон."""
    shape = PulseShape(phase_width_us=200.0)
    montage = Montage("L2", cathode=Contact("c3", "L2", area_mm2=0.30))
    limits = SafetyLimits(max_amplitude_ua=700.0, max_charge_density_uc_cm2=60.0)
    return SearchSpace(), shape, montage, limits


def animal_response(p: StimParams, limits: SafetyLimits, rng, noise: float) -> float:
    """
    Модель отклика: гладкий оптимум внутри диапазона плюс штраф за дозу.
    Возвращает шумное наблюдение, как и настоящая проба.
    """
    a = (p.amplitude_ua - 100.0) / 500.0            # 0..1
    f = (p.frequency_hz - 20.0) / 60.0
    t = (p.train_ms - 50.0) / 150.0
    c = p.interchannel_ms / 60.0

    eff = (np.exp(-((a - 0.55) ** 2) / 0.055)        # оптимум по амплитуде
           * np.exp(-((f - 0.40) ** 2) / 0.10)       # предпочтительная частота
           * (1.0 - np.exp(-3.0 * t))                # насыщение по длительности
           * np.exp(-((c - 0.35) ** 2) / 0.20))      # оптимум по сдвигу
    true = 0.95 * eff - 0.20 * normalized_dose(p, limits)
    return float(true + rng.normal(0.0, noise)), float(true)


def run_bo(trials, seed, noise, space, shape, montage, limits, prior=None):
    opt = StimOptimizer(space, shape, montage, limits, noise=noise, seed=seed)
    if prior:
        opt.add_prior(prior)
    rng = np.random.default_rng(seed + 10_000)
    best_true = -np.inf
    curve = []
    for _ in range(trials):
        p = opt.suggest()
        obs, true = animal_response(p, limits, rng, noise)
        opt.observe(p, obs)
        best_true = max(best_true, true)
        curve.append(best_true)
    return curve


def run_random(trials, seed, noise, space, shape, montage, limits,
               max_amp_step=100.0):
    """
    Случайный перебор при ТЕХ ЖЕ ограничениях, что у BO. Ограничение на
    наращивание амплитуды - требование безопасности, а не свойство алгоритма;
    без него сравнение нечестное: перебор сразу берёт из всего диапазона, а BO
    обязан подниматься постепенно.
    """
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 10_000)
    bounds = np.asarray(space.as_bounds(), float)
    i_amp = space.names().index("amplitude_ua")
    best_true = -np.inf
    tried_amp = [bounds[i_amp, 0]]
    curve = []
    got = 0
    guard = 0
    while got < trials and guard < 100_000:
        guard += 1
        v = bounds[:, 0] + rng.random(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        if max_amp_step is not None and v[i_amp] > max(tried_amp) + max_amp_step:
            continue
        p = from_vector(v, space, shape, montage)
        if not p.is_safe(limits):
            continue
        tried_amp.append(p.amplitude_ua)
        _, true = animal_response(p, limits, obs_rng, noise)
        best_true = max(best_true, true)
        curve.append(best_true)
        got += 1
    return curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.06,
                    help="разброс счёта между блоками, измерен на записях")
    a = ap.parse_args(argv)

    space, shape, montage, limits = make_setup()

    # верхняя планка: лучшее достижимое на плотной сетке
    rng = np.random.default_rng(0)
    grid = np.asarray(space.as_bounds(), float)
    samples = grid[:, 0] + rng.random((200_000, 4)) * (grid[:, 1] - grid[:, 0])
    ceiling = -np.inf
    for v in samples[:20_000]:
        p = from_vector(v, space, shape, montage)
        if p.is_safe(limits):
            _, true = animal_response(p, limits, np.random.default_rng(0), 0.0)
            ceiling = max(ceiling, true)
    print(f"проб {a.trials}, повторов {a.repeats}, шум счёта {a.noise}")
    print(f"верхняя планка модели: {ceiling:.3f}\n")

    bo = np.array([run_bo(a.trials, s, a.noise, space, shape, montage, limits)
                   for s in range(a.repeats)])
    rnd = np.array([run_random(a.trials, s, a.noise, space, shape, montage, limits)
                    for s in range(a.repeats)])
    rnd_free = np.array([run_random(a.trials, s, a.noise, space, shape, montage,
                                    limits, max_amp_step=None)
                         for s in range(a.repeats)])

    print(f"{'проба':>7} {'BO':>18} {'случайный перебор':>20} {'выигрыш':>10}")
    print("-" * 60)
    for k in (5, 10, 15, 20, 25, a.trials):
        if k > a.trials:
            continue
        b, r = bo[:, k - 1], rnd[:, k - 1]
        print(f"{k:>7} {b.mean():>10.3f} +- {b.std():>4.3f} "
              f"{r.mean():>12.3f} +- {r.std():>4.3f} {b.mean() - r.mean():>+10.3f}")

    print()
    print("без ограничения на наращивание (нереалистично, для справки):")
    print(f"  случайный перебор на пробе {a.trials}: {rnd_free[:, -1].mean():.3f}")

    b_end, r_end = bo[:, -1], rnd[:, -1]
    frac_bo = float(np.mean(b_end > ceiling * 0.95))
    frac_rnd = float(np.mean(r_end > ceiling * 0.95))
    print(f"\nдоля прогонов, достигших 95% планки за {a.trials} проб:")
    print(f"  BO                {100 * frac_bo:.0f}%")
    print(f"  случайный перебор {100 * frac_rnd:.0f}%")

    # тёплый старт: prior с "другого животного" со сдвинутым оптимумом
    prior = []
    prng = np.random.default_rng(77)
    for _ in range(12):
        v = grid[:, 0] + prng.random(4) * (grid[:, 1] - grid[:, 0])
        p = from_vector(v, space, shape, montage)
        if p.is_safe(limits):
            obs, _ = animal_response(p, limits, prng, a.noise)
            prior.append((p, obs))
    warm = np.array([run_bo(a.trials, s, a.noise, space, shape, montage, limits, prior)
                     for s in range(a.repeats)])
    print(f"\nтёплый старт ({len(prior)} проб с другого животного, вес 0.3):")
    for k in (5, 10, 15):
        print(f"  проба {k:>2}: без переноса {bo[:, k-1].mean():.3f}, "
              f"с переносом {warm[:, k-1].mean():.3f}")

    ok = b_end.mean() > r_end.mean()
    print("\nвывод:", "оптимизатор обгоняет случайный перебор"
          if ok else "ВЫИГРЫША НЕТ: при таком шуме BO не лучше перебора")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
