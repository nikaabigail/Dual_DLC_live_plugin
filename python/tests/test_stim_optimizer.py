"""
Тесты медленной петли: гауссов процесс, EI и сам оптимизатор.

Главное, что тут стережётся, - безопасность. Оптимизатор не имеет права
предложить набор, нарушающий пределы, и не имеет права прыгнуть по амплитуде.
Это свойства КОНСТРУКЦИИ, а не результата обучения, поэтому они и проверяются
отдельными тестами, а не "качеством сходимости".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stim_optimizer import GP, StimOptimizer, expected_improvement
from stim_params import (Bounds, Contact, Montage, PulseShape, SafetyLimits,
                         SearchSpace, from_vector)


def setup(area_mm2: float = 0.30, max_amp: float = 700.0):
    space = SearchSpace()
    shape = PulseShape(phase_width_us=200.0)
    montage = Montage("L2", cathode=Contact("c3", "L2", area_mm2=area_mm2))
    limits = SafetyLimits(max_amplitude_ua=max_amp, max_charge_density_uc_cm2=60.0)
    return space, shape, montage, limits


def make_opt(**kw):
    space, shape, montage, limits = setup()
    return StimOptimizer(space, shape, montage, limits,
                         **{"noise": 0.06, "seed": 0, **kw})


# --------------------------------------------------------------------------- #
# гауссов процесс
# --------------------------------------------------------------------------- #

def test_gp_recovers_a_known_function():
    rng = np.random.default_rng(0)
    x = rng.random((40, 2))
    y = np.sin(6 * x[:, 0]) + x[:, 1]
    gp = GP(noise=0.01).fit(x, y)
    xs = rng.random((200, 2))
    mu, sd = gp.predict(xs)
    truth = np.sin(6 * xs[:, 0]) + xs[:, 1]
    assert np.mean(np.abs(mu - truth)) < 0.15
    assert np.all(sd > 0)


def test_gp_uncertainty_grows_away_from_data():
    """Без этого EI не имеет смысла: исследовать станет нечего."""
    x = np.array([[0.5, 0.5]])
    gp = GP(noise=0.01).fit(x, np.array([1.0]))
    _, sd_near = gp.predict(np.array([[0.5, 0.52]]))
    _, sd_far = gp.predict(np.array([[0.05, 0.95]]))
    assert sd_far[0] > sd_near[0]


def test_expected_improvement_is_non_negative_and_rewards_uncertainty():
    mu = np.array([0.5, 0.5])
    sd = np.array([0.01, 0.30])
    ei = expected_improvement(mu, sd, best=0.5)
    assert np.all(ei >= 0)
    assert ei[1] > ei[0], "при равном среднем должна выигрывать неизвестность"


def test_gp_downweights_prior_observations():
    """
    Данные с других животных должны информировать, но не доминировать. Проверка:
    при конфликте с собственным наблюдением предсказание тянется к своему.
    """
    x = np.array([[0.5, 0.5], [0.5, 0.5]])
    y = np.array([1.0, 0.0])                       # своё = 1.0, чужое = 0.0
    gp = GP(noise=0.05).fit(x, y, w=np.array([1.0, 0.05]))
    mu, _ = gp.predict(np.array([[0.5, 0.5]]))
    assert mu[0] > 0.5, f"предсказание {mu[0]:.3f} ушло к чужому наблюдению"


def test_gp_shrinks_noisy_observations():
    """
    Свойство, ради которого инкумбент берётся по предсказанию GP, а не по
    максимуму наблюдений: при шуме предсказание в опробованных точках сжато к
    среднему, поэтому максимум предсказаний ниже максимума наблюдений. Иначе
    инкумбент систематически завышен на величину шумового выброса, и EI
    занижает выгоду исследования.
    """
    rng = np.random.default_rng(3)
    x = rng.random((30, 2))
    truth = np.sin(4 * x[:, 0])
    y = truth + rng.normal(0, 0.25, x.shape[0])
    gp = GP(noise=0.25).fit(x, y)
    mu, _ = gp.predict(x)

    # Строгого "<" мало: без шумового члена GP интерполирует, mu.max() ровно
    # равен y.max(), и тест проходит вхолостую. Требуем ЗАМЕТНОГО подтягивания:
    # измерено 19% пути к медиане при шуме 0.25 и 0% при интерполяции.
    pull = (y.max() - mu.max()) / (y.max() - np.median(y))
    assert pull > 0.10, f"выброс почти не сглажен (подтянут на {100 * pull:.0f}%)"
    assert mu.max() > np.median(y), "GP сгладил всё в среднее"


# --------------------------------------------------------------------------- #
# безопасность: свойства конструкции
# --------------------------------------------------------------------------- #

def test_never_suggests_unsafe_parameters():
    """
    Небезопасная точка не должна предлагаться ВООБЩЕ, даже на разогреве.

    Пределы взяты ЖЁСТКИМИ намеренно: при мягких весь диапазон поиска и так
    безопасен, фильтру нечего отбраковывать, и тест проходит даже с выключенным
    фильтром. На этом он уже один раз оказался декоративным - мутация "снять
    проверку безопасности" его не роняла. Поэтому ниже стоит guard, который
    падает, если пределы перестали резать пространство.
    """
    space, shape, montage = setup()[0], setup()[1], setup()[2]
    tight = SafetyLimits(max_amplitude_ua=700.0, max_charge_density_uc_cm2=30.0)

    rng = np.random.default_rng(0)
    bounds = np.asarray(space.as_bounds(), float)
    probe = bounds[:, 0] + rng.random((400, 4)) * (bounds[:, 1] - bounds[:, 0])
    unsafe = sum(0 if from_vector(v, space, shape, montage).is_safe(tight) else 1
                 for v in probe)
    assert unsafe > 40, ("пределы не режут пространство поиска: тест выродится "
                         f"(небезопасных {unsafe} из 400)")

    opt = StimOptimizer(space, shape, montage, tight, noise=0.06, seed=1,
                        max_amplitude_step_ua=1e9)
    for _ in range(40):
        p = opt.suggest()
        assert p.is_safe(tight), f"предложен небезопасный набор: {p.violations(tight)}"
        opt.observe(p, float(rng.normal()))


def test_amplitude_ramp_is_enforced():
    """
    На животном амплитуду наращивают постепенно. Прыжок с нижней границы к
    максимуму недопустим, даже если оптимизатору там интереснее.
    """
    step = 80.0
    opt = make_opt(max_amplitude_step_ua=step, seed=3)
    lo = opt.space.amplitude_ua.low
    tried = [lo]
    for _ in range(25):
        p = opt.suggest()
        assert p.amplitude_ua <= max(tried) + step + 1e-6, (
            f"прыжок амплитуды: {p.amplitude_ua:.0f} при потолке "
            f"{max(tried) + step:.0f}")
        tried.append(p.amplitude_ua)
        opt.observe(p, 0.5)


def test_raises_when_safety_leaves_no_candidates():
    """Лучше явная ошибка, чем тихая выдача чего попало."""
    space, shape, montage, _ = setup(area_mm2=0.01)
    impossible = SafetyLimits(max_charge_density_uc_cm2=0.001)
    opt = StimOptimizer(space, shape, montage, impossible, seed=0)
    with pytest.raises(RuntimeError, match="допустимых кандидатов"):
        opt.suggest()


# --------------------------------------------------------------------------- #
# поведение оптимизатора
# --------------------------------------------------------------------------- #

def test_is_deterministic_for_a_given_seed():
    a = [make_opt(seed=7).suggest().amplitude_ua for _ in range(3)]
    assert len(set(a)) == 1, "один и тот же seed даёт разные предложения"


def test_warmup_explores_before_modelling():
    """Разогрев должен давать РАЗНЫЕ точки, иначе GP нечему учиться."""
    opt = make_opt(n_warmup=6, seed=5)
    amps = []
    for _ in range(6):
        p = opt.suggest()
        amps.append(p.amplitude_ua)
        opt.observe(p, 0.5)
    assert len(set(np.round(amps, 3))) >= 4


def test_finds_optimum_on_a_clean_objective():
    """
    Без шума и без ограничения на наращивание оптимизатор обязан подойти к
    оптимуму. Если не сходится здесь - дело в самом алгоритме, а не в шуме.
    """
    space, shape, montage, limits = setup()
    opt = StimOptimizer(space, shape, montage, limits, noise=1e-4, seed=0,
                        max_amplitude_step_ua=1e9)
    target = 375.0
    for _ in range(40):
        p = opt.suggest()
        opt.observe(p, float(np.exp(-((p.amplitude_ua - target) / 60.0) ** 2)))
    best, _ = opt.best()
    assert abs(best.amplitude_ua - target) < 90.0, (
        f"оптимум не найден: {best.amplitude_ua:.0f} вместо {target:.0f}")


def test_prior_does_not_count_as_own_experience():
    """
    Чужие пробы не должны отменять разогрев на своём животном и не должны
    попадать в best(): иначе "лучшим" окажется набор, который на этом животном
    вообще не проверяли.
    """
    opt = make_opt(n_warmup=5, seed=0)
    space, shape, montage, limits = setup()
    p = from_vector([300.0, 40.0, 100.0, 20.0], space, shape, montage)
    opt.add_prior([(p, 99.0)])
    assert opt.best() == (None, float("-inf"))
    own = opt.suggest()
    opt.observe(own, 0.1)
    got, sc = opt.best()
    assert sc == 0.1 and got is own


def test_history_records_both_given_and_derived():
    opt = make_opt(seed=0)
    p = opt.suggest()
    opt.observe(p, 0.42)
    h = opt.history()[0]
    assert h["score"] == 0.42 and h["source"] == "self"
    for k in ("amplitude_ua", "interpulse_ms", "charge_density_uc_cm2"):
        assert k in h
