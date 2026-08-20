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
                         SearchSpace, from_vector, normalized_dose)


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


# --------------------------------------------------------------------------- #
# разогрев: покрытие как гарантия, а не как везение
# --------------------------------------------------------------------------- #

def test_latin_hypercube_is_stratified():
    """
    По каждой оси ровно одна точка в каждой из n равных полос. Это и есть всё
    отличие от случайной выборки, и именно оно чинит отказы.
    """
    opt = make_opt(seed=0)
    n = 12
    u = opt._latin_hypercube(n)
    assert u.shape == (n, 4)
    for j in range(u.shape[1]):
        bins = np.floor(u[:, j] * n).astype(int)
        assert len(set(bins)) == n, f"ось {j}: полосы {sorted(bins)}"


def test_warmup_plan_covers_every_axis():
    """
    То же свойство, но на готовом плане, в физических единицах.

    Зачем этот тест существует. Со случайным разогревом из шести точек четверть
    прогонов на стенде заканчивалась счётом около нуля: разогрев промахивался
    мимо узкого хребта пользы, GP выучивал только штраф за дозу и сходился в
    угол "почти не стимулировать". Ни увеличение xi, ни принудительное
    исследование это не лечили - лечило только гарантированное покрытие.
    Подробности и цифры в шапке stim_optimizer.
    """
    for seed in range(5):
        opt = make_opt(n_warmup=12, seed=seed)
        plan = np.array(opt.warmup_plan())
        assert len(plan) == 12
        b = np.asarray(opt.space.as_bounds(), float)
        lo, hi = b[:, 0], b[:, 1]
        for j, name in enumerate(opt.space.names()):
            k = np.floor((plan[:, j] - lo[j]) / (hi[j] - lo[j]) * 12).astype(int)
            k = np.clip(k, 0, 11)
            assert len(set(k)) == 12, f"seed {seed}, ось {name}: {sorted(k)}"


def test_warmup_is_a_staircase_and_is_never_clipped():
    """
    План отсортирован по возрастанию амплитуды не для красоты: разгон не пускает
    выше чем на шаг от опробованного. При обходе в случайном порядке высокие
    точки срезались бы по потолку и покрытие по амплитуде разваливалось бы.
    """
    opt = make_opt(n_warmup=12, seed=2)
    plan = np.array(opt.warmup_plan())
    assert np.all(np.diff(plan[:, 0]) >= 0), "план не отсортирован по амплитуде"
    for i in range(len(plan)):
        p = opt.suggest()
        assert p.amplitude_ua == pytest.approx(plan[i, 0]), (
            f"проба {i}: разгон срезал план {plan[i, 0]:.0f} -> {p.amplitude_ua:.0f}")
        opt.observe(p, 0.0)


def test_transfer_does_not_shorten_warmup():
    """
    Соблазн засчитать чужие пробы за покрытие и начать оптимизировать раньше
    измерен и отвергнут: укороченный разогрев давал 0.517 к пробе 15 против
    0.548 у полного. Оптимум у каждого животного свой, чужими точками своё
    пространство не покрывается.
    """
    space, shape, montage, limits = setup()
    prior_pt = from_vector([300.0, 40.0, 100.0, 20.0], space, shape, montage)

    cold = make_opt(n_warmup=6, seed=0)
    warm = make_opt(n_warmup=6, seed=0)
    warm.add_prior([(prior_pt, 0.5)] * 50)
    assert len(warm.warmup_plan()) == len(cold.warmup_plan()) == 6


# --------------------------------------------------------------------------- #
# отклик не постоянен: забывание и правило отчёта
# --------------------------------------------------------------------------- #

def test_old_trials_are_forgotten():
    """
    Гауссов процесс сам одинаково верит первой пробе и последней. Животное за
    сессию устаёт и оптимум уезжает, поэтому вес своей пробы обязан падать с
    возрастом. Без этого промах на уезжающем оптимуме 0.193 вместо 0.126.
    """
    space, shape, montage, limits = setup()
    p = from_vector([300.0, 40.0, 100.0, 20.0], space, shape, montage)

    opt = make_opt(seed=0, forget_half_life=10.0)
    for _ in range(21):
        opt.observe(p, 0.5)
    _, _, w = opt._observations()
    assert w[-1] == pytest.approx(1.0), "свежая проба должна весить единицу"
    assert w[-11] == pytest.approx(0.5, abs=1e-6), "через период - половину"
    assert w[0] == pytest.approx(0.25, abs=1e-6), "через два периода - четверть"
    assert np.all(np.diff(w) > 0), "вес обязан расти к свежим пробам"

    off = make_opt(seed=0, forget_half_life=None)
    for _ in range(21):
        off.observe(p, 0.5)
    assert np.allclose(off._observations()[2], 1.0), "выключенное забывание должно быть выключено"

    # По умолчанию забывание ВКЛЮЧЕНО. Это решение по замеру, а не вкус:
    # на неподвижной поверхности оно бесплатно, на уезжающей вдвое снижает
    # промах. Если менять умолчание - сначала перегнать validate_stim_drift.py.
    default = make_opt(seed=0)
    for _ in range(21):
        default.observe(p, 0.5)
    w_def = default._observations()[2]
    assert not np.allclose(w_def, 1.0), (
        "забывание по умолчанию выключено - см. validate_stim_drift.py")
    assert w_def[0] < 0.5 * w_def[-1]


def test_report_uses_smoothed_prediction_not_best_observation():
    """
    Ловушка проклятия победителя. Отклика нет вообще - счёт это чистый шум.
    Максимум шести десятков шумов заметно больше нуля, и если отчитываться им,
    эффект "получается" там, где его нет: на стенде +0.130.

    Правило отчёта - recommend(), сглаженное предсказание. Тест сравнивает оба
    на одних и тех же данных, поэтому не зависит от траектории поиска.
    """
    space, shape, montage, limits = setup()
    claims_naive, claims_smooth = [], []
    for seed in range(4):
        opt = StimOptimizer(space, shape, montage, limits, noise=0.06,
                            seed=seed, n_candidates=256)
        rng = np.random.default_rng(seed + 500)
        for _ in range(40):
            p = opt.suggest()
            opt.observe(p, float(rng.normal(0.0, 0.06)))   # эффекта нет
        claims_naive.append(opt.best()[1])
        _, mu, _ = opt.recommend()
        claims_smooth.append(mu)
    naive, smooth = float(np.mean(claims_naive)), float(np.mean(claims_smooth))
    assert naive > 0.08, (
        f"стенд не воспроизводит ловушку: наивный отчёт {naive:.3f}")
    assert smooth < 0.5 * naive, (
        f"сглаженный отчёт {smooth:.3f} не лучше наивного {naive:.3f}")


def _ridge_response(p, limits):
    """
    Стенд для проверки на отказ: узкий хребет пользы плюс гладкий штраф за дозу.
    Ловушка именно в их сочетании - штраф выучивается легко, польза нет.
    """
    a = (p.amplitude_ua - 100.0) / 500.0
    f = (p.frequency_hz - 20.0) / 60.0
    t = (p.train_ms - 50.0) / 150.0
    c = p.interchannel_ms / 60.0
    eff = (np.exp(-((a - 0.55) ** 2) / 0.055)
           * np.exp(-((f - 0.40) ** 2) / 0.10)
           * (1.0 - np.exp(-3.0 * t))
           * np.exp(-((c - 0.35) ** 2) / 0.20))
    return 0.95 * eff - 0.20 * normalized_dose(p, limits)


def test_never_collapses_to_the_minimum_dose_corner():
    """
    Регрессия на найденный отказ, сквозная.

    "Угол" - это набор с минимальной дозой: пользы нет, штрафа тоже почти нет,
    счёт около нуля. Для оптимизатора это локальный оптимум, из которого EI сам
    не выходит: угол одновременно и лучший по среднему, и лучше всех изучен.

    Замер в этой самой конфигурации (20 прогонов по 30 проб, 512 кандидатов):

        разогрев планом по гиперкубу   1 отказ  (мин  0.048)
        план, но точки случайные       2 отказа (мин -0.014)
        старый код, без плана          5 отказов(мин -0.007)

    Порог 2 стоит между "починено" и "старый код" с запасом в три прогона.
    Промежуточный вариант (план из случайных точек) этот тест не ловит - его
    стережёт test_warmup_plan_covers_every_axis, где свойство проверяется точно,
    а не статистически.

    В боевой конфигурации (4096 кандидатов, 60 проб) отказов 0 из 40, но такой
    прогон идёт шесть минут и в наборе тестов ему не место.
    """
    space, shape, montage, limits = setup()
    best = []
    for seed in range(20):
        opt = StimOptimizer(space, shape, montage, limits, noise=0.06,
                            seed=seed, n_candidates=512)
        rng = np.random.default_rng(seed + 10_000)
        top = -np.inf
        for _ in range(30):
            p = opt.suggest()
            true = _ridge_response(p, limits)
            opt.observe(p, true + rng.normal(0.0, 0.06))
            top = max(top, true)
        best.append(top)
    best = np.asarray(best)
    bad = int((best < 0.30).sum())
    assert bad <= 2, (
        f"{bad} прогонов из 20 свалились в угол минимальной дозы "
        f"(норма <= 2), худший счёт {best.min():.3f}")


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
