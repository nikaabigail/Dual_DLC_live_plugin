"""
Тесты схемы параметров стимуляции и счёта походки.

Данные синтетические, GPU и записи не нужны. Каждый тест закрывает конкретную
ошибку, которая уже случалась или которую легко допустить:

  * пять пауз перепутать между собой (одно поле "интервал");
  * забыть, что заряд и плотность заряда - производные, и рассинхронизировать;
  * позволить оптимизатору выйти за пределы безопасности;
  * посчитать счёт по выжившим циклам - на этом проверка монотонности уже
    поймала реальный дефект: при сильном дрожании ритма счёт РОС.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gait_score as GS
from gait_score import GaitMetrics, IntactNorm, measure, measure_norm, score
from stim_params import (Bounds, Contact, Montage, PulseShape, SafetyLimits,
                         SearchSpace, StimParams, from_vector, normalized_dose)

FPS = 100.0
STANCE_N, SWING_N = 30, 10
CYCLE_N = STANCE_N + SWING_N


def make_params(**over) -> StimParams:
    kw = dict(shape=PulseShape(),
              montage=Montage("L2", cathode=Contact("c3", "L2")),
              amplitude_ua=300.0, frequency_hz=40.0, train_ms=100.0)
    kw.update(over)
    return StimParams(**kw)


# --------------------------------------------------------------------------- #
# схема параметров
# --------------------------------------------------------------------------- #

def test_five_intervals_are_separately_addressable():
    """
    Пять пауз должны существовать по отдельности. Одно поле "интервал"
    неоднозначно: через полгода не восстановить, что именно меняли.
    """
    p = make_params(intertrain_ms=400.0, intertrial_s=90.0, interchannel_ms=25.0)
    assert p.shape.interphase_us == 50.0          # 1: внутри импульса
    assert p.interpulse_ms == pytest.approx(25.0)  # 2: 1000/40 Гц
    assert p.interchannel_ms == 25.0               # 3: между каналами
    assert p.intertrain_ms == 400.0                # 4: между сериями
    assert p.intertrial_s == 90.0                  # 5: между пробами
    # совпадение чисел у 2 и 3 не должно их смешивать: это разные поля
    assert p.interpulse_ms is not p.interchannel_ms


def test_interpulse_follows_frequency_and_is_not_stored():
    """Частота и interpulse - одно и то же; хранить оба значит рассинхронизировать."""
    assert make_params(frequency_hz=50.0).interpulse_ms == pytest.approx(20.0)
    assert make_params(frequency_hz=25.0).interpulse_ms == pytest.approx(40.0)
    assert "interpulse_ms" not in StimParams.__dataclass_fields__


def test_derived_charge_math():
    """Q = I*t; плотность - на площадь катода. Числа считаем руками."""
    p = make_params(amplitude_ua=300.0)            # 300 мкА * 200 мкс
    assert p.charge_per_phase_nc == pytest.approx(60.0)
    # 60 нКл / 0.05 мм2 -> мкКл/см2
    assert p.charge_density_uc_cm2 == pytest.approx(120.0)
    assert p.pulses_per_train == 4                 # 100 мс при 25 мс между импульсами
    assert p.duty_cycle == pytest.approx(100.0 / 600.0)


def test_safety_limits_catch_violations():
    strict = SafetyLimits(max_amplitude_ua=200.0, max_charge_density_uc_cm2=30.0)
    bad = make_params(amplitude_ua=500.0)
    v = bad.violations(strict)
    assert not bad.is_safe(strict)
    assert any("амплитуда" in x for x in v)
    assert any("плотность" in x for x in v)


def test_duty_cycle_limit_is_enforced():
    p = make_params(train_ms=200.0, intertrain_ms=50.0)     # duty = 0.8
    assert any("duty" in x for x in p.violations(SafetyLimits(max_duty_cycle=0.5)))


def test_charge_balance_is_required_by_default():
    p = make_params(shape=PulseShape(charge_balanced=False))
    assert any("charge-balanced" in x for x in p.violations(SafetyLimits()))


def test_from_vector_clips_to_search_space():
    """Численный шум оптимизатора не должен выносить за диапазон поиска."""
    space = SearchSpace(amplitude_ua=Bounds(100.0, 600.0))
    p = from_vector([1e6, 40.0, 100.0, 0.0], space, PulseShape(),
                    Montage("L2", cathode=Contact("c3", "L2")))
    assert p.amplitude_ua == 600.0
    p2 = from_vector([-50.0, 40.0, 100.0, 0.0], space, PulseShape(),
                     Montage("L2", cathode=Contact("c3", "L2")))
    assert p2.amplitude_ua == 100.0


def test_from_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        from_vector([300.0, 40.0], SearchSpace(), PulseShape(),
                    Montage("L2", cathode=Contact("c3", "L2")))


def test_record_carries_both_given_and_derived():
    rec = make_params().to_record()
    for k in ("amplitude_ua", "frequency_hz", "interpulse_ms",
              "charge_per_phase_nc", "charge_density_uc_cm2", "duty_cycle"):
        assert k in rec, f"в журнале пробы нет поля {k}"


def test_dose_grows_with_amplitude():
    lim = SafetyLimits()
    assert normalized_dose(make_params(amplitude_ua=500.0), lim) > \
           normalized_dose(make_params(amplitude_ua=200.0), lim)


# --------------------------------------------------------------------------- #
# счёт походки
# --------------------------------------------------------------------------- #

def synth_gait(n_cycles=30, clearance=12.0, span=60.0, seed=0, jitter=0):
    """
    Носок и iliac для интактного шага. iliac неподвижен, носок описывает петлю:
    в опоре едет назад по прямой, в переносе возвращается вперёд с подъёмом.
    """
    rng = np.random.default_rng(seed)
    bx, by = 500.0, 60.0
    tx, ty, td, lo = [], [], [], []
    f = 0
    for _ in range(n_cycles):
        st = STANCE_N + (int(rng.integers(-jitter, jitter + 1)) if jitter else 0)
        sw = SWING_N + (int(rng.integers(-1, 2)) if jitter else 0)
        td.append(f)
        for k in range(st):                                  # опора
            tx.append(bx + span / 2 - span * k / max(1, st - 1))
            ty.append(by + 120.0)
        lo.append(f + st)
        for k in range(sw):                                  # перенос
            tx.append(bx - span / 2 + span * (k + 1) / sw)
            ty.append(by + 120.0 - clearance * np.sin(np.pi * (k + .5) / sw))
        f += st + sw
    td.append(f)
    n = len(tx)
    return (np.array(tx), np.array(ty), np.full(n, bx), np.full(n, by),
            np.array(td, float), np.array(lo, float))


def measure_synth(**kw):
    tx, ty, bx, by, td, lo = synth_gait(**kw)
    return measure(tx, ty, bx, by, td, lo, l_ref=64.0, fps=FPS)


def test_measure_recovers_known_geometry():
    m = measure_synth(clearance=12.0, span=60.0)
    assert m.n_cycles > 20
    assert m.clearance_lref == pytest.approx(12.0 / 64.0, rel=.15)
    assert m.step_length_lref == pytest.approx(60.0 / 64.0, rel=.15)
    assert m.cycle_yield == pytest.approx(1.0)


@pytest.mark.parametrize("field,worse", [("clearance", 4.0), ("span", 25.0)])
def test_score_falls_when_gait_degrades(field, worse):
    """Волочение и укороченный шаг обязаны снижать счёт."""
    norm = measure_norm([measure_synth()])
    good = score(measure_synth(), norm)
    bad = score(measure_synth(**{field: worse}), norm)
    assert bad < good, f"{field}: ухудшение не снизило счёт ({bad:.3f} >= {good:.3f})"


def test_cycle_yield_penalises_unrecognised_steps():
    """
    Регрессия на реальный дефект. Раньше счёт считался только по распознанным
    циклам, поэтому при развале ритма он РОС: выживали самые чистые шаги.
    Теперь доля нераспознанных входит множителем.
    """
    norm = measure_norm([measure_synth()])
    m = measure_synth()
    full = score(m, norm)

    half = GaitMetrics(**{**m.as_dict(), "n_cycles": m.n_cycles})
    half.cycle_yield = 0.5
    assert score(half, norm) == pytest.approx(full * 0.5, rel=1e-6)
    assert score(half, norm) < full


def test_variability_penalty_is_zero_at_and_below_the_norm():
    """
    Изменчивость штрафуется ОТНОСИТЕЛЬНО нормы. У интакта свой ненулевой
    разброс, и абсолютный штраф просто сдвигал бы шкалу.

    Свойство, которое это отличает: на норме и ниже нормы штрафа нет вовсе,
    поэтому счёт для обоих случаев ОДИНАКОВ. При абсолютном штрафе более
    ровная походка получила бы больше, и равенство сломалось бы.
    """
    # Синтетика строго периодична, поэтому её собственный cycle_cv = 0 и тест
    # был бы вхолостую: обе формулы дали бы ноль. Задаём норме реальный
    # ненулевой разброс, измеренный на записи.
    norm = replace(measure_norm([measure_synth()]), cycle_cv=0.20)
    m = measure_synth()

    at_norm = GaitMetrics(**m.as_dict())
    at_norm.cycle_cv = at_norm.step_cv = norm.cycle_cv
    below = GaitMetrics(**m.as_dict())
    below.cycle_cv = below.step_cv = norm.cycle_cv * 0.4

    assert at_norm.cycle_cv > 0, "норма должна иметь ненулевой разброс"
    assert score(at_norm, norm) == pytest.approx(score(below, norm))


def test_score_falls_when_rhythm_gets_irregular():
    """Разброс ВЫШЕ нормы обязан снижать счёт."""
    norm = measure_norm([measure_synth()])
    calm = measure_synth()
    shaky = measure_synth(jitter=6, seed=3)
    assert shaky.cycle_cv > calm.cycle_cv
    assert score(shaky, norm) < score(calm, norm)


def test_score_handles_missing_second_leg():
    """
    Одна камера видит одну ногу. Слагаемые про вторую ногу должны отсутствовать,
    а не обнуляться: иначе счёт занижен и несравним с двусторонним.
    """
    norm = measure_norm([measure_synth()])
    m = measure_synth()
    assert m.interlimb_phase is None and m.stance_asymmetry is None
    s = score(m, norm)
    assert 0.5 < s <= 1.05, f"счёт без второй ноги вне разумного диапазона: {s}"


def test_dose_penalty_prefers_gentler_stimulation():
    """При равном результате побеждает меньшая доза."""
    norm = measure_norm([measure_synth()])
    m = measure_synth()
    assert score(m, norm, dose=0.0) > score(m, norm, dose=0.8)


def test_measure_rejects_too_few_cycles():
    tx, ty, bx, by, td, lo = synth_gait(n_cycles=2)
    with pytest.raises(ValueError):
        measure(tx, ty, bx, by, td, lo, l_ref=64.0, fps=FPS)
