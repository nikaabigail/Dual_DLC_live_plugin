"""
Схема параметров электрической стимуляции.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Трёх полей "частота, амплитуда, интервал" не хватает.
В серии импульсов ПЯТЬ физически разных пауз, и каждая делает своё:

    interphase        мкс   пауза между фазами ОДНОГО импульса
    interpulse        мс    между импульсами внутри серии (= 1000 / частота)
    interchannel      мс    сдвиг второго канала относительно первого
    intertrain        мс    между сериями
    intertrial        с     отдых между пробами

Одно поле "interval" неоднозначно: через полгода не восстановить, что меняли.
Поэтому они разведены явно, а частота и interpulse связаны формулой, а не
дублируются полями (иначе рассинхронизируются).

ТРИ ГРУППЫ. Зафиксированное до эксперимента (форма импульса, монтаж),
подбираемое оптимизатором (амплитуда, частота, длительность серии, межканальный
сдвиг) и производное (заряд, плотность заряда, duty cycle). Производное не
хранится, а считается свойством.

БЕЗОПАСНОСТЬ - ЖЁСТКИЕ ОГРАНИЧЕНИЯ, А НЕ ШТРАФЫ. Плотность заряда и потолок
амплитуды проверяются до выдачи; оптимизатор не имеет права их нарушить.
Значения пределов по умолчанию - ЗАГЛУШКИ, их надо задать из протокола на
животных и документации электрода до первой сессии.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


# --------------------------------------------------------------------------- #
# зафиксировано до эксперимента
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PulseShape:
    """Электрическая форма одного импульса."""

    mode: str = "current"                 # current | voltage
    biphasic: bool = True
    charge_balanced: bool = True
    cathodic_first: bool = True
    phase_width_us: float = 200.0         # длительность одной фазы
    interphase_us: float = 50.0           # ПАУЗА 1 из 5: внутри импульса
    ramp_up_ms: float = 0.0
    ramp_down_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("current", "voltage"):
            raise ValueError(f"mode: current или voltage, задано {self.mode!r}")
        if self.phase_width_us <= 0:
            raise ValueError("phase_width_us должна быть положительной")
        if self.interphase_us < 0:
            raise ValueError("interphase_us не может быть отрицательной")


@dataclass(frozen=True)
class Contact:
    """Один контакт электрода."""

    contact_id: str
    segment: str                          # L2, S1, ...
    side: str = "ipsi"                    # ipsi | contra | midline
    area_mm2: float = 0.05
    impedance_kohm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.side not in ("ipsi", "contra", "midline"):
            raise ValueError(f"side: ipsi/contra/midline, задано {self.side!r}")
        if self.area_mm2 <= 0:
            raise ValueError("area_mm2 должна быть положительной")


@dataclass(frozen=True)
class Montage:
    """Пространственная конфигурация одного канала стимуляции."""

    name: str
    cathode: Contact
    anode: Optional[Contact] = None       # None = monopolar, возврат на корпус
    return_electrode: str = "chassis"

    @property
    def is_monopolar(self) -> bool:
        return self.anode is None


# --------------------------------------------------------------------------- #
# подбирается оптимизатором
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Bounds:
    """Границы одного непрерывного параметра поиска."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"пустой диапазон: [{self.low}, {self.high}]")

    def clip(self, x: float) -> float:
        return min(max(float(x), self.low), self.high)

    def contains(self, x: float) -> bool:
        return self.low <= float(x) <= self.high


@dataclass(frozen=True)
class SearchSpace:
    """
    Что варьируем. Четыре непрерывных параметра - столько же варьировал
    Drainville 2025 на латеральной гемисекции (амплитуда, длительность серии,
    частота, тайминг). Тайминг у нас вынесен отдельно и задаётся процентом
    фазы шага, поэтому вместо него здесь межканальный сдвиг.

    Больше четырёх брать не стоит: на 20-50 пробах за сессию гауссов процесс
    в пространстве большей размерности не сойдётся.
    """

    amplitude_ua: Bounds = field(default_factory=lambda: Bounds(100.0, 600.0))
    frequency_hz: Bounds = field(default_factory=lambda: Bounds(20.0, 80.0))
    train_ms: Bounds = field(default_factory=lambda: Bounds(50.0, 200.0))
    interchannel_ms: Bounds = field(default_factory=lambda: Bounds(0.0, 60.0))

    def names(self) -> list[str]:
        return ["amplitude_ua", "frequency_hz", "train_ms", "interchannel_ms"]

    def as_bounds(self) -> list[tuple[float, float]]:
        return [(getattr(self, n).low, getattr(self, n).high) for n in self.names()]


# --------------------------------------------------------------------------- #
# ограничения
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SafetyLimits:
    """Жёсткие пределы. ЗАГЛУШКИ - заменить значениями из протокола."""

    max_amplitude_ua: float = 800.0
    max_charge_per_phase_nc: float = 200.0
    max_charge_density_uc_cm2: float = 30.0
    max_duty_cycle: float = 0.5
    require_charge_balanced: bool = True


# --------------------------------------------------------------------------- #
# полный набор
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StimParams:
    """Полный набор параметров одной пробы: всё, что нужно, чтобы её повторить."""

    shape: PulseShape
    montage: Montage
    amplitude_ua: float
    frequency_hz: float
    train_ms: float
    intertrain_ms: float = 500.0          # ПАУЗА 4 из 5
    intertrial_s: float = 60.0            # ПАУЗА 5 из 5
    interchannel_ms: float = 0.0          # ПАУЗА 3 из 5
    trains_per_trial: int = 1

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz должна быть положительной")
        if self.train_ms <= 0:
            raise ValueError("train_ms должна быть положительной")
        if self.amplitude_ua < 0:
            raise ValueError("amplitude_ua не может быть отрицательной")

    # ---- производные ------------------------------------------------------
    @property
    def interpulse_ms(self) -> float:
        """ПАУЗА 2 из 5. Не хранится: однозначно задана частотой."""
        return 1000.0 / self.frequency_hz

    @property
    def pulses_per_train(self) -> int:
        return max(1, int(round(self.train_ms / self.interpulse_ms)))

    @property
    def charge_per_phase_nc(self) -> float:
        """Q = I * t. мкА * мкс = пКл, переводим в нКл."""
        return self.amplitude_ua * self.shape.phase_width_us / 1000.0

    @property
    def charge_density_uc_cm2(self) -> float:
        """нКл/мм2 -> мкКл/см2: /1000 (нКл->мкКл), *100 (мм2->см2)."""
        return self.charge_per_phase_nc / self.montage.cathode.area_mm2 / 10.0

    @property
    def duty_cycle(self) -> float:
        period = self.train_ms + self.intertrain_ms
        return self.train_ms / period if period > 0 else 1.0

    # ---- проверка ---------------------------------------------------------
    def violations(self, limits: SafetyLimits) -> list[str]:
        """Список нарушений. Пустой список = набор допустим."""
        out: list[str] = []
        if self.amplitude_ua > limits.max_amplitude_ua:
            out.append(f"амплитуда {self.amplitude_ua:.0f} > "
                       f"{limits.max_amplitude_ua:.0f} мкА")
        if self.charge_per_phase_nc > limits.max_charge_per_phase_nc:
            out.append(f"заряд {self.charge_per_phase_nc:.1f} > "
                       f"{limits.max_charge_per_phase_nc:.1f} нКл/фазу")
        if self.charge_density_uc_cm2 > limits.max_charge_density_uc_cm2:
            out.append(f"плотность заряда {self.charge_density_uc_cm2:.1f} > "
                       f"{limits.max_charge_density_uc_cm2:.1f} мкКл/см2")
        if self.duty_cycle > limits.max_duty_cycle:
            out.append(f"duty cycle {self.duty_cycle:.2f} > {limits.max_duty_cycle:.2f}")
        if limits.require_charge_balanced and not self.shape.charge_balanced:
            out.append("стимуляция не charge-balanced")
        return out

    def is_safe(self, limits: SafetyLimits) -> bool:
        return not self.violations(limits)

    # ---- журналирование ---------------------------------------------------
    def to_record(self) -> dict:
        """Плоская запись для лога пробы: и заданное, и производное."""
        rec = asdict(self)
        rec.update({
            "interpulse_ms": round(self.interpulse_ms, 4),
            "pulses_per_train": self.pulses_per_train,
            "charge_per_phase_nc": round(self.charge_per_phase_nc, 4),
            "charge_density_uc_cm2": round(self.charge_density_uc_cm2, 4),
            "duty_cycle": round(self.duty_cycle, 4),
        })
        return rec

    def to_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, sort_keys=True)


def from_vector(x: Sequence[float], space: SearchSpace, shape: PulseShape,
                montage: Montage, **rest) -> StimParams:
    """
    Вектор оптимизатора -> набор параметров. Значения обрезаются по границам:
    численный шум оптимизатора не должен выносить за диапазон поиска.
    """
    names = space.names()
    if len(x) != len(names):
        raise ValueError(f"ожидалось {len(names)} значений ({names}), дано {len(x)}")
    vals = {n: getattr(space, n).clip(v) for n, v in zip(names, x)}
    return StimParams(shape=shape, montage=montage, **vals, **rest)


def normalized_dose(p: StimParams, limits: SafetyLimits) -> float:
    """
    Доза в долях от предела, 0..1+. Нужна счёту походки как штраф: при равном
    результате должна побеждать более щадящая стимуляция.
    """
    parts = [
        p.charge_density_uc_cm2 / limits.max_charge_density_uc_cm2,
        p.duty_cycle / limits.max_duty_cycle,
    ]
    return float(sum(parts) / len(parts))
