"""
Медленная петля: подбор параметров стимуляции байесовской оптимизацией.

ДВЕ ПЕТЛИ, КОТОРЫЕ НЕЛЬЗЯ СМЕШИВАТЬ.
  быстрая  - "когда": фаза шага -> поджиг на целевом проценте -> TTL. 52 Гц,
             детерминированная, никакого обучения. Это контур безопасности.
  медленная (здесь) - "чем": вектор параметров выбирается раз в пробу, а не раз
             в шаг. Здесь и место оптимизатору.
Если дать модели менять амплитуду внутри цикла, теряются и воспроизводимость, и
возможность что-либо доказать про фазовую специфичность.

ПОЧЕМУ ГАУССОВ ПРОЦЕСС, А НЕ НЕЙРОСЕТЬ. Персонализация под животное - это
десятки, максимум пара сотен проб за сессию. Глубокая сеть в таком режиме
переобучится на шум. GP-BO работает от 20-50 проб, сам решает, что пробовать
дальше, и умеет жёсткие ограничения. Для спинальной стимуляции у животных это
показано в Bonizzato et al., Cell Reports Medicine 2023 (doi 10.1016/j.xcrm.2023.101008).

ПОЧЕМУ СВОЙ GP, А НЕ sklearn. Боевое окружение закреплено (пины torch), sklearn
там нет и ставить нельзя. Для 4 измерений и <=100 точек GP это разложение
Холецкого на матрице 100x100 - пишется на numpy и работает где угодно.

БЕЗОПАСНОСТЬ ВСТРОЕНА, А НЕ ВЫУЧЕНА. Небезопасная точка не предлагается вообще:
кандидаты фильтруются по SafetyLimits ДО выбора. Плюс ограничение на шаг по
амплитуде - оптимизатор не может прыгнуть с 150 на 600 мкА, даже если очень
хочется: на животном наращивают постепенно.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from stim_params import (Montage, PulseShape, SafetyLimits, SearchSpace,
                         StimParams, from_vector, normalized_dose)


# --------------------------------------------------------------------------- #
# компактный гауссов процесс
# --------------------------------------------------------------------------- #

def _matern52(a: np.ndarray, b: np.ndarray, ls: float) -> np.ndarray:
    """Matern 5/2: гладкая, но не бесконечно - разумное допущение для отклика."""
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0)) / ls
    return (1.0 + math.sqrt(5) * d + 5.0 / 3.0 * d ** 2) * np.exp(-math.sqrt(5) * d)


class GP:
    """
    Гауссов процесс на нормированном кубе [0,1]^d.

    Длина корреляции подбирается перебором по маргинальному правдоподобию: для
    сетки из восьми значений это дешевле и надёжнее градиентной оптимизации на
    двух десятках точек, где правдоподобие легко имеет несколько максимумов.
    """

    LENGTH_SCALES = (0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 0.9, 1.3)

    def __init__(self, noise: float = 0.06):
        self.noise = float(noise)
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None      # веса наблюдений (1 = своё)
        self.ls = 0.3
        self.y_mean = 0.0
        self.y_scale = 1.0
        self._L = None
        self._alpha = None

    def fit(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "GP":
        self.x = np.atleast_2d(np.asarray(x, float))
        y = np.asarray(y, float).ravel()
        self.w = np.ones_like(y) if w is None else np.asarray(w, float).ravel()
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std()) or 1.0
        self.y = (y - self.y_mean) / self.y_scale

        best = (-np.inf, self.LENGTH_SCALES[0])
        for ls in self.LENGTH_SCALES:
            try:
                L, alpha = self._chol(ls)
            except np.linalg.LinAlgError:
                continue
            # log p(y|X) без констант
            lml = -0.5 * float(self.y @ alpha) - float(np.log(np.diag(L)).sum())
            if lml > best[0]:
                best = (lml, ls)
        self.ls = best[1]
        self._L, self._alpha = self._chol(self.ls)
        return self

    def _chol(self, ls: float):
        k = _matern52(self.x, self.x, ls)
        # шум на диагонали: чем меньше вес наблюдения, тем больше его шум,
        # так данные с других животных информируют, но не доминируют
        k[np.diag_indices_from(k)] += (self.noise / self.y_scale) ** 2 / np.maximum(self.w, 1e-6)
        L = np.linalg.cholesky(k)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return L, alpha

    def predict(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.atleast_2d(np.asarray(xs, float))
        ks = _matern52(xs, self.x, self.ls)
        mu = ks @ self._alpha
        v = np.linalg.solve(self._L, ks.T)
        var = np.maximum(1.0 - (v ** 2).sum(0), 1e-12)
        return mu * self.y_scale + self.y_mean, np.sqrt(var) * self.y_scale


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float,
                         xi: float = 0.01) -> np.ndarray:
    """EI для МАКСИМИЗАЦИИ. erf из stdlib, чтобы не тянуть scipy."""
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return (mu - best - xi) * cdf + sd * pdf


# --------------------------------------------------------------------------- #
# оптимизатор
# --------------------------------------------------------------------------- #

@dataclass
class Trial:
    """Одна проба: что подали и что получили."""

    params: StimParams
    score: float
    vector: np.ndarray
    source: str = "self"                       # self | prior


class StimOptimizer:
    """
    Подбор параметров стимуляции под КОНКРЕТНОЕ животное.

    Персонализация делается переносом: наблюдения с других животных кладутся в
    ту же модель, но с пониженным весом (prior_weight). Это и есть "дообучение
    под животное", просто реализованное как перенос апостериора, а не как
    файнтюнинг сети - на 20-50 пробах иначе не выйдет.
    """

    def __init__(self, space: SearchSpace, shape: PulseShape, montage: Montage,
                 limits: SafetyLimits, *, noise: float = 0.06, seed: int = 0,
                 n_warmup: int = 6, max_amplitude_step_ua: float = 100.0,
                 prior_weight: float = 0.3, n_candidates: int = 4096,
                 fixed: Optional[dict] = None):
        self.space = space
        self.shape = shape
        self.montage = montage
        self.limits = limits
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.n_warmup = int(n_warmup)
        self.max_amp_step = float(max_amplitude_step_ua)
        self.prior_weight = float(prior_weight)
        self.n_candidates = int(n_candidates)
        self.fixed = dict(fixed or {})
        self.trials: list[Trial] = []
        self._bounds = np.asarray(space.as_bounds(), float)
        self._names = space.names()
        self._i_amp = self._names.index("amplitude_ua")

    # ---- преобразования ---------------------------------------------------
    def _to_unit(self, v: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        return (np.asarray(v, float) - lo) / (hi - lo)

    def _to_real(self, u: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        return lo + np.asarray(u, float) * (hi - lo)

    def _params(self, v: Sequence[float]) -> StimParams:
        return from_vector(v, self.space, self.shape, self.montage, **self.fixed)

    # ---- предел наращивания ----------------------------------------------
    def _amplitude_ceiling(self) -> float:
        """
        Выше чем на max_amp_step от уже опробованного не поднимаемся. На
        животном амплитуду наращивают постепенно, а не прыжком к максимуму.
        """
        own = [t.params.amplitude_ua for t in self.trials if t.source == "self"]
        base = max(own) if own else self._bounds[self._i_amp, 0]
        return min(base + self.max_amp_step, self._bounds[self._i_amp, 1])

    def _candidates(self) -> np.ndarray:
        """Случайные точки куба, отфильтрованные по безопасности и потолку."""
        u = self.rng.random((self.n_candidates, len(self._names)))
        real = np.array([self._to_real(x) for x in u])
        ceil = self._amplitude_ceiling()
        keep = []
        for i, v in enumerate(real):
            if v[self._i_amp] > ceil:
                continue
            if self._params(v).is_safe(self.limits):
                keep.append(i)
        return u[keep] if keep else np.empty((0, len(self._names)))

    # ---- основной интерфейс ----------------------------------------------
    def suggest(self) -> StimParams:
        """Следующий набор параметров для пробы."""
        cand = self._candidates()
        if cand.size == 0:
            raise RuntimeError(
                "не осталось допустимых кандидатов: пределы безопасности "
                "исключают всё пространство поиска")

        own = [t for t in self.trials if t.source == "self"]
        if len(own) < self.n_warmup:
            # разогрев: случайные допустимые точки, чтобы GP было на чём учиться
            return self._params(self._to_real(cand[self.rng.integers(len(cand))]))

        x = np.array([self._to_unit(t.vector) for t in self.trials])
        y = np.array([t.score for t in self.trials])
        w = np.array([1.0 if t.source == "self" else self.prior_weight
                      for t in self.trials])
        gp = GP(noise=self.noise).fit(x, y, w)
        mu, sd = gp.predict(cand)

        # Инкумбент берём по СГЛАЖЕННОМУ предсказанию в уже опробованных точках,
        # а не по максимальному наблюдению. Максимум зашумлённых наблюдений
        # систематически завышен (при шуме 0.06 и трёх десятках проб - примерно
        # на 0.1), из-за чего EI занижает выгоду и оптимизатор недоисследует.
        own_x = np.array([self._to_unit(t.vector) for t in own])
        best = float(gp.predict(own_x)[0].max())
        ei = expected_improvement(mu, sd, best)
        return self._params(self._to_real(cand[int(np.argmax(ei))]))

    def observe(self, params: StimParams, score: float, source: str = "self") -> None:
        v = np.array([getattr(params, n) for n in self._names], float)
        self.trials.append(Trial(params=params, score=float(score),
                                 vector=v, source=source))

    def add_prior(self, records: Sequence[tuple[StimParams, float]]) -> None:
        """Наблюдения с других животных: тёплый старт для нового."""
        for p, s in records:
            self.observe(p, s, source="prior")

    def best(self) -> tuple[Optional[StimParams], float]:
        own = [t for t in self.trials if t.source == "self"]
        if not own:
            return None, float("-inf")
        t = max(own, key=lambda z: z.score)
        return t.params, t.score

    def history(self) -> list[dict]:
        return [{**t.params.to_record(), "score": t.score, "source": t.source}
                for t in self.trials]
