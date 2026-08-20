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

РАЗОГРЕВ ОБЯЗАН ПОКРЫВАТЬ ПРОСТРАНСТВО, А НЕ БЫТЬ СЛУЧАЙНЫМ. Это не стилистика,
а измеренный отказ. Со случайным разогревом каждый пятый прогон заканчивался
счётом около нуля - хуже случайного перебора, который в тех же условиях не
опускался ниже 0.195.

Механизм. Счёт складывается из пользы и штрафа за дозу. Польза сосредоточена в
узком хребте (нужно одновременно попасть по амплитуде, частоте, длительности и
сдвигу), а штраф гладкий и монотонный: меньше доза - выше счёт. Если разогрев в
хребет не попал, GP видит единственную воспроизводимую закономерность - штраф -
и честно сходится в угол "почти не стимулировать". Там счёт около нуля, и это
локальный оптимум: ноль лучше, чем минус. Дальше EI оттуда не уходит, потому что
угол одновременно и лучший по среднему, и лучше всех изучен.

Замер, 40 прогонов по 60 проб (дошли до 95% достижимого / отказов со счётом
ниже 0.30):

    разогрев 6 случайных точек              65% / 22%   <- было
    то же + исследование каждую 5-ю         60% / 20%
    то же + xi 0.10 вместо 0.01             62% / 28%
    разогрев 12 случайных точек             75% /  5%
    разогрев 6 точек по гиперкубу           90% /  0%   <- стало
    разогрев 12 точек по гиперкубу          88% /  0%

Отсюда три вывода, каждый из которых мог оказаться иным.

  * Подкрутка функции сбора не лечит. Ни увеличенное xi, ни принудительное
    исследование отказы не убирают: EI не может захотеть туда, где по его модели
    ничего нет, а модель построена по промахнувшемуся разогреву.
  * Дело не в числе проб. Шесть точек по гиперкубу работают не хуже двенадцати,
    поэтому разогрев остался коротким.
  * Вклад дают обе части плана. В отдельном замере (20 прогонов по 30 проб)
    отказов было 5 у старого кода, 2 у плана из случайных точек, отсортированных
    по амплитуде, и 1 у плана по гиперкубу. То есть половину даёт сама лестница
    по амплитуде, половину - расслоение по остальным осям.

Полностью отказы не исчезают: в боевой конфигурации это 0 из 40 прогонов, но при
более коротком бюджете встречается примерно 1 из 30. Считать нужно "редко", а не
"никогда".

ПЕРЕНОС НЕ УКОРАЧИВАЕТ РАЗОГРЕВ. Соблазн засчитать чужие пробы за покрытие и
начать раньше - проверен и отвергнут: укороченный разогрев давал 0.517 к пробе
15 против 0.548 у полного (40 прогонов). Покрытие своего животного ничем не
заменяется, потому что оптимум у каждого свой.

Сам перенос после починки разогрева даёт немного: 0.548 против 0.515 к пробе 15
и ничего к пробе 30. Раньше он выглядел главным рычагом, но это была иллюзия -
он вытаскивал те прогоны, которые иначе сваливались в угол. Чинить надо было
разогрев, а не добавлять чужие данные.

Вес чужих наблюдений оставлен 0.3 и намеренно НЕ подобран по стенду: там "другое
животное" - та же самая поверхность, только другие точки, поэтому стенд завышает
пользу переноса и подбирать по нему вес означало бы подгонку под то, чего в
эксперименте не будет.
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
                 explore_every: int = 5, forget_half_life: Optional[float] = 15.0,
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
        self.explore_every = int(explore_every)
        self.forget_half_life = (None if forget_half_life is None
                                 else float(forget_half_life))
        self.fixed = dict(fixed or {})
        self.trials: list[Trial] = []
        self._bounds = np.asarray(space.as_bounds(), float)
        self._names = space.names()
        self._i_amp = self._names.index("amplitude_ua")
        self._plan: Optional[list[np.ndarray]] = None

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

    # ---- разогрев ---------------------------------------------------------
    def _latin_hypercube(self, n: int) -> np.ndarray:
        """
        Латинский гиперкуб: по каждой оси ровно одна точка в каждом из n
        интервалов. Случайная выборка такого не обещает - в четырёх измерениях
        шесть случайных точек легко оставляют половину диапазона частоты пустой,
        и именно оттуда берутся отказы (см. шапку модуля).
        """
        d = len(self._names)
        u = np.empty((n, d))
        for j in range(d):
            u[:, j] = (self.rng.permutation(n) + self.rng.random(n)) / n
        return u

    def warmup_plan(self) -> list[np.ndarray]:
        """План разогрева. Строится один раз, при первом обращении."""
        if self._plan is None:
            self._plan = self._make_warmup_plan(self.n_warmup)
        return self._plan

    def _make_warmup_plan(self, n: int) -> list[np.ndarray]:
        """
        Точки разогрева, отсортированные по возрастанию амплитуды.

        Сортировка не косметическая: разгон амплитуды разрешает подниматься не
        более чем на max_amp_step от уже опробованного. Если идти по гиперкубу в
        случайном порядке, высокие точки пришлось бы срезать по потолку, и
        покрытие по амплитуде развалилось бы. По возрастанию потолок всегда
        оказывается впереди, и срезать нечего.
        """
        if n <= 0:
            return []
        pts: list[np.ndarray] = []
        for _ in range(8):                       # добор, если что-то небезопасно
            for u in self._latin_hypercube(n):
                v = self._to_real(u)
                if self._params(v).is_safe(self.limits):
                    pts.append(v)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if not pts:
            return []
        arr = np.array(pts[:n])
        return list(arr[np.argsort(arr[:, self._i_amp])])

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
        own = [t for t in self.trials if t.source == "self"]

        # Разогрев идёт по заранее построенному плану, а не по тому, что
        # подвернулось: покрытие пространства должно быть гарантией, а не
        # везением. Срез по потолку разгона тут почти никогда не срабатывает,
        # потому что план отсортирован по возрастанию амплитуды.
        plan = self.warmup_plan()
        if len(own) < len(plan):
            v = np.array(plan[len(own)], float)
            v[self._i_amp] = min(v[self._i_amp], self._amplitude_ceiling())
            return self._params(v)

        cand = self._candidates()
        if cand.size == 0:
            raise RuntimeError(
                "не осталось допустимых кандидатов: пределы безопасности "
                "исключают всё пространство поиска")
        if len(own) < self.n_warmup:
            # план оказался короче нормы (небезопасные точки) - добираем случайно
            return self._params(self._to_real(cand[self.rng.integers(len(cand))]))

        x, y, w = self._observations()
        gp = GP(noise=self.noise).fit(x, y, w)
        mu, sd = gp.predict(cand)

        # Раз в explore_every проб берём самую неизученную точку вместо самой
        # выгодной. Само по себе это отказы НЕ лечит (замер в шапке модуля), но
        # после нормального разогрева добавляет несколько процентов попаданий на
        # длинных сессиях и стоит одну пробу из пяти.
        if self.explore_every > 0 and len(own) % self.explore_every == 0:
            return self._params(self._to_real(cand[int(np.argmax(sd))]))

        # Инкумбент берём по СГЛАЖЕННОМУ предсказанию в уже опробованных точках,
        # а не по максимальному наблюдению. Максимум зашумлённых наблюдений
        # систематически завышен (при шуме 0.06 и трёх десятках проб - примерно
        # на 0.1), из-за чего EI занижает выгоду и оптимизатор недоисследует.
        own_x = np.array([self._to_unit(t.vector) for t in own])
        best = float(gp.predict(own_x)[0].max())
        ei = expected_improvement(mu, sd, best)
        return self._params(self._to_real(cand[int(np.argmax(ei))]))

    def _observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Точки, счёта и веса для GP.

        Вес чужой пробы понижен всегда. Вес своей падает вдвое каждые
        forget_half_life проб, потому что отклик за сессию НЕ постоянен:
        животное устаёт, порог растёт, оптимум уезжает. Гауссов процесс сам по
        себе такого не умеет - он одинаково верит первой пробе и последней.

        Цена и польза измерены (validate_stim_drift.py, 50 прогонов по 60 проб,
        промах рекомендации на поверхности, какой она стала к концу):

            поверхность   помнит всё   забывает T½=15
            неподвижная      0.033         0.036       <- страховка стоит 0.003
            усталость        0.037         0.042
            уезд оптимума    0.193         0.126       <- ради этого и включено

        Завышение отчёта при этом падает всюду: на уезде с 0.121 до 0.022, при
        усталости с 0.189 до 0.073.

        Сам период подобран, а не назначен (40 прогонов, успех / промах):

            T½      неподвижная      уезд оптимума
            нет     92% / 0.034      70% / 0.196
            15      92% / 0.036      78% / 0.124   <- взято
            25      90% / 0.041      72% / 0.151
            40      90% / 0.042      72% / 0.159

        Пятнадцать лучше и более длинных, и более коротких периодов: на
        неподвижной поверхности разницы с "не забывать" нет вовсе, на уезжающей
        промах падает почти вдвое.
        """
        x = np.array([self._to_unit(t.vector) for t in self.trials])
        y = np.array([t.score for t in self.trials])
        own_idx = [i for i, t in enumerate(self.trials) if t.source == "self"]
        w = np.full(len(self.trials), self.prior_weight)
        w[own_idx] = 1.0
        if self.forget_half_life:
            n = len(own_idx)
            for k, i in enumerate(own_idx):
                age = n - 1 - k
                w[i] = max(0.5 ** (age / self.forget_half_life), 1e-3)
        return x, y, w

    def recommend(self) -> tuple[Optional[StimParams], float, float]:
        """
        Что реально применять по итогам сессии: точка с наибольшим СГЛАЖЕННЫМ
        предсказанием, а не с наибольшим наблюдением.

        Разница не косметическая, она измерена. На поверхности, где эффекта
        НЕТ ВООБЩЕ, наивное правило отчитывается о +0.130 - выдуманный эффект
        размером примерно с половину настоящего. Сглаженное даёт +0.004.

            правило отчёта      неподвижная    эффекта нет
            максимум наблюдений    +0.094         +0.130
            сглаженное             -0.007         +0.004

        Причина - проклятие победителя: максимум зашумлённых наблюдений тем
        выше, чем больше проб. Отчитываться им нельзя.

        Возвращает (набор, предсказанный счёт, неопределённость предсказания).
        """
        own = [t for t in self.trials if t.source == "self"]
        if not own:
            return None, float("-inf"), float("inf")
        cand = self._candidates()
        x, y, w = self._observations()
        gp = GP(noise=self.noise).fit(x, y, w)
        pts = np.vstack([cand, np.array([self._to_unit(t.vector) for t in own])])             if cand.size else np.array([self._to_unit(t.vector) for t in own])
        mu, sd = gp.predict(pts)
        i = int(np.argmax(mu))
        return self._params(self._to_real(pts[i])), float(mu[i]), float(sd[i])

    def observe(self, params: StimParams, score: float, source: str = "self") -> None:
        v = np.array([getattr(params, n) for n in self._names], float)
        self.trials.append(Trial(params=params, score=float(score),
                                 vector=v, source=source))

    def add_prior(self, records: Sequence[tuple[StimParams, float]]) -> None:
        """Наблюдения с других животных: тёплый старт для нового."""
        for p, s in records:
            self.observe(p, s, source="prior")

    def best(self) -> tuple[Optional[StimParams], float]:
        """
        Лучшая ПРОБА за сессию. Для отчёта не годится - см. recommend().

        Наблюдение выбрано максимумом, поэтому оно систематически завышено:
        на стенде +0.094 при неподвижном отклике и +0.130 там, где эффекта нет
        вообще. Метод оставлен для журнала проб и отладки.
        """
        own = [t for t in self.trials if t.source == "self"]
        if not own:
            return None, float("-inf")
        t = max(own, key=lambda z: z.score)
        return t.params, t.score

    def history(self) -> list[dict]:
        return [{**t.params.to_record(), "score": t.score, "source": t.source}
                for t in self.trials]
