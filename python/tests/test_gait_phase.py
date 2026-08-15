"""
Тесты фазового тракта: разметка шага, потоковый детектор, процент фазы, поджиг.

Данные СИНТЕТИЧЕСКИЕ, GPU и записи не нужны: тест должен проходить на любой
машине за секунду. Синтетика воспроизводит физику тредмила - в опоре лента
тащит носок с постоянной скоростью, в переносе носок быстро идёт вперёд.

Почти каждый тест здесь закрывает ОШИБКУ, которая уже случалась:
  * направленная оценка скорости ленты молча испортила 5 записей из 12;
  * лаг детектора протекал в разметку эталона и давал смещение +9.5%;
  * поджиг по правилу "первый кадр за целью" перелетал на полшага;
  * взвод снимался одним кадром и был уязвим к выбросу проекции.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gait_phase_labeler as G
from realtime_phase_sim import STANCE, SWING, StreamingPhaseDetector
from realtime_phase_percent import StreamingPhasePercent, truth_percent

FPS = 100.0
STANCE_N, SWING_N = 30, 10          # кадров: цикл 40 = 400 мс
CYCLE_N = STANCE_N + SWING_N
V_BELT = -4.5                       # px/кадр, то есть -450 px/с


def synth_gait(n_cycles=40, v_belt=V_BELT, noise=0.0, seed=0, body_x=500.0,
               jitter=0):
    """
    Носок и iliac по кадрам. Петля замкнута: за перенос носок возвращает ровно
    то, что лента утащила за опору.

    jitter - разброс длительности фаз в кадрах. Он нужен не для красоты: без
    него шаг строго периодичен, цель пересекается каждый цикл в одном и том же
    кадре, и поправка на полшага перестаёт быть наблюдаемой (проверено - без
    джиттера смещение одинаково с поправкой и без неё).
    """
    rng = np.random.default_rng(seed)
    x, y = [], []
    pos = body_x + 60.0
    for _ in range(n_cycles):
        st = STANCE_N + (int(rng.integers(-jitter, jitter + 1)) if jitter else 0)
        sw = SWING_N + (int(rng.integers(-1, 2)) if jitter else 0)
        for _ in range(st):                             # опора: едем с лентой
            pos += v_belt
            x.append(pos)
            y.append(120.0)
        back = -v_belt * st / sw                        # перенос: возврат
        for k in range(sw):
            pos += back
            x.append(pos)
            y.append(120.0 - 18.0 * np.sin(np.pi * (k + 0.5) / sw))
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if noise:
        x = x + rng.normal(0, noise, x.size)
        y = y + rng.normal(0, noise, y.size)
    toe = np.column_stack([x, y, np.full(x.size, 0.95)])
    body = np.column_stack([np.full(x.size, body_x), np.full(x.size, 60.0),
                            np.full(x.size, 0.95)])
    return toe, body


# --------------------------------------------------------------------------- #
# скорость ленты: направление не должно иметь значения
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("v_belt", [-450.0, +450.0])
def test_belt_velocity_direction_agnostic(v_belt):
    """
    Регрессия на баг, из-за которого 5 записей из 12 считались неправильно.

    Прежняя оценка брала медиану НИЖНЕЙ половины скоростей и тем самым молча
    предполагала, что носок в опоре едет в -x. На камере с другой стороны
    дорожки он едет в +x, и оценка цепляла кластер переноса: получалось -28 px/с
    вместо +439. Разметка при этом не падала, а выдавала правдоподобный мусор.
    """
    v = np.concatenate([
        np.full(700, v_belt),                    # опора, 70% времени
        np.full(300, -3.0 * v_belt),             # перенос, втрое быстрее назад
        np.zeros(200),                           # паузы: крыса стоит
    ])
    got = G.estimate_belt_velocity(v)
    assert np.sign(got) == np.sign(v_belt), "знак скорости ленты определён неверно"
    assert abs(got - v_belt) < 0.05 * abs(v_belt), f"{got} против {v_belt}"


def test_belt_velocity_ignores_standing_cluster():
    """Пик на нуле от пауз не должен утягивать оценку на себя."""
    v = np.concatenate([np.full(300, -450.0), np.zeros(900)])
    assert abs(G.estimate_belt_velocity(v) + 450.0) < 25.0


# --------------------------------------------------------------------------- #
# соглашение о процентах
# --------------------------------------------------------------------------- #

def test_percent_convention_boundaries():
    """Опора это 0..100, перенос 100..200, нормировка ВНУТРИ каждой фазы."""
    td = np.array([0.0, 40.0, 80.0])
    lo = np.array([30.0, 70.0])
    pct = truth_percent(120, td, lo)
    assert pct[0] == pytest.approx(0.0)
    assert pct[30] == pytest.approx(100.0)          # отрыв
    assert pct[15] == pytest.approx(50.0)           # середина опоры
    assert pct[35] == pytest.approx(150.0)          # середина переноса
    assert np.all(pct[:30] < 100.0) and np.all(pct[30:40] >= 100.0)


def test_percent_scale_is_not_linear_in_time():
    """
    Опора 30 кадров и перенос 10 кадров получают по 100% каждая, значит один
    процент переноса втрое дешевле по времени. Это не дефект, а соглашение,
    но от него зависят все требования к точности.
    """
    pct = truth_percent(80, np.array([0.0, 40.0]), np.array([30.0]))
    per_pct_stance = 30.0 / 100.0
    per_pct_swing = 10.0 / 100.0
    assert per_pct_stance == pytest.approx(3 * per_pct_swing)
    assert pct[29] - pct[28] == pytest.approx(100.0 / 30.0)
    assert pct[35] - pct[34] == pytest.approx(100.0 / 10.0)


# --------------------------------------------------------------------------- #
# потоковый детектор событий
# --------------------------------------------------------------------------- #

def _run_detector(toe, body, **kw):
    det = StreamingPhaseDetector(fps=FPS, **kw)
    phases, events = [], []
    for i in range(len(toe)):
        ph, ev = det.update(toe[i, 0], toe[i, 2], body[i, 0], body[i, 2])
        phases.append(ph)
        if ev:
            events.append((i, ev))
    return phases, events


def test_detector_finds_both_phases():
    toe, body = synth_gait()
    phases, events = _run_detector(toe, body)
    assert STANCE in phases and SWING in phases
    kinds = {e for _, e in events}
    assert kinds == {"touch_down", "lift_off"}


def test_detector_cycle_length_matches_truth():
    toe, body = synth_gait()
    _, events = _run_detector(toe, body)
    td = [i for i, e in events if e == "touch_down"]
    assert len(td) > 20
    assert np.median(np.diff(td)) == pytest.approx(CYCLE_N, abs=1)


def test_detector_is_causal():
    """Дообработка кадров не должна менять уже выданные ответы."""
    toe, body = synth_gait(n_cycles=20)
    full, _ = _run_detector(toe, body)
    half = len(toe) // 2
    part, _ = _run_detector(toe[:half], body[:half])
    assert full[:half] == part


@pytest.mark.parametrize("v_belt", [-4.5, +4.5])
def test_detector_works_on_mirrored_camera(v_belt):
    """Зеркальная камера: носок в опоре едет в другую сторону."""
    toe, body = synth_gait(v_belt=v_belt)
    phases, events = _run_detector(toe, body)
    td = [i for i, e in events if e == "touch_down"]
    assert len(td) > 20, "на зеркальной камере события не найдены"
    assert np.median(np.diff(td)) == pytest.approx(CYCLE_N, abs=1)


# --------------------------------------------------------------------------- #
# процент фазы в потоке
# --------------------------------------------------------------------------- #

def _run_percent(toe, body, **kw):
    det = StreamingPhaseDetector(fps=FPS)
    est = StreamingPhasePercent(fps=FPS, **kw)
    pct, pred, fires = [], [], []
    for i in range(len(toe)):
        _, ev = det.update(toe[i, 0], toe[i, 2], body[i, 0], body[i, 2])
        p, q, f = est.update(tuple(toe[i]), tuple(body[i]), ev)
        pct.append(p)
        pred.append(q)
        if f:
            fires.append(i)
    return np.asarray(pct), np.asarray(pred), fires, est


def test_percent_builds_reference_and_reports():
    toe, body = synth_gait()
    pct, _, _, est = _run_percent(toe, body)
    assert est.n_rebuild > 5, "эталон так и не собрался"
    ok = np.isfinite(pct)
    assert ok.sum() > 0.5 * len(pct)
    assert pct[ok].min() >= 0.0 and pct[ok].max() < 200.0


def test_percent_advances_and_wraps_once_per_cycle():
    """Фаза должна расти и оборачиваться ровно раз на шаг."""
    toe, body = synth_gait(n_cycles=40)
    pct, _, _, _ = _run_percent(toe, body)
    ok = np.isfinite(pct)
    p = pct[ok]
    d = np.diff(p)
    forward = np.sum((d > 0) & (d < 40))
    backward = np.sum((d < -2) & (d > -150))       # откат, не оборот цикла
    assert backward < 0.02 * forward, f"откатов назад {backward} на {forward} шагов"
    wraps = np.sum(d < -150)
    assert wraps == pytest.approx(len(p) / CYCLE_N, rel=0.2)


def test_percent_is_causal():
    toe, body = synth_gait(n_cycles=20)
    full, _, _, _ = _run_percent(toe, body)
    half = len(toe) // 2
    part, _, _, _ = _run_percent(toe[:half], body[:half])
    np.testing.assert_allclose(full[:half], part, equal_nan=True)


# --------------------------------------------------------------------------- #
# поджиг
# --------------------------------------------------------------------------- #

def test_fires_about_once_per_cycle():
    toe, body = synth_gait(n_cycles=40)
    _, _, fires, _ = _run_percent(toe, body, target=145.0)
    n_cyc = len(toe) / CYCLE_N
    assert 0.7 * n_cyc < len(fires) <= n_cyc + 1, f"импульсов {len(fires)} на {n_cyc} циклов"


def test_refractory_blocks_close_repeats():
    """
    Между импульсами должно пройти не меньше min_gap процентов накопленной фазы.
    Прежний взвод снимался ОДНИМ кадром с pct < 50 и был уязвим к одиночному
    выбросу проекции; накопитель к таким выбросам нечувствителен.
    """
    toe, body = synth_gait(n_cycles=40, noise=1.5, seed=3)
    _, _, fires, _ = _run_percent(toe, body, target=145.0, min_gap=100.0)
    if len(fires) > 1:
        gaps = np.diff(fires)
        assert gaps.min() >= 0.4 * CYCLE_N, f"слишком близкие импульсы: {gaps.min()} кадров"


def _freeze_near_target(bout_std, n_frames=800, target=145.0, seed=2):
    """
    Животное замирает В ТОЧКЕ, проекция которой стоит у самой цели, и лишь
    подрагивает. Это единственный способ проверить гейт движения: если просто
    заморозить носок где попало, фаза не растёт и поджиг молчит сам по себе,
    без всякого гейта (проверено мутацией - тест был декоративным).
    """
    toe, body = synth_gait(n_cycles=20)
    det = StreamingPhaseDetector(fps=FPS)
    est = StreamingPhasePercent(fps=FPS, target=target, bout_std=bout_std)
    pct = []
    for i in range(len(toe)):
        _, ev = det.update(toe[i, 0], toe[i, 2], body[i, 0], body[i, 2])
        p, _, _ = est.update(tuple(toe[i]), tuple(body[i]), ev)
        pct.append(p)
    pct = np.asarray(pct)
    tail = range(len(pct) - 40, len(pct))
    near = [i for i in tail if np.isfinite(pct[i]) and target - 7 <= pct[i] <= target - 1]
    assert near, "не нашлось кадра с процентом у самой цели"
    k = near[0]

    rng = np.random.default_rng(seed)

    def jitter_frames(count):
        fired = 0
        for _ in range(count):
            t = toe[k].copy()
            t[0] += rng.normal(0, 3.0)
            t[1] += rng.normal(0, 3.0)
            _, _, f = est.update(tuple(t), tuple(body[k]), None)
            fired += int(f)
        return fired

    jitter_frames(100)             # окно признака движения должно заполниться
    return jitter_frames(n_frames), est


def test_no_fire_when_animal_stands():
    """Дрожит на месте у самой цели: поджиг обязан молчать."""
    fired, est = _freeze_near_target(bout_std=8.0)
    assert est.walking is False, "признак движения не опустился"
    assert fired == 0, f"выдано {fired} импульсов по стоящему животному"


def test_walk_gate_is_what_blocks_standing_fire():
    """
    Тот же сценарий с ОТКЛЮЧЁННЫМ гейтом обязан стрелять. Иначе предыдущий тест
    ничего не стережёт: молчание могло бы объясняться не гейтом, а тем, что
    фаза не растёт.
    """
    fired_off, est = _freeze_near_target(bout_std=0.0)
    assert est.walking is True
    assert fired_off > 0, "без гейта поджиг тоже молчит, значит гейт не проверен"


def test_fire_lands_near_target():
    """
    Импульс кладётся через lat кадров после решения. Поджиг идёт по ПРОГНОЗУ и
    с поправкой на полшага: без неё правило "первый кадр за целью" перелетает.
    Порог 2% выбран по измерению: с поправкой -1.0%, без неё +3.0%.
    Джиттер обязателен, на строго периодическом шаге разницы не видно.
    """
    toe, body = synth_gait(n_cycles=60, jitter=4)
    pct, _, fires, est = _run_percent(toe, body, target=145.0, latency_ms=30.0)
    landed = [pct[i + est.lat] for i in fires
              if i + est.lat < len(pct) and np.isfinite(pct[i + est.lat])]
    assert len(landed) > 20
    err = (np.asarray(landed) - 145.0 + 100.0) % 200.0 - 100.0
    assert abs(np.median(err)) < 2.0, f"систематическое смещение {np.median(err):+.1f}%"


def test_event_lag_compensation_shifts_labels():
    """
    Компенсация лага событий сдвигает ЯРЛЫКИ на петле. Без неё точка "100%"
    оказывается там, где перенос уже начался, и импульс систематически
    опаздывает. Проверяем, что параметр реально влияет и в нужную сторону.
    """
    toe, body = synth_gait(n_cycles=40)
    p0, _, _, _ = _run_percent(toe, body, event_lag=0)
    p2, _, _, _ = _run_percent(toe, body, event_lag=2)
    ok = np.isfinite(p0) & np.isfinite(p2)
    assert ok.sum() > 500
    d = (p2[ok] - p0[ok] + 100.0) % 200.0 - 100.0
    assert np.median(d) > 1.0, "event_lag не сдвигает шкалу"


def test_target_is_respected():
    """Разные цели дают разные моменты поджига."""
    toe, body = synth_gait(n_cycles=40)
    pct, _, f_low, est = _run_percent(toe, body, target=120.0)
    pct2, _, f_high, est2 = _run_percent(toe, body, target=180.0)
    a = [pct[i + est.lat] for i in f_low if i + est.lat < len(pct) and np.isfinite(pct[i + est.lat])]
    b = [pct2[i + est2.lat] for i in f_high if i + est2.lat < len(pct2) and np.isfinite(pct2[i + est2.lat])]
    assert len(a) > 10 and len(b) > 10
    assert np.median(a) < np.median(b), "цель не влияет на момент поджига"


# --------------------------------------------------------------------------- #
# копия боевого ROI-трекера не должна разъезжаться с оригиналом
# --------------------------------------------------------------------------- #

def _load_production_tracker():
    """
    Достаём класс из боевого моста БЕЗ импорта самого моста: он тянет за собой
    камеру, конфиг и torch, чего в тестах быть не должно. Разбираем файл через
    ast и исполняем только нужный класс.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "single_rt_dlc_live_bridge.py"
    if not src.exists():
        pytest.skip("боевой мост рядом не найден (другая раскладка репозитория)")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LegRoiTracker":
            ns: dict = {"np": np}
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(src), "exec"), ns)
            except NameError as exc:
                pytest.skip(f"класс зависит от внешних имён: {exc}")
            return ns["LegRoiTracker"]
    pytest.fail("LegRoiTracker не найден в боевом мосте")


def test_roi_tracker_copy_matches_production():
    """
    realtime_phase_sim.LegRoiTracker это КОПИЯ боевого трекера, взятая ради
    того, чтобы офлайн-стенд не тянул за собой камеру и torch. Копия может
    незаметно разъехаться с оригиналом, поэтому сверяем поведение: одна и та же
    последовательность поз должна давать одинаковые окна ROI, кадр в кадр.
    """
    from realtime_phase_sim import BODYPARTS, LegRoiTracker as Copy

    Prod = _load_production_tracker()
    legs = [i for i, nm in enumerate(BODYPARTS) if nm.startswith("hl_")]
    fw, fh = 1920, 220

    # У боевого класса умолчаний нет, значения приходят из конфига. Передаём
    # ровно те, что стоят умолчаниями в копии: их совпадение с конфигом
    # проверяет отдельный тест ниже.
    a = Copy(fw, fh, legs, width=256)
    b = Prod(fw, fh, legs, width=256, detect_thresh=0.30,
             hold_frames=100, center_ema=0.35)

    rng = np.random.default_rng(11)
    for i in range(600):
        pose = np.zeros((len(BODYPARTS), 3), dtype=float)
        centre = 400.0 + 300.0 * np.sin(i / 40.0)
        for k, j in enumerate(legs):
            pose[j] = (centre + 12.0 * k + rng.normal(0, 1.5),
                       110.0, 0.9 if (i // 70) % 5 else 0.05)   # периодические потери
        a.update(pose)
        b.update(pose)
        assert a.window() == b.window(), (
            f"копия разъехалась с боевым трекером на кадре {i}: "
            f"{a.window()} против {b.window()}")


def test_roi_tracker_defaults_match_config():
    """
    Умолчания копии обязаны совпадать с боевым конфигом. Иначе офлайн-стенд
    молча меряет не ту систему, что работает в контуре: параметры ROI влияют и
    на устойчивость трека, и на долю кадров в поиске.
    """
    import inspect
    import re

    from realtime_phase_sim import LegRoiTracker as Copy

    cfg = Path(__file__).resolve().parents[1] / "config_dual_rt_dlc_live.py"
    if not cfg.exists():
        pytest.skip("боевой конфиг рядом не найден")
    text = cfg.read_text(encoding="utf-8")

    want = {}
    for key in ("LEG_ROI_WIDTH", "LEG_ROI_DETECT_THRESH",
                "LEG_ROI_HOLD_FRAMES", "LEG_ROI_CENTER_EMA"):
        m = re.search(rf"^{key}\s*=\s*([0-9.]+)", text, re.M)
        assert m, f"{key} не найден в конфиге"
        want[key] = float(m.group(1))

    got = {p.name: p.default for p in inspect.signature(Copy.__init__).parameters.values()}
    assert float(got["width"]) == want["LEG_ROI_WIDTH"]
    assert float(got["detect_thresh"]) == want["LEG_ROI_DETECT_THRESH"]
    assert float(got["hold_frames"]) == want["LEG_ROI_HOLD_FRAMES"]
    assert float(got["center_ema"]) == want["LEG_ROI_CENTER_EMA"]
