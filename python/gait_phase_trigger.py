"""
Фазовый триггер для боевого моста: процент шага -> TTL-линия.

Тонкая обвязка над StreamingPhaseDetector и StreamingPhasePercent, которая
читает конфиг и отдаёт мосту один булев ответ на кадр: жечь или нет.
Вся физика и все поправки живут в тех классах, здесь только склейка.

ПОЧЕМУ РЕЖИМ TTL, А НЕ БИНАРНЫЙ. Плагин держит ОДНО восьмибитное TTL-слово и
пересобирает его целиком из каждого пришедшего пакета (DualDLCLiveBridge.cpp,
`ttlWord = 0; for (line...) if (nextStates[line]) ttlWord |= 1 << line;`).
Для бинарных pose-пакетов состояния линий вычисляет сам плагин из позы, и бит,
выставленный отдельным пакетом от Python, будет погашен уже следующим кадром.
Поэтому фазовый триггер работает только в режиме, где слово формирует Python:

    DUAL_OE_BRIDGE_PACKET_MODE = "ttl"

Мост об этом предупреждает в логе при старте, если режим другой.

ЧТО ЭТО СТОИТ. Замерено на записи 15687 кадров: 0.33 мс на кадр при инференсе
18.0 мс, то есть 1.8% цикла. Точность попадания в цель +0.5% (0.5 мс) при
разбросе p90 10.6% (11.1 мс), и разброс этот определяется изменчивостью самого
животного, а не контуром.

ЧЕГО ЗДЕСЬ НЕТ. Выходное плечо (от решения до тока в электроде) не измерено.
Программная часть 0.18 мс, но TTL выставляется из process() поблочно, и при
буфере 1024 отсчёта это добавляет джиттер около 10 мс. Подробности и что с этим
делать - в docs/GAIT_PHASE_PERCENT_2026-08-15.md.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import config_dual_rt_dlc_live as config
from realtime_phase_percent import StreamingPhasePercent
from realtime_phase_sim import StreamingPhaseDetector


class PhaseTrigger:
    """Процент фазы шага и решение о поджиге. Один экземпляр на ногу."""

    def __init__(self, logger: logging.Logger, fps: float,
                 bodypart_to_idx: dict[str, int]) -> None:
        self.logger = logger
        self.enabled = bool(getattr(config, "PHASE_TRIGGER_ENABLED", False))
        self.leg = str(getattr(config, "PHASE_TRIGGER_LEG", "r")).strip().lower()
        self.line = int(getattr(config, "PHASE_TRIGGER_TTL_LINE", 4))
        self.target = float(getattr(config, "PHASE_TRIGGER_TARGET_PCT", 145.0))
        self.hold_frames = max(1, int(getattr(config, "PHASE_TRIGGER_HOLD_FRAMES", 2)))

        self.pct: float = float("nan")
        self.pred: float = float("nan")
        self.walking = False
        self.fired_total = 0
        self._hold = 0
        self.i_toe: Optional[int] = None
        self.i_body: Optional[int] = None
        self.det: Optional[StreamingPhaseDetector] = None
        self.est: Optional[StreamingPhasePercent] = None

        if not self.enabled:
            return
        if not 0 <= self.line <= 7:
            raise ValueError(f"PHASE_TRIGGER_TTL_LINE вне 0..7: {self.line}")
        if self.leg not in ("l", "r"):
            # Автовыбор ноги сознательно не поддержан: эталонная петля строится
            # для КОНКРЕТНОЙ ноги, и переключение посреди сессии её обнулит.
            raise ValueError(f"PHASE_TRIGGER_LEG должен быть 'l' или 'r', задано {self.leg!r}")

        toe, body = f"hl_toes_{self.leg}", f"hl_iliac_{self.leg}"
        missing = [n for n in (toe, body) if n not in bodypart_to_idx]
        if missing:
            raise KeyError(f"в модели нет точек {missing}, фазовый триггер невозможен")
        self.i_toe, self.i_body = bodypart_to_idx[toe], bodypart_to_idx[body]

        self.det = StreamingPhaseDetector(fps=fps)
        self.est = StreamingPhasePercent(
            fps=fps,
            target=self.target,
            latency_ms=float(getattr(config, "PHASE_TRIGGER_LATENCY_MS", 28.0)),
            ref_cycles=int(getattr(config, "PHASE_TRIGGER_REF_CYCLES", 10)),
            event_lag=int(getattr(config, "PHASE_TRIGGER_EVENT_LAG", 1)),
            med_win=int(getattr(config, "PHASE_TRIGGER_MED_WIN", 1)),
            min_gap=float(getattr(config, "PHASE_TRIGGER_MIN_GAP_PCT", 100.0)),
            bout_std=float(getattr(config, "PHASE_TRIGGER_BOUT_STD_PX", 8.0)),
        )

        mode = str(getattr(config, "DUAL_OE_BRIDGE_PACKET_MODE", "pose")).strip().lower()
        logger.info(
            "Phase trigger enabled: leg=%s target=%.0f%% line=%d latency=%.0fms "
            "ref_cycles=%d min_gap=%.0f%% hold=%d frames",
            self.leg, self.target, self.line,
            float(getattr(config, "PHASE_TRIGGER_LATENCY_MS", 28.0)),
            int(getattr(config, "PHASE_TRIGGER_REF_CYCLES", 10)),
            float(getattr(config, "PHASE_TRIGGER_MIN_GAP_PCT", 100.0)),
            self.hold_frames,
        )
        if mode != "ttl":
            logger.warning(
                "Phase trigger НЕ БУДЕТ ВИДЕН плагину: DUAL_OE_BRIDGE_PACKET_MODE=%r. "
                "Плагин пересобирает TTL-слово из каждого пакета, и в режиме %r линии "
                "считает он сам. Нужен режим 'ttl'.", mode, mode)

    # ------------------------------------------------------------------ #
    def update(self, pose: np.ndarray) -> bool:
        """Один кадр позы (K, 3) -> держать ли линию поднятой."""
        if not self.enabled or self.est is None or self.det is None:
            return False
        toe = pose[self.i_toe]
        body = pose[self.i_body]
        _, event = self.det.update(float(toe[0]), float(toe[2]),
                                   float(body[0]), float(body[2]))
        pct, pred, fire = self.est.update(
            (float(toe[0]), float(toe[1]), float(toe[2])),
            (float(body[0]), float(body[1]), float(body[2])),
            event,
        )
        self.pct, self.pred, self.walking = pct, pred, self.est.walking
        if fire:
            self.fired_total += 1
            self._hold = self.hold_frames
        if self._hold > 0:
            self._hold -= 1
            return True
        return False

    # ------------------------------------------------------------------ #
    def overlay_text(self) -> str:
        if not self.enabled:
            return ""
        if not np.isfinite(self.pct):
            return f"phase {self.leg.upper()}: --  cel {self.target:.0f}%  n={self.fired_total}"
        return (f"phase {self.leg.upper()}: {self.pct:5.1f}%  cel {self.target:.0f}%  "
                f"{'walk' if self.walking else 'stand'}  n={self.fired_total}")
