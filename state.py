"""
Core emotional state model — ALMA-inspired three-layer architecture.

Trait  →  stable personality baseline (OCEAN), cross-session persistent
Mood   →  slowly drifting affective tone, Ornstein-Uhlenbeck mean-reverting
Emotion → event-triggered acute state, exponential decay

All dimensions clamped to [0, 1].  Computed properties (patience,
silence_threshold) are derived, not stored.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---- constants ----------------------------------------------------

# emotion labels (discrete)
JOY = "joy"
SADNESS = "sadness"
ANGER = "anger"
FEAR = "fear"
SURPRISE = "surprise"
DISGUST = "disgust"
TRUST = "trust"
ANTICIPATION = "anticipation"
GUILT = "guilt"
GRATITUDE = "gratitude"
HURT = "hurt"           # transient negative, agency=other but < anger

EMOTION_LABELS: tuple[str, ...] = (
    JOY, SADNESS, ANGER, FEAR, SURPRISE,
    DISGUST, TRUST, ANTICIPATION, GUILT, GRATITUDE, HURT,
)

# OU process parameters
THETA_DEFAULT = 0.12       # regression strength (per hour)
SIGMA_DEFAULT = 0.025      # volatility (per hour)

# emotion half-lives (seconds)
EMOTION_HALF_LIVES: dict[str, float] = {
    JOY: 180,               #  3 min
    SADNESS: 900,           # 15 min
    ANGER: 600,             # 10 min
    FEAR: 300,              #  5 min
    SURPRISE: 60,           #  1 min
    DISGUST: 450,           #  7.5 min
    TRUST: 600,             # 10 min
    ANTICIPATION: 300,      #  5 min
    GUILT: 900,             # 15 min
    GRATITUDE: 300,         #  5 min
    HURT: 600,              # 10 min
}

# emotion → PAD centroid
_EMOTION_PAD: dict[str, tuple[float, float, float]] = {
    JOY:          (0.80, 0.60, 0.65),
    SADNESS:      (0.15, 0.10, 0.15),
    ANGER:        (0.10, 0.85, 0.75),
    FEAR:         (0.15, 0.75, 0.10),
    SURPRISE:     (0.55, 0.70, 0.40),
    DISGUST:      (0.15, 0.50, 0.55),
    TRUST:        (0.70, 0.35, 0.55),
    ANTICIPATION: (0.60, 0.50, 0.55),
    GUILT:        (0.15, 0.40, 0.15),
    GRATITUDE:    (0.75, 0.45, 0.50),
    HURT:         (0.15, 0.55, 0.20),
}

# emotion absorption weight (how much decayed emotion feeds into mood)
ABSORPTION_WEIGHT = 0.12

# silence / decay thresholds
EMOTION_FLOOR = 0.05
SILENCE_LONG_REPLY = 150     # chars — "already fully responded"
PERSIST_EVERY_N = 10         # save to disk every N interactions


# ---- helpers ------------------------------------------------------

def _c(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, x)), 3)


# ---- dataclasses --------------------------------------------------

@dataclass
class Trait:
    """Stable personality baseline (Big Five OCEAN)."""

    openness: float = 0.55
    conscientiousness: float = 0.60
    extraversion: float = 0.45
    agreeableness: float = 0.65
    neuroticism: float = 0.40

    # derived regulation preferences (set in __post_init__)
    reappraisal: float = 0.55
    suppression: float = 0.30
    rumination: float = 0.25

    def __post_init__(self) -> None:
        self.reappraisal = _c(0.25 + self.openness * 0.50 + self.agreeableness * 0.35)
        self.suppression = _c(0.10 + self.neuroticism * 0.55 + (1.0 - self.extraversion) * 0.20)
        self.rumination   = _c(0.10 + self.neuroticism * 0.65)

    @property
    def mood_baseline(self) -> dict[str, float]:
        return {
            "valence":   _c(0.25 + (1.0 - self.neuroticism) * 0.50 + self.extraversion * 0.20),
            "arousal":   _c(0.25 + self.extraversion * 0.55),
            "dominance": _c(0.25 + self.conscientiousness * 0.35 + (1.0 - self.neuroticism) * 0.30),
        }

    @classmethod
    def random(cls) -> Trait:
        return cls(
            openness=round(random.uniform(0.15, 0.95), 2),
            conscientiousness=round(random.uniform(0.15, 0.95), 2),
            extraversion=round(random.uniform(0.15, 0.95), 2),
            agreeableness=round(random.uniform(0.15, 0.95), 2),
            neuroticism=round(random.uniform(0.10, 0.85), 2),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "openness": self.openness, "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion, "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "reappraisal": self.reappraisal, "suppression": self.suppression,
            "rumination": self.rumination,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trait:
        t = cls(
            openness=d.get("openness", 0.55),
            conscientiousness=d.get("conscientiousness", 0.60),
            extraversion=d.get("extraversion", 0.45),
            agreeableness=d.get("agreeableness", 0.65),
            neuroticism=d.get("neuroticism", 0.40),
        )
        # override derived if stored explicitly
        t.reappraisal = d.get("reappraisal", t.reappraisal)
        t.suppression = d.get("suppression", t.suppression)
        t.rumination   = d.get("rumination", t.rumination)
        return t


@dataclass
class Mood:
    """Slowly-changing affective tone.  Mean-reverts to Trait.mood_baseline."""

    valence: float = 0.50
    arousal: float = 0.50
    dominance: float = 0.50
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"v": self.valence, "a": self.arousal, "d": self.dominance, "ts": self.updated_at}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mood:
        return cls(
            valence=d.get("v", 0.50), arousal=d.get("a", 0.50),
            dominance=d.get("d", 0.50), updated_at=d.get("ts", 0.0),
        )


@dataclass
class Emotion:
    """Acute emotional state triggered by an appraisal event."""

    primary: str | None = None
    secondary: str | None = None
    v: float = 0.0          # valence shift
    a: float = 0.0          # arousal shift
    d: float = 0.0          # dominance shift
    intensity: float = 0.0
    started_at: float = 0.0
    half_life: float = 300.0

    @property
    def is_active(self) -> bool:
        return self.primary is not None and self.intensity > EMOTION_FLOOR

    def to_dict(self) -> dict[str, Any] | None:
        if self.primary is None:
            return None
        return {
            "p": self.primary, "s": self.secondary,
            "v": self.v, "a": self.a, "d": self.d,
            "i": self.intensity, "st": self.started_at, "hl": self.half_life,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Emotion | None:
        if d is None:
            return None
        return cls(
            primary=d.get("p"), secondary=d.get("s"),
            v=d.get("v", 0.0), a=d.get("a", 0.0), d=d.get("d", 0.0),
            intensity=d.get("i", 0.0), started_at=d.get("st", 0.0),
            half_life=d.get("hl", 300.0),
        )


@dataclass
class SessionState:
    trait: Trait
    mood: Mood
    emotion: Emotion | None = None
    enabled: bool = True
    msg_count: int = 0
    last_response_len: int = 0
    last_active: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trait": self.trait.to_dict(),
            "mood": self.mood.to_dict(),
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "enabled": self.enabled,
            "msg_count": self.msg_count,
            "last_response_len": self.last_response_len,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SessionState:
        return cls(
            trait=Trait.from_dict(d.get("trait", {})),
            mood=Mood.from_dict(d.get("mood", {})),
            emotion=Emotion.from_dict(d.get("emotion")),
            enabled=d.get("enabled", True),
            msg_count=d.get("msg_count", 0),
            last_response_len=d.get("last_response_len", 0),
            last_active=d.get("last_active", 0.0),
        )


# ---- StateManager -------------------------------------------------

class StateManager:
    """Owns per-session states, persistence, drift, decay, and computed props."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "states_v2.json")
        self.states: dict[str, SessionState] = {}
        self._dirty = 0
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.data_file):
            # try v1 migration
            old = os.path.join(self.data_dir, "states.json")
            if os.path.exists(old):
                self._migrate_v1(old)
            return
        try:
            with open(self.data_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            sessions = raw.get("sessions", raw)  # tolerate both formats
            for sid, obj in sessions.items():
                self.states[sid] = SessionState.from_dict(obj)
        except Exception:
            self.states = {}

    def save(self) -> None:
        try:
            payload: dict[str, Any] = {
                "_version": 2,
                "sessions": {sid: s.to_dict() for sid, s in self.states.items()},
            }
            with open(self.data_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self._dirty = 0
        except Exception:
            pass

    def _maybe_save(self) -> None:
        self._dirty += 1
        if self._dirty >= PERSIST_EVERY_N:
            self.save()

    # -- migration ---------------------------------------------------

    def _migrate_v1(self, old_path: str) -> None:
        try:
            with open(old_path, encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:
            return

        first = next(iter(old.values()), None)
        if first and "trait" in first:
            # already v2 shape in old file — copy it
            for sid, obj in old.items():
                self.states[sid] = SessionState.from_dict(obj)
            self.save()
            os.rename(old_path, old_path + ".v1.bak")
            return

        for sid, v1 in old.items():
            trait = Trait(
                openness=v1.get("openness", 0.55),
                extraversion=v1.get("energy", 0.60),
                neuroticism=round(max(0.10, 0.85 - v1.get("patience", 0.72)), 2),
                agreeableness=v1.get("valence", 0.40) + 0.2,
                conscientiousness=0.60,
            )
            trait.__post_init__()
            mood = Mood(
                valence=v1.get("valence", 0.40),
                arousal=v1.get("energy", 0.60),
                dominance=0.50,
                updated_at=v1.get("last_active", time.time()),
            )
            self.states[sid] = SessionState(
                trait=trait, mood=mood,
                enabled=v1.get("enabled", True),
                msg_count=v1.get("msg_count", 0),
                last_response_len=v1.get("last_response_len", 0),
                last_active=mood.updated_at,
            )
        self.save()
        os.rename(old_path, old_path + ".v1.bak")

    # -- lifecycle ---------------------------------------------------

    def get_or_init(self, session_id: str) -> SessionState:
        if session_id not in self.states:
            self.states[session_id] = self._random_init()
            return self.states[session_id]

        state = self.states[session_id]
        elapsed = time.time() - state.last_active
        # long silence → chance to re-randomise
        if elapsed > 1800 and random.random() < 0.30:
            self.states[session_id] = self._random_init()
        return self.states[session_id]

    def get_state(self, session_id: str) -> SessionState | None:
        return self.states.get(session_id)

    def _random_init(self) -> SessionState:
        trait = Trait.random()
        bl = trait.mood_baseline
        mood = Mood(
            valence=bl["valence"], arousal=bl["arousal"],
            dominance=bl["dominance"], updated_at=time.time(),
        )
        return SessionState(trait=trait, mood=mood, last_active=time.time())

    # -- drift (OU process) -----------------------------------------

    def drift_mood(self, session_id: str) -> None:
        """Apply OU mean-reverting drift to mood since last tick."""
        state = self.get_state(session_id)
        if state is None:
            return

        mood = state.mood
        baseline = state.trait.mood_baseline
        now = time.time()
        dt_hours = max(0.0, (now - mood.updated_at) / 3600.0)
        if dt_hours <= 0:
            return

        for dim, key in [("valence", "valence"), ("arousal", "arousal"), ("dominance", "dominance")]:
            mu = baseline[dim]
            x = getattr(mood, key)
            drift = THETA_DEFAULT * (mu - x) * dt_hours
            noise = SIGMA_DEFAULT * math.sqrt(dt_hours) * random.gauss(0, 1)
            setattr(mood, key, _c(x + drift + noise))

        mood.updated_at = now

    # -- emotion -----------------------------------------------------

    def trigger_emotion(
        self,
        session_id: str,
        label: str,
        intensity: float = 0.6,
    ) -> Emotion:
        """Set an active emotion on the session."""
        state = self.get_or_init(session_id)

        padv = _EMOTION_PAD.get(label, (0.50, 0.50, 0.50))
        hl = EMOTION_HALF_LIVES.get(label, 300.0)

        em = Emotion(
            primary=label,
            v=padv[0], a=padv[1], d=padv[2],
            intensity=_c(intensity),
            started_at=time.time(),
            half_life=hl,
        )
        state.emotion = em
        return em

    def decay_emotion(self, session_id: str) -> None:
        """Apply exponential decay; absorb residue into mood on expiry."""
        state = self.get_state(session_id)
        if state is None or state.emotion is None:
            return

        em = state.emotion
        elapsed = max(0.0, time.time() - em.started_at)
        current = em.intensity * (2.0 ** (-elapsed / max(1.0, em.half_life)))

        if current < EMOTION_FLOOR:
            self._absorb_into_mood(state.mood, em, elapsed)
            state.emotion = None
        else:
            em.intensity = _c(current)

    def _absorb_into_mood(self, mood: Mood, em: Emotion, duration: float) -> None:
        """Feed decayed emotional experience into mood (cumulative effect)."""
        impact = _c(em.intensity * min(1.0, duration / max(1.0, em.half_life)) * ABSORPTION_WEIGHT)
        mood.valence   = _c(mood.valence   + em.v * impact * 0.5)  # half-weight on v/a/d
        mood.arousal   = _c(mood.arousal   + em.a * impact * 0.5)
        mood.dominance = _c(mood.dominance + em.d * impact * 0.5)

    # -- computed properties -----------------------------------------

    def patience(self, session_id: str) -> float:
        state = self.get_state(session_id)
        if state is None:
            return 0.70
        t = state.trait
        m = state.mood
        n = state.msg_count
        base = 1.0 - t.neuroticism * 0.55
        mood_penalty = max(0.0, 0.5 - m.valence) * 0.45
        fatigue = min(0.45, n * 0.012)
        # active anger eats patience faster
        if state.emotion and state.emotion.primary == ANGER:
            fatigue += 0.10
        return _c(max(0.05, base - mood_penalty - fatigue))

    def silence_threshold(self, session_id: str) -> float:
        """0→1; higher = more likely to stay silent."""
        state = self.get_state(session_id)
        if state is None:
            return 0.25
        t = state.trait
        m = state.mood
        base = (1.0 - t.extraversion) * 0.45 + t.neuroticism * 0.15
        mood_factor = max(0.0, 0.5 - m.valence) * 0.40
        ars_factor  = max(0.0, 0.5 - m.arousal) * 0.25
        emo_factor = 0.0
        if state.emotion:
            if state.emotion.primary == SADNESS:
                emo_factor = 0.30
            elif state.emotion.primary == ANGER:
                emo_factor = 0.15
            elif state.emotion.primary == HURT:
                emo_factor = 0.35
        return _c(min(1.0, base + mood_factor + ars_factor + emo_factor))

    def current_mode(self, session_id: str) -> str:
        """Map current state to expression mode label for backward compat."""
        state = self.get_state(session_id)
        if state is None:
            return "议论"
        m = state.mood
        # rough heuristic mapping
        if m.arousal < 0.25 and m.valence < 0.35:
            return "说明"
        if m.arousal > 0.70 and m.valence > 0.55:
            return "抒情"
        if m.arousal > 0.55:
            return "议论"
        if m.valence > 0.60:
            return "记叙"
        return "描写"

    # -- manual overrides --------------------------------------------

    def set_trait(self, session_id: str, **kwargs: Any) -> SessionState:
        state = self.get_or_init(session_id)
        for k, v in kwargs.items():
            if hasattr(state.trait, k):
                setattr(state.trait, k, _c(float(v)))
        state.trait.__post_init__()
        self.save()
        return state

    def set_mood(self, session_id: str, **kwargs: Any) -> SessionState:
        state = self.get_or_init(session_id)
        for k, v in kwargs.items():
            key = {"valence": "valence", "arousal": "arousal", "dominance": "dominance"}.get(k, k)
            if hasattr(state.mood, key):
                setattr(state.mood, key, _c(float(v)))
        state.mood.updated_at = time.time()
        self.save()
        return state

    def set_enabled(self, session_id: str, enabled: bool) -> None:
        state = self.get_or_init(session_id)
        state.enabled = enabled
        self.save()

    def reset(self, session_id: str) -> SessionState:
        self.states[session_id] = self._random_init()
        self.save()
        return self.states[session_id]
