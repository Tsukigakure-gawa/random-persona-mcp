"""
Interpersonal relationship model — social penetration theory applied
to human-AI conversation.

Tracks relationship stage, self-disclosure depth, and interaction
history per user.  Regulates downstream emotional expressiveness,
silence thresholds, and language style.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


# ---- constants ----------------------------------------------------

STAGE_STRANGER      = "stranger"
STAGE_ACQUAINTANCE  = "acquaintance"
STAGE_FRIEND        = "friend"
STAGE_CLOSE         = "close"

# upgrade thresholds
THRESHOLD_ACQUAINTANCE = 20      # interactions
THRESHOLD_FRIEND       = 80
THRESHOLD_CLOSE        = 200
MIN_POSITIVE_RATIO     = 0.70

# disclosure onion layers (cumulative)
DISCLOSURE_SURFACE  = 0.0   # hobbies, daily life, weather
DISCLOSURE_SHALLOW  = 0.2   # opinions, preferences, anecdotes
DISCLOSURE_MID      = 0.4   # values, goals, confusion
DISCLOSURE_DEEP     = 0.6   # weaknesses, failures, unease
DISCLOSURE_CORE     = 0.8   # trauma, fear, core beliefs

# max AI disclosure relative to user's observed depth
MAX_DISCLOSURE_GAP = 0.15


# ---- dataclass ----------------------------------------------------

@dataclass
class Relationship:
    user_id: str
    stage: str = STAGE_STRANGER
    self_disclosure_depth: float = 0.0       # AI's side
    user_disclosure_depth: float = 0.0        # observed from user
    interaction_count: int = 0
    positive_exchanges: int = 0
    negative_exchanges: int = 0
    first_interaction: float = 0.0
    last_interaction: float = 0.0
    formality_match: float = 0.5

    @property
    def positive_ratio(self) -> float:
        total = self.positive_exchanges + self.negative_exchanges
        return self.positive_exchanges / total if total > 0 else 0.5

    @property
    def max_disclosure(self) -> float:
        """AI should not disclose deeper than user + gap."""
        return min(1.0, self.user_disclosure_depth + MAX_DISCLOSURE_GAP)

    @property
    def is_established(self) -> bool:
        return self.stage != STAGE_STRANGER

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "sd": self.self_disclosure_depth,
            "ud": self.user_disclosure_depth,
            "ic": self.interaction_count,
            "pos": self.positive_exchanges,
            "neg": self.negative_exchanges,
            "first": self.first_interaction,
            "last": self.last_interaction,
            "fm": self.formality_match,
        }

    @classmethod
    def from_dict(cls, uid: str, d: dict[str, Any]) -> Relationship:
        return cls(
            user_id=uid,
            stage=d.get("stage", STAGE_STRANGER),
            self_disclosure_depth=d.get("sd", 0.0),
            user_disclosure_depth=d.get("ud", 0.0),
            interaction_count=d.get("ic", 0),
            positive_exchanges=d.get("pos", 0),
            negative_exchanges=d.get("neg", 0),
            first_interaction=d.get("first", 0.0),
            last_interaction=d.get("last", 0.0),
            formality_match=d.get("fm", 0.5),
        )


# ---- RelationshipManager -------------------------------------------

class RelationshipManager:
    """Load, persist, and update per-user relationship state."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_file = os.path.join(data_dir, "relationships.json")
        self._rels: dict[str, Relationship] = {}
        self._load()

    # -- persistence -------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.data_file):
            return
        try:
            with open(self.data_file, encoding="utf-8") as fh:
                raw = json.load(fh)
            for uid, obj in raw.items():
                self._rels[uid] = Relationship.from_dict(uid, obj)
        except Exception:
            self._rels = {}

    def save(self) -> None:
        try:
            payload = {uid: r.to_dict() for uid, r in self._rels.items()}
            with open(self.data_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # -- access ------------------------------------------------------

    def get(self, user_id: str) -> Relationship:
        if user_id not in self._rels:
            r = Relationship(user_id=user_id, first_interaction=time.time())
            self._rels[user_id] = r
            return r
        return self._rels[user_id]

    def get_stage(self, user_id: str) -> str:
        return self.get(user_id).stage

    def get_disclosure_limit(self, user_id: str) -> float:
        return self.get(user_id).max_disclosure

    # -- updates -----------------------------------------------------

    def record_interaction(
        self, user_id: str, sentiment: float = 0.0, user_msg: str = ""
    ) -> None:
        """Record one round of interaction with optional sentiment."""
        r = self.get(user_id)
        r.interaction_count += 1
        r.last_interaction = time.time()

        if sentiment > 0.15:
            r.positive_exchanges += 1
        elif sentiment < -0.15:
            r.negative_exchanges += 1

        # estimate user disclosure from message length + intimacy cues
        if user_msg:
            r.user_disclosure_depth = round(
                r.user_disclosure_depth * 0.95
                + self._estimate_disclosure(user_msg) * 0.05,
                2,
            )

        self._maybe_upgrade(r)
        self.save()

    def _estimate_disclosure(self, text: str) -> float:
        """Heuristic: estimate how deeply personal a user message is."""
        score = 0.0
        length = len(text)

        # length correlates weakly with disclosure
        if length > 50:
            score += 0.05
        if length > 150:
            score += 0.10
        if length > 400:
            score += 0.10

        # emotional / personal keywords → deeper
        deep_words = [
            "我觉得", "我感觉", "我害怕", "我担心", "我后悔",
            "我小时候", "我妈", "我爸", "我家人", "我对象", "我男", "我女",
            "压力", "焦虑", "抑郁", "失眠", "崩溃", "分手", "失恋",
            "秘密", "梦想", "理想", "信仰", "价值观",
            "说实话", "说真的", "其实我", "一直以来",
        ]
        for w in deep_words:
            if w in text:
                score += 0.08

        return min(1.0, score)

    def _maybe_upgrade(self, r: Relationship) -> None:
        ic = r.interaction_count
        ratio = r.positive_ratio

        if r.stage == STAGE_STRANGER and ic >= THRESHOLD_ACQUAINTANCE and ratio >= MIN_POSITIVE_RATIO:
            r.stage = STAGE_ACQUAINTANCE
        elif r.stage == STAGE_ACQUAINTANCE and ic >= THRESHOLD_FRIEND and ratio >= MIN_POSITIVE_RATIO and r.user_disclosure_depth > 0.15:
            r.stage = STAGE_FRIEND
        elif r.stage == STAGE_FRIEND and ic >= THRESHOLD_CLOSE and ratio >= MIN_POSITIVE_RATIO and r.user_disclosure_depth > 0.40:
            r.stage = STAGE_CLOSE

    def update_disclosure(self, user_id: str, ai_used_depth: float) -> None:
        """Called after AI makes a self-disclosure to track depth."""
        r = self.get(user_id)
        r.self_disclosure_depth = round(
            max(r.self_disclosure_depth, ai_used_depth) * 0.6
            + r.self_disclosure_depth * 0.4,
            2,
        )

    def reset(self, user_id: str) -> None:
        r = Relationship(user_id=user_id, first_interaction=time.time())
        self._rels[user_id] = r
        self.save()

    # -- silence threshold modulation ---------------------------------

    def silence_mod(self, user_id: str) -> float:
        """Returns a multiplier on base silence threshold.

        Strangers: less silence (be polite, always respond).
        Close: more natural silence allowed."""
        stage = self.get_stage(user_id)
        return {
            STAGE_STRANGER: 0.5,
            STAGE_ACQUAINTANCE: 0.8,
            STAGE_FRIEND: 1.0,
            STAGE_CLOSE: 1.2,
        }.get(stage, 0.8)
