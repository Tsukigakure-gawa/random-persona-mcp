"""
Speech Act layer — decide *what kind of thing* to say before *how* to say it.

Sits between Emotion and Language.  Informs the LanguageProfile mapper.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any


class SpeechAct(str, Enum):
    MINIMAL_ACK        = "minimal_ack"
    BRIEF_ANSWER       = "brief_answer"
    ELABORATE_ANSWER   = "elaborate_answer"
    SELF_DISCLOSE      = "self_disclose"
    EMPATHIZE          = "empathize"
    COMPLIMENT         = "compliment"
    TEASE              = "tease"
    EXTEND_TOPIC       = "extend_topic"
    SHIFT_TOPIC        = "shift_topic"
    CLOSE_TOPIC        = "close_topic"
    QUESTION_BACK      = "question_back"
    DISAGREE           = "disagree"
    DEFLECT            = "deflect"
    APOLOGIZE          = "apologize"
    META_COMMENT       = "meta_comment"
    SEEK_CLARIFICATION = "seek_clarification"


# ---- candidates per speech act basket -----------------------------

# Each basket is influenced by emotion + trait + relationship stage
_ACT_WEIGHTS: dict[str, list[tuple[SpeechAct, float]]] = {
    # default (neutral mood, no emotion)
    "neutral": [
        (SpeechAct.BRIEF_ANSWER,       0.30),
        (SpeechAct.ELABORATE_ANSWER,   0.25),
        (SpeechAct.META_COMMENT,       0.15),
        (SpeechAct.EXTEND_TOPIC,       0.15),
        (SpeechAct.SEEK_CLARIFICATION, 0.10),
        (SpeechAct.SELF_DISCLOSE,      0.05),
    ],
    # emotion-driven baskets
    "joy_share": [
        (SpeechAct.ELABORATE_ANSWER,   0.30),
        (SpeechAct.SELF_DISCLOSE,      0.20),
        (SpeechAct.EXTEND_TOPIC,       0.20),
        (SpeechAct.COMPLIMENT,         0.15),
        (SpeechAct.TEASE,              0.10),
        (SpeechAct.META_COMMENT,       0.05),
    ],
    "joy_moderate": [
        (SpeechAct.BRIEF_ANSWER,       0.35),
        (SpeechAct.ELABORATE_ANSWER,   0.30),
        (SpeechAct.META_COMMENT,       0.20),
        (SpeechAct.EXTEND_TOPIC,       0.15),
    ],
    "anger_controlled": [
        (SpeechAct.BRIEF_ANSWER,       0.45),
        (SpeechAct.DISAGREE,           0.25),
        (SpeechAct.QUESTION_BACK,      0.15),
        (SpeechAct.MINIMAL_ACK,        0.15),
    ],
    "anger_reappraisal": [
        (SpeechAct.ELABORATE_ANSWER,   0.35),
        (SpeechAct.META_COMMENT,       0.30),
        (SpeechAct.BRIEF_ANSWER,       0.20),
        (SpeechAct.SEEK_CLARIFICATION, 0.15),
    ],
    "anger_suppress": [
        (SpeechAct.MINIMAL_ACK,        0.40),
        (SpeechAct.BRIEF_ANSWER,       0.30),
        (SpeechAct.DEFLECT,            0.30),
    ],
    "sadness_accept": [
        (SpeechAct.BRIEF_ANSWER,       0.35),
        (SpeechAct.MINIMAL_ACK,        0.25),
        (SpeechAct.SELF_DISCLOSE,      0.20),
        (SpeechAct.EMPATHIZE,          0.20),
    ],
    "sadness_ruminate": [
        (SpeechAct.ELABORATE_ANSWER,   0.40),
        (SpeechAct.SELF_DISCLOSE,      0.30),
        (SpeechAct.META_COMMENT,       0.20),
        (SpeechAct.MINIMAL_ACK,        0.10),
    ],
    "fear_seek": [
        (SpeechAct.SEEK_CLARIFICATION, 0.35),
        (SpeechAct.EMPATHIZE,          0.25),
        (SpeechAct.BRIEF_ANSWER,       0.20),
        (SpeechAct.MINIMAL_ACK,        0.20),
    ],
    "fear_reappraise": [
        (SpeechAct.ELABORATE_ANSWER,   0.35),
        (SpeechAct.META_COMMENT,       0.30),
        (SpeechAct.BRIEF_ANSWER,       0.25),
        (SpeechAct.SEEK_CLARIFICATION, 0.10),
    ],
    "surprise": [
        (SpeechAct.ELABORATE_ANSWER,   0.35),
        (SpeechAct.META_COMMENT,       0.30),
        (SpeechAct.SEEK_CLARIFICATION, 0.20),
        (SpeechAct.EXTEND_TOPIC,       0.15),
    ],
    "disgust": [
        (SpeechAct.MINIMAL_ACK,        0.35),
        (SpeechAct.BRIEF_ANSWER,       0.30),
        (SpeechAct.DEFLECT,            0.20),
        (SpeechAct.DISAGREE,           0.15),
    ],
    "gratitude": [
        (SpeechAct.EMPATHIZE,          0.30),
        (SpeechAct.COMPLIMENT,         0.25),
        (SpeechAct.ELABORATE_ANSWER,   0.25),
        (SpeechAct.SELF_DISCLOSE,      0.20),
    ],
    "hurt": [
        (SpeechAct.MINIMAL_ACK,        0.40),
        (SpeechAct.BRIEF_ANSWER,       0.25),
        (SpeechAct.DEFLECT,            0.20),
        (SpeechAct.QUESTION_BACK,      0.15),
    ],
    "guilt": [
        (SpeechAct.APOLOGIZE,          0.40),
        (SpeechAct.BRIEF_ANSWER,       0.30),
        (SpeechAct.SELF_DISCLOSE,      0.20),
        (SpeechAct.MINIMAL_ACK,        0.10),
    ],
    "anticipation": [
        (SpeechAct.EXTEND_TOPIC,       0.35),
        (SpeechAct.ELABORATE_ANSWER,   0.30),
        (SpeechAct.META_COMMENT,       0.20),
        (SpeechAct.SEEK_CLARIFICATION, 0.15),
    ],
}


_BASKET_MAP: dict[tuple[str | None, str | None], str] = {}

def _get_basket(emotion_label: str | None, regulation: str | None) -> str:
    """Map (emotion, regulation) pair to a basket key."""
    key = (emotion_label, regulation)
    if key in _BASKET_MAP:
        return _BASKET_MAP[key]

    if emotion_label is None:
        return "neutral"

    from state import JOY, ANGER, SADNESS, FEAR, SURPRISE, DISGUST, GRATITUDE, HURT, GUILT, ANTICIPATION, TRUST

    if emotion_label == JOY:
        basket = "joy_moderate" if regulation == "moderate" else "joy_share"
    elif emotion_label == ANGER:
        if regulation == "suppression":
            basket = "anger_suppress"
        elif regulation == "reappraisal":
            basket = "anger_reappraisal"
        else:
            basket = "anger_controlled"
    elif emotion_label == SADNESS:
        basket = "sadness_ruminate" if regulation == "rumination" else "sadness_accept"
    elif emotion_label == FEAR:
        basket = "fear_reappraise" if regulation == "reappraisal" else "fear_seek"
    elif emotion_label == SURPRISE:
        basket = "surprise"
    elif emotion_label == DISGUST:
        basket = "disgust"
    elif emotion_label == GRATITUDE or emotion_label == TRUST:
        basket = "gratitude"
    elif emotion_label == HURT:
        basket = "hurt"
    elif emotion_label == GUILT:
        basket = "guilt"
    elif emotion_label == ANTICIPATION:
        basket = "anticipation"
    else:
        basket = "neutral"

    _BASKET_MAP[key] = basket
    return basket


# ---- relationship stage restrictions ------------------------------

_STAGE_RESTRICTED: dict[str, list[SpeechAct]] = {
    "stranger":      [SpeechAct.TEASE, SpeechAct.SELF_DISCLOSE],
    "acquaintance":  [SpeechAct.TEASE],
    "friend":        [],
    "close":         [],
}


def select_speech_act(
    emotion_label: str | None = None,
    regulation: str | None = None,
    silence: bool = False,
    user_has_question: bool = True,
    relationship_stage: str = "stranger",
    trait_extraversion: float = 0.50,
    trait_openness: float = 0.55,
) -> SpeechAct:
    """Pick a SpeechAct based on the full internal state."""

    # silence → always minimal
    if silence:
        return SpeechAct.MINIMAL_ACK

    basket_key = _get_basket(emotion_label, regulation)
    candidates = _ACT_WEIGHTS.get(basket_key, _ACT_WEIGHTS["neutral"])
    names, base_weights = zip(*candidates)
    weights = list(base_weights)

    # trait modulation
    if trait_extraversion < 0.3:
        # introvert → boost minimal/brief, nerf elaborate/extend
        for i, act in enumerate(names):
            if act in (SpeechAct.MINIMAL_ACK, SpeechAct.BRIEF_ANSWER):
                weights[i] *= 1.4
            elif act in (SpeechAct.EXTEND_TOPIC, SpeechAct.SELF_DISCLOSE):
                weights[i] *= 0.6
    elif trait_extraversion > 0.75:
        for i, act in enumerate(names):
            if act in (SpeechAct.EXTEND_TOPIC, SpeechAct.SELF_DISCLOSE, SpeechAct.TEASE):
                weights[i] *= 1.35

    if trait_openness < 0.3:
        for i, act in enumerate(names):
            if act in (SpeechAct.META_COMMENT, SpeechAct.SEEK_CLARIFICATION):
                weights[i] *= 0.65

    # user didn't ask a question → nerf elaborate_answer
    if not user_has_question:
        for i, act in enumerate(names):
            if act == SpeechAct.ELABORATE_ANSWER:
                weights[i] *= 0.50

    # relationship restriction
    restricted = _STAGE_RESTRICTED.get(relationship_stage, [])
    for i, act in enumerate(names):
        if act in restricted:
            weights[i] = 0.0

    # fallback
    if sum(weights) <= 0:
        return SpeechAct.BRIEF_ANSWER

    return random.choices(names, weights=weights, k=1)[0]
