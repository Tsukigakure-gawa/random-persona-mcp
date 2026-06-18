"""
Language profile mapper + lexicon-backed prompt injection.

v2.1:  Replaces vague "语气温暖" instructions with data-driven word
       recommendations from the combined emotion lexicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .lexicon import EmotionLexicon

# ---- LanguageProfile -----------------------------------------------

@dataclass
class LanguageProfile:
    # ── lexical ──
    intensifier_rate: float = 0.3
    hedge_rate: float = 0.3
    positive_lexicon: float = 0.5
    negative_lexicon: float = 0.2
    emoji_rate: float = 0.2
    filler_rate: float = 0.15
    exclamation_rate: float = 0.3

    # ── syntactic ──
    avg_sentence_len: int = 25
    complexity: float = 0.5
    ellipsis_rate: float = 0.1
    question_rate: float = 0.2

    # ── discourse ──
    response_length: str = "normal"
    politeness_strategy: str = "positive"
    turn_initiative: str = "neutral"
    self_disclosure_depth: float = 0.0
    humor_license: bool = False


# ---- mapper --------------------------------------------------------

def map_to_profile(
    mood: Any,
    trait: Any,
    emotion: Any | None = None,
    speech_act: str = "brief_answer",
    regulation: str | None = None,
    relationship_stage: str = "stranger",
) -> LanguageProfile:
    """Convert internal state → quantified linguistic parameters."""

    p = LanguageProfile()

    # ── from Mood ──
    p.positive_lexicon = round(0.15 + mood.valence * 0.70, 2)
    p.negative_lexicon = round(0.05 + (1.0 - mood.valence) * 0.50, 2)
    p.intensifier_rate = round(0.10 + mood.arousal * 0.50, 2)
    p.filler_rate        = round(0.05 + max(0.0, 1.0 - mood.arousal) * 0.25, 2)
    p.exclamation_rate   = round(0.05 + mood.arousal * 0.55, 2)
    p.complexity         = round(0.25 + mood.dominance * 0.40, 2)
    p.hedge_rate         = round(0.10 + max(0.0, 1.0 - mood.dominance) * 0.40, 2)

    # ── from Trait ──
    p.turn_initiative = "active" if trait.extraversion > 0.65 \
        else "passive" if trait.extraversion < 0.35 else "neutral"
    p.self_disclosure_depth = round(trait.extraversion * 0.35 + trait.openness * 0.25, 2)
    p.avg_sentence_len = 8 + int(trait.openness * 25) + int(trait.extraversion * 10)
    p.ellipsis_rate = round(0.05 + (1.0 - trait.conscientiousness) * 0.30, 2)

    # ── from Emotion ──
    if emotion is not None and getattr(emotion, 'is_active', False):
        p.positive_lexicon = round(p.positive_lexicon + emotion.v * emotion.intensity * 0.30, 2)
        p.negative_lexicon = round(p.negative_lexicon + (1.0 - emotion.v) * emotion.intensity * 0.30, 2)
        p.intensifier_rate = round(p.intensifier_rate + emotion.a * emotion.intensity * 0.30, 2)
        p.exclamation_rate = round(p.exclamation_rate + emotion.a * emotion.intensity * 0.25, 2)

    # ── from Speech Act ──
    _apply_speech_act_mods(p, speech_act)

    # ── from Regulation ──
    if regulation == "suppression":
        p.intensifier_rate   *= 0.50
        p.exclamation_rate   *= 0.40
        p.negative_lexicon   *= 0.60
        p.turn_initiative     = "passive"
    elif regulation == "reappraisal":
        p.complexity         += 0.15
        p.positive_lexicon   += 0.10
        p.hedge_rate         += 0.10
    elif regulation == "rumination":
        p.avg_sentence_len   += 5
        p.negative_lexicon   += 0.15
        p.turn_initiative     = "active"
    elif regulation == "amplify":
        p.intensifier_rate   *= 1.30
        p.exclamation_rate   *= 1.30
        p.filler_rate        *= 0.60
    elif regulation == "controlled_expression":
        p.intensifier_rate   *= 0.70
        p.exclamation_rate   *= 0.60
        p.complexity         += 0.05

    # ── from Relationship ──
    if relationship_stage == "stranger":
        p.humor_license = False
        p.self_disclosure_depth *= 0.30
        p.emoji_rate *= 0.40
        p.politeness_strategy = "negative"
        p.turn_initiative = "passive"
    elif relationship_stage == "acquaintance":
        p.humor_license = False
        p.self_disclosure_depth *= 0.60
        p.emoji_rate *= 0.70
        p.politeness_strategy = "positive"
    elif relationship_stage == "friend":
        p.humor_license = True
        p.emoji_rate *= 0.90
        p.politeness_strategy = "positive"
    elif relationship_stage == "close":
        p.humor_license = True
        p.politeness_strategy = "bald"
        p.turn_initiative = "active" if trait.extraversion > 0.4 else "neutral"

    # ── clamp ──
    for attr in ("intensifier_rate", "hedge_rate", "positive_lexicon", "negative_lexicon",
                 "emoji_rate", "filler_rate", "exclamation_rate", "complexity",
                 "ellipsis_rate", "question_rate", "self_disclosure_depth"):
        setattr(p, attr, round(max(0.0, min(1.0, getattr(p, attr))), 2))
    p.avg_sentence_len = max(5, min(60, p.avg_sentence_len))

    return p


def _apply_speech_act_mods(p: LanguageProfile, act: str) -> None:
    mods: dict[str, Any] = {
        "minimal_ack":        {"response_length": "minimal", "avg_sentence_len": 8,
                               "turn_initiative": "passive", "question_rate": 0.0},
        "brief_answer":       {"response_length": "brief", "avg_sentence_len": 18,
                               "turn_initiative": "passive", "question_rate": 0.1},
        "elaborate_answer":   {"response_length": "elaborate", "avg_sentence_len": 35,
                               "turn_initiative": "active"},
        "self_disclose":      {"response_length": "elaborate", "self_disclosure_depth": 0.6,
                               "filler_rate": 0.20},
        "empathize":          {"positive_lexicon": 0.65, "hedge_rate": 0.35,
                               "intensifier_rate": 0.25, "emoji_rate": 0.35},
        "compliment":         {"positive_lexicon": 0.80, "intensifier_rate": 0.55,
                               "exclamation_rate": 0.45},
        "tease":              {"humor_license": True, "question_rate": 0.30,
                               "exclamation_rate": 0.40, "complexity": 0.40},
        "extend_topic":       {"turn_initiative": "active", "question_rate": 0.40,
                               "response_length": "normal"},
        "shift_topic":        {"turn_initiative": "active", "question_rate": 0.30},
        "close_topic":        {"response_length": "brief", "turn_initiative": "passive",
                               "question_rate": 0.0},
        "question_back":      {"question_rate": 0.55, "turn_initiative": "active",
                               "response_length": "brief"},
        "disagree":           {"negative_lexicon": 0.40, "complexity": 0.55,
                               "hedge_rate": 0.15, "response_length": "brief"},
        "deflect":            {"response_length": "minimal", "turn_initiative": "passive",
                               "hedge_rate": 0.45, "question_rate": 0.0},
        "apologize":          {"hedge_rate": 0.45, "positive_lexicon": 0.50,
                               "intensifier_rate": 0.20, "response_length": "brief"},
        "meta_comment":       {"complexity": 0.65, "turn_initiative": "active",
                               "response_length": "normal"},
        "seek_clarification": {"question_rate": 0.60, "hedge_rate": 0.30,
                               "turn_initiative": "active", "response_length": "brief"},
    }
    overrides = mods.get(act, {})
    for k, v in overrides.items():
        if hasattr(p, k):
            setattr(p, k, v)


# ---- lexicon-backed prompt injection --------------------------------

class LexiconPromptBuilder:
    """Builds prompt injection using emotion lexicon for word recommendations."""

    def __init__(self, data_dir: str) -> None:
        self.lexicon = EmotionLexicon(data_dir)

    def build(
        self,
        mood: Any,
        trait: Any = None,
        emotion: Any | None = None,
        speech_act: str = "brief_answer",
        regulation: str | None = None,
        relationship_stage: str = "stranger",
        silence_muted: bool = False,
    ) -> str:
        """Build a compact behaviour prompt with lexicon-backed word hints."""

        # 1. Get full profile
        profile = map_to_profile(
            mood=mood, trait=trait, emotion=emotion,
            speech_act=speech_act, regulation=regulation,
            relationship_stage=relationship_stage,
        )
        if silence_muted:
            profile.response_length = "minimal"
            profile.turn_initiative = "passive"

        # 2. Lexicon query — preferred words
        preferred = self.lexicon.query_pad_weighted(
            mood.valence, mood.arousal, mood.dominance,
            top_k=12,
        )

        # 3. Lexicon query — words to avoid
        avoid = self.lexicon.query_avoid(
            mood.valence, mood.arousal, mood.dominance,
            top_k=6,
        )

        # 4. If active emotion, also pull category words
        if emotion is not None and getattr(emotion, 'is_active', False):
            emotion_cat = getattr(emotion, 'primary', None)
            if emotion_cat:
                cat_words = self.lexicon.query_category(emotion_cat, top_k=6)
                # merge with preferred, dedupe
                preferred = preferred[:8] + [w for w in cat_words if w not in preferred]
                preferred = preferred[:14]

        # 5. Build prompt block
        return _render_prompt(profile=profile, preferred=preferred, avoid=avoid)


def profile_to_prompt(p: LanguageProfile) -> str:
    """Legacy API: render profile without lexicon."""
    return _render_prompt(p, preferred=[], avoid=[])


def _render_prompt(
    profile: LanguageProfile,
    preferred: list[str],
    avoid: list[str],
) -> str:
    """Render behaviour instructions + word hints into ~50-token block."""

    parts: list[str] = []

    # tone
    tone_parts: list[str] = []
    if profile.positive_lexicon > 0.65:
        tone_parts.append("温暖")
    elif profile.negative_lexicon > 0.40:
        tone_parts.append("克制")
    if profile.intensifier_rate > 0.55:
        tone_parts.append("有力")
    if profile.hedge_rate > 0.50:
        tone_parts.append("委婉")
    if tone_parts:
        parts.append("语气: " + "、".join(tone_parts))
    else:
        parts.append("语气: 自然平和")

    # length
    len_map = {"minimal": "极短，一句话", "brief": "简洁", "normal": "适中", "elaborate": "偏长，可以展开"}
    parts.append("长度: " + len_map.get(profile.response_length, "适中"))

    # style
    style: list[str] = []
    if profile.emoji_rate > 0.35:
        style.append("适当 emoji")
    if profile.humor_license:
        style.append("可以轻松幽默")
    if profile.self_disclosure_depth > 0.3:
        style.append("可以带出个人感受")
    if profile.complexity < 0.35:
        style.append("短句为主")
    if profile.ellipsis_rate > 0.25:
        style.append("可以用省略")
    if style:
        parts.append("风格: " + "、".join(style))

    # turn
    if profile.turn_initiative == "active":
        parts.append("主动: 可以延伸或追问")
    elif profile.turn_initiative == "passive":
        parts.append("克制: 回应即可")

    # constraint
    constraints: list[str] = []
    if profile.exclamation_rate < 0.15:
        constraints.append("少用感叹号")
    if profile.intensifier_rate < 0.20:
        constraints.append("不用夸张修辞")

    # ── lexicon word hints ──
    word_hints: list[str] = []
    if preferred:
        word_hints.append("倾向用词: " + "、".join(preferred[:8]))
    if avoid:
        word_hints.append("避免用词: " + "、".join(avoid[:5]))

    block = "[PERSONA]\n"
    block += "\n".join(parts)
    if word_hints:
        block += "\n" + "\n".join(word_hints)
    if constraints:
        block += "\n" + "、".join(constraints)
    block += "\n[/PERSONA]"

    return block
