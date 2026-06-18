"""
Appraisal engine — detect cognitive-appraisal cues from user messages
and trigger discrete emotions with intensity.

ALL rule-based.  Zero LLM token cost.
"""

from __future__ import annotations

import re
from typing import Any

from .state import (
    JOY, SADNESS, ANGER, FEAR, SURPRISE, DISGUST,
    TRUST, ANTICIPATION, GUILT, GRATITUDE, HURT,
)

# ---- rule table ---------------------------------------------------

# Each rule: (regex_pattern, appraisal_hints, emotion_label, base_intensity)

_AppraisalHint = dict[str, float]
_Rule = tuple[str, _AppraisalHint, str, float]

RULES: list[_Rule] = [
    # ── ANGER ──
    (r"烦死了|烦死|气死我了|气死|气炸|无语[了死]?|受不了|忍不了|忍无可忍",
     {"goal_congruence": -0.70, "agency": 1.0}, ANGER, 0.75),
    (r"凭什么|怎么这样|太过分|太离谱|就这[?？]?|搞笑呢",
     {"goal_congruence": -0.55, "agency": 1.0, "norm_compatibility": -0.50}, ANGER, 0.60),
    (r"别烦我|滚[开蛋]?|闭嘴|走开|够了|别说了|少来",
     {"goal_congruence": -0.80, "agency": 1.0}, ANGER, 0.80),
    (r"你是不是[有在]?病|你[真很]?[有]?毛病|什么玩意|什么东西",
     {"goal_congruence": -0.60, "agency": 1.0, "norm_compatibility": -0.40}, ANGER, 0.65),
    (r"操|草[泥尼]|tm|他妈|tnnd|sb|傻[逼叉]|脑残|智障|弱智",
     {"goal_congruence": -0.75, "agency": 1.0, "norm_compatibility": -0.70}, ANGER, 0.85),

    # ── JOY ──
    (r"哈哈{2,}|笑死[我了]?|太好笑|笑[喷尿抽]|233+|www+",
     {"goal_congruence": 0.55, "expectedness": 0.25}, JOY, 0.55),
    (r"开心|高兴|快乐|幸福|满足|爽[!！]?$|真爽|太爽",
     {"goal_congruence": 0.65}, JOY, 0.65),
    (r"太棒了|好耶|nice|奈斯|赞[!！]|牛逼|nb[!！]?$|厉害",
     {"goal_congruence": 0.70, "expectedness": 0.20}, JOY, 0.60),
    (r"终于|可算|总算[是]?",
     {"goal_congruence": 0.55, "expectedness": -0.30}, JOY, 0.50),

    # ── SADNESS ──
    (r"难过|伤心|想哭|哭[了死]|好累|崩溃|绝望|没意思|没劲",
     {"goal_congruence": -0.60, "coping_potential": -0.40}, SADNESS, 0.65),
    (r"对不起|我的错|怪我|都是我[的]?问题|是我不好",
     {"goal_congruence": -0.55, "agency": -1.0}, SADNESS, 0.55),
    (r"失败[了]?|完了|毁了|没救了|放弃[了]?",
     {"goal_congruence": -0.70, "coping_potential": -0.55}, SADNESS, 0.70),
    (r"想[死挂]|不想活|活[着得]好累",
     {"goal_congruence": -0.85, "coping_potential": -0.70}, SADNESS, 0.85),

    # ── FEAR / ANXIETY ──
    (r"怎么办[啊呀]?|我好怕|好怕|害怕|不敢|吓[死我]|吓人|担心|焦虑",
     {"coping_potential": -0.50, "goal_relevance": 0.75}, FEAR, 0.55),
    (r"万一|要是.*怎么办|如果.*怎么办|不会[是又要].*吧",
     {"coping_potential": -0.45, "goal_relevance": 0.70}, FEAR, 0.50),
    (r"紧张|不安|慌[了张]?|发慌",
     {"coping_potential": -0.40}, FEAR, 0.50),

    # ── SURPRISE ──
    (r"卧槽|我靠|天哪|天呐|不是吧|居然|没想到|竟然",
     {"expectedness": -0.70}, SURPRISE, 0.65),
    (r"真的假的[!！?？]|什么[!！?？]$|啥[!！?？]$|啊[!！?？]$",
     {"expectedness": -0.60}, SURPRISE, 0.55),

    # ── DISGUST ──
    (r"恶心|恶[臭心]|下头|辣眼睛|吐了|反胃|想吐",
     {"goal_congruence": -0.45, "norm_compatibility": -0.65}, DISGUST, 0.65),
    (r"恶心[死我]了|吐了|呕|yue",
     {"goal_congruence": -0.50, "norm_compatibility": -0.70}, DISGUST, 0.70),

    # ── TRUST / GRATITUDE ──
    (r"谢谢[你]?|多谢|感谢|太感谢|感恩|3q|thx",
     {"agency": 1.0, "goal_congruence": 0.75}, GRATITUDE, 0.65),
    (r"还好有你|多亏[你了]?|幸亏|幸好[有你]?",
     {"agency": 1.0, "goal_congruence": 0.80}, GRATITUDE, 0.70),
    (r"信[任得]过你|靠谱|靠得住|就靠你了",
     {"agency": 1.0, "goal_congruence": 0.60}, TRUST, 0.55),

    # ── ANTICIPATION ──
    (r"期待|等不及|好想[要]?|快点|赶紧|迫不及待",
     {"goal_congruence": 0.45, "goal_relevance": 0.70}, ANTICIPATION, 0.55),
    (r"下一[步个章]|然后呢|后来呢|接着呢",
     {"goal_relevance": 0.65}, ANTICIPATION, 0.45),

    # ── GUILT ──
    (r"我对不起|辜负|亏欠|欠你的|是我的疏忽",
     {"goal_congruence": -0.60, "agency": -1.0}, GUILT, 0.60),

    # ── HURT ──
    (r"你[怎么]?这样|你太[让]?.*了|失望|寒心|心寒",
     {"goal_congruence": -0.50, "agency": 1.0}, HURT, 0.55),
    (r"[不别]理我|不想[说讲话聊天]|让我静静|冷静[一下会]",
     {"goal_congruence": -0.35, "agency": 1.0}, HURT, 0.45),
]

# compile once
_RULES_COMPILED: list[tuple[re.Pattern, _AppraisalHint, str, float]] = [
    (re.compile(pat), hints, label, intensity) for pat, hints, label, intensity in RULES
]


# ---- public API ----------------------------------------------------

class AppraisalEngine:
    """Evaluates user text and returns triggered emotion, if any."""

    def evaluate(self, text: str) -> dict[str, Any] | None:
        """Return {label, intensity, hints} or None."""
        if not text or not text.strip():
            return None

        text_lower = text.lower()
        best: tuple[str, float, _AppraisalHint] | None = None

        for pat, hints, label, intensity in _RULES_COMPILED:
            m = pat.search(text_lower)
            if m:
                # slightly boost intensity for longer matches
                matched_len = m.end() - m.start()
                adj_intensity = min(1.0, intensity + matched_len * 0.01)
                if best is None or adj_intensity > best[1]:
                    best = (label, adj_intensity, hints)

        if best is None:
            return None

        return {
            "label": best[0],
            "intensity": round(best[1], 2),
            "hints": best[2],
        }

    def evaluate_with_regulation(
        self, text: str, trait: Any, mood: Any
    ) -> dict[str, Any] | None:
        """Evaluate and attach a regulation strategy hint."""
        result = self.evaluate(text)
        if result is None:
            return None

        label = result["label"]
        strategies = _REGULATION_OPTIONS.get(label, [("acceptance", 1.0)])
        names, weights = zip(*strategies)
        weights = list(weights)

        # trait modulation
        if trait.reappraisal > 0.6 and "reappraisal" in names:
            idx = names.index("reappraisal")
            weights[idx] *= 1.6
        if trait.suppression > 0.5 and "suppression" in names:
            idx = names.index("suppression")
            weights[idx] *= 1.5
        if trait.rumination > 0.5 and "rumination" in names:
            idx = names.index("rumination")
            weights[idx] *= 1.4

        # mood modulation — low valence boosts reappraisal for negative emotions
        if mood.valence < 0.35 and "reappraisal" in names:
            idx = names.index("reappraisal")
            weights[idx] *= 1.3

        import random
        strategy = random.choices(names, weights=weights, k=1)[0]
        result["regulation"] = strategy
        return result


# ---- regulation strategies per emotion ----------------------------

_REGULATION_OPTIONS: dict[str, list[tuple[str, float]]] = {
    ANGER: [
        ("suppression",       0.25),
        ("reappraisal",       0.35),
        ("controlled_expression", 0.40),
    ],
    JOY: [
        ("amplify",          0.20),
        ("share",            0.55),
        ("moderate",         0.25),
    ],
    SADNESS: [
        ("acceptance",       0.35),
        ("rumination",       0.25),
        ("distraction",      0.25),
        ("reappraisal",      0.15),
    ],
    FEAR: [
        ("seek_reassurance", 0.35),
        ("avoidance",        0.25),
        ("reappraisal",      0.40),
    ],
    SURPRISE: [
        ("express",          0.60),
        ("moderate",         0.40),
    ],
    DISGUST: [
        ("suppression",      0.45),
        ("controlled_expression", 0.35),
        ("reappraisal",      0.20),
    ],
    TRUST: [
        ("share",            0.60),
        ("moderate",         0.40),
    ],
    ANTICIPATION: [
        ("express",          0.50),
        ("moderate",         0.50),
    ],
    GUILT: [
        ("acceptance",       0.30),
        ("rumination",       0.35),
        ("reappraisal",      0.35),
    ],
    GRATITUDE: [
        ("share",            0.70),
        ("moderate",         0.30),
    ],
    HURT: [
        ("suppression",      0.40),
        ("controlled_expression", 0.30),
        ("reappraisal",      0.30),
    ],
}
