"""
Emotion Lexicon — combined DUTIR + NRC-VAD backed word lookup.

Provides PAD-continuous querying for Chinese emotion vocabulary,
used by language.py to inject data-driven word recommendations into
the system prompt instead of vague "语气温暖" instructions.

Architecture:
  - curated_lexicon.json: ~600 Chinese emotion words with V/A/D ratings
    (curated from DUTIR categories + NRC-VAD approximate mappings)
  - Query by PAD proximity (cosine or Euclidean)
  - Query by emotion category (joy/anger/sadness/etc.)
"""


import json
import math
import os
from typing import Any


# ---- loader -------------------------------------------------------

class EmotionLexicon:
    """In-memory emotion lexicon with PAD similarity search."""

    def __init__(self, data_dir: str) -> None:
        self.words: "List[Dict[str, Any]]" = []
        self._by_category: "Dict[str, List[Dict[str, Any]]]" = {}
        self._loaded = False
        self._data_dir = data_dir
        self._load()

    @property
    def size(self) -> int:
        return len(self.words)

    def _load(self) -> None:
        """Load curated_lexicon.json if available, merging with built-in seed."""
        # Always start with built-in seed (hand-curated, high quality)
        self.words = list(_BUILTIN_LEXICON)
        builtin_words = {w["word"] for w in self.words}

        path = os.path.join(self._data_dir, "curated_lexicon.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                dutir_words = data.get("words", [])
                # Merge: add DUTIR words not already in built-in seed
                added = 0
                for dw in dutir_words:
                    if dw["word"] not in builtin_words:
                        self.words.append(dw)
                        added += 1
                if added:
                    print(f"[Lexicon] Merged {added} DUTIR words ({len(self.words)} total)")
            except Exception:
                pass  # fall through to built-in only

        self._index()
        self._loaded = True

    def _index(self) -> None:
        self._by_category = {}
        for w in self.words:
            cat = w.get("category", "neutral")
            self._by_category.setdefault(cat, []).append(w)

    # ---- query by PAD ----------------------------------------------

    def query_pad(
        self,
        v: float,
        a: float,
        d: float,
        top_k: int = 15,
        exclude_negative: bool = True,
    ) -> "List[str]":
        """Return top-K words closest to target (v,a,d) by Euclidean distance.

        Args:
            v/a/d: target PAD values (0-1)
            top_k: number of words to return
            exclude_negative: if True, exclude words too far away (>0.5 dist)
        """
        scored: list[tuple[float, str, float]] = []
        for w in self.words:
            wv = w.get("v", 0.5)
            wa = w.get("a", 0.5)
            wd = w.get("d", 0.5)
            dist = math.sqrt((v - wv) ** 2 + (a - wa) ** 2 + (d - wd) ** 2)
            if exclude_negative and dist > 0.50:
                continue
            scored.append((dist, w["word"], w.get("intensity", 0.5)))

        scored.sort(key=lambda x: x[0])
        return [s[1] for s in scored[:top_k]]

    def query_pad_weighted(
        self,
        v: float,
        a: float,
        d: float,
        top_k: int = 12,
        v_weight: float = 1.0,
        a_weight: float = 0.8,
        d_weight: float = 0.6,
    ) -> "List[str]":
        """Weighted PAD query — valence matters most by default."""
        scored: list[tuple[float, str]] = []
        for w in self.words:
            wv = w.get("v", 0.5)
            wa = w.get("a", 0.5)
            wd = w.get("d", 0.5)
            dist = math.sqrt(
                v_weight * (v - wv) ** 2
                + a_weight * (a - wa) ** 2
                + d_weight * (d - wd) ** 2
            )
            scored.append((dist, w["word"]))
        scored.sort(key=lambda x: x[0])
        return [s[1] for s in scored[:top_k]]

    # ---- query by category -----------------------------------------

    def query_category(self, category: str, top_k: int = 10) -> "List[str]":
        """Return top words in emotion category, sorted by intensity desc."""
        words = self._by_category.get(category, [])
        words = sorted(words, key=lambda w: w.get("intensity", 0.5), reverse=True)
        return [w["word"] for w in words[:top_k]]

    def query_avoid(self, v: float, a: float, d: float, top_k: int = 8) -> "List[str]":
        """Return words FARTHEST from target (for 'avoid' list)."""
        scored: list[tuple[float, str]] = []
        for w in self.words:
            wv = w.get("v", 0.5)
            wa = w.get("a", 0.5)
            wd = w.get("d", 0.5)
            dist = math.sqrt((v - wv) ** 2 + (a - wa) ** 2 + (d - wd) ** 2)
            scored.append((dist, w["word"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k]]


# ---- built-in curated lexicon (~600 words) ------------------------

# Each entry: {word, v, a, d, category, intensity}
# v=valence, a=arousal, d=dominance (all 0-1)
# category: joy/anger/sadness/fear/surprise/disgust/trust/anticipation/neutral
# intensity: 0-1

_BUILTIN_LEXICON: "List[Dict[str, Any]]" = [
    # ── JOY (高valence, 中高arousal, 中高dominance) ──
    {"word": "开心", "v": 0.82, "a": 0.65, "d": 0.70, "category": "joy", "intensity": 0.60},
    {"word": "快乐", "v": 0.85, "a": 0.60, "d": 0.72, "category": "joy", "intensity": 0.65},
    {"word": "高兴", "v": 0.80, "a": 0.58, "d": 0.68, "category": "joy", "intensity": 0.58},
    {"word": "喜悦", "v": 0.88, "a": 0.52, "d": 0.75, "category": "joy", "intensity": 0.70},
    {"word": "愉快", "v": 0.83, "a": 0.50, "d": 0.72, "category": "joy", "intensity": 0.60},
    {"word": "欢喜", "v": 0.85, "a": 0.62, "d": 0.70, "category": "joy", "intensity": 0.65},
    {"word": "幸福", "v": 0.92, "a": 0.45, "d": 0.80, "category": "joy", "intensity": 0.80},
    {"word": "满足", "v": 0.82, "a": 0.30, "d": 0.75, "category": "joy", "intensity": 0.55},
    {"word": "兴奋", "v": 0.78, "a": 0.88, "d": 0.65, "category": "joy", "intensity": 0.75},
    {"word": "激动", "v": 0.75, "a": 0.90, "d": 0.60, "category": "joy", "intensity": 0.80},
    {"word": "惊喜", "v": 0.82, "a": 0.85, "d": 0.58, "category": "joy", "intensity": 0.75},
    {"word": "雀跃", "v": 0.88, "a": 0.85, "d": 0.70, "category": "joy", "intensity": 0.78},
    {"word": "欢快", "v": 0.84, "a": 0.70, "d": 0.68, "category": "joy", "intensity": 0.65},
    {"word": "舒畅", "v": 0.80, "a": 0.40, "d": 0.75, "category": "joy", "intensity": 0.55},
    {"word": "欣慰", "v": 0.78, "a": 0.35, "d": 0.72, "category": "joy", "intensity": 0.55},
    {"word": "美好", "v": 0.88, "a": 0.48, "d": 0.76, "category": "joy", "intensity": 0.70},
    {"word": "惬意", "v": 0.80, "a": 0.30, "d": 0.78, "category": "joy", "intensity": 0.55},
    {"word": "庆幸", "v": 0.72, "a": 0.55, "d": 0.60, "category": "joy", "intensity": 0.60},
    {"word": "欢笑", "v": 0.85, "a": 0.75, "d": 0.70, "category": "joy", "intensity": 0.70},
    {"word": "享受", "v": 0.82, "a": 0.35, "d": 0.78, "category": "joy", "intensity": 0.60},
    {"word": "满意", "v": 0.78, "a": 0.30, "d": 0.80, "category": "joy", "intensity": 0.55},
    {"word": "乐观", "v": 0.80, "a": 0.55, "d": 0.75, "category": "joy", "intensity": 0.60},
    {"word": "热情", "v": 0.78, "a": 0.75, "d": 0.72, "category": "joy", "intensity": 0.65},
    {"word": "向往", "v": 0.80, "a": 0.55, "d": 0.70, "category": "joy", "intensity": 0.60},
    {"word": "热爱", "v": 0.88, "a": 0.65, "d": 0.75, "category": "joy", "intensity": 0.75},
    {"word": "感动", "v": 0.78, "a": 0.60, "d": 0.45, "category": "joy", "intensity": 0.70},

    # ── ANGER (低valence, 高arousal, 高dominance) ──
    {"word": "愤怒", "v": 0.10, "a": 0.88, "d": 0.78, "category": "anger", "intensity": 0.85},
    {"word": "生气", "v": 0.12, "a": 0.80, "d": 0.72, "category": "anger", "intensity": 0.70},
    {"word": "恼怒", "v": 0.12, "a": 0.82, "d": 0.70, "category": "anger", "intensity": 0.75},
    {"word": "愤慨", "v": 0.08, "a": 0.85, "d": 0.80, "category": "anger", "intensity": 0.80},
    {"word": "气愤", "v": 0.10, "a": 0.80, "d": 0.75, "category": "anger", "intensity": 0.70},
    {"word": "烦躁", "v": 0.15, "a": 0.78, "d": 0.60, "category": "anger", "intensity": 0.65},
    {"word": "恼火", "v": 0.12, "a": 0.82, "d": 0.72, "category": "anger", "intensity": 0.72},
    {"word": "暴躁", "v": 0.08, "a": 0.90, "d": 0.80, "category": "anger", "intensity": 0.85},
    {"word": "狂怒", "v": 0.05, "a": 0.95, "d": 0.85, "category": "anger", "intensity": 0.90},
    {"word": "怨恨", "v": 0.08, "a": 0.65, "d": 0.55, "category": "anger", "intensity": 0.75},
    {"word": "不满", "v": 0.20, "a": 0.55, "d": 0.65, "category": "anger", "intensity": 0.55},
    {"word": "不快", "v": 0.22, "a": 0.50, "d": 0.60, "category": "anger", "intensity": 0.50},
    {"word": "窝火", "v": 0.12, "a": 0.72, "d": 0.58, "category": "anger", "intensity": 0.65},
    {"word": "可气", "v": 0.15, "a": 0.70, "d": 0.65, "category": "anger", "intensity": 0.60},
    {"word": "憋屈", "v": 0.12, "a": 0.60, "d": 0.35, "category": "anger", "intensity": 0.60},
    {"word": "忍无可忍", "v": 0.05, "a": 0.88, "d": 0.78, "category": "anger", "intensity": 0.85},

    # ── SADNESS (低valence, 低arousal, 低dominance) ──
    {"word": "悲伤", "v": 0.10, "a": 0.15, "d": 0.12, "category": "sadness", "intensity": 0.75},
    {"word": "难过", "v": 0.12, "a": 0.20, "d": 0.15, "category": "sadness", "intensity": 0.65},
    {"word": "伤心", "v": 0.10, "a": 0.25, "d": 0.12, "category": "sadness", "intensity": 0.70},
    {"word": "悲哀", "v": 0.08, "a": 0.12, "d": 0.10, "category": "sadness", "intensity": 0.75},
    {"word": "忧伤", "v": 0.12, "a": 0.15, "d": 0.15, "category": "sadness", "intensity": 0.65},
    {"word": "悲痛", "v": 0.05, "a": 0.25, "d": 0.08, "category": "sadness", "intensity": 0.85},
    {"word": "哀伤", "v": 0.08, "a": 0.15, "d": 0.10, "category": "sadness", "intensity": 0.70},
    {"word": "沮丧", "v": 0.12, "a": 0.15, "d": 0.15, "category": "sadness", "intensity": 0.60},
    {"word": "消沉", "v": 0.15, "a": 0.10, "d": 0.18, "category": "sadness", "intensity": 0.55},
    {"word": "失落", "v": 0.15, "a": 0.18, "d": 0.20, "category": "sadness", "intensity": 0.55},
    {"word": "孤独", "v": 0.15, "a": 0.20, "d": 0.18, "category": "sadness", "intensity": 0.60},
    {"word": "寂寞", "v": 0.18, "a": 0.15, "d": 0.22, "category": "sadness", "intensity": 0.55},
    {"word": "绝望", "v": 0.02, "a": 0.25, "d": 0.05, "category": "sadness", "intensity": 0.90},
    {"word": "无助", "v": 0.10, "a": 0.25, "d": 0.05, "category": "sadness", "intensity": 0.65},
    {"word": "心灰意冷", "v": 0.08, "a": 0.08, "d": 0.10, "category": "sadness", "intensity": 0.70},
    {"word": "怅然", "v": 0.18, "a": 0.12, "d": 0.22, "category": "sadness", "intensity": 0.50},
    {"word": "惋惜", "v": 0.20, "a": 0.20, "d": 0.30, "category": "sadness", "intensity": 0.50},
    {"word": "遗憾", "v": 0.22, "a": 0.25, "d": 0.30, "category": "sadness", "intensity": 0.50},

    # ── FEAR (低valence, 高arousal, 低dominance) ──
    {"word": "恐惧", "v": 0.08, "a": 0.82, "d": 0.08, "category": "fear", "intensity": 0.80},
    {"word": "害怕", "v": 0.10, "a": 0.75, "d": 0.10, "category": "fear", "intensity": 0.70},
    {"word": "惊恐", "v": 0.05, "a": 0.90, "d": 0.05, "category": "fear", "intensity": 0.85},
    {"word": "畏惧", "v": 0.08, "a": 0.70, "d": 0.08, "category": "fear", "intensity": 0.70},
    {"word": "恐慌", "v": 0.05, "a": 0.88, "d": 0.05, "category": "fear", "intensity": 0.82},
    {"word": "焦虑", "v": 0.12, "a": 0.72, "d": 0.15, "category": "fear", "intensity": 0.65},
    {"word": "不安", "v": 0.15, "a": 0.65, "d": 0.18, "category": "fear", "intensity": 0.58},
    {"word": "紧张", "v": 0.18, "a": 0.75, "d": 0.20, "category": "fear", "intensity": 0.60},
    {"word": "担忧", "v": 0.15, "a": 0.60, "d": 0.22, "category": "fear", "intensity": 0.55},
    {"word": "忐忑", "v": 0.15, "a": 0.65, "d": 0.15, "category": "fear", "intensity": 0.60},
    {"word": "慌张", "v": 0.12, "a": 0.80, "d": 0.12, "category": "fear", "intensity": 0.68},
    {"word": "胆怯", "v": 0.12, "a": 0.60, "d": 0.10, "category": "fear", "intensity": 0.60},
    {"word": "战栗", "v": 0.05, "a": 0.88, "d": 0.05, "category": "fear", "intensity": 0.80},
    {"word": "惊惶", "v": 0.06, "a": 0.88, "d": 0.06, "category": "fear", "intensity": 0.82},

    # ── DISGUST (低valence, 中arousal, 中dominance) ──
    {"word": "厌恶", "v": 0.10, "a": 0.55, "d": 0.60, "category": "disgust", "intensity": 0.70},
    {"word": "反感", "v": 0.12, "a": 0.50, "d": 0.58, "category": "disgust", "intensity": 0.62},
    {"word": "讨厌", "v": 0.12, "a": 0.55, "d": 0.60, "category": "disgust", "intensity": 0.60},
    {"word": "恶心", "v": 0.08, "a": 0.65, "d": 0.55, "category": "disgust", "intensity": 0.72},
    {"word": "嫌弃", "v": 0.10, "a": 0.50, "d": 0.65, "category": "disgust", "intensity": 0.60},
    {"word": "憎恶", "v": 0.05, "a": 0.65, "d": 0.62, "category": "disgust", "intensity": 0.78},
    {"word": "厌烦", "v": 0.12, "a": 0.52, "d": 0.55, "category": "disgust", "intensity": 0.58},
    {"word": "不屑", "v": 0.18, "a": 0.40, "d": 0.72, "category": "disgust", "intensity": 0.50},
    {"word": "鄙夷", "v": 0.12, "a": 0.45, "d": 0.75, "category": "disgust", "intensity": 0.62},
    {"word": "反胃", "v": 0.08, "a": 0.60, "d": 0.50, "category": "disgust", "intensity": 0.65},

    # ── SURPRISE (中valence, 高arousal, 中dominance) ──
    {"word": "惊讶", "v": 0.55, "a": 0.75, "d": 0.40, "category": "surprise", "intensity": 0.65},
    {"word": "吃惊", "v": 0.50, "a": 0.78, "d": 0.38, "category": "surprise", "intensity": 0.68},
    {"word": "震惊", "v": 0.35, "a": 0.90, "d": 0.30, "category": "surprise", "intensity": 0.82},
    {"word": "诧异", "v": 0.48, "a": 0.70, "d": 0.42, "category": "surprise", "intensity": 0.62},
    {"word": "意外", "v": 0.50, "a": 0.65, "d": 0.45, "category": "surprise", "intensity": 0.58},
    {"word": "愕然", "v": 0.40, "a": 0.72, "d": 0.35, "category": "surprise", "intensity": 0.65},
    {"word": "惊叹", "v": 0.60, "a": 0.78, "d": 0.45, "category": "surprise", "intensity": 0.68},
    {"word": "目瞪口呆", "v": 0.40, "a": 0.88, "d": 0.30, "category": "surprise", "intensity": 0.80},

    # ── TRUST (高valence, 中低arousal, 中高dominance) ──
    {"word": "信任", "v": 0.78, "a": 0.35, "d": 0.75, "category": "trust", "intensity": 0.65},
    {"word": "信赖", "v": 0.80, "a": 0.30, "d": 0.78, "category": "trust", "intensity": 0.68},
    {"word": "放心", "v": 0.75, "a": 0.25, "d": 0.72, "category": "trust", "intensity": 0.58},
    {"word": "可靠", "v": 0.78, "a": 0.30, "d": 0.80, "category": "trust", "intensity": 0.62},
    {"word": "安心", "v": 0.80, "a": 0.20, "d": 0.75, "category": "trust", "intensity": 0.60},
    {"word": "安稳", "v": 0.78, "a": 0.15, "d": 0.78, "category": "trust", "intensity": 0.55},

    # ── ANTICIPATION (中高valence, 中高arousal, 中dominance) ──
    {"word": "期待", "v": 0.68, "a": 0.58, "d": 0.60, "category": "anticipation", "intensity": 0.60},
    {"word": "期望", "v": 0.65, "a": 0.50, "d": 0.62, "category": "anticipation", "intensity": 0.55},
    {"word": "盼望", "v": 0.70, "a": 0.55, "d": 0.58, "category": "anticipation", "intensity": 0.62},
    {"word": "憧憬", "v": 0.78, "a": 0.55, "d": 0.65, "category": "anticipation", "intensity": 0.65},
    {"word": "好奇", "v": 0.65, "a": 0.60, "d": 0.55, "category": "anticipation", "intensity": 0.58},
    {"word": "跃跃欲试", "v": 0.72, "a": 0.72, "d": 0.68, "category": "anticipation", "intensity": 0.68},

    # ── GRATITUDE (高valence, 中arousal, 中dominance) ──
    {"word": "感激", "v": 0.80, "a": 0.50, "d": 0.55, "category": "trust", "intensity": 0.68},
    {"word": "感谢", "v": 0.78, "a": 0.48, "d": 0.58, "category": "trust", "intensity": 0.62},
    {"word": "感动", "v": 0.82, "a": 0.58, "d": 0.45, "category": "trust", "intensity": 0.70},

    # ── GUILT (低valence, 中arousal, 低dominance) ──
    {"word": "愧疚", "v": 0.10, "a": 0.45, "d": 0.12, "category": "sadness", "intensity": 0.65},
    {"word": "自责", "v": 0.08, "a": 0.50, "d": 0.10, "category": "sadness", "intensity": 0.68},
    {"word": "后悔", "v": 0.10, "a": 0.48, "d": 0.15, "category": "sadness", "intensity": 0.62},
    {"word": "抱歉", "v": 0.18, "a": 0.40, "d": 0.20, "category": "sadness", "intensity": 0.55},
    {"word": "遗憾", "v": 0.22, "a": 0.35, "d": 0.25, "category": "sadness", "intensity": 0.50},
    {"word": "忏悔", "v": 0.08, "a": 0.50, "d": 0.10, "category": "sadness", "intensity": 0.72},

    # ── HURT (低valence, 中arousal, 中低dominance) ──
    {"word": "受伤", "v": 0.10, "a": 0.55, "d": 0.18, "category": "sadness", "intensity": 0.65},
    {"word": "委屈", "v": 0.10, "a": 0.58, "d": 0.15, "category": "sadness", "intensity": 0.65},
    {"word": "失望", "v": 0.12, "a": 0.45, "d": 0.25, "category": "sadness", "intensity": 0.60},
    {"word": "寒心", "v": 0.08, "a": 0.40, "d": 0.15, "category": "sadness", "intensity": 0.62},
    {"word": "痛心", "v": 0.05, "a": 0.55, "d": 0.12, "category": "sadness", "intensity": 0.70},

    # ── HIGH-AROUSAL NEUTRAL ──
    {"word": "紧张", "v": 0.25, "a": 0.78, "d": 0.25, "category": "neutral", "intensity": 0.55},
    {"word": "兴奋", "v": 0.65, "a": 0.82, "d": 0.60, "category": "neutral", "intensity": 0.62},
    {"word": "激烈", "v": 0.40, "a": 0.85, "d": 0.55, "category": "neutral", "intensity": 0.65},
    {"word": "急迫", "v": 0.30, "a": 0.80, "d": 0.50, "category": "neutral", "intensity": 0.60},
    {"word": "匆忙", "v": 0.35, "a": 0.78, "d": 0.40, "category": "neutral", "intensity": 0.55},
    {"word": "热烈", "v": 0.68, "a": 0.80, "d": 0.62, "category": "neutral", "intensity": 0.65},

    # ── LOW-AROUSAL NEUTRAL ──
    {"word": "平静", "v": 0.60, "a": 0.15, "d": 0.65, "category": "neutral", "intensity": 0.45},
    {"word": "冷静", "v": 0.55, "a": 0.15, "d": 0.70, "category": "neutral", "intensity": 0.50},
    {"word": "安静", "v": 0.58, "a": 0.10, "d": 0.60, "category": "neutral", "intensity": 0.40},
    {"word": "沉着", "v": 0.55, "a": 0.12, "d": 0.75, "category": "neutral", "intensity": 0.50},
    {"word": "沉稳", "v": 0.55, "a": 0.12, "d": 0.78, "category": "neutral", "intensity": 0.50},
    {"word": "从容", "v": 0.60, "a": 0.18, "d": 0.78, "category": "neutral", "intensity": 0.52},
    {"word": "平淡", "v": 0.48, "a": 0.12, "d": 0.50, "category": "neutral", "intensity": 0.30},
    {"word": "宁静", "v": 0.65, "a": 0.08, "d": 0.65, "category": "neutral", "intensity": 0.45},
    {"word": "轻松", "v": 0.70, "a": 0.25, "d": 0.68, "category": "neutral", "intensity": 0.50},
    {"word": "悠闲", "v": 0.72, "a": 0.18, "d": 0.70, "category": "neutral", "intensity": 0.48},
    {"word": "自在", "v": 0.75, "a": 0.22, "d": 0.75, "category": "neutral", "intensity": 0.55},
    {"word": "舒适", "v": 0.72, "a": 0.18, "d": 0.70, "category": "neutral", "intensity": 0.48},
    {"word": "放松", "v": 0.70, "a": 0.15, "d": 0.70, "category": "neutral", "intensity": 0.45},

    # ── HIGH-DOMINANCE NEUTRAL ──
    {"word": "自信", "v": 0.72, "a": 0.45, "d": 0.88, "category": "neutral", "intensity": 0.62},
    {"word": "坚定", "v": 0.65, "a": 0.48, "d": 0.90, "category": "neutral", "intensity": 0.65},
    {"word": "果断", "v": 0.62, "a": 0.50, "d": 0.88, "category": "neutral", "intensity": 0.62},
    {"word": "果敢", "v": 0.62, "a": 0.55, "d": 0.90, "category": "neutral", "intensity": 0.65},
    {"word": "坚决", "v": 0.58, "a": 0.52, "d": 0.88, "category": "neutral", "intensity": 0.62},
    {"word": "主动", "v": 0.65, "a": 0.55, "d": 0.82, "category": "neutral", "intensity": 0.58},
    {"word": "强力", "v": 0.58, "a": 0.58, "d": 0.88, "category": "neutral", "intensity": 0.60},
    {"word": "主导", "v": 0.55, "a": 0.50, "d": 0.90, "category": "neutral", "intensity": 0.60},
    {"word": "掌控", "v": 0.55, "a": 0.48, "d": 0.92, "category": "neutral", "intensity": 0.62},

    # ── LOW-DOMINANCE NEUTRAL ──
    {"word": "迷茫", "v": 0.35, "a": 0.40, "d": 0.15, "category": "neutral", "intensity": 0.52},
    {"word": "犹豫", "v": 0.38, "a": 0.42, "d": 0.18, "category": "neutral", "intensity": 0.50},
    {"word": "困惑", "v": 0.35, "a": 0.48, "d": 0.22, "category": "neutral", "intensity": 0.55},
    {"word": "迟疑", "v": 0.38, "a": 0.40, "d": 0.20, "category": "neutral", "intensity": 0.48},
    {"word": "被动", "v": 0.35, "a": 0.25, "d": 0.15, "category": "neutral", "intensity": 0.42},
    {"word": "顺从", "v": 0.42, "a": 0.20, "d": 0.12, "category": "neutral", "intensity": 0.40},
    {"word": "依赖", "v": 0.40, "a": 0.35, "d": 0.10, "category": "neutral", "intensity": 0.45},
    {"word": "无奈", "v": 0.25, "a": 0.30, "d": 0.15, "category": "neutral", "intensity": 0.50},

    # ── POSITIVE-HIGH AROUSAL (warm) ──
    {"word": "热心", "v": 0.78, "a": 0.62, "d": 0.70, "category": "joy", "intensity": 0.60},
    {"word": "真诚", "v": 0.82, "a": 0.45, "d": 0.72, "category": "joy", "intensity": 0.62},
    {"word": "友善", "v": 0.80, "a": 0.42, "d": 0.68, "category": "joy", "intensity": 0.55},
    {"word": "亲切", "v": 0.82, "a": 0.48, "d": 0.70, "category": "joy", "intensity": 0.58},
    {"word": "温暖", "v": 0.85, "a": 0.35, "d": 0.72, "category": "joy", "intensity": 0.60},
    {"word": "体贴", "v": 0.82, "a": 0.30, "d": 0.68, "category": "joy", "intensity": 0.55},
    {"word": "温馨", "v": 0.85, "a": 0.25, "d": 0.70, "category": "joy", "intensity": 0.58},
    {"word": "舒服", "v": 0.80, "a": 0.22, "d": 0.72, "category": "joy", "intensity": 0.55},
    {"word": "自然", "v": 0.72, "a": 0.35, "d": 0.72, "category": "neutral", "intensity": 0.50},
    {"word": "好玩", "v": 0.78, "a": 0.62, "d": 0.58, "category": "joy", "intensity": 0.55},
    {"word": "有趣", "v": 0.78, "a": 0.55, "d": 0.62, "category": "joy", "intensity": 0.58},
    {"word": "有意思", "v": 0.78, "a": 0.52, "d": 0.60, "category": "joy", "intensity": 0.55},
    {"word": "精彩", "v": 0.82, "a": 0.65, "d": 0.65, "category": "joy", "intensity": 0.65},
    {"word": "棒", "v": 0.82, "a": 0.62, "d": 0.68, "category": "joy", "intensity": 0.60},
    {"word": "妙", "v": 0.80, "a": 0.55, "d": 0.65, "category": "joy", "intensity": 0.58},
    {"word": "厉害", "v": 0.78, "a": 0.62, "d": 0.72, "category": "joy", "intensity": 0.60},
    {"word": "了不起", "v": 0.82, "a": 0.62, "d": 0.72, "category": "joy", "intensity": 0.65},
    {"word": "佩服", "v": 0.78, "a": 0.48, "d": 0.58, "category": "joy", "intensity": 0.60},

    # ── negative descriptors (for "avoid" lists) ──
    {"word": "糟糕", "v": 0.10, "a": 0.55, "d": 0.35, "category": "anger", "intensity": 0.55},
    {"word": "可怕", "v": 0.08, "a": 0.72, "d": 0.15, "category": "fear", "intensity": 0.62},
    {"word": "可怜", "v": 0.22, "a": 0.35, "d": 0.20, "category": "sadness", "intensity": 0.50},
    {"word": "可悲", "v": 0.10, "a": 0.30, "d": 0.18, "category": "sadness", "intensity": 0.58},
    {"word": "无聊", "v": 0.20, "a": 0.18, "d": 0.30, "category": "neutral", "intensity": 0.40},
    {"word": "枯燥", "v": 0.18, "a": 0.15, "d": 0.28, "category": "neutral", "intensity": 0.38},
    {"word": "乏味", "v": 0.18, "a": 0.12, "d": 0.28, "category": "neutral", "intensity": 0.35},
    {"word": "尴尬", "v": 0.20, "a": 0.58, "d": 0.25, "category": "neutral", "intensity": 0.52},
    {"word": "丢脸", "v": 0.12, "a": 0.60, "d": 0.10, "category": "sadness", "intensity": 0.58},
    {"word": "羞耻", "v": 0.10, "a": 0.55, "d": 0.08, "category": "sadness", "intensity": 0.60},

    # ── COGNITIVE / ANALYTICAL ──
    {"word": "冷静", "v": 0.55, "a": 0.15, "d": 0.72, "category": "neutral", "intensity": 0.48},
    {"word": "理性", "v": 0.55, "a": 0.18, "d": 0.75, "category": "neutral", "intensity": 0.50},
    {"word": "客观", "v": 0.52, "a": 0.18, "d": 0.72, "category": "neutral", "intensity": 0.48},
    {"word": "清晰", "v": 0.62, "a": 0.35, "d": 0.78, "category": "neutral", "intensity": 0.52},
    {"word": "明确", "v": 0.60, "a": 0.35, "d": 0.80, "category": "neutral", "intensity": 0.52},
    {"word": "直接", "v": 0.52, "a": 0.40, "d": 0.78, "category": "neutral", "intensity": 0.50},
    {"word": "简洁", "v": 0.55, "a": 0.30, "d": 0.75, "category": "neutral", "intensity": 0.45},
    {"word": "精确", "v": 0.55, "a": 0.28, "d": 0.80, "category": "neutral", "intensity": 0.48},
    {"word": "严谨", "v": 0.52, "a": 0.25, "d": 0.82, "category": "neutral", "intensity": 0.50},
    {"word": "稳重", "v": 0.58, "a": 0.18, "d": 0.82, "category": "neutral", "intensity": 0.50},
    {"word": "务实", "v": 0.55, "a": 0.25, "d": 0.78, "category": "neutral", "intensity": 0.48},
    {"word": "实际", "v": 0.50, "a": 0.22, "d": 0.75, "category": "neutral", "intensity": 0.45},

    # ── POETIC / LITERARY ──
    {"word": "柔美", "v": 0.82, "a": 0.28, "d": 0.60, "category": "joy", "intensity": 0.52},
    {"word": "绚烂", "v": 0.80, "a": 0.58, "d": 0.62, "category": "joy", "intensity": 0.60},
    {"word": "清澈", "v": 0.78, "a": 0.30, "d": 0.65, "category": "joy", "intensity": 0.52},
    {"word": "朦胧", "v": 0.52, "a": 0.22, "d": 0.35, "category": "neutral", "intensity": 0.40},
    {"word": "苍凉", "v": 0.18, "a": 0.15, "d": 0.25, "category": "sadness", "intensity": 0.52},
    {"word": "凄美", "v": 0.28, "a": 0.22, "d": 0.22, "category": "sadness", "intensity": 0.52},
]

# Keep growing this list — the full DUTIR+NRC-VAD merge target is ~30k entries.
# Current curated set: ~170 entries covering all 8 emotion categories + neutral.
