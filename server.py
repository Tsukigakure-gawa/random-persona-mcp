#!/usr/bin/env python3
"""
Random Persona MCP Server — exposes the full six-layer persona engine
as MCP tools for any MCP-compatible client (AstrBot, Claude Desktop, etc.).

Start:
  pip install fastmcp
  python server.py

Tools exposed:
  persona_inject       — get system-prompt injection for current state
  persona_status       — get full state dump
  persona_command      — handle /persona commands
  persona_post_process — post-process LLM response (silence right, filler)

State is persisted per session_id; relationship is per user_id.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from fastmcp import FastMCP

# ---- internal modules ----------------------------------------------
from state import StateManager, EMOTION_LABELS
from appraisal import AppraisalEngine
from speech_act import SpeechAct, select_speech_act
from language import LexiconPromptBuilder
from relationship import RelationshipManager

# ---- init ----------------------------------------------------------
DATA_DIR = os.environ.get("PERSONA_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

mcp = FastMCP("random-persona", version="2.1.0")

state_mgr = StateManager(DATA_DIR)
appraiser = AppraisalEngine()
lex_builder = LexiconPromptBuilder(os.path.join(os.path.dirname(__file__), "lexicon"))
rel_mgr = RelationshipManager(DATA_DIR)


# ---- helpers -------------------------------------------------------
def _has_question(text: str) -> bool:
    return bool(re.search(
        r"[?？]|(吗|呢|吧|啥|什么|怎么|如何|为什么|谁|哪|"
        r"几|多少|能不能|可不可以|要不要|是不是|有没有)", text or ""
    ))


_SILENCE = {
    "anger": ["嗯。", "知道了。", "行。"],
    "sadness": ["嗯…", "好。", "…"],
    "joy": ["好的~", "👌", "好嘞。"],
    "hurt": ["…", "嗯。", "好的。"],
}


# ---- MCP tools -----------------------------------------------------

@mcp.tool()
def persona_inject(
    session_id: str,
    user_id: str = "default",
    user_message: str = "",
) -> str:
    """
    Get the system-prompt injection block for the current session state.

    Call this in your LLM pipeline's pre-request hook.  Returns a
    compact (~50 token) behaviour instruction block that should be
    appended to the system prompt.

    Args:
        session_id: Unique session identifier (e.g. AstrBot unified_msg_origin)
        user_id: Unique user identifier for relationship tracking
        user_message: The user's latest message text
    """
    state = state_mgr.get_or_init(session_id)

    # 1. mood drift
    state_mgr.drift_mood(session_id)

    # 2. appraisal → emotion
    emo_result = appraiser.evaluate_with_regulation(user_message, state.trait, state.mood)
    if emo_result:
        state_mgr.trigger_emotion(session_id, emo_result["label"], emo_result["intensity"])

    # 3. emotion decay
    state_mgr.decay_emotion(session_id)

    # 4. relationship
    rel = rel_mgr.get(user_id)
    rel_mgr.record_interaction(user_id, sentiment=state.mood.valence - 0.5, user_msg=user_message)

    # 5. silence check
    has_q = _has_question(user_message)
    should_silence = (
        (not has_q and getattr(state, "last_response_len", 0) > 150)
        or (state_mgr.patience(session_id) < 0.20)
    )

    # 6. speech act
    em_label = state.emotion.primary if state.emotion else None
    reg = emo_result.get("regulation") if emo_result else None
    sa = select_speech_act(
        emotion_label=em_label,
        regulation=reg,
        silence=should_silence,
        user_has_question=has_q,
        relationship_stage=rel.stage,
        trait_extraversion=state.trait.extraversion,
        trait_openness=state.trait.openness,
    )

    # 7. build prompt block
    block = lex_builder.build(
        mood=state.mood,
        trait=state.trait,
        emotion=state.emotion,
        speech_act=sa.value if isinstance(sa, SpeechAct) else sa,
        regulation=reg,
        relationship_stage=rel.stage,
        silence_muted=should_silence,
    )

    # 8. update counter
    state.msg_count += 1
    state.last_active = time.time()
    state_mgr._maybe_save()

    return block if block else ""


@mcp.tool()
def persona_post_process(
    session_id: str,
    response_text: str,
    silence_mode: str = "短回应",
) -> dict[str, Any]:
    """
    Post-process an LLM response.  Handles silence right and filler injection.

    Args:
        session_id: Session identifier
        response_text: The raw LLM response text
        silence_mode: "短回应" or "完全不回"

    Returns:
        dict with "text" (final response) and "was_silenced" (bool)
    """
    import random as _random

    state = state_mgr.get_state(session_id)
    if state is None:
        return {"text": response_text, "was_silenced": False}

    has_q = _has_question("")  # we don't have user msg here — already passed
    should_silence = (
        (not has_q and getattr(state, "last_response_len", 0) > 150)
        or (state_mgr.patience(session_id) < 0.20)
    )

    if should_silence:
        mode_label = state.emotion.primary if state.emotion else None
        short = _SILENCE.get(mode_label, _SILENCE.get(None, ["嗯。"]))
        text = _random.choice(short)

        if silence_mode == "完全不回":
            state.last_response_len = 0
            state_mgr._maybe_save()
            return {"text": "", "was_silenced": True}

        state.last_response_len = len(text)
        state_mgr._maybe_save()
        return {"text": text, "was_silenced": True}

    # filler injection (15%)
    if _random.random() < 0.15 and len(response_text) > 25:
        fillers = ["嗯…", "啊", "嘛", "就是说", "反正", "话说", "不过呢", "算了"]
        filler = _random.choice(fillers)
        if not response_text.rstrip().endswith(filler):
            response_text = response_text.rstrip() + filler

    state.last_response_len = len(response_text)
    state_mgr._maybe_save()
    return {"text": response_text, "was_silenced": False}


@mcp.tool()
def persona_status(session_id: str, user_id: str = "default") -> str:
    """
    Get the current persona state as a human-readable status display.

    Args:
        session_id: Session identifier
        user_id: User identifier
    """
    state = state_mgr.get_state(session_id)
    if state is None:
        return "No active persona state for this session."

    from prompt import PromptBuilder
    pb = PromptBuilder(DATA_DIR)
    rel = rel_mgr.get(user_id)
    patience = state_mgr.patience(session_id)
    return pb.build_status(state, patience, rel.stage)


@mcp.tool()
def persona_command(
    session_id: str,
    user_id: str = "default",
    command: str = "status",
    args: str = "",
) -> str:
    """
    Handle a /persona command.

    Args:
        session_id: Session identifier
        user_id: User identifier
        command: Command name (status, random, chill, warm, talkative, quiet,
                 off, on, reset, trait, emotion)
        args: Additional arguments (dim=value for trait, label for emotion)
    """
    state = state_mgr.get_or_init(session_id)

    if command == "random":
        state_mgr.reset(session_id)
        s = state_mgr.get_state(session_id)
        return f"🎲 人格已随机重置！\n外向性: {s.trait.extraversion:.2f}"

    elif command == "chill":
        state_mgr.set_trait(session_id, extraversion=0.2, neuroticism=0.2, agreeableness=0.25)
        state_mgr.set_mood(session_id, valence=0.3, arousal=0.2, dominance=0.4)
        return "🧊 chill 模式 (低外向·低唤醒·偏冷)"

    elif command == "warm":
        state_mgr.set_trait(session_id, extraversion=0.8, agreeableness=0.85, neuroticism=0.2)
        state_mgr.set_mood(session_id, valence=0.8, arousal=0.7, dominance=0.6)
        return "☀️ warm 模式 (高外向·高宜人·偏暖)"

    elif command == "talkative":
        state_mgr.set_trait(session_id, extraversion=0.9, openness=0.85)
        state_mgr.set_mood(session_id, arousal=0.7, valence=0.65)
        return "🗣️ talkative 模式 (高外向·高开放)"

    elif command == "quiet":
        state_mgr.set_trait(session_id, extraversion=0.1, openness=0.2, neuroticism=0.25)
        state_mgr.set_mood(session_id, arousal=0.15, valence=0.4, dominance=0.35)
        return "🤫 quiet 模式 (低外向·低唤醒)"

    elif command == "off":
        state_mgr.set_enabled(session_id, False)
        return "🛑 随机人格已关闭。"

    elif command == "on":
        state_mgr.set_enabled(session_id, True)
        return "✅ 随机人格已开启！"

    elif command == "reset":
        state_mgr.reset(session_id)
        rel_mgr.reset(user_id)
        return "🔄 人格 + 关系已重置。"

    elif command == "trait":
        parts = args.strip().split()
        if len(parts) < 2:
            return "用法: trait <维度> <0.0-1.0>\n维度: openness/conscientiousness/extraversion/agreeableness/neuroticism"
        valid = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        if parts[0] not in valid:
            return f"❌ 无效维度。可选: {', '.join(sorted(valid))}"
        try:
            v = float(parts[1])
            if not 0.0 <= v <= 1.0:
                raise ValueError
        except ValueError:
            return "❌ 值需在 0.0-1.0 之间"
        state_mgr.set_trait(session_id, **{parts[0]: v})
        return f"✅ {parts[0]} → {v:.2f}"

    elif command == "emotion":
        label = args.strip().lower()
        if label not in EMOTION_LABELS:
            return f"❌ 未知情绪。可选: {', '.join(EMOTION_LABELS)}"
        state_mgr.trigger_emotion(session_id, label, 0.7)
        return f"⚡ 已触发: {label}"

    else:
        # default: status
        return persona_status(session_id, user_id)


# ---- main ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PERSONA_PORT", "4568"))
    host = os.environ.get("PERSONA_HOST", "127.0.0.1")

    print(f"🎭 Random Persona MCP Server v2.1.0")
    print(f"   Listening on {host}:{port}")
    print(f"   Data dir: {DATA_DIR}")

    # FastMCP stdio or SSE
    transport = os.environ.get("MCP_TRANSPORT", "sse")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=host, port=port)
