#!/usr/bin/env python3
"""HTTP API wrapper — simple POST endpoints for the AstrBot plugin.
Runs on port 4569, wraps the same state/logic as the MCP server."""

import json, os, time
from http.server import HTTPServer, BaseHTTPRequestHandler

import state as st
from appraisal import AppraisalEngine
from language import LexiconPromptBuilder
from relationship import RelationshipManager

DATA_DIR = os.environ.get("PERSONA_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
LEX_DIR = os.path.join(os.path.dirname(__file__), "lexicon")

state_mgr = st.StateManager(DATA_DIR)
appraiser = AppraisalEngine()
lex_builder = LexiconPromptBuilder(LEX_DIR)
rel_mgr = RelationshipManager(DATA_DIR)
prompt_builder = None


def get_prompt_builder():
    global prompt_builder
    if prompt_builder is None:
        from prompt import PromptBuilder
        prompt_builder = PromptBuilder(DATA_DIR)
    return prompt_builder


class Handler(BaseHTTPRequestHandler):
    def _ok(self, body: str):
        self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "data": body}, ensure_ascii=False).encode())

    def _err(self, msg: str):
        self.send_response(500); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(json.dumps({"ok": False, "error": msg}).encode())

    def _parse(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        if self.path in ("/", "/dashboard"):
            import os as _os
            html_path = _os.path.join(_os.path.dirname(__file__), "dashboard.html")
            if _os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
                return
        self.send_error(404)

    def do_POST(self):
        try:
            body = self._parse()
            if self.path == "/api/inject":
                sid = body.get("session_id", "default")
                uid = body.get("user_id", "default")
                msg = body.get("user_message", "")

                stt = state_mgr.get_or_init(sid)
                state_mgr.drift_mood(sid)
                result = appraiser.evaluate_with_regulation(msg, stt.trait, stt.mood)
                if result:
                    state_mgr.trigger_emotion(sid, result["label"], result["intensity"])
                state_mgr.decay_emotion(sid)
                rel = rel_mgr.get(uid)
                rel_mgr.record_interaction(uid, sentiment=stt.mood.valence - 0.5, user_msg=msg)
                block = lex_builder.build(
                    mood=stt.mood, trait=stt.trait, emotion=stt.emotion,
                    speech_act="brief_answer", relationship_stage=rel.stage,
                )
                stt.msg_count += 1; stt.last_active = time.time()
                state_mgr._maybe_save()
                self._ok(block)

            elif self.path == "/api/status":
                sid = body.get("session_id", "default")
                uid = body.get("user_id", "default")
                stt = state_mgr.get_state(sid)
                if not stt:
                    return self._err("no state")
                rel = rel_mgr.get(uid)
                patience = state_mgr.patience(sid)
                pb = get_prompt_builder()
                self._ok(pb.build_status(stt, patience, rel.stage))

            elif self.path == "/api/command":
                sid = body.get("session_id", "default")
                uid = body.get("user_id", "default")
                cmd = body.get("command", "status")
                args = body.get("args", "")

                if cmd == "random":
                    state_mgr.reset(sid)
                    s = state_mgr.get_state(sid)
                    self._ok(f"🎲 已重置\n外向性: {s.trait.extraversion:.2f}")
                elif cmd == "chill":
                    state_mgr.set_trait(sid, extraversion=0.2, neuroticism=0.2, agreeableness=0.25)
                    state_mgr.set_mood(sid, valence=0.3, arousal=0.2, dominance=0.4)
                    self._ok("🧊 chill 模式")
                elif cmd == "warm":
                    state_mgr.set_trait(sid, extraversion=0.8, agreeableness=0.85, neuroticism=0.2)
                    state_mgr.set_mood(sid, valence=0.8, arousal=0.7, dominance=0.6)
                    self._ok("☀️ warm 模式")
                elif cmd == "talkative":
                    state_mgr.set_trait(sid, extraversion=0.9, openness=0.85)
                    state_mgr.set_mood(sid, arousal=0.7, valence=0.65)
                    self._ok("🗣️ talkative 模式")
                elif cmd == "quiet":
                    state_mgr.set_trait(sid, extraversion=0.1, openness=0.2, neuroticism=0.25)
                    state_mgr.set_mood(sid, arousal=0.15, valence=0.4, dominance=0.35)
                    self._ok("🤫 quiet 模式")
                elif cmd == "off":
                    state_mgr.set_enabled(sid, False); self._ok("🛑 已关闭")
                elif cmd == "on":
                    state_mgr.set_enabled(sid, True); self._ok("✅ 已开启")
                elif cmd == "reset":
                    state_mgr.reset(sid); rel_mgr.reset(uid); self._ok("🔄 已重置")
                elif cmd == "trait":
                    parts = args.strip().split()
                    valid = {"openness","conscientiousness","extraversion","agreeableness","neuroticism"}
                    if len(parts) >= 2 and parts[0] in valid:
                        state_mgr.set_trait(sid, **{parts[0]: float(parts[1])})
                        self._ok(f"✅ {parts[0]} → {parts[1]}")
                    else:
                        self._err(f"用法: trait 维度 值. 有效维度: {', '.join(sorted(valid))}")
                elif cmd == "emotion":
                    label = args.strip().lower()
                    if label in st.EMOTION_LABELS:
                        state_mgr.trigger_emotion(sid, label, 0.7)
                        self._ok(f"⚡ 已触发: {label}")
                    else:
                        self._err(f"未知情绪: {label}")
                else:  # status
                    self.do_POST.__wrapped__(self)  # fallback

            elif self.path == "/health":
                self._ok("ok")
            else:
                self.send_error(404)
        except Exception as e:
            self._err(str(e))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", "4569"))
    print(f"🎭 Persona HTTP API on :{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
