"""Prompt status display — used by persona_status MCP tool."""

from relationship import STAGE_STRANGER


def _bar(value: float) -> str:
    filled = max(0, min(10, int(value * 10)))
    return "█" * filled + "░" * (10 - filled)


def _energy_desc(v: float) -> str:
    if v < 0.25:  return "极低能量"
    if v < 0.45:  return "偏低能量"
    if v < 0.65:  return "中等能量"
    if v < 0.85:  return "偏高能量"
    return "高能量"


def _valence_desc(v: float) -> str:
    if v < 0.25:  return "冷淡疏离"
    if v < 0.45:  return "偏冷理性"
    if v < 0.65:  return "中性平和"
    if v < 0.85:  return "偏暖亲切"
    return "热情亲密"


_OCEAN_LABELS = {"openness":"开放性","conscientiousness":"尽责性","extraversion":"外向性","agreeableness":"宜人性","neuroticism":"神经质"}
_OCEAN_HIGH = {"openness":"好奇·审美","conscientiousness":"自律·条理","extraversion":"社交·活跃","agreeableness":"合作·共情","neuroticism":"敏感·易波动"}
_OCEAN_LOW = {"openness":"务实·惯例","conscientiousness":"随性·灵活","extraversion":"内向·安静","agreeableness":"独立·直言","neuroticism":"稳定·抗压"}


class PromptBuilder:
    def __init__(self, data_dir: str): pass

    def build_status(self, state, patience, relationship_stage="stranger", regulation=None):
        t, m, em = state.trait, state.mood, state.emotion
        lines = ["🎭 Trait (OCEAN基线)"]
        for key in ("extraversion","agreeableness","openness","conscientiousness","neuroticism"):
            val = getattr(t, key, 0.5)
            desc = _OCEAN_HIGH[key] if val > 0.5 else _OCEAN_LOW[key]
            lines.append(f"  {_OCEAN_LABELS[key]}: {_bar(val)} {val:.2f}  {desc}")
        lines += ["", "🌊 Mood (当前心境)"]
        lines.append(f"  愉悦: {_bar(m.valence)} {m.valence:.2f}  ({_valence_desc(m.valence)})")
        lines.append(f"  唤醒: {_bar(m.arousal)} {m.arousal:.2f}  ({_energy_desc(m.arousal)})")
        lines.append(f"  支配: {_bar(m.dominance)} {m.dominance:.2f}")
        lines += ["", f"📊 耐心: {_bar(patience)} {patience:.2f}"]
        if em and getattr(em, 'is_active', False):
            lines += ["", f"⚡ 情绪: {em.primary} (强度: {em.intensity:.2f})"]
        if relationship_stage != STAGE_STRANGER:
            lines += ["", f"👥 关系: {relationship_stage}"]
        return "\n".join(lines)
