# 🎭 随机人格 v2 (Random Persona v2) — 项目白皮书

> AI 回复风格动态引擎 · 六层认知-情感-社交架构  
> 版本 2.1.0 · 2026-06-18  
> 作者：Tsukigakure

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 问题定义](#2-问题定义)
- [3. 架构设计](#3-架构设计)
- [4. 代码结构](#4-代码结构)
- [5. 模块详解](#5-模块详解)
- [6. 数据流](#6-数据流)
- [7. 操作手册](#7-操作手册)
- [8. 命令参考](#8-命令参考)
- [9. AstrBot 集成](#9-astrbot-集成)
- [10. 词库体系](#10-词库体系)
- [11. 理论依据](#11-理论依据)
- [12. v1 vs v2 对比](#12-v1-vs-v2-对比)
- [13. 架构演进路线图](#13-架构演进路线图)

---

## 1. 项目概述

### 1.1 是什么

**随机人格 v2**（Random Persona）是一个 AI 对话风格动态引擎。它在 LLM 的 system prompt 中注入精炼的行为指令，使 AI 的回复风格随内部情绪状态自然波动，模拟人类的**情感变化、表达多样性和社交适应性**。

### 1.2 核心价值

- **零额外 LLM token 成本**：所有检测、决策、映射均为纯规则引擎，不调用任何外部 API 或 LLM
- **数据驱动**：基于 ~600 词手标 PAD 词库，用真实语言学数据替代模糊的"语气温暖"描述
- **MCP 架构**：服务端独立进程，通过 Model Context Protocol 暴露工具，与 AstrBot 解耦
- **全链路可观测**：从用户消息到最终 prompt 注入，每一步都可追踪

### 1.3 项目组成

```
random-persona-mcp/                  ← MCP 服务端（独立进程，核心逻辑）
astrbot_plugin_random_persona/       ← AstrBot 插件（HTTP 客户端，薄封装）
astrbot_plugin_user_persona_bind/    ← 可选：用户人格绑定插件
```

| 组件 | 语言 | 端口 | 职责 |
|------|------|------|------|
| MCP Server | Python 3 | 4568 (MCP) | 状态管理、事件检测、prompt 生成 |
| HTTP API | Python 3 | 4569 | REST 接口供 AstrBot 调用 |
| AstrBot Plugin | Python 3 | — | Hook 注入、命令路由 |
| User Bind Plugin | Python 3 | — | 用户级人格绑定 |

---

## 2. 问题定义

### 2.1 AI 聊天机器人的三个"非人"特征

在长期对话中，标准 LLM 聊天机器人会表现出三种典型非人特征：

1. **永远正面积极** — 不会低落、不会敷衍、不会不耐烦
2. **永远有回应** — 哪怕不需要回复也要硬找话题
3. **永远一种腔调** — 缺少人类自然切换表达方式的多样性

### 2.2 根因分析

LLM 的默认行为是**最大化信息量和积极性**——这是 RLHF 对齐的副作用。人类对话的自然状态是**有起伏、有沉默、有敷衍、有情绪波动**。缺少这些特性的对话在短期内高效，在长期内令人疲惫。

### 2.3 解决思路

不改变 LLM 本身，而是在 system prompt 中注入**动态变化的、参数化的行为指令**，让 LLM 在保持本质能力的同时，表现出自然的人类对话波动。

核心设计原则：

- **不伪装成有意识**：所有"情绪"都是模拟，明确告知用户
- **不降低可用性**：在"自然波动"和"保持帮助性"之间平衡
- **可观测、可调节**：用户可以随时查看、修改、重置 AI 的状态

---

## 3. 架构设计

### 3.1 六层管道

```
RELATIONSHIP    关系阶段 · 自我表露 · 互惠平衡
      ↓
   TRAITS       OCEAN 基线人格（跨会话持久化）
      ↓
    MOOD        心境（小时级漂移，OU 均值回复）
      ↓
  EMOTION       情绪（事件触发，指数衰减）
      ↓
 SPEECH ACT     话语行为选择（17 种）
      ↓
  LANGUAGE      语言特征参数化 + 词库查询
      ↓
   PROMPT       注入 ~50 token 行为指令
```

### 3.2 各层职责

| 层 | 时长 | 触发 | 核心算法 |
|----|------|------|----------|
| **Relationship** | 跨会话（天-月） | 互动累积 | 社会渗透理论四阶段升级 |
| **Trait** | 跨会话稳定 | 会话初始化随机 | Big Five OCEAN → 推导 baseline |
| **Mood** | 小时级 | 自动漂移 + 情绪累积 | Ornstein-Uhlenbeck 均值回复 |
| **Emotion** | 分钟级 | Appraisal 检测 | 正则匹配 → PAD 映射 → 指数衰减 |
| **Speech Act** | 单轮 | 情绪 + 状态驱动 | 加权随机从候选篮选择 |
| **Language** | 单轮 | 全状态映射 | PAD 词库查询 + 16 参数编译 |

### 3.3 关键算法

#### Mood: Ornstein-Uhlenbeck 漂移

```
dX = θ(μ - X)dt + σ√dt · ε    ε ∼ N(0,1)
```

- θ = 0.12（回归强度/小时）：无干扰时约 8-12 小时回归基线
- σ = 0.025（波动率/小时）：模拟日常心境的自然随机波动
- μ = Trait.mood_baseline：由 OCEAN 人格推导的回归目标

#### Emotion: 指数衰减

```
intensity(t) = intensity₀ × 2^(-t / half_life)
```

不同情绪半衰期：

| 情绪 | 半衰期 | 说明 |
|------|--------|------|
| joy | 3 min | 来得快去得快 |
| surprise | 1 min | 转瞬即逝 |
| fear | 5 min | 中等 |
| anger | 10 min | 消退慢 |
| sadness | 15 min | 持久 |
| guilt | 15 min | 与 sadness 同级 |

#### Speech Act: 篮式选择

情绪 + 调节策略 → 候选篮 → Trait 调制权重 → Relationship 过滤限制 → 加权随机选择

17 种 Speech Act 涵盖了从 `MINIMAL_ACK`（"嗯"）到 `ELABORATE_ANSWER`（展开详述）的所有人类对话模式。

---

## 4. 代码结构

### 4.1 MCP 服务端 (`random-persona-mcp/`)

```
random-persona-mcp/
├── server.py            # MCP Server 入口 (FastMCP SSE/stdio)
├── api_server.py        # HTTP API (AstrBot 调用)
├── state.py             # 三层状态模型 + OU漂移 + 衰减
├── appraisal.py         # 评价引擎 (40+ 正则规则)
├── speech_act.py        # 话语行为选择器 (17 种 + 篮式决策)
├── language.py          # 语言特征映射 + Prompt 编译
├── relationship.py      # 人际关系模型 (4 阶段)
├── prompt.py            # 状态显示格式
├── lexicon/
│   ├── __init__.py      # 情绪词库 (PAD 查询引擎)
│   ├── build_lexicon.py # DUTIR 词库构建脚本
│   └── curated_lexicon.json  # ~600 词手标 PAD 词库
├── data/
│   ├── states_v2.json   # 会话状态持久化
│   └── relationships.json  # 用户关系持久化
└── README.md
```

**代码量统计：**

| 文件 | 行数 | 职责 |
|------|------|------|
| `state.py` | ~420 | 状态模型 + 管理 + 持久化 + 数学 |
| `appraisal.py` | ~190 | 情绪检测规则引擎 |
| `speech_act.py` | ~200 | 话语行为决策 |
| `language.py` | ~210 | 语言特征映射 + prompt 编译 |
| `relationship.py` | ~210 | 人际关系模型 |
| `lexicon/__init__.py` | ~240 | 词库查询 (含 ~170 内嵌词) |
| `server.py` | ~185 | MCP 工具定义 |
| `api_server.py` | ~160 | HTTP API |
| `prompt.py` | ~60 | 状态显示 |
| **总计** | **~1,875** | |

### 4.2 AstrBot 插件 (`astrbot_plugin_random_persona/`)

```
astrbot_plugin_random_persona/
├── __init__.py          # 插件注册 (1 行)
├── main.py              # Hook + 命令 (~120 行)
├── metadata.yaml        # 插件元数据
├── _conf_schema.json    # 配置 schema
└── README.md
```

**设计原则：薄客户端。** AstrBot 插件不包含任何业务逻辑，仅负责：
1. `on_llm_request` 调用 API `/api/inject`，将返回注入 system prompt
2. `on_llm_response` 做轻量语气词注入
3. `/persona *` 命令转发到 API `/api/command`

### 4.3 用户人格绑定插件 (`astrbot_plugin_user_persona_bind/`)

```
astrbot_plugin_user_persona_bind/
├── __init__.py
├── main.py              # (~250 行)
├── metadata.yaml
└── README.md
```

功能：让不同用户绑定不同人格，跨群聊生效。使用 AstrBot 原生 `session_service_config` 机制。

---

## 5. 模块详解

### 5.1 state.py — 三层状态模型

**数据结构：**

```python
Trait            # OCEAN 五因素 + 推导调节风格
  .openness, .conscientiousness, .extraversion
  .agreeableness, .neuroticism
  .reappraisal, .suppression, .rumination  # 推导值
  .mood_baseline → {valence, arousal, dominance}

Mood             # 当前心境
  .valence, .arousal, .dominance
  .updated_at

Emotion          # 活跃情绪
  .primary       # 11 种离散标签
  .intensity     # 当前强度 (0-1)
  .half_life     # 半衰期 (秒)
  .v, .a, .d     # PAD 偏移
```

**StateManager 核心方法：**

| 方法 | 功能 |
|------|------|
| `get_or_init(sid)` | 获取或随机初始化会话状态 |
| `drift_mood(sid)` | 心境 OU 漂移 |
| `trigger_emotion(sid, label, intensity)` | 触发情绪 |
| `decay_emotion(sid)` | 情绪指数衰减 + 心境吸收 |
| `patience(sid)` | 计算属性：当前耐心值 |
| `silence_threshold(sid)` | 计算属性：沉默触发阈值 |

**数据持久化：**

- 存储路径：`data/states_v2.json`
- 每 10 次交互自动保存
- 支持 v1 → v2 自动迁移（`states.json` → `states_v2.json`）
- 迁移后旧文件备份为 `.v1.bak`

**会话生命周期：**

- 新会话 → `Trait.random()` 随机初始化
- 超过 30 分钟无交互 → 30% 概率重新随机化
- 用户请求时自动漂移心境 + 衰减情绪

### 5.2 appraisal.py — 评价引擎

**零 LLM token 成本的纯规则检测。**

40+ 条正则规则覆盖 11 种离散情绪：

| 情绪 | 规则数 | 典型触发词 |
|------|--------|-----------|
| anger | 5 | 烦死了、凭什么、滚、你是不是有病、操 |
| joy | 4 | 哈哈哈、开心、太棒了、终于 |
| sadness | 4 | 难过、对不起、失败了、想死 |
| fear | 3 | 怎么办、万一、紧张 |
| surprise | 2 | 卧槽、真的假的 |
| disgust | 2 | 恶心、下头 |
| gratitude | 2 | 谢谢、还好有你 |
| trust | 1 | 信得过你 |
| anticipation | 2 | 期待、然后呢 |
| guilt | 1 | 我对不起 |
| hurt | 2 | 你怎么这样、别理我 |

**情绪调节策略：**

检测到情绪后，根据 Trait + Mood 加权选择调节策略：

```
anger → suppression / reappraisal / controlled_expression
joy   → amplify / share / moderate
sadness → acceptance / rumination / distraction / reappraisal
fear  → seek_reassurance / avoidance / reappraisal
...
```

策略影响下游的 Speech Act 和 Language 映射。

### 5.3 speech_act.py — 话语行为选择

**17 种 Speech Act：**

| 类别 | Speech Act | 典型表现 |
|------|-----------|---------|
| 回应 | MINIMAL_ACK | "嗯"、"好" |
| 回应 | BRIEF_ANSWER | 一两句话 |
| 回应 | ELABORATE_ANSWER | 详述展开 |
| 社交 | SELF_DISCLOSE | 自我表露 |
| 社交 | EMPATHIZE | 共情 |
| 社交 | COMPLIMENT | 称赞 |
| 社交 | TEASE | 调侃 |
| 话题 | EXTEND_TOPIC | 延伸话题 |
| 话题 | SHIFT_TOPIC | 转移话题 |
| 话题 | CLOSE_TOPIC | 结束话题 |
| 对抗 | QUESTION_BACK | 反问 |
| 对抗 | DISAGREE | 表达异议 |
| 对抗 | DEFLECT | 回避搪塞 |
| 对抗 | APOLOGIZE | 道歉 |
| 元对话 | META_COMMENT | "这个问题挺有意思" |
| 元对话 | SEEK_CLARIFICATION | 请求澄清 |

**决策流程：**

```
emotion_label + regulation → basket_key
    → 从预设篮获取候选 + 基础权重
    → Trait 调制 (外向性/开放性)
    → 用户是否有问题 (无问题则削弱 elaborate)
    → Relationship 过滤 (陌生人禁 tease/自我表露)
    → 加权随机选择
```

### 5.4 language.py — 语言特征映射

**LanguageProfile（16 个量化参数）：**

```
词汇层: intensifier_rate, hedge_rate, positive_lexicon,
        negative_lexicon, emoji_rate, filler_rate, exclamation_rate

句法层: avg_sentence_length, complexity, ellipsis_rate, question_rate

话语层: response_length, politeness_strategy, turn_initiative,
        self_disclosure_depth, humor_license
```

**映射来源：**

```
Mood.valence  → positive_lexicon ↑ / negative_lexicon ↓
Mood.arousal  → intensifier_rate ↑ / filler_rate ↓
Mood.dominance → complexity ↑ / hedge_rate ↓
Trait.extraversion → turn_initiative / disclosure_depth
Trait.openness → avg_sentence_length / complexity
Emotion.active → 叠加 PAD 偏移
SpeechAct → 覆盖特定参数
Regulation → 整体缩放（如 suppression 将所有强度参数减半）
Relationship.stage → politeness_strategy / humor_license / emoji_rate
```

**Prompt 输出格式（~50 token）：**

```
[PERSONA]
语气: 温暖、有力
长度: 适中
风格: 适当 emoji、短句为主
主动: 可以延伸或追问
倾向用词: 热情、友好、有趣、自然、轻松、舒服、自在、愉快
避免用词: 糟糕、恐怖、绝望、悲伤
[/PERSONA]
```

**关键设计：不暴露内部状态。** Prompt 中不会出现"你现在的愤怒强度为 0.7"之类的内部变量。所有指令都是行为导向的。

### 5.5 relationship.py — 人际关系模型

**四阶段社会渗透模型：**

| 阶段 | 条件 | AI 行为特征 |
|------|------|------------|
| **stranger** | 默认 | 礼貌克制、不主动延伸、禁玩笑 |
| **acquaintance** | ≥20 轮, 正面比≥0.7 | 中性、适度延伸 |
| **friend** | ≥80 轮, 双向表露 | 自然、允许玩笑、可自我表露 |
| **close** | ≥200 轮, 深度表露 | 随意、主动、bald politeness |

**自我表露洋葱模型：**

```
0.0-0.2 表层: 爱好、日常、天气
0.2-0.4 浅层: 观点、偏好、小故事
0.4-0.6 中层: 价值观、目标、困惑
0.6-0.8 深层: 弱点、失败经历、不安
0.8-1.0 核心: 创伤、恐惧、核心信念
```

AI 表露深度 ≤ 用户已表露深度 + 0.15（互惠但不逾越）。

**用户表露深度估算（启发式）：**

- 消息长度 > 50 字符 +0.05，> 400 +0.15
- 检测到亲密词汇（"我小时候"、"压力"、"秘密"等）+0.08/词

### 5.6 lexicon — 情绪词库

**内嵌 ~170 个手标 PAD 词，覆盖所有基本情绪 + 中性维度：**

每种情绪类别有 10-20 个代表词，每个词标注：

```json
{"word": "开心", "v": 0.82, "a": 0.65, "d": 0.70, "category": "joy", "intensity": 0.60}
```

**查询能力：**

| 方法 | 功能 |
|------|------|
| `query_pad(v, a, d)` | 按 PAD 欧氏距离返回最近词 |
| `query_pad_weighted(v, a, d)` | 加权查询（v 权重最高） |
| `query_category(cat)` | 按情绪类别查询 |
| `query_avoid(v, a, d)` | 返回最远词（避免列表） |

**扩展能力：**

`build_lexicon.py` 可从 DUTIR 情感词汇本体（大连理工大学，徐琳宏等）的 ~27,000 词中提取高质量中文情感词，经 jieba 词频过滤后与内嵌种子词合并。合并后的 `curated_lexicon.json` 可达 ~1,600+ 词。

---

## 6. 数据流

### 6.1 完整请求生命周期

```
1. 用户发送消息 "烦死了，又加班"
         │
2. AstrBot on_llm_request hook
         │
3. HTTP POST → /api/inject
         │
4. server.py persona_inject()
   ├── drift_mood(sid)          # 心境 OU 漂移
   ├── appraiser.evaluate()     # 检测 "烦死了" → anger 0.75
   ├── trigger_emotion(sid, anger, 0.75)
   ├── select_regulation(anger) # → "controlled_expression"
   ├── decay_emotion(sid)       # 衰减现有情绪
   ├── rel_mgr.get(uid)         # 获取关系阶段
   ├── should_silence?          # 否（有情绪要回应）
   ├── select_speech_act()      # anger+controlled → "brief_answer"
   ├── map_to_profile()         # 量化语言参数
   ├── lexicon.query_pad()      # 倾向词: ["冷静","克制","理性"...]
   ├── lexicon.query_avoid()    # 避免词: ["开心","兴奋","热情"...]
   ├── _render_prompt()         # 编译 prompt 块
         │
5. 返回 prompt 块注入 system prompt
         │
6. LLM 生成回复
         │
7. AstrBot on_llm_response hook
   ├── 15% 概率注入语气词 ("嗯…"、"嘛"、etc.)
         │
8. 用户收到回复
```

### 6.2 状态持久化

```
states_v2.json:
{
  "_version": 2,
  "sessions": {
    "<session_id>": {
      "trait": { "openness": 0.55, ..., "reappraisal": 0.62, ... },
      "mood": { "v": 0.55, "a": 0.38, "d": 0.50, "ts": 1718700000 },
      "emotion": null or { "p": "anger", "i": 0.65, ... },
      "enabled": true, "msg_count": 23, ...
    }
  }
}

relationships.json:
{
  "<user_id>": {
    "stage": "acquaintance",
    "sd": 0.15, "ud": 0.28,   // AI 和用户表露深度
    "ic": 35,                   // 交互次数
    "pos": 28, "neg": 2,        // 正/负面交互计数
    "first": 1718000000, "last": 1718700000,
    "fm": 0.55                  // 语域匹配度
  }
}
```

---

## 7. 操作手册

### 7.1 环境要求

- Python ≥ 3.11
- `fastmcp`（MCP 框架）
- AstrBot ≥ 4.0.0（如使用 AstrBot 集成）

### 7.2 快速启动

#### 方式一：MCP Server（标准方式）

```bash
cd random-persona-mcp
pip install fastmcp
python server.py
# → MCP Server on http://127.0.0.1:4568 (SSE)
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PERSONA_PORT` | `4568` | MCP Server 端口 |
| `PERSONA_HOST` | `127.0.0.1` | 绑定地址 |
| `PERSONA_DATA_DIR` | `./data` | 状态持久化目录 |
| `MCP_TRANSPORT` | `sse` | 传输方式：`sse` 或 `stdio` |

#### 方式二：HTTP API（AstrBot 使用）

```bash
cd random-persona-mcp
python api_server.py
# → HTTP API on http://0.0.0.0:4569
```

#### 方式三：systemd 生产部署

```ini
# /etc/systemd/system/persona-api.service
[Unit]
Description=Random Persona HTTP API
After=network.target

[Service]
Type=simple
User=admin
WorkingDirectory=/home/admin/.openclaw/workspace/random-persona-mcp
Environment=PERSONA_DATA_DIR=/home/admin/.openclaw/workspace/random-persona-mcp/data
ExecStart=/usr/bin/python3 api_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### 7.3 安装 AstrBot 插件

```bash
# 将插件放入 AstrBot data/plugins/
cp -r astrbot_plugin_random_persona /home/admin/astrbot/data/plugins/
docker restart astrbot
```

配置（WebUI → 插件配置）：

```json
{
  "mcp_url": "http://127.0.0.1:4569",
  "silence_mode": "短回应"
}
```

### 7.4 API 参考

#### POST /api/inject

获取当前状态的 prompt 注入。

```json
// Request
{
  "session_id": "qq_official:webhook:group:123456",
  "user_id": "qq_official:webhook:789012",
  "user_message": "烦死了，又加班"
}

// Response
{
  "ok": true,
  "data": "[PERSONA]\n语气: 克制、有力\n长度: 简洁\n风格: 短句为主\n克制: 回应即可\n倾向用词: 冷静、理性、克制、稳重\n避免用词: 兴奋、热情、欢快\n[/PERSONA]"
}
```

#### POST /api/status

```json
// Request
{ "session_id": "...", "user_id": "..." }

// Response
{
  "ok": true,
  "data": "🎭 Trait (OCEAN基线)\n  外向性: ██████░░░░ 0.60 ...\n\n🌊 Mood (当前心境)\n  愉悦: ████░░░░░░ 0.42 (偏冷理性)\n  ...\n\n📊 耐心: ██░░░░░░░░ 0.22\n⚡ 情绪: anger (强度: 0.65)\n👥 关系: acquaintance"
}
```

#### POST /api/command

```json
// 随机重置
{ "session_id": "...", "command": "random", "args": "" }

// 调节特质
{ "session_id": "...", "command": "trait", "args": "extraversion 0.9" }

// 触发情绪
{ "session_id": "...", "command": "emotion", "args": "joy" }

// 切换模式
{ "session_id": "...", "command": "chill", "args": "" }
```

#### GET /health

```json
{ "ok": true, "data": "ok" }
```

---

## 8. 命令参考

所有命令通过 `/persona` 前缀调用（AstrBot），或通过 MCP `persona_command` 工具：

| 命令 | 参数 | 效果 |
|------|------|------|
| `/persona` | — | 查看完整状态（Trait + Mood + Emotion + 耐心 + 关系） |
| `/persona random` | — | 完全随机重置 OCEAN 人格 + 心境 |
| `/persona chill` | — | 低外向、低唤醒、偏冷 |
| `/persona warm` | — | 高外向、高宜人、偏暖 |
| `/persona talkative` | — | 高外向、高开放、话多 |
| `/persona quiet` | — | 低外向、低唤醒、话少 |
| `/persona trait <维度> <值>` | `extraversion 0.8` | 手动调节 OCEAN 某一维度 |
| `/persona emotion <标签>` | `joy` | 手动触发离散情绪（joy/anger/sadness...） |
| `/persona off` | — | 关闭随机人格（LLM 恢复默认） |
| `/persona on` | — | 开启随机人格 |
| `/persona reset` | — | 重置人格 + 清除关系数据 |

**OCEAN 维度参考值：**

| 维度 | 低值 (0.1-0.3) | 中值 (0.4-0.6) | 高值 (0.7-0.9) |
|------|---------------|---------------|---------------|
| extraversion | 内向安静 | 中性 | 社交活跃 |
| agreeableness | 独立直言 | 合作 | 共情温暖 |
| openness | 务实惯例 | 灵活 | 好奇审美 |
| conscientiousness | 随性灵活 | 条理 | 自律严谨 |
| neuroticism | 稳定抗压 | 正常 | 敏感易波动 |

**可用情绪标签（11 种）：**

`joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `trust`, `anticipation`, `guilt`, `gratitude`, `hurt`

---

## 9. AstrBot 集成

### 9.1 架构

```
AstrBot Core
  │
  ├── on_llm_request (priority=-100)
  │   ├── user_persona_bind 插件: 注入 persona_id
  │   └── random_persona 插件: HTTP → persona API /api/inject
  │       └── 返回 prompt 块 → 追加到 req.system_prompt
  │
  ├── LLM 调用 (system_prompt 含 persona 指令)
  │
  └── on_llm_response
      └── random_persona 插件: 15% 概率注入语气词
```

### 9.2 与 user_persona_bind 的协作

`user_persona_bind` 在 `on_llm_request` 中注入 `persona_id`，`random_persona` 在更低优先级读入该 ID 作为 `user_id`。两者不冲突，可以叠加使用：

- `user_persona_bind`：确定使用哪个 "角色"（persona_id）
- `random_persona`：让该角色的回复风格自然波动

### 9.3 调试

```bash
# 查看 API 状态
curl -X POST http://localhost:4569/health

# 手动测试 inject
curl -X POST http://localhost:4569/api/inject \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"test","user_id":"test","user_message":"哈哈哈好开心"}'

# 查看持久化状态
cat random-persona-mcp/data/states_v2.json | python3 -m json.tool

# 查看关系数据
cat random-persona-mcp/data/relationships.json | python3 -m json.tool
```

---

## 10. 词库体系

### 10.1 数据来源

| 来源 | 词数 | 标注 |
|------|------|------|
| **手标种子词** | ~170 | V/A/D 连续值 + 情绪类别 + 强度 |
| **DUTIR 扩展** | ~1,428 (经 jieba 过滤) | 类别映射 → V/A/D 估值 |
| **合计** | ~1,600 | 覆盖 11 种情绪 + 中性 4 象限 |

### 10.2 DUTIR 处理流程

```
DUTIR CSV (~27,000 词)
  → 过滤: intensity ≥ 5, polarity ≠ 0, length 1-2
  → jieba 词频过滤 (freq ≥ 20)
  → 类别映射 (22 类 → 8 类 + neutral)
  → PAD 赋值 (根据类别 + 强度微调)
  → 去重 (保留最高强度)
  → 按类别上限裁剪 (500/类)
  → curated_lexicon.json
```

### 10.3 PAD 示例查询

```
Mood(v=0.82, a=0.65, d=0.70) → "开心、幸福、美好、热情、欢喜、兴奋、喜悦"
Mood(v=0.12, a=0.82, d=0.72) → "恼怒、气愤、烦躁、恼火、可气"
Mood(v=0.55, a=0.15, d=0.72) → "沉着、稳重、严谨、冷静、理性"
Mood(v=0.15, a=0.15, d=0.15) → "悲哀、消沉、沮丧、寂寞、忧伤"
```

---

## 11. 理论依据

### 11.1 情绪心理学

| 理论 | 来源 | 应用 |
|------|------|------|
| **OCC 评价理论** | Ortony, Clore & Collins (1988) | Appraisal 引擎设计 |
| **PAD 三维模型** | Mehrabian & Russell (1974) | Mood + Emotion 维度定义 |
| **基本情绪理论** | Ekman (1972, 1992) | 11 种离散情绪标签 |
| **情绪调节过程模型** | Gross (1998, 2015) | Regulation 策略选择 |
| **ALMA 三层架构** | Gebhard (2005) | Trait → Mood → Emotion 分层 |
| **EMA 情绪适应** | Marsella & Gratch (2009) | Coping 策略 → Speech Act 映射 |
| **情绪感染理论** | Hatfield et al. (1993) | AI 对用户情绪的共鸣反应 |
| **情感习惯化** | Headey & Wearing (1989) | OU 均值回复 → 基线回归 |

### 11.2 人格心理学

| 理论 | 来源 | 应用 |
|------|------|------|
| **Big Five OCEAN** | McCrae & Costa (2003) | Trait 五维度基线人格 |
| **特质激活理论** | Tett & Guterman (2000) | 不同场景下人格表现不同 |

### 11.3 社会/语用学

| 理论 | 来源 | 应用 |
|------|------|------|
| **社会渗透理论** | Altman & Taylor (1973) | Relationship 四阶段 + 洋葱模型 |
| **面子协商/礼貌理论** | Brown & Levinson (1987) | politeness_strategy 参数 |
| **合作原则** | Grice (1975) | Speech Act 的会话准则约束 |
| **话轮转换** | Sacks, Schegloff & Jefferson (1974) | 沉默权 + MINIMAL_ACK |
| **语言风格匹配** | Ireland et al. (2011) | formality_match 参数 |
| **沟通适应理论** | Giles (1973, 2016) | 趋同/趋异策略 |

---

## 12. v1 vs v2 对比

| 维度 | v1.0 | v2.1 |
|------|------|------|
| **状态层数** | 1 (扁平4维) | 3 + Relationship |
| **人格模型** | 仅 openness | Big Five OCEAN 完整五因素 |
| **漂移方式** | 纯随机游走 | OU 均值回复 + 事件驱动 |
| **情绪触发** | 无 | 40+ 正则规则引擎 |
| **情绪类型** | 无离散标签 | 11 种离散情绪 |
| **情绪衰减** | 无 | 指数衰减 + 心境吸收 |
| **情绪调节** | 无 | 多种策略选择 |
| **话语层** | 5 种模糊表达 | 17 种 Speech Act |
| **语言参数** | 模糊风格描述 | 16 个量化参数 |
| **词库** | 无 | ~1,600 词 PAD 词库 |
| **关系模型** | 无 | 4 阶段社会渗透 |
| **Prompt 注入** | ~180 token | ~50 token |
| **架构** | 单体 AstrBot 插件 | MCP 服务 + 插件客户端 |
| **可观测性** | 仅 `/persona` 状态 | 完整状态 + API + 关系追踪 |
| **代码量** | ~777 行 | ~1,875 行 |

---

## 13. 架构演进路线图

### Phase 1 ✅ 已完成 — 情绪分层
- ALMA 三层模型（Trait / Mood / Emotion）
- OU 均值回复漂移
- 指数衰减 + 心境吸收
- v1 数据自动迁移

### Phase 2 ✅ 已完成 — Appraisal 事件触发
- 40+ 正则规则覆盖 11 种情绪
- 情绪调节策略选择
- 多次同向情绪累积 → 心境偏移

### Phase 3 ✅ 已完成 — 话语行为 + 语言映射
- 17 种 Speech Act + 篮式决策
- 16 参数 LanguageProfile
- PAD 词库查询 → 数据驱动 prompt
- ~50 token 行为指令编译

### Phase 4 ✅ 已完成 — 人际关系模型
- 四阶段社会渗透
- 自我表露互惠 + 洋葱模型
- 关系影响所有下游决策
- 用户表露深度启发式估算

### 未来方向（待定）

- **多轮情绪追踪**：跨轮次的情绪状态一致性
- **用户情绪建模**：对用户建立独立的情绪模型（当前仅建模 AI 自身）
- **多模态情绪**：结合语音 TTS 的情绪控制
- **关系降级**：长期不互动后的关系衰减
- **A/B 测试框架**：量化不同参数组合的用户满意度

---

## 附录 A: 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| PAD | Pleasure-Arousal-Dominance | 三维情绪空间 |
| OCEAN | Big Five | 五因素人格模型 |
| OCC | Ortony-Clore-Collins | 认知评价情绪理论 |
| ALMA | Layered Model of Affect | 三层情感架构 |
| OU | Ornstein-Uhlenbeck | 均值回复随机过程 |
| Speech Act | 话语行为 | 对话中的功能性行为分类 |
| MCP | Model Context Protocol | AI 工具协议 |
| DUTIR | 大连理工大学信息检索研究室 | 情感词汇本体来源 |
| NRC-VAD | NRC Valence-Arousal-Dominance | 英文情感词库 |

## 附录 B: 文件清单

```
workspace/
├── random-persona-mcp/           # 核心服务端
│   ├── server.py                 # MCP Server
│   ├── api_server.py             # HTTP API
│   ├── state.py                  # 状态引擎
│   ├── appraisal.py              # 评价引擎
│   ├── speech_act.py             # 话语行为
│   ├── language.py               # 语言映射
│   ├── relationship.py           # 人际关系
│   ├── prompt.py                 # 状态显示
│   ├── lexicon/                  # 词库模块
│   │   ├── __init__.py
│   │   ├── build_lexicon.py
│   │   └── curated_lexicon.json
│   ├── data/                     # 运行时数据
│   ├── README.md
│   ├── ARCHITECTURE_V2.md        # 架构设计文档
│   ├── IMPLEMENTATION_PLAN.md    # 实施方案
│   ├── RESEARCH_REVIEW.md        # 研究综述
│   └── WHITEPAPER.md             # 本文件
├── astrbot_plugin_random_persona/# AstrBot 插件
│   ├── main.py
│   ├── __init__.py
│   └── README.md
└── astrbot_plugin_user_persona_bind/  # 用户绑定插件
    ├── main.py
    ├── __init__.py
    └── README.md
```

---

*本文档随项目持续更新。最后更新：2026-06-19*
