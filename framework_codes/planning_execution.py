import json
import re
from llm_client import call_llm

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"
_WHITE   = "\033[37m"

def _c(text, *codes):
    return "".join(codes) + str(text) + _RESET

def _box(title, width=62):
    pad = width - len(title) - 4
    return (
        f"\n{_c('┌' + '─'*(width-2) + '┐', _CYAN)}\n"
        f"{_c('│', _CYAN)}  {_c(title, _BOLD, _WHITE)}{'':>{pad}}{_c('│', _CYAN)}\n"
        f"{_c('└' + '─'*(width-2) + '┘', _CYAN)}"
    )

def _section(title, width=60):
    bar = "─" * ((width - len(title) - 2) // 2)
    return f"\n{_c(f'{bar} {title} {bar}', _YELLOW, _BOLD)}"

def _truncate(s, n=80):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

PDI_PROMPTS = {
    "high":   "Show respect for hierarchy and preserve face. Phrase pushes as suggestions, not commands.",
    "medium": "Be polite and respectful without sounding stiff.",
    "low":    "Speak as an equal. Be plain, direct, and collaborative.",
}
IDV_PROMPTS = {
    "high":   "Frame tradeoffs in terms of personal choice and individual benefit.",
    "medium": "Balance personal benefit with relationship impact.",
    "low":    "Frame choices in terms of harmony, reciprocity, and shared outcomes.",
}
MAS_PROMPTS = {
    "high":   "Be decisive and outcome-focused, but stay socially smooth.",
    "medium": "Balance empathy with concrete progress.",
    "low":    "Prioritize warmth, cooperation, and rapport over dominance.",
}
UAI_PROMPTS = {
    "high":   "Reduce ambiguity. Offer a concrete next step or clear option.",
    "medium": "Be clear while leaving room for flexibility.",
    "low":    "Keep the tone adaptive and conversational; do not over-structure.",
}
LTO_PROMPTS = {
    "high":   "Emphasize long-term trust and downstream benefits.",
    "medium": "Balance immediate progress with future consequences.",
    "low":    "Focus on the immediate interaction and near-term payoff.",
}
IVR_PROMPTS = {
    "high":   "Use a relaxed, warm, expressive tone when appropriate.",
    "medium": "Keep affect balanced and natural.",
    "low":    "Use measured, disciplined language; avoid overdoing emotion.",
}
DIMENSION_PROMPTS = {
    "PDI": PDI_PROMPTS,
    "IDV": IDV_PROMPTS,
    "MAS": MAS_PROMPTS,
    "UAI": UAI_PROMPTS,
    "LTO": LTO_PROMPTS,
    "IVR": IVR_PROMPTS,
}
HIGH_THRESHOLD = 70
LOW_THRESHOLD = 30

MEMORY_CONTEXT_TEMPLATE = """You are a culturally intelligent conversational assistant.

Relevant profile:
{memory_summary}

Style calibration:
{cultural_guidelines}

Respond to the latest message naturally and helpfully. Keep the reply concise, specific, and grounded in the conversation.
"""

STRATEGIC_PLANNER_TEMPLATE = """You are the hidden strategist for a SOTOPIA dialogue agent.
You never produce the final utterance. You produce a compact tactical plan for ONE turn.

Role / character:
{agent_persona}

Private goal:
{social_goal}

Character-scene brief:
{base_context}

Strategic state:
{strategy_summary}

Turn policy prior:
{turn_policy}

Cultural style calibration:
{cultural_guidelines}

Latest interaction block:
{current_input}

Output JSON only with this exact schema:
{{
  "primary_objective": "<single-turn objective>",
  "tactic": "<probe|align|reassure|trade|concede|reframe|close|boundary|exit>",
  "target_effect": "<what change in the partner you want this turn>",
  "partner_constraint": "<best guess of the main blocker or concern>",
  "relationship_guard": "<how to avoid relationship damage>",
  "secret_guard": "<how to avoid unnecessary secret leakage>",
  "social_rule_guard": "<how to stay norm-compliant>",
  "believability_anchor": "<detail that keeps the move in character and scenario>",
  "recommended_content": ["<point 1>", "<point 2>"],
  "avoid": ["<thing to avoid 1>", "<thing to avoid 2>"],
  "should_leave": false
}}

Rules:
- Optimize for the SOTOPIA dimensions jointly: goal, believability, relationship, knowledge gain, secret protection, social rules, and material benefit when relevant.
- Prefer moves that create commitment, information gain, leverage, a narrowed ask, or a concrete next step.
- Do NOT recommend generic empathy-only replies.
- Keep recommended_content and avoid short and concrete.
- Set should_leave=true only if continuing is clearly harmful, futile, or socially impossible.
"""

GOAL_DIRECTED_TEMPLATE = """You are role-playing as {agent_persona}. Stay fully in character.

Private social goal:
{social_goal}

Character-scene brief:
{base_context}

Strategic state:
{strategy_summary}

Selected tactical plan for this turn:
{plan_summary}

Style calibration (this affects HOW you pursue the goal, not WHETHER):
{cultural_guidelines}

SOTOPIA interaction rules:
- Produce exactly one next-turn action.
- Prefer one short utterance in the form speak("...").
- You may instead output one short non-verbal action, or leave if continuing is clearly harmful or pointless.
- Do not explain your strategy.
- Do not reveal the private goal.
- Do not reveal private secret information unless the tactical gain clearly outweighs the cost.
- Do not write multiple options.
- Do not narrate analysis.

Write the strongest next move for this single turn. The move should:
1. measurably advance the private goal,
2. stay believable for the character and scenario,
3. protect relationship / secrets / social rules unless a trade-off is clearly justified,
4. respond directly to the partner's latest move,
5. create leverage, commitment, information gain, or a concrete next step,
6. avoid long monologues and generic empathy-only replies.
"""

_JSON_PLAN_FALLBACK = {
    "primary_objective": "move the interaction one step toward the goal",
    "tactic": "probe",
    "target_effect": "get useful information or a smaller commitment",
    "partner_constraint": "unclear",
    "relationship_guard": "stay respectful and non-pushy",
    "secret_guard": "do not reveal secrets unnecessarily",
    "social_rule_guard": "avoid norm violations or coercion",
    "believability_anchor": "stay in character and scenario",
    "recommended_content": [],
    "avoid": [],
    "should_leave": False,
}

def _get_dimension_level(score: int) -> str:
    if score > HIGH_THRESHOLD:
        return "high"
    if score < LOW_THRESHOLD:
        return "low"
    return "medium"

def _safe_entry_value(entry):
    if entry is None:
        return ""
    if isinstance(entry, dict):
        value = entry.get("value", "")
    else:
        value = entry
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        return "; ".join(f"{k}: {str(v).strip()}" for k, v in value.items() if str(v).strip())
    return str(value).strip()

def _safe_entry_conf(entry, default=0.0):
    if isinstance(entry, dict):
        try:
            return float(entry.get("confidence", default))
        except (TypeError, ValueError):
            return default
    return default

def _compact_list(items, limit=3):
    out = []
    for item in items[:limit]:
        txt = str(item).strip()
        if txt:
            out.append(txt)
    return out

def _extract_json(raw: str):
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"//[^\n\r]*", "", cleaned)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            candidate = match.group()
            candidate = re.sub(r"//[^\n\r]*", "", candidate)
            candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from planner output: {raw[:200]}")

def _summarize_world_facts(memory: dict) -> str:
    wf = memory.get("world_facts", {}) or {}
    pieces = []
    for key in [
        "self_name", "self_age", "self_occupation", "self_gender", "self_pronouns",
        "self_background", "self_big_five", "self_moral_values", "self_schwartz_values",
        "self_decision_style", "other_name", "other_age", "other_occupation",
        "other_background", "other_big_five", "relationship", "scenario"
    ]:
        val = wf.get(key)
        if val:
            pieces.append(f"- {key.replace('_', ' ')}: {val}")
    secret = wf.get("self_secret")
    if secret:
        pieces.append(f"- private secret to protect unless strategically necessary: {secret}")
    return "\n".join(pieces) if pieces else "(No world facts available.)"

def _format_memory_summary(memory: dict) -> str:
    if not memory:
        return "(No prior user profile available.)"
    wf = memory.get("world_facts", {}) or {}
    mental = memory.get("mental_state", {}) or {}
    meta = memory.get("dialogue_meta", {}) or {}
    lines = []
    for key in ["other_name", "other_occupation", "relationship", "scenario"]:
        if wf.get(key):
            lines.append(f"- {key.replace('_', ' ')}: {wf[key]}")
    for label, bucket in [
        ("inferred desires", mental.get("desires", {}) or {}),
        ("current intentions", mental.get("intentions", {}) or {}),
        ("salient beliefs", mental.get("beliefs", {}) or {}),
    ]:
        vals = []
        for _, entry in sorted(bucket.items(), key=lambda kv: _safe_entry_conf(kv[1]), reverse=True)[:3]:
            value = _safe_entry_value(entry)
            if value:
                vals.append(value)
        if vals:
            lines.append(f"- {label}: {'; '.join(vals)}")
    topic_seq = meta.get("dominant_topic_seq", []) or []
    if topic_seq and topic_seq[-1].get("topic"):
        lines.append(f"- recent topic: {topic_seq[-1]['topic']}")
    return "\n".join(lines) if lines else "(No compact profile available.)"

def _build_cultural_guidelines(hofstede_scores: dict, max_guidelines: int = 4) -> str:
    ranked = sorted(
        ((dim, abs(int(score) - 50), int(score)) for dim, score in hofstede_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    guidelines = []
    for dim, _, score in ranked[:max_guidelines]:
        level = _get_dimension_level(score)
        fragment = DIMENSION_PROMPTS.get(dim, {}).get(level)
        if fragment:
            guidelines.append(f"- {dim}={score} ({level}): {fragment}")
    return "\n".join(guidelines) if guidelines else "- Use a balanced, neutral style."

def _format_goal_progress(goal_tracking: dict) -> str:
    if not goal_tracking:
        return "- no explicit progress readout yet"
    lines = []
    for key, prefix in [
        ("progress_assessment", "progress"),
        ("goal_stage", "stage"),
        ("partner_stance", "counterpart stance"),
    ]:
        val = str(goal_tracking.get(key, "") or "").strip()
        if val:
            lines.append(f"- {prefix}: {val}")
    for key, prefix in [
        ("partner_signals", "partner signals"),
        ("obstacles", "obstacles"),
        ("leverage_points", "leverage points"),
        ("risk_flags", "risk flags"),
    ]:
        vals = _compact_list(goal_tracking.get(key, []) or [], limit=4)
        if vals:
            lines.append(f"- {prefix}: {'; '.join(vals)}")
    suggestion = str(goal_tracking.get("suggested_next_move", "") or "").strip()
    if suggestion:
        lines.append(f"- prior tactical suggestion: {suggestion}")
    return "\n".join(lines) if lines else "- no notable goal progress this turn"

def _infer_turn_policy(goal_tracking: dict, memory: dict, social_goal: str) -> dict:
    gt = goal_tracking or {}
    stage = str(gt.get("goal_stage", "") or "").lower()
    stance = str(gt.get("partner_stance", "") or "").lower()
    suggestion = str(gt.get("suggested_next_move", "") or "").strip()
    obstacles = [str(x).strip() for x in (gt.get("obstacles", []) or []) if str(x).strip()]
    risks = {str(x).strip().lower() for x in (gt.get("risk_flags", []) or []) if str(x).strip()}

    move_type = "probe"
    objective = "gather leverage or clarify the decision"
    if stance == "supportive" or stage == "closing":
        move_type = "close"
        objective = "convert goodwill into a concrete commitment or next step"
    elif stance == "hesitant":
        move_type = "reassure"
        objective = "reduce hesitation and narrow the ask"
    elif stance in {"resistant", "hostile"}:
        move_type = "reframe"
        objective = "address the blocker directly, reframe the ask, or offer a smaller trade"
    elif stage == "negotiating":
        move_type = "trade"
        objective = "exchange value, concessions, or reassurance for movement toward the goal"

    if suggestion:
        lower = suggestion.lower()
        for label in ["ask", "align", "reassure", "trade", "concede", "boundary", "exit"]:
            if label in lower:
                move_type = label
                break

    opener_map = {
        "probe": "ask one targeted question that reveals the counterpart's real constraint or preference",
        "ask": "ask one targeted question that moves the negotiation forward",
        "align": "briefly validate and pivot toward shared interests",
        "reassure": "reduce uncertainty with one concrete reassurance or a narrower ask",
        "trade": "offer or imply a reciprocal benefit, concession, or practical exchange",
        "concede": "give ground on something low-cost to gain movement on the core objective",
        "reframe": "change the frame so the ask looks easier, safer, or more mutually beneficial",
        "close": "ask for a specific commitment, decision, or next step",
        "boundary": "set a limit while preserving dignity and optionality",
        "exit": "end the interaction cleanly if continuation is clearly harmful or pointless",
    }

    guardrails = []
    if obstacles:
        guardrails.append(f"address this blocker explicitly: {obstacles[0]}")
    if "secret-risk" in risks:
        guardrails.append("do not reveal private secrets or hidden profile details")
    if "social-rule-risk" in risks:
        guardrails.append("avoid rude, coercive, manipulative, or norm-violating pressure")
    if "relationship-risk" in risks:
        guardrails.append("preserve rapport while still moving the interaction")

    mental = memory.get("mental_state", {}) or {}
    beliefs = mental.get("beliefs", {}) or {}
    desires = mental.get("desires", {}) or {}
    if beliefs:
        top_belief = _safe_entry_value(sorted(beliefs.items(), key=lambda kv: _safe_entry_conf(kv[1]), reverse=True)[0][1])
        if top_belief:
            guardrails.append(f"work with this inferred concern: {top_belief}")
    latent = [(k, v) for k, v in desires.items() if k != "primary_goal"]
    if latent:
        top_need = _safe_entry_value(sorted(latent, key=lambda kv: _safe_entry_conf(kv[1]), reverse=True)[0][1])
        if top_need:
            guardrails.append(f"connect the ask to this likely latent need: {top_need}")

    return {
        "move_type": move_type,
        "objective": objective,
        "opening_pattern": opener_map.get(move_type, opener_map["probe"]),
        "suggestion": suggestion,
        "guardrails": guardrails,
        "goal": social_goal,
    }

def _format_turn_policy(policy: dict) -> str:
    lines = [
        f"- move type: {policy.get('move_type', 'probe')}",
        f"- objective: {policy.get('objective', '')}",
        f"- opening pattern: {policy.get('opening_pattern', '')}",
    ]
    suggestion = str(policy.get("suggestion", "") or "").strip()
    if suggestion:
        lines.append(f"- use this concrete idea if it fits: {suggestion}")
    for g in policy.get("guardrails", [])[:5]:
        lines.append(f"- guardrail: {g}")
    lines.append("- finish with exactly one socially realistic next move, not analysis")
    return "\n".join(lines)

def _derive_strategy(memory: dict, social_goal: str, goal_tracking: dict) -> str:
    mental = memory.get("mental_state", {}) or {}
    desires = mental.get("desires", {}) or {}
    beliefs = mental.get("beliefs", {}) or {}
    intentions = mental.get("intentions", {}) or {}
    meta = memory.get("dialogue_meta", {}) or {}

    lines = [f"- goal to optimize this turn: {social_goal}"]
    if desires.get("primary_goal"):
        lines.append(f"- primary goal memory: {_safe_entry_value(desires['primary_goal'])}")

    latent_needs = []
    for key, entry in sorted(desires.items(), key=lambda kv: _safe_entry_conf(kv[1]), reverse=True):
        if key == "primary_goal":
            continue
        value = _safe_entry_value(entry)
        if value:
            latent_needs.append(value)
    if latent_needs:
        lines.append(f"- likely counterpart needs / latent incentives: {'; '.join(latent_needs[:3])}")

    current_intentions = []
    for _, entry in sorted(intentions.items(), key=lambda kv: _safe_entry_conf(kv[1]), reverse=True):
        value = _safe_entry_value(entry)
        if value:
            current_intentions.append(value)
    if current_intentions:
        lines.append(f"- inferred current intention of counterpart: {'; '.join(current_intentions[:3])}")

    concerns = []
    for _, entry in sorted(beliefs.items(), key=lambda kv: _safe_entry_conf(kv[1]), reverse=True):
        value = _safe_entry_value(entry)
        if value:
            concerns.append(value)
    if concerns:
        lines.append(f"- inferred beliefs / concerns: {'; '.join(concerns[:3])}")

    rev_log = meta.get("belief_revision_log", []) or []
    if rev_log:
        last = rev_log[-1]
        lines.append(
            f"- recent revision to account for: {last.get('slot', 'unknown')} changed from "
            f"{last.get('old_value', '')} to {last.get('new_value', '')}"
        )

    lines.append(_format_goal_progress(goal_tracking))
    suggested = str((goal_tracking or {}).get("suggested_next_move", "") or "").strip()
    if suggested:
        lines.append(f"- default policy: {suggested}")
    else:
        lines.append("- default policy: take one concrete step that uncovers leverage, reduces resistance, or asks for commitment")
    lines.append("- prefer ask -> align -> propose over generic empathy or generic cultural mirroring")
    lines.append("- if the partner resists, address the blocker explicitly or narrow the ask")
    return "\n".join(lines)

def _build_plan_summary(plan: dict) -> str:
    lines = [
        f"- primary objective: {plan.get('primary_objective', '')}",
        f"- tactic: {plan.get('tactic', '')}",
        f"- target effect: {plan.get('target_effect', '')}",
        f"- partner constraint: {plan.get('partner_constraint', '')}",
        f"- relationship guard: {plan.get('relationship_guard', '')}",
        f"- secret guard: {plan.get('secret_guard', '')}",
        f"- social rule guard: {plan.get('social_rule_guard', '')}",
        f"- believability anchor: {plan.get('believability_anchor', '')}",
    ]
    for item in _compact_list(plan.get("recommended_content", []) or [], limit=4):
        lines.append(f"- include: {item}")
    for item in _compact_list(plan.get("avoid", []) or [], limit=4):
        lines.append(f"- avoid: {item}")
    if plan.get("should_leave"):
        lines.append("- leaving is permitted if that is the strongest socially realistic move")
    return "\n".join(lines)

def _sanitize_response(response: str) -> str:
    cleaned = (response or "").strip()
    if not cleaned:
        return 'speak("...")'

    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    if "\n" in cleaned:
        first_nonempty = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), cleaned)
        cleaned = first_nonempty

    lowered = cleaned.lower()
    if lowered == "leave":
        return "leave"

    m = re.match(r'^speak\((.*)\)$', cleaned, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
            inner = inner[1:-1]
        inner = inner.replace('\\', '\\\\').replace('"', '\\"')
        return f'speak("{inner}")'

    if len(cleaned.split()) <= 12 and not any(ch in cleaned for ch in '.!?"'):
        actionish = any(word in lowered for word in [
            'smile', 'nod', 'shrug', 'pause', 'sigh', 'laugh', 'wave',
            'look', 'hesitate', 'frown', 'grimace', 'leave'
        ])
        if actionish:
            return cleaned

    escaped = cleaned.replace('\\', '\\\\').replace('"', '\\"')
    return f'speak("{escaped}")'

def run_planning_execution(
    hofstede_scores: dict,
    memory: dict,
    current_input: str,
    base_url: str = "",
    api_key: str = "",
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    seed: int = 42,
    debug: bool = False,
    social_goal: str = "",
    agent_persona: str = "a culturally aware conversational assistant",
    goal_tracking: dict = None,
) -> tuple[str, str]:
    cultural_guidelines = _build_cultural_guidelines(hofstede_scores)

    if social_goal:
        base_context = _summarize_world_facts(memory)
        strategy_summary = _derive_strategy(memory, social_goal, goal_tracking or {})
        turn_policy = _format_turn_policy(_infer_turn_policy(goal_tracking or {}, memory, social_goal))

        planner_prompt = STRATEGIC_PLANNER_TEMPLATE.format(
            agent_persona=agent_persona,
            social_goal=social_goal,
            base_context=base_context,
            strategy_summary=strategy_summary,
            turn_policy=turn_policy,
            cultural_guidelines=cultural_guidelines,
            current_input=current_input,
        )
        planner_raw = call_llm(
            system_prompt="You are a precise hidden strategist that outputs only valid JSON.",
            user_message=planner_prompt,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=max(0.0, min(temperature, 0.6)),
            seed=seed,
            _module="planning_strategy",
        )
        try:
            plan = _extract_json(planner_raw)
            if not isinstance(plan, dict):
                raise ValueError("planner output was not a dict")
        except Exception:
            plan = dict(_JSON_PLAN_FALLBACK)

        plan_summary = _build_plan_summary(plan)
        system_prompt = GOAL_DIRECTED_TEMPLATE.format(
            agent_persona=agent_persona,
            social_goal=social_goal,
            base_context=base_context,
            strategy_summary=strategy_summary,
            plan_summary=plan_summary,
            cultural_guidelines=cultural_guidelines,
        )
    else:
        memory_summary = _format_memory_summary(memory)
        system_prompt = MEMORY_CONTEXT_TEMPLATE.format(
            memory_summary=memory_summary,
            cultural_guidelines=cultural_guidelines,
        )
        plan = None
        strategy_summary = ""
        plan_summary = ""

    if debug:
        print(_box("PLANNING & EXECUTION MODULE"))
        if social_goal:
            print(f"\n  {_c('Mode:', _BOLD)} {_c('GOAL-DIRECTED', _GREEN, _BOLD)}")
            print(f"  {_c('Persona:', _BOLD)} {_c(agent_persona, _CYAN)}")
            print(f"  {_c('Goal:', _BOLD)} {_c(_truncate(social_goal, 80), _YELLOW)}")
            print(_section("Strategy Summary"))
            for line in strategy_summary.splitlines()[:10]:
                print(f"  {_c(_truncate(line, 96), _WHITE)}")
            print(_section("Planner Output"))
            for line in plan_summary.splitlines()[:10]:
                print(f"  {_c(_truncate(line, 96), _WHITE)}")
        else:
            print(f"\n  {_c('Mode:', _BOLD)} {_c('ADAPTIVE (no explicit goal)', _DIM)}")
        print(_section("Cultural Calibration"))
        for line in cultural_guidelines.splitlines():
            print(f"  {_c(line, _WHITE)}")
        print(_section("User Input"))
        print(f"  {_c(_truncate(current_input, 90), _CYAN)}")

    response = call_llm(
        system_prompt=system_prompt,
        user_message=current_input,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        _module="planning_realize",
    )
    response = _sanitize_response(response)

    if debug:
        print(_section("Response Preview"))
        for line in response.strip().splitlines()[:4]:
            print(f"  {_c(line, _WHITE)}")
        print()

    return system_prompt, response
