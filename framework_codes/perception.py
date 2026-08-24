import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from llm_client import call_llm

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"
_MAGENTA = "\033[35m"
_RED     = "\033[31m"
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
    return s if len(s) <= n else s[:n-1] + "…"

_FACTS_SYSTEM = (
    "You are a dialogue analyst. Extract objective facts from dialogue text. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_FACTS_USER_TMPL = """\
Extract objective facts about the latest counterpart move, using recent dialogue only as supporting context.

Output this exact JSON shape (no extra keys):
{{
  "static_attributes": {{}},
  "dynamic_events": []
}}

static_attributes: key-value pairs for stable user attributes (name, age, gender, \
occupation, location, relationships, explicit preferences). Only include attributes \
that are explicitly mentioned. Omit null/unknown fields entirely.

dynamic_events: list of objects like {{"event": "...", "time_reference": "..."}} \
for each event with a time anchor (e.g. "yesterday", "last week", "unspecified").

{memory_hint}

Dialogue:
{dialogue}"""

def _call_facts(dialogue: str, prior_memory: Optional[dict], **llm_kw) -> dict:
    memory_hint = ""
    if prior_memory:
        known = list(prior_memory.get("world_facts", {}).keys())
        if known:
            memory_hint = (
                "Already known attributes (do NOT re-extract unless updated): "
                + ", ".join(known[:10])
            )
    user_msg = _FACTS_USER_TMPL.format(dialogue=dialogue, memory_hint=memory_hint)
    raw = call_llm(system_prompt=_FACTS_SYSTEM, user_message=user_msg,
                   _module="perception_facts", **llm_kw)
    parsed = _extract_json(raw)
    return {
        "static_attributes": parsed.get("static_attributes") or {},
        "dynamic_events":    parsed.get("dynamic_events") or [],
    }

_MENTAL_SYSTEM = (
    "You are a cognitive analyst. Identify the speaker's mental state from dialogue. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_MENTAL_USER_TMPL = """\
Analyze the latest counterpart speaker's mental state. Use recent dialogue only as context and focus on the latest-message section.

Output this exact JSON shape (no extra keys):
{{
  "immediate_intent": "<the speaker's direct goal or purpose in this turn>",
  "emotion": {{
    "category": "<one word label, e.g. anxious, happy, frustrated, neutral>",
    "intensity": "<low | medium | high>"
  }},
  "values_and_obsessions": ["<implicit deep belief or recurring concern>"]
}}

values_and_obsessions: list up to 3 implicit values inferred from the text. \
Return an empty list if none are detectable.

Dialogue:
{dialogue}"""

def _call_mental(dialogue: str, **llm_kw) -> dict:
    user_msg = _MENTAL_USER_TMPL.format(dialogue=dialogue)
    raw = call_llm(system_prompt=_MENTAL_SYSTEM, user_message=user_msg,
                   _module="perception_mental", **llm_kw)
    parsed = _extract_json(raw)
    return {
        "immediate_intent":     parsed.get("immediate_intent", ""),
        "emotion":              parsed.get("emotion") or {"category": "neutral", "intensity": "low"},
        "values_and_obsessions": parsed.get("values_and_obsessions") or [],
    }

_CULTURAL_SYSTEM = (
    "You are a cross-cultural communication expert. Identify cultural signals in dialogue. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_CULTURAL_USER_TMPL = """\
Identify cultural communication patterns expressed by the latest counterpart speaker. Use prior turns only as context.

Output this exact JSON shape (no extra keys):
{{
  "communication_style": "<e.g. direct, indirect, formal, informal, assertive, hedging>",
  "social_tendencies": ["<observable cultural pattern, e.g. collectivism, hierarchy awareness>"]
}}

social_tendencies: list up to 4 cultural behavioral patterns. \
Return an empty list if none are observable.

Dialogue:
{dialogue}"""

def _call_cultural(dialogue: str, **llm_kw) -> dict:
    user_msg = _CULTURAL_USER_TMPL.format(dialogue=dialogue)
    raw = call_llm(system_prompt=_CULTURAL_SYSTEM, user_message=user_msg,
                   _module="perception_cultural", **llm_kw)
    parsed = _extract_json(raw)
    return {
        "communication_style": parsed.get("communication_style", ""),
        "social_tendencies":   parsed.get("social_tendencies") or [],
    }

_GOAL_SYSTEM = (
    "You are a social dynamics analyst tracking conversational goal progress. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_GOAL_USER_TMPL = """\
The agent has an active social goal:
  {social_goal}

Analyze the latest counterpart move, using recent dialogue as context, and assess how this goal is progressing.

Output this exact JSON shape (no extra keys):
{{
  "progress_assessment": "<one sentence on current goal progress>",
  "partner_signals": ["<observable signal from the partner relevant to the goal>"],
  "obstacles": ["<specific blocker, hesitation, or resistance detected>"],
  "leverage_points": ["<useful fact, incentive, commitment, or opening created by the partner>"],
  "suggested_next_move": "<one concrete next move such as ask, align, reassure, trade, concede, boundary, or exit>",
  "goal_stage": "<opening | probing | negotiating | closing | stalled>",
  "partner_stance": "<supportive | neutral | hesitant | resistant | hostile>",
  "risk_flags": ["<secret-risk, social-rule-risk, relationship-risk, or none>"]
}}

Rules:
- Focus on tactical social progress, not generic empathy.
- partner_signals, obstacles, leverage_points, and risk_flags should be short phrases.
- If the partner already gave useful information, suggested_next_move should exploit it.
- If the partner is resistant, suggested_next_move should address the blocker or narrow the ask.

Dialogue:
{dialogue}"""

def _call_goal_tracking(dialogue: str, social_goal: str, **llm_kw) -> dict:
    user_msg = _GOAL_USER_TMPL.format(dialogue=dialogue, social_goal=social_goal)
    raw = call_llm(system_prompt=_GOAL_SYSTEM, user_message=user_msg,
                   _module="perception_goal", **llm_kw)
    parsed = _extract_json(raw)
    return {
        "progress_assessment": parsed.get("progress_assessment", ""),
        "partner_signals":     parsed.get("partner_signals") or [],
        "obstacles":           parsed.get("obstacles") or [],
        "leverage_points":     parsed.get("leverage_points") or [],
        "suggested_next_move": parsed.get("suggested_next_move", ""),
        "goal_stage": parsed.get("goal_stage", ""),
        "partner_stance": parsed.get("partner_stance", ""),
        "risk_flags": parsed.get("risk_flags") or [],
    }

PERCEPTION_SYSTEM_PROMPT_BASE = (
    _FACTS_SYSTEM + "\n" + _MENTAL_SYSTEM + "\n" + _CULTURAL_SYSTEM
)

GOAL_TRACKING_SCHEMA_EXTENSION = _GOAL_USER_TMPL

PERCEPTION_SYSTEM_PROMPT = PERCEPTION_SYSTEM_PROMPT_BASE

def _build_perception_prompt(social_goal: str = "") -> str:
    """Backward-compat shim — not used by the new parallel implementation."""
    return PERCEPTION_SYSTEM_PROMPT_BASE

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from raw LLM output.

    Improvements over the original:
      1. Strips ``// line comments`` before parsing — 7B models sometimes
         reproduce comment annotations from the schema examples.
      2. Removes trailing commas before ``}`` / ``]`` — another common 7B artefact.
      3. Falls back to a greedy regex search for the outermost JSON object when
         full-string parsing fails.
      4. Error message includes a 200-char preview of the raw output.
    """

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

    preview = raw[:200].replace("\n", " ")
    raise ValueError(f"Could not extract valid JSON from LLM output:\n{preview}")

def run_perception(
    current_input: str,
    prior_memory: Optional[dict] = None,
    base_url: str = "",
    api_key: str = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.2,
    seed: int = 42,
    debug: bool = False,
    max_retries: int = 2,
    social_goal: str = "",
) -> dict:
    """
    Perceive and extract structured information from the current dialogue input.

    Parallel sub-task design
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Instead of one large multi-section prompt, this function dispatches 3 (or 4)
    focused sub-tasks concurrently:

      A · objective_facts  — static attributes + dated events
      B · mental_state     — intent, emotion, values
      C · cultural_cues    — communication style, social tendencies
      D · goal_tracking    — only when social_goal is non-empty

    Each sub-task uses a short, flat JSON schema that is easier for smaller models
    to follow reliably.  Results are merged to recreate the original output schema.

    Parameters
    ----------
    current_input : str
        The dialogue text at timestep t.
    prior_memory : dict, optional
        The memory dictionary from timestep t-1.
    base_url, api_key, model_name, temperature, seed : LLM parameters.
    debug : bool
        Pretty-printed debug output.
    max_retries : int
        Parse-failure retries per sub-task (each sub-task retries independently).
    social_goal : str
        If non-empty, activates the goal_tracking sub-task.

    Returns
    -------
    dict
        Structured Perception dictionary — same schema as the original module:
        {objective_facts, mental_state, cultural_cues[, goal_tracking]}
    """
    llm_kw = dict(
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=temperature, seed=seed,
    )

    try:
        from telemetry import get_active_collector, set_active_collector, clear_active_collector as _tel_clear
        _parent_collector = get_active_collector()
    except ImportError:
        _parent_collector = None

    def _propagate_telemetry(fn):
        """Wrap fn so the active TelemetryCollector is re-set on the sub-thread."""
        def _wrapper():
            if _parent_collector is not None:
                set_active_collector(_parent_collector)
            try:
                return fn()
            finally:
                if _parent_collector is not None:
                    _tel_clear()
        return _wrapper

    if debug:
        print(_box("PERCEPTION MODULE  [parallel sub-tasks]"))
        print(f"\n  {_c('Input:', _BOLD)} {_c(_truncate(current_input, 100), _CYAN)}")
        n_tasks = 4 if social_goal else 3
        print(f"  {_c('Sub-tasks:', _BOLD)} {_c(str(n_tasks), _YELLOW)} "
              f"({_c('A facts · B mental · C cultural' + (' · D goal' if social_goal else ''), _DIM)})")
        if prior_memory:
            known = list(prior_memory.keys())
            print(f"  {_c('Prior memory keys:', _BOLD)} {_c(', '.join(known[:8]), _DIM)}"
                  + (_c(f"  (+{len(known)-8} more)", _DIM) if len(known) > 8 else ""))

    def _with_retry(fn, label: str, fallback: dict):
        """Run fn() up to max_retries times; return fallback on persistent failure."""
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if debug:
                    print(f"  {_c(f'⚠ [{label}] attempt {attempt} failed: {e}', _RED)}")
        if debug:
            print(f"  {_c(f'⚠ [{label}] all retries exhausted — using fallback', _RED)}")
        return fallback

    def _run_facts():
        return _with_retry(
            lambda: _call_facts(current_input, prior_memory, **llm_kw),
            label="facts",
            fallback={"static_attributes": {}, "dynamic_events": []},
        )

    def _run_mental():
        return _with_retry(
            lambda: _call_mental(current_input, **llm_kw),
            label="mental",
            fallback={"immediate_intent": "", "emotion": {"category": "neutral", "intensity": "low"},
                      "values_and_obsessions": []},
        )

    def _run_cultural():
        return _with_retry(
            lambda: _call_cultural(current_input, **llm_kw),
            label="cultural",
            fallback={"communication_style": "", "social_tendencies": []},
        )

    def _run_goal():
        return _with_retry(
            lambda: _call_goal_tracking(current_input, social_goal, **llm_kw),
            label="goal",
            fallback={"progress_assessment": "", "partner_signals": [],
                      "obstacles": [], "suggested_next_move": ""},
        )

    tasks = {
        "facts":    _propagate_telemetry(_run_facts),
        "mental":   _propagate_telemetry(_run_mental),
        "cultural": _propagate_telemetry(_run_cultural),
    }
    if social_goal:
        tasks["goal"] = _propagate_telemetry(_run_goal)

    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_map = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                if debug:
                    print(f"  {_c(f'⚠ [{name}] executor error: {e}', _RED)}")
                results[name] = {}

    perception = {
        "objective_facts": results.get("facts", {}),
        "mental_state":    results.get("mental", {}),
        "cultural_cues":   results.get("cultural", {}),
    }
    if social_goal and "goal" in results:
        perception["goal_tracking"] = results["goal"]

    if debug:
        print(_section("Extracted Perception"))

        obj   = perception.get("objective_facts", {})
        attrs = obj.get("static_attributes", {})
        events = obj.get("dynamic_events", [])
        non_null = {k: v for k, v in attrs.items() if v is not None}
        print(f"\n  {_c('📋 Objective Facts', _BOLD)}")
        if non_null:
            for k, v in non_null.items():
                print(f"    {_c(k, _DIM)}: {_c(str(v), _WHITE)}")
        else:
            print(f"    {_c('(no static attributes)', _DIM)}")
        for ev in events[:3]:
            print(f"    {_c('⏱', _YELLOW)} {_c(ev.get('event',''), _WHITE)} "
                  f"{_c('['+ev.get('time_reference','?')+']', _DIM)}")

        ms    = perception.get("mental_state", {})
        emo   = ms.get("emotion", {})
        intent = ms.get("immediate_intent", "—")
        vals  = ms.get("values_and_obsessions", [])
        emo_color = (_GREEN if emo.get("intensity") == "low"
                     else _YELLOW if emo.get("intensity") == "medium" else _RED)
        print(f"\n  {_c('🧠 Mental State', _BOLD)}")
        print(f"    Intent  : {_c(_truncate(intent, 70), _CYAN)}")
        print(f"    Emotion : {_c(emo.get('category','—'), emo_color, _BOLD)} "
              f"({_c(emo.get('intensity','—'), _DIM)})")
        for v in vals[:2]:
            print(f"    Value   : {_c(_truncate(v, 70), _MAGENTA)}")

        cc    = perception.get("cultural_cues", {})
        style = cc.get("communication_style", "—")
        social = cc.get("social_tendencies", [])
        print(f"\n  {_c('🌐 Cultural Cues', _BOLD)}")
        print(f"    Style   : {_c(style, _GREEN)}")
        if social:
            print(f"    Tendencies: {_c(', '.join(social[:3]), _WHITE)}")

        gt = perception.get("goal_tracking")
        if gt:
            print(f"\n  {_c('🎯 Goal Tracking', _BOLD)}")
            print(f"    Progress : {_c(_truncate(gt.get('progress_assessment','—'), 70), _YELLOW)}")
            if gt.get("partner_signals"):
                print(f"    Signals  : {_c(', '.join(gt['partner_signals'][:2]), _WHITE)}")
            if gt.get("obstacles"):
                print(f"    Obstacles: {_c(', '.join(gt['obstacles'][:2]), _RED)}")
            print(f"    Next move: {_c(_truncate(gt.get('suggested_next_move','—'), 70), _GREEN)}")
        print()

    return perception
