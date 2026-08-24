from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from llm_client import call_llm

logger = logging.getLogger("sotopia_framework")

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

def _box(title, width=66):
    pad = width - len(title) - 4
    return (
        f"\n{_c('┌' + '─'*(width-2) + '┐', _CYAN)}\n"
        f"{_c('│', _CYAN)}  {_c(title, _BOLD, _WHITE)}{'':>{pad}}{_c('│', _CYAN)}\n"
        f"{_c('└' + '─'*(width-2) + '┘', _CYAN)}"
    )

def _section(title, width=64):
    bar = "─" * ((width - len(title) - 2) // 2)
    return f"\n{_c(f'{bar} {title} {bar}', _YELLOW, _BOLD)}"

def _truncate(s, n=60):
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

LAYER_WORLD   = "world_facts"
LAYER_MENTAL  = "mental_state"
LAYER_META    = "dialogue_meta"

BDI_BELIEFS    = "beliefs"
BDI_DESIRES    = "desires"
BDI_INTENTIONS = "intentions"

DEFAULT_CONFIDENCE_WORLD = 0.90
DEFAULT_CONFIDENCE_BDI   = 0.65
RETRACT_THRESHOLD        = 0.20

def _empty_memory() -> dict:
    return {
        LAYER_WORLD: {},
        LAYER_MENTAL: {
            BDI_BELIEFS:    {},
            BDI_DESIRES:    {},
            BDI_INTENTIONS: {},
        },
        LAYER_META: {
            "turn_count":          0,
            "dominant_topic_seq":  [],
            "belief_revision_log": [],
            "user_model_of_ai":    None,
        },
    }

def _ensure_schema(memory: dict) -> dict:
    """Guarantee the three-layer schema is present (backward-compatible)."""
    base = _empty_memory()
    for layer in (LAYER_WORLD, LAYER_MENTAL, LAYER_META):
        if layer not in memory:
            memory[layer] = base[layer]
    for bdi_key in (BDI_BELIEFS, BDI_DESIRES, BDI_INTENTIONS):
        existing = memory[LAYER_MENTAL].get(bdi_key)
        if existing is None:
            memory[LAYER_MENTAL][bdi_key] = {}
        elif isinstance(existing, list):
            converted = {}
            for i, item in enumerate(existing):
                if isinstance(item, dict):
                    val  = item.get("value") or item.get("content", "")
                    conf = item.get("confidence", DEFAULT_CONFIDENCE_BDI)
                    converted["item_" + str(i)] = {"value": val, "confidence": conf}
            memory[LAYER_MENTAL][bdi_key] = converted
    for meta_key, default in base[LAYER_META].items():
        memory[LAYER_META].setdefault(meta_key, default)
    return memory

def _bayes_update(prior: float, likelihood: float) -> float:
    eps = 1e-6
    p = max(eps, min(1 - eps, prior))
    e = max(eps, min(1 - eps, likelihood))
    numerator = e * p
    return round(numerator / (numerator + (1 - e) * (1 - p)), 4)

def _safe_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

_TOM_SYSTEM = (
    "You are a Theory-of-Mind reasoner embedded in a dialogue agent. "
    "Given a memory snapshot and the latest dialogue perception, infer the user's hidden mental states. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_TOM_USER_TMPL = """\
prior_memory:
{memory_json}

perception:
{perception_json}

Infer hidden mental states. Output this exact JSON shape (no extra keys):
{{
  "user_model_of_ai": "<what the user believes the AI currently knows about them; use 'unknown' on the first turn>",
  "hidden_desires": [
    {{
      "desire": "<latent need NOT directly stated>",
      "confidence": 0.0,
      "evidence": "<brief rationale citing perception signals>"
    }}
  ],
  "belief_revision_event": {{
    "detected": false,
    "slot": null,
    "old_value": null,
    "new_value": null,
    "rationale": ""
  }},
  "dominant_topic": "<2-4 word label for the main topic>"
}}

Rules:
- hidden_desires: include 1-3 items with confidence >= 0.50; return empty list if none.
- belief_revision_event: set detected=true only when the user clearly contradicts a prior intent in memory.
- Do NOT fabricate. Use null/false when something cannot be reliably inferred."""

_MEM_OPS_SYSTEM = (
    "You are a memory operations agent. Given a memory snapshot and dialogue perception, "
    "generate the minimal set of update operations needed. "
    "Output ONLY valid JSON. No preamble, no explanation, no markdown fences."
)

_MEM_OPS_USER_TMPL = """\
prior_memory:
{memory_json}

perception:
{perception_json}

Generate memory update operations. Output this exact JSON shape (no extra keys):
{{
  "memory_operations": [
    {{
      "action": "<ASSERT | REVISE | RETRACT | HOLD>",
      "layer": "<world_facts | mental_state>",
      "bdi_key": "<beliefs | desires | intentions | null>",
      "slot": "<snake_case_slot_name>",
      "value": "<new value; required for ASSERT and REVISE>",
      "confidence": 0.0,
      "likelihood": 0.0,
      "rationale": "<one sentence justification>"
    }}
  ]
}}

Action semantics:
- ASSERT  : Add a new slot not yet in prior_memory.
- REVISE  : Update an existing slot. likelihood = P(evidence | new_value_correct). Bayes rule is applied externally.
- RETRACT : Remove a slot that is false, completed, or superseded.
- HOLD    : Explicitly acknowledge no change is warranted.

Rules:
- world_facts layer: stable factual attributes (name, job, location, relationships). Set bdi_key to null.
- mental_state layer: BDI slots only. bdi_key MUST be exactly one of: beliefs, desires, intentions.
- Do NOT emit operations for layer "dialogue_meta" — it is managed automatically.
- Be conservative: prefer HOLD over noisy assertions.
- confidence values must reflect genuine epistemic uncertainty; avoid extremes.

[CRITICAL] bdi_key mapping (perception field names are NOT valid bdi_key values):
  perception "values_and_obsessions"  →  bdi_key: "beliefs"   (user's deep convictions)
  perception "immediate_intent"       →  bdi_key: "intentions" (what the user plans to do)
  perception "emotion"                →  bdi_key: "beliefs"    (user's belief about their situation)
  perception "communication_style"    →  layer: world_facts,   bdi_key: null
  perception "social_tendencies"      →  layer: world_facts,   bdi_key: null
  Never use "values_and_obsessions", "emotion", or "immediate_intent" as the bdi_key field."""

COMBINED_MEMORY_SYSTEM_PROMPT = _TOM_SYSTEM
TOM_INFERENCE_SYSTEM_PROMPT   = _TOM_SYSTEM
MEMORY_OPS_SYSTEM_PROMPT      = _MEM_OPS_SYSTEM

def _extract_json(raw: str):
    """
    Robustly extract a JSON object or array from raw LLM output.

    Improvements over the original:
      1. Strips // line comments — 7B models sometimes reproduce them from schema examples.
      2. Removes trailing commas before } / ] — another common 7B artefact.
      3. Tries both object and array patterns in the greedy fallback.
      4. Error preview capped at 200 chars to keep logs readable.
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

_ACTION_STYLE = {
    "ASSERT":  (_GREEN,  "✚"),
    "REVISE":  (_YELLOW, "↻"),
    "RETRACT": (_RED,    "✖"),
    "HOLD":    (_DIM,    "·"),
}

def _conf_bar(conf: float, width: int = 12) -> str:
    filled = round(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    color = _GREEN if conf >= 0.70 else _YELLOW if conf >= 0.40 else _RED
    return _c(bar, color) + f" {_c(f'{conf:.2f}', _BOLD)}"

def _execute_instructions(
    memory: dict,
    instructions: list[dict],
    tom: dict,
    turn_count: int,
    debug: bool,
) -> tuple[dict, dict]:
    VALID_ACTIONS = {"ASSERT", "REVISE", "RETRACT", "HOLD"}
    counts = {a: 0 for a in VALID_ACTIONS}
    revision_log = memory[LAYER_META]["belief_revision_log"]

    _BDI_KEY_COERCE: dict[str, str] = {
        "values_and_obsessions": BDI_BELIEFS,
        "immediate_intent":      BDI_INTENTIONS,
        "emotion":               BDI_BELIEFS,
        "communication_style":   None,
        "social_tendencies":     None,
    }

    if debug:
        print(_section("Operations"))

    for idx, op in enumerate(instructions):
        action     = op.get("action", "").upper()
        layer      = op.get("layer", LAYER_WORLD)
        bdi_key    = op.get("bdi_key")
        slot       = op.get("slot")
        value      = op.get("value")
        confidence = _safe_float(op.get("confidence"), DEFAULT_CONFIDENCE_WORLD)
        likelihood = _safe_float(op.get("likelihood"), 0.7)
        rationale  = op.get("rationale", "")

        if action not in VALID_ACTIONS:
            logger.warning("[MemoryUpdateModule] Unknown action '%s' at index %d — skipped.", action, idx)
            continue
        if slot is None:
            logger.warning("[MemoryUpdateModule] Op at index %d missing 'slot' — skipped.", idx)
            continue

        if layer == LAYER_WORLD:
            target = memory[LAYER_WORLD]
        elif layer == LAYER_MENTAL:

            if bdi_key not in (BDI_BELIEFS, BDI_DESIRES, BDI_INTENTIONS):
                coerced = _BDI_KEY_COERCE.get(bdi_key)
                if coerced is None and bdi_key in _BDI_KEY_COERCE:

                    layer = LAYER_WORLD
                    bdi_key = None
                    target = memory[LAYER_WORLD]
                    logger.debug(
                        "[MemoryUpdateModule] bdi_key '%s' at index %d redirected → world_facts.",
                        op.get("bdi_key"), idx,
                    )
                elif coerced:
                    logger.debug(
                        "[MemoryUpdateModule] bdi_key '%s' at index %d auto-corrected → '%s'.",
                        bdi_key, idx, coerced,
                    )
                    bdi_key = coerced
                    target = memory[LAYER_MENTAL][bdi_key]
                else:
                    logger.warning(
                        "[MemoryUpdateModule] Invalid bdi_key '%s' at index %d — skipped.",
                        bdi_key, idx,
                    )
                    continue
            else:
                target = memory[LAYER_MENTAL][bdi_key]
        else:
            continue

        if action == "ASSERT":
            target[slot] = {"value": value, "confidence": confidence}

        elif action == "REVISE":
            if slot in target:
                prior_conf = _safe_float(target[slot].get("confidence"), DEFAULT_CONFIDENCE_BDI)
                posterior  = _bayes_update(prior_conf, likelihood)
                old_value  = target[slot].get("value")
                if (layer == LAYER_MENTAL
                        and old_value is not None
                        and str(old_value) != str(value)):
                    revision_log.append({
                        "turn":      turn_count,
                        "bdi_key":   bdi_key,
                        "slot":      slot,
                        "old_value": old_value,
                        "new_value": value,
                        "conf_delta": round(posterior - prior_conf, 4),
                    })
                target[slot] = {"value": value, "confidence": posterior}
                if posterior < RETRACT_THRESHOLD:
                    del target[slot]
                    if debug:
                        print(f"  {_c('⤷ auto-retract', _RED)} {_c(slot, _WHITE)} "
                              f"posterior={posterior:.2f} < threshold")
            else:
                target[slot] = {"value": value, "confidence": confidence}

        elif action == "RETRACT":
            target.pop(slot, None)

        counts[action] += 1

        if debug:
            color, icon = _ACTION_STYLE.get(action, (_WHITE, "?"))
            loc = (f"{_c(layer, _DIM)}"
                   + (f"/{_c(bdi_key, _MAGENTA)}" if bdi_key else ""))
            val_str  = _truncate(str(value), 36) if value is not None else ""
            conf_str = _conf_bar(confidence, 8) if action in ("ASSERT", "REVISE") else ""
            rat_str  = _c(_truncate(rationale, 40), _DIM)
            print(f"  {_c(icon, color, _BOLD)} {_c(action, color):<10} "
                  f"{loc:<36} {_c(slot, _WHITE):<28} {val_str}")
            if conf_str:
                print(f"    {'':>10} conf: {conf_str}  — {rat_str}")

    return memory, counts

def _update_meta(memory: dict, tom: dict, turn_count: int) -> dict:
    meta = memory[LAYER_META]
    meta["turn_count"] = turn_count + 1
    dominant_topic = tom.get("dominant_topic")
    if dominant_topic:
        topics: list = meta.get("dominant_topic_seq", [])
        topics.append({"turn": turn_count, "topic": dominant_topic})
        meta["dominant_topic_seq"] = topics[-10:]
    user_model = tom.get("user_model_of_ai")
    if user_model and user_model != "unknown":
        meta["user_model_of_ai"] = user_model
    return memory

def run_memory_update(
    prior_memory: dict,
    perception: dict,
    base_url: str = "",
    api_key: str = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    seed: int = 42,
    debug: bool = False,
    max_retries: int = 2,
) -> dict:
    """
    Update the three-layer Bayesian ToM memory with information from the current turn.

    Parallel two-prompt design
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    This function replaces the previous single combined LLM call with two
    independent calls dispatched in parallel:

      Call T — ToM Inference  → infers hidden desires, belief revisions, topic
      Call M — Memory Ops     → generates ASSERT/REVISE/RETRACT/HOLD instructions

    Both calls receive the same inputs (prior_memory + perception) and are fully
    independent.  The Python-side merge and all downstream logic (Bayesian updates,
    desire injection, meta updates) are identical to the original implementation.

    Parameters
    ----------
    prior_memory : dict
        Memory dict from timestep t-1.  Pass {} for the first turn.
    perception : dict
        Structured Perception dict produced by the Perception Module.
    base_url, api_key, model_name, temperature, seed : LLM parameters.
    debug : bool
        Print detailed, colour-coded debug output.
    max_retries : int
        Parse-failure retries per call.

    Returns
    -------
    dict
        Updated memory dict with three-layer schema preserved.
    """
    memory = _ensure_schema(dict(prior_memory))
    turn_count = memory[LAYER_META].get("turn_count", 0)

    if debug:
        print(_box("MEMORY UPDATE  ·  Bayesian ToM Edition  [parallel 2-prompt]"))
        wf_n  = len(memory[LAYER_WORLD])
        bdi_n = sum(len(memory[LAYER_MENTAL][k])
                    for k in (BDI_BELIEFS, BDI_DESIRES, BDI_INTENTIONS))
        print(f"\n  {_c('Turn:', _BOLD)} {_c(str(turn_count), _CYAN)}   "
              f"{_c('world_facts:', _BOLD)} {_c(str(wf_n), _CYAN)}   "
              f"{_c('BDI slots:', _BOLD)} {_c(str(bdi_n), _CYAN)}")
        print(f"  {_c('Sub-calls:', _DIM)} T (ToM)  ∥  M (MemOps)  — dispatched concurrently")

    llm_kw = dict(
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=temperature, seed=seed,
    )

    try:
        from telemetry import get_active_collector, set_active_collector, clear_active_collector as _tel_clear
        _parent_collector = get_active_collector()
    except ImportError:
        _parent_collector = None

    memory_json     = json.dumps(memory, ensure_ascii=False, indent=2)
    perception_json = json.dumps(perception, ensure_ascii=False, indent=2)

    def _call_tom() -> dict:
        if _parent_collector is not None:
            set_active_collector(_parent_collector)
        try:
          user_msg = _TOM_USER_TMPL.format(
            memory_json=memory_json,
            perception_json=perception_json,
          )
          last_err = None
          for attempt in range(1, max_retries + 1):
            raw = call_llm(system_prompt=_TOM_SYSTEM, user_message=user_msg,
                           _module="memory_tom", **llm_kw)
            try:
                parsed = _extract_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a JSON object.")
                return parsed
            except (ValueError, TypeError) as e:
                last_err = e
                if debug:
                    print(f"  {_c(f'⚠ [ToM] attempt {attempt}: {e}', _RED)}")
          if debug:
            print(f"  {_c('⚠ [ToM] all retries exhausted — fallback to empty', _RED)}")
          return {}
        finally:
            if _parent_collector is not None:
                _tel_clear()

    def _call_mem_ops() -> list:
        if _parent_collector is not None:
            set_active_collector(_parent_collector)
        try:
          user_msg = _MEM_OPS_USER_TMPL.format(
            memory_json=memory_json,
            perception_json=perception_json,
          )
          last_err = None
          for attempt in range(1, max_retries + 1):
            raw = call_llm(system_prompt=_MEM_OPS_SYSTEM, user_message=user_msg,
                           _module="memory_ops", **llm_kw)
            try:
                parsed = _extract_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a JSON object.")
                ops = parsed.get("memory_operations", [])
                if not isinstance(ops, list):
                    raise ValueError(f"'memory_operations' must be a list, got {type(ops)}")
                return ops
            except (ValueError, TypeError) as e:
                last_err = e
                if debug:
                    print(f"  {_c(f'⚠ [MemOps] attempt {attempt}: {e}', _RED)}")
          if debug:
            print(f"  {_c('⚠ [MemOps] all retries exhausted — fallback to empty list', _RED)}")
          return []
        finally:
            if _parent_collector is not None:
                _tel_clear()

    if debug:
        print(_section("Parallel Sub-Calls  T ∥ M"))

    tom: dict          = {}
    instructions: list = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_tom = executor.submit(_call_tom)
        future_ops = executor.submit(_call_mem_ops)
        try:
            tom = future_tom.result()
        except Exception as e:
            if debug:
                print(f"  {_c(f'⚠ [ToM] executor error: {e}', _RED)}")
        try:
            instructions = future_ops.result()
        except Exception as e:
            if debug:
                print(f"  {_c(f'⚠ [MemOps] executor error: {e}', _RED)}")

    if debug:
        topic  = tom.get("dominant_topic", "—")
        umai   = _truncate(str(tom.get("user_model_of_ai", "—")), 60)
        hd_list = tom.get("hidden_desires", [])
        rev_ev  = tom.get("belief_revision_event", {})

        print(f"  {_c('Topic:', _BOLD)}         {_c(topic, _CYAN)}")
        print(f"  {_c('User→AI model:', _BOLD)}  {_c(umai, _WHITE)}")
        if hd_list:
            print(f"  {_c('Hidden desires:', _BOLD)}")
            for hd in hd_list:
                conf = hd.get("confidence", 0)
                print(f"    {_conf_bar(conf, 8)}  {_c(_truncate(hd.get('desire',''), 55), _MAGENTA)}")
        if rev_ev.get("detected"):
            print(f"  {_c('⚡ Belief revision:', _YELLOW, _BOLD)} "
                  f"{_c(rev_ev.get('slot','?'), _WHITE)}: "
                  f"{_c(str(rev_ev.get('old_value','')), _RED)} → "
                  f"{_c(str(rev_ev.get('new_value','')), _GREEN)}")
        print(f"  {_c(f'Memory ops to apply: {len(instructions)}', _DIM)}")

    memory, counts = _execute_instructions(memory, instructions, tom, turn_count, debug)

    memory = _update_meta(memory, tom, turn_count)

    for hd in tom.get("hidden_desires", []):
        conf = _safe_float(hd.get("confidence"), 0.0)
        if conf < 0.50:
            continue
        desire_text = hd.get("desire", "").strip()
        if not desire_text:
            continue
        slot_key = "hd_" + re.sub(r"\W+", "_", desire_text[:40]).strip("_").lower()
        desires = memory[LAYER_MENTAL][BDI_DESIRES]
        if slot_key not in desires:
            desires[slot_key] = {"value": desire_text, "confidence": conf}
            if debug:
                print(f"  {_c('⤷ ToM→desire', _MAGENTA)} "
                      f"{_c(slot_key, _WHITE)}  {_conf_bar(conf, 8)}")
        else:
            prior_conf = _safe_float(desires[slot_key].get("confidence"), DEFAULT_CONFIDENCE_BDI)
            posterior  = _bayes_update(prior_conf, conf)
            desires[slot_key]["confidence"] = posterior
            if debug:
                print(f"  {_c('⤷ ToM→desire(revised)', _MAGENTA)} "
                      f"{_c(slot_key, _WHITE)}  {_conf_bar(posterior, 8)}")

    if debug:
        print(_section("Memory Summary"))
        wf_n = len(memory[LAYER_WORLD])
        b_n  = len(memory[LAYER_MENTAL][BDI_BELIEFS])
        d_n  = len(memory[LAYER_MENTAL][BDI_DESIRES])
        i_n  = len(memory[LAYER_MENTAL][BDI_INTENTIONS])
        rev_n = len(memory[LAYER_META]["belief_revision_log"])

        print(f"\n  {_c('world_facts:', _BOLD)} {_c(str(wf_n), _CYAN)} slots   "
              f"{_c('beliefs:', _BOLD)} {_c(str(b_n), _BLUE)}   "
              f"{_c('desires:', _BOLD)} {_c(str(d_n), _MAGENTA)}   "
              f"{_c('intentions:', _BOLD)} {_c(str(i_n), _GREEN)}")
        print(f"  {_c('Belief revisions (cumulative):', _BOLD)} "
              f"{_c(str(rev_n), _YELLOW)}")
        print(f"\n  {_c('✚ ASSERT:', _GREEN, _BOLD)}  {counts.get('ASSERT',0)}   "
              f"{_c('↻ REVISE:', _YELLOW, _BOLD)}  {counts.get('REVISE',0)}   "
              f"{_c('✖ RETRACT:', _RED, _BOLD)} {counts.get('RETRACT',0)}   "
              f"{_c('· HOLD:', _DIM)}    {counts.get('HOLD',0)}")

        desires = memory[LAYER_MENTAL][BDI_DESIRES]
        if desires:
            print(f"\n  {_c('Top desires (by confidence):', _BOLD, _MAGENTA)}")
            sorted_d = sorted(desires.items(),
                              key=lambda kv: kv[1].get("confidence", 0), reverse=True)
            for slot, entry in sorted_d[:4]:
                c = entry.get("confidence", 0)
                v = _truncate(str(entry.get("value", "")), 50)
                print(f"    {_conf_bar(c, 10)}  {_c(v, _WHITE)}")
        print()

    return memory
