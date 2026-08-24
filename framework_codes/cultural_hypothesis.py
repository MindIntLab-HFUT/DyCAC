import json
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import call_llm

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_MAGENTA= "\033[35m"
_RED    = "\033[31m"
_WHITE  = "\033[37m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + _RESET

def _box(title: str, width: int = 62) -> str:
    pad = width - len(title) - 4
    return (
        f"\n{_c('┌' + '─'*(width-2) + '┐', _CYAN)}\n"
        f"{_c('│', _CYAN)}  {_c(title, _BOLD, _WHITE)}{'':>{pad}}{_c('│', _CYAN)}\n"
        f"{_c('└' + '─'*(width-2) + '┘', _CYAN)}"
    )

def _section(title: str, width: int = 60) -> str:
    bar = "─" * ((width - len(title) - 2) // 2)
    return f"\n{_c(f'{bar} {title} {bar}', _YELLOW, _BOLD)}"

def _bar_chart(label: str, value: float, max_val: float = 1.0,
               width: int = 28, color: str = _GREEN) -> str:
    filled = int(round(value / max_val * width))
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{value*100:5.1f}%"
    return f"  {label:<24} {_c(bar, color)} {_c(pct, _BOLD)}"

def _prob_bars(dist: dict, top_n: int = 5) -> str:
    lines = []
    sorted_items = sorted(dist.items(), key=lambda x: -x[1])[:top_n]
    for country, prob in sorted_items:
        color = _GREEN if prob > 0.3 else _YELLOW if prob > 0.1 else _DIM + _WHITE
        lines.append(_bar_chart(country[:24], prob, 1.0, 28, color))
    return "\n".join(lines)

def _hofstede_row(scores: dict) -> str:
    dims = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
    parts = []
    for d in dims:
        v = scores.get(d, "?")
        color = _GREEN if int(v) > 70 else _YELLOW if int(v) > 40 else _BLUE
        parts.append(f"{_c(d, _DIM)}:{_c(str(v), color, _BOLD)}")
    return "  " + "  ".join(parts)

HOFSTEDE_DB = {

    "Africa East":           {"PDI": 64, "IDV": 27, "MAS": 41, "UAI": 52,  "LTO": 32, "IVR": 40},
    "Africa West":           {"PDI": 77, "IDV": 20, "MAS": 46, "UAI": 54,  "LTO":  9, "IVR": 78},
    "Morocco":               {"PDI": 70, "IDV": 46, "MAS": 53, "UAI": 68,  "LTO": 14, "IVR": 25},
    "Nigeria":               {"PDI": 80, "IDV": 30, "MAS": 60, "UAI": 55,  "LTO": 13, "IVR": 84},
    "South Africa":          {"PDI": 49, "IDV": 65, "MAS": 63, "UAI": 49,  "LTO": 34, "IVR": 63},

    "Arab countries":        {"PDI": 80, "IDV": 38, "MAS": 53, "UAI": 68,  "LTO": 23, "IVR": 34},
    "Egypt":                 {"PDI": 70, "IDV": 25, "MAS": 45, "UAI": 80,  "LTO":  7, "IVR":  4},
    "Iran":                  {"PDI": 58, "IDV": 41, "MAS": 43, "UAI": 59,  "LTO": 14, "IVR": 40},
    "Israel":                {"PDI": 13, "IDV": 54, "MAS": 47, "UAI": 81,  "LTO": 38, "IVR":  0},
    "Saudi Arabia":          {"PDI": 95, "IDV": 25, "MAS": 60, "UAI": 80,  "LTO": 36, "IVR": 52},
    "Turkey":                {"PDI": 66, "IDV": 37, "MAS": 45, "UAI": 85,  "LTO": 46, "IVR": 49},

    "Bangladesh":            {"PDI": 80, "IDV": 20, "MAS": 55, "UAI": 60,  "LTO": 47, "IVR": 20},
    "China":                 {"PDI": 80, "IDV": 20, "MAS": 66, "UAI": 30,  "LTO": 87, "IVR": 24},
    "Hong Kong":             {"PDI": 68, "IDV": 25, "MAS": 57, "UAI": 29,  "LTO": 61, "IVR": 17},
    "India":                 {"PDI": 77, "IDV": 48, "MAS": 56, "UAI": 40,  "LTO": 51, "IVR": 26},
    "Indonesia":             {"PDI": 78, "IDV": 14, "MAS": 46, "UAI": 48,  "LTO": 62, "IVR": 38},
    "Japan":                 {"PDI": 54, "IDV": 46, "MAS": 95, "UAI": 92,  "LTO": 88, "IVR": 42},
    "Malaysia":              {"PDI":104, "IDV": 26, "MAS": 50, "UAI": 36,  "LTO": 41, "IVR": 57},
    "Pakistan":              {"PDI": 55, "IDV": 14, "MAS": 50, "UAI": 70,  "LTO": 50, "IVR":  0},
    "Philippines":           {"PDI": 94, "IDV": 32, "MAS": 64, "UAI": 44,  "LTO": 27, "IVR": 42},
    "Singapore":             {"PDI": 74, "IDV": 20, "MAS": 48, "UAI":  8,  "LTO": 72, "IVR": 46},
    "South Korea":           {"PDI": 60, "IDV": 18, "MAS": 39, "UAI": 85,  "LTO":100, "IVR": 29},
    "Taiwan":                {"PDI": 58, "IDV": 17, "MAS": 45, "UAI": 69,  "LTO": 93, "IVR": 49},
    "Thailand":              {"PDI": 64, "IDV": 20, "MAS": 34, "UAI": 64,  "LTO": 32, "IVR": 45},
    "Vietnam":               {"PDI": 70, "IDV": 20, "MAS": 40, "UAI": 30,  "LTO": 57, "IVR": 35},

    "Australia":             {"PDI": 38, "IDV": 90, "MAS": 61, "UAI": 51,  "LTO": 21, "IVR": 71},
    "New Zealand":           {"PDI": 22, "IDV": 79, "MAS": 58, "UAI": 49,  "LTO": 33, "IVR": 75},

    "Canada":                {"PDI": 39, "IDV": 80, "MAS": 52, "UAI": 48,  "LTO": 36, "IVR": 68},
    "United States":         {"PDI": 40, "IDV": 91, "MAS": 62, "UAI": 46,  "LTO": 26, "IVR": 68},

    "Argentina":             {"PDI": 49, "IDV": 46, "MAS": 56, "UAI": 86,  "LTO": 20, "IVR": 62},
    "Brazil":                {"PDI": 69, "IDV": 38, "MAS": 49, "UAI": 76,  "LTO": 44, "IVR": 59},
    "Chile":                 {"PDI": 63, "IDV": 23, "MAS": 28, "UAI": 86,  "LTO": 31, "IVR": 68},
    "Colombia":              {"PDI": 67, "IDV": 13, "MAS": 64, "UAI": 80,  "LTO": 13, "IVR": 83},
    "El Salvador":           {"PDI": 66, "IDV": 19, "MAS": 40, "UAI": 94,  "LTO": 20, "IVR": 89},
    "Mexico":                {"PDI": 81, "IDV": 30, "MAS": 69, "UAI": 82,  "LTO": 24, "IVR": 97},
    "Peru":                  {"PDI": 64, "IDV": 16, "MAS": 42, "UAI": 87,  "LTO": 25, "IVR": 46},
    "Trinidad and Tobago":   {"PDI": 47, "IDV": 16, "MAS": 58, "UAI": 55,  "LTO": 13, "IVR": 80},
    "Uruguay":               {"PDI": 61, "IDV": 36, "MAS": 38, "UAI":100,  "LTO": 26, "IVR": 53},
    "Venezuela":             {"PDI": 81, "IDV": 12, "MAS": 73, "UAI": 76,  "LTO": 16, "IVR":100},

    "Austria":               {"PDI": 11, "IDV": 55, "MAS": 79, "UAI": 70,  "LTO": 60, "IVR": 63},
    "Belgium":               {"PDI": 65, "IDV": 75, "MAS": 54, "UAI": 94,  "LTO": 82, "IVR": 57},
    "Denmark":               {"PDI": 18, "IDV": 74, "MAS": 16, "UAI": 23,  "LTO": 35, "IVR": 70},
    "Finland":               {"PDI": 33, "IDV": 63, "MAS": 26, "UAI": 59,  "LTO": 38, "IVR": 57},
    "France":                {"PDI": 68, "IDV": 71, "MAS": 43, "UAI": 86,  "LTO": 63, "IVR": 48},
    "Germany":               {"PDI": 35, "IDV": 67, "MAS": 66, "UAI": 65,  "LTO": 83, "IVR": 40},
    "Greece":                {"PDI": 60, "IDV": 35, "MAS": 57, "UAI":112,  "LTO": 45, "IVR": 50},
    "Ireland":               {"PDI": 28, "IDV": 70, "MAS": 68, "UAI": 35,  "LTO": 24, "IVR": 65},
    "Italy":                 {"PDI": 50, "IDV": 76, "MAS": 70, "UAI": 75,  "LTO": 61, "IVR": 30},
    "Luxembourg":            {"PDI": 40, "IDV": 60, "MAS": 50, "UAI": 70,  "LTO": 64, "IVR": 56},
    "Malta":                 {"PDI": 56, "IDV": 59, "MAS": 47, "UAI": 96,  "LTO": 47, "IVR": 66},
    "Netherlands":           {"PDI": 38, "IDV": 80, "MAS": 14, "UAI": 53,  "LTO": 67, "IVR": 68},
    "Norway":                {"PDI": 31, "IDV": 69, "MAS":  8, "UAI": 50,  "LTO": 35, "IVR": 55},
    "Portugal":              {"PDI": 63, "IDV": 27, "MAS": 31, "UAI":104,  "LTO": 28, "IVR": 33},
    "Spain":                 {"PDI": 57, "IDV": 51, "MAS": 42, "UAI": 86,  "LTO": 48, "IVR": 44},
    "Sweden":                {"PDI": 31, "IDV": 71, "MAS":  5, "UAI": 29,  "LTO": 53, "IVR": 78},
    "Switzerland":           {"PDI": 34, "IDV": 68, "MAS": 70, "UAI": 58,  "LTO": 74, "IVR": 66},
    "United Kingdom":        {"PDI": 35, "IDV": 89, "MAS": 66, "UAI": 35,  "LTO": 51, "IVR": 69},

    "Bulgaria":              {"PDI": 70, "IDV": 30, "MAS": 40, "UAI": 85,  "LTO": 69, "IVR": 16},
    "Croatia":               {"PDI": 73, "IDV": 33, "MAS": 40, "UAI": 80,  "LTO": 58, "IVR": 33},
    "Czech Republic":        {"PDI": 57, "IDV": 58, "MAS": 57, "UAI": 74,  "LTO": 70, "IVR": 29},
    "Estonia":               {"PDI": 40, "IDV": 60, "MAS": 30, "UAI": 60,  "LTO": 82, "IVR": 16},
    "Hungary":               {"PDI": 46, "IDV": 80, "MAS": 88, "UAI": 82,  "LTO": 58, "IVR": 31},
    "Latvia":                {"PDI": 44, "IDV": 70, "MAS":  9, "UAI": 63,  "LTO": 69, "IVR": 13},
    "Lithuania":             {"PDI": 42, "IDV": 60, "MAS": 19, "UAI": 65,  "LTO": 82, "IVR": 16},
    "Poland":                {"PDI": 68, "IDV": 60, "MAS": 64, "UAI": 93,  "LTO": 38, "IVR": 29},
    "Romania":               {"PDI": 90, "IDV": 30, "MAS": 42, "UAI": 90,  "LTO": 52, "IVR": 20},
    "Russia":                {"PDI": 93, "IDV": 39, "MAS": 36, "UAI": 95,  "LTO": 81, "IVR": 20},
    "Serbia":                {"PDI": 86, "IDV": 25, "MAS": 43, "UAI": 92,  "LTO": 52, "IVR": 28},
    "Slovakia":              {"PDI":104, "IDV": 52, "MAS":110, "UAI": 51,  "LTO": 77, "IVR": 28},
    "Slovenia":              {"PDI": 71, "IDV": 27, "MAS": 19, "UAI": 88,  "LTO": 49, "IVR": 48},
}

COUNTRY_ALIASES: dict[str, str] = {
    "united states of america": "United States", "usa": "United States",
    "u.s.a.": "United States", "u.s.": "United States", "us": "United States",
    "america": "United States", "the united states": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "england": "United Kingdom", "wales": "United Kingdom",
    "scotland": "United Kingdom", "the united kingdom": "United Kingdom",
    "prc": "China", "people's republic of china": "China", "mainland china": "China",
    "korea": "South Korea", "republic of korea": "South Korea",
    "rok": "South Korea", "korean": "South Korea",
    "ksa": "Saudi Arabia", "kingdom of saudi arabia": "Saudi Arabia",
    "rsa": "South Africa",
    "russian federation": "Russia", "ussr": "Russia",
    "west germany": "Germany", "deutschland": "Germany",
    "the netherlands": "Netherlands", "holland": "Netherlands",
    "the philippines": "Philippines",
    "brasil": "Brazil",
    "turkiye": "Turkey", "türkiye": "Turkey",
    "czech rep": "Czech Republic", "czechia": "Czech Republic", "czech": "Czech Republic",
    "slovak rep": "Slovakia", "slovak republic": "Slovakia",
    "trinidad": "Trinidad and Tobago", "tobago": "Trinidad and Tobago", "t&t": "Trinidad and Tobago",
    "hk": "Hong Kong",
    "roc": "Taiwan", "republic of china": "Taiwan",
    "nz": "New Zealand", "new zeland": "New Zealand",
    "aus": "Australia", "oz": "Australia",
    "eire": "Ireland", "republic of ireland": "Ireland",
    "arab world": "Arab countries", "middle east": "Arab countries",
    "east africa": "Africa East", "eastern africa": "Africa East",
    "west africa": "Africa West", "western africa": "Africa West",
    "persia": "Iran", "islamic republic of iran": "Iran",
    "columbia": "Colombia",
    "lux": "Luxembourg",
}

def _normalise_country(name: str) -> str:
    if name in HOFSTEDE_DB:
        return name
    lower = name.lower().strip()
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]
    for key in HOFSTEDE_DB:
        if key.lower() == lower:
            return key
    return name

GLOBAL_DEFAULT = {"PDI": 50, "IDV": 50, "MAS": 50, "UAI": 50, "LTO": 50, "IVR": 50}
DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

HYPOTHESIS_SELECTION_PROMPT = """\
You are a cultural analyst expert in Hofstede's dimensions.

Based on the structured perception data extracted from the user's dialogue below, identify the {n} most likely countries or regions of cultural origin.
Use communication style, social tendencies, values, emotional cues, behavioral patterns, and negotiation style as evidence.
Prefer broad regions (e.g. Arab countries, Africa West) when the signal is coarse.
Return exactly {n} country or region names ordered from most to least plausible.

Perception data:
{perception_data}

Output JSON only. No preamble, no markdown fences:
{{"countries": ["<country>", ...]}}"""

PRIOR_SCORE_PROMPT = """\
You are a cross-cultural psychologist.

Rate how plausible it is that the speaker originates from {country}, given the structured perception data below.
Score from 0.0 (very implausible) to 1.0 (very plausible). Evaluate only this country independently.

Perception data:
{perception_data}

Output JSON only. No preamble, no markdown fences:
{{"country": "{country}", "prior_score": <float 0.0-1.0>}}"""

LIKELIHOOD_EVALUATION_PROMPT = """\
You are a cross-cultural psychologist.

Assume the speaker is from {country} (PDI={PDI}, IDV={IDV}, MAS={MAS}, UAI={UAI}, LTO={LTO}, IVR={IVR}).

Dialogue:
{dialogue}

How naturally does this dialogue fit a speaker from {country}?
Score 0.0 (very unnatural for this culture) to 1.0 (very natural). Evaluate this country independently.

Output JSON only. No preamble, no markdown fences:
{{"country": "{country}", "likelihood": <float 0.0-1.0>}}"""

DIRECT_DIMENSION_PROMPT = """\
You are a cross-cultural psychologist using Hofstede-style dimensions as soft latent variables, not hard labels.

Infer the speaker's probable style on each dimension from 0 to 100, where 50 means unclear / mixed.
Use the perception summary and the latest dialogue jointly.

Perception data:
{perception_data}

Latest dialogue:
{dialogue}

Output JSON only. No preamble, no markdown fences:
{{
  "PDI": <0-100>,
  "IDV": <0-100>,
  "MAS": <0-100>,
  "UAI": <0-100>,
  "LTO": <0-100>,
  "IVR": <0-100>
}}

Rules:
- Use 40-60 when evidence is weak.
- Base scores on observable interaction style, not demographics.
- Do not output explanations."""

def _format_perception_for_prompt(perception: dict) -> str:
    """
    Serialise the perception dict into a structured plain-text block.
    Keep the prompt fairly rich because smaller models often reason better when
    cues are separated explicitly rather than compressed too aggressively.
    """
    lines: list[str] = []

    facts = perception.get("objective_facts", {}) or {}
    if facts:
        lines.append("[Objective Facts]")
        static = facts.get("static_attributes", {}) or {}
        dynamic = facts.get("dynamic_events", []) or []
        for k, v in static.items():
            if v:
                lines.append(f"  {k.replace('_', ' ').capitalize()}: {v}")
        if dynamic:
            for ev in dynamic[:5]:
                if isinstance(ev, dict):
                    desc = ev.get("event", "")
                    tref = ev.get("time_reference", "")
                    if desc:
                        suffix = f" [{tref}]" if tref else ""
                        lines.append(f"  Event: {desc}{suffix}")
                elif ev:
                    lines.append(f"  Event: {ev}")

    mental = perception.get("mental_state", {}) or {}
    if mental:
        lines.append("[Mental State]")
        intent = mental.get("immediate_intent")
        if intent:
            lines.append(f"  Immediate intent: {intent}")
        emotion = mental.get("emotion", {}) or {}
        cat = emotion.get("category", "")
        intensity = emotion.get("intensity", "")
        if cat:
            lines.append(f"  Emotion: {cat}" + (f" ({intensity})" if intensity else ""))
        values = mental.get("values_and_obsessions", []) or []
        if values:
            lines.append(f"  Values / obsessions: {'; '.join(str(v) for v in values[:5])}")

    cues = perception.get("cultural_cues", {}) or {}
    if cues:
        lines.append("[Cultural Cues]")
        comm_style = cues.get("communication_style")
        if comm_style:
            lines.append(f"  Communication style: {comm_style}")
        social = cues.get("social_tendencies", []) or []
        if social:
            lines.append(f"  Social tendencies: {', '.join(str(s) for s in social[:6])}")

    goal = perception.get("goal_tracking", {}) or {}
    if goal:
        lines.append("[Goal Tracking]")
        for key in ["progress_assessment", "suggested_next_move", "goal_stage", "partner_stance"]:
            val = goal.get(key)
            if val:
                lines.append(f"  {key.replace('_', ' ').capitalize()}: {val}")
        for key in ["partner_signals", "obstacles", "risk_flags"]:
            vals = goal.get(key, []) or []
            if vals:
                lines.append(f"  {key.replace('_', ' ').capitalize()}: {'; '.join(str(v) for v in vals[:5])}")

    covered = {"objective_facts", "mental_state", "cultural_cues", "goal_tracking"}
    for k, v in perception.items():
        if k not in covered and v:
            label = k.replace("_", " ").capitalize()
            if isinstance(v, (list, dict)):
                import json as _json
                lines.append(f"[{label}]\n  {_json.dumps(v, ensure_ascii=False)}")
            else:
                lines.append(f"[{label}]\n  {v}")

    return "\n".join(lines) if lines else "(No perception data available.)"

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from raw LLM output.

    Improvements over the original:
      1. Strips // line comments — 7B models sometimes reproduce them from examples.
      2. Removes trailing commas before } / ] — another common small-model artefact.
      3. Greedy regex fallback applies the same cleanup before attempting parse.
      4. Error preview capped at 200 chars.
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

def _extract_float_field(raw: str, key: str, fallback: float = 0.5) -> float:
    """
    Extract a single float value from raw LLM output with multiple fallback strategies.

    Strategy order:
      1. Full JSON parse → read key.
      2. Regex search for ``"key": <float>`` pattern.
      3. Regex search for any float in the [0, 1] range anywhere in the text.
      4. Return fallback.

    This handles the failure mode where the model outputs a plain number, a sentence
    like "I would rate this 0.7", or JSON with the key misspelled.
    """

    try:
        parsed = _extract_json(raw)
        val = parsed.get(key)
        if val is not None:
            return max(0.0, min(1.0, float(val)))
    except (ValueError, TypeError, AttributeError):
        pass

    pat2 = rf'"{key}"\s*:\s*([01]?\.?\d+)'
    m = re.search(pat2, raw, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    for pat in (r'\b(0\.\d+)\b', r'\b(1\.0+)\b', r'\b([01])\b'):
        m = re.search(pat, raw)
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass

    return fallback

def _select_candidate_countries(
    perception_data: str, n: int,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> list[str]:
    """Ask LLM to nominate N candidate country names."""
    prompt = HYPOTHESIS_SELECTION_PROMPT.format(
        n=n,
        perception_data=perception_data,
    )
    raw = call_llm(
        system_prompt="You are a precise JSON-outputting cultural analysis assistant.",
        user_message=prompt,
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=temperature, seed=seed,
        _module="culture_select",
    )
    try:
        result  = _extract_json(raw)
        countries = [_normalise_country(c) for c in result["countries"]]
        valid = [c for c in countries if c in HOFSTEDE_DB]
        if not valid:
            raise ValueError("No valid countries returned.")
        return valid
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"[CulturalHypothesisModule] Failed to parse country selection.\n"
            f"Error: {e}\nRaw: {raw}"
        )

def _score_prior_for_country(
    country: str, perception_data: str,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> tuple[str, float]:
    """Independently evaluate prior plausibility score (0–1) for a single country."""
    prompt = PRIOR_SCORE_PROMPT.format(
        country=country,
        perception_data=perception_data,
    )
    raw = call_llm(
        system_prompt="You are a precise JSON-outputting cultural analysis assistant.",
        user_message=prompt,
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=temperature, seed=seed,
        _module="culture_prior",
    )

    score = _extract_float_field(raw, key="prior_score", fallback=0.5)
    return country, score

def _generate_hypotheses(
    perception: dict, current_input: str, n: int,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> dict:
    """
    Generate N country hypotheses with independently-scored and normalised priors.
      1. Format the perception dict into a structured text block.
      2. LLM selects N candidate country names based on perception data.
      3. Each country's prior score is evaluated independently (in parallel)
         against the same perception data.
      4. Scores are normalised to sum to 1.0.
    """

    perception_data = _format_perception_for_prompt(perception)

    try:
        from telemetry import get_active_collector, set_active_collector, clear_active_collector as _tel_clear
        _collector = get_active_collector()
    except ImportError:
        _collector = None

    candidates = _select_candidate_countries(
        perception_data, n, base_url, api_key, model_name, temperature, seed, debug
    )
    if debug:
        print(f"\n  {_c('Candidates selected:', _BOLD)} {', '.join(candidates)}")

    def _score_with_telemetry(country: str) -> tuple[str, float]:
        """Wrapper that propagates the telemetry collector into this sub-thread."""
        if _collector is not None:
            set_active_collector(_collector)
        try:
            return _score_prior_for_country(
                country, perception_data, base_url, api_key, model_name, temperature, seed, debug,
            )
        finally:
            if _collector is not None:
                _tel_clear()

    raw_scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = {
            executor.submit(_score_with_telemetry, country): country
            for country in candidates
        }
        for future in as_completed(futures):
            country, score = future.result()
            raw_scores[country] = score

    total = sum(raw_scores.values())
    hypotheses = (
        {c: s / total for c, s in raw_scores.items()}
        if total > 0
        else {c: 1.0 / len(raw_scores) for c in raw_scores}
    )
    return hypotheses

def _evaluate_single_likelihood(
    country: str, current_input: str,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> tuple[str, float]:
    """Evaluate P(original dialogue | culture) for a single country."""
    scores = HOFSTEDE_DB[country]
    prompt = LIKELIHOOD_EVALUATION_PROMPT.format(
        country=country,
        PDI=scores["PDI"], IDV=scores["IDV"], MAS=scores["MAS"],
        UAI=scores["UAI"], LTO=scores["LTO"], IVR=scores["IVR"],
        dialogue=current_input,
    )
    raw = call_llm(
        system_prompt="You are a precise JSON-outputting cultural analysis assistant.",
        user_message=prompt,
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=temperature, seed=seed,
        _module="culture_likelihood",
    )

    likelihood = _extract_float_field(raw, key="likelihood", fallback=0.5)
    return country, likelihood

def _evaluate_likelihoods(
    current_input: str, hypotheses: dict,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> dict:
    """Evaluate P(original dialogue | culture_i) for ALL hypothesis countries in parallel."""

    try:
        from telemetry import get_active_collector, set_active_collector, clear_active_collector as _tel_clear
        _collector = get_active_collector()
    except ImportError:
        _collector = None

    def _evaluate_with_telemetry(country: str) -> tuple[str, float]:
        """Wrapper that propagates the telemetry collector into this sub-thread."""
        if _collector is not None:
            set_active_collector(_collector)
        try:
            return _evaluate_single_likelihood(
                country, current_input, base_url, api_key, model_name, temperature, seed, debug,
            )
        finally:
            if _collector is not None:
                _tel_clear()

    likelihoods: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(hypotheses)) as executor:
        futures = {
            executor.submit(_evaluate_with_telemetry, country): country
            for country in hypotheses
        }
        for future in as_completed(futures):
            country, likelihood = future.result()
            likelihoods[country] = likelihood
    return likelihoods

def _infer_direct_dimension_scores(
    perception: dict, current_input: str,
    base_url: str, api_key: str, model_name: str,
    temperature: float, seed: int, debug: bool,
) -> dict:
    perception_data = _format_perception_for_prompt(perception)
    prompt = DIRECT_DIMENSION_PROMPT.format(
        perception_data=perception_data,
        dialogue=current_input,
    )
    raw = call_llm(
        system_prompt="You are a precise JSON-outputting cultural analysis assistant.",
        user_message=prompt,
        base_url=base_url, api_key=api_key, model_name=model_name,
        temperature=max(0.0, min(temperature, 0.5)), seed=seed,
        _module="culture_direct_dims",
    )
    parsed = _extract_json(raw)
    out = {}
    for dim in DIMENSIONS:
        try:
            out[dim] = max(0, min(100, int(round(float(parsed.get(dim, 50))))))
        except (TypeError, ValueError):
            out[dim] = 50
    return out

def _blend_dimension_scores(country_scores: dict, direct_scores: dict, weight_direct: float = 0.38) -> dict:
    weight_country = 1.0 - weight_direct
    blended = {}
    for dim in DIMENSIONS:
        c = int(country_scores.get(dim, 50))
        d = int(direct_scores.get(dim, 50))
        blended[dim] = round(weight_country * c + weight_direct * d)
    return blended

def run_cultural_hypothesis(
    perception: dict,
    current_input: str,
    prior_cultural_state: Optional[dict] = None,
    n_hypotheses: int = 5,
    theta_high: float = 0.9,
    theta_low_filter: float = 0.03,
    theta_low_likelihood: float = 0.1,
    base_url: str = "",
    api_key: str = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
    seed: int = 42,
    debug: bool = False,
) -> tuple[dict, dict]:
    """
    Infer Hofstede cultural dimension scores using Bayesian hypothesis updating.

    (Public API and return schema unchanged — see original docstring.)
    """
    if debug:
        print(_box("CULTURAL HYPOTHESIS MODULE"))
        comm    = perception.get("cultural_cues", {}).get("communication_style", "—")
        emotion = perception.get("mental_state", {}).get("emotion", {})
        intent  = perception.get("mental_state", {}).get("immediate_intent", "—")
        social  = perception.get("cultural_cues", {}).get("social_tendencies", [])
        print(f"\n  {_c('Perception snapshot:', _BOLD)}")
        print(f"    Intent     : {_c(intent, _CYAN)}")
        print(f"    Emotion    : {_c(emotion.get('category','—'), _YELLOW)} "
              f"({emotion.get('intensity','—')})")
        print(f"    Comm style : {_c(comm, _GREEN)}")
        if social:
            print(f"    Social cues: {_c(', '.join(social[:3]), _DIM)}")
        if prior_cultural_state:
            print(f"\n  {_c('Prior state:', _BOLD)} "
                  f"{len(prior_cultural_state.get('hypotheses',{}))} hypotheses carried over")

    if prior_cultural_state and "hypotheses" in prior_cultural_state:
        hypotheses = dict(prior_cultural_state["hypotheses"])
        if debug:
            print(_section("Prior Hypotheses (reused)"))
            print(_prob_bars(hypotheses))
    else:
        if debug:
            print(_section("Generating Hypotheses"))
        hypotheses = _generate_hypotheses(
            perception, current_input, n_hypotheses, base_url, api_key, model_name, temperature, seed, debug
        )
        if debug:
            print(_section("Initial Prior Distribution"))
            print(_prob_bars(hypotheses))

    def _run_bayesian_update(hypotheses_in: dict, attempt: int = 1) -> tuple[dict, bool]:
        likelihoods = _evaluate_likelihoods(
            current_input, hypotheses_in, base_url, api_key, model_name, temperature, seed, debug
        )
        max_likelihood = max(likelihoods.values()) if likelihoods else 0.0

        if debug:
            print(_section(f"Likelihoods (attempt {attempt})"))
            for country, lk in sorted(likelihoods.items(), key=lambda x: -x[1]):
                bar = "█" * int(lk * 20) + "░" * (20 - int(lk * 20))
                color = _GREEN if lk > 0.6 else _YELLOW if lk > 0.3 else _DIM
                print(f"  {country:<24} {_c(bar, color)} {_c(f'{lk:.2f}', _BOLD)}")
            print(f"\n  {_c('Max likelihood:', _DIM)} {_c(f'{max_likelihood:.3f}', _BOLD)}", end="")
            if max_likelihood < theta_low_likelihood:
                print(f"  {_c('⚠ Below threshold — will regenerate', _RED)}")
            else:
                print()

        if max_likelihood < theta_low_likelihood:
            return {}, False

        posteriors = {c: likelihoods.get(c, 0.0) * p for c, p in hypotheses_in.items()}
        total = sum(posteriors.values())
        if total == 0:
            return {}, False
        return {c: p / total for c, p in posteriors.items()}, True

    posteriors, success = _run_bayesian_update(hypotheses, attempt=1)

    if not success:
        if debug:
            print(f"\n  {_c('↺ Regenerating hypotheses…', _YELLOW)}")
        hypotheses = _generate_hypotheses(
            perception, current_input, n_hypotheses, base_url, api_key, model_name, temperature, seed, debug
        )
        posteriors, success = _run_bayesian_update(hypotheses, attempt=2)
        if not success:
            if debug:
                print(f"\n  {_c('✗ Fallback to global default scores.', _RED)}")
            return GLOBAL_DEFAULT.copy(), {"hypotheses": {}, "scores": GLOBAL_DEFAULT.copy()}

    pruned = {c: p for c, p in posteriors.items() if p > theta_low_filter}
    if pruned:
        total = sum(pruned.values())
        pruned = {c: p / total for c, p in pruned.items()}

    if not pruned:
        if debug:
            print(f"\n  {_c('✗ All pruned — fallback to global default.', _RED)}")
        return GLOBAL_DEFAULT.copy(), {"hypotheses": {}, "scores": GLOBAL_DEFAULT.copy()}

    if debug:
        print(_section("Posterior Distribution (after pruning)"))
        print(_prob_bars(pruned))
        top = max(pruned, key=pruned.get)
        print(f"\n  {_c('Most likely:', _BOLD)} {_c(top, _GREEN, _BOLD)} "
              f"{_c(f'({pruned[top]*100:.1f}%)', _DIM)}")

    country_scores = {}
    for dim in DIMENSIONS:
        weighted_sum = sum(
            prob * HOFSTEDE_DB[country][dim]
            for country, prob in pruned.items()
        )
        country_scores[dim] = round(weighted_sum)

    try:
        direct_scores = _infer_direct_dimension_scores(
            perception, current_input, base_url, api_key, model_name, temperature, seed, debug
        )
    except Exception:
        direct_scores = GLOBAL_DEFAULT.copy()

    hofstede_scores = _blend_dimension_scores(country_scores, direct_scores, weight_direct=0.38)

    if debug:
        print(_section("Inferred Hofstede Scores"))
        print(f"  Country-mixture : {_hofstede_row(country_scores)}")
        print(f"  Direct-style    : {_hofstede_row(direct_scores)}")
        print(f"  Blended-final   : {_hofstede_row(hofstede_scores)}")
        for dim in DIMENSIONS:
            v = hofstede_scores[dim]
            color = _GREEN if v > 70 else _YELLOW if v > 40 else _BLUE
            bar = "█" * int(v / 120 * 30) + "░" * (30 - int(v / 120 * 30))
            print(f"  {dim:<5} {_c(bar, color)} {_c(str(v), _BOLD)}")
        print()

    cultural_state = {
        "hypotheses": pruned,
        "scores": hofstede_scores,
        "country_scores": country_scores,
        "direct_scores": direct_scores,
    }
    return hofstede_scores, cultural_state
