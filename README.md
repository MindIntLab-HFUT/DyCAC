# Don' t Box Me In: Dynamic Cultural Adaptation and Cognitive Tracking for Social Understanding

This repository contains the four core modules of **DyCAC**, a training-free framework for dynamic cultural adaptation and continuous cognitive tracking in dialogue.

## Framework overview

DyCAC processes each dialogue turn through four stages:

1. **Perception** extracts objective facts, mental-state signals, cultural cues, and optional goal-progress signals.
2. **Memory Update** maintains explicit dialogue memory and Theory-of-Mind (ToM) state estimates.
3. **Cultural Hypothesis** updates a probabilistic culture space and estimates Hofstede-style cultural dimensions.
4. **Planning and Execution** uses the current memory and cultural profile to plan and generate the response.

The memory and cultural state are carried across turns, allowing the framework to adapt continuously during a conversation.

## Repository structure

```text
framework_codes/
├── perception.py             # Perception module
├── memory_update.py          # Memory and ToM update module
├── cultural_hypothesis.py    # Dynamic cultural inference module
├── planning_execution.py     # Response planning and generation module
└── llm_client.py             # Shared OpenAI-compatible API helper
README.md
```

`llm_client.py` is a runtime utility used by all four modules; it is not an additional framework module.

## Requirements

- Python 3.9 or later
- The `openai` Python package
- Access to an OpenAI-compatible chat-completions API

Install the dependency:

```bash
pip install openai
```

Set your API credentials:

```bash
export OPENAI_API_KEY="your-api-key"
```

If you use another OpenAI-compatible provider, pass its API base URL through `base_url` in the example.
