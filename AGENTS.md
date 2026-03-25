# Agent Guidance (Local-Windows Minimal Mode)

This repository contains optional integrations (OpenClaw bridge, showcase docs, website assets).
For local Windows research runs, use a minimal context by default.

## Default Context Scope

Load these first:

1. `README.md`
2. `local-win.minimal.yaml`
3. `researchclaw/config.py`
4. `researchclaw/cli.py`
5. `researchclaw/pipeline/` (only files relevant to the current stage)

## Skip Unless Explicitly Requested

- `docs/showcase/**`
- `image/**`
- `website/**`
- `docs/README_*.md` (non-English variants)
- `docs/*integration*` (unless user asks integration questions)
- OpenClaw/MetaClaw related docs and examples

## Local-Only Assumptions

- OS: Windows
- Run mode: local (`experiment.mode: sandbox`)
- No OpenClaw dependency required
- No remote/ssh mode required

## Execution Preference

Use `uv` and the local profile:

```powershell
uv run --python .venv researchclaw run --config local-win.minimal.yaml --topic "..."
```
