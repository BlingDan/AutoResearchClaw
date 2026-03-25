# Local Windows Minimal Setup

Use this when you only run ResearchClaw locally on Windows and want smaller working context.

## 1) Use the minimal config

Run with:

```powershell
uv run --python .venv researchclaw run --config local-win.minimal.yaml --topic "Your topic" --auto-approve
```

This profile keeps non-local features off by default:
- OpenClaw bridge off
- MetaClaw bridge off
- Figure agent off
- OpenCode beast mode off

## 2) Keep assistant context small

When working with an AI coding assistant, load these files first:

1. `README.md`
2. `local-win.minimal.yaml`
3. `researchclaw/config.py`
4. `researchclaw/cli.py`
5. Relevant file under `researchclaw/pipeline/`

Avoid loading unless needed:
- `docs/showcase/**`
- `image/**`
- `website/**`
- multilingual README files under `docs/README_*.md`
- integration docs (`docs/*integration*`)

## 3) Optional quality checks

```powershell
uv run --python .venv researchclaw validate --config local-win.minimal.yaml
uv run --python .venv researchclaw doctor --config local-win.minimal.yaml
```
