# Contributing to ASD Early Detection Pipeline

Thank you for your interest in contributing. This project sits at the intersection
of clinical screening research and production ML engineering — contributions in
both directions are welcome.

---

## What we welcome

| Area | Examples |
|------|---------|
| **Bug fixes** | Test failures, edge cases in PSI / SHAP, Gradio validation |
| **New evaluation metrics** | Calibration curves, DCA (decision curve analysis) |
| **Dataset support** | Adapters for ABIDE, SPARK, or other ASD datasets |
| **UI improvements** | Gradio layout, accessibility, mobile responsiveness |
| **Documentation** | Notebook clarity, docstring completeness |
| **Tests** | Additional pytest coverage for `src/` modules |

## What we do NOT accept via PR

- Changes to the novel composite risk scoring logic (IP-protected, see README)
- Model weight files (`.pkl`, `.onnx`) — these are distributed separately
- Full dataset files — only the 50-row sample belongs in the repo

---

## Development setup

```bash
git clone https://github.com/<your-username>/asd-early-detection.git
cd asd-early-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v        # all tests must pass before submitting PR
```

## Code style

- Formatter: `black` (line length 100)
- Linter: `flake8` (configured in CI)
- All public functions must have a NumPy-style docstring
- Type hints required for function signatures

```bash
black src/ app/ tests/
flake8 src/ app/ tests/ --max-line-length=100
```

## Submitting a pull request

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`)
2. Write tests for any new logic in `tests/`
3. Ensure `pytest tests/ -v` passes locally
4. Open a PR against `main` with a clear description

---

## Reporting issues

Use GitHub Issues. For security-related issues (e.g. accidental weight exposure),
email the maintainers directly rather than opening a public issue.
