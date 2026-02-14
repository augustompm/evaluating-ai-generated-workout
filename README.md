# Evaluating the Safety of Local LLM-Generated Exercise Prescriptions Against Clinical Guidelines

Source code and data for the SEGAH 2026 paper.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Generate plans (requires [Ollama](https://ollama.com/) with models pulled):

```bash
python src/generate_plans.py
```

Evaluate plans against 15 clinical criteria:

```bash
python src/rule_engine.py
```

Analyze results:

```bash
python src/analyze.py
```

## License

MIT
