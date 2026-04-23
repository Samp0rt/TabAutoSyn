! [logo](assets/image.png)

# 📊 TabAutoSyn

**TabAutoSyn** is an **AutoML** framework for **tabular data synthesis**: it automates model choice, training, hyperparameter search, and quality-driven refinement end to end. It combines [Synthcity](https://github.com/vanderschaarlab/synthcity) generators, **genetic curation** of synthetic rows, **tail extension** of distributions, and (in the main pipeline) **LLM agents** on [OpenRouter](https://openrouter.ai/) to discover and repair logical/structural dependencies between columns. The stack is aimed at workflows where you need both statistical fidelity to real data and domain-consistent constraints.

## ✨ Features

- 🔌 **Synthcity plugins** for `ml`, `privacy`, and `universal` tasks when `model="task_specific"`.
- 🦙 **Local LLM generation** via the OpenAI-compatible **Ollama** API (`model="LLM"`): the default model tag in code is `gpt-oss:20b`.
- 🤖 **Async pipeline `TabAutoSyn.generate`**: synthesis → dependency discovery → `DependencyFixer` → tail correction → class alignment → genetic optimization; optional artifacts (CSV + Markdown) and **Langfuse** traces.
- 📈 **Metrics and Optuna** for quality evaluation and plugin hyperparameter search (see `tabautosyn/config.py`, `tabautosyn/metrics.py`, `tabautosyn/optimization.py`).

## 🏗️ Architecture

| Component | Role |
|-----------|------|
| `tabautosyn/automl/base.py` | Main entry point: `task_specific` / `LLM` modes, `generate`, `old_generate`, `run_generator`, curation and tail stages. |
| `tabautosyn/gen/` | Genetic algorithm to align synthetic data with the real distribution. |
| `tabautosyn/tail_extension/` | Merges tail synthetic rows into the main pool using divergence-based criteria. |
| `tabautosyn/agents/` | Prompts and dependency repair (`deps_reconstruction`). |
| `tabautosyn/utils/dataset_processor.py` | Dataset description and preprocessing helpers. |
| `curation/gen/` | Standalone GA curation implementation (see notebooks/scripts under `curation/`). |

**`generate` (primary method)** expects an in-memory `pandas.DataFrame`, a required `target_column` for classification workflows, and a **required** `OPENROUTER_API_KEY` (meta-models go through OpenRouter). On this path, synthesis runs over multiple Synthcity plugins in sequence—by default **`ctgan`**, **`ddpm`**, and **`dpgan`** (see the `plugins` list in `tabautosyn/automl/base.py`; edit there to change or extend models). Local **Ollama** is **not** used by `generate`; Ollama is used on the **`model="LLM"`** branch (`_generate_synthetics_llm`).

## 🎬 `generate` demo

![demo](assets/demo.gif)

## 📥 Clone the repository

```bash
git clone https://github.com/Samp0rt/TabAutoSyn.git
cd TabAutoSyn
```

Create a virtual environment if you like, then install dependencies — see [Installation](#-installation).

## 🔐 Environment variables

1. 📋 Copy the template:

   ```bash
   cp .env.example .env
   ```

2. 🔑 Fill in secrets in `.env`. Variable names and comments are documented in **[`.env.example`](.env.example)**.

**Minimum for `generate`:** `OPENROUTER_API_KEY`. Langfuse and fine-grained OpenRouter knobs are optional.

## 🦙 Ollama and gpt-oss

Behavior matches `tabautosyn/automl/base.py`: for **`TabAutoSyn(model="LLM")`**, `_generate_synthetics_llm` builds an OpenAI client with **`base_url="http://localhost:11434/v1"`** and **`api_key="ollama"`** — the standard OpenAI-compatible **Ollama** endpoint. The model id is passed as `plugin_name` (default **`gpt-oss:20b`**).

To match the code out of the box:

1. ⬇️ Install [Ollama](https://ollama.com/download) and ensure it listens on **`127.0.0.1:11434`** (default).
2. 📦 Pull the model (tag matches the [Ollama library](https://ollama.com/library/gpt-oss:20b)):

   ```bash
   ollama pull gpt-oss:20b
   ```

3. ✅ Smoke test:

   ```bash
   ollama run gpt-oss:20b "ping"
   curl http://localhost:11434/v1/models
   ```

For **20B**, plan for enough RAM/VRAM (often **≥16 GB** unified or GPU memory; exact needs depend on quantization and OS). For **`gpt-oss:120b`**, run `ollama pull gpt-oss:120b` and pass the matching `plugin_name` in `run_generator` / `_generate_synthetics_llm` call sites.

If Ollama runs on another host or port, **`base_url` is currently hard-coded** in `base.py`; use a tunnel to `localhost:11434` or change the source.

## 📦 Installation

Dependencies and supported Python versions are declared in **`pyproject.toml`** (`[project.dependencies]` and `[project.optional-dependencies]`). `pip install -e .` installs the package using those declarations.

**Supported Python:** `3.10`–`3.12` (see `requires-python` in `pyproject.toml`; `synthcity` 0.2.x expects a PyTorch stack that is not compatible with CPython 3.13+ yet).

### 💻 Editable install (usual dev setup)

From the repository root:

```bash
cd TabAutoSyn
pip install -e .
```

That installs the `[project.dependencies]` set and links `tabautosyn` into your environment so code changes apply immediately.

### 🧪 Optional dev tools (`pyproject.toml`)

For tests and linting:

```bash
pip install -e ".[dev]"
```

Installs `pytest` and `ruff` in addition to the core dependencies. Not required to run `TabAutoSyn` or `generate`.

### 🔧 Other tools

Any PEP 517–compatible installer can use `pyproject.toml`, for example **`uv pip install -e .`** or **`pdm install`** (with a PDM config) — same dependency semantics as `pip install -e .`.

## ⚡ Quick start

### 🚀 Full `generate` pipeline (recommended)

Set **`OPENROUTER_API_KEY`** in `.env` (see [`.env.example`](.env.example)), then run the example driver **[`examples/run_generate.py`](examples/run_generate.py)**:

```bash
python examples/run_generate.py \
  --input-csv path/to/train.csv \
  --target-column your_label_column \
  --output-dir examples/out \
  --save-pipeline-summary
```

`TabAutoSyn.generate` is invoked with `model="task_specific"` and `task="ml"` (see the script). With `--output-dir` set, the pipeline writes timestamped **`final_synthetic_*.csv`** and **`generate_summary_*.md`** under that directory (see `TabAutoSyn.generate` in `tabautosyn/automl/base.py`).

Useful flags (full list: `python examples/run_generate.py --help`):

| Flag | Role |
|------|------|
| `--user-df-info` | One-line dataset description; if omitted, `generate()` infers it. |
| `--separator` | CSV delimiter (default `,`). |
| `--optimization-trials` / `--params` | Optional Optuna HPO or pickled best params. |
| `--temperature`, `--max-tokens`, `--retries`, `--timeout` | OpenRouter agent call settings. |
| `--quiet` | Turn off verbose logging. |

### 🔌 Synthcity-only generation (`run_generator`)

```python
from tabautosyn import TabAutoSyn
import pandas as pd

df = pd.read_csv("train.csv")

syn = TabAutoSyn(model="task_specific", task="ml", verbose=True)
out = syn.run_generator("train.csv", n_samples=1000, target_column="target")
```

### 🦙 Local LLM via Ollama (`gpt-oss:20b`)

Ensure Ollama is running and `ollama pull gpt-oss:20b` has completed (see [Ollama and gpt-oss](#-ollama-and-gpt-oss)).

```python
from tabautosyn import TabAutoSyn

syn = TabAutoSyn(model="LLM", verbose=True)
df_syn = syn.run_generator(
    train_data_path="train.csv",
    n_samples=500,
    batch_size=10,
    target_column="your_label_column",
)
```

## 📁 Project layout (summary)

```
TabAutoSyn/
├── tabautosyn/
│   ├── automl/base.py      # TabAutoSyn, generate, LLM / task_specific paths
│   ├── gen/                # genetic algorithm
│   ├── tail_extension/     # distribution tails
│   ├── agents/             # LLM agents
│   ├── llm_generator.py    # row generation via chat model
│   ├── optimization.py     # Optuna
│   └── metrics.py
├── curation/gen/             # standalone GA curation module
├── examples/
│   └── run_generate.py       # CLI example for TabAutoSyn.generate()
├── .env.example              # environment variable template
├── pyproject.toml            # dependencies & build (single source of truth)
└── LICENSE
```

## 📊 Metrics

Metric names are grouped in `tabautosyn/config.py`: sanity checks, statistics, downstream model quality, synthetic detection, and privacy-oriented scores.

## 📄 License

BSD 3-Clause — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- 🔬 [Synthcity](https://github.com/vanderschaarlab/synthcity)
- 🎯 [Optuna](https://optuna.org/)
- 🧬 [DEAP](https://github.com/DEAP/deap)
- 🦙 [Ollama](https://ollama.com/) and **gpt-oss** models
