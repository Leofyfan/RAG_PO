# RAG Poisoning for Social Media Event Summaries

This project implements the `plan.md` attack/defense pipeline using the ECNU OpenAI-compatible API models:

- Chat/generation/judge: `ecnu-plus` by default, optionally `ecnu-max`
- Embedding: `ecnu-embedding-small` (1024 dimensions)

The implementation stays dependency-light (`requests` only) and stores all data, cache, and results inside this project directory.

## Quick Start

```bash
export PYTHONPATH=src
export ECNU_API_KEY='your-key'
python -m rag_po.cli download-data
python -m rag_po.cli prepare-data --max-per-event-label 12
python -m rag_po.cli smoke
```

`smoke` runs a tiny real ECNU-backed experiment with `parallelism=1`. Full experiments use `run`, for example:

```bash
python -m rag_po.cli run \
  --events charliehebdo,sydneysiege,ferguson,germanwings-crash,ottawashooting \
  --attacks random,semantic,llm \
  --ratios 0,0.1,0.3,0.5 \
  --defenses D0,D1,D2,D3,D4,D_all \
  --parallelism 8 \
  --max-per-event-label -1 \
  --max-clean-docs 0 \
  --max-rumour-pool 0
```

## Main Modules

- `src/rag_po/data_prep`: PHEME download/parse/query building
- `src/rag_po/attack`: random, semantic, and LLM-generated poisoning
- `src/rag_po/defense`: semantic outlier, consistency, social rerank, critical prompt
- `src/rag_po/rag`: ECNU client, vector search, retriever, generator
- `src/rag_po/evaluation`: retrieval metrics, ECNU LLM-as-Judge, CSV flattening
- `src/rag_po/pipeline.py`: attack -> defense -> generation -> evaluation experiment loop

## Outputs

Each run writes to `experiments/results/<timestamp>/`:

- `config.json`: run configuration
- `<event>-<attack>-<ratio>-<defense>.json`: per-cell details
- `results.json`: all result rows
- `results.csv`: flat metrics for plotting
