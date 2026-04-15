# Quant Multi-Agent Alpha Research

This project combines:

- Alpha factor retrieval (RAG over a CSV knowledge base)
- Genetic programming (GP) for alpha expression search
- QSTrader backtesting
- A LangGraph-based multi-agent workflow
- A FastAPI endpoint for evaluation

## Project Layout

- `main.py`: FastAPI app exposing `/health` and `/evaluate`
- `agent.py`: LangGraph multi-agent workflow orchestration
- `alpha_rag_chroma.py`: CSV ingestion and query against ChromaDB
- `gp.py`: Genetic programming search for alpha expressions
- `backtest/executor.py`: Backtest runners (`Template` and `GpTemplate`)
- `stock_data.py`: Download and store market data as CSV
- `Alphas/`: Alpha model implementations
- `alpha_factors.csv`: RAG source dataset
- `chroma_db/`: Local Chroma persistence

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. (Optional) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Set Gemini API key

Use an environment variable:

```bash
export GEMINI_API_KEY="your_api_key"
```

Note: Some scripts currently include hardcoded API key fields. Environment variables are safer and recommended.

### 3. Prepare RAG vector store

Ingest `alpha_factors.csv` into ChromaDB:

```bash
python alpha_rag_chroma.py ingest --csv alpha_factors.csv --db ./chroma_db --collection alpha_factors
```

Test query:

```bash
python alpha_rag_chroma.py query --query "momentum alpha with volume" -k 5 --db ./chroma_db --collection alpha_factors
```

### 4. Run the multi-agent workflow (CLI)

```bash
python agent.py --message "find a robust alpha and evaluate it"
```

Useful options:

```bash
python agent.py \
  --message "optimize gp fitness function" \
  --top-k 3 \
  --db ./chroma_db \
  --collection alpha_factors \
  --gp-npop 30 \
  --gp-seed 42 \
  --gp-crossover 0.4 \
  --gp-mutation 0.4 \
  --gp-fitness-function pearson_fitness
```

### 5. Run GP standalone

```bash
python gp.py --message "search alpha" --npop 20 --generations 5 --fitness-function pearson_fitness
```

With backtest after GP:

```bash
python gp.py --message "search alpha" --run-backtest
```

### 6. Run FastAPI server

Install ASGI server if needed:

```bash
pip install uvicorn
```

Start API:

```bash
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Evaluate query:

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"find an alpha for Taiwan market and evaluate risk"}'
```

## Data Notes

- Backtest modules auto-check local CSV data under `./stock_data/`.
- Missing files may be downloaded via `yfinance`.
- GP backtest currently uses `0050_10year` data.

## Troubleshooting

- Chroma/Gemini errors:
  - Confirm `GEMINI_API_KEY` is set and valid.
  - Ensure `chroma_db` path is writable.
- Backtest import errors:
  - Verify `qstrader` installed successfully in your environment.
- No API report returned:
  - Check logs from `agent.py` path and validate external API access.

## Development Tips

- Rebuild RAG DB after changing `alpha_factors.csv`.
- Tune GP via `--gp-npop`, `--gp-crossover`, and `--gp-mutation`.
- Keep generated/backtest artifacts out of version control if needed.
