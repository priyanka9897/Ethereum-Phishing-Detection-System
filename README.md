# ChainGuard

Ethereum phishing detection with a FastAPI backend, a Streamlit dashboard, and a static HTML frontend.

## What it does

ChainGuard fetches Ethereum account activity from Etherscan and Alchemy, builds a 2-hop heterogeneous transaction graph, runs an HT-GNN-based inference pipeline, and blends the model output with heuristic signals to produce a phishing risk score, explanation, evidence, and feature importance.

The project also includes a small in-memory marketplace for community validation reports and a simple HTML dashboard for scanning addresses.

## Repository Layout

```
.
├── Backend/
│   ├── main.py
│   ├── dashboard_app.py
│   ├── config.py
│   ├── api/
│   │   ├── cache.py
│   │   ├── marketplace.py
│   │   └── validators.py
│   ├── pipeline/
│   │   ├── fetcher.py
│   │   └── graph_builder.py
│   ├── model/
│   │   ├── htgnn.py
│   │   └── inference.py
│   ├── data/
│   ├── model/
│   ├── requirements.txt
│   └── run_backend_and_dashboard.sh
├── Frontend/
│   └── index.html
├── nginx.conf
└── report/
```

## Prerequisites

- Python 3.11+
- An Alchemy API key
- An Etherscan API key
- Optional: Redis, if you want a persistent cache instead of the in-memory fallback

The code ships with default values in `Backend/config.py`, but you should override them in a local `.env` file for real use.

## Setup

```bash
cd Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to override configuration, create `Backend/.env` and set values like:

```env
ALCHEMY_API_KEY=...
ALCHEMY_MAINNET_URL=...
ALCHEMY_WS_URL=...
ETHERSCAN_API_KEY=...
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=./model/htgnn_phishing.pt
```

## Run the backend

From `Backend/`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or start the backend and Streamlit dashboard together:

```bash
./run_backend_and_dashboard.sh
```

The backend exposes:

- `GET /health`
- `POST /api/v1/analyze`
- `GET /api/v1/analyze/{address}`
- `GET /api/v1/subgraph/{address}`
- `POST /api/v1/fetch-and-save`
- `POST /api/v1/fetch-raw`
- `GET /api/v1/marketplace`
- `POST /api/v1/marketplace`
- `POST /api/v1/marketplace/{id}/vote`

Open `http://localhost:8000/docs` for the Swagger UI.

## Run the dashboard

The Streamlit app is in `Backend/dashboard_app.py`.

```bash
cd Backend
streamlit run dashboard_app.py
```

The dashboard talks to the backend API at `http://127.0.0.1:8000` by default. You can change the backend URL in the sidebar.

## Run the frontend

The static frontend lives in `Frontend/index.html`. Serve it locally:

```bash
cd Frontend
python3 -m http.server 3000
```

Then open `http://localhost:3000`.

## How the pipeline works

1. `Backend/pipeline/fetcher.py` collects balances, normal transactions, internal transactions, token transfers, approvals, and nearby neighbors.
2. `Backend/pipeline/graph_builder.py` converts the raw payload into a PyTorch Geometric `HeteroData` graph.
3. `Backend/model/htgnn.py` builds the HT-GNN model and loads `Backend/model/htgnn_phishing.pt` if it exists.
4. `Backend/model/inference.py` runs the model, extracts heuristic signals, blends the scores, and generates the explanation payload returned by the API.
5. `Backend/api/cache.py` caches recent analyses in Redis when available, otherwise it falls back to an in-memory cache.

If the checkpoint file is missing, the app still runs and falls back to heuristic-heavy scoring.

## Key behavior

- The analysis endpoint returns phishing score, confidence, verdict, risk tier, evidence, feature importance, suspicious neighbors, and a rendered subgraph.
- Marketplace reports are in-memory only and reset on restart.
- The backend limits Etherscan request bursts and retries transient failures.
- The frontend and dashboard expect the backend to be running on port `8000`.


## Troubleshooting

- If `http://localhost:3000` says the site can’t be reached, make sure the static server is still running in the `Frontend/` terminal.
- If the backend fails to start, check that your virtual environment is active and that the required Python packages are installed.
- If analysis fails, verify your Alchemy and Etherscan credentials and confirm the API can reach the network.
- A missing model checkpoint is not fatal; it just means inference will rely more on heuristic signals.
 Then : Bash -
 1: lsof -nP -iTCP:8000 -sTCP:LISTEN || true
 2: lsof -nP -iTCP:3000 -sTCP:LISTEN || true
 3: source venv/bin/activate
 4:from Backend : python -m uvicorn main:app --host 0.0.0.0 --port 8000
 5: curl -sS http://127.0.0.1:8000/health && curl -sS -I http://127.0.0.1:3000 | head -n 1
 


## Model and graph settings

The default graph and model settings are defined in `Backend/config.py`.

- Model path: `./model/htgnn_phishing.pt`
- Cache TTL: 3600 seconds
- Hop depth: 2
- Maximum neighbors: 30

The HT-GNN uses EOA and contract node types, with relation types for sends, calls, returns, and approvals.

## Development entry points

- Backend API: `Backend/main.py`
- Streamlit dashboard: `Backend/dashboard_app.py`
- Frontend UI: `Frontend/index.html`
- Graph construction: `Backend/pipeline/graph_builder.py`
- On-chain fetcher: `Backend/pipeline/fetcher.py`
- Inference logic: `Backend/model/inference.py`

## License

No license file is included in this repository snapshot.
