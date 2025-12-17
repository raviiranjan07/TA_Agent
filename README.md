# TA_AGENT

Lightweight toolkit and pipeline to detect, backfill and maintain OHLCV candle data (1m and higher timeframes) for crypto pairs (default: BTCUSDT) using Binance (REST + archives) and TimescaleDB/Postgres. Includes a FastAPI backend (dashboard API) and orchestration scripts.

## Project overview & flow
- Database (TimescaleDB/Postgres) stores 1m OHLCV rows.
- Orchestrator (run.py) coordinates:
  - Tests (DB, Binance)
  - Historical backfill (archive downloads)
  - Gap-filling (REST API + archive routing)
  - Real-time collector
- Data acquisition:
  - Recent data via Binance REST in batched requests.
  - Historical data via Binance monthly ZIP archives (cached).
- Transformations: raw Binance klines → DB-ready tuples (timestamps normalized to UTC).
- Persistence: bulk insert into ohlcv_data managed by `scripts.database`.
- Serving: `dashboard.py` provides a FastAPI backend with REST endpoints and WebSocket for live updates. (A Dash UI is present but commented out.)

## Key files
- run.py — Master orchestrator and CLI-like menu.
- scripts/database.py — Database wrapper (connection pool, gap detection, insert helpers).
- scripts/spot/binance_api.py — Binance REST wrapper.
- dashboard.py — FastAPI backend (and an optional Dash app commented out).
- test.py — quick REST/test helper.
- data/archive_cache/ — cached Binance ZIP archives (created at runtime).
- logs/ — runtime logs (scripts create when needed).

## Requirements
- macOS (development tested)
- Python 3.8+
- PostgreSQL + TimescaleDB (tested with TimescaleDB 2.x)
- Recommended Python packages (install into a virtualenv):
  - python-binance
  - python-dotenv
  - requests
  - psycopg2-binary
  - fastapi, uvicorn
  - pandas (optional for dashboard)
  - pandas-ta (optional)
- Install: pip install -r requirements.txt (if present) or install packages individually.

## Environment variables (.env)
Create a `.env` file in project root with at least:
- BINANCE_API_KEY=
- BINANCE_API_SECRET=
- DB_HOST=localhost
- DB_PORT=5432
- DB_USER=your_user
- DB_PASSWORD=your_password
- DB_NAME=crypto_data
- PAIRS=BTCUSDT,ETHUSDT
- TIMEFRAME=1m
- PORT (optional for dashboards)

## Quickstart (recommended)
1. Create & activate venv (macOS / zsh):
   python3 -m venv .venv
   source .venv/bin/activate

2. Install deps:
   pip install -r requirements.txt
   (or pip install python-binance python-dotenv requests psycopg2-binary fastapi uvicorn)

3. Configure `.env` with DB and Binance credentials.

4. Run master orchestrator:
   python3 run.py

5. Run Dashboard API (FastAPI):
   python3 dashboard.py
   - Default: http://localhost:8000
   - Docs: http://localhost:8000/docs

   Or run with uvicorn:
   uvicorn dashboard:app --host 0.0.0.0 --port 8000 --reload

6. (Optional) Dash UI
   - The Dash app is included but commented out inside `dashboard.py`. Enable and run on a different port (e.g., 8050) if needed.

## REST endpoints (FastAPI)
- GET /api/health — health & DB status
- GET /api/candles — get OHLCV candles (supports aggregation via time_bucket)
- GET /api/stats — 24h stats
- GET /api/indicators — MA / Bollinger, RSI (computed in Python)
- GET /api/orderbook — buy/sell pressure summary
- GET /api/pairs — available trading pairs
- GET /api/timeframes — supported timeframes
- GET /api/database/stats — DB-level stats
- WebSocket `/ws/live` — broadcast latest 1m data periodically

## Troubleshooting & tips
- "Import ... could not be resolved" in VS Code / Pylance:
  - Ensure VS Code uses the `.venv` interpreter (Cmd+Shift+P → Python: Select Interpreter).
  - Add project root to python.analysis.extraPaths in `.vscode/settings.json`:
    {
      "python.analysis.extraPaths": ["./"]
    }
  - Or use explicit imports that match project layout (e.g., `from scripts.spot.binance_api import BinanceAPI`).

- Flask/port conflict:
  - If a service says "Port 5000 is in use", find/kill the process:
    lsof -iTCP:5000 -sTCP:LISTEN -n -P
    kill <PID>
  - Or start the server on a different port:
    PORT=5001 python3 dashboard.py
  - The FastAPI app uses port 8000 by default — pick ports carefully.

- Database connection:
  - The project expects a TimescaleDB-enabled Postgres. Ensure `scripts.database` configuration matches your DB (user, password, host, port).
  - Check migrations/schema: the table `ohlcv_data` must exist with expected columns.

- Timezones:
  - Backend returns ISO datetimes converted to IST (Asia/Kolkata). Internally the pipeline stores UTC 1m rows and aggregates them with time_bucket. Ensure your consumers expect IST or UTC accordingly.

- Rate limits & retries:
  - Binance API calls may be rate limited. Production runs should include exponential backoff / retry logic (some helpers exist in code — consider hardening).

## Development notes
- To run modules from project root, set PYTHONPATH=. or run with `python -m` from package root.
- To make imports stable, consider converting `scripts/` to a package (add __init__.py) or install the project in editable mode:
  pip install -e .

## Logging
- Logs are written under `logs/` (e.g., `logs/master.log`, `logs/fill_gaps.log`). Scripts should create the `logs/` directory at startup — ensure write permissions.

## Contributing
- Bug fixes, tests and small focused PRs welcome. Add unit tests for transform/parse functions and integration tests against a test DB.

## License
- Add a LICENSE file appropriate for your project (e.g., MIT).