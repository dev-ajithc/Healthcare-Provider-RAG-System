# Healthcare Provider RAG System

> **Synthetic data only** — no real patient or provider information is used.

An AI-powered natural-language search system over 10,000 synthetic healthcare providers.
Ask questions like *"Top-rated cardiologists in California accepting Medicare"* and get
cited, source-grounded answers in under 5 seconds.

---

## Architecture

```
Streamlit UI  →  FastAPI  →  RAG Pipeline
                              ├── Sanitise + PII scrub
                              ├── Semantic cache (Redis)
                              ├── HyDE (optional)
                              ├── Embed (OpenAI text-embedding-3-small)
                              ├── Hybrid retrieve (pgvector + BM25 + RRF)
                              ├── Cross-encoder re-rank (MiniLM)
                              ├── Claude claude-sonnet-4-5 + GPT-4o-mini fallback
                              └── Citation integrity check
```

## Quick Start (Local)

### Prerequisites

- Docker + Docker Compose
- OpenAI API key
- Anthropic API key

### 1. Clone & configure

```bash
git clone https://github.com/your-org/healthcare-provider-rag.git
cd healthcare-provider-rag
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Start services

```bash
docker-compose up --build -d
```

### 3. Run migrations

```bash
docker-compose exec app alembic upgrade head
```

### 4. Generate & ingest synthetic data

```bash
# Generate 10k providers (~30s)
docker-compose exec app python scripts/generate_data.py

# Embed & ingest (~8 min on OpenAI free tier)
docker-compose exec app python scripts/ingest.py
```

### 5. Open the UI

- **Streamlit UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **Jaeger traces:** http://localhost:16686

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Natural language provider search |
| `GET` | `/health` | Liveness + readiness |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/session/{id}` | Retrieve session history |
| `DELETE` | `/session/{id}` | Clear session |
| `GET` | `/ingest/status` | Ingestion job status |
| `POST` | `/ingest/trigger` | Trigger re-ingestion (admin) |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "cardiologists in California accepting Medicare"}'
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | Yes | OpenAI embeddings key |
| `DATABASE_URL` | Yes | PostgreSQL async URL |
| `REDIS_URL` | Yes | Redis connection URL |
| `ADMIN_KEY` | Yes | Key for admin endpoints |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |
| `HYDE_ENABLED` | No | Enable HyDE retrieval (default: false) |
| `QUERY_KILLSWITCH` | No | Emergency stop all queries (default: false) |
| `MAX_DAILY_TOKENS` | No | Daily LLM token cap (default: 5000000) |
| `CACHE_TTL_SECONDS` | No | Semantic cache TTL (default: 86400) |

See `.env.example` for the full list.

---

## Development

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Run tests

```bash
# Unit + integration tests with coverage
pytest tests/unit tests/integration \
  --cov=app --cov-fail-under=80 -v

# All tests
pytest
```

### Code quality

```bash
flake8 app/ tests/ scripts/ frontend/
isort --check-only app/ tests/ scripts/ frontend/
mypy app/
pip-audit -r requirements.txt
```

### Run locally without Docker

```bash
# Start PG + Redis via Docker Compose only
docker-compose up -d db redis

# Apply migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload --port 8000

# Start UI (separate terminal)
streamlit run frontend/streamlit_app.py
```

---

## Deployment (Render)

1. Fork this repo on GitHub.
2. Create a new Render service from `render.yaml` (Blueprint deploy).
3. Set secrets in Render dashboard:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY`
4. Push to `main` — CI runs, then deploys automatically.
5. After first deploy, trigger ingestion via:

```bash
curl -X POST https://your-app.onrender.com/ingest/trigger \
  -H "X-Admin-Key: <your_admin_key>"
```

---

## Project Structure

```
healthcare-provider-rag/
├── app/
│   ├── api/           # FastAPI endpoints
│   ├── core/          # Config, security, middleware, telemetry
│   ├── db/            # SQLAlchemy models, session, queries
│   ├── rag/           # Pipeline, embeddings, LLM, cache, retrieval
│   └── schemas.py     # Pydantic request/response models
├── frontend/          # Streamlit UI + UI components
├── scripts/           # Data generation + ingestion
├── tests/             # Unit, integration, fixtures
├── alembic/           # Database migrations
├── docker/            # Dockerfile
├── .github/workflows/ # CI/CD
├── docker-compose.yml
├── render.yaml
├── SPEC.md            # Full design specification
└── .env.example
```

---

## Security

- All data is 100% synthetic — zero real PII.
- API keys stored in environment variables only.
- Rate limiting: 10 req/min per IP.
- Prompt injection detection on all queries.
- SQL injection prevention via SQLAlchemy parameterised queries.
- Dependency CVE scanning via `pip-audit` in CI.
- HTTPS enforced on Render; HSTS headers applied.

## License

MIT — see [LICENSE](LICENSE).
