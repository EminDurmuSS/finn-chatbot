# Statement Copilot

AI-powered financial assistant with LangGraph orchestration. bunq-aligned, production-ready architecture.

## Features

- **Multi-Agent Architecture**: Orchestrator + specialized agents (Finance Analyst, Search Agent, Action Planner)
- **SQL-First Truth**: LLM generates parameters, deterministic code executes SQL
- **Hybrid Search**: BM25 + Dense embeddings for best of both worlds
- **Structured Outputs**: Anthropic's guaranteed JSON (beta)
- **Human-in-the-Loop**: Action confirmation with LangGraph `interrupt()`
- **Multi-Layered Guardrails**: Rule-based → LLM-based → PII masking

## Architecture

```
Input Guard -> Orchestrator -> [Finance Analyst | Search Agent | Action Planner] -> Synthesizer -> Output
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/statement-copilot.git
cd statement-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[all]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
from statement_copilot import chat

# Simple query
response = chat("How much did I spend this month?")
print(response["answer"])

# With session
response = chat(
    message="How much did I spend on groceries?",
    session_id="user-123",
    tenant_id="my-company",
)
```

### Full Copilot Instance

```python
from statement_copilot import StatementCopilot

copilot = StatementCopilot()

# Chat
response = copilot.chat(
    message="Create this month's report",
    session_id="session-1",
    tenant_id="tenant-1",
)

# If action needs confirmation
if response["needs_confirmation"]:
    # Show plan to user
    print(response["action_plan"]["human_plan"])

    # After user approves
    result = copilot.confirm_action(
        session_id="session-1",
        action_id=response["action_plan"]["action_id"],
        approved=True,
    )
```

### API Server

```bash
# Start server
python -m statement_copilot.api

# Or with uvicorn
uvicorn statement_copilot.api.main:app --reload
```

API Endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How much did I spend this month?"}'

# Confirm action
curl -X POST http://localhost:8000/chat/confirm \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action_id": "...", "approved": true}'
```

## Configuration

Environment variables (`.env`):

| Variable              | Description                        | Default                          |
| --------------------- | ---------------------------------- | -------------------------------- |
| `ANTHROPIC_API_KEY`   | Anthropic API key (required)       | -                                |
| `OPENROUTER_API_KEY`  | OpenRouter API key for embeddings  | -                                |
| `PINECONE_API_KEY`    | Pinecone API key for vector search | -                                |
| `PINECONE_INDEX_NAME` | Pinecone index name                | `statement-copilot-hybrid`       |
| `POSTGRES_URL`        | PostgreSQL URL for checkpointing   | `postgresql://localhost/copilot` |
| `DEBUG`               | Enable debug mode                  | `false`                          |
| `LOG_LEVEL`           | Logging level                      | `INFO`                           |

## Project Structure

```
statement_copilot/
- __init__.py          # Package entry point
- config.py            # Configuration management
- workflow.py          # LangGraph workflow
- core/
  - schemas.py         # Pydantic models
  - state.py           # LangGraph state
  - llm.py             # Anthropic client
  - database.py        # DuckDB + SQL builder
  - embeddings.py      # OpenRouter embeddings
  - vector_store.py    # Pinecone hybrid search
- agents/
  - prompts.py         # System prompts (English)
  - guardrails.py      # Input/output safety
  - orchestrator.py    # Central routing
  - finance_analyst.py # SQL analytics
  - search_agent.py    # Vector search
  - action_planner.py  # Export/reports
- api/
  - main.py            # FastAPI endpoints
- tests/
  - test_copilot.py     # Test suite
```

## Intent Types

| Intent      | Description                      | Agents          |
| ----------- | -------------------------------- | --------------- |
| `ANALYTICS` | Calculations, trends, breakdowns | Finance Analyst |
| `LOOKUP`    | Find specific transactions       | Search Agent    |
| `ACTION`    | Export, report, alert            | Action Planner  |
| `EXPLAIN`   | Explain previous results         | Synthesizer     |
| `CLARIFY`   | Need more info                   | Synthesizer     |
| `CHITCHAT`  | General conversation             | Synthesizer     |

## Metrics Available

- `sum_amount` - Total amount
- `count_tx` - Transaction count
- `avg_amount` - Average amount
- `top_merchants` - Top merchants by spending
- `top_categories` - Top categories
- `category_breakdown` - Category distribution
- `daily_trend` / `weekly_trend` / `monthly_trend`
- `monthly_comparison` - This vs last month
- `subscription_list` - Recurring payments
- `anomaly_detection` - Unusual transactions
- `cashflow_summary` - Income vs expenses

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=statement_copilot

# Run specific test
pytest tests/test_copilot.py::TestGuardrails -v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check .

# Run type checker
mypy statement_copilot

# Format code
ruff format .
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000
CMD ["uvicorn", "statement_copilot.api.main:app", "--host", "0.0.0.0"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: copilot
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

## License

MIT

## Contributing

PRs welcome! Please read CONTRIBUTING.md first.
