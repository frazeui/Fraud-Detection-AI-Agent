# Fraud Detection Agent

An AI agent that analyzes financial transactions for fraud risk using multi-signal reasoning, with a human-in-the-loop safety layer for high-risk decisions. Built with LangGraph and deployed as a REST API.

## Problem

Automated fraud detection systems face a core tension: they need to act fast, but fully autonomous blocking of transactions carries real financial and reputational risk if the system gets it wrong. This project explores a practical middle ground — an AI agent that reasons over multiple risk signals and explains its findings, but defers the final call on high-risk cases to a human reviewer rather than acting unilaterally.

## Architecture

```
Transaction Input
       │
       ▼
┌─────────────┐
│  LLM Node   │──────► Decides which checks to run
└─────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│           Tool Node                  │
│  • check_amount_risk                 │
│  • check_velocity                    │
│  • check_location_mismatch           │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  LLM Node   │──────► Combines results into risk assessment
└─────────────┘
       │
       ▼
┌─────────────────────┐
│   Human Review Node  │
│                       │
│  LOW/MEDIUM risk  ────┼──► Auto-approved, no interruption
│                       │
│  HIGH risk / BLOCK ───┼──► Graph PAUSES (interrupt)
│                       │    Waits for human decision
│                       │    Resumes with final verdict
└─────────────────────┘
```

Built as a stateful graph (LangGraph) rather than a linear script, so the agent can loop between reasoning and tool calls an arbitrary number of times, and can be paused/resumed mid-execution for human input — something a simple `for` loop can't do cleanly.

## Tech Stack

- **LangGraph** — agent orchestration, conditional routing, human-in-the-loop via `interrupt()`
- **Groq (Llama 3.3 70B)** — LLM inference (free tier)
- **SQLite (via `SqliteSaver`)** — checkpointing, so paused/in-progress reviews survive a restart
- **FastAPI** — REST API layer (`/analyze_transactions`, `/human_decision`)
- **Tenacity** — retry logic for transient LLM tool-calling failures
- **ngrok** — public tunnel for testing the API from outside the notebook environment

## Key Design Decisions

**Why human-in-the-loop, not full automation.** In fraud detection specifically, a false positive (blocking a genuine customer) and a false negative (missing real fraud) are both costly. Routing only HIGH-risk / BLOCK-recommended cases to a human reviewer, while letting LOW/MEDIUM risk transactions clear automatically, balances speed with accountability.

**Why raw tool output is surfaced separately from the LLM's summary.** During development, the LLM occasionally paraphrased or misstated tool results in its final answer (e.g., reporting "Low Risk, matched" for a transaction that its own tool had flagged as a country mismatch). Since this is a domain where an incorrect summary has real consequences, the API and logs always expose the exact, unmodified tool output alongside the LLM's interpretation — the model's summary is treated as an aid, not a source of truth.

## What Went Wrong (and What It Taught Me)

Building this surfaced a number of real, non-obvious bugs — documenting them because they were more instructive than anything that worked on the first try:

- **Node/edge name mismatches** — a conditional edge returning `"tool_node"` while the routing dictionary only defined `"tools"` caused silent misrouting. LangGraph doesn't always fail loudly on this; it's worth explicitly logging state transitions during development.
- **`append()` vs `extend()`** — wrapping a system + user message pair in a list and `append`-ing it created a list-inside-a-list, which the API rejected with a cryptic "must be an object with property 'role'" error.
- **Persisted messages need to be plain dicts** — saving raw SDK response objects (`msg.model_dump()`) to JSON included fields (like `annotations`) that were valid in a *response* but rejected when replayed in a *request*. Fix: explicitly reconstruct only `role`, `content`, and `tool_calls` before persisting.
- **LLM tool-call formatting is not 100% reliable** — the same request occasionally produced a malformed, non-JSON tool call (`<function=check_amount_risk>{...}`) instead of a proper structured call. Wrapping the LLM call in a retry (via `tenacity`) resolved this in practice.
- **Variable name collisions matter in interactive notebooks** — naming both the LangGraph app and the FastAPI app `app` in the same session caused `'FastAPI' object has no attribute 'invoke'`, since the FastAPI instance silently shadowed the compiled graph.

## Running It

```bash
pip install langgraph langchain-groq langgraph-checkpoint-sqlite tenacity fastapi uvicorn
export GROQ_API_KEY="your-key-here"
```

Run the script — it will start a local API on port 8000. If running in a notebook environment (e.g. Colab), use `nest_asyncio` + `pyngrok` to expose it publicly; see `fraud_agent_api.py` for the full setup.

## API Usage

**Submit a transaction for analysis:**
```bash
curl -X POST 'http://localhost:8000/analyze_transactions' \
  -H 'Content-Type: application/json' \
  -d '{
    "thread_id": "txn-001",
    "description": "Analyze this transaction: user_id=user_101, amount=10000, transaction_country=Kuwait, transaction_count_last_hour=6"
  }'
```

If the risk is LOW/MEDIUM, the response includes a final result immediately. If HIGH, it returns `"status": "PENDING_REVIEW"` along with the AI's assessment.

**Submit a human decision for a pending review:**
```bash
curl -X POST 'http://localhost:8000/human_decision' \
  -H 'Content-Type: application/json' \
  -d '{
    "thread_id": "txn-001",
    "decision": "BLOCK"
  }'
```

