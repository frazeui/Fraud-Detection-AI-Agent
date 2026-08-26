# Fraud Detection Agent

A multi-agent AI system that analyzes financial transactions for fraud risk — combining structured transaction signals with multimodal document verification, a human-in-the-loop safety layer for high-risk decisions, and a systematic evaluation suite. Built with LangGraph and deployed as a REST API.

**v1 → v2 → v3:**
- **v1**: Single agent handling both analysis and decision-making.
- **v2**: Split into a Risk Analyst (evidence-gathering, has tool access) and a Decision Agent (reasoning-only, no tools) after finding this made failures traceable to a specific stage — which enabled evaluation-driven prompt fixes (62.5% → 75% accuracy, see below).
- **v3**: Added a Document Verification agent that analyzes an uploaded ID/document image (multimodal vision) alongside the transaction data, so the Decision Agent weighs both transaction-pattern risk and document authenticity before making a final call.

## Problem

Automated fraud detection systems face a core tension: they need to act fast, but fully autonomous blocking of transactions carries real financial and reputational risk if the system gets it wrong. This project explores a practical middle ground — a multi-agent system that reasons over transaction signals *and* supporting document evidence, explains its findings, and defers the final call on high-risk cases to a human reviewer rather than acting unilaterally.

## Architecture (v3 — Multimodal Multi-Agent)

```
                    Transaction Description + Document Image
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                 ▼
         ┌─────────────────────┐          ┌─────────────────────────┐
         │   Risk Analyst Agent  │          │ Document Verification    │
         │   (has tools)          │          │ Agent (multimodal vision) │
         │                        │          │                           │
         │ • check_amount_risk    │          │ • Extracts document type,│
         │ • check_velocity       │          │   name, ID number, dates │
         │ • check_location_      │          │ • Assesses authenticity  │
         │   mismatch             │          │ • Flags specific red     │
         │ (runs in parallel)     │          │   flags (tampering,      │
         │                        │          │   placeholder text, etc.)│
         └─────────────────────┘          └─────────────────────────┘
                    │                                 │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────┐
                    │      Decision Agent        │  No tool access. Reasons
                    │  (combines both findings)   │  only over what it's given.
                    └─────────────────────────┘  Classifies risk, recommends
                                    │              APPROVE / REVIEW / BLOCK.
                                    ▼
                    ┌─────────────────────┐
                    │   Human Review Node   │
                    │                       │
                    │  LOW/MEDIUM risk  ────┼──► Auto-approved
                    │  HIGH risk / BLOCK ───┼──► Graph PAUSES, waits for
                    │                       │    human decision, resumes
                    └─────────────────────┘
```

**Why the Decision Agent has no tool access.** Keeping it reasoning-only (fed structured findings from the other two agents, never raw user input or tool access) made two things possible: failures could be traced to a specific stage during evaluation, and — observed as a side effect during security testing — it closed off the most direct path for prompt injection, since injected instructions in the original transaction text never reach the agent making the final call.

## Tech Stack

- **LangGraph** — multi-agent orchestration, conditional routing, human-in-the-loop via `interrupt()`, native parallel tool execution
- **Groq (Qwen3.6-27B, vision-capable)** — LLM inference, including multimodal document analysis
- **SQLite (via `SqliteSaver`)** — checkpointing, so paused/in-progress reviews survive a restart
- **FastAPI** — REST API layer, including multipart file upload for document images (`/analyze_transactions_with_documents`, `/human_decision`)
- **Pydantic** — structured output for both the Decision Agent's verdict and the Document Verification agent's extraction, replacing free-text parsing
- **Tenacity** — retry logic tuned for the vision model's intermittent tool-calling/formatting failures
- **LangSmith** — tracing for debugging multi-agent runs (surfaced several bugs that manual print-debugging missed — see below)
- **ngrok** — public tunnel for testing the API from outside the notebook environment

## Evaluation Results

Built a small labeled evaluation suite (8 scenarios spanning single-signal and multi-signal cases, both users, and deliberately borderline amounts) to measure the Risk Analyst → Decision Agent pipeline's accuracy objectively instead of relying on manual testing — see `agent_evaluation.py`.

**First run: 62.5% accuracy.** Failures clustered around one pattern: the Decision Agent was inconsistently escalating single medium-risk signals to HIGH.

**Fix:** Added an explicit escalation rule to the Decision Agent's prompt (HIGH requires either one independently-HIGH signal or two-or-more MEDIUM signals).

**Second run: 75% accuracy**, zero false negatives across both runs — the system never classified a genuinely risky transaction as LOW.

## Security Testing

Tested prompt injection resistance with three attack variations embedded directly in the transaction description field (explicit system override, debug-mode roleplay, subtle bias injection). All three were correctly ignored — the agent classified each test case as HIGH risk based on the actual signals, regardless of the injected instruction. See the architecture note above on why this appears structural rather than incidental.

## What Went Wrong (and What It Taught Me)

Documenting these because they were more instructive than anything that worked on the first try:

- **Node/edge name mismatches** — a conditional edge returning a string that didn't match any key in the routing dictionary caused silent misrouting more than once. LangGraph doesn't always fail loudly on this.
- **A node returning early inside a conditional block** — the Risk Analyst node's LLM call and `return` were accidentally nested inside an `if` condition that was structurally always `False`, so the node silently returned `None` on every run — no exception, no obvious signal. Caught by checking `app.get_graph().nodes` and adding debug prints per node.
- **State key typos are silent** — returning `{"message": [...]}` instead of `{"messages": [...]}` doesn't raise an error; LangGraph just doesn't merge it into state, and the response quietly disappears.
- **A Python tuple where a string was expected** — building a "summary" string as a parenthesized, comma-separated sequence of f-strings (with stray `print()` calls mixed in) silently produced a tuple instead of a string, which only surfaced as a cryptic Pydantic validation error several layers downstream (`AIMessage.content` expects `str`, not `tuple`).
- **Structured-output field names must match exactly what the model naturally produces, or vice versa** — the Decision Agent's model consistently emitted `overall_risk`/`justification` regardless of prompt wording; matching the Pydantic schema's field names to the model's natural output was more reliable than trying to prompt the model into different field names.
- **Groq's `json_mode` requires the literal word "json" somewhere in the prompt** — a specific, easy-to-miss API requirement that produces an otherwise-unexplained 400 error.
- **Vision + structured-output + multi-agent is a genuinely harder combination for a preview-tier model** — the document verification step fails intermittently (~10-20% of calls) with an unparseable/empty generation, most likely from the model's verbose internal reasoning exhausting the token budget before it emits the structured response. Mitigated with a higher `max_tokens` budget, a "be concise" instruction in the prompt, and — critically — a try/except in the node itself that falls back to a "manual review required" message rather than letting the whole pipeline crash. This is treated as an accepted, monitored limitation of the current model tier rather than something fully "fixed."
- **LLM tool-call formatting is not 100% reliable in general** — occasionally produces a malformed, non-JSON tool call, or invents a tool call (like a nonexistent `"json"` tool) that isn't in the registered tool list. Traced back in one case to an instruction ("respond only with valid JSON") left over from an earlier prompt draft that no longer matched the actual output mechanism in use.
- **Free-tier rate limits (TPM) are a real constraint once multiple agents share a token budget** — giving every agent the same generous `max_tokens` caused the combined request to exceed the per-minute token limit; splitting into a smaller-budget LLM for simple tool-calling tasks and a larger-budget LLM for the vision/structured-output task resolved it.
- **Stale request bodies in API testing tools** — reusing a `thread_id` and copy-pasting a request body between two different endpoints in Swagger UI produced confusing 422s that looked like a code bug but weren't.

## Running It

```bash
pip install langgraph langchain-groq langgraph-checkpoint-sqlite tenacity fastapi uvicorn pillow
export GROQ_API_KEY="your-key-here"
```

Run the notebook/script — it starts a local API on port 8000. In a notebook environment (e.g. Colab), use `nest_asyncio` + `pyngrok` to expose it publicly.

## API Usage

**Submit a transaction with a supporting document for analysis:**
```bash
curl -X POST 'http://localhost:8000/analyze_transactions_with_documents' \
  -F 'thread_id=txn-001' \
  -F 'description=Analyze this transaction: user_id=user_101, amount=10000, transaction_country=Kuwait, transaction_count_last_hour=6' \
  -F 'document=@sample_id.jpg'
```

If risk is LOW/MEDIUM, the response includes `final_result` immediately. If HIGH, it returns `"status": "PENDING_REVIEW"` with the Decision Agent's assessment (which references both transaction and document findings).

**Submit a human decision for a pending review (same `thread_id`):**
```bash
curl -X POST 'http://localhost:8000/human_decision' \
  -H 'Content-Type: application/json' \
  -d '{"thread_id": "txn-001", "decision": "BLOCK"}'
```

