import base64
import logging
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.tools.base import BaseTool

load_dotenv()

from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from google import genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langsmith import Client
from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from prometheus_client import CollectorRegistry, make_asgi_app
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from redis_client import client as redis_client

# Observability -> OpenTelemetry,Prometheus ---------------


resource = Resource.create(
    {
        "service.name": "fraud-detection-agent v3",
        "service.version": "3.0.0",
        "deployment.environment": "production",
    }
)


tracer = trace.get_tracer("fraud-detection-agent v3")

# Prometheus
promethus_registry = CollectorRegistry()
reader = PrometheusMetricReader(registry=promethus_registry)
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])

metrics.set_meter_provider(meter_provider)

meter = metrics.get_meter("fraud-detection-agent v3")

request_counter = meter.create_counter(
    "fraud_detection_requests",
    description="Counts the number of requests to the fraud detection agent",
)

request_duration = meter.create_histogram(
    "fraud_detection_request_duration",
    description="Measures the duration of requests to the fraud detection agent",
)

error_counter = meter.create_counter(
    "fraud_detection_errors",
    description="Counts the number of errors in the fraud detection agent",
)

decision_counter = meter.create_counter(
    "fraud_detection_decisions",
    description="Counts the number of decisions made by the fraud detection agent",
)

recommendation_counter = meter.create_counter(
    "fraud_detection_recommendations",
    description="Counts the number of recommendations made by the fraud detection agent",
)

llm_duration = meter.create_histogram(
    "fraud_detection_llm_duration",
    description="Measures the duration of LLM calls in the fraud detection agent",
)


# Audit Logging -------------------
logger = logging.getLogger("fraud-detection-agent v3")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Langsmith setup ------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = "fraud-detection-MultiAgent"

langsmith_key = os.environ.get("LANGSMITH_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key

logging.basicConfig(level=logging.INFO)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_VISION_MODEL = "gemini-3.5-flash-lite"

USER_PROFILES = {
    "user_101": {"home_country": "UAE", "avg_transaction": 500},
    "user_102": {"home_country": "Pakistan", "avg_transaction": 200},
}

# -----------------------------------------------------------


@tool(
    description="check the transaction amount that either it's in normal range or not "
)
def check_amount_risk(amount: float, user_id: str) -> str:
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return "Error: User profile not found"
    avg = profile["avg_transaction"]
    if amount > avg * 10:  # type: ignore
        return f"HIGH RISK: Amount ${amount} is {round(amount / avg, 1)}x higher than user's average (${avg})"  # type: ignore
    elif amount > avg * 3:  # type: ignore
        return f"MEDIUM RISK: Amount ${amount} is notably higher than average (${avg})"
    return f"LOW RISK: Amount ${amount} is within normal range (avg: ${avg})"


@tool
def check_velocity(transaction_count_last_hour: int) -> str:
    """Checks how many transactions occurred in the last hour."""
    if transaction_count_last_hour >= 5:
        return f"HIGH RISK: {transaction_count_last_hour} transactions in the last hour - unusual velocity"
    elif transaction_count_last_hour >= 3:
        return (
            f"MEDIUM RISK: {transaction_count_last_hour} transactions in the last hour"
        )
    return f"LOW RISK: {transaction_count_last_hour} transaction(s) in the last hour - normal"


@tool(
    description="check the location of the transaction against the user's home country"
)
def check_location_mismatch(user_id: str, transaction_country: str) -> str:
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return "Error: User profile not found"
    home = profile["home_country"]
    if home.lower() != transaction_country.lower(): #type: ignore
        return f"MEDIUM RISK: Transaction from {transaction_country}, but user's home country is {home}"
    return (
        f"LOW RISK: Transaction location ({transaction_country}) matches home country"
    )


risk_tools = [check_amount_risk, check_velocity, check_location_mismatch]

from concurrent.futures import Future, ThreadPoolExecutor


def run_risk_tools_parallel(
    user_id: str, amount: float, transaction_count_last_hour: int, country: str
):
    with ThreadPoolExecutor() as executor:
        futures: list[Future[str]] = [
            executor.submit(check_amount_risk, amount, user_id),  # type: ignore
            executor.submit(check_velocity, transaction_count_last_hour),  # type: ignore
            executor.submit(check_location_mismatch, user_id, country),  # type: ignore
        ]
        return [f.result() for f in futures]


class Risk_Assessment(BaseModel):
    overall_risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Overall risk classification"
    )
    recommendation: Literal["APPROVE", "REVIEW", "BLOCK"] = Field(
        description="Action to take"
    )
    justification: str = Field(description="Reasoning referencing specific findings")


class Document_Extraction_Result(BaseModel):
    document_type: str = Field(
        description="Type of document, e.g. Passport, Driver License, Bank Statement"
    )
    name: str | None = Field(
        default=None, description="Full name found on the document, if visible"
    )
    id_number: str | None = Field(
        default=None, description="ID/document number, if visible"
    )
    date_of_birth: str | None = Field(
        default=None, description="Date of birth, if visible"
    )
    appears_authentic: Literal["yes", "no"] = Field(
        description="Overall authenticity assessment"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Signs of any digital editing or tampering, if any",
    )
    font_consistency: Literal["consistent", "inconsistent", "cannot_determine"] = Field(
        description="Whether fonts/styles appear consistent"
    )
    tampering_indicators: list[str] = Field(
        description="Signs of digital editing or tampering, if any"
    )
    confidence_level: Literal["Low", "Medium", "High"] = Field(
        description="Confidence in this assessment"
    )


from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult


class OpenRouterLLM(BaseChatModel):
    client: object
    model: str

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        formatted_message = []
        for m in messages:
            role = getattr(m, "role", "user")
            safe_content = " "
            if isinstance(m.content, list):
                formatted_message.append({"role": role, "content": m.content})
            else:
                safe_content = str(m.content)
            formatted_message.append({"role": role, "content": safe_content})
        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_message, **kwargs
        )
        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def bind_tools(self, tools: list[BaseTool], **kwargs) -> "OpenRouterLLM":  # type: ignore
        """
        this is the function that binds the tools to the llm and returns a new llm instance with the tools bound
        """
        return self


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    document_image: str | None


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Lighter weight vision models for document verification


OPENROUTER_API_KEY: str | None = os.environ.get("OPEN_ROUTER_API_KEY")

llm_risk = ChatOpenAI(
    api_key=OPENROUTER_API_KEY, #type: ignore
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3.1-8b-instruct",
)


risk_analyst_llm = llm_risk.bind_tools(risk_tools)
decision_llm = llm_risk
structured_decision_llm = llm_risk.with_structured_output(Risk_Assessment)

# deep weight vision model for document verification
llm_vision = ChatGroq(model="qwen/qwen3.8-27b", api_key=GROQ_API_KEY)  # type: ignore
document_verification_llm = llm_vision.with_structured_output(
    Document_Extraction_Result
)


RISK_ANALYST_PROMPT = """You are a Risk Analyst Agent. Your only job is to run risk checks
on a transaction using your tools - you do not make final decisions or recommendations.

Rules:
1. Always run ALL THREE tools: check_amount_risk, check_velocity, check_location_mismatch.
2. Call them together in one turn, not one at a time.
3. After getting all results, summarize each finding factually - quote exact tool output, do not hallucinate.
4. Respond with PLAIN TEXT only after the tools return their results. Do NOT invent or call
   any additional tool (like 'json' or any other name) to format your response.
5. Do not give a final risk classification or recommendation - that is the Decision Agent's job.
6. End your summary clearly so the next agent can read it easily."""

DECISION_AGENT_PROMPT = """You are Decision Agent. Your job is to review the Risk Analyst's
findings and respond with a JSON object with exactly these three fields: overall_risk, recommendation, justification.

Critical: Keep the justification concise - 2-3 sentences maximum.

overall_risk must be one of: LOW, MEDIUM, HIGH
recommendation must be one of: APPROVE, REVIEW, BLOCK

Classification Rules - apply these to the actual findings given to you, do not use a default value:
- HIGH: at least ONE signal (amount, velocity, or location) is independently HIGH RISK, OR two-or-more signals are MEDIUM RISK.
- MEDIUM: exactly ONE signal is MEDIUM RISK and no signal is HIGH RISK.
- LOW: ALL signals are LOW RISK.

EXAMPLE 1:
Findings: "Amount: HIGH RISK (20x average). Velocity: LOW RISK. Location: LOW RISK."
Output: overall_risk=HIGH, recommendation=BLOCK

EXAMPLE 2:
Findings: "Amount: MEDIUM RISK. Location: MEDIUM RISK. Velocity: LOW RISK."
Output: overall_risk=HIGH, recommendation=BLOCK (two MEDIUM signals escalate to HIGH)

EXAMPLE 3:
Findings: "Amount: LOW RISK. Velocity: LOW RISK. Location: LOW RISK."
Output: overall_risk=LOW, recommendation=APPROVE

Recommendation mapping: LOW->APPROVE, MEDIUM->REVIEW, HIGH->BLOCK


Additional Rules:

1. Justification must reference the specific findings from the Risk Analyst - never invent new data.

2. When referencing the amount finding, use the exact multiplier format
   (e.g., '4.0x higher'), never percentage.

3. Use exactly these fields:
   overall_risk
   recommendation
   justification

Return the answer using the required structured schema.
Do not add additional fields.
"""

DOCUMENT_VERIFICATION_PROMPT = """
Examine the provided document image.

Extract the actual information visible in the document.

Return the result using the required structured schema.

Rules:

- document_type: identify the actual document type.
- name: extract the actual visible name, otherwise null.
- id_number: extract the actual visible document number, otherwise null.
- date_of_birth: extract the actual visible date of birth, otherwise null.
- appears_authentic: choose "yes" or "no".
- red_flags: list up to 2 specific visual red flags.
- font_consistency: choose "consistent", "inconsistent", or "cannot_determine".
- tampering_indicators: list up to 2 specific signs of editing or tampering.
- confidence_level: choose "Low", "Medium", or "High".

Important:

- Use actual values from the image.
- Never output placeholder values such as "string", "string or null", or "example".
- red_flags must be a list of strings.
- tampering_indicators must be a list of strings.
- If there are no red flags, return an empty list.
- If there are no tampering indicators, return an empty list.
- Do not invent information that cannot be seen.
"""

LIGHTWEIGHT_VISION_PROMPT = """You are analyzing a document image. 

Output ONLY a single JSON object with the ACTUAL values you observe in the image. 
Do not explain, do not write code, do not describe the schema - ONLY output the filled JSON.

Required JSON keys and their ACTUAL extracted values:
- document_type: the real document type you see (e.g. "Driver License")
- name: the real name visible, or null
- id_number: the real ID number visible, or null
- date_of_birth: the real date visible, or null
- appears_authentic: "yes" or "no" (your actual assessment)
- red_flags: list of actual red flags you see, max 2, or []
- font_consistency: "consistent", "inconsistent", or "cannot_determine"
- tampering_indicators: list of actual signs, max 2, or []
- confidence_level: "Low", "Medium", or "High"

Output format example (fill with REAL data, this is just showing the structure):
{"document_type": "Driver License", "name": "John Smith", "id_number": "ABC123", "date_of_birth": "1990-01-01", "appears_authentic": "yes", "red_flags": [], "font_consistency": "consistent", "tampering_indicators": [], "confidence_level": "High"}

Now output the JSON for the actual image shown:"""


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(2))
def risk_analyst_node(state: AgentState):
    with tracer.start_as_current_span("risk_analyst") as span:
        span.set_attribute("llm_provider", "openrouter")
        span.set_attribute("llm.model", llm_risk.model)
        span.set_attribute("operation", "risk_analysis")

        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=RISK_ANALYST_PROMPT)] + messages

        llm_start = time.time()
        response = risk_analyst_llm.invoke(messages)

        llm_duration.record(
            time.time() - llm_start,
            {"model": llm_risk.model, "operation": "Risk Analysis"},
        )
        return {"messages": [response]}


risk_tool_node = ToolNode(risk_tools)


def should_continue_risk_analysis(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "risk_tools"
    return "document_verification"


import json


def extract_document_fields_lightweight(
    base64_image: str,
) -> Document_Extraction_Result | None:

    with tracer.start_as_current_span("extract_document_fields_lightweight") as span:
        span.set_attribute("llm_provider", "google")
        span.set_attribute("llm.model", GEMINI_VISION_MODEL)
        span.set_attribute("operation", "document_extraction")
        span.set_attribute("vision.state", "lightweight")

        try:
            vision_start = time.time()

            response = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[ #type: ignore
                    {
                        "text": LIGHTWEIGHT_VISION_PROMPT,
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
                config={
                    "response_mime_type": "application/json",
                },
            )

            llm_duration.record(
                time.time() - vision_start,
                {
                    "model": GEMINI_VISION_MODEL,
                    "operation": "document extraction with lightweight model",
                },
            )

        except RuntimeError as e:
            error_counter.add(1)

            logger.error(f"[Lightweight Vision] Call failed: {type(e).__name__}: {e}")

            span.record_exception(e)

            return None

        span.set_attribute(
            "llm.response_received",
            response is not None,
        )

        if not response or not response.text:
            error_counter.add(1)
            logger.error("[Lightweight Vision] Empty response from Gemini")
            return None

        try:
            parsed = json.loads(response.text)

        except json.JSONDecodeError as e:
            error_counter.add(1)

            logger.error(f"[Lightweight Vision] JSON parsing failed: {e}")

            span.record_exception(e)

            return None

        try:
            return Document_Extraction_Result.model_validate(parsed)

        except RuntimeError as e:
            error_counter.add(1)

            logger.error(f"[Lightweight Vision] Schema validation failed: {e}")

            span.record_exception(e)

            return None


def extract_document_fields_deep_reasoning(
    base64_image: str,
) -> Document_Extraction_Result:
    with tracer.start_as_current_span("extract_document_fields_deep_reasoning") as span:
        span.set_attribute("llm_provider", "groq")
        span.set_attribute("llm.model", "qwen/qwen3.8-27b")
        span.set_attribute("operation", "document_verification")
        span.set_attribute("vision_stage", "deep_reasoning")

        message = HumanMessage(
            content=[
                {"type": "text", "text": DOCUMENT_VERIFICATION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        try:
            vision_time = time.time()
            result: Document_Extraction_Result = document_verification_llm.invoke(
                [message]  # type: ignore
            )
            llm_duration.record(
                time.time() - vision_time,
                {
                    "model": "qwen/qwen3.8-27b",
                    "operation": "extrat document field with deepweight model",
                },
            )
            return (
                result
                if isinstance(result, Document_Extraction_Result)
                else Document_Extraction_Result.model_validate(result)
            )

        except RuntimeError as e:
            error_counter.add(1)
            print(f"[Deep Reasoning Vision] Model failed: {e}")
            return None  # type: ignore


def document_verification_node(state: AgentState):
    with tracer.start_as_current_span("verify document") as span:
        span.set_attribute("operation", "document_verification")
        span.set_attribute("llm_provider", bool(state.get("document_image")))

        base64_image = state.get("document_image")

        if not base64_image:
            return {
                "messages": [
                    AIMessage(
                        content="[Document Verification] No document provided - skipping document check."
                    )
                ]
            }

        try:
            extraction = extract_document_fields_lightweight(base64_image=base64_image)

            if extraction:
                span.set_attribute("vision.state", "lightweight")
                span.set_attribute("document.authentic", extraction.appears_authentic)
                span.set_attribute(
                    "document.confidence_level", extraction.confidence_level
                )
                span.set_attribute("vision.escalated", False)

            deep_extraction = extract_document_fields_deep_reasoning(
                base64_image=base64_image
            )

            if not extraction:
                summary_text = (
                    "[Document Verification]\n"
                    "No extraction result available. Manual review required."
                )
            else:
                if (
                    extraction.confidence_level == "Low"
                    or extraction.appears_authentic in ["no", False]
                ):
                    print(
                        "[Document Verification] Low confidence/Risk detected. Escalating to deep reasoning model..."
                    )

                    if deep_extraction:
                        span.set_attribute("vision.state", "deep_reasoning")
                        span.set_attribute(
                            "document.authentic", deep_extraction.appears_authentic
                        )
                        span.set_attribute(
                            "document.confidence_level",
                            deep_extraction.confidence_level,
                        )
                        span.set_attribute("vision.escalated", True)

                summary_text = (
                    f"[Document Verification]\n"
                    f"Document Type: {getattr(deep_extraction, 'document_type', 'N/A')}\n"
                    f"Appears Authentic: {getattr(deep_extraction, 'appears_authentic', 'N/A')}\n"
                    f"Red Flags: {getattr(deep_extraction, 'red_flags', 'N/A')}\n"
                    f"Font Consistency: {getattr(deep_extraction, 'font_consistency', 'N/A')}\n"
                    f"Confidence Level: {getattr(deep_extraction, 'confidence_level', 'N/A')}"
                )

        except RuntimeError as e:
            error_counter.add(1)
            print(f"[Document Verification Error] {type(e).__name__}: {e}")
            summary_text = (
                "[Document Verification]\n"
                "Automated document verification failed due to a technical issue. "
                "This transaction requires MANUAL document review before approval."
            )

        return {"messages": [AIMessage(content=summary_text)]}


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(2))
def decision_agent_node(state: AgentState):
    with tracer.start_as_current_span("decision_agent") as span:
        span.set_attribute("operation", "decision_agent")
        span.set_attribute("llm_provider", "openrouter")
        span.set_attribute("llm.model", decision_llm.model)

        risk_findings = None
        document_findings = None

        for m in reversed(state["messages"]):
            if (
                isinstance(m, AIMessage)
                and m.content
                and "[Document Verification]" in m.content
                and document_findings is None
            ):
                document_findings = m.content
            elif (
                isinstance(m, AIMessage)
                and not getattr(m, "tool_calls", None)
                and m.content
                and risk_findings is None
                and "[Document Verification]" not in m.content
            ):
                risk_findings = m.content

        decision_input = [
            SystemMessage(content=DECISION_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Transaction Risk Findings:\n{risk_findings}\n\n"
                    f"Document Verification Findings:\n{document_findings or 'None provided'}\n\n"
                    f"Provide your final risk classification and recommendation, considering BOTH sources."
                )
            ),
        ]

        try:
            decision_time = time.time()
            llm_duration.record(
                time.time() - decision_time,
                {"model": decision_llm.model, "operation": "Fraud Detecting"},
            )

            response = decision_llm.invoke(decision_input)

            structure_result = (
                response
                if isinstance(response, Risk_Assessment)
                else Risk_Assessment.model_validate(response)
            )

            logger.info(f"[Debug] Decision raw response: {response.content!r}")

            span.set_attribute("decision.overall_risk", structure_result.overall_risk)
            span.set_attribute(
                "decision.recommendation", structure_result.recommendation
            )

        except json.JSONDecodeError as e:
            error_counter.add(1)
            span.record_exception(e)
            span.set_attribute("decision_agent_faied", True)

            logger.error(f"[Decision Agent] Parse failed: {e}")

            structure_result = Risk_Assessment(
                overall_risk="HIGH",
                recommendation="BLOCK",
                justification="Automated decision-parsing failed — manual review required.",
            )

        span.set_attribute("decision_agent_faied", False)
        formatted_text = (
            f"Overall Risk: {structure_result.overall_risk}\n"
            f"Recommendation: {structure_result.recommendation}\n"
            f"Justification: {structure_result.justification}"
        )
        response_msg = AIMessage(content=formatted_text)
        decision_counter.add(1, {"overall_risk": structure_result.overall_risk})
        return {"messages": [response_msg]}


def human_review_node(state: AgentState):
    last_message = state["messages"][-1]
    assessment_text = last_message.content
    needs_review = (
        "HIGH" in assessment_text.upper() or "BLOCK" in assessment_text.upper()
    )

    if not needs_review:
        return {"messages": []}

    human_decision = interrupt(
        {
            "question": "Decision Agent flagged this as HIGH risk / BLOCK. Please confirm.",
            "decision_agent_assessment": assessment_text,
        }
    )
    confirmation_message = HumanMessage(content=f"[HUMAN REVIEW]: {human_decision}")
    return {"messages": [confirmation_message]}


graph = StateGraph(AgentState)
graph.add_node("risk_analyst", risk_analyst_node)
graph.add_node("risk_tools", risk_tool_node)
graph.add_node("document_verification", document_verification_node)
graph.add_node("decision_agent", decision_agent_node)
graph.add_node("human_review", human_review_node)

graph.set_entry_point("risk_analyst")
graph.add_conditional_edges(
    "risk_analyst",
    should_continue_risk_analysis,
    {"risk_tools": "risk_tools", "document_verification": "document_verification"},
)
graph.add_edge("risk_tools", "risk_analyst")
graph.add_edge("document_verification", "decision_agent")
graph.add_edge("decision_agent", "human_review")
graph.add_edge("human_review", END)


class PatchedRedisSaver(RedisSaver):
    def _get_latest_checkpoint_document(self, thread_id, checkpoint_ns):
        checkpoint_key = f"{thread_id}:{checkpoint_ns}"
        return self._redis.json().get(str(checkpoint_key), "$")


memory = PatchedRedisSaver(
    redis_client=redis_client, ttl={"default_ttl": 3600, "refresh_on_read": True}
)
fraud_agent_app = graph.compile(checkpointer=memory)

# LLM Obserability by using opentelemetry


provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)


app = FastAPI(title="Fraud Detection Agent v3")

metrics_app = make_asgi_app(registry=promethus_registry)
app.mount("/metrics", metrics_app)


FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class HumanDecisionRequest(BaseModel):
    thread_id: str
    decision: str


class TransactionExtraction(BaseModel):
    user_id: str | None = None
    amount: float | None = None
    country: str | None = None
    transaction_count_last_hour: int | None = None


transaction_extraction_llm = llm_risk.with_structured_output(TransactionExtraction)

TRANSACTION_EXTRACTION_PROMPT = """
Extract transaction information from the user's description.

Extract only information that is explicitly present.

Fields:
-user_id
- amount
- transaction_count_last_hour
- country


If a field is not present, return null.
"""


def extract_transaction_data(description: str) -> dict:
    with tracer.start_as_current_span("extract_transaction_data") as span:
        span.set_attribute("operation", "transaction_extraction")
        span.set_attribute("llm_provider", "openrouter")
        span.set_attribute("llm.model", llm_risk.model)  

        prompt = f"""{TRANSACTION_EXTRACTION_PROMPT}

    Respond with ONLY a valid JSON object, no markdown, no extra text."""

        messages = [SystemMessage(content=prompt), HumanMessage(content=description)]

        try:
            response = llm_risk.invoke(messages)
            raw_text = response.content.strip()  # type: ignore
            cleaned = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            error_counter.add(1)
            logger.error(
                f"[Transaction Extraction] Failed: {e} | raw: {locals().get('raw_text', 'N/A')}"
            )
            return {}

        result = {k: v for k, v in parsed.items() if v is not None}

        span.set_attribute("transaction.fields_extracted", len(result))

        return result


@app.post("/analyze_transactions_with_documents")
async def analyze_transactions_with_documents(
    thread_id: str = Form(...),
    description: str = Form(...),
    document: UploadFile | None = None,
):
    start = time.time()
    request_counter.add(1, {"endpoint": "/analyze_transactions_with_documents"})
    try:
        logger.info("Extracting transaction data...")
        transaction_data = extract_transaction_data(description)

        if transaction_data:
            logger.info(f"DEBUG: {type(transaction_data)} {transaction_data}")
            with tracer.start_as_current_span("redis_store_transaction"):
                redis_mapping = {k: str(v) for k, v in transaction_data.items()}
                redis_client.hset(f"transaction:{thread_id}", mapping=redis_mapping)
    except RuntimeError as e:
        error_counter.add(1)
        logger.error(f"Redis Error: {e}")

    if not document:
        logger.info("[Document Verification] No document uploaded - skipping")
    base64_image = None
    if document:
        max_file_size_mb = 5
        image_bytes = await document.read()

        if len(image_bytes) > max_file_size_mb * 1024 * 1024:
            return {
                "status": "ERROR",
                "message": f"Document file size exceeds {max_file_size_mb} MB limit.",
            }

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

    config = {"configurable": {"thread_id": thread_id}}

    with tracer.start_as_current_span("fraud_agent_graph") as span:
        span.set_attribute("operation", "fraud_agent_graph")
        span.set_attribute("document.provided", document is not None)

        result = fraud_agent_app.invoke(  # type: ignore
            {
                "messages": [HumanMessage(content=description)],
                "document_image": base64_image,
            },
            config=config,
        )

        span.set_attribute("fraud.workflow_status", "completed")
    if "__interrupt__" in result:
        span.set_attribute("fraud.workflow_status", "pending_human_review")

        interrupt_data = result["__interrupt__"][0].value
        return {
            "status": "PENDING_REVIEW",
            "thread_id": thread_id,
            "ai_assessment": interrupt_data["decision_agent_assessment"],
            "message": "High risk detected. Call /human_decision with your decision.",
            "processing_time_seconds": round(time.time() - start, 2),
        }
    duration = time.time() - start
    request_duration.record(
        duration, {"endpoint": "/analyze_transactions_with_documents"}
    )
    return {
        "status": "COMPLETED",
        "thread_id": thread_id,
        "final_result": result["messages"][-1].content,
        "processing_time_seconds": round(duration, 2),
    }


@app.post("/human_decision")
def human_decision(req: HumanDecisionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = fraud_agent_app.invoke(Command(resume=req.decision), config=config)  # type: ignore
    return {
        "status": "COMPLETED",
        "thread_id": req.thread_id,
        "final_result": result["messages"][-1].content,
    }


@app.get("/")
def health_check():
    return {"status": "Fraud Detection Agent v3 API is running"}
