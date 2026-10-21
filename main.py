import base64
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
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
from opentelemetry.trace import Status, StatusCode
from prometheus_client import CollectorRegistry, make_asgi_app
from pydantic import BaseModel, Field, SecretStr
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

print("Meter provider:", metrics.get_meter_provider())
print("Custom registry collectors:", list(promethus_registry.collect()))

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

USER_PROFILES:dict[str,dict[str,str|float]] = {
    "user_101": {"home_country": "UAE", "avg_transaction": 500.0},
    "user_102": {"home_country": "Pakistan", "avg_transaction": 200.0},
}

# -----------------------------------------------------------


@tool(
    description="check the transaction amount that either it's in normal range or not "
)
def check_amount_risk(amount: float, user_id: str) -> str:
    profile = USER_PROFILES.get(user_id)
    if profile is None:
        return "Error: User profile not found"
    
    avg = profile["avg_transaction"]
    
    if not isinstance(avg,(int,float)):
        return "Error: Invalid average transaction amount"
    
    avg=float(avg)
    if avg<=0.0: return "Error: Invalid average transaction amount."

    if amount > avg * 10.0:  
        return f"HIGH RISK: Amount ${amount} is {round(amount / avg, 1)}x higher than user's average (${avg})" 
    elif amount > avg * 3.0:  
        return f"MEDIUM RISK: Amount ${amount} is notably higher than average (${avg})"
    return f"LOW RISK: Amount ${amount} is within normal range (avg: ${avg})"


@tool
def check_velocity(transaction_count_last_hour: int) -> str:
    """Checks how many transactions occurred in the last hour."""

    if transaction_count_last_hour<0:return "Error: Transaction count cannot be negative "

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
    if home.lower() != transaction_country.lower():  # type: ignore
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
            executor.submit(check_amount_risk.invoke, {"amount":amount,"user_id": user_id}),  
            executor.submit(check_velocity.invoke, {"transaction_count_last_hour":transaction_count_last_hour}),  
            executor.submit(check_location_mismatch.invoke, {"user_id":user_id, "country":country}),  
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




OPENROUTER_API_KEY: str | None = os.environ.get("OPEN_ROUTER_API_KEY")

if not OPENROUTER_API_KEY: raise RuntimeError('OPEN_ROUTER_API_KEY is missing')


llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,  # type: ignore
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3.1-8b-instruct",
)



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_assessment: Risk_Assessment | None
    document_image: str | None


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing or empty")
api_key=SecretStr(GROQ_API_KEY)


MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

llm_risk = llm.bind_tools(risk_tools)

structured_decision_llm = llm.with_structured_output(Risk_Assessment)

# deep weight vision model for document verification
llm_vision = ChatGroq(
    name="vision_model",    
    model="qwen/qwen3.8-27b", api_key=api_key, timeout=30, max_retries=0
) 
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
        span.set_attribute("llm.model", MODEL_NAME)
        span.set_attribute("operation", "risk_analysis")

        messages = state["messages"]

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=RISK_ANALYST_PROMPT), *messages]

        llm_start = time.perf_counter()
        try:
            response = llm_risk.invoke(messages)
            has_tool_calls = bool(response.tool_calls)

            span.set_attribute("risk_analysis.success", True)
            span.set_attribute("risk_analysis.has_tool_calls", has_tool_calls)

            return {"messages": [response]}

        except Exception as e:
            error_counter.add(1, {"operation": "risk_analysis"})
            span.set_attribute("risk_analysis.success", False)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, type(e).__name__))

            raise

        finally:
            llm_duration.record(
                time.perf_counter() - llm_start,
                {"model": MODEL_NAME, "operation": "Risk Analysis"},
            )


risk_tool_node = ToolNode(risk_tools)


def should_continue_risk_analysis(state: AgentState) -> str:

    message = state.get("messages", [])

    if not message:
        raise ValueError("Risk analysis cannot continue: no message found")

    last_message = message[-1]

    if not isinstance(last_message, AIMessage):

        raise TypeError("Risk analysis cannot continue: expected AI Message")

    if last_message.tool_calls:return "risk_tools"

    
    if not last_message.content:
        raise TypeError("Risk analyst returned neither content nor tool calls.")

    return "document_verification"


import json

from pydantic import ValidationError


def extract_document_fields_lightweight(
    base64_image: str,
) -> Document_Extraction_Result | None:

    with tracer.start_as_current_span("extract_document_fields_lightweight") as span:
        span.set_attribute("llm_provider", "google")
        span.set_attribute("llm.model", GEMINI_VISION_MODEL)
        span.set_attribute("operation", "document_extraction")
        span.set_attribute("vision.state", "lightweight")

        vision_start = time.perf_counter()

        import filetype
        SUPPORTED_TYPES={"application/pdf","image/jpeg","image/png"}

        try:
            file_bytes = base64.b64decode(base64_image, validate=True)
            kind=filetype.guess(file_bytes)
            if kind is None or kind.mime not in SUPPORTED_TYPES: 
                raise ValueError("Unspported or invalid document format")
            mime_type=kind.mime
            image_part=types.Part.from_bytes(data=file_bytes,mime_type=mime_type)
            response = gemini_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[  # type: ignore
                    LIGHTWEIGHT_VISION_PROMPT,
                    image_part
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            if not response or not response.text:
                raise ValueError("Gemini returned an empty response")

            parsed = json.loads(response.text)

            result = Document_Extraction_Result.model_validate(parsed)

            span.set_attribute("llm.response_received", True)
            span.set_attribute("document.extraction_success", True)

            return result
        except (ValueError,json.JSONDecodeError,ValidationError) as e:
            error_counter.add(1)

            span.set_attribute("llm.response_received", False)
            span.set_attribute("document.extraction_success", False)

            logger.error(f"[Lightweight Vision] Call failed: {type(e).__name__}: {e}")

            span.record_exception(e)

            return None

        except Exception as e:

            error_counter.add(1, {"operation": "document_extraction"})
            span.set_attribute("document.extraction_success", False)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, type(e).__name__))
            raise


        finally:
            llm_duration.record(
                time.perf_counter() - vision_start,
                {
                    "model": GEMINI_VISION_MODEL,
                    "operation": "document extraction with lightweight model",
                },
            )


def extract_document_fields_deep_reasoning(
    base64_image: str,
    mime_type:str="image/jpeg"
) -> Document_Extraction_Result:
    with tracer.start_as_current_span("extract_document_fields_deep_reasoning") as span:
        span.set_attribute("llm_provider", "groq")
        span.set_attribute("llm.model", "qwen/qwen3.8-27b")
        span.set_attribute("operation", "document_verification")
        span.set_attribute("vision_stage", "deep_reasoning")

        vision_time = time.perf_counter()
        message = HumanMessage(
            content=[
                {"type": "text", "text": DOCUMENT_VERIFICATION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )
        try:
            result: Document_Extraction_Result = document_verification_llm.invoke(
                [message]  # type: ignore
            )

            if isinstance(result, Document_Extraction_Result):
                validated_result = result

            elif isinstance(result,dict):validated_result=Document_Extraction_Result.model_validate(result)

            else:raise TypeError(f"Unexpected model response type: {type(result).__name__}")

            span.set_attribute("document.extraction_success", True)

            return validated_result

        except (ValueError,ValidationError) as e:
            error_counter.add(1)
            span.set_attribute("document.extraction_success", False)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR,type(e).__name__))

            logger.error(f"[Deep Reasoning Vision] Model failed: {type(e).__name__}")
            return None  # type: ignore

        finally:
            llm_duration.record(
                time.perf_counter()-vision_time,
                {
                    "model": "qwen/qwen3.8-27b",
                    "operation": "extract document fields with deepweight model",
                },
            )


def document_verification_node(state: AgentState):

    with tracer.start_as_current_span("verify document") as span:
        span.set_attribute("operation", "document_verification")

        base64_image = state.get("document_image")

        if not base64_image:
            span.set_attribute("vision.state", "skipped")
            span.set_attribute("vision.manual_review_required", True)
            return {
                "messages": [
                    AIMessage(
                        content="[Document Verification] No document provided - skipping document check."
                    )
                ]
            }

        try:
            span.set_attribute("llm_provider", "google")
            span.set_attribute("vision.stage", "lightweight")

            extraction = extract_document_fields_lightweight(base64_image=base64_image)

            if extraction is None:
                span.set_attribute("vision.state", "failed")
                span.set_attribute("vision.escalated", False)
                span.set_attribute("vision.manual_review_required", True)

                summary_text = (
                    "[Document Verification]\n"
                    "Lightweight extraction returned no result.\n"
                    "Manual document review required."
                )
                return {"messages": [AIMessage(content=summary_text)]}

            selected_extraction = extraction

            confidence = (
                str(getattr(extraction, "confidence_level", None)or "Unknown").strip().lower()
            )
            authenticity = getattr(extraction, "appears_authentic", None)

            authenticity_is_positive = authenticity is True or (
                isinstance(authenticity, str)
                and authenticity.strip().lower() in {"true", "yes"}
            )

            needs_escalation = confidence != "high" or not authenticity_is_positive
            span.set_attribute("vision.escalated", needs_escalation)

            if needs_escalation:
                span.set_attribute("vision.escalated", True)
                span.set_attribute("llm_provider", "groq")
                span.set_attribute("vision.stage", "deep_reasoning")

                deep_extraction = extract_document_fields_deep_reasoning(
                    base64_image=base64_image
                )

                if deep_extraction is None:
                    span.set_attribute("vision.state", "failed")
                    span.set_attribute("vision.manual_review_required", True)

                    summary_text = (
                        "[Document Verification]\n"
                        "Deep verification failed\n"
                        "Manual document review required"
                    )

                    return {"messages": [AIMessage(content=summary_text)]}

                selected_extraction = deep_extraction

                span.set_attribute("vision.state", "deep_reasoning")
                span.set_attribute("llm_provider", "groq")

            else:
                span.set_attribute("vision.state", "lightweight")
                span.set_attribute("vision.escalated", False)

            final_authenticity = getattr(selected_extraction, "appears_authentic", None)

            final_authenticity_is_positive = (
                final_authenticity is True
                or isinstance(final_authenticity, str)
                and final_authenticity.strip().lower() in {"true", "yes"}
            )

            final_confidence = str(
                getattr(selected_extraction, "confidence_level", None)or "Unknown").strip().lower()

            summary_text = (
                f"[Document Verification]\n"
                f"Document Type: {getattr(selected_extraction, 'document_type', 'N/A')}\n"
                f"Appears Authentic: {getattr(selected_extraction, 'appears_authentic', 'N/A')}\n"
                f"Red Flags: {getattr(selected_extraction, 'red_flags', 'N/A')}\n"
                f"Font Consistency: {getattr(selected_extraction, 'font_consistency', 'N/A')}\n"
                f"Confidence Level: {getattr(selected_extraction, 'confidence_level', 'N/A')}"
            )

            manual_review_required = (
                not final_authenticity_is_positive or final_confidence != "high"
            )

            span.set_attribute("document.appears_authentic", str(final_authenticity))
            span.set_attribute("vision.manual_review_required", manual_review_required)

            if manual_review_required:
                summary_text += "\nMANUAL document review required before approval."

            return {"messages": [AIMessage(content=summary_text)],'manual_review_required':manual_review_required}

        except Exception as e:
            error_counter.add(1, {"operation": "document_verification"})
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, "Document Verification failed")
            span.set_attribute("vision.stage", "error")
            span.set_attribute("vision.manual_review_required", True)

            logger.exception(f"[Document Verification Error] {type(e).__name__}")

            summary_text = (
                "[Document Verification]\n"
                "Automated document verification failed due to a technical issue. \n"
                "This transaction requires MANUAL document review before approval."
            )

            return {"messages": [AIMessage(content=summary_text)],'manual_review_required':True}


def decision_agent_node(state: AgentState):
    with tracer.start_as_current_span("decision_agent") as span:
        span.set_attribute("operation", "decision_agent")
        span.set_attribute("llm_provider", "openrouter")
        span.set_attribute("llm.model", MODEL_NAME)

        risk_findings = None
        document_findings = None

        for m in reversed(state["messages"]):
            if not isinstance(m, AIMessage) or not m.content:
                continue

            if getattr(m, "tool_calls", None):
                continue

            if "[Document Verification]" in m.content:
                if document_findings is None:
                    document_findings = m.content
            elif risk_findings is None:
                risk_findings = m.content

            if risk_findings and document_findings:
                break

        decision_input = [
            SystemMessage(content=DECISION_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Transaction Risk Findings:\n"
                    f"{risk_findings or 'None provided'}\n\n"
                    f"Document Verification Findings:\n"
                    f"{document_findings or 'None provided'}\n\n"
                    "Provide your final risk classification and recommendation, "
                    "considering BOTH sources."
                )
            ),
        ]

        start_time = time.perf_counter()
        decision_status="unknown"
        try:
            decision_status="success"
            response = structured_decision_llm.invoke(decision_input)

            if isinstance(response, Risk_Assessment):
                structure_result = response
            else:
                structure_result = Risk_Assessment.model_validate(response)

            span.set_attribute(
                "decision.overall_risk",
                structure_result.overall_risk,
            )
            span.set_attribute(
                "decision.recommendation",
                structure_result.recommendation,
            )
            span.set_attribute("decision_agent_failed", False)

            logger.info(
                "[Decision Agent] Result: %s",
                structure_result.model_dump(),
            )

        except Exception as e:
            decision_status = "fallback"
            decision_counter.add(
                1,
                {"operation": "decision_agent"},
            )
            error_counter.add(
                1,
                {"operation": "decision_agent"},
            )
            span.record_exception(e)
            span.set_attribute("decision_agent_failed", True)
            span.set_status(Status(StatusCode.ERROR, str(e)))

            logger.exception("[Decision Agent] Failed")

            structure_result = Risk_Assessment(
                overall_risk="HIGH",
                recommendation="REVIEW",
                justification=(
                    "Automated decision failed. "
                    "Fail-safe classification applied; manual review required."
                ),
            )
        
        finally:
            llm_duration.record(
                time.perf_counter() - start_time,
                {
                    "model": MODEL_NAME,
                    "operation": "decision_agent",
                    "status": decision_status,
                },
            )
        if state.get('manual_review_required',False):
            structure_result=structure_result.model_copy(update={
                "Recommendation":"REVIEW",
                "justification":f"{structure_result.justification}"
                "Manual document review is mendatory"
            })
        formatted_text = (
            f"Overall Risk: {structure_result.overall_risk}\n"
            f"Recommendation: {structure_result.recommendation}\n"
            f"Justification: {structure_result.justification}"
        )

        response_msg = AIMessage(content=formatted_text)

        decision_counter.add(
            1,
            {"overall_risk": structure_result.overall_risk},
        )

        return {"messages": [response_msg], "risk_assessment": structure_result}


def human_review_node(state: AgentState):

    assessment = state.get("risk_assessment")

    needs_review = (
        assessment is None
        or assessment.overall_risk == "HIGH"
        or assessment.recommendation in {"BLOCK", "REVIEW"}
        or state.get("manual_review_required", False)
    )

    if not needs_review:
        return {"messages": []}

    human_decision = interrupt({
        "question": "Manual review required. Please confirm the decision.",
        "decision_agent_assessment": (
            state["messages"][-1].content
            if state.get("messages")
            else "No assessment available"
        ),
    })

    return {
        "messages": [HumanMessage(content=f"[HUMAN REVIEW]: {human_decision}")]
    }


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
        result=self._redis.json().get(str(checkpoint_key), "$")
        
        if result is None:
            return None

        if isinstance(result, list):
            result = result[0] if result else None

        if result is None:
            return None

        if not isinstance(result, dict):
            raise TypeError("Redis checkpoint must be a dictionary")

        return result


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


transaction_extraction_llm = llm.with_structured_output(TransactionExtraction)

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
        span.set_attribute("llm.model", MODEL_NAME)

        prompt = f"""{TRANSACTION_EXTRACTION_PROMPT}

    Respond with ONLY a valid JSON object, no markdown, no extra text."""

        messages = [SystemMessage(content=prompt), HumanMessage(content=description)]

        try:
            response: TransactionExtraction = transaction_extraction_llm.invoke(
                messages
            )  # type: ignore
            result = response.model_dump(exclude_none=True)

            span.set_attribute("transaction.fields_extracted", len(result))
            span.set_attribute("transaction.extraction_success", bool(result))

            return result

        except RuntimeError as e:
            error_counter.add(1)
            span.set_attribute("transaction.extraction_success", False)
            span.record_exception(e)

            logger.error(
                f"[Transaction Extraction] Failed: {e} | raw: {locals().get('raw_text', 'N/A')}"
            )
            return {}


@app.post("/analyze_transactions_with_documents")
async def analyze_transactions_with_documents(
    thread_id: str = Form(...),
    description: str = Form(...),
    document: UploadFile | None = None,
):
    start = time.perf_counter()
    request_counter.add(1, {"endpoint": "/analyze_transactions_with_documents"})
    try:
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
            logger.error(f"Transaction Error/Redis Error: {e}")
            raise

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

        config:RunnableConfig = {"configurable": {"thread_id": thread_id}}
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=description)],
            "risk_assessment":None,
            "document_image": base64_image,
        }

        with tracer.start_as_current_span("fraud_agent_graph") as span:
            span.set_attribute("operation", "fraud_agent_graph")
            span.set_attribute("document.provided", document is not None)

            result = fraud_agent_app.invoke(  
                initial_state,
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
                "processing_time_seconds": round(time.perf_counter() - start, 2),
            }

        return {
            "status": "COMPLETED",
            "thread_id": thread_id,
            "final_result": result["messages"][-1].content,
            "processing_time_seconds": round(time.perf_counter() - start, 2),
        }

    finally:
        duration = time.perf_counter() - start
        request_duration.record(
            duration, {"endpoint": "/analyze_transactions_with_documents"}
        )


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
