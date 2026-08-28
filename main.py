

import os
import re
import time
import base64
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from tenacity import retry, wait_random_exponential, stop_after_attempt
from fastapi import FastAPI, Form, UploadFile, File
from redis_client import redis_client
import json



load_dotenv()  

USER_PROFILES = {
    "user_101": {"home_country": "UAE", "avg_transaction": 500},
    "user_102": {"home_country": "Pakistan", "avg_transaction": 200},
}


@tool
def check_amount_risk(amount: float, user_id: str) -> str:
    """Compares the transaction amount against the user's average spending."""
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return "Error: User profile not found"
    avg = profile["avg_transaction"]
    if amount > avg * 10:
        return f"HIGH RISK: Amount ${amount} is {round(amount/avg, 1)}x higher than user's average (${avg})"
    elif amount > avg * 3:
        return f"MEDIUM RISK: Amount ${amount} is notably higher than average (${avg})"
    return f"LOW RISK: Amount ${amount} is within normal range (avg: ${avg})"


@tool
def check_velocity(transaction_count_last_hour: int) -> str:
    """Checks how many transactions occurred in the last hour."""
    if transaction_count_last_hour >= 5:
        return f"HIGH RISK: {transaction_count_last_hour} transactions in the last hour - unusual velocity"
    elif transaction_count_last_hour >= 3:
        return f"MEDIUM RISK: {transaction_count_last_hour} transactions in the last hour"
    return f"LOW RISK: {transaction_count_last_hour} transaction(s) in the last hour - normal"


@tool
def check_location_mismatch(user_id: str, transaction_country: str) -> str:
    """Compares the transaction's origin country against the user's home country."""
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return "Error: User profile not found"
    home = profile["home_country"]
    if home.lower() != transaction_country.lower():
        return f"MEDIUM RISK: Transaction from {transaction_country}, but user's home country is {home}"
    return f"LOW RISK: Transaction location ({transaction_country}) matches home country"


risk_tools = [check_amount_risk, check_velocity, check_location_mismatch]


class Risk_Assessment(BaseModel):
    overall_risk: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Overall risk classification")
    recommendation: Literal["APPROVE", "REVIEW", "BLOCK"] = Field(description="Action to take")
    justification: str = Field(description="Reasoning referencing specific findings")


class Document_Extraction_Result(BaseModel):
    document_type: str = Field(description="Type of document, e.g. Passport, Driver License, Bank Statement")
    name: Optional[str] = Field(default=None, description="Full name found on the document, if visible")
    id_number: Optional[str] = Field(default=None, description="ID/document number, if visible")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth, if visible")
    appears_authentic: Literal["yes", "no"] = Field(description="Overall authenticity assessment")
    red_flags: list[str] = Field(default_factory=list,description="Specific visual red flags found, if any. Empty list if none.")
    font_consistency: Literal["consistent", "inconsistent", "cannot_determine"] = Field(description="Whether fonts/styles appear consistent")
    tampering_indicators: list[str] = Field(description="Signs of digital editing or tampering, if any")
    confidence_level: Literal["Low", "Medium", "High"] = Field(description="Confidence in this assessment")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    document_image: Optional[str]


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm_risk = ChatGroq(model="qwen/qwen3.6-27b", max_tokens=1500, api_key=GROQ_API_KEY)
risk_analyst_llm = llm_risk.bind_tools(risk_tools)
decision_llm = llm_risk
structured_decision_llm = decision_llm.with_structured_output(Risk_Assessment)

llm_vision = ChatGroq(model="qwen/qwen3.6-27b", max_tokens=5000, api_key=GROQ_API_KEY)



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
2. When referencing the amount finding, use the exact multiplier format (e.g., '4.0x higher'), never percentage.
3. Use exactly these field names: overall_risk, recommendation, justification."""

DOCUMENT_VERIFICATION_PROMPT = """
Examine the provided document image.

Return ONLY one valid JSON object.
Do not return markdown.
Do not use ```json.
Do not write explanations before or after the JSON.

The JSON must contain exactly these fields:

{
  "document_type": "string",
  "name": "string or null",
  "id_number": "string or null",
  "date_of_birth": "string or null",
  "appears_authentic": "yes",
  "red_flags": [],
  "font_consistency": "consistent",
  "tampering_indicators": [],
  "confidence_level": "High"
}

Rules:
- red_flags must be a JSON array of strings.
- Maximum 2 red_flags.
- Each red_flag must contain fewer than 10 words.
- tampering_indicators must be a JSON array of strings.
- Maximum 2 tampering_indicators.
- Each tampering_indicator must contain fewer than 10 words.
- If there are no red flags, return [].
- If there are no tampering indicators, return [].
- Never put an array inside a string.
- Use null when a field is not visible.
- Return valid JSON only.
"""



# @retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt(5))
def risk_analyst_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=RISK_ANALYST_PROMPT)] + messages
    response = risk_analyst_llm.invoke(messages)
    return {"messages": [response]}


risk_tool_node = ToolNode(risk_tools)


def should_continue_risk_analysis(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "risk_tools"
    return "document_verification"


def extract_document_fields(base64_image: str) -> Document_Extraction_Result:
    message = HumanMessage(content=[
        {"type": "text", "text": DOCUMENT_VERIFICATION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])
    response=llm_vision.invoke([message])
    print(repr(response.content))
    print(type(respnse.content))
    data=json.loads(response.content)

    return Document_Extraction_Result.model_validate(data)

def document_verification_node(state: AgentState):
    base64_image = state.get("document_image")

    if not base64_image:
        return {"messages": [AIMessage(content="[Document Verification] No document provided - skipping document check.")]}

    try:
        extraction = extract_document_fields(base64_image=base64_image)
        summary_text = (
            f"[Document Verification]\n"
            f"Document Type: {extraction.document_type}\n"
            f"Appears Authentic: {extraction.appears_authentic}\n"
            f"Red Flags: {extraction.red_flags}\n"
            f"Font Consistency: {extraction.font_consistency}\n"
            f"Confidence Level: {extraction.confidence_level}"
        )
    except Exception as e:
        print(f"[Document Verification Error] {type(e).__name__}: {e}")
        raise

    return {"messages": [AIMessage(content=summary_text)]}


# @retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt(5))
def decision_agent_node(state: AgentState):
    risk_findings = None
    document_findings = None

    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content and "[Document Verification]" in m.content and document_findings is None:
            document_findings = m.content
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content and risk_findings is None:
            if "[Document Verification]" not in m.content:
                risk_findings = m.content

    decision_input = [
        SystemMessage(content=DECISION_AGENT_PROMPT),
        HumanMessage(content=(
            f"Transaction Risk Findings:\n{risk_findings}\n\n"
            f"Document Verification Findings:\n{document_findings or 'None provided'}\n\n"
            f"Provide your final risk classification and recommendation, considering BOTH sources."
        ))
    ]
    structure_result: Risk_Assessment = structured_decision_llm.invoke(decision_input)
    formatted_text = (
        f"Overall Risk: {structure_result.overall_risk}\n"
        f"Recommendation: {structure_result.recommendation}\n"
        f"Justification: {structure_result.justification}"
    )
    response = AIMessage(content=formatted_text)
    return {"messages": [response]}


def human_review_node(state: AgentState):
    last_message = state["messages"][-1]
    assessment_text = last_message.content
    needs_review = "HIGH" in assessment_text.upper() or "BLOCK" in assessment_text.upper()

    if not needs_review:
        return {"messages": []}

    human_decision = interrupt({
        "question": "Decision Agent flagged this as HIGH risk / BLOCK. Please confirm.",
        "decision_agent_assessment": assessment_text
    })
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
    {"risk_tools": "risk_tools", "document_verification": "document_verification"}
)
graph.add_edge("risk_tools", "risk_analyst")
graph.add_edge("document_verification", "decision_agent")
graph.add_edge("decision_agent", "human_review")
graph.add_edge("human_review", END)


DB_PATH = os.environ.get("DB_PATH", "fraud_agent_memory.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
fraud_agent_app = graph.compile(checkpointer=memory)



api = FastAPI(title="Fraud Detection Agent v3")


class HumanDecisionRequest(BaseModel):
    thread_id: str
    decision: str

class TransactionExtraction(BaseModel):
    user_id:Optional[str]=None
    amount:Optional[float]=None
    country:Optional[str]=None
    last_hour_transaction: Optional[int]=None

transaction_extraction_llm=llm_risk.with_structured_output(TransactionExtraction)

TRANSACTION_EXTRACTION_PROMPT = """
Extract transaction information from the user's description.

Extract only information that is explicitly present.

Fields:
-user_id
- amount
- last_hour_transaction | velocity
- country


If a field is not present, return null.
"""


def extract_transaction_data(description:str)->str:
    messages=[
        SystemMessage(content=TRANSACTION_EXTRACTION_PROMPT),
        HumanMessage(content=description)
    ]
    result:TransactionExtraction=transaction_extraction_llm.invoke(messages)

    data=result.model_dump(exclude_none=True)

    return data



@api.post("/analyze_transactions_with_documents")
async def analyze_transactions_with_documents(
    thread_id: str = Form(...),
    description: str = Form(...),
    document: UploadFile = File(...)
):
    start = time.time() 

    transaction_data=extract_transaction_data(description)
    
    if transaction_data:
        redis_client.hset(f"transaction: {thread_id}",mapping=transaction_data)
    

    image_bytes = await document.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = fraud_agent_app.invoke({
            "messages": [HumanMessage(content=description)],
            "document_image": base64_image
        }, config=config)
    except Exception as e:
        return {"status": "Error", "error": str(e)}

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        return {
            "status": "PENDING_REVIEW",
            "thread_id": thread_id,
            "ai_assessment": interrupt_data["decision_agent_assessment"],
            "message": "High risk detected. Call /human_decision with your decision.",
            "processing_time_seconds": round(time.time() - start, 2)
        }

    return {
        "status": "COMPLETED",
        "thread_id": thread_id,
        "final_result": result["messages"][-1].content,
        "processing_time_seconds": round(time.time() - start, 2)
    }


@api.post("/human_decision")
def human_decision(req: HumanDecisionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = fraud_agent_app.invoke(Command(resume=req.decision), config=config)
    return {
        "status": "COMPLETED",
        "thread_id": req.thread_id,
        "final_result": result["messages"][-1].content
    }


@api.get("/")
def health_check():
    return {"status": "Fraud Detection Agent v3 API is running"}