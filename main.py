

import os
import re
import time
import base64
import logging
from dotenv import load_dotenv
load_dotenv()  

from typing import Annotated, TypedDict, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.redis import RedisSaver
from tenacity import retry, wait_random_exponential, stop_after_attempt
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from redis_client import client as redis_client
from openai import OpenAI
from langsmith import Client
import langsmith as ls

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"]=os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "fraud-detection-MultiAgent"

langsmith_key = os.environ.get("LANGSMITH_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key

logging.basicConfig(level=logging.INFO)


client=Client(api_key=os.environ["LANGSMITH_API_KEY"])


USER_PROFILES = {
    "user_101": {"home_country": "UAE", "avg_transaction": 500},
    "user_102": {"home_country": "Pakistan", "avg_transaction": 200},
}


@tool(description="check the transaction amount that either it's in normal range or not ")
def check_amount_risk(amount: float, user_id: str) -> str:
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


@tool(description="check the location of the transaction against the user's home country")
def check_location_mismatch(user_id: str, transaction_country: str) -> str:
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return "Error: User profile not found"
    home = profile["home_country"]
    if home.lower() != transaction_country.lower():
        return f"MEDIUM RISK: Transaction from {transaction_country}, but user's home country is {home}"
    return f"LOW RISK: Transaction location ({transaction_country}) matches home country"


risk_tools = [check_amount_risk, check_velocity, check_location_mismatch]

from concurrent.futures import ThreadPoolExecutor

def run_risk_tools_parallel(user_id: str, amount: float, transaction_count_last_hour: int, country: str):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(check_amount_risk, amount, user_id),
            executor.submit(check_velocity, transaction_count_last_hour),
            executor.submit(check_location_mismatch, user_id, country),
        ]
        return [f.result() for f in futures]


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
    red_flags: list[str] = Field(default_factory=list,description="Signs of any digital editing or tampering, if any")
    font_consistency: Literal["consistent", "inconsistent", "cannot_determine"] = Field(description="Whether fonts/styles appear consistent")
    tampering_indicators: list[str] = Field(description="Signs of digital editing or tampering, if any")
    confidence_level: Literal["Low", "Medium", "High"] = Field(description="Confidence in this assessment")


from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult,ChatGeneration

class OpenRouterLLM(BaseChatModel):
    client:object
    model:str

    def _llm_type(self) -> str:
        return "openrouter"

    
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        formatted_message=[]
        for m in messages:
            role=getattr(m,"role","user")
            if isinstance(m.content,list):
                safe_content=json.dumps(m.content)
            else:
                safe_content=str(m.content)
            formatted_message.append({"role":role,"content":safe_content})
        response=self.client.chat.completions.create(
            model=self.model,
            messages=formatted_message,
            **kwargs
        )
        content=response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    
    def bind_tools(self, tools: list[BaseTool],**kwargs):
        """
        this is the function that binds the tools to the llm and returns a new llm instance with the tools bound
        """
        return self

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    document_image: Optional[str]


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

#Lighter weight vision models for document verification


OPENROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")

openrouter_client=OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


llm_risk =OpenRouterLLM(client=openrouter_client,model="meta-llama/llama-3.1-8b-instruct")
risk_analyst_llm = llm_risk.bind_tools(risk_tools)
decision_llm = llm_risk
structured_decision_llm = decision_llm.with_structured_output(Risk_Assessment)

#deep weight vision model for document verification
llm_vision =ChatGroq(model="qwen3.6-27b",api_key=GROQ_API_KEY)
document_verification_llm=llm_vision.with_structured_output(Document_Extraction_Result)


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
2. When referencing the amount finding, use the exact multiplier format (e.g., '4.0x higher'), never percentage.
3. Use exactly these field names: overall_risk, recommendation, justification.
But Respond with a valid JSON Object:
like {{"overall_risk": "...", "recommendation": "...", "justification": "..."}}"""

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

LIGHTWEIGHT_VISION_PROMPT = """Analyze this document quickly. Return strictly a JSON object matching this schema:
{
    "document_type": "Passport/Driver License/Bank Statement/Unknown",
    "name": "Full Name or null",
    "id_number": "ID Number or null",
    "date_of_birth": "YYYY-MM-DD or null",
    "appears_authentic": "yes" or "no",
    "red_flags": ["flag 1", "flag 2"],
    "font_consistency": "consistent", "inconsistent", or "cannot_determine",
    "tampering_indicators": ["indicator 1"],
    "confidence_level": "Low", "Medium", or "High"
}"""


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(2))
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


import re
import json

def extract_document_fields_lightweight(base64_image:str)->Document_Extraction_Result:
    message=[{
        "role":"user",
        "content":[
        {"type":"text","text":LIGHTWEIGHT_VISION_PROMPT},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64_image}"}}
    ]}]
    response=None
    try:
        response=OpenRouterLLM(client=openrouter_client,model="meta-llama/llama-3.1-8b-instruct",).invoke([HumanMessage(content=message)])

    except Exception as e: 
        print(f"[Lightweight Vision] Fallback model failed: {e}")
        try:
            response=OpenRouterLLM(client=openrouter_client,model="meta-llama/llama-3.1-8b-instruct",).invoke([HumanMessage(content=message)])
        except Exception as e:
            print(f"[Lightweight Vision] Fallback model failed: {e}")
            return None

    if not response or not response.content:
        return None

    content=response.content
    cleaned_content=re.sub(r"```json\s*|\s*```","",content).strip()
    try:
        parsed=json.loads(cleaned_content)
    except json.JSONDecodeError :
        print(f"[Lightweight Vision] JSON parsing failed. Raw content: {cleaned_content}")
        return None
    return Document_Extraction_Result.model_validate(parsed)

def extract_document_fields_deep_reasoning(base64_image: str) -> Document_Extraction_Result:
    message = HumanMessage(content=[
        {"type": "text", "text": DOCUMENT_VERIFICATION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ])
    try:
        result:Document_Extraction_Result=document_verification_llm.invoke([message])
        return result if isinstance(result, Document_Extraction_Result) else Document_Extraction_Result.model_validate(result)

    except Exception as e:
        print(f"[Deep Reasoning Vision] Model failed: {e}")
        return None
    
def document_verification_node(state: AgentState):
    base64_image = state.get("document_image")

    if not base64_image:
        return {"messages": [AIMessage(content="[Document Verification] No document provided - skipping document check.")]}

    try:
        extraction = extract_document_fields_lightweight(base64_image=base64_image)

        if not extraction:
            summary_text = (
                "[Document Verification]\n"
                "No extraction result available. Manual review required."
            )
        else:
                
            if extraction.confidence_level=="Low" or extraction.appears_authentic in ["no",False]:
                print(f"[Document Verification] Low confidence/Risk detected. Escalating to deep reasoning model...")
                extraction=extract_document_fields_deep_reasoning(base64_image=base64_image)
            
            summary_text = (
                    f"[Document Verification]\n"
                    f"Document Type: {getattr(extraction, 'document_type', 'N/A')}\n"
                    f"Appears Authentic: {getattr(extraction, 'appears_authentic', 'N/A')}\n"
                    f"Red Flags: {getattr(extraction, 'red_flags', 'N/A')}\n"
                    f"Font Consistency: {getattr(extraction, 'font_consistency', 'N/A')}\n"
                    f"Confidence Level: {getattr(extraction, 'confidence_level', 'N/A')}"
                )
            
    except Exception as e:
            print(f"[Document Verification Error] {type(e).__name__}: {e}")
            summary_text = (
            "[Document Verification]\n"
            "Automated document verification failed due to a technical issue. "
            "This transaction requires MANUAL document review before approval."
        )

    return {"messages": [AIMessage(content=summary_text)]}


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(2))
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
    parsed=structure_result.model_dump(exclude_none=True)
    if not parsed:
        return {"messages": [AIMessage(content="[Decision Agent] Unable to parse structured output. Manual review required.")]}
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


memory = RedisSaver(redis_client=redis_client, ttl=3600)
fraud_agent_app = graph.compile(checkpointer=memory)



app = FastAPI(title="Fraud Detection Agent v3")

if os.path.isdir("static"):
    app.mount("/static",StaticFiles(directory="static"),name="static")

class HumanDecisionRequest(BaseModel):
    thread_id: str
    decision: str

class TransactionExtraction(BaseModel):
    user_id:Optional[str]=None
    amount:Optional[float]=None
    country:Optional[str]=None
    transaction_count_last_hour: Optional[int]=None

transaction_extraction_llm=llm_risk.with_structured_output(TransactionExtraction)

TRANSACTION_EXTRACTION_PROMPT = """
Extract transaction information from the user's description.

Extract only information that is explicitly present.

Fields:
-user_id
- amount
- transaction_count_last_hour
- country


If a field is not present, return null.
But respond with a valid JSON object containing all fields, even if some are null.
like:{{'user_id':'...','amount':'...','transaction_count_last_hour':'...','country':'...'}}
"""


def extract_transaction_data(description:str)->dict:
    messages=[
        SystemMessage(content=TRANSACTION_EXTRACTION_PROMPT),
        HumanMessage(content=description)
    ]
    result:TransactionExtraction=transaction_extraction_llm.invoke(messages)
    if result is None:
        logging.error(f"[Transaction Extraction] Failed to extract transaction data from description: {description}")
        return {}
    return result.model_dump(exclude_none=True)




@app.post("/analyze_transactions_with_documents")
async def analyze_transactions_with_documents(
    thread_id: str = Form(...),
    description: str = Form(...),
    document: UploadFile |None = File(None)
):
    start = time.time() 

    try:
        logging.info("Extracting transaction data...")
        transaction_data=extract_transaction_data(description)
    
        if transaction_data:
            logging.info(f"DEBUG: {type(transaction_data)} {transaction_data}")
            redis_mapping={k:str(v) for k,v in transaction_data.items()}
            redis_client.hset(f"transaction:{thread_id}",mapping=redis_mapping)
    except Exception as e:
        logging.error(f"Redis Error: {e}")

    if not document:
        logging.info(f"[Document Verification] No document uploaded - skipping")
    base64_image = None
    if document: 
        max_file_size_mb = 5
        image_bytes=await document.read()
        
        if len(image_bytes)>max_file_size_mb * 1024 * 1024:
                return {
                    "status":"ERROR",
                    "message":f"Document file size exceeds {max_file_size_mb} MB limit."
                }

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

    config = {"configurable": {"thread_id": thread_id}}

    
   
                
    result = fraud_agent_app.invoke({
                "messages": [HumanMessage(content=description)],
                "document_image": base64_image
            }, config=config)
    
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


@app.post("/human_decision")
def human_decision(req: HumanDecisionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = fraud_agent_app.invoke(Command(resume=req.decision), config=config)
    return {
        "status": "COMPLETED",
        "thread_id": req.thread_id,
        "final_result": result["messages"][-1].content
    }


@app.get("/")
def health_check():
    return {"status": "Fraud Detection Agent v3 API is running"}