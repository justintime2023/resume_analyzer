# resume_agent_core.py
# LangGraph-based Resume Screening Agent (core logic)
# Designed to be called from Streamlit or any other frontend
# Uses GENAI_API_KEY, GENAI_BASE_URL, GENAI_MODEL (same pattern as hi.py)

import os
import json
import operator
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import chromadb

load_dotenv()

# --- Configuration ---
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
TOP_K_RESULTS = 3


# =====================================================
# LLM Setup (same pattern as hi.py)
# =====================================================

def get_llm(temperature=0.3):
    return ChatOpenAI(
        api_key=os.getenv("GENAI_API_KEY"),
        base_url=os.getenv("GENAI_BASE_URL"),
        model=os.getenv("GENAI_MODEL"),
        temperature=temperature,
        max_tokens=1500,
    )


# =====================================================
# State Definitions
# =====================================================

class ResumeInfo(TypedDict):
    filename: str
    text: str


class EvaluationResult(TypedDict):
    filename: str
    score: int
    summary: str
    strengths: List[str]
    gaps: List[str]
    recommendation: str


class AgentState(TypedDict):
    job_description: str
    resumes: List[ResumeInfo]
    pending_resumes: List[ResumeInfo]
    current_resume: Optional[ResumeInfo]
    evaluations: Annotated[List[EvaluationResult], operator.add]
    final_ranking: str
    error: str
    status: str
    progress_log: Annotated[List[str], operator.add]  # log messages for UI


# =====================================================
# Helpers
# =====================================================

def create_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def get_relevant_chunks(resume_text, job_description, top_k=TOP_K_RESULTS):
    chunks = create_chunks(resume_text)
    if not chunks:
        return [], []

    client = chromadb.Client()
    col_name = f"temp_{abs(hash(resume_text)) % 999999}"

    existing = [c.name for c in client.list_collections()]
    if col_name in existing:
        client.delete_collection(col_name)

    collection = client.create_collection(name=col_name)
    ids = [f"c_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)

    results = collection.query(query_texts=[job_description], n_results=top_k)
    client.delete_collection(col_name)

    return results["documents"][0], results["distances"][0]


# =====================================================
# LangGraph Nodes
# =====================================================

def validate_inputs_node(state: AgentState) -> dict:
    """Node 1: Validate that we have resumes and a JD."""
    resumes = state["resumes"]
    jd = state["job_description"]

    if not resumes:
        return {"error": "No resumes provided.", "status": "error", "progress_log": ["❌ No resumes found."]}
    if not jd or not jd.strip():
        return {"error": "No job description provided.", "status": "error", "progress_log": ["❌ No job description."]}

    valid = [r for r in resumes if r["text"].strip() and len(r["text"].split()) >= 20]
    if not valid:
        return {"error": "All resumes were empty or too short.", "status": "error", "progress_log": ["❌ All resumes invalid."]}

    log = [f"✅ Loaded {len(valid)} valid resume(s): {', '.join(r['filename'] for r in valid)}"]
    return {
        "resumes": valid,
        "pending_resumes": valid.copy(),
        "status": "validated",
        "progress_log": log
    }


def pick_next_resume_node(state: AgentState) -> dict:
    """Node 2: Pick next resume from the queue."""
    pending = state["pending_resumes"]
    if not pending:
        return {"current_resume": None, "status": "all_evaluated", "progress_log": []}

    current = pending[0]
    remaining = pending[1:]
    total = len(state["resumes"])
    idx = total - len(pending) + 1

    return {
        "current_resume": current,
        "pending_resumes": remaining,
        "status": "evaluating",
        "progress_log": [f"📄 [{idx}/{total}] Evaluating: {current['filename']}..."]
    }


def evaluate_resume_node(state: AgentState) -> dict:
    """Node 3: Evaluate current resume via RAG + LLM."""
    resume = state["current_resume"]
    job_description = state["job_description"]

    matched_chunks, distances = get_relevant_chunks(resume["text"], job_description)

    if not matched_chunks:
        return {
            "evaluations": [{
                "filename": resume["filename"],
                "score": 0,
                "summary": "Resume too short or empty to analyze.",
                "strengths": [],
                "gaps": [],
                "recommendation": "Weak Fit - insufficient content"
            }],
            "status": "evaluated",
            "progress_log": [f"  ⚠️ {resume['filename']}: No content to analyze"]
        }

    context = "\n\n".join(matched_chunks)
    llm = get_llm(temperature=0.2)

    prompt = f"""You are a professional HR assistant evaluating a resume against a job description.

Analyze the resume sections below and provide your evaluation in STRICT JSON format.

Job Description:
{job_description}

Relevant Resume Sections:
{context}

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
    "score": <integer 1-100 indicating overall fit>,
    "summary": "<2-3 sentence match summary>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "gaps": ["<gap 1>", "<gap 2>"],
    "recommendation": "<Strong Fit / Moderate Fit / Weak Fit with brief reason>"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        result["filename"] = resume["filename"]
        score = result.get("score", "N/A")

        return {
            "evaluations": [result],
            "status": "evaluated",
            "progress_log": [f"  ✅ {resume['filename']} → Score: {score}/100"]
        }

    except json.JSONDecodeError:
        return {
            "evaluations": [{
                "filename": resume["filename"], "score": 0,
                "summary": "Could not parse model response.",
                "strengths": [], "gaps": [],
                "recommendation": "Error in evaluation"
            }],
            "status": "evaluated",
            "progress_log": [f"  ⚠️ {resume['filename']}: Parse error"]
        }
    except Exception as e:
        return {
            "evaluations": [{
                "filename": resume["filename"], "score": 0,
                "summary": f"Error: {str(e)}",
                "strengths": [], "gaps": [],
                "recommendation": "Error in evaluation"
            }],
            "status": "evaluated",
            "progress_log": [f"  ❌ {resume['filename']}: {str(e)[:80]}"]
        }


def rank_candidates_node(state: AgentState) -> dict:
    """Node 4: Generate final comparative ranking."""
    evaluations = state["evaluations"]
    sorted_evals = sorted(evaluations, key=lambda x: x.get("score", 0), reverse=True)

    if len(sorted_evals) <= 1:
        name = sorted_evals[0]["filename"] if sorted_evals else "N/A"
        return {
            "final_ranking": f"Only one candidate: {name}",
            "status": "done",
            "progress_log": ["🏆 Single candidate — no ranking needed."]
        }

    candidates_summary = ""
    for i, r in enumerate(sorted_evals, 1):
        candidates_summary += f"""
Candidate {i}: {r['filename']}
  Score: {r.get('score', 'N/A')}/100
  Summary: {r.get('summary', 'N/A')}
  Strengths: {', '.join(r.get('strengths', []))}
  Gaps: {', '.join(r.get('gaps', []))}
"""

    llm = get_llm(temperature=0.3)
    prompt = f"""You are a senior HR consultant. You have evaluated multiple resumes for a position.

Job Description:
{state['job_description']}

Candidate Evaluations:
{candidates_summary}

Please provide:
1. A final ranking of ALL candidates from best to worst fit
2. Your TOP PICK with a clear justification
3. A brief comparison highlighting what sets the top candidate apart
4. Any candidates worth considering for an interview

Be specific and reference actual skills/experience from the evaluations."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_ranking": response.content,
        "status": "done",
        "progress_log": ["🏆 Final ranking generated!"]
    }


# =====================================================
# Routing
# =====================================================

def should_continue_or_error(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


def has_more_resumes(state: AgentState) -> str:
    if state.get("pending_resumes"):
        return "more"
    return "done"


# =====================================================
# Build Graph
# =====================================================

def build_graph():
    """
    LangGraph workflow:
      validate → pick_next → evaluate → [more?] → pick_next (loop)
                                       → [done]  → rank → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_inputs", validate_inputs_node)
    workflow.add_node("pick_next", pick_next_resume_node)
    workflow.add_node("evaluate", evaluate_resume_node)
    workflow.add_node("rank_candidates", rank_candidates_node)

    workflow.set_entry_point("validate_inputs")

    workflow.add_conditional_edges(
        "validate_inputs",
        should_continue_or_error,
        {"continue": "pick_next", "error": END}
    )

    workflow.add_edge("pick_next", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        has_more_resumes,
        {"more": "pick_next", "done": "rank_candidates"}
    )

    workflow.add_edge("rank_candidates", END)

    return workflow.compile()


def run_screening(resumes: List[dict], job_description: str) -> dict:
    """
    Main entry point for the agent.
    Args:
        resumes: list of {"filename": str, "text": str}
        job_description: str
    Returns:
        Final agent state dict with evaluations, final_ranking, progress_log, etc.
    """
    graph = build_graph()

    initial_state = {
        "job_description": job_description,
        "resumes": resumes,
        "pending_resumes": [],
        "current_resume": None,
        "evaluations": [],
        "final_ranking": "",
        "error": "",
        "status": "starting",
        "progress_log": ["🚀 Starting Resume Screening Agent..."]
    }

    return graph.invoke(initial_state)
