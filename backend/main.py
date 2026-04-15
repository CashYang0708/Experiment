#!/usr/bin/env python3
"""FastAPI server for quant multi-agent evaluation.

Input: user query
Output: evaluation agent report
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    # Works when launched from project root: uvicorn backend.main:app --reload
    from backend.agent import AgentState, build_multi_agent_graph
except ModuleNotFoundError:
    # Works when launched inside backend/: uvicorn main:app --reload
    from agent import AgentState, build_multi_agent_graph


app = FastAPI(title="Quant Evaluation API", version="1.0.0")
agent_graph = build_multi_agent_graph()


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")


class EvaluateResponse(BaseModel):
    evaluation_report: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    state: AgentState = {
        "user_message": payload.query,
        "label": "",
        "rag_output": "",
        "gp_output": "",
        "best_alpha": "",
        "backtest_output": "",
        "evaluation_report": "",
    }

    try:
        result = agent_graph.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {exc}") from exc

    report = str(result.get("evaluation_report", "")).strip()
    if not report:
        raise HTTPException(status_code=500, detail="Evaluation report is empty")

    return EvaluateResponse(evaluation_report=report)
