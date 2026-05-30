#!/usr/bin/env python3
"""FastAPI server for quant multi-agent evaluation.

Input: user query
Output: evaluation agent report
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from backend.mongo_store import save_report
except ModuleNotFoundError:
    from mongo_store import save_report

try:
    # Works when launched from project root: uvicorn backend.main:app --reload
    from backend.agent import AgentState, build_multi_agent_graph, build_system_hint
except ModuleNotFoundError:
    # Works when launched inside backend/: uvicorn main:app --reload
    from agent import AgentState, build_multi_agent_graph, build_system_hint


app = FastAPI(title="Quant Evaluation API", version="1.0.0")
agent_graph = build_multi_agent_graph()


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")


class EvaluateResponse(BaseModel):
    label: str = ""
    notice: str = ""
    system_hint: str = ""
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

    label = str(result.get("label", ""))

    if label == "unrelated":
        system_hint = build_system_hint(payload.query, label)
        save_report(payload.query, system_hint, "unrelated")
        return EvaluateResponse(
            label="unrelated",
            notice="",
            system_hint=system_hint,
            evaluation_report=system_hint,
        )

    report = str(result.get("evaluation_report", "")).strip()
    if not report:
        raise HTTPException(status_code=500, detail="Evaluation report is empty")
    
    # 針對不同label給予不同的notice提示，讓使用者知道接下來會發生什麼事
    notice = ""
    if label == "alpha_search":
        notice = "即將執行RAG搜尋alpha內部因子庫找出符合市場情境的alpha並進行回測驗證與分析"

    if label == "genetic_programming":
        notice = "即將執行基因演算法尋找alpha組合並進行回測驗證與分析"

    save_report(payload.query, report, label)

    return EvaluateResponse(label=label, notice=notice, evaluation_report=report)
