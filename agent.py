#!/usr/bin/env python3
"""LangGraph multi-agent workflow.

Agents:
1) orchestrator_agent: classify and route message.
2) alpha_search_agent: run RAG search for alpha queries.
3) gp_agent: execute genetic programming via gp.py.
4) backtesting_agent: run backtest for the best alpha from gp_agent.
5) evaluation_agent: aggregate outputs and generate final evaluation report.



Usage:
    python agent.py --message "how to design fitness function in gp"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Optional
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from google import genai


# Optional: hardcode key if you do not want env vars.
GEMINI_API_KEY = "your api key"
GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_LABEL = "alpha_search"
RAG_DB_PATH = "./chroma_db"
RAG_COLLECTION = "alpha_factors"
RAG_TOP_K = 1
GP_NPOP = 20
GP_SEED = 42
GP_CROSSOVER = 0.4
GP_MUTATION = 0.4
GP_FITNESS_FUNCTION = "pearson_fitness"

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
             Classify the user message as either:
             - 'genetic_programming':  if the message is about genetic programming and contains terms like fitness function, gp
             - 'alpha_search': if the message is about alpha your trading thoughts and market situation
             '""",
}


class AgentState(TypedDict):
    user_message: str
    label: str
    rag_output: str
    gp_output: str
    best_alpha: str
    backtest_output: str
    evaluation_report: str


def orchestrator_classify(message: str) -> str:
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return DEFAULT_LABEL

    client = genai.Client(api_key=api_key)

    prompt = (
        f"System:\n{SYSTEM_PROMPT['content']}\n\n"
        "Return ONLY JSON with this schema:\n"
        '{"label":"genetic_programming"|"alpha_search"}\n\n'
        f"User message:\n{message}"
    )

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        parsed = json.loads(text)
        label = str(parsed.get("label", "")).strip()
        if label in {"genetic_programming", "alpha_search"}:
            return label
        return DEFAULT_LABEL
    except Exception:
        return DEFAULT_LABEL


def orchestrator_agent_node(state: AgentState) -> AgentState:
    """Orchestrator agent: classify the message for downstream routing."""
    label = orchestrator_classify(state["user_message"])
    return {
        "user_message": state["user_message"],
        "label": label,
        "rag_output": "",
        "gp_output": "",
        "best_alpha": "",
        "backtest_output": "",
        "evaluation_report": "",
    }


@tool("rag_search")
def rag_search_tool(user_message: str, top_k: int, db_path: str, collection: str) -> str:
    """Run Chroma RAG query and return stdout text."""
    script_path = os.path.join(os.path.dirname(__file__), "alpha_rag_chroma.py")
    cmd = [
        sys.executable,
        script_path,
        "--db",
        db_path,
        "--collection",
        collection,
        "query",
        "--query",
        user_message,
        "-k",
        str(top_k),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"RAG query failed (exit={proc.returncode}).\n{err or output}"
        if err:
            return f"{output}\n\n[stderr]\n{err}".strip()
        return output
    except Exception as exc:
        return f"RAG query failed with exception: {exc}"


@tool("gp_run")
def gp_run_tool(
    user_message: str,
    npop: int,
    generations: int,
    seed: int,
    crossover: float,
    mutation: float,
    fitness_function: str,
) -> str:
    """Run gp.py for genetic programming queries and return stdout text."""
    script_path = os.path.join(os.path.dirname(__file__), "gp.py")
    cmd = [
        sys.executable,
        script_path,
        "--message",
        user_message,
        "--npop",
        str(npop),
        "--generations",
        str(generations),
        "--seed",
        str(seed),
        "--crossover",
        str(crossover),
        "--mutation",
        str(mutation),
        "--fitness-function",
        fitness_function,
    ]

    try:
        last_output = ""
        last_err = ""
        last_code = -1
        for _ in range(2):
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            output = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode == 0:
                if err:
                    return f"{output}\n\n[stderr]\n{err}".strip()
                return output
            last_output = output
            last_err = err
            last_code = proc.returncode

        return f"GP run failed after retry (exit={last_code}).\n{last_err or last_output}"
    except Exception as exc:
        return f"GP run failed with exception: {exc}"


@tool("run_backtest")
def run_backtest_tool(alpha_expression: str, period: str = "10y") -> str:
    """Run backtest for a given alpha expression via backtest/executor.py."""
    try:
        from backtest.executor import GpTemplate
    except Exception as exc:
        return f"Backtest failed: cannot import GpTemplate ({exc})"

    try:
        result = GpTemplate(alpha_expression, period=period).run()
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return f"Backtest failed: {exc}"


@tool("run_alpha_backtest")
def run_alpha_backtest_tool(alpha_name: str) -> str:
    """Run backtest for an Alpha101 model name via backtest/executor.py."""
    try:
        from backtest.executor import Template
    except Exception as exc:
        return f"Backtest failed: cannot import Template ({exc})"

    try:
        normalized_alpha_name = normalize_alpha101_name(alpha_name)
        result = Template(normalized_alpha_name).run()
        # Template returns an image object, which cannot be JSON serialized.
        if isinstance(result, dict) and "image" in result:
            result = {k: v for k, v in result.items() if k != "image"}
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return f"Backtest failed: {exc}"


@tool("adjust_backtest_period")
def adjust_backtest_period_tool(
    sharpe: float,
    cum_returns: float,
    max_drawdown_pct: float,
    current_period: str,
) -> str:
    """Ask LLM to decide if metrics are extreme and whether to retest on another period."""
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    period_order = ["10y", "5y", "3y"]
    default_next = current_period
    if current_period in period_order:
        idx = period_order.index(current_period)
        if idx + 1 < len(period_order):
            default_next = period_order[idx + 1]

    default_payload = {
        "extreme": False,
        "current_period": current_period,
        "next_period": default_next,
        "should_retest": False,
        "reason": "llm_unavailable_or_invalid_output",
    }

    if not api_key:
        return json.dumps(default_payload)

    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception:
        return json.dumps(default_payload)

    prompt = (
        "You are a backtest reviewer. Decide if a backtest result is extremely high or low "
        "using financial reasoning, not fixed thresholds.\n"
        "Given metrics and current period, return ONLY JSON with keys:\n"
        "extreme (bool), should_retest (bool), next_period (one of 10y/5y/3y), reason (short string).\n"
        f"Input metrics: sharpe={sharpe}, cum_returns={cum_returns}, max_drawdown_pct={max_drawdown_pct}, current_period={current_period}.\n"
        "If should_retest is true, choose a shorter period than current when possible."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        obj = json.loads(text)

        next_period = str(obj.get("next_period", default_next))
        if next_period not in period_order:
            next_period = default_next

        payload = {
            "extreme": bool(obj.get("extreme", False)),
            "current_period": current_period,
            "next_period": next_period,
            "should_retest": bool(obj.get("should_retest", False)) and next_period != current_period,
            "reason": str(obj.get("reason", "llm_decision")),
        }
        return json.dumps(payload)
    except Exception:
        return json.dumps(default_payload)


@tool("gp_adjust_params")
def gp_adjust_params_tool(
    attempt: int,
    current_crossover: float,
    current_mutation: float,
    current_score: float,
    previous_score: float,
) -> str:
    """Ask LLM to suggest next GP crossover/mutation from score trend.

    Returns JSON string: {"crossover": float, "mutation": float, "reason": str}
    """
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    fallback_payload = {
        "crossover": round(current_crossover, 4),
        "mutation": round(current_mutation, 4),
        "reason": "llm_unavailable_keep_params",
        "attempt": attempt,
    }

    if not api_key:
        return json.dumps(fallback_payload)

    prompt = (
        "You are a GP hyperparameter tuner. Suggest next crossover/mutation based on recent score trend.\n"
        "Do not use fixed numeric thresholds. Use trend-based reasoning.\n"
        "Return ONLY JSON with keys: crossover (float), mutation (float), reason (string).\n"
        "Constraints: 0.05 <= crossover <= 0.9, 0.05 <= mutation <= 0.9, crossover+mutation <= 0.95.\n"
        f"Input: attempt={attempt}, current_crossover={current_crossover}, current_mutation={current_mutation}, current_score={current_score}, previous_score={previous_score}."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        obj = json.loads((response.text or "").strip())
        next_crossover = float(obj.get("crossover", current_crossover))
        next_mutation = float(obj.get("mutation", current_mutation))
        reason = str(obj.get("reason", "llm_adjustment"))

        # Enforce safe bounds and probability feasibility.
        next_crossover = min(0.9, max(0.05, next_crossover))
        next_mutation = min(0.9, max(0.05, next_mutation))
        total = next_crossover + next_mutation
        if total > 0.95:
            scale = 0.95 / total
            next_crossover *= scale
            next_mutation *= scale

        payload = {
            "crossover": round(next_crossover, 4),
            "mutation": round(next_mutation, 4),
            "reason": reason,
            "attempt": attempt,
        }
        return json.dumps(payload)
    except Exception:
        return json.dumps(fallback_payload)


def parse_best_fitness(output: str) -> Optional[float]:
    match = re.search(r"Best Fitness:\s*([-+]?\d*\.?\d+)", output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_best_alpha(output: str) -> str:
    match = re.search(r"Best Alpha:\s*(.+)", output)
    if not match:
        return ""
    return match.group(1).strip()


def parse_top_alpha_name(rag_output: str) -> str:
    """Extract top alpha name from rag output, e.g. alpha=Alpha#44."""
    match = re.search(r"alpha\s*=\s*(Alpha#\d+)", rag_output, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def normalize_alpha101_name(alpha_name: str) -> str:
    """Normalize alpha references (e.g. Alpha#21/alpha21/21) to 3-digit Alpha101 id."""
    raw = (alpha_name or "").strip()
    if not raw:
        return ""

    match = re.search(r"alpha\s*#?\s*0*(\d{1,3})", raw, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b0*(\d{1,3})\b", raw)
    if not match:
        return raw

    number = int(match.group(1))
    if number < 1 or number > 999:
        return raw
    return f"{number:03d}"


def parse_backtest_metrics(backtest_output: str) -> Optional[dict]:
    try:
        obj = json.loads(backtest_output)
    except Exception:
        return None
    if isinstance(obj, dict) and "sharpe" in obj:
        return obj
    return None


def should_continue_tuning(score_history: list[float], attempt: int) -> bool:
    """Ask LLM whether GP tuning should continue based on score trend.

    Falls back to stopping conservatively when LLM is unavailable.
    """
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False

    trend_text = ", ".join([f"{s:.6f}" for s in score_history]) if score_history else "N/A"
    prompt = (
        "You are a GP tuning controller. Decide if we should continue tuning.\n"
        "Do not use fixed score thresholds; use trend and stability reasoning only.\n"
        "Return ONLY JSON with keys: continue (bool), reason (string).\n"
        f"Input: attempt={attempt}, score_history=[{trend_text}]"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        obj = json.loads((response.text or "").strip())
        return bool(obj.get("continue", False))
    except Exception:
        return False


def generate_evaluation_report(state: AgentState) -> str:
    """Generate a performance report using only backtesting output data."""
    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()

    backtest_summary = state.get("backtest_output", "")
    backtest_metrics = parse_backtest_metrics(backtest_summary)

    if not api_key:
        lines = [
            "# Evaluation Report",
            "## Data Source",
            "backtesting_agent output only",
        ]
        if backtest_metrics:
            lines.extend(
                [
                    "## Backtest Metrics",
                    f"- sharpe: {backtest_metrics.get('sharpe')}",
                    f"- max_drawdown_pct: {backtest_metrics.get('max_drawdown_pct')}",
                    f"- cum_returns: {backtest_metrics.get('cum_returns')}",
                ]
            )
        else:
            lines.extend(["## Backtest Metrics", "- unavailable or non-JSON output"])
        lines.extend(["## Conclusion", "LLM unavailable, returned deterministic summary."])
        return "\n".join(lines)

    payload = {
        "backtest_output": backtest_summary,
        "backtest_metrics": backtest_metrics,
    }

    prompt = (
        "You are an evaluation agent for alpha performance assessment.\n"
        "Use ONLY the backtesting data provided in payload.\n"
        "Do NOT reference route labels, RAG output, GP output, user intent, or alpha metadata.\n"
        "Generate a concise report with sections:\n"
        "1) Backtest Data Availability\n"
        "2) Performance Assessment\n"
        "3) Risk Assessment\n"
        "4) Recommendation\n"
        "Keep the report practical and avoid hallucinating missing metrics.\n"
        "If metrics are missing, explicitly say unavailable.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass

    return (
        "# Evaluation Report\n"
        "Data Source: backtesting_agent output only\n"
        "Backtest summary present but LLM report generation failed."
    )


def alpha_search_agent_node(state: AgentState) -> AgentState:
    """Alpha search agent: run RAG for alpha-related queries."""
    output = rag_search_tool.invoke(
        {
            "user_message": state["user_message"],
            "top_k": RAG_TOP_K,
            "db_path": RAG_DB_PATH,
            "collection": RAG_COLLECTION,
        }
    )
    top_alpha = parse_top_alpha_name(output)

    return {
        "user_message": state["user_message"],
        "label": state["label"],
        "rag_output": output,
        "gp_output": state.get("gp_output", ""),
        "best_alpha": top_alpha,
        "backtest_output": state.get("backtest_output", ""),
        "evaluation_report": state.get("evaluation_report", ""),
    }


def gp_agent_node(state: AgentState) -> AgentState:
    """GP agent: execute gp.py for genetic programming queries."""
    attempt_generations = [1, 3, 5]
    crossover = GP_CROSSOVER
    mutation = GP_MUTATION

    best_output = ""
    tuning_log = []
    score_history: list[float] = []
    idx = 1

    while True:
        generations = attempt_generations[min(idx - 1, len(attempt_generations) - 1)]
        output = gp_run_tool.invoke(
            {
                "user_message": state["user_message"],
                "npop": GP_NPOP,
                "generations": generations,
                "seed": GP_SEED + idx - 1,
                "crossover": crossover,
                "mutation": mutation,
                "fitness_function": GP_FITNESS_FUNCTION,
            }
        )
        score = parse_best_fitness(output)
        score_text = f"{score:.6f}" if score is not None else "N/A"
        tuning_log.append(
            f"Attempt {idx}: generations={generations}, crossover={crossover:.3f}, mutation={mutation:.3f}, best_fitness={score_text}"
        )
        best_output = output
        if score is not None:
            score_history.append(score)

        if score is None:
            break

        if not should_continue_tuning(score_history, idx):
            break

        previous_score = score_history[-2] if len(score_history) >= 2 else score
        adjust_raw = gp_adjust_params_tool.invoke(
            {
                "attempt": idx,
                "current_crossover": crossover,
                "current_mutation": mutation,
                "current_score": score,
                "previous_score": previous_score,
            }
        )
        try:
            adjust = json.loads(adjust_raw)
            crossover = float(adjust.get("crossover", crossover))
            mutation = float(adjust.get("mutation", mutation))
            reason = str(adjust.get("reason", "no_reason"))
            tuning_log.append(
                f"  Adjust -> crossover={crossover:.3f}, mutation={mutation:.3f}, reason={reason}"
            )
        except Exception:
            # Keep current params if tool payload is malformed.
            tuning_log.append("  Adjust -> skipped (invalid tool payload)")

        idx += 1

    output_with_log = "\n".join(tuning_log) + "\n\n" + best_output
    best_alpha = parse_best_alpha(best_output)

    return {
        "user_message": state["user_message"],
        "label": "genetic_programming",
        "rag_output": state.get("rag_output", ""),
        "gp_output": output_with_log,
        "best_alpha": best_alpha,
        "backtest_output": state.get("backtest_output", ""),
        "evaluation_report": state.get("evaluation_report", ""),
    }


def backtesting_agent_node(state: AgentState) -> AgentState:
    """Backtesting agent: run backtest for best alpha found by gp_agent."""
    alpha_expr = state.get("best_alpha", "").strip()
    if not alpha_expr:
        output = "Backtest skipped: no best alpha from gp_agent"
    else:
        if state.get("label") == "alpha_search":
            output = run_alpha_backtest_tool.invoke({"alpha_name": alpha_expr})
        else:
            first_output = run_backtest_tool.invoke({"alpha_expression": alpha_expr, "period": "10y"})
            first_metrics = parse_backtest_metrics(first_output)

            if first_metrics is None:
                output = first_output
            else:
                decision_raw = adjust_backtest_period_tool.invoke(
                    {
                        "sharpe": float(first_metrics.get("sharpe", 0.0)),
                        "cum_returns": float(first_metrics.get("cum_returns", 0.0)),
                        "max_drawdown_pct": float(first_metrics.get("max_drawdown_pct", 0.0)),
                        "current_period": str(first_metrics.get("period", "10y")),
                    }
                )
                try:
                    decision = json.loads(decision_raw)
                except Exception:
                    decision = {"should_retest": False, "reason": "invalid_adjustment_payload"}

                if decision.get("should_retest", False):
                    next_period = decision.get("next_period", "5y")
                    second_output = run_backtest_tool.invoke(
                        {"alpha_expression": alpha_expr, "period": next_period}
                    )
                    output = (
                        "Primary Backtest (10y):\n"
                        f"{first_output}\n\n"
                        f"Adjustment: {decision}\n\n"
                        f"Retest ({next_period}):\n"
                        f"{second_output}"
                    )
                else:
                    output = (
                        "Primary Backtest (10y):\n"
                        f"{first_output}\n\n"
                        f"Adjustment: {decision}"
                    )

    return {
        "user_message": state["user_message"],
        "label": state["label"],
        "rag_output": state.get("rag_output", ""),
        "gp_output": state.get("gp_output", ""),
        "best_alpha": alpha_expr,
        "backtest_output": output,
        "evaluation_report": state.get("evaluation_report", ""),
    }


def evaluation_agent_node(state: AgentState) -> AgentState:
    """Evaluation agent: collect previous outputs and generate a final report."""
    report = generate_evaluation_report(state)
    return {
        "user_message": state["user_message"],
        "label": state["label"],
        "rag_output": state.get("rag_output", ""),
        "gp_output": state.get("gp_output", ""),
        "best_alpha": state.get("best_alpha", ""),
        "backtest_output": state.get("backtest_output", ""),
        "evaluation_report": report,
    }


def orchestrator_route(state: AgentState) -> str:
    """Routing policy from orchestrator to specialist agents."""
    text = state["user_message"].lower()
    gp_hint = bool(
        re.search(r"\b(genetic programming|fitness function|gp|crossover|mutation|evolutionary)\b", text)
    )

    if gp_hint:
        return "gp_agent"
    if state["label"] == "alpha_search":
        return "alpha_search_agent"
    if state["label"] == "genetic_programming":
        return "gp_agent"
    return "end"


def build_multi_agent_graph():
    """Build graph with orchestrator -> specialist agents -> backtesting -> evaluation."""
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator_agent", orchestrator_agent_node)
    graph.add_node("alpha_search_agent", alpha_search_agent_node)
    graph.add_node("gp_agent", gp_agent_node)
    graph.add_node("backtesting_agent", backtesting_agent_node)
    graph.add_node("evaluation_agent", evaluation_agent_node)
    graph.set_entry_point("orchestrator_agent")
    graph.add_conditional_edges(
        "orchestrator_agent",
        orchestrator_route,
        {
            "alpha_search_agent": "alpha_search_agent",
            "gp_agent": "gp_agent",
            "end": END,
        },
    )
    graph.add_edge("alpha_search_agent", "backtesting_agent")
    graph.add_edge("gp_agent", "backtesting_agent")
    graph.add_edge("backtesting_agent", "evaluation_agent")
    graph.add_edge("evaluation_agent", END)
    return graph.compile()


def main() -> None:
    global RAG_TOP_K, RAG_DB_PATH, RAG_COLLECTION
    global GP_NPOP, GP_SEED, GP_CROSSOVER, GP_MUTATION, GP_FITNESS_FUNCTION

    parser = argparse.ArgumentParser(description="LangGraph multi-agent workflow")
    parser.add_argument("--message", required=True, help="User message to classify")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K, help="RAG top-k for alpha_search")
    parser.add_argument("--db", default=RAG_DB_PATH, help="ChromaDB path for RAG query")
    parser.add_argument("--collection", default=RAG_COLLECTION, help="Chroma collection name")
    parser.add_argument("--gp-npop", type=int, default=GP_NPOP, help="GP population size")
    parser.add_argument("--gp-seed", type=int, default=GP_SEED, help="GP seed")
    parser.add_argument("--gp-crossover", type=float, default=GP_CROSSOVER, help="Initial GP crossover")
    parser.add_argument("--gp-mutation", type=float, default=GP_MUTATION, help="Initial GP mutation")
    parser.add_argument(
        "--gp-fitness-function",
        default=GP_FITNESS_FUNCTION,
        help="GP fitness function name from fitness_function.py, e.g. pearson_fitness",
    )
    args = parser.parse_args()

    RAG_TOP_K = args.top_k
    RAG_DB_PATH = args.db
    RAG_COLLECTION = args.collection
    GP_NPOP = args.gp_npop
    GP_SEED = args.gp_seed
    GP_CROSSOVER = args.gp_crossover
    GP_MUTATION = args.gp_mutation
    GP_FITNESS_FUNCTION = args.gp_fitness_function

    agent = build_multi_agent_graph()
    result = agent.invoke(
        {
            "user_message": args.message,
            "label": "",
            "rag_output": "",
            "gp_output": "",
            "best_alpha": "",
            "backtest_output": "",
            "evaluation_report": "",
        }
    )
    print(result["label"])
    if result.get("gp_output", ""):
        print("\n=== GP Result ===")
        print(result.get("gp_output", ""))
        print("\n=== Backtest Result ===")
        print(result.get("backtest_output", ""))
        print("\n=== Evaluation Report ===")
        print(result.get("evaluation_report", ""))
    elif result["label"] == "alpha_search":
        print("\n=== RAG Search Result ===")
        print(result.get("rag_output", ""))
        print("\n=== Backtest Result ===")
        print(result.get("backtest_output", ""))
        print("\n=== Evaluation Report ===")
        print(result.get("evaluation_report", ""))


if __name__ == "__main__":
    main()
