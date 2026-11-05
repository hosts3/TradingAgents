#!/usr/bin/env python3
"""AAPL analysis using TradingAgents with Ollama cloud model.

Uses the official TradingAgentsGraph.propagate() method which handles:
- Memory log resolution & storage
- Structured state logging to disk (JSON)
- Proper signal processing & report generation
- Checkpoint support for resumable runs
"""
import os, sys, datetime, json

os.environ["OPENAI_API_KEY"] = "ollama"
sys.path.insert(0, os.path.expanduser("~/TradingAgents"))

LOG = os.path.expanduser("~/trading_aapl_run.log")

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("TradingAgents: AAPL Analysis Starting (using propagate())")
log("=" * 60)

log("Testing LLM connection...")
from tradingagents.llm_clients.factory import create_llm_client
client = create_llm_client(provider='ollama', model='glm-5.1:cloud')
llm = client.get_llm()
log(f"LLM ready: {llm.model_name} @ {llm.openai_api_base}")

try:
    from langchain_core.messages import HumanMessage
    resp = llm.invoke([HumanMessage(content="Say OK if you can hear me.")])
    log(f"LLM response: {resp.content[:80]}")
except Exception as e:
    log(f"LLM test FAILED: {e}")
    sys.exit(1)

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "glm-5.1:cloud"
config["quick_think_llm"] = "glm-5.1:cloud"

selected_analysts = ["market", "social", "news", "fundamentals"]

log(f"Creating graph: {config['llm_provider']} / {config['deep_think_llm']}")
graph = TradingAgentsGraph(selected_analysts, config=config, debug=False)

ticker = "AAPL"
analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

log(f"Running propagate() for {ticker} @ {analysis_date}...")
log(f"This will: fetch data → run analysts → debate → risk analysis → produce final decision")
log(f"Estimated time: 15-30 minutes with a cloud model. Please be patient.")

try:
    final_state, decision = graph.propagate(ticker, analysis_date)

    log(f"\n{'='*60}")
    log(f"🎯 FINAL DECISION FOR {ticker}: {decision}")
    log(f"{'='*60}")

    # _log_state() saves the state directly as a flat dict (not nested under date key)
    # The JSON file is: ~/.tradingagents/logs/AAPL/TradingAgentsStrategy_logs/full_states_log_YYYY-MM-DD.json
    results_dir = os.path.expanduser(f"~/.tradingagents/logs/{ticker}")
    json_log = os.path.join(
        results_dir,
        "TradingAgentsStrategy_logs",
        f"full_states_log_{analysis_date}.json",
    )

    if os.path.exists(json_log):
        log(f"📄 Full state JSON log: {json_log}")
        with open(json_log) as f:
            state = json.load(f)
        # state is a flat dict — the date key wrapping was from the old code
        if not isinstance(state, dict):
            log(f"⚠️ Unexpected state type: {type(state)}")
            state = {}
    else:
        log(f"⚠️ JSON log not found at {json_log}")
        state = final_state if isinstance(final_state, dict) else {}

    # Generate the markdown report
    report_dir = os.path.join(results_dir, analysis_date)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "report.md")

    with open(report_path, "w") as f:
        f.write(f"# {ticker} Analysis Report\n\n")
        f.write(f"**Date:** {analysis_date}\n\n")
        f.write(f"**Final Decision:** {decision}\n\n")
        f.write("---\n\n")

        report_keys = [
            ("market_report", "Market Report"),
            ("sentiment_report", "Sentiment Report"),
            ("news_report", "News Report"),
            ("fundamentals_report", "Fundamentals Report"),
            ("trader_investment_plan", "Trader Investment Plan"),
            ("investment_plan", "Investment Plan"),
            ("final_trade_decision", "Final Trade Decision"),
        ]

        for key, title in report_keys:
            val = state.get(key, "")
            if val and str(val).strip():
                f.write(f"## {title}\n\n")
                f.write(str(val) + "\n\n")

        # Investment debate
        debate_state = state.get("investment_debate_state", {})
        if debate_state and isinstance(debate_state, dict):
            f.write("## Investment Debate\n\n")
            judge = debate_state.get("judge_decision", "")
            if judge:
                f.write(f"**Judge Decision:** {judge[:500]}...\n\n")

        # Risk debate
        risk_state = state.get("risk_debate_state", {})
        if risk_state and isinstance(risk_state, dict):
            f.write("## Risk Debate\n\n")
            judge = risk_state.get("judge_decision", "")
            if judge:
                f.write(f"**Judge Decision:** {judge[:500]}...\n\n")

    log(f"📄 Report saved: {report_path}")

except KeyboardInterrupt:
    log("⛔ Analysis interrupted by user (Ctrl+C)")
    sys.exit(1)
except Exception as e:
    log(f"❌ Analysis failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log("✅ Analysis complete!")