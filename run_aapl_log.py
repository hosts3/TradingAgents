#!/usr/bin/env python3
"""Headless AAPL analysis - writes output to file for monitoring."""
import os, sys, datetime, logging

os.environ["OPENAI_API_KEY"] = "ollama"
sys.path.insert(0, os.path.expanduser("~/TradingAgents"))

LOG = os.path.expanduser("~/trading_aapl.log")

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("TradingAgents: AAPL Analysis Starting")
log("=" * 60)

# Test LLM first
log("Testing LLM connection...")
from tradingagents.llm_clients.factory import create_llm_client
client = create_llm_client(provider='ollama', model='glm-5.1:cloud')
llm = client.get_llm()
log(f"LLM ready: {llm.model_name} @ {llm.openai_api_base}")

# Quick LLM test
try:
    from langchain_core.messages import HumanMessage
    resp = llm.invoke([HumanMessage(content="Say OK if you can hear me.")])
    log(f"LLM test response: {resp.content[:50]}")
except Exception as e:
    log(f"LLM test FAILED: {e}")
    sys.exit(1)

# Import graph
log("Importing TradingAgentsGraph...")
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "glm-5.1:cloud"
config["quick_think_llm"] = "glm-5.1:cloud"

selected_analysts = ["market", "social", "news", "fundamentals"]

log(f"Creating graph with {config['llm_provider']} / {config['deep_think_llm']}...")
graph = TradingAgentsGraph(selected_analysts, config=config, debug=True)

ticker = "AAPL"
analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

log(f"Creating initial state for {ticker}...")
init_state = graph.propagator.create_initial_state(ticker, analysis_date)
args = graph.propagator.get_graph_args()

log(f"Starting stream for {ticker} @ {analysis_date}...")

step = 0
trace = []
for chunk in graph.graph.stream(init_state, **args):
    step += 1
    keys = list(chunk.keys())
    log(f"[Step {step}] {keys}")

    for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report",
                 "investment_debate_state", "trader_investment_plan",
                 "risk_debate_state", "final_trade_decision"]:
        if chunk.get(key):
            val = str(chunk[key])
            log(f"  → {key}: {val[:100]}...")

    trace.append(chunk)

    # Log every 10 steps with progress
    if step % 10 == 0:
        log(f"  ({step} steps completed, continuing...)")

if trace:
    final_state = trace[-1]
    decision_raw = final_state.get("final_trade_decision", "")
    decision = graph.process_signal(decision_raw) if decision_raw else "N/A"
    log(f"\n🎯 FINAL DECISION FOR {ticker}: {decision}")

    # Save report
    results_dir = os.path.expanduser(f"~/.tradingagents/logs/{ticker}/{analysis_date}")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "report.md")

    with open(report_path, "w") as f:
        f.write(f"# {ticker} Analysis Report\n\nDate: {analysis_date}\n\n")
        f.write(f"**Final Decision: {decision}**\n\n")
        for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report",
                     "trader_investment_plan", "final_trade_decision"]:
            if final_state.get(key):
                f.write(f"\n## {key.replace('_', ' ').title()}\n\n")
                f.write(str(final_state[key]) + "\n")

    log(f"📄 Report saved: {report_path}")
    log(f"📄 Log file: {LOG}")
else:
    log("❌ No results received from analysis!")

log("✅ Analysis complete!")