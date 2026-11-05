#!/usr/bin/env python3
"""Generate a markdown report from the TradingAgents JSON state log."""
import json, os, sys

date = sys.argv[1] if len(sys.argv) > 1 else "2026-05-01"
ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

json_path = os.path.expanduser(
    f"~/.tradingagents/logs/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json"
)

if not os.path.exists(json_path):
    print(f"ERROR: {json_path} not found")
    sys.exit(1)

with open(json_path) as f:
    state = json.load(f)

# The _log_state method saves the state dict directly (not nested under a date key)
# So state IS the dict we need

report_dir = os.path.expanduser(f"~/.tradingagents/logs/{ticker}/{date}")
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "report.md")

decision = state.get("final_trade_decision", "N/A")

with open(report_path, "w") as f:
    f.write(f"# {ticker} Analysis Report\n\n")
    f.write(f"**Date:** {date}\n\n")
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

    # Investment debate state
    debate = state.get("investment_debate_state", {})
    if debate and isinstance(debate, dict):
        f.write("## Investment Debate\n\n")
        for subkey, subtitle in [
            ("judge_decision", "Judge Decision"),
        ]:
            v = debate.get(subkey, "")
            if v and str(v).strip():
                f.write(f"**{subtitle}:** {str(v)[:500]}...\n\n")

    # Risk debate state
    risk = state.get("risk_debate_state", {})
    if risk and isinstance(risk, dict):
        f.write("## Risk Debate\n\n")
        judge = risk.get("judge_decision", "")
        if judge and str(judge).strip():
            f.write(f"**Judge Decision:** {str(judge)[:500]}...\n\n")

print(f"✅ Report saved to {report_path}")
print(f"📄 Decision: {decision[:100]}")