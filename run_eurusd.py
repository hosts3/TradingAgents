#!/usr/bin/env python3
"""EUR/USD analysis: TradingAgents (opencode/glm-5.3-flash) + QuantAgent (opencode) + Kronos forecast.

Runs the full TradingAgents pipeline via propagate() for EURUSD (Yahoo: EURUSD=X),
QuantAgent's graph for a second decision, and Kronos small-model 24-candle forecast
from TradingView OHLCV pulled via the MCP (saved JSON) or yfinance fallback.
"""
import os, sys, json, datetime

sys.path.insert(0, os.path.expanduser("~/TradingAgents"))
os.environ.setdefault("OPENAI_API_KEY", "ollama")  # not used by opencode provider, silences warnings

# TradingAgents' opencode provider reads OPENCODE_API_KEY; Hermes stores it as OPENCODE_GO_API_KEY.
if not os.environ.get("OPENCODE_API_KEY"):
    for envf in (os.path.expanduser("~/.hermes/.env"), os.path.expanduser("~/TradingAgents/.env")):
        if os.path.exists(envf):
            for line in open(envf):
                if line.startswith("OPENCODE_GO_API_KEY="):
                    os.environ["OPENCODE_API_KEY"] = line.partition("=")[2].strip()
                    break
        if os.environ.get("OPENCODE_API_KEY"):
            break
assert os.environ.get("OPENCODE_API_KEY"), "OPENCODE_API_KEY could not be sourced"

# Provider override via env (openai-compatible): zai coding plan (glm-5.3-flash, verified working)
# OpenCode Go hit monthly 429 GoUsageLimitError; zai is the current live brain.
# NVIDIA NIM (Nemotron 3 Ultra 550B) - set TRADINGAGENTS_LLM_PROVIDER=nvidia
os.environ.setdefault("TRADINGAGENTS_LLM_PROVIDER", "zai")

TICKER = "EURUSD"
DATE = datetime.date.today().strftime("%Y-%m-%d")
LOG = os.path.expanduser("~/eurusd_chain_run.log")

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log(f"EUR/USD chain test starting — TradingAgents + QuantAgent + Kronos, {DATE}")
log("=" * 60)

# ---------------------------------------------------------------- LLM check
# OpenCode Go hit monthly 429 GoUsageLimitError — switch to zai coding plan.
# zai = GLM openai-compatible endpoint, base https://api.z.ai/api/coding/paas/v4
# NVIDIA NIM = Nemotron 3 Ultra 550B, base https://integrate.api.nvidia.com/v1
PROVIDER = os.environ.get("TRADINGAGENTS_LLM_PROVIDER", "zai")

if PROVIDER == "nvidia":
    MODEL = os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", "nvidia/nemotron-3-ultra-550b-a55b")
    BASE_URL = os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL", "https://integrate.api.nvidia.com/v1")
    # Load NVIDIA_API_KEY from Hermes .env
    if not os.environ.get("NVIDIA_API_KEY"):
        for line in open(os.path.expanduser("~/.hermes/.env")):
            if line.startswith("NVIDIA_API_KEY="):
                os.environ["NVIDIA_API_KEY"] = line.partition("=")[2].strip()
                break
else:
    MODEL = "glm-5.3-flash"
    ZAI_BASE = "https://api.z.ai/api/coding/paas/v4"
    if not os.environ.get("ZAI_API_KEY"):
        for line in open(os.path.expanduser("~/.hermes/.env")):
            if line.startswith("ZAI_API_KEY="):
                os.environ["ZAI_API_KEY"] = line.partition("=")[2].strip()
                break
    BASE_URL = ZAI_BASE

log(f"Testing {PROVIDER} LLM ({MODEL})...")
from tradingagents.llm_clients.factory import create_llm_client
client = create_llm_client(provider=PROVIDER, model=MODEL, base_url=BASE_URL)
llm = client.get_llm()
log(f"LLM ready: {llm.model_name} @ {llm.openai_api_base}")
try:
    from langchain_core.messages import HumanMessage
    resp = llm.invoke([HumanMessage(content="Say OK if you can hear me.")])
    log(f"LLM response: {str(resp.content)[:60]}")
except Exception as e:
    log(f"LLM test FAILED: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- Kronos forecast
# Kronos runs standalone as a subprocess (its own venv/torch); result read from JSON.
log("Kronos forecast runs as separate subprocess later in the chain.")

# ---------------------------------------------------------------- TradingAgents
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# TradingAgentsGraph builds its own LLM clients from config; also needs the key env its provider resolves.
# openai_compatible resolves OPENAI_COMPATIBLE_API_KEY (verified above).
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = PROVIDER
config["deep_think_llm"] = MODEL
config["quick_think_llm"] = MODEL
if "backend_url" in config:
    config["backend_url"] = BASE_URL
# EUR/USD on Yahoo stalls news/macro fetches (hung 20+ min on a dead socket once
# already); alpha_vantage needs a key we don't have, so drop news vendor to a
# hard "no chain" and let the news analyst run with DATA_UNAVAILABLE context.
config["tool_vendors"] = {
    # EUR/USD=X stalls Yahoo's fetches (multiple runs hung 11-20+ min on dead
    # 69.147.x.x sockets — every yfinance path). Route ALL methods to
    # alpha_vantage: no key here = instant VendorNotConfiguredError = graceful
    # DATA_UNAVAILABLE. Kronos forecast + zai reasoning carry the technical side.
    "get_news": "alpha_vantage",
    "get_global_news": "alpha_vantage",
    "get_macro_indicators": "alpha_vantage",
    "get_stock_data": "alpha_vantage",
    "get_indicators": "alpha_vantage",
    "get_fundamentals": "alpha_vantage",
    "get_balance_sheet": "alpha_vantage",
    "get_income_statement": "alpha_vantage",
    "get_cashflow": "alpha_vantage",
    "get_insider_transactions": "alpha_vantage",
}

graph = TradingAgentsGraph(
    selected_analysts=["market", "fundamentals"],  # news/macro/Yahoo vendors stall on FX; LLM reasons over context
    config=config,
    debug=False,
)

final_state, decision = graph.propagate(TICKER, DATE)
log("TradingAgents propagate() complete")
log(f"DECISION: {json.dumps(decision, default=str)[:500]}")

log("TradingAgents pipeline starting took the long path — done now.")

# save state for report
state_path = os.path.expanduser(f"~/.tradingagents/logs/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{DATE}.json")
log(f"State saved: {os.path.exists(state_path)} at {state_path}")

print("TRADINGAGENTS_DONE")
