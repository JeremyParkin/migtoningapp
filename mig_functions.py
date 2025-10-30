# mig_functions.py
from __future__ import annotations
import streamlit as st

# ----------------------------
# Sidebar / Chrome Utilities
# ----------------------------
def standard_sidebar() -> None:
    """Render branding + feedback link + session cost meter."""
    st.sidebar.image(
        "https://app.agilitypr.com/app/assets/images/agility-logo-vertical.png",
        width=180,
    )
    st.sidebar.subheader("MIG Toning App")
    st.sidebar.caption("Version: October 2025")

    # Sidebar width/overflow tweaks (keep nav readable on long titles)
    st.markdown(
        """
        <style>
          .eczjsme9, .st-emotion-cache-1wqrzgl { overflow: visible !important; max-width: 250px !important; }
          .st-emotion-cache-a8w3f8 { overflow: visible !important; }
          .st-emotion-cache-1cypcdb { max-width: 250px !important; }
          .e1mq0gaz1, .e1mq0gaz0 { filter: brightness(10); max-width: 150px !important; }
          
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Feedback link
    st.sidebar.markdown(
        "[App Feedback](https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u)"
    )

    # Session API meter
    _init_api_meter()
    cost_usd = float(st.session_state.api_meter.get("cost_usd", 0.0) or 0.0)
    st.sidebar.caption(f"Est. session cost USD${cost_usd:,.4f}")


# ----------------------------
# Programmatic Navigation
# ----------------------------
def build_pages(is_admin: bool) -> list:
    """
    Return the list of st.Page entries in the app.
    The Sentiment Tester page is included only for admins.
    Paths are relative to the repo root where `app.py` lives.
    """
    pages = [
        st.Page("pages/1-Getting_Started.py", title="Getting Started", icon=":material/flight_takeoff:"),
        st.Page("pages/2-Toning_Sample.py", title="Toning Sample", icon=":material/science:"),
        st.Page("pages/3-Toning_Configuration.py", title="Toning Configuration", icon=":material/tune:"),
        st.Page("pages/4-Toning_Interface.py", title="Toning Interface", icon=":material/dashboard_customize:"),
        st.Page("pages/5-AI_Sentiment.py", title="AI Sentiment", icon=":material/auto_awesome:"),
        st.Page("pages/6-Spot_Checks.py", title="Spot Checks", icon=":material/fact_check:"),
        st.Page("pages/8-Download.py", title="Download", icon=":material/download:"),
    ]
    if is_admin:
        pages.append(
            st.Page("pages/9-Sentiment_Tester.py", title="Sentiment Tester", icon=":material/analytics:")
        )
    return pages


def run_navigation(position: str = "sidebar") -> None:
    """
    Create the nav with the correct page set (admin-gated),
    render the standard sidebar chrome, then run the selected page.
    Call this from your root launcher (e.g., app.py).
    """
    is_admin = bool(st.session_state.get("is_admin"))
    nav = st.navigation(build_pages(is_admin), position=position)
    standard_sidebar()
    nav.run()


# ----------------------------
# Display helpers
# ----------------------------
def format_number(num: float | int) -> str:
    """Helper to display large integers with K/M/B suffixes."""
    try:
        n = float(num)
    except Exception:
        return str(num)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f} K"
    else:
        # Preserve int look if it's actually an int
        return str(int(n)) if n.is_integer() else str(n)


# ----------------------------
# OpenAI Usage Metering
# ----------------------------
_OPENAI_PRICES = {
    "gpt-5-mini":  {"in": 0.25, "out": 2.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-5": {"in": 1.25, "out": 10.00},
    # Optional extras if you ever use them:
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-5-nano":  {"in": 0.05, "out": 0.40},
}

def _init_api_meter() -> None:
    """Ensure the session usage meter object exists."""
    if "api_meter" not in st.session_state:
        st.session_state.api_meter = {
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},
        }

def add_api_usage(resp, model_name: str) -> None:
    """
    Record usage & cost from an OpenAI response object.

    Usage:
        resp = client.chat.completions.create(...)
        mig.add_api_usage(resp, model_id)
    """
    _init_api_meter()
    usage = getattr(resp, "usage", None)
    if not usage:
        return

    in_t  = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_t = int(getattr(usage, "completion_tokens", 0) or 0)

    meter = st.session_state.api_meter
    meter["in_tokens"]  += in_t
    meter["out_tokens"] += out_t

    prices = _OPENAI_PRICES.get((model_name or "").lower(), {"in": 0.15, "out": 0.60})
    in_cost  = (in_t  / 1_000_000) * prices["in"]
    out_cost = (out_t / 1_000_000) * prices["out"]
    meter["cost_usd"] += (in_cost + out_cost)

    # Per-model breakdown
    bym = meter["by_model"].setdefault(model_name, {"in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0})
    bym["in_tokens"]  += in_t
    bym["out_tokens"] += out_t
    bym["cost_usd"]   += (in_cost + out_cost)

def reset_api_meter() -> None:
    """Reset the usage meter (call this from your 'Start Over' button handler)."""
    st.session_state.api_meter = {
        "in_tokens": 0,
        "out_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }