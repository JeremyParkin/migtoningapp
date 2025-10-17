# 6-Spot_Check.py

import re
import pandas as pd
import streamlit as st
import mig_functions as mig

# --- Configure Streamlit page ---
st.set_page_config(page_title="Spot Check AI Labels",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
                   layout="wide")
mig.standard_sidebar()
st.session_state.current_page = "Spot Check"

# --- Validate required workflow steps ---
if not st.session_state.get("upload_step"):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()
if not st.session_state.get("config_step"):
    st.error("Please run the configuration step before trying this step.")
    st.stop()
if not isinstance(st.session_state.get("unique_stories"), pd.DataFrame):
    st.error("Unique stories not found. Please complete earlier steps.")
    st.stop()

# --- Ensure expected columns exist ---
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name)
    if isinstance(df, pd.DataFrame):
        if "Assigned Sentiment" not in df.columns:
            df["Assigned Sentiment"] = pd.NA
        for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col not in df.columns:
                df[col] = None
        st.session_state[df_name] = df

# --- Normalise sentiment type (3-way/5-way) ---
_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# --- Prepare keywords for highlighting ---
keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]

# --- Utility helpers ---
def escape_markdown(text: str) -> str:
    text = str(text or "")
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    url_pattern = r"https?:\/\/[^\s]+"
    def esc(part): return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)
    parts = re.split(r"(" + url_pattern + r")", text)
    return "".join(part if re.match(url_pattern, part) else esc(part) for part in parts)

def highlight_keywords(text: str, kw: list, bg="goldenrod", fg="black") -> str:
    text = str(text or "")
    if not kw:
        return text
    escaped = []
    for k in kw:
        k_esc = re.escape(k)
        escaped.append(rf"\b{k_esc}\b" if " " not in k else k_esc)
    pattern = r"(?:%s)" % "|".join(escaped)
    def repl(m): return f"<span style='background-color:{bg};color:{fg};'>{m.group(0)}</span>"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def set_assigned_sentiment(group_id, label):
    """Write the human label to both DataFrames for every row in this group."""
    st.session_state.unique_stories.loc[
        st.session_state.unique_stories["Group ID"] == group_id, "Assigned Sentiment"
    ] = label
    st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Group ID"] == group_id, "Assigned Sentiment"
    ] = label

def compute_candidates(n_to_review: int, conf_thresh: int):
    """Prioritise AI-labelled stories that warrant human review."""
    # Pool definition: AI sentiment present AND no human label yet
    df = st.session_state.unique_stories.copy()
    pool = df[df["Assigned Sentiment"].isna() & df["AI Sentiment"].notna()].copy()
    if pool.empty:
        return pool

    pool["AI_UPPER"] = pool["AI Sentiment"].astype(str).str.upper().str.strip()
    # Confidence as numeric (default 100 so it is not flagged as low-confidence)
    pool["AI_CONF"] = pd.to_numeric(pool["AI Sentiment Confidence"], errors="coerce").fillna(100)
    pool["GROUP_CT"] = pd.to_numeric(pool.get("Group Count", 1), errors="coerce").fillna(1)

    # Negative emphasis
    if sentiment_type == "3-way":
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({
            "NEGATIVE": 1.0
        }).fillna(0.0)
    else:
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({
            "VERY NEGATIVE": 1.0,
            "SOMEWHAT NEGATIVE": 0.7
        }).fillna(0.0)

    # Low confidence component
    conf_thresh = max(1, int(conf_thresh))
    pool["LOWCONF"] = (conf_thresh - pool["AI_CONF"]) / conf_thresh
    pool["LOWCONF"] = pool["LOWCONF"].clip(lower=0, upper=1)

    # Normalise group count
    max_gc = pool["GROUP_CT"].max()
    pool["GC_NORM"] = pool["GROUP_CT"] / max_gc if max_gc > 0 else 0

    # Weighted score (more weight to higher group counts)
    pool["SCORE"] = 0.45 * pool["GC_NORM"] + 0.35 * pool["NEG_SCORE"] + 0.20 * pool["LOWCONF"]

    # Sort and take the top N results
    pool = pool.sort_values(["SCORE", "GROUP_CT"], ascending=[False, False]).reset_index(drop=True)
    return pool.head(n_to_review)

# --- Controls ---
st.title("Spot Check AI Sentiment")

n_to_review = st.number_input("How many stories to spot check?", min_value=1, max_value=200, value=15, step=1)

conf_thresh = 75

candidates = compute_candidates(n_to_review, conf_thresh)

if candidates.empty:
    st.success("No stories need spot checking. (Either no AI labels yet, or everything already has a human label.)")
    st.stop()

# Maintain a sticky navigation index
st.session_state.setdefault("spot_idx", 0)
st.session_state.spot_idx = min(st.session_state.spot_idx, len(candidates)-1)
idx = st.session_state.spot_idx

row = candidates.iloc[idx]
current_group_id = row["Group ID"]
URL = str(row.get("URL", "") or "")
head_raw = row.get("Headline", "") or ""
body_raw = row.get("Snippet", "") or ""

# Highlight story text
head = escape_markdown(head_raw)
body = escape_markdown(body_raw)
highlighted_body = highlight_keywords(body, keywords)

# --- Layout ---
col1, col2 = st.columns([3,1], gap="large")

with col1:
    if URL:
        st.markdown(URL)
    st.subheader(head)
    st.markdown(highlighted_body, unsafe_allow_html=True)

with col2:
    # st.caption(f"Active label set: **{sentiment_type}**")
    st.info(f"Queue: {idx+1} of {len(candidates)}")

    # AI opinion summary
    ai_label = row.get("AI Sentiment")
    ai_conf  = row.get("AI Sentiment Confidence")
    ai_why   = row.get("AI Sentiment Rationale")
    if pd.notna(ai_label):
        st.write(f"{ai_label}")
        if pd.notna(ai_why) and str(ai_why).strip():
            st.caption(str(ai_why))
    else:
        st.warning("No AI sentiment is available on this story (it won't usually appear in this page).")


    # --- Action 1: Accept AI sentiment as human label ---
    if st.button("✅ Accept AI Sentiment", use_container_width=True):
        if pd.notna(ai_label) and str(ai_label).strip():
            set_assigned_sentiment(current_group_id, str(ai_label).strip())
            # Move to next item after accepting
            st.session_state.spot_idx = min(idx + 1, len(candidates) - 1)
            st.rerun()
        else:
            st.warning("No AI label available to accept on this item.")

    # --- Action 2: Manual override ---
    with st.form("override_form", clear_on_submit=True):
        choices = (['POSITIVE','NEUTRAL','NEGATIVE','NOT RELEVANT']
                   if sentiment_type == '3-way'
                   else ['VERY POSITIVE','SOMEWHAT POSITIVE','NEUTRAL','SOMEWHAT NEGATIVE','VERY NEGATIVE','NOT RELEVANT'])
        default_idx = choices.index('NEUTRAL') if 'NEUTRAL' in choices else 0
        picked = st.radio("Or apply a different label:", choices, index=default_idx)
        if st.form_submit_button("Apply", type="primary", use_container_width=True):
            set_assigned_sentiment(current_group_id, picked)
            st.session_state.spot_idx = min(idx + 1, len(candidates) - 1)
            st.rerun()


    # Navigation controls
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("◄ Back", disabled=(idx == 0), use_container_width=True):
            st.session_state.spot_idx = max(0, idx - 1)
            st.rerun()
    with next_col:
        if st.button("Next ►", disabled=(idx >= len(candidates)-1), use_container_width=True):
            st.session_state.spot_idx = min(len(candidates)-1, idx + 1)
            st.rerun()

# Footer hint
st.caption("This list updates automatically as you accept/override labels. Items drop out once they have an **Assigned Sentiment**.")

