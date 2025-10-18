# 6-Spot_Check.py

import re
import json
import pandas as pd
import streamlit as st
from streamlit import sidebar

import mig_functions as mig
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from streamlit_extras.stylable_container import stylable_container

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="Spot Check AI Labels",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
# Global CSS: pad the top
st.markdown(
    "<style>.block-container{padding-top:3rem !important;}</style>",
    unsafe_allow_html=True
)

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

# --- OpenAI client ---
client = OpenAI(api_key=st.secrets["key"])

# --- Pull config / prompts reused from Toning ---
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])

# --- Normalise sentiment type (3-way/5-way) ---
_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# --- Model resolver (same as Toning; default gpt-5-mini) ---
def resolve_model_choice(choice: str) -> str:
    if not choice:
        return "gpt-5-mini"
    c = str(choice).strip().lower()
    if "gpt-5-mini" in c:
        return "gpt-5-mini"
    if "gpt-4o" in c and "mini" in c:
        return "gpt-4o-mini"
    if c in {"gpt-4o", "gpt4o", "gpt 4o"}:
        return "gpt-4o"
    return "gpt-5-mini"

model_id = resolve_model_choice(st.session_state.get("model_choice", "gpt-5-mini"))

# --- Ensure expected columns exist + translation cols ---
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name)
    if isinstance(df, pd.DataFrame):
        if "Assigned Sentiment" not in df.columns:
            df["Assigned Sentiment"] = pd.NA
        for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
                    "Translated Headline", "Translated Body"]:
            if col not in df.columns:
                df[col] = None
        st.session_state[df_name] = df

# --- Prepare keywords for highlighting ---
keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]

# --- AI loading flags (for disabling refresh/accept) ---
st.session_state.setdefault("spot_ai_loading", False)
st.session_state.setdefault("spot_ai_refresh_requested", False)

# --- Utility helpers (match Toning Interface) ---
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

def split_text(text, limit=700, sentence_limit=350):
    sentences = re.split(r"(?<=[.!?])\s+", str(text or ""))
    chunks, current = [], ""
    for s in sentences:
        s = s or ""
        while len(s) > sentence_limit:
            part, s = s[:sentence_limit], s[sentence_limit:]
            if current: chunks.append(current); current = part
            else: current = part
        if len(current) + len(s) <= limit: current += (" " if current else "") + s
        else: chunks.append(current); current = s
    if current: chunks.append(current)
    return chunks

def translate_concurrently(chunks):
    translator = GoogleTranslator(source="auto", target="en")
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = [(i, ex.submit(translator.translate, c)) for i, c in enumerate(chunks)]
        for i, fut in futures:
            try: results[i] = fut.result()
            except Exception as e: results[i] = f"Error: {e}"
    return results

def translate(text):
    return " ".join(translate_concurrently(split_text(text)))

def build_story_prompt(headline: str, snippet: str) -> str:
    parts = []
    if pre_prompt: parts.append(pre_prompt)
    if sentiment_instruction: parts.append(sentiment_instruction)
    if post_prompt: parts.append(post_prompt)
    parts.append("This is the news story:")
    parts.append(f"HEADLINE: {headline}")
    parts.append(f"BODY: {snippet}")
    return "\n\n".join(parts)

def call_ai_sentiment(story_prompt: str):
    """Function-call first; plain-text fallback."""
    try:
        if functions:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                    {"role": "user", "content": story_prompt},
                ],
                functions=functions,
                function_call={"name": "analyze_sentiment"},
            )
            choice = resp.choices[0]
            if getattr(choice.message, "function_call", None):
                fc = choice.message.function_call
                if fc and fc.name == "analyze_sentiment":
                    args = json.loads(fc.arguments or "{}")
                    return {
                        "named_entity": args.get("named_entity"),
                        "sentiment": args.get("sentiment"),
                        "confidence": args.get("confidence"),
                        "explanation": args.get("explanation"),
                    }
    except Exception as e:
        st.caption(f"Function-calling fallback used due to: {e}")

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                {"role": "user", "content": story_prompt},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        cand3 = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
        cand5 = ["VERY POSITIVE","SOMEWHAT POSITIVE","NEUTRAL","SOMEWHAT NEGATIVE","VERY NEGATIVE","NOT RELEVANT"]
        cand = cand3 if sentiment_type == "3-way" else cand5
        sent = next((c for c in cand if re.search(rf"\b{re.escape(c)}\b", txt)), None)
        m = re.search(r"confidence[^0-9]{0,10}(\d{1,3})", txt, flags=re.I)
        conf = max(0, min(100, int(m.group(1)))) if m else None
        return {"named_entity": None, "sentiment": sent, "confidence": conf, "explanation": txt}
    except Exception as e:
        st.error(f"AI sentiment failed: {e}")
        return None

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
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({"NEGATIVE": 1.0}).fillna(0.0)
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
# st.title("Spot Check AI Sentiment")

with sidebar:
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

# Prefer translated text when available
trans_head = row.get("Translated Headline")
trans_body = row.get("Translated Body")
head_to_show = trans_head if isinstance(trans_head, str) and trans_head.strip() else head_raw
body_to_show = trans_body if isinstance(trans_body, str) and trans_body.strip() else body_raw

# Highlight story text
head = escape_markdown(head_to_show)
body = escape_markdown(body_to_show)
highlighted_body = highlight_keywords(body, keywords)

# --- Sidebar: Translate ---
with sidebar:
    if st.button("Translate"):
        try:
            from_text_h = head_raw or ""
            from_text_b = body_raw or ""
            th = translate(from_text_h) if from_text_h.strip() else None
            tb = translate(from_text_b) if from_text_b.strip() else None
            # Persist to both dataframes by Group ID
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["Translated Headline", "Translated Body"]] = [th, tb]
            # Force rerun to render translated preference
            st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {e}")

# --- Layout ---
col1, col2 = st.columns([3,1], gap="large")

with col1:
    st.subheader(head)
    st.markdown(highlighted_body, unsafe_allow_html=True)
    st.divider()
    if URL:
        st.markdown(URL)

with col2:
    # st.info(f"Queue: {idx+1} of {len(candidates)}")

    # ---------- AI opinion (standardised storage + controlled loading) ----------
    sentiment_placeholder = st.empty()
    story_prompt = build_story_prompt(head_raw, body_raw)

    # Current cached AI
    ai_label = row.get("AI Sentiment")
    ai_rsn   = row.get("AI Sentiment Rationale")

    # If there's no AI yet and we're not already loading, trigger a first load
    if (not ai_label) and (not st.session_state.spot_ai_loading):
        st.session_state.spot_ai_loading = True
        st.session_state.spot_ai_refresh_requested = True
        st.rerun()

    # If a refresh/first-load is requested, do the call here
    if st.session_state.spot_ai_loading and st.session_state.spot_ai_refresh_requested:
        with st.spinner("Generating AI opinion..."):
            ai_result = call_ai_sentiment(story_prompt)
        if ai_result:
            label = ai_result.get("sentiment")
            conf  = ai_result.get("confidence")
            why   = ai_result.get("explanation")
            # Persist to both DFs for this group
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [label, conf, why]
        # flip flags and rerun
        st.session_state.spot_ai_loading = False
        st.session_state.spot_ai_refresh_requested = False
        st.rerun()

    # Re-read fresh values after any refresh
    row = st.session_state.unique_stories.loc[st.session_state.unique_stories["Group ID"] == current_group_id].iloc[0]
    ai_label = row.get("AI Sentiment")
    ai_rsn   = row.get("AI Sentiment Rationale")

    with sentiment_placeholder.container():
        if st.session_state.spot_ai_loading:
            st.info("AI is working…")
        elif ai_label:
            st.write(f"**{ai_label}**")
            if ai_rsn:
                st.caption(str(ai_rsn))
        else:
            st.caption("No AI opinion yet.")

    # --- Accept + Second opinion side-by-side (disable while loading; Accept also disabled if no AI yet) ---
    acc_col, sec_col = st.columns(2)
    accept_disabled = (not bool(ai_label)) or st.session_state.spot_ai_loading
    accept_help = None if not accept_disabled else "AI label not ready yet."

    with acc_col:
        if st.button("✅ Accept opinion", disabled=accept_disabled, help=accept_help, use_container_width=True):
            if ai_label:
                set_assigned_sentiment(current_group_id, str(ai_label).strip())
                # advance to next candidate
                st.session_state.spot_idx = min(idx + 1, len(candidates) - 1)
                st.rerun()

    with sec_col:
        if st.button("↻ Second opinion", disabled=st.session_state.spot_ai_loading, use_container_width=True):
            st.session_state.spot_ai_loading = True
            st.session_state.spot_ai_refresh_requested = True
            # Clear cached AI fields for this group
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state.get(df_name)
                if isinstance(df, pd.DataFrame):
                    mask = df["Group ID"] == current_group_id
                    df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [None, None, None]
            st.rerun()

    # --- Stacked, color-coded manual sentiment buttons ---
    st.caption("Or pick a different label:")
    if sentiment_type == "5-way":
        manual_labels = [
            "VERY NEGATIVE",
            "SOMEWHAT NEGATIVE",
            "NEUTRAL",
            "SOMEWHAT POSITIVE",
            "VERY POSITIVE",
            "NOT RELEVANT",
        ]
        palette = {
            "VERY NEGATIVE": "#c0392b",  # deep red
            "SOMEWHAT NEGATIVE": "#e67e22",  # orange
            "NEUTRAL": "#f1c40f",        # yellow
            "SOMEWHAT POSITIVE": "#2ecc71",  # green
            "VERY POSITIVE": "#27ae60",  # deep green
            "NOT RELEVANT": "#7f8c8d",   # grey
        }
    else:
        manual_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE", "NOT RELEVANT"]
        palette = {
            "NEGATIVE": "#e74c3c",
            "NEUTRAL": "#f1c40f",
            "POSITIVE": "#2ecc71",
            "NOT RELEVANT": "#7f8c8d",
        }

    def colored_button(label: str, key: str):
        css = f"""
        button {{
            width: 100%;
            background-color: {palette[label]} !important;
            color: black !important;
            border: 0;
            padding: 0.2rem 0.6rem;
            font-weight: bold !important;
            font-size: 14px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        """
        with stylable_container(key=f"btn_{key}", css_styles=css):
            return st.button(label, key=key)

    clicked_override = None
    for lbl in manual_labels:
        if colored_button(lbl, key=f"{lbl.replace(' ', '_')}_{current_group_id}"):
            clicked_override = lbl
            break

    if clicked_override:
        set_assigned_sentiment(current_group_id, clicked_override)
        st.session_state.spot_idx = min(idx + 1, len(candidates) - 1)
        st.rerun()

    st.divider()

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

    st.info(f"Queue: {idx + 1} of {len(candidates)}")

# Footer hint
st.caption("This list updates automatically as you accept/override labels. Items drop out once they have an **Assigned Sentiment**.")
