# 5-Bulk_AI_Toning.py

import re
import json
import time
import pandas as pd
import streamlit as st
import mig_functions as mig
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI

# ================== Setup ==================
st.set_page_config(
    page_title="Bulk AI Sentiment",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
mig.standard_sidebar()
st.session_state.current_page = "Bulk AI Toning"

client = OpenAI(api_key=st.secrets["key"])

# ================== Guards ==================
if not st.session_state.get("upload_step"):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()
if not st.session_state.get("config_step"):
    st.error("Please run the Configuration step before trying this step.")
    st.stop()
if not st.session_state.get("toning_config_step"):
    st.error("Please complete the Toning Configuration step before running bulk AI toning.")
    st.stop()

# ================== Session pulls ==================
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])

# Label set (3-way/5-way) from config page
_raw_st = st.session_state.get("sentiment_type") or st.session_state.get("ui_sentiment_type") or "3-way"
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# Ensure result columns exist in both DFs
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name, pd.DataFrame())
    for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
        if col not in df.columns:
            df[col] = None
    st.session_state[df_name] = df

# ================== Helpers ==================
def build_story_prompt(headline: str, snippet: str) -> str:
    parts = []
    if pre_prompt: parts.append(pre_prompt)
    if sentiment_instruction: parts.append(sentiment_instruction)
    if post_prompt: parts.append(post_prompt)
    parts.append("This is the news story:")
    parts.append(f"HEADLINE: {headline or ''}")
    parts.append(f"BODY: {snippet or ''}")
    return "\n\n".join(parts)

def call_ai_sentiment(story_prompt: str):
    """
    Try legacy function-calling first using your saved schema.
    Fallback to plain text and light parse.
    Returns dict: {sentiment, confidence, explanation, usage?}
    """
    # 1) Function-calling path (uses your Page-3 `functions` schema)
    if functions:
        try:
            resp = client.chat.completions.create(
                model="gpt-5-mini",
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
                        "sentiment": args.get("sentiment"),
                        "confidence": args.get("confidence"),
                        "explanation": args.get("explanation"),
                        "usage": getattr(resp, "usage", None),
                    }
        except Exception:
            # fall through to plaintext
            pass

    # 2) Plain text fallback
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
            {"role": "user", "content": story_prompt},
        ],
    )
    txt = resp.choices[0].message.content.strip()

    candidates_3 = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
    candidates_5 = [
        "VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL",
        "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"
    ]
    cand = candidates_3 if sentiment_type == "3-way" else candidates_5

    sent = next((c for c in cand if re.search(rf"\b{re.escape(c)}\b", txt)), None)
    m = re.search(r"confidence[^0-9]{0,10}(\d{1,3})", txt, flags=re.I)
    conf = max(0, min(100, int(m.group(1)))) if m else None
    return {"sentiment": sent, "confidence": conf, "explanation": txt, "usage": getattr(resp, "usage", None)}

# ================== Determine remaining groups ==================
# Groups with ANY human label in df_traditional are excluded from AI
human_labeled_groups = set(
    st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Assigned Sentiment"].notna(), "Group ID"
    ].unique()
)

# We also skip groups where an AI Sentiment already exists (to avoid duplicates)
already_ai_groups = set(
    st.session_state.unique_stories.loc[
        st.session_state.unique_stories["AI Sentiment"].notna(), "Group ID"
    ].unique()
)

# Remaining = no human label AND no existing AI
remaining_mask = (~st.session_state.unique_stories["Group ID"].isin(human_labeled_groups)) & \
                 (~st.session_state.unique_stories["Group ID"].isin(already_ai_groups))
remaining = st.session_state.unique_stories.loc[remaining_mask].reset_index(drop=False)  # keep original index

st.title("Bulk AI Sentiment Toning")
st.caption(f"Active label set: **{sentiment_type}**  •  Model: **gpt-5-mini**")
st.write(f"Groups remaining for AI (no human label & no prior AI): **{len(remaining)}**")

run_clicked = st.button("Run AI on All Remaining Groups", type="primary", disabled=(len(remaining) == 0))
reset_ai_clicked = st.button("Clear AI Results (keep human labels)")

if reset_ai_clicked:
    for df_name in ["unique_stories", "df_traditional"]:
        df = st.session_state[df_name]
        df.loc[:, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = None
        st.session_state[df_name] = df
    st.success("Cleared AI results. Human labels remain unchanged.")

# ================== Run ==================
if run_clicked:
    MAX_WORKERS = 8   # fixed, best-practice default
    lock = Lock()
    progress_bar = st.progress(0.0)
    stats = {"done": 0, "total": len(remaining), "input_tokens": 0, "output_tokens": 0}
    start = time.time()

    def work(row):
        gid = row["Group ID"]
        headline = row.get("Headline", "")
        snippet = row.get("Snippet", "")
        prompt = build_story_prompt(headline, snippet)
        try:
            result = call_ai_sentiment(prompt)
            return {"group_id": gid, "result": result, "error": None}
        except Exception as e:
            return {"group_id": gid, "result": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, row): i for i, row in remaining.iterrows()}
        for fut in as_completed(futures):
            out = fut.result()

            with lock:
                stats["done"] += 1
                progress_bar.progress(stats["done"] / max(1, stats["total"]))

            if out["error"]:
                st.warning(f"Group {out['group_id']}: {out['error']}")
                continue

            res = out["result"] or {}
            ai_label = res.get("sentiment")
            ai_conf  = res.get("confidence")
            ai_rsn   = res.get("explanation")

            # Update token usage if available
            usage = res.get("usage")
            if usage:
                try:
                    stats["input_tokens"]  += getattr(usage, "prompt_tokens", 0) or 0
                    stats["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                except Exception:
                    pass

            gid = out["group_id"]

            # Update unique_stories for this group
            st.session_state.unique_stories.loc[
                st.session_state.unique_stories["Group ID"] == gid,
                ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]
            ] = [ai_label, ai_conf, ai_rsn]

            # Update all rows in df_traditional for this group
            st.session_state.df_traditional.loc[
                st.session_state.df_traditional["Group ID"] == gid,
                ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]
            ] = [ai_label, ai_conf, ai_rsn]

    progress_bar.progress(1.0)
    elapsed = time.time() - start
    st.success(f"Completed AI toning for {stats['done']} group(s) in {elapsed:.1f}s.")
    st.caption(f"Token usage (if reported): input={stats['input_tokens']}, output={stats['output_tokens']}")

    with st.expander("Preview: Unique Stories (updated)"):
        st.dataframe(
            st.session_state.unique_stories.sort_values(by=["Group ID"]).reset_index(drop=True)
        )

    with st.expander("Preview: Full Dataset (df_traditional) sample"):
        st.dataframe(st.session_state.df_traditional.head(50))

