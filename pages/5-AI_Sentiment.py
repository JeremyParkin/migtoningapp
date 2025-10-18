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

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="Bulk AI Sentiment",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)

# Standard sidebar (reads running totals from session_state)
mig.standard_sidebar()
st.session_state.current_page = "Bulk AI Toning"

client = OpenAI(api_key=st.secrets["key"])

# --- If we just reran to refresh the sidebar, show the last batch summary now ---
_last = st.session_state.pop("__last_batch_summary__", None)
if _last:
    st.success(f"Completed AI toning for {_last['done']} group(s) in {_last['elapsed']:.1f}s.")
    st.caption(f"Token usage this batch (if reported): input={_last['in_tok']:,} • output={_last['out_tok']:,}")

# --- Guards ---
if not st.session_state.get("upload_step"):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()
if not st.session_state.get("config_step"):
    st.error("Please run the Configuration step before trying this step.")
    st.stop()
if not st.session_state.get("toning_config_step"):
    st.error("Please complete the Toning Configuration step before running bulk AI toning.")
    st.stop()

# --- Config pulls ---
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])
model_id = (st.session_state.get("model_choice") or "gpt-5-mini").strip()

# --- Sentiment set ---
_raw_st = st.session_state.get("sentiment_type") or st.session_state.get("ui_sentiment_type") or "3-way"
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# --- Ensure columns ---
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name, pd.DataFrame())
    for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
        if col not in df.columns:
            df[col] = None
    st.session_state[df_name] = df

# --- Sidebar usage counters (fallback if mig.add_api_usage isn't wired) ---
st.session_state.setdefault("api_tokens_in", 0)
st.session_state.setdefault("api_tokens_out", 0)
st.session_state.setdefault("api_cost_usd", 0.0)

def get_prices(model: str) -> tuple[float, float]:
    tbl = getattr(mig, "_OPENAI_PRICES", {})
    p = tbl.get((model or "").strip())
    if not p:
        return 0.25, 2.00
    return float(p.get("in", 0.25)), float(p.get("out", 2.00))

def apply_usage_to_session(in_tok: int, out_tok: int, model: str):
    st.session_state.api_tokens_in += int(in_tok or 0)
    st.session_state.api_tokens_out += int(out_tok or 0)
    pin, pout = get_prices(model)
    st.session_state.api_cost_usd += (in_tok / 1_000_000) * pin + (out_tok / 1_000_000) * pout
    # Optional: central log, if implemented
    try:
        class _U:  # minimal shim
            prompt_tokens = int(in_tok or 0)
            completion_tokens = int(out_tok or 0)
        class _R: usage = _U()
        mig.add_api_usage(_R(), model)
    except Exception:
        pass

# --- Helpers ---
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
    def parse_plain_text(txt: str):
        cand3 = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
        cand5 = ["VERY POSITIVE","SOMEWHAT POSITIVE","NEUTRAL","SOMEWHAT NEGATIVE","VERY NEGATIVE","NOT RELEVANT"]
        cand = cand3 if sentiment_type == "3-way" else cand5
        sent = next((c for c in cand if re.search(rf"\b{re.escape(c)}\b", txt)), None)
        m = re.search(r"confidence[^0-9]{0,10}(\d{1,3})", txt, flags=re.I)
        conf = max(0, min(100, int(m.group(1)))) if m else None
        return sent, conf, txt

    # Function-calling path
    if functions:
        try:
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
            in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            if getattr(choice.message, "function_call", None):
                fc = choice.message.function_call
                if fc and fc.name == "analyze_sentiment":
                    args = json.loads(fc.arguments or "{}")
                    return {
                        "sentiment": args.get("sentiment"),
                        "confidence": args.get("confidence"),
                        "explanation": args.get("explanation"),
                        "in_tok": in_tok, "out_tok": out_tok,
                    }
        except Exception:
            pass

    # Plain text fallback
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
            {"role": "user", "content": story_prompt},
        ],
    )
    txt = resp.choices[0].message.content.strip()
    in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
    sent, conf, why = parse_plain_text(txt)
    return {"sentiment": sent, "confidence": conf, "explanation": why, "in_tok": in_tok, "out_tok": out_tok}

# --- Compute remaining groups ---
human_labeled_groups = set(
    st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Assigned Sentiment"].notna(), "Group ID"
    ].unique()
)
already_ai_groups = set(
    st.session_state.unique_stories.loc[
        st.session_state.unique_stories["AI Sentiment"].notna(), "Group ID"
    ].unique()
)
remaining_mask = (~st.session_state.unique_stories["Group ID"].isin(human_labeled_groups)) & \
                 (~st.session_state.unique_stories["Group ID"].isin(already_ai_groups))
remaining = st.session_state.unique_stories.loc[remaining_mask].reset_index(drop=False)

# ==================== UI ====================
st.title("Bulk AI Sentiment Toning")
st.caption(f"Active label set: **{sentiment_type}** • Model: **{model_id}**")

# Controls (batch size + buttons under it)
batch_size = st.number_input(
    "Batch size", min_value=1, max_value=2000,
    value=min(25, max(1, len(remaining))), step=25
)
run_clicked = st.button("▶️ Run next batch", type="primary", disabled=(len(remaining) == 0))
reset_ai_clicked = st.button("🧹 Clear AI results (keep human labels)")

st.divider()

if reset_ai_clicked:
    for df_name in ["unique_stories", "df_traditional"]:
        df = st.session_state[df_name]
        df.loc[:, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = None
        st.session_state[df_name] = df
    st.success("Cleared AI results. Human labels remain unchanged.")
    st.stop()

# --- Run a batch ---
if run_clicked:
    batch_df = remaining.head(batch_size).copy()
    if batch_df.empty:
        st.info("No eligible groups found to process.")
        st.stop()

    MAX_WORKERS = 8
    lock = Lock()
    progress_bar = st.progress(0.0)
    stats = {"done": 0, "total": len(batch_df)}
    total_in, total_out = 0, 0

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

    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(work, row): i for i, row in batch_df.iterrows()}
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
            total_in  += int(res.get("in_tok", 0) or 0)
            total_out += int(res.get("out_tok", 0) or 0)

            gid = out["group_id"]

            # Update both DataFrames for this group
            st.session_state.unique_stories.loc[
                st.session_state.unique_stories["Group ID"] == gid,
                ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]
            ] = [ai_label, ai_conf, ai_rsn]
            st.session_state.df_traditional.loc[
                st.session_state.df_traditional["Group ID"] == gid,
                ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]
            ] = [ai_label, ai_conf, ai_rsn]

    elapsed = time.time() - start
    progress_bar.progress(1.0)

    # Update usage totals (MAIN THREAD), stash summary, then force a rerun so sidebar re-renders
    apply_usage_to_session(total_in, total_out, model_id)
    st.session_state["__last_batch_summary__"] = {
        "done": stats["done"], "elapsed": elapsed, "in_tok": total_in, "out_tok": total_out
    }
    st.rerun()
else:
    st.info("Ready. Set your batch size, then click **Run next batch**.")

# Bottom status
remaining_mask = (~st.session_state.unique_stories["Group ID"].isin(
    st.session_state.df_traditional.loc[st.session_state.df_traditional["Assigned Sentiment"].notna(), "Group ID"]
)) & (~st.session_state.unique_stories["Group ID"].isin(
    st.session_state.unique_stories.loc[st.session_state.unique_stories["AI Sentiment"].notna(), "Group ID"]
))
remaining_now = st.session_state.unique_stories.loc[remaining_mask]
st.caption(f"Groups remaining (no human label & no AI): **{len(remaining_now)}**")
