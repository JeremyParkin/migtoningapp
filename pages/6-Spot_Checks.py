# 6-Spot_Checks.py

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

# ==================== Page ====================
# st.set_page_config(
#     page_title="Spot Check AI Labels",
#     page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
#     layout="wide",
# )
st.markdown("<style>.block-container{padding-top:3rem !important;}</style>", unsafe_allow_html=True)
st.session_state.current_page = "Spot Check"

# ==================== Guards ====================
if not st.session_state.get("upload_step"):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()
if not st.session_state.get("config_step"):
    st.error("Please run the configuration step before trying this step.")
    st.stop()
if not isinstance(st.session_state.get("unique_stories"), pd.DataFrame):
    st.error("Unique stories not found. Please complete earlier steps.")
    st.stop()

# ==================== OpenAI / Config ====================
client = OpenAI(api_key=st.secrets["key"])
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])
model_id = st.session_state.get("model_choice", "gpt-5-mini")

SECOND_OPINION_MODEL = "gpt-5"
st.session_state.setdefault("spot_ai_model_override", None)

_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# Ensure columns exist
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name, pd.DataFrame())
    if "Assigned Sentiment" not in df.columns:
        df["Assigned Sentiment"] = pd.NA
    for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
                "Translated Headline", "Translated Body"]:
        if col not in df.columns:
            df[col] = None
    st.session_state[df_name] = df

# ==================== State ====================
st.session_state.setdefault("initial_ai_label", {})         # {group_id: "LABEL"}
st.session_state.setdefault("spot_checked_groups", set())    # set of group_ids
st.session_state.setdefault("accepted_initial", set())       # subset of above
st.session_state.setdefault("spot_ai_loading", False)
st.session_state.setdefault("spot_ai_refresh_requested", False)
st.session_state.setdefault("spot_idx", 0)

# ==================== Priority Weights ====================
W_GROUP = 0.40
W_NEG   = 0.35
W_IMP   = 0.15
W_LOWCF = 0.10
CONF_THRESH = 75  # confidence threshold for "low confidence" term

# ==================== Helpers ====================
def escape_markdown(text: str) -> str:
    text = str(text or "")
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    url_pattern = r"https?:\/\/[^\s]+"
    def esc(part: str) -> str:
        return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)
    parts = re.split(r"(" + url_pattern + r")", text)
    return "".join(part if re.match(url_pattern, part) else esc(part) for part in parts)

def _simple_highlight(text: str, kw: list, bg="goldenrod", fg="black") -> str:
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

keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
tolerant_pat_str = st.session_state.get("highlight_regex_str")

def highlight_with_tolerant_regex(text: str, fallback_keywords: list, bg="goldenrod", fg="black") -> str:
    s = str(text or "")
    pat_str = tolerant_pat_str
    if pat_str:
        try:
            rx = re.compile(pat_str)  # (?i) embedded by builder
            return rx.sub(lambda m: f"<span style='background-color:{bg};color:{fg};'>{m.group(0)}</span>", s)
        except re.error:
            pass
    return _simple_highlight(s, fallback_keywords, bg=bg, fg=fg)

def split_text(text, limit=700, sentence_limit=350):
    sentences = re.split(r"(?<=[.!?])\s+", str(text or ""))
    chunks, current = [], ""
    for s in sentences:
        s = s or ""
        while len(s) > sentence_limit:
            part, s = s[:sentence_limit], s[sentence_limit:]
            current = (current + " " + part).strip() if current else part
            if len(current) >= limit:
                chunks.append(current); current = ""
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

def call_ai_sentiment(story_prompt: str, model_override: str | None = None):
    model_to_use = model_override or model_id
    try:
        if functions:
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                    {"role": "user", "content": story_prompt},
                ],
                functions=functions,
                function_call={"name": "analyze_sentiment"},
            )
            mig.add_api_usage(resp, model_to_use)
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
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                {"role": "user", "content": story_prompt},
            ],
        )
        mig.add_api_usage(resp, model_to_use)
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
    st.session_state.unique_stories.loc[
        st.session_state.unique_stories["Group ID"] == group_id, "Assigned Sentiment"
    ] = label
    st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Group ID"] == group_id, "Assigned Sentiment"
    ] = label

def get_group_count(gid: int) -> int:
    """Prefer precomputed 'Group Count' on unique_stories; fallback to counting in df_traditional."""
    try:
        us = st.session_state.unique_stories
        val = us.loc[us["Group ID"] == gid, "Group Count"]
        if len(val) and pd.notna(val.iloc[0]):  # use first available
            return int(val.iloc[0])
    except Exception:
        pass
    try:
        return int((st.session_state.df_traditional["Group ID"] == gid).sum())
    except Exception:
        return 1

# ==================== Candidate Scoring ====================
def compute_candidates(conf_thresh: int) -> pd.DataFrame:
    us = st.session_state.unique_stories.copy()
    dt = st.session_state.df_traditional

    # Exclude any group that already has a human label in either DF
    assigned_us = set(us.loc[us["Assigned Sentiment"].notna(), "Group ID"].unique())
    assigned_dt = set(dt.loc[dt["Assigned Sentiment"].notna(), "Group ID"].unique())
    assigned_any = assigned_us | assigned_dt

    pool = us[(~us["Group ID"].isin(assigned_any)) & (us["AI Sentiment"].notna())].copy()
    if pool.empty:
        return pool

    pool["AI_UPPER"] = pool["AI Sentiment"].astype(str).str.upper().str.strip()
    pool["AI_CONF"]  = pd.to_numeric(pool["AI Sentiment Confidence"], errors="coerce").fillna(100)
    pool["GROUP_CT"] = pd.to_numeric(pool.get("Group Count", 1), errors="coerce").fillna(1)

    # Negativity signal
    if sentiment_type == "3-way":
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({"NEGATIVE": 1.0}).fillna(0.0)
    else:
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({"VERY NEGATIVE": 1.0, "SOMEWHAT NEGATIVE": 0.7}).fillna(0.0)

    # Low-confidence factor (only below threshold)
    conf_thresh = max(1, int(conf_thresh))
    pool["LOWCONF"] = ((conf_thresh - pool["AI_CONF"]) / conf_thresh).clip(lower=0, upper=1)

    # Group size (normalized)
    max_gc = pool["GROUP_CT"].max()
    pool["GC_NORM"] = (pool["GROUP_CT"] / max_gc) if max_gc > 0 else 0

    # Impressions (normalized) — optional
    imp_col = next((c for c in ["Impressions","impressions","IMPRESSIONS"] if c in pool.columns), None)
    if imp_col:
        pool["_IMP"] = pd.to_numeric(pool[imp_col], errors="coerce").fillna(0)
        max_imp = pool["_IMP"].max()
        pool["IMP_NORM"] = (pool["_IMP"] / max_imp) if max_imp > 0 else 0.0
    else:
        pool["IMP_NORM"] = 0.0

    # Composite score (0..1)
    pool["SCORE"] = (
        W_GROUP*pool["GC_NORM"] +
        W_NEG*pool["NEG_SCORE"] +
        W_IMP*pool["IMP_NORM"] +
        W_LOWCF*pool["LOWCONF"]
    )

    # Sort priority
    pool = pool.sort_values(["SCORE", "GROUP_CT", "IMP_NORM"], ascending=[False, False, False]).reset_index(drop=True)
    return pool

# ==================== Build Candidate List ====================
candidates = compute_candidates(CONF_THRESH)

if candidates.empty:
    st.success("All set — no remaining stories need spot checks.")
    checked = len(st.session_state.spot_checked_groups)
    accepted = len(st.session_state.accepted_initial)
    rate = (accepted / checked) if checked else 0.0
    m1, m2 = st.columns(2)
    with m1: st.metric("Spot-checked", checked)
    with m2: st.metric("Acceptance rate", f"{rate:.0%}")
    st.stop()

st.session_state.spot_idx = min(st.session_state.spot_idx, len(candidates) - 1)
idx = st.session_state.spot_idx
row = candidates.iloc[idx]

current_group_id = int(row["Group ID"])
URL = str(row.get("URL", "") or "")
head_raw = row.get("Headline", "") or ""
body_raw = row.get("Snippet", "") or ""

# Prefer translated text
trans_head = row.get("Translated Headline")
trans_body = row.get("Translated Body")
head_to_show = trans_head if (isinstance(trans_head, str) and trans_head.strip()) else head_raw
body_to_show = trans_body if (isinstance(trans_body, str) and trans_body.strip()) else body_raw

# Escape + tolerant highlight
head_display = escape_markdown(head_to_show)
body_display = escape_markdown(body_to_show)
highlighted_head = highlight_with_tolerant_regex(head_display, keywords)
highlighted_body = highlight_with_tolerant_regex(body_display, keywords)

# ==================== Sidebar: Translate ====================
with sidebar:
    if st.button("Translate"):
        try:
            from_text_h = head_raw or ""
            from_text_b = body_raw or ""
            th = translate(from_text_h) if from_text_h.strip() else None
            tb = translate(from_text_b) if from_text_b.strip() else None
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["Translated Headline", "Translated Body"]] = [th, tb]
            st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {e}")

# ==================== Layout ====================
col1, col2 = st.columns([3, 1], gap="large")

with col1:
    st.markdown(f"### {highlighted_head}", unsafe_allow_html=True)
    st.markdown(highlighted_body, unsafe_allow_html=True)
    st.divider()
    if URL:
        st.markdown(URL)

with col2:
    # Group size badge
    group_count = get_group_count(current_group_id)
    st.caption(f"Group size: {group_count} stor{'y' if group_count == 1 else 'ies'}")

    # ---------- AI Opinion ----------
    # sentiment_placeholder = st.empty()
    story_prompt = build_story_prompt(head_raw, body_raw)

    ai_label = row.get("AI Sentiment")
    ai_rsn   = row.get("AI Sentiment Rationale")

    # Capture the initial AI label once for acceptance-rate stats
    if ai_label and current_group_id not in st.session_state.initial_ai_label:
        st.session_state.initial_ai_label[current_group_id] = str(ai_label).strip()

    # Fetch if missing
    if (not ai_label) and (not st.session_state.spot_ai_loading):
        st.session_state.spot_ai_loading = True
        st.session_state.spot_ai_refresh_requested = True
        st.rerun()

    if st.session_state.spot_ai_loading and st.session_state.spot_ai_refresh_requested:
        with st.spinner("Generating AI opinion..."):
            ai_result = call_ai_sentiment(
                story_prompt,
                model_override=st.session_state.spot_ai_model_override
            )
        if ai_result:
            label = ai_result.get("sentiment")
            conf  = ai_result.get("confidence")
            why   = ai_result.get("explanation")
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [label, conf, why]
            if current_group_id not in st.session_state.initial_ai_label and label:
                st.session_state.initial_ai_label[current_group_id] = str(label).strip()
        st.session_state.spot_ai_loading = False
        st.session_state.spot_ai_refresh_requested = False
        st.session_state.spot_ai_model_override = None
        st.rerun()

    # Re-read after potential refresh
    row_fresh = st.session_state.unique_stories.loc[
        st.session_state.unique_stories["Group ID"] == current_group_id
    ].iloc[0]
    ai_label = row_fresh.get("AI Sentiment")
    ai_rsn   = row_fresh.get("AI Sentiment Rationale")


    # DIRECT render (no placeholder)
    if st.session_state.spot_ai_loading:
        st.info("AI is working…")
    elif ai_label:
        st.write(f"**{ai_label}**")
        if ai_rsn:
            st.caption(str(ai_rsn))
    else:
        st.caption("No AI opinion yet.")

    # --- Accept / Second opinion ---
    acc_col, sec_col = st.columns(2)
    accept_disabled = (not bool(ai_label)) or st.session_state.spot_ai_loading
    accept_help = None if not accept_disabled else "AI label not ready yet."

    def _rebuild_and_advance(final_label: str):
        # Write human label
        set_assigned_sentiment(current_group_id, final_label)
        # Stats
        st.session_state.spot_checked_groups.add(current_group_id)
        init_map = st.session_state.initial_ai_label
        if (current_group_id in init_map and
            final_label.strip().upper() == str(init_map[current_group_id]).strip().upper()):
            st.session_state.accepted_initial.add(current_group_id)
        # Rebuild queue and move to next valid index
        new_cands = compute_candidates(CONF_THRESH)
        if new_cands.empty:
            st.rerun()
        else:
            st.session_state.spot_idx = min(idx, len(new_cands) - 1)
            st.rerun()

    with acc_col:
        if st.button("✅ Accept opinion", disabled=accept_disabled, help=accept_help, use_container_width=True):
            if ai_label:
                _rebuild_and_advance(str(ai_label).strip())

    with sec_col:
        if st.button("↻ Second opinion", disabled=st.session_state.spot_ai_loading, use_container_width=True):
            st.session_state.spot_ai_loading = True
            st.session_state.spot_ai_refresh_requested = True
            st.session_state.spot_ai_model_override = SECOND_OPINION_MODEL
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state.get(df_name)
                if isinstance(df, pd.DataFrame):
                    mask = df["Group ID"] == current_group_id
                    df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [None, None, None]
            st.rerun()

    # --- Manual override buttons ---
    st.caption("Or pick a different label:")
    if sentiment_type == "5-way":
        manual_labels = [
            "VERY NEGATIVE", "SOMEWHAT NEGATIVE", "NEUTRAL",
            "SOMEWHAT POSITIVE", "VERY POSITIVE", "NOT RELEVANT",
        ]
        palette = {
            "VERY NEGATIVE": "#c0392b",
            "SOMEWHAT NEGATIVE": "#e67e22",
            "NEUTRAL": "#f1c40f",
            "SOMEWHAT POSITIVE": "#2ecc71",
            "VERY POSITIVE": "#27ae60",
            "NOT RELEVANT": "#7f8c8d",
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
        # use a STABLE key per label (no group_id)
        stable_key = f"manbtn_{label.replace(' ', '_')}"
        with stylable_container(key=f"wrap_{stable_key}", css_styles=css):
            return st.button(label, key=stable_key, use_container_width=True)


    clicked_override = None
    for lbl in manual_labels:
        if colored_button(lbl, key=lbl):
            clicked_override = lbl
            break


    if clicked_override:
        _rebuild_and_advance(clicked_override)

    st.divider()

    # --- Navigation ---
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("◄ Back", disabled=(idx == 0), use_container_width=True):
            st.session_state.spot_idx = max(0, idx - 1)
            st.rerun()
    with next_col:
        if st.button("Next ►", disabled=(idx >= len(candidates) - 1), use_container_width=True):
            st.session_state.spot_idx = min(len(candidates) - 1, idx + 1)
            st.rerun()

    # --- Metrics (two rows) ---
    checked = len(st.session_state.spot_checked_groups)
    accepted = len(st.session_state.accepted_initial)
    rate = (accepted / checked) if checked else 0.0

    remaining_cnt = len(candidates)  # unique stories still eligible right now
    priority_score = float(row.get("SCORE", 0.0))  # composite score 0..1

    m1, m2 = st.columns(2)
    with m1: st.metric("Spot-checked", checked)
    with m2: st.metric("Acceptance rate", f"{rate:.0%}")
    m3, m4 = st.columns(2)
    with m3: st.metric("Remaining", remaining_cnt)
    with m4: st.metric("Priority score", f"{priority_score:.0%}")


# Footer hint
st.caption(
    "This list updates automatically as you accept/override labels. "
    "Items drop out once they get an **Assigned Sentiment**. "
    "Acceptance rate is measured against the original AI label."
)
