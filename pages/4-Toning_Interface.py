# 4-Toning_Interface.py
import re
import json
import pandas as pd
import streamlit as st
import mig_functions as mig
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ================== Setup ==================
st.set_page_config(
    page_title="MIG Sentiment Tool",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
mig.standard_sidebar()
st.session_state.current_page = "Toning Interface"

# OpenAI client (expects st.secrets["key"])
client = OpenAI(api_key=st.secrets["key"])

# ================== Guards ==================
if not st.session_state.get("upload_step"):
    st.title("Toning Interface")
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()

if not st.session_state.get("config_step"):
    st.title("Toning Interface")
    st.error("Please run the configuration step before trying this step.")
    st.stop()

if not st.session_state.get("toning_config_step"):
    st.title("Toning Interface")
    st.error("Please complete the Toning Configuration page first.")
    st.stop()

# ================== Session pulls / normalization ==================
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])

# Normalize sentiment type once
_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# Model resolver (prefer gpt-5-mini)
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

# model_id = resolve_model_choice(st.session_state.get("model_choice", "gpt-5-mini"))
model_id = "gpt-5-nano"

# ================== Ensure columns ==================
# Human label column lives on the full dataset
if "Assigned Sentiment" not in st.session_state.df_traditional.columns:
    st.session_state.df_traditional["Assigned Sentiment"] = pd.NA

# AI + Translation columns on both dfs
for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name, pd.DataFrame())
    for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
                "Translated Headline", "Translated Body"]:
        if col not in df.columns:
            df[col] = None
    st.session_state[df_name] = df

# Story list (analyst tones per group) + counter
st.session_state.filtered_stories = st.session_state.unique_stories.copy()
st.session_state.setdefault("counter", 0)
counter = st.session_state.counter

# Keywords compiled on P3
keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]

# ================== Utils ==================
def escape_markdown(text: str) -> str:
    text = str(text or "")
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    url_pattern = r"https?:\/\/[^\s]+"
    def escape_chars(part):
        return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)
    parts = re.split(r"(" + url_pattern + r")", text)
    return "".join(part if re.match(url_pattern, part) else escape_chars(part) for part in parts)

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
    # Function-calling path
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

    # Plain text fallback
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

# ================== Layout ==================
col1, col2 = st.columns([3, 1], gap="large")

# End-of-list guard
if counter >= len(st.session_state.filtered_stories):
    st.info("You have reached the end of the stories.")
    if st.button("Back to the first story"):
        st.session_state.counter = 0
        st.rerun()
    st.stop()

# Current row
row = st.session_state.filtered_stories.iloc[counter]
current_group_id = row["Group ID"]
URL = str(row.get("URL", "") or "")
head_raw = row.get("Headline", "") or ""
body_raw = row.get("Snippet", "") or ""
count = int(row.get("Group Count", 1) or 1)

# ---- Prefer translated text if present ----
trans_head = row.get("Translated Headline")
trans_body = row.get("Translated Body")
head_to_show = trans_head if isinstance(trans_head, str) and trans_head.strip() else head_raw
body_to_show = trans_body if isinstance(trans_body, str) and trans_body.strip() else body_raw

# Escape + highlight
head = escape_markdown(head_to_show)
body = escape_markdown(body_to_show)
highlighted_body = highlight_keywords(body, keywords)

# -------- Left: Story --------
with col1:
    if URL: st.markdown(URL)
    st.subheader(f"{head}")
    st.markdown(highlighted_body, unsafe_allow_html=True)

# -------- Right: Tools / Opinion / Nav --------
with col2:
    # Translate button
    tcol1, _ = st.columns(2)
    with tcol1:
        if st.button("Translate"):
            try:
                th = translate(head_raw) if (head_raw or "").strip() else None
                tb = translate(body_raw) if (body_raw or "").strip() else None

                # Persist to ALL DFs by Group ID (so it survives reruns)
                for df_name in ["unique_stories", "df_traditional"]:
                    df = st.session_state[df_name]
                    mask = df["Group ID"] == current_group_id
                    df.loc[mask, ["Translated Headline", "Translated Body"]] = [th, tb]

                # (Optional) also update the in-memory filtered row this frame
                st.session_state.filtered_stories.at[counter, "Translated Headline"] = th
                st.session_state.filtered_stories.at[counter, "Translated Body"] = tb

                st.rerun()
            except Exception as e:
                st.error(f"Translation failed: {e}")

    # Label selection (HUMAN)
    with st.form("Sentiment Selector"):
        if sentiment_type == "3-way":
            choices = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
            default_idx = 1
        else:
            choices = ["VERY POSITIVE","SOMEWHAT POSITIVE","NEUTRAL","SOMEWHAT NEGATIVE","VERY NEGATIVE","NOT RELEVANT"]
            default_idx = 2
        sentiment_choice = st.radio("Sentiment Choice", choices, index=default_idx)
        submitted = st.form_submit_button("Confirm Sentiment", type="primary")

    st.info(f"Grouped Stories: {count}")

    if submitted:
        st.session_state.df_traditional.loc[
            st.session_state.df_traditional["Group ID"] == current_group_id, "Assigned Sentiment"
        ] = sentiment_choice
        st.session_state.counter = min(len(st.session_state.filtered_stories) - 1, st.session_state.counter + 1)
        st.rerun()

    # Navigation
    prev_button, next_button = st.columns(2)
    with prev_button:
        if st.button("◄ Back", disabled=(st.session_state.counter == 0)):
            st.session_state.counter = max(0, st.session_state.counter - 1)
            st.rerun()
    with next_button:
        if st.button("Next ►", disabled=(st.session_state.counter == len(st.session_state.filtered_stories) - 1)):
            st.session_state.counter = min(len(st.session_state.filtered_stories) - 1, st.session_state.counter + 1)
            st.rerun()

    # Progress
    numbers, progress = st.columns(2)
    with progress:
        assigned_articles_count = st.session_state.df_traditional["Assigned Sentiment"].notna().sum()
        percent_done = assigned_articles_count / max(1, len(st.session_state.df_traditional))
        st.metric("Total done", "{:.1%}".format(percent_done), "")
    with numbers:
        total_stories = len(st.session_state.unique_stories)
        st.metric("Unique story", f"{counter + 1}/{total_stories}", "")

    # Jump back at end
    if (counter + 1) == len(st.session_state.filtered_stories):
        if st.button("Back to the first story"):
            st.session_state.counter = 0
            st.rerun()

    # ---------- AI Opinion (standardized storage + Regenerate) ----------
    sentiment_placeholder = st.empty()
    story_prompt = build_story_prompt(head_raw, body_raw)

    # Regenerate button clears AI fields for this group
    regen = st.button("↻ Regenerate AI opinion", key=f"regen_{current_group_id}")
    if regen:
        for df_name in ["filtered_stories", "unique_stories", "df_traditional"]:
            df = st.session_state.get(df_name)
            if isinstance(df, pd.DataFrame):
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [None, None, None]

    # Read latest values (after optional clear)
    row_now = st.session_state.filtered_stories.iloc[counter]
    ai_label = row_now.get("AI Sentiment")
    ai_conf  = row_now.get("AI Sentiment Confidence")
    ai_rsn   = row_now.get("AI Sentiment Rationale")

    if ai_label and not regen:
        with sentiment_placeholder.container():
            st.write(f"**{ai_label}**")
            if ai_rsn:
                st.caption(str(ai_rsn))
    else:
        with st.spinner("Generating AI opinion..."):
            ai_result = call_ai_sentiment(story_prompt)
        if ai_result:
            label = ai_result.get("sentiment")
            conf  = ai_result.get("confidence")
            why   = ai_result.get("explanation")

            # Persist to ALL DFs for this Group ID
            for df_name in ["filtered_stories", "unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [label, conf, why]

            with sentiment_placeholder.container():
                st.write(f"**{label or '(none)'}**")
                if why:
                    st.caption(str(why))

    # Already-assigned human label?
    assigned_sentiment = st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Group ID"] == current_group_id, "Assigned Sentiment"
    ].iloc[0]
    if pd.notna(assigned_sentiment):
        st.info(f"Assigned Sentiment: {assigned_sentiment}")
