# 4-Toning_Interface.py

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
    page_title="MIG Sentiment Tool",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
# Global CSS: add top padding
st.markdown("<style>.block-container{padding-top:3rem !important;}</style>", unsafe_allow_html=True)

# mig.standard_sidebar()
st.session_state.current_page = "Toning Interface"

# --- OpenAI client (expects st.secrets["key"]) ---
client = OpenAI(api_key=st.secrets["key"])

# --- Guards: required workflow steps ---
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

# --- Pull session config ---
pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])
model_id = st.session_state.get("model_choice", "gpt-5-mini")

# Force second opinion to GPT-5 (pricing already in mig_functions)
SECOND_OPINION_MODEL = "gpt-5"
st.session_state.setdefault("ai_model_override", None)  # one-shot override holder

# --- Normalise sentiment type (3-way vs 5-way) ---
_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

# --- Ensure required columns exist (human + AI + translation) ---
if "Assigned Sentiment" not in st.session_state.df_traditional.columns:
    st.session_state.df_traditional["Assigned Sentiment"] = pd.NA

for df_name in ["unique_stories", "df_traditional"]:
    df = st.session_state.get(df_name, pd.DataFrame())
    for col in [
        "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
        "Translated Headline", "Translated Body"
    ]:
        if col not in df.columns:
            df[col] = None
    st.session_state[df_name] = df

# --- Working list + index ---
st.session_state.filtered_stories = st.session_state.unique_stories.copy()
st.session_state.setdefault("counter", 0)
counter = st.session_state.counter

# --- Highlighting inputs (display list kept for fallback) ---
keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
tolerant_pat_str = st.session_state.get("highlight_regex_str")  # built/saved by config page

# --- AI state flags (disable buttons during work) ---
st.session_state.setdefault("ai_loading", False)
st.session_state.setdefault("ai_refresh_requested", False)

# ===================== Utils =====================
def escape_markdown(text: str) -> str:
    """Escape Markdown special chars outside of URLs for safe st.markdown rendering."""
    text = str(text or "")
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    url_pattern = r"https?:\/\/[^\s]+"

    def escape_chars(part: str) -> str:
        return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)

    parts = re.split(r"(" + url_pattern + r")", text)
    return "".join(part if re.match(url_pattern, part) else escape_chars(part) for part in parts)

def _simple_highlight(text: str, kw: list, bg="goldenrod", fg="black") -> str:
    """Fallback highlighter: exact keyword/phrase with word-ish boundaries."""
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

def highlight_with_tolerant_regex(text: str, fallback_keywords: list, bg="goldenrod", fg="black") -> str:
    """Use tolerant regex string from config; fallback to simple highlighter."""
    s = str(text or "")
    pat_str = tolerant_pat_str
    if pat_str:
        try:
            # The pattern string already embeds (?i) for case-insensitive
            rx = re.compile(pat_str)
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
            if current:
                chunks.append(current)
                current = part
            else:
                current = part
        if len(current) + len(s) <= limit:
            current += (" " if current else "") + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def translate_concurrently(chunks):
    translator = GoogleTranslator(source="auto", target="en")
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = [(i, ex.submit(translator.translate, c)) for i, c in enumerate(chunks)]
        for i, fut in futures:
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = f"Error: {e}"
    return results

def translate(text):
    return " ".join(translate_concurrently(split_text(text)))

def build_story_prompt(headline: str, snippet: str) -> str:
    parts = []
    if pre_prompt:
        parts.append(pre_prompt)
    if sentiment_instruction:
        parts.append(sentiment_instruction)
    if post_prompt:
        parts.append(post_prompt)
    parts.append("This is the news story:")
    parts.append(f"HEADLINE: {headline}")
    parts.append(f"BODY: {snippet}")
    return "\n\n".join(parts)

def call_ai_sentiment(story_prompt: str, model_override: str | None = None):
    """Generate sentiment using selected model; allow one-shot override (e.g., GPT-5 for 'Second opinion')."""
    model_to_use = model_override or model_id
    # 1) Function-calling path
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

    # 2) Plain-text fallback
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

def get_group_count(gid: int) -> int:
    """Return how many rows belong to this Group ID.
    Prefer the precomputed 'Group Count' on unique_stories; fallback to df_traditional."""
    try:
        # try the precomputed column on the current row first
        gc = row.get("Group Count")
        if pd.notna(gc) and str(gc).strip():
            return int(gc)
    except Exception:
        pass
    # fallback: count in the full dataset
    try:
        return int((st.session_state.df_traditional["Group ID"] == gid).sum())
    except Exception:
        return 1


# ===================== Layout =====================
col1, col2 = st.columns([5, 2], gap="large")

# --- End-of-list guard ---
if counter >= len(st.session_state.filtered_stories):
    st.info("You have reached the end of the stories.")
    if st.button("Back to the first story"):
        st.session_state.counter = 0
        st.rerun()
    st.stop()

# --- Active story ---
row = st.session_state.filtered_stories.iloc[counter]
current_group_id = row["Group ID"]
URL = str(row.get("URL", "") or "")
head_raw = row.get("Headline", "") or ""
body_raw = row.get("Snippet", "") or ""
count = int(row.get("Group Count", 1) or 1)
group_count = get_group_count(current_group_id)

# Prefer translated text when available
trans_head = row.get("Translated Headline")
trans_body = row.get("Translated Body")
head_to_show = trans_head if isinstance(trans_head, str) and trans_head.strip() else head_raw
body_to_show = trans_body if isinstance(trans_body, str) and trans_body.strip() else body_raw

# Escape + tolerant highlight (headline & body)
head_display = escape_markdown(head_to_show)
body_display = escape_markdown(body_to_show)
highlighted_head = highlight_with_tolerant_regex(head_display, keywords)
highlighted_body = highlight_with_tolerant_regex(body_display, keywords)

# ===== Sidebar: Translate =====
with sidebar:
    if st.button("Translate"):
        try:
            th = translate(head_raw) if (head_raw or "").strip() else None
            tb = translate(body_raw) if (body_raw or "").strip() else None
            for df_name in ["unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["Translated Headline", "Translated Body"]] = [th, tb]
            st.session_state.filtered_stories.at[counter, "Translated Headline"] = th
            st.session_state.filtered_stories.at[counter, "Translated Body"] = tb
            st.rerun()
        except Exception as e:
            st.error(f"Translation failed: {e}")

# ===== Left column: Story =====
with col1:
    # Use markdown heading so highlights render (subheader doesn't render HTML)
    st.markdown(f"### {highlighted_head}", unsafe_allow_html=True)
    st.markdown(highlighted_body, unsafe_allow_html=True)
    st.divider()
    if URL:
        st.markdown(URL)

# ===== Right column: Tools, labels, navigation =====
with col2:
    st.caption(f"Group size: {group_count} stor{'y' if group_count == 1 else 'ies'}")

    # ---------- AI Opinion (controlled loading) ----------
    sentiment_placeholder = st.empty()
    story_prompt = build_story_prompt(head_raw, body_raw)

    # Current cached AI values
    row_now = st.session_state.filtered_stories.iloc[counter]
    ai_label = row_now.get("AI Sentiment")
    ai_conf  = row_now.get("AI Sentiment Confidence")
    ai_rsn   = row_now.get("AI Sentiment Rationale")

    # If there's no AI yet and we're not already loading, trigger initial load (default model)
    if (not ai_label) and (not st.session_state.ai_loading):
        st.session_state.ai_loading = True
        st.session_state.ai_refresh_requested = True
        st.rerun()

    # Handle refresh / initial fetch here (buttons disabled while working)
    if st.session_state.ai_loading and st.session_state.ai_refresh_requested:
        with st.spinner("Generating AI opinion..."):
            ai_result = call_ai_sentiment(
                story_prompt,
                model_override=st.session_state.ai_model_override  # may force gpt-5 for second opinion
            )
        if ai_result:
            label = ai_result.get("sentiment")
            conf  = ai_result.get("confidence")
            why   = ai_result.get("explanation")
            for df_name in ["filtered_stories", "unique_stories", "df_traditional"]:
                df = st.session_state[df_name]
                mask = df["Group ID"] == current_group_id
                df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [label, conf, why]

        # Reset flags & override, then rerun to render enabled buttons
        st.session_state.ai_loading = False
        st.session_state.ai_refresh_requested = False
        st.session_state.ai_model_override = None
        st.rerun()

    # Read again after possible refresh
    row_now = st.session_state.filtered_stories.iloc[counter]
    ai_label = row_now.get("AI Sentiment")
    ai_rsn   = row_now.get("AI Sentiment Rationale")

    with sentiment_placeholder.container():
        if st.session_state.ai_loading:
            st.info("AI is working…")
        elif ai_label:
            st.write(f"**{ai_label}**")
            if ai_rsn:
                st.caption(str(ai_rsn))
        else:
            st.caption("No AI opinion yet.")

    # --- Accept + Second opinion (side-by-side) ---
    acc_col, sec_col = st.columns(2)
    accept_disabled = (not bool(ai_label)) or st.session_state.ai_loading
    accept_help = None if not accept_disabled else "AI label not ready yet."

    with acc_col:
        # inside 4-Toning_Interface.py, in the Accept button handler
        if st.button("✅ Accept opinion", disabled=accept_disabled, help=accept_help):
            if ai_label:
                # write to BOTH dataframes
                st.session_state.unique_stories.loc[
                    st.session_state.unique_stories["Group ID"] == current_group_id, "Assigned Sentiment"
                ] = ai_label
                st.session_state.df_traditional.loc[
                    st.session_state.df_traditional["Group ID"] == current_group_id, "Assigned Sentiment"
                ] = ai_label
                # keep the working view aligned if you use it
                st.session_state.filtered_stories.loc[
                    st.session_state.filtered_stories["Group ID"] == current_group_id, "Assigned Sentiment"
                ] = ai_label

                st.session_state.counter = min(
                    len(st.session_state.filtered_stories) - 1,
                    st.session_state.counter + 1
                )
                st.rerun()

        # if st.button("✅ Accept opinion", disabled=accept_disabled, help=accept_help):
        #     if ai_label:
        #         mask_trad = st.session_state.df_traditional["Group ID"] == current_group_id
        #         st.session_state.df_traditional.loc[mask_trad, "Assigned Sentiment"] = ai_label
        #         st.session_state.counter = min(
        #             len(st.session_state.filtered_stories) - 1,
        #             st.session_state.counter + 1
        #         )
        #         st.rerun()

    with sec_col:
        # Second opinion = force GPT-5 once, then reset override after call completes
        if st.button("↻ Second opinion", disabled=st.session_state.ai_loading):
            st.session_state.ai_loading = True
            st.session_state.ai_refresh_requested = True
            st.session_state.ai_model_override = SECOND_OPINION_MODEL  # <- force GPT-5 for next run
            # Clear cached AI fields for this group so the UI shows the spinner & disables Accept
            for df_name in ["filtered_stories", "unique_stories", "df_traditional"]:
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
        with stylable_container(key=f"btn_{key}", css_styles=css):
            return st.button(label, key=key)

    clicked_override = None
    for lbl in manual_labels:
        if colored_button(lbl, key=f"{lbl.replace(' ', '_')}_{current_group_id}"):
            clicked_override = lbl
            break

    if clicked_override:
        mask_trad = st.session_state.df_traditional["Group ID"] == current_group_id
        st.session_state.df_traditional.loc[mask_trad, "Assigned Sentiment"] = clicked_override
        st.session_state.counter = min(len(st.session_state.filtered_stories) - 1, st.session_state.counter + 1)
        st.rerun()

    st.divider()
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


    # Progress indicators
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

    # Show any existing human label
    assigned_sentiment = st.session_state.df_traditional.loc[
        st.session_state.df_traditional["Group ID"] == current_group_id, "Assigned Sentiment"
    ].iloc[0]
    if pd.notna(assigned_sentiment):
        st.info(f"Assigned Sentiment: {assigned_sentiment}")
