# 3-Toning_Configuration.py

import re
import html
import unicodedata
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
import mig_functions as mig
from datetime import datetime
from typing import List, Dict, Tuple

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="MIG Sentiment Tool",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
st.session_state.current_page = "Toning Configuration"
mig.standard_sidebar()

st.title("Toning Configuration")

# --- Validate required workflow steps ---
if not st.session_state.get("upload_step", False):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()
if not st.session_state.get("config_step", False):
    st.error("Please run the configuration step before trying this step.")
    st.stop()

# -------------------- Helpers: columns & state --------------------
def _init_df_columns():
    # Ensure human labels exist on the full dataset
    if "Assigned Sentiment" not in st.session_state.df_traditional.columns:
        st.session_state.df_traditional["Assigned Sentiment"] = pd.NA
    # Ensure AI output columns exist on both DataFrames
    for df_name in ["unique_stories", "df_traditional"]:
        df = st.session_state.get(df_name, pd.DataFrame())
        for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col not in df.columns:
                df[col] = None
        st.session_state[df_name] = df

def _clean_list(lst: List[str]) -> List[str]:
    return [s.strip() for s in (lst or []) if isinstance(s, str) and s.strip()]

# -------------------- Tolerant regex builder (store STRING, compile on UI pages) --------------------
def _kw_variant_pattern(kw: str) -> str:
    """
    Build a tolerant regex for a single keyword/phrase:
      - apostrophes: straight ' and curly ’ and prime ′
      - hyphens/dashes: -, ‐, ‒, –, —, −
      - spaces: normal + NBSP
      - optional dots in acronyms
    Everything else is escaped literally.
    """
    APOS  = r"[\'\u2019\u2032]"                 # ' ’ ′
    HYPH  = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"  # - ‐ ‒ – — −
    SPACE = r"[ \t\u00A0]+"                     # space, tab, NBSP

    out = []
    for ch in kw:
        if ch in ("'", "’", "′"):
            out.append(APOS)
        elif ch in ("-", "‐", "‒", "–", "—", "−"):
            out.append(HYPH)
        elif ch.isspace():
            out.append(SPACE)
        elif ch == ".":
            out.append(r"\.?")  # optional dot for acronyms
        else:
            out.append(re.escape(ch))
    return "".join(out)

def build_tolerant_regex_str(keywords: List[str]) -> str | None:
    """Return a single alternation pattern string with case-insensitive flag embedded."""
    kws = [k for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not kws:
        return None
    parts = [_kw_variant_pattern(k) for k in kws]
    # Word-ish boundaries instead of \b to play nice with Unicode punctuation
    return r"(?i)(?<!\w)(?:%s)(?!\w)" % "|".join(parts)

# -------------------- Persist UI defaults across reruns --------------------
st.session_state.setdefault(
    "ui_primary_names",
    [st.session_state.get("client_name", "")] if st.session_state.get("client_name") else [],
)
st.session_state.setdefault("ui_alternate_names", [])
st.session_state.setdefault("ui_spokespeople", [])
st.session_state.setdefault("ui_products", [])
st.session_state.setdefault("ui_toning_rationale", "")

_default_sent_type = st.session_state.get("sentiment_type", "3-way")
st.session_state.setdefault("ui_sentiment_type", _default_sent_type)

st.session_state.setdefault("toning_config_step", False)
st.session_state.setdefault("last_saved", None)

# -------------------- Configuration form --------------------
with st.form("toning_config_form", clear_on_submit=False):
    primary_names = st_tags(
        label="**Primary name(s)**",
        text="Press enter to add more",
        maxtags=1,
        value=st.session_state.ui_primary_names,
        key="primary_names_tags",
    )

    col1, col2 = st.columns(2)
    with col1:
        sentiment_type = st.selectbox(
            "**Sentiment Type**",
            ["3-way", "5-way"],
            index=0 if st.session_state.ui_sentiment_type == "3-way" else 1,
            help="3-way is standard (Positive/Neutral/Negative). 5-way adds more gradations.",
            key="sentiment_type_select",
        )
    with col2:
        model = st.selectbox(
            "Select Model",
            ["gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"],
            help="GPT-5-mini is recommended for most tasks.",
        )

    st.divider()

    alternate_names = st_tags(
        label="**Alternate names**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_alternate_names,
        key="alternate_names_tags",
    )
    spokespeople = st_tags(
        label="**Key spokespeople**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_spokespeople,
        key="spokespeople_tags",
    )
    products = st_tags(
        label="**Products or sub-brands**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_products,
        key="products_tags",
    )

    toning_rationale = st.text_area(
        "**Additional rationale, context, or guidance** (optional):",
        st.session_state.ui_toning_rationale,
        key="toning_rationale_text",
    )

    col_a, col_b = st.columns([1, 1])
    submitted = col_a.form_submit_button("Save Configuration", type="primary")
    reset_clicked = col_b.form_submit_button("Reset Inputs")

# -------------------- Reset handler (keeps form visible) --------------------
if reset_clicked:
    st.session_state.ui_primary_names = [st.session_state.get("client_name", "")] if st.session_state.get("client_name") else []
    st.session_state.ui_alternate_names = []
    st.session_state.ui_spokespeople = []
    st.session_state.ui_products = []
    st.session_state.ui_toning_rationale = ""
    st.session_state.ui_sentiment_type = "3-way"

    # Clear derived/session config
    for k in [
        "sentiment_type", "model_choice",
        "pre_prompt", "post_prompt", "sentiment_instruction", "functions",
        "highlight_keyword", "highlight_regex_str"
    ]:
        st.session_state.pop(k, None)

    st.session_state.toning_config_step = False
    st.session_state.last_saved = None
    st.rerun()

# -------------------- Save handler --------------------
if submitted:
    if not primary_names or not str(primary_names[0]).strip():
        st.warning("Add at least one **Primary name** (e.g., the brand/entity being toned) before saving.")
    else:
        # Persist UI selections
        st.session_state.ui_primary_names = _clean_list(primary_names)
        st.session_state.ui_alternate_names = _clean_list(alternate_names)
        st.session_state.ui_spokespeople = _clean_list(spokespeople)
        st.session_state.ui_products = _clean_list(products)
        st.session_state.ui_toning_rationale = toning_rationale or ""
        st.session_state.ui_sentiment_type = sentiment_type
        st.session_state.sentiment_type = sentiment_type
        st.session_state.model_choice = model

        named_entity = st.session_state.ui_primary_names[0]
        aliases = st.session_state.ui_alternate_names
        spokes  = st.session_state.ui_spokespeople
        prods   = st.session_state.ui_products
        rationale_str = st.session_state.ui_toning_rationale.strip() if st.session_state.ui_toning_rationale else None

        _init_df_columns()

        # --- Build highlight keywords (display) + tolerant regex (string) ---
        display_keywords = list(st.session_state.ui_primary_names) + aliases + spokes + prods
        # Dedupe case-insensitively while preserving original display form
        seen_cf, deduped_display = set(), []
        for k in display_keywords:
            cf = k.casefold()
            if cf not in seen_cf:
                seen_cf.add(cf)
                deduped_display.append(k.strip())
        st.session_state.highlight_keyword = deduped_display
        st.session_state.highlight_regex_str = build_tolerant_regex_str(deduped_display)

        # --- Build prompts ---
        pre_lines = [
            f"PRIMARY ENTITY (focus): {named_entity}",
            "Your task: Analyze sentiment toward the PRIMARY ENTITY only.",
            "Consider references where sentiment should carry over to the primary entity when acting on its behalf.",
        ]
        if aliases:
            pre_lines += ["", "ALIASES / ALTERNATE NAMES (treat as the same entity when present):", ", ".join(aliases)]
        if spokes:
            pre_lines += ["", "KEY SPOKESPEOPLE (attribute their on-record statements/actions to the entity unless clearly personal/unrelated):", ", ".join(spokes)]
        if prods:
            pre_lines += ["", "PRODUCTS / SUB-BRANDS (attribute product sentiment to the parent unless clearly isolated/unrelated):", ", ".join(prods)]
        pre_lines += ["", "Focus ONLY on how the coverage portrays the primary entity (including legitimate carry-over per rules above)."]
        st.session_state.pre_prompt = "\n".join(pre_lines).strip()

        context_lines = [
            "Scope Clarifications:",
            f"- Research by {named_entity} on a negative topic is not automatically negative toward the entity.",
            "- Hosting/sponsoring an event about a negative issue is not automatically negative.",
            "- Straight factual coverage is Neutral.",
            "- Passing mentions without a strong stance are generally Neutral.",
            "",
            "Attribution Rules:",
            f"1) When a spokesperson acts explicitly for {named_entity}, attribute that sentiment to the entity.",
            f"2) When a product or sub-brand is discussed, attribute sentiment to {named_entity} unless clearly unrelated.",
            "3) Do not transfer sentiment from unrelated external topics unless the article directly links them.",
            f"4) If {named_entity} (or valid aliases) is not mentioned, respond with NOT RELEVANT.",
        ]
        if rationale_str:
            context_lines += ["", "Analyst Rationale/Context (apply when judging gray areas):", f"- {rationale_str}"]
        st.session_state.post_prompt = "\n".join(context_lines).strip()

        # --- Function schema / labeling instruction ---
        if sentiment_type == "3-way":
            st.session_state.sentiment_instruction = f"""
LABEL SET: POSITIVE, NEUTRAL, NEGATIVE, NOT RELEVANT

CRITERIA:
- POSITIVE: Praises, favorable framing, or beneficial outcomes attributed to {named_entity}.
- NEUTRAL: Factual/balanced coverage without clear positive/negative framing of {named_entity}.
- NEGATIVE: Criticism, unfavorable framing, or negative outcomes attributed to {named_entity}.
- NOT RELEVANT: {named_entity} (or aliases) not mentioned.

OUTPUT: Provide the uppercase label, a confidence (0–100), and a 1–2 sentence explanation focused on {named_entity}.
""".strip()

            st.session_state.functions = [{
                "name": "analyze_sentiment",
                "description": "Analyze sentiment toward the named entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "named_entity": {"type": "string", "description": "The brand/entity being analyzed."},
                        "sentiment": {
                            "type": "string",
                            "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
                            "description": "Sentiment toward the entity."
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 100, "description": "Confidence (percentage)."},
                        "explanation": {"type": "string", "description": "1–2 sentence rationale tied to how the story portrays the entity."}
                    },
                    "required": ["named_entity", "sentiment", "confidence", "explanation"]
                }
            }]
        else:
            st.session_state.sentiment_instruction = f"""
LABEL SET: VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, VERY NEGATIVE, NOT RELEVANT

CRITERIA:
- VERY POSITIVE: Strong praise or substantial positive impact credited to {named_entity}.
- SOMEWHAT POSITIVE: Moderately favorable framing or minor positive outcomes.
- NEUTRAL: Factual/balanced with no clear stance on {named_entity}.
- SOMEWHAT NEGATIVE: Mild criticism or limited negative impact.
- VERY NEGATIVE: Strong criticism or substantial negative impact attributed to {named_entity}.
- NOT RELEVANT: {named_entity} (or aliases) not mentioned.

OUTPUT: Provide the uppercase label, a confidence (0–100), and a 1–2 sentence explanation focused on {named_entity}.
""".strip()

            st.session_state.functions = [{
                "name": "analyze_sentiment",
                "description": "Analyze sentiment toward the named entity with intensity levels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "named_entity": {"type": "string", "description": "The brand/entity being analyzed."},
                        "sentiment": {
                            "type": "string",
                            "enum": [
                                "VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL",
                                "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"
                            ],
                            "description": "Sentiment toward the entity with intensity."
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 100, "description": "Confidence (percentage)."},
                        "explanation": {"type": "string", "description": "1–2 sentence rationale tied to how the story portrays the entity."}
                    },
                    "required": ["named_entity", "sentiment", "confidence", "explanation"]
                }
            }]

        st.session_state.toning_config_step = True
        st.session_state.last_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"✅ Configuration saved for **{named_entity}**.")
        st.toast("Toning configuration saved.", icon="✅")

# -------------------- Prompt & schema preview --------------------
with st.expander("Generated Prompts, Keywords & Function"):
    if st.session_state.get("last_saved"):
        st.caption(f"Last saved: {st.session_state.last_saved}")

    if "pre_prompt" in st.session_state:
        st.write("**Pre-prompt:**")
        st.code(st.session_state.pre_prompt)
    if "post_prompt" in st.session_state:
        st.write("**Labeling Clarifications:**")
        st.code(st.session_state.post_prompt)
    if "sentiment_instruction" in st.session_state:
        st.write("**Labeling Instruction:**")
        st.code(st.session_state.sentiment_instruction)

    st.write("**Highlight keywords (display):**")
    st.write(st.session_state.get("highlight_keyword", []))

    st.write("**Tolerant highlight regex (string stored for UI pages):**")
    st.code(st.session_state.get("highlight_regex_str", "") or "")
