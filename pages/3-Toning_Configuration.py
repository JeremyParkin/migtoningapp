# 3-Toning_Configuration.py

import re
import html
import unicodedata
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
import mig_functions as mig
from datetime import datetime
from typing import List

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="MIG Sentiment Tool",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
st.session_state.current_page = "Toning Configuration"
# mig.standard_sidebar()

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
      - Apostrophes: match straight ' and curly ’ / prime ′ … and make them OPTIONAL.
      - If the keyword itself omits an apostrophe before a terminal 's' in a word
        (e.g., 'Alzheimers'), allow an optional apostrophe in that spot so it still
        matches 'Alzheimer's'.
      - Hyphens/dashes: -, ‐, ‒, –, —, −
      - Spaces: normal + NBSP
      - Optional dots in acronyms
    Everything else is escaped literally.
    """
    APOS  = r"[\'\u2019\u2032]"                           # ' ’ ′
    HYPH  = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"   # - ‐ ‒ – — −
    SPACE = r"[ \t\u00A0]+"                                # space, tab, NBSP

    hyphen_chars = "-\u2010\u2011\u2012\u2013\u2014\u2212"
    apos_chars   = "'\u2019\u2032"

    out = []
    L = len(kw)
    i = 0
    while i < L:
        ch       = kw[i]
        prev_ch  = kw[i-1] if i > 0 else ""
        next_ch  = kw[i+1] if i+1 < L else ""

        # Treat any apostrophe-like in the keyword as OPTIONAL in the pattern
        if ch in apos_chars:
            out.append(f"(?:{APOS})?")

        # Hyphens/dashes
        elif ch in hyphen_chars:
            out.append(HYPH)

        # Spaces
        elif ch.isspace():
            out.append(SPACE)

        # Optional dots for acronyms
        elif ch == ".":
            out.append(r"\.?")

        # If this is an 's' that ENDS a token (end of string or before space/hyphen),
        # and it wasn't already preceded by an apostrophe in the *keyword*,
        # allow an OPTIONAL apostrophe before it. This makes "Alzheimers" match "Alzheimer's".
        elif (ch.lower() == "s"
              and (i+1 == L or next_ch.isspace() or next_ch in hyphen_chars)
              and prev_ch not in apos_chars
              and prev_ch.isalnum()):
            out.append(f"(?:{APOS})?s")

        else:
            out.append(re.escape(ch))

        i += 1

    return "".join(out)



# def _kw_variant_pattern(kw: str) -> str:
#     """
#     Build a tolerant regex for a single keyword/phrase:
#       - apostrophes: straight ' and curly ’ and prime ′
#       - hyphens/dashes: -, ‐, ‒, –, —, −
#       - spaces: normal + NBSP
#       - optional dots in acronyms
#     Everything else is escaped literally.
#     """
#     APOS  = r"[\'\u2019\u2032]"                 # ' ’ ′
#     HYPH  = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"  # - ‐ ‒ – — −
#     SPACE = r"[ \t\u00A0]+"                     # space, tab, NBSP
#
#     out = []
#     for ch in kw:
#         if ch in ("'", "’", "′"):
#             out.append(APOS)
#         elif ch in ("-", "‐", "‒", "–", "—", "−"):
#             out.append(HYPH)
#         elif ch.isspace():
#             out.append(SPACE)
#         elif ch == ".":
#             out.append(r"\.?")  # optional dot for acronyms
#         else:
#             out.append(re.escape(ch))
#     return "".join(out)

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

# --- Page-specific sidebar prompt ---
client = st.session_state.get("client_name", "<client name>")
st.sidebar.markdown(
    f"""## ChatGPT Prompt:
For **{client}**, I would like to know: 
- alternate names or aliases 
- key spokespeople or public representatives 
- main programs, products, initiatives or sub-brands
"""
)

# -------------------- Admin flag (set on Getting Started) --------------------
is_admin = bool(st.session_state.get("is_admin", False))

# -------------------- Configuration form --------------------
with st.form("toning_config_form", clear_on_submit=False):
    primary_names = st_tags(
        label="**Primary name(s)**",
        text="Press enter to add more",
        maxtags=1,
        value=st.session_state.ui_primary_names,
        key="primary_names_tags",
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        sentiment_type = st.selectbox(
            "**Sentiment Type**",
            ["3-way", "5-way"],
            index=0 if st.session_state.ui_sentiment_type == "3-way" else 1,
            help="3-way is standard (Positive/Neutral/Negative). 5-way adds more gradations.",
            key="sentiment_type_select",
        )
    with col2:
        # Build model list; expose GPT-5 only to admins
        base_models = ["gpt-5-mini", "gpt-4.1-mini"]
        if is_admin:
            base_models.append("gpt-5")

        # Determine current selection (coerce hidden value if necessary)
        current_choice = st.session_state.get("model_choice", "gpt-5-mini")
        if current_choice not in base_models:
            # If a non-admin somehow had gpt-5 in state, show mini instead
            current_choice = "gpt-5-mini"

        model = st.selectbox(
            "Select Model",
            base_models,
            index=base_models.index(current_choice) if current_choice in base_models else 0,
            help=("gpt-5 is available (admin). " if is_admin else "") + "GPT-5-mini is recommended for most tasks.",
            key="model_choice_select",
        )
    with col3:
        st.markdown("<small>Model notes:</small>", unsafe_allow_html=True)
        st.caption("* gpt-5-mini: latest model but slower.\n* gpt-4.1-mini is good, and much faster.")


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

    # Clear derived/session config (do not touch is_admin)
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

        # Coerce model if necessary (admin-only protection)
        chosen_model = model
        if (not is_admin) and chosen_model == "gpt-5":
            chosen_model = "gpt-5-mini"
        st.session_state.model_choice = chosen_model

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
        # pre_lines = [
        #     f"PRIMARY ENTITY (focus): {named_entity}",
        #     "Your task: Analyze sentiment toward the PRIMARY ENTITY only.",
        #     "Consider references where sentiment should carry over to the primary entity when acting on its behalf.",
        # ]
        # if aliases:
        #     pre_lines += ["", "ALIASES / ALTERNATE NAMES (treat as the same entity when present):", ", ".join(aliases)]
        # if spokes:
        #     pre_lines += ["", "KEY SPOKESPEOPLE (attribute their on-record statements/actions to the entity unless clearly personal/unrelated):", ", ".join(spokes)]
        # if prods:
        #     pre_lines += ["", "PRODUCTS / SUB-BRANDS (attribute product sentiment to the parent unless clearly isolated/unrelated):", ", ".join(prods)]
        # pre_lines += ["", "Focus ONLY on how the coverage portrays the primary entity (including legitimate carry-over per rules above)."]
        # st.session_state.pre_prompt = "\n".join(pre_lines).strip()

        # --- Build prompts ---
        # PRE-PROMPT — Collective entity framing (brand umbrella)
        pre_lines = [
            f"PRIMARY ENTITY (brand umbrella): {named_entity}",
            "Assess sentiment toward the *collective entity* defined as:",
            "  • the primary entity,",
            "  • any listed aliases/alternate names,",
            "  • any named spokespeople when acting on behalf of the entity, and",
            "  • any listed products or sub-brands when discussed as part of the entity’s activity/portfolio.",
        ]

        if aliases:
            pre_lines += ["", "ALIASES / ALTERNATE NAMES (treat as the same entity):", ", ".join(aliases)]
        if spokes:
            pre_lines += ["",
                          "SPOKESPEOPLE (attribute on-record statements/actions to the entity unless clearly personal/unrelated):",
                          ", ".join(spokes)]
        if prods:
            pre_lines += ["",
                          "PRODUCTS / SUB-BRANDS (attribute product sentiment to the parent unless clearly isolated/unrelated):",
                          ", ".join(prods)]

        pre_lines += [
            "",
            "Judge the *net sentiment* the coverage conveys to a typical reader/viewer about the collective entity.",
            "Ignore sentiment about unrelated third parties unless the story explicitly connects it to the entity.",
        ]
        st.session_state.pre_prompt = "\n".join(pre_lines).strip()


        # context_lines = [
        #     "Scope Clarifications:",
        #     f"- Research by {named_entity} on a negative topic is not automatically negative toward the entity.",
        #     "- Hosting/sponsoring an event about a negative issue is not automatically negative.",
        #     "- Straight factual coverage is Neutral.",
        #     "- Passing mentions without a strong stance are generally Neutral.",
        #     "",
        #     "Attribution Rules:",
        #     f"1) When a spokesperson acts explicitly for {named_entity}, attribute that sentiment to the entity.",
        #     f"2) When a product or sub-brand is discussed, attribute sentiment to {named_entity} unless clearly unrelated.",
        #     "3) Do not transfer sentiment from unrelated external topics unless the article directly links them.",
        #     f"4) If {named_entity} (or valid aliases) is not mentioned, respond with NOT RELEVANT.",
        # ]
        # if rationale_str:
        #     context_lines += ["", "Analyst Rationale/Context (apply when judging gray areas):", f"- {rationale_str}"]
        # st.session_state.post_prompt = "\n".join(context_lines).strip()

        # POST-PROMPT — Clarifications + Analyst guidance priority
        context_lines = [
            "Scope Clarifications:",
            f"- Research by {named_entity} on a negative topic is not automatically negative toward the entity.",
            "- Hosting/sponsoring an event about a negative issue is not automatically negative.",
            "- Straight factual coverage is Neutral.",
            "- Passing mentions without a strong stance are generally Neutral.",
            "- Brief/Passing Mentions: If the collective entity appears only briefly in a longer story without explicit praise/criticism or clear attribution of outcomes to the entity, default to NEUTRAL.",
            "- Passing mentions: If the entity is referenced only briefly, default to NEUTRAL unless the coverage explicitly evaluates the entity or attributes outcomes to it.",

            "",
            "Carry-over / Attribution Rules (collective entity):",
            f"1) When a spokesperson acts explicitly for {named_entity}, attribute their stance to the entity.",
            f"2) When a product or sub-brand is discussed, attribute sentiment to {named_entity} unless clearly unrelated.",
            "3) No sentiment transfer: Do not infer sentiment toward the entity from third parties or adjacent topics. Only attribute sentiment when the coverage explicitly links the stance or outcome to the entity (or its spokespeople/products acting for it).",
            f"4) If the collective entity (primary/aliases/spokespeople-as-entity/products) is not present, label NOT RELEVANT.",
            "",
            "Tie-breakers & Gray Areas:",
            "- Use the audience takeaway as the deciding factor when positive and negative elements coexist.",
            "- Prefer explicit attributions, direct quotes, headlines, and framing to infer stance.",
        ]

        if rationale_str:
            context_lines += [
                "",
                "Analyst Guidance — PRIORITY (overrides defaults in gray areas):",
                f"- {rationale_str}",
                "If this guidance changes the default outcome, follow it and reflect that in your explanation.",
            ]

        st.session_state.post_prompt = "\n".join(context_lines).strip()



        # --- Function schema / labeling instruction ---
        if sentiment_type == "3-way":
            st.session_state.sentiment_instruction = f"""
            LABEL SET: POSITIVE, NEUTRAL, NEGATIVE, NOT RELEVANT

            WHAT TO JUDGE:
            - The *collective entity*: {named_entity} + aliases + spokespeople acting on its behalf + products/sub-brands (per rules).

            CRITERIA:
            - POSITIVE: Praise, favorable framing, or beneficial outcomes credited to the collective entity.
            - NEUTRAL: Factual/balanced coverage with no clear stance on the collective entity; **brief/passing mentions in otherwise unrelated coverage should default to NEUTRAL unless explicit positive/negative framing or clear attribution is present.** Includes brief/incidental mentions without evaluation; treat as NEUTRAL unless there is clear positive/negative framing about the entity.
            - NEGATIVE: Criticism, unfavorable framing, or negative outcomes attributed to the collective entity.
            - NOT RELEVANT: The collective entity (as defined) is not present.

            OUTPUT:
            - Provide the UPPERCASE label, a confidence (0–100), and a 1–2 sentence explanation focused on the collective entity.
            - If Analyst Guidance was provided and altered the default outcome, mention that in the explanation.
            """.strip()

            st.session_state.functions = [{
                "name": "analyze_sentiment",
                "description": "Analyze sentiment toward the collective entity (brand + aliases + spokespeople-as-entity + products/sub-brands).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "named_entity": {"type": "string",
                                         "description": "The brand/entity being analyzed (primary name)."},
                        "sentiment": {
                            "type": "string",
                            "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
                            "description": "Sentiment toward the collective entity."
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 100,
                                       "description": "Confidence (percentage)."},
                        "explanation": {"type": "string",
                                        "description": "1–2 sentence rationale tied to how the story portrays the collective entity."}
                    },
                    "required": ["named_entity", "sentiment", "confidence", "explanation"]
                }
            }]

        #             st.session_state.sentiment_instruction = f"""
# LABEL SET: POSITIVE, NEUTRAL, NEGATIVE, NOT RELEVANT
#
# CRITERIA:
# - POSITIVE: Praises, favorable framing, or beneficial outcomes attributed to {named_entity}.
# - NEUTRAL: Factual/balanced coverage without clear positive/negative framing of {named_entity}.
# - NEGATIVE: Criticism, unfavorable framing, or negative outcomes attributed to {named_entity}.
# - NOT RELEVANT: {named_entity} (or aliases) not mentioned.
#
# OUTPUT: Provide the uppercase label, a confidence (0–100), and a 1–2 sentence explanation focused on {named_entity}.
# """.strip()
#
#             st.session_state.functions = [{
#                 "name": "analyze_sentiment",
#                 "description": "Analyze sentiment toward the named entity.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "named_entity": {"type": "string", "description": "The brand/entity being analyzed."},
#                         "sentiment": {
#                             "type": "string",
#                             "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
#                             "description": "Sentiment toward the entity."
#                         },
#                         "confidence": {"type": "number", "minimum": 0, "maximum": 100, "description": "Confidence (percentage)."},
#                         "explanation": {"type": "string", "description": "1–2 sentence rationale tied to how the story portrays the entity."}
#                     },
#                     "required": ["named_entity", "sentiment", "confidence", "explanation"]
#                 }
#             }]
        else:
            st.session_state.sentiment_instruction = f"""
            LABEL SET: VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, VERY NEGATIVE, NOT RELEVANT

            WHAT TO JUDGE:
            - The *collective entity*: {named_entity} + aliases + spokespeople acting on its behalf + products/sub-brands (per rules).

            CRITERIA:
            - VERY POSITIVE: Strong praise or substantial positive impact credited to the collective entity.
            - SOMEWHAT POSITIVE: Moderate praise or minor positive outcomes.
            - NEUTRAL: Factual/balanced coverage with no clear stance on the collective entity; **brief/passing mentions in otherwise unrelated coverage should default to NEUTRAL unless explicit positive/negative framing or clear attribution is present.** Includes brief/incidental mentions without evaluation; treat as NEUTRAL unless there is clear positive/negative framing about the entity.
            - SOMEWHAT NEGATIVE: Mild criticism or limited negative impact.
            - VERY NEGATIVE: Strong criticism or substantial negative impact attributed to the collective entity.
            - NOT RELEVANT: The collective entity (as defined) is not present.

            OUTPUT:
            - Provide the UPPERCASE label, a confidence (0–100), and a 1–2 sentence explanation focused on the collective entity.
            - If Analyst Guidance was provided and altered the default outcome, mention that in the explanation.
            """.strip()

            st.session_state.functions = [{
                "name": "analyze_sentiment",
                "description": "Analyze sentiment toward the collective entity (brand + aliases + spokespeople-as-entity + products/sub-brands) with intensity levels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "named_entity": {"type": "string",
                                         "description": "The brand/entity being analyzed (primary name)."},
                        "sentiment": {
                            "type": "string",
                            "enum": [
                                "VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL",
                                "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"
                            ],
                            "description": "Sentiment toward the collective entity with intensity."
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 100,
                                       "description": "Confidence (percentage)."},
                        "explanation": {"type": "string",
                                        "description": "1–2 sentence rationale tied to how the story portrays the collective entity."}
                    },
                    "required": ["named_entity", "sentiment", "confidence", "explanation"]
                }
            }]

        #             st.session_state.sentiment_instruction = f"""
# LABEL SET: VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, VERY NEGATIVE, NOT RELEVANT
#
# CRITERIA:
# - VERY POSITIVE: Strong praise or substantial positive impact credited to {named_entity}.
# - SOMEWHAT POSITIVE: Moderately favorable framing or minor positive outcomes.
# - NEUTRAL: Factual/balanced with no clear stance on {named_entity}.
# - SOMEWHAT NEGATIVE: Mild criticism or limited negative impact.
# - VERY NEGATIVE: Strong criticism or substantial negative impact attributed to {named_entity}.
# - NOT RELEVANT: {named_entity} (or aliases) not mentioned.
#
# OUTPUT: Provide the uppercase label, a confidence (0–100), and a 1–2 sentence explanation focused on {named_entity}.
# """.strip()
#
#             st.session_state.functions = [{
#                 "name": "analyze_sentiment",
#                 "description": "Analyze sentiment toward the named entity with intensity levels.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "named_entity": {"type": "string", "description": "The brand/entity being analyzed."},
#                         "sentiment": {
#                             "type": "string",
#                             "enum": [
#                                 "VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL",
#                                 "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"
#                             ],
#                             "description": "Sentiment toward the entity with intensity."
#                         },
#                         "confidence": {"type": "number", "minimum": 0, "maximum": 100, "description": "Confidence (percentage)."},
#                         "explanation": {"type": "string", "description": "1–2 sentence rationale tied to how the story portrays the entity."}
#                     },
#                     "required": ["named_entity", "sentiment", "confidence", "explanation"]
#                 }
#             }]

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
