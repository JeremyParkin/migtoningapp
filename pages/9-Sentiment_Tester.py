# 9-Sentiment_Tester.py

import pandas as pd
import streamlit as st
import mig_functions as mig

# ============== Page Setup ==============
st.set_page_config(
    page_title="MIG Sentiment Tester",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
mig.standard_sidebar()
st.title("AI Sentiment Analysis Tester")

# ============== Guards ==============
if not st.session_state.get("upload_step"):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()

if not st.session_state.get("config_step"):
    st.error("Please run the configuration step before trying this step.")
    st.stop()

# ============== Helpers ==============
def _first_non_na(series: pd.Series):
    for x in series:
        if pd.notna(x) and str(x).strip():
            return x
    return pd.NA

def build_group_hybrid():
    """
    Build a per-Group-ID hybrid label:
      Hybrid = Assigned Sentiment (if any in the group) else AI Sentiment (if any).
    Also returns group-level Assigned/AI for reference, plus Group_Count as fallback.
    """
    dft = st.session_state.df_traditional.copy()

    # Ensure columns exist
    for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Rationale"]:
        if col not in dft.columns:
            dft[col] = pd.NA

    grp = dft.groupby("Group ID", as_index=False).agg(
        Assigned_Sentiment=("Assigned Sentiment", _first_non_na),
        AI_Sentiment=("AI Sentiment", _first_non_na),
        AI_Sentiment_Rationale=("AI Sentiment Rationale", _first_non_na),
        Group_Count=("Group ID", "size"),
    )

    grp["Hybrid Sentiment"] = grp["Assigned_Sentiment"].where(
        pd.notna(grp["Assigned_Sentiment"]) & grp["Assigned_Sentiment"].astype(str).str.strip().ne(""),
        grp["AI_Sentiment"]
    )
    return grp

def normalize_label(s):
    return (str(s).upper().strip()) if pd.notna(s) else ""

def label_set_from_session():
    stype = str(st.session_state.get("sentiment_type", "3-way")).lower().strip()
    if stype.startswith("5"):
        return ["VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL", "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"]
    return ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]

def distribution(series: pd.Series, labels):
    vc = series.value_counts(normalize=True)
    out = {lab: float(vc.get(lab, 0.0)) for lab in labels}
    other = max(0.0, 1.0 - sum(out.values()))
    out["OTHER"] = other
    return out

# ============== Build Hybrid + Comparison ==============
# Build or rebuild Hybrid each run
group_hybrid = build_group_hybrid()

# Merge onto unique_stories to compare per group
u = st.session_state.unique_stories.merge(group_hybrid, on="Group ID", how="left")


# --- Ensure 'Hybrid Sentiment' exists after merge (handle suffixing/overlap) ---
if "Hybrid Sentiment" not in u.columns:
    # If pandas suffixed during merge, grab the right-hand version
    candidates = [c for c in u.columns if c.endswith("Hybrid Sentiment")]
    if candidates:
        u["Hybrid Sentiment"] = u[candidates[0]]
    else:
        # Derive it on the fly from Assigned then AI (as a fallback)
        u["Hybrid Sentiment"] = u.get("Assigned_Sentiment")
        mask = u["Hybrid Sentiment"].isna() | (u["Hybrid Sentiment"].astype(str).str.strip() == "")
        u.loc[mask, "Hybrid Sentiment"] = u.get("AI_Sentiment")


# Compare only where we have both a human Sentiment and a Hybrid
comp = u.copy()
comp = comp[comp["Hybrid Sentiment"].notna() & comp["Sentiment"].notna()].copy()

# Normalize labels
comp["Human_Upper"]  = comp["Sentiment"].apply(normalize_label)
comp["Hybrid_Upper"] = comp["Hybrid Sentiment"].apply(normalize_label)
comp["Match"] = comp["Human_Upper"] == comp["Hybrid_Upper"]

# Weighted match rate (by Group Count if present)
gc_col = "Group Count" if "Group Count" in comp.columns else ("Group_Count" if "Group_Count" in comp.columns else None)
if gc_col:
    comp[gc_col] = pd.to_numeric(comp[gc_col], errors="coerce").fillna(1)
    total_weight = comp[gc_col].sum()
    weighted_matches = comp.loc[comp["Match"], gc_col].sum()
    weighted_match_rate = (weighted_matches / total_weight) * 100 if total_weight else 0.0
else:
    weighted_match_rate = None

# Unweighted match rate
match_rate = comp["Match"].mean() * 100 if len(comp) else 0.0

# Distributions (use the app’s active label set)
labels = label_set_from_session()
human_dist  = distribution(comp["Human_Upper"], labels)
hybrid_dist = distribution(comp["Hybrid_Upper"], labels)

# ============== Summary ==============
st.markdown("### 🔍 Human vs Hybrid Sentiment — Summary")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Unique Stories Compared", f"{len(comp)}")
with c2:
    st.metric("Match Rate", f"{match_rate:.1f}%")
with c3:
    st.metric("Weighted Match Rate", f"{weighted_match_rate:.1f}%" if weighted_match_rate is not None else "N/A")

# Label distribution table
dist_rows = []
for lab in labels + ["OTHER"]:
    dist_rows.append({
        "Label": lab.title(),
        "Human %": f"{human_dist[lab]*100:.1f}%",
        "Hybrid %": f"{hybrid_dist[lab]*100:.1f}%"
    })
# st.dataframe(pd.DataFrame(dist_rows), hide_index=True, use_container_width=True)


# === Build Full Comparison Summary Table ===
# Inputs expected: named_entity, model, toning_rationale, st.session_state.uploaded_file_name
# Dataframe expected: u with columns ['Hybrid Sentiment','Sentiment','Group Count']

# ---- Page vars used in the summary table ----
named_entity = (
    st.session_state.get("client_name")
    or (st.session_state.get("ui_primary_names") or [None])[0]
    or "Unknown Entity"
)

# whatever model you actually used for AI on this page/run
model = st.session_state.get("model_choice", "gpt-5-mini")

# guidance/rationale captured on p3 (or empty if none)
toning_rationale = st.session_state.get("ui_toning_rationale", "") or ""


def _base_label(s: str) -> str:
    """Map any 3-way/5-way label to POSITIVE/NEUTRAL/NEGATIVE, else OTHER."""
    if not isinstance(s, str):
        return "OTHER"
    t = s.strip().upper()
    if t in {"POSITIVE", "VERY POSITIVE", "SOMEWHAT POSITIVE"}:
        return "POSITIVE"
    if t in {"NEUTRAL"}:
        return "NEUTRAL"
    if t in {"NEGATIVE", "VERY NEGATIVE", "SOMEWHAT NEGATIVE"}:
        return "NEGATIVE"
    # treat NOT RELEVANT or anything else as OTHER
    return "OTHER"

# Keep rows with both human and model labels
comp = u.copy()
if "Sentiment" not in comp.columns:
    comp["Sentiment"] = pd.NA
comp = comp[comp["Hybrid Sentiment"].notna() & comp["Sentiment"].notna()].copy()

# Normalize for comparison
comp["Human_UP"] = comp["Sentiment"].astype(str).str.upper().str.strip()
comp["AI_UP"]    = comp["Hybrid Sentiment"].astype(str).str.upper().str.strip()
comp["Match"]    = comp["Human_UP"] == comp["AI_UP"]

# Weighted match rate (by Group Count if available)
if "Group Count" in comp.columns:
    comp["Group Count"] = pd.to_numeric(comp["Group Count"], errors="coerce").fillna(1)
    total_weight = comp["Group Count"].sum() if len(comp) else 1
    weighted_matches = comp.loc[comp["Match"], "Group Count"].sum() if len(comp) else 0
    weighted_match_rate = (weighted_matches / total_weight) * 100 if total_weight else 0.0
else:
    weighted_match_rate = None

# Simple match rate (unweighted)
match_rate = (comp["Match"].mean() * 100) if len(comp) else 0.0

# Distributions in POS/NEU/NEG/OTHER buckets
labels = ["POSITIVE", "NEUTRAL", "NEGATIVE", "OTHER"]
human_dist = comp["Human_UP"].map(_base_label).value_counts(normalize=True).reindex(labels, fill_value=0.0)
ai_dist    = comp["AI_UP"].map(_base_label).value_counts(normalize=True).reindex(labels, fill_value=0.0)

# Hypothetical Hybrid (first 10 human-toned) Weighted Match Rate
sim = comp.copy()
if "Group Count" in sim.columns:
    sim = sim.sort_values(by="Group Count", ascending=False).reset_index(drop=True)
    sim["Match_Sim"] = sim["Match"]
    sim.loc[:9, "Match_Sim"] = True  # force top 10 to match
    sim_total_w = sim["Group Count"].sum() if len(sim) else 1
    sim_weighted = (sim["Group Count"] * sim["Match_Sim"]).sum() if len(sim) else 0
    sim_weighted_match_rate = (sim_weighted / sim_total_w) * 100 if sim_total_w else 0.0
else:
    sim_weighted_match_rate = None

# Assemble the one-row summary table
summary_row = {
    "Named Entity": named_entity,
    "File Name": st.session_state.get("uploaded_file_name", "Not Available"),
    "Model Used": model,  # or st.session_state.get("model_choice", "gpt-5-mini")
    "Additional Guidance Provided": toning_rationale.strip() if toning_rationale and toning_rationale.strip() else "No",
    "Match Rate (%)": f"{match_rate:.1f}%",
    "Weighted Match Rate (%)": f"{weighted_match_rate:.1f}%" if weighted_match_rate is not None else "N/A",
    "Unique Stories Compared": int(len(comp)),

    "Human Positive": f"{human_dist['POSITIVE'] * 100:.1f}%",
    "AI Positive":    f"{ai_dist['POSITIVE'] * 100:.1f}%",
    "Human Neutral":  f"{human_dist['NEUTRAL'] * 100:.1f}%",
    "AI Neutral":     f"{ai_dist['NEUTRAL'] * 100:.1f}%",
    "Human Negative": f"{human_dist['NEGATIVE'] * 100:.1f}%",
    "AI Negative":    f"{ai_dist['NEGATIVE'] * 100:.1f}%",
    "Human Other":    f"{human_dist['OTHER'] * 100:.1f}%",
    "AI Other":       f"{ai_dist['OTHER'] * 100:.1f}%",

    # "Hypothetical Hybrid (first 10 Human toned) Weighted Match Rate":
    #     f"{sim_weighted_match_rate:.1f}%" if sim_weighted_match_rate is not None else "N/A",
}

summary_df = pd.DataFrame([summary_row])
# st.markdown("### 🔍 AI vs Human (Hybrid) Sentiment — Summary")
st.table(summary_df)


# ============== Detailed View ==============
st.subheader("Detailed Comparison Table")

# Columns to display
cols_to_show = [
    "Group ID", "Headline",
    "Sentiment", "Hybrid Sentiment",
    "Assigned_Sentiment", "AI_Sentiment",
    "AI_Sentiment_Rationale",
    "Match", "URL", "Snippet"
]
# Include group count column if available
if gc_col and gc_col not in cols_to_show:
    cols_to_show.insert(3, gc_col)  # put near the left side

# Filter mismatches toggle
only_mismatch = st.checkbox("Show mismatches only")
df_view = comp.copy()
if only_mismatch:
    df_view = df_view[~df_view["Match"]]

# Keep only existing columns
cols_to_show = [c for c in cols_to_show if c in df_view.columns]
st.dataframe(df_view[cols_to_show].reset_index(drop=True), hide_index=True, use_container_width=True)

# ============== Footer Actions ==============
colA, colB = st.columns([1, 3])
with colA:
    if st.button("Recompute Hybrid Now"):
        st.rerun()

with colB:
    st.caption(
        "Hybrid = human **Assigned Sentiment** if present within the group, otherwise the group’s **AI Sentiment**. "
        "Weights (if shown) use Group Count."
    )
