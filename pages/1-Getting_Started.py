# 1-Getting_Started.py

import csv, io, hashlib
import pandas as pd
import streamlit as st
import mig_functions as mig

# ── Title / page marker ───────────────────────────────────────────────────────
st.title("Getting Started")
st.session_state.setdefault("current_page", "Getting Started")

# ── Initialize session defaults (idempotent) ──────────────────────────────────
string_vars = {
    "page": "1: Getting Started",
    "sentiment_type": "3-way",
    "client_name": "",
    "focus": "",
    "model_choice": "gpt-5-mini",
    "counter": 0,
    "analysis_note": "",
    "group_ids": "",
    "sample_size": 0,
    "highlight_keyword": "",
    "min_impressions": 0,
    "min_domain_authority": 0,
    "pre_prompt": "",
    "post_prompt": "",
    "functions": "",
    "sentiment_examples": "",
    "uploaded_file_name": "",
}
for k, v in string_vars.items():
    st.session_state.setdefault(k, v)

for name in ["df_traditional", "unique_stories", "full_dataset"]:
    st.session_state.setdefault(name, pd.DataFrame())

for b in ["upload_step", "config_step", "sentiment_opinion", "random_sample",
          "toning_config_step", "processing_started"]:
    st.session_state.setdefault(b, False)

# Additional state for file handling
st.session_state.setdefault("uploaded_file", None)
st.session_state.setdefault("excel_sheet_names", [])
st.session_state.setdefault("excel_selected_sheet", None)

# ── Admin unlock helper ───────────────────────────────────────────────────────
ADMIN_SHA256 = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")
st.session_state.setdefault("is_admin", False)

def _sync_admin_from_reporting():
    rp = (
        st.session_state.get("reporting_period_or_focus")
        or st.session_state.get("period")
        or st.session_state.get("focus")
        or ""
    ).strip()
    is_admin = hashlib.sha256(rp.encode()).hexdigest() == ADMIN_SHA256
    prev = st.session_state.get("is_admin")
    st.session_state["is_admin"] = bool(is_admin)
    if is_admin and prev is not True:
        st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
    elif (prev is True) and not is_admin:
        st.toast("🔒 Admin tools locked.", icon="🔒")

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_csv_smart(uploaded_file) -> pd.DataFrame:
    """Robust CSV reader w/ BOM strip + delimiter & header detection + header promotion."""
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    sample = text[:10000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    except Exception:
        delim = ","
    try:
        has_header = csv.Sniffer().has_header(sample)
    except Exception:
        has_header = True

    df = pd.read_csv(io.StringIO(text), sep=delim, header=0 if has_header else None,
                     dtype="object", engine="python")

    # Promote first row to header if sniffer said no header or headers look wrong
    if not has_header or not df.columns.tolist() or \
       sum(1 for c in map(str, df.columns) if c.lower().startswith("unnamed") or c.isdigit()) > len(df.columns)/2:
        if len(df) > 0:
            df.columns = df.iloc[0].astype(str)
            df = df.iloc[1:].reset_index(drop=True)

    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Coverage Snippet": "Snippet",
        "Content": "Snippet",
        "Network": "Type",
        "Title": "Headline",
    }, inplace=False)

# ── Already uploaded: show preview AFTER load ─────────────────────────────────
if st.session_state.upload_step:
    st.success("File loaded.")
    with st.expander("Uploaded File Preview", expanded=False):
        st.dataframe(st.session_state.df_traditional, use_container_width=True, height=320)
    if st.button("Start Over?"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Initial form (no pre-submit previews) ─────────────────────────────────────
if not st.session_state.upload_step:
    client = st.text_input(
        "Client",
        key="client_name_input",
        placeholder="e.g., Air Canada",
        help="Required to build export file name."
    )
    reporting_period = st.text_input(
        "Reporting period or focus*",
        key="reporting_period_or_focus",
        placeholder="e.g., March 2025",
        help="Required to build export file name (also unlocks admin if you know the key)."
    )
    _sync_admin_from_reporting()

    uploaded = st.file_uploader(
        label="Upload your CSV or XLSX*",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="uploader"
    )

    # Reset file-related session state when a new file is chosen
    if uploaded is not None and uploaded != st.session_state.get("uploaded_file"):
        st.session_state.uploaded_file = uploaded
        st.session_state.excel_sheet_names = []
        st.session_state.excel_selected_sheet = None
        st.session_state.pop("df_traditional", None)
        st.session_state.pop("full_dataset", None)

    # Sheet selection BEFORE submit (no data preview)
    is_excel = False
    if st.session_state.get("uploaded_file") is not None:
        fname = st.session_state.uploaded_file.name.lower()
        is_excel = fname.endswith(".xlsx")

        if is_excel and not st.session_state.excel_sheet_names:
            try:
                xls = pd.ExcelFile(st.session_state.uploaded_file)
                st.session_state.excel_sheet_names = xls.sheet_names
                if xls.sheet_names:
                    st.session_state.excel_selected_sheet = xls.sheet_names[0]
            except Exception as e:
                st.error(f"Could not read workbook sheets: {e}")
                st.stop()

        if is_excel and st.session_state.excel_sheet_names:
            st.selectbox(
                "Choose a worksheet",
                options=st.session_state.excel_sheet_names,
                index=(st.session_state.excel_sheet_names.index(st.session_state.excel_selected_sheet)
                       if st.session_state.excel_selected_sheet in st.session_state.excel_sheet_names else 0),
                key="excel_selected_sheet",
                help="Select which worksheet to load."
            )

    can_submit = (
        bool(client.strip()) and
        bool(reporting_period.strip()) and
        (st.session_state.get("uploaded_file") is not None) and
        ((not is_excel) or (is_excel and st.session_state.get("excel_selected_sheet")))
    )

    if st.button("Load dataset", type="primary", disabled=not can_submit):
        try:
            if is_excel:
                df = pd.read_excel(
                    st.session_state.uploaded_file,
                    sheet_name=st.session_state.excel_selected_sheet,
                    dtype="object",
                )
            else:
                df = read_csv_smart(st.session_state.uploaded_file)

            df = normalize_columns(df)

            # Persist metadata/state
            st.session_state.df_traditional = df.copy()
            st.session_state.full_dataset = df.copy()
            st.session_state.uploaded_file_name = st.session_state.uploaded_file.name
            st.session_state.client_name = client.strip()
            st.session_state.focus = reporting_period.strip()

            _sync_admin_from_reporting()
            st.session_state.upload_step = True
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.session_state.upload_step = False
            st.stop()

# Hard stop so later steps don’t run with half-initialized state
if not st.session_state.get("upload_step"):
    st.stop()


# # 1-Getting_Started.py
#
# import csv
# import io
# import hashlib
# import pandas as pd
# import streamlit as st
# import mig_functions as mig
#
# # ── Title / page marker ───────────────────────────────────────────────────────
# st.title("Getting Started")
# st.session_state.setdefault("current_page", "Getting Started")
#
# # ── Initialize session defaults (idempotent) ──────────────────────────────────
# string_vars = {
#     "page": "1: Getting Started",
#     "sentiment_type": "3-way",
#     "client_name": "",
#     "focus": "",
#     "model_choice": "gpt-5-mini",
#     "counter": 0,
#     "analysis_note": "",
#     "group_ids": "",
#     "sample_size": 0,
#     "highlight_keyword": "",
#     "min_impressions": 0,
#     "min_domain_authority": 0,
#     "pre_prompt": "",
#     "post_prompt": "",
#     "functions": "",
#     "sentiment_examples": "",
#     "uploaded_file_name": "",
# }
# for k, v in string_vars.items():
#     st.session_state.setdefault(k, v)
#
# for name in ["df_traditional", "unique_stories", "full_dataset"]:
#     st.session_state.setdefault(name, pd.DataFrame())
#
# for b in ["upload_step", "config_step", "sentiment_opinion", "random_sample",
#           "toning_config_step", "processing_started"]:
#     st.session_state.setdefault(b, False)
#
# # Additional state for file handling
# st.session_state.setdefault("uploaded_file", None)
# st.session_state.setdefault("excel_sheet_names", [])
# st.session_state.setdefault("excel_selected_sheet", None)
#
# # ── Admin unlock helper ───────────────────────────────────────────────────────
# ADMIN_SHA256 = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")
# st.session_state.setdefault("is_admin", False)
#
# def _sync_admin_from_reporting():
#     rp = (
#         st.session_state.get("reporting_period_or_focus")
#         or st.session_state.get("period")
#         or st.session_state.get("focus")
#         or ""
#     ).strip()
#     is_admin = hashlib.sha256(rp.encode()).hexdigest() == ADMIN_SHA256
#     prev = st.session_state.get("is_admin")
#     st.session_state["is_admin"] = bool(is_admin)
#     if is_admin and prev is not True:
#         st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
#     elif (prev is True) and not is_admin:
#         st.toast("🔒 Admin tools locked.", icon="🔒")
#
# # ── Helpers ───────────────────────────────────────────────────────────────────
# def read_csv_smart(uploaded_file) -> pd.DataFrame:
#     """
#     Robust CSV reader:
#       - Strips BOM (utf-8-sig)
#       - Detects delimiter & header with csv.Sniffer
#       - If Sniffer says 'no header', we still promote first row to header (common exports)
#     """
#     raw = uploaded_file.read()
#     uploaded_file.seek(0)
#
#     # Try utf-8-sig first to drop BOM; fall back to utf-8
#     try:
#         text = raw.decode("utf-8-sig")
#     except UnicodeDecodeError:
#         text = raw.decode("utf-8", errors="replace")
#
#     # Sample for sniffer
#     sample = text[:10000]
#     try:
#         dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
#     except Exception:
#         dialect = csv.get_dialect("excel")  # default comma
#
#     try:
#         has_header = csv.Sniffer().has_header(sample)
#     except Exception:
#         has_header = True  # sane default
#
#     # Read once with detected delimiter
#     df = pd.read_csv(io.StringIO(text),
#                      sep=dialect.delimiter if getattr(dialect, "delimiter", ",") else ",",
#                      header=0 if has_header else None,
#                      dtype="object",
#                      engine="python")
#
#     # If no header, promote first row to header
#     if not has_header:
#         if len(df) > 0:
#             df.columns = df.iloc[0].astype(str)
#             df = df.iloc[1:].reset_index(drop=True)
#
#     # Extra guard: if lots of columns are named like 0/1/2 or 'Unnamed',
#     # assume first row is header and promote.
#     col_str = [str(c) for c in df.columns]
#     unnamed_ratio = sum(1 for c in col_str if c.lower().startswith("unnamed")) / max(1, len(col_str))
#     numeric_like = sum(1 for c in col_str if c.isdigit()) / max(1, len(col_str))
#     if unnamed_ratio > 0.5 or numeric_like > 0.5:
#         if len(df) > 0:
#             df.columns = df.iloc[0].astype(str)
#             df = df.iloc[1:].reset_index(drop=True)
#
#     return df
#
# def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     # Standardize common names used downstream
#     return df.rename(columns={
#         "Coverage Snippet": "Snippet",
#         "Content": "Snippet",
#         "Network": "Type",
#         "Title": "Headline",
#     }, inplace=False)
#
# # ── Already uploaded: minimal controls (no previews) ──────────────────────────
# if st.session_state.upload_step:
#     st.success("File loaded.")
#     if st.button("Start Over?"):
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.rerun()
#
# # ── Initial form ──────────────────────────────────────────────────────────────
# if not st.session_state.upload_step:
#     client = st.text_input(
#         "Client",
#         key="client_name_input",
#         placeholder="e.g., Air Canada",
#         help="Required to build export file name."
#     )
#
#     reporting_period = st.text_input(
#         "Reporting period or focus*",
#         key="reporting_period_or_focus",
#         placeholder="e.g., March 2025",
#         help="Required to build export file name (also unlocks admin if you know the key)."
#     )
#     _sync_admin_from_reporting()
#
#     uploaded = st.file_uploader(
#         label="Upload your CSV or XLSX*",
#         type=["csv", "xlsx"],
#         accept_multiple_files=False,
#         key="uploader"
#     )
#
#     # Reset file-related session state when a new file is chosen
#     if uploaded is not None and uploaded != st.session_state.get("uploaded_file"):
#         st.session_state.uploaded_file = uploaded
#         st.session_state.excel_sheet_names = []
#         st.session_state.excel_selected_sheet = None
#         st.session_state.pop("df_traditional", None)
#         st.session_state.pop("full_dataset", None)
#
#     # Sheet selection (shown BEFORE submit)
#     is_excel = False
#     if st.session_state.get("uploaded_file") is not None:
#         fname = st.session_state.uploaded_file.name.lower()
#         is_excel = fname.endswith(".xlsx")
#
#         if is_excel and not st.session_state.excel_sheet_names:
#             try:
#                 xls = pd.ExcelFile(st.session_state.uploaded_file)
#                 st.session_state.excel_sheet_names = xls.sheet_names
#                 if xls.sheet_names:
#                     st.session_state.excel_selected_sheet = xls.sheet_names[0]
#             except Exception as e:
#                 st.error(f"Could not read workbook sheets: {e}")
#                 st.stop()
#
#         if is_excel and st.session_state.excel_sheet_names:
#             st.selectbox(
#                 "Choose a worksheet",
#                 options=st.session_state.excel_sheet_names,
#                 index=(st.session_state.excel_sheet_names.index(st.session_state.excel_selected_sheet)
#                        if st.session_state.excel_selected_sheet in st.session_state.excel_sheet_names else 0),
#                 key="excel_selected_sheet",
#                 help="Select which worksheet to load."
#             )
#
#     can_submit = (
#         bool(client.strip()) and
#         bool(reporting_period.strip()) and
#         (st.session_state.get("uploaded_file") is not None) and
#         ((not is_excel) or (is_excel and st.session_state.get("excel_selected_sheet")))
#     )
#
#     if st.button("Load dataset", type="primary", disabled=not can_submit):
#         try:
#             if is_excel:
#                 df = pd.read_excel(
#                     st.session_state.uploaded_file,
#                     sheet_name=st.session_state.excel_selected_sheet,
#                     dtype="object",
#                 )
#             else:
#                 # Use robust CSV reader
#                 df = read_csv_smart(st.session_state.uploaded_file)
#
#             df = normalize_columns(df)
#
#             # Persist metadata/state
#             st.session_state.df_traditional = df.copy()
#             st.session_state.full_dataset = df.copy()
#             st.session_state.uploaded_file_name = st.session_state.uploaded_file.name
#             st.session_state.client_name = client.strip()
#             st.session_state.focus = reporting_period.strip()
#
#             _sync_admin_from_reporting()
#
#             st.session_state.upload_step = True
#             st.rerun()
#
#         except Exception as e:
#             st.error(f"Failed to load dataset: {e}")
#             st.session_state.upload_step = False
#             st.stop()
#
# # Hard stop to prevent downstream pages running half-initialized
# if not st.session_state.get("upload_step"):
#     st.stop()
#
