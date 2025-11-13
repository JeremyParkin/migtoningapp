# 1-Getting_Started.py

import streamlit as st
import pandas as pd
import hashlib
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
    "model_choice": "gpt-5-mini",     # normalized default
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

# ── Admin unlock helper (runs every rerun, before guards) ─────────────────────
ADMIN_SHA256 = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")
st.session_state.setdefault("is_admin", False)

def _sync_admin_from_reporting():
    rp = (
        st.session_state.get("reporting_period_or_focus")
        or st.session_state.get("period")  # legacy
        or st.session_state.get("focus")   # legacy storage
        or ""
    ).strip()
    is_admin = hashlib.sha256(rp.encode()).hexdigest() == ADMIN_SHA256
    prev = st.session_state.get("is_admin")
    st.session_state["is_admin"] = bool(is_admin)
    if is_admin and prev is not True:
        st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
    elif (prev is True) and not is_admin:
        st.toast("🔒 Admin tools locked.", icon="🔒")

# ── If already uploaded, show preview + start over ────────────────────────────
if st.session_state.upload_step:
    st.success("File loaded.")
    with st.expander("Uploaded File Preview"):
        st.dataframe(st.session_state.df_traditional, use_container_width=True)

    if st.button("Start Over?"):
        # Clear all state safely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Initial form (when no upload yet) ─────────────────────────────────────────
if not st.session_state.upload_step:
    # Inputs (kept outside a form so the sheet picker/previews react immediately)
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

    # Sheet selection (shown BEFORE submit)
    is_excel = False
    if st.session_state.get("uploaded_file") is not None:
        fname = st.session_state.uploaded_file.name.lower()
        is_excel = fname.endswith(".xlsx")

        if is_excel:
            if not st.session_state.excel_sheet_names:
                try:
                    xls = pd.ExcelFile(st.session_state.uploaded_file)
                    st.session_state.excel_sheet_names = xls.sheet_names
                    if xls.sheet_names:
                        st.session_state.excel_selected_sheet = xls.sheet_names[0]
                except Exception as e:
                    st.error(f"Could not read workbook sheets: {e}")
                    st.stop()

            if st.session_state.excel_sheet_names:
                st.selectbox(
                    "Choose a worksheet",
                    options=st.session_state.excel_sheet_names,
                    index=(st.session_state.excel_sheet_names.index(st.session_state.excel_selected_sheet)
                           if st.session_state.excel_selected_sheet in st.session_state.excel_sheet_names else 0),
                    key="excel_selected_sheet",
                    help="Select which worksheet to load."
                )

                # Optional preview (first 8 rows)
                try:
                    preview = pd.read_excel(
                        st.session_state.uploaded_file,
                        sheet_name=st.session_state.excel_selected_sheet,
                        nrows=4
                    )
                    st.caption("Preview (first 4 rows)")
                    st.dataframe(preview, use_container_width=True)
                except Exception as e:
                    st.warning(f"Preview unavailable: {e}")
        else:
            # CSV preview
            try:
                preview = pd.read_csv(st.session_state.uploaded_file, nrows=8)
                st.caption("Preview (first 8 rows)")
                st.dataframe(preview, use_container_width=True)
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")

    # Submit is enabled only when all required inputs are ready
    can_submit = (
        bool(client.strip()) and
        bool(reporting_period.strip()) and
        (st.session_state.get("uploaded_file") is not None) and
        ((not is_excel) or (is_excel and st.session_state.get("excel_selected_sheet")))
    )

    if st.button("Load dataset", type="primary", disabled=not can_submit):
        try:
            # Read the chosen file
            if is_excel:
                df = pd.read_excel(
                    st.session_state.uploaded_file,
                    sheet_name=st.session_state.excel_selected_sheet
                )
            else:
                df = pd.read_csv(st.session_state.uploaded_file)

            # Normalize/rename common columns for downstream pages
            df.rename(columns={
                "Coverage Snippet": "Snippet",
                "Content": "Snippet",
                "Network": "Type",
                "Title": "Headline",
            }, inplace=True)

            # Persist metadata/state
            st.session_state.df_traditional = df.copy()
            st.session_state.full_dataset = df.copy()
            st.session_state.uploaded_file_name = st.session_state.uploaded_file.name
            st.session_state.client_name = client.strip()
            st.session_state.focus = reporting_period.strip()

            # Re-sync admin (in case value changed just before submit)
            _sync_admin_from_reporting()

            # Advance
            st.session_state.upload_step = True
            st.success(f"Loaded {len(df):,} rows from "
                       f"{'sheet “'+st.session_state.excel_selected_sheet+'”' if is_excel else 'CSV'}.")
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
# import streamlit as st
# import pandas as pd
# import hashlib
# import mig_functions as mig
#
#
# # ── Sidebar / title ───────────────────────────────────────────────────────────
# st.title("Getting Started")
#
# # ── Initialize session defaults (idempotent) ──────────────────────────────────
# string_vars = {
#     "page": "1: Getting Started",
#     "sentiment_type": "3-way",
#     "client_name": "",
#     "focus": "",
#     "model_choice": "gpt-5-mini",     # normalized default
#     "counter": 0,
#     "analysis_note": "",
#     "group_ids": "",
#     "sample_size": 0,
#     "highlight_keyword": "",
#     "current_page": "Getting Started",
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
# for b in ["upload_step", "config_step", "sentiment_opinion", "random_sample", "toning_config_step", "processing_started"]:
#     st.session_state.setdefault(b, False)
#
# # ── Admin unlock helper (runs every rerun, before guards) ─────────────────────
# ADMIN_SHA256 = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")
#
# def _sync_admin_from_reporting():
#     # Try the canonical key first; fall back to legacy key names if present
#     rp = (
#         st.session_state.get("reporting_period_or_focus")
#         or st.session_state.get("period")  # legacy
#         or st.session_state.get("focus")   # legacy storage
#         or ""
#     ).strip()
#     is_admin = hashlib.sha256(rp.encode()).hexdigest() == ADMIN_SHA256
#     prev = st.session_state.get("is_admin")
#     st.session_state["is_admin"] = bool(is_admin)
#     # Optional toasts only when this page is interacted with
#     # if prev is not None and prev != is_admin:
#     #     st.toast("🔓 Admin tools unlocked for this session.", icon="✅") if is_admin else st.toast("🔒 Admin tools locked.", icon="🔒")
#
#     if is_admin and prev is not True:
#         _ = st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
#     elif (prev is True) and not is_admin:
#         _ = st.toast("🔒 Admin tools locked.", icon="🔒")
#     # Do NOT wrap that in st.write(), st.text(), lists, tuples, or return it from a function used by Streamlit.
#
#
#
# st.session_state.setdefault("is_admin", False)
# _sync_admin_from_reporting()
#
# st.session_state.current_page = "Getting Started"
#
# # ── If already uploaded, show preview + start over ────────────────────────────
# if st.session_state.upload_step:
#     st.success("File uploaded.")
#     with st.expander("Uploaded File Preview"):
#         st.dataframe(st.session_state.df_traditional, use_container_width=True)
#
#     if st.button("Start Over?"):
#         # Clear all state safely
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.rerun()
#
# # ── Initial form (when no upload yet) ─────────────────────────────────────────
# if not st.session_state.upload_step:
#     client = st.text_input(
#         "Client",
#         key="client_name_input",
#         placeholder="eg. Air Canada",
#         help="Required to build export file name."
#     )
#
#     # Use canonical key so the admin helper picks it up live
#     reporting_period = st.text_input(
#         "Reporting period or focus*",
#         key="reporting_period_or_focus",
#         placeholder="eg. March 2025",
#         help="Required to build export file name."
#     )
#
#     # Keep admin flag synced live as they type (no submit needed)
#     _sync_admin_from_reporting()
#
#     uploaded_file = st.file_uploader(
#         label="Upload your CSV or XLSX*",
#         type=["csv", "xlsx"],
#         accept_multiple_files=False,
#     )
#
#     submitted = st.button("Submit", type="primary")
#
#     if submitted and (not client or not reporting_period or uploaded_file is None):
#         st.error("Missing required form inputs above.")
#     elif submitted:
#         # Try to read the file
#         try:
#             if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
#                 xls = pd.ExcelFile(uploaded_file)
#                 if len(xls.sheet_names) > 1:
#                     sheet = st.selectbox("Select a sheet:", xls.sheet_names)
#                     st.session_state.df_traditional = pd.read_excel(xls, sheet_name=sheet)
#                 else:
#                     st.session_state.df_traditional = pd.read_excel(xls)
#             elif uploaded_file.type == "text/csv":
#                 st.session_state.df_traditional = pd.read_csv(uploaded_file)
#             else:
#                 st.error("Unsupported file type.")
#                 st.stop()
#         except Exception as e:
#             st.error(f"Could not read file: {e}")
#             st.stop()
#
#         # Normalize/rename common columns for downstream pages
#         st.session_state.df_traditional.rename(columns={
#             "Coverage Snippet": "Snippet",
#             "Content": "Snippet",
#             "Network": "Type",
#             "Title": "Headline",
#         }, inplace=True)
#
#         # Persist metadata
#         st.session_state.uploaded_file_name = uploaded_file.name
#         st.session_state.client_name = client.strip()
#         st.session_state.focus = reporting_period.strip()
#
#         # Keep a pristine copy
#         st.session_state.full_dataset = st.session_state.df_traditional.copy()
#
#         # Re-sync admin (in case the field changed right before submit)
#         _sync_admin_from_reporting()
#
#         # Mark step complete and advance
#         st.session_state.upload_step = True
#         st.rerun()
#
