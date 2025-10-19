import streamlit as st
import pandas as pd
import hashlib
import mig_functions as mig

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIG Toning App",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)

# ── Sidebar / title ───────────────────────────────────────────────────────────
# mig.standard_sidebar()
st.title("Getting Started")

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
    "current_page": "Getting Started",
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

for b in ["upload_step", "config_step", "sentiment_opinion", "random_sample", "toning_config_step", "processing_started"]:
    st.session_state.setdefault(b, False)

# ── Admin unlock helper (runs every rerun, before guards) ─────────────────────
ADMIN_SHA256 = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")

def _sync_admin_from_reporting():
    # Try the canonical key first; fall back to legacy key names if present
    rp = (
        st.session_state.get("reporting_period_or_focus")
        or st.session_state.get("period")  # legacy
        or st.session_state.get("focus")   # legacy storage
        or ""
    ).strip()
    is_admin = hashlib.sha256(rp.encode()).hexdigest() == ADMIN_SHA256
    prev = st.session_state.get("is_admin")
    st.session_state["is_admin"] = bool(is_admin)
    # Optional toasts only when this page is interacted with
    # if prev is not None and prev != is_admin:
    #     st.toast("🔓 Admin tools unlocked for this session.", icon="✅") if is_admin else st.toast("🔒 Admin tools locked.", icon="🔒")

    if is_admin and prev is not True:
        _ = st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
    elif (prev is True) and not is_admin:
        _ = st.toast("🔒 Admin tools locked.", icon="🔒")
    # Do NOT wrap that in st.write(), st.text(), lists, tuples, or return it from a function used by Streamlit.



st.session_state.setdefault("is_admin", False)
_sync_admin_from_reporting()

st.session_state.current_page = "Getting Started"

# ── If already uploaded, show preview + start over ────────────────────────────
if st.session_state.upload_step:
    st.success("File uploaded.")
    with st.expander("Uploaded File Preview"):
        st.dataframe(st.session_state.df_traditional, use_container_width=True)

    if st.button("Start Over?"):
        # Clear all state safely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Initial form (when no upload yet) ─────────────────────────────────────────
if not st.session_state.upload_step:
    client = st.text_input(
        "Client",
        key="client_name_input",
        placeholder="eg. Air Canada",
        help="Required to build export file name."
    )

    # Use canonical key so the admin helper picks it up live
    reporting_period = st.text_input(
        "Reporting period or focus*",
        key="reporting_period_or_focus",
        placeholder="eg. March 2025",
        help="Required to build export file name."
    )

    # Keep admin flag synced live as they type (no submit needed)
    _sync_admin_from_reporting()

    uploaded_file = st.file_uploader(
        label="Upload your CSV or XLSX*",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )

    submitted = st.button("Submit", type="primary")

    if submitted and (not client or not reporting_period or uploaded_file is None):
        st.error("Missing required form inputs above.")
    elif submitted:
        # Try to read the file
        try:
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                xls = pd.ExcelFile(uploaded_file)
                if len(xls.sheet_names) > 1:
                    sheet = st.selectbox("Select a sheet:", xls.sheet_names)
                    st.session_state.df_traditional = pd.read_excel(xls, sheet_name=sheet)
                else:
                    st.session_state.df_traditional = pd.read_excel(xls)
            elif uploaded_file.type == "text/csv":
                st.session_state.df_traditional = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        # Normalize/rename common columns for downstream pages
        st.session_state.df_traditional.rename(columns={
            "Coverage Snippet": "Snippet",
            "Content": "Snippet",
            "Network": "Type",
            "Title": "Headline",
        }, inplace=True)

        # Persist metadata
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.client_name = client.strip()
        st.session_state.focus = reporting_period.strip()

        # Keep a pristine copy
        st.session_state.full_dataset = st.session_state.df_traditional.copy()

        # Re-sync admin (in case the field changed right before submit)
        _sync_admin_from_reporting()

        # Mark step complete and advance
        st.session_state.upload_step = True
        st.rerun()



# import streamlit as st
# import pandas as pd
# import mig_functions as mig
#
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Toning App",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
#                    layout="wide")
#
# # Sidebar configuration
# mig.standard_sidebar()
#
# st.title("Getting Started")
#
#
#
# # Initialize Session State Variables
# string_vars = {'page': '1: Getting Started', 'sentiment_type': '3-way', 'client_name': '', 'focus': '',
#                'model_choice': 'GPT-5-mini', 'counter': 0, 'analysis_note': '', 'group_ids':'',
#                'sample_size': 0, 'highlight_keyword':'', 'current_page': 'Getting Started', 'min_impressions': 0, 'min_domain_authority': 0,
#                'pre_prompt':'','post_prompt':'','functions':'','sentiment_examples':'', 'uploaded_file_name':''}
#
# for key, value in string_vars.items():
#     if key not in st.session_state:
#         st.session_state[key] = value
#
# df_vars = ['df_traditional', 'unique_stories', 'full_dataset']
# for _ in df_vars:
#     if _ not in st.session_state:
#         st.session_state[_] = pd.DataFrame()
#
#
#
# bool_vars = ['upload_step', 'config_step', 'sentiment_opinion', 'random_sample', 'toning_config_step', 'processing_started']
# for _ in bool_vars:
#     if _ not in st.session_state:
#         st.session_state[_] = False
#
# st.session_state.current_page = 'Getting Started'
#
#
# if st.session_state.upload_step:
#     st.success('File uploaded.')
#     with st.expander('Uploaded File Preview'):
#         st.dataframe(st.session_state.df_traditional)
#
#     if st.button('Start Over?'):
#         for key in st.session_state.keys():
#             del st.session_state[key]
#         st.rerun()
#
#
# if not st.session_state.upload_step:
#
#     client = st.text_input('Client', placeholder='eg. Air Canada', key='client',
#                            help='Required to build export file name.')
#     focus = st.text_input('Reporting period or focus*', placeholder='eg. March 2025', key='period',
#                           help='Required to build export file name.')
#     uploaded_file = st.file_uploader(label='Upload your CSV or XLSX*', type=['csv', 'xlsx'],
#                                      accept_multiple_files=False,
#                                      )
#
#
#     if not uploaded_file == None:
#         if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
#             # Read the xlsx file
#             excel_file = pd.ExcelFile(uploaded_file)
#             # Get the sheet names
#             sheet_names = excel_file.sheet_names
#             # If there is more than one sheet, let the user select which one to use
#             if len(sheet_names) > 1:
#
#                 sheet = st.selectbox('Select a sheet:', sheet_names)
#                 st.session_state.df_traditional = pd.read_excel(excel_file, sheet_name=sheet)
#             else:
#                 st.session_state.df_traditional = pd.read_excel(excel_file)
#         elif uploaded_file.type == 'text/csv':
#             st.session_state.df_traditional = pd.read_csv(uploaded_file)
#
#
#     submitted = st.button("Submit", type="primary")
#
#     if submitted and (client == "" or focus == "" or uploaded_file is None):
#         st.error('Missing required form inputs above.')
#
#     elif submitted:
#         with st.spinner("Converting file format."):
#
#             st.session_state.uploaded_file_name = uploaded_file.name
#
#
#
#             st.session_state.client_name = client
#             st.session_state.focus = focus
#             # --- Admin unlock via "Reporting period or focus*" ---
#             import hashlib
#
#             ADMIN_HASH = "60fe74406e7f353ed979f350f2fbb6a2e8690a5fa7d1b0c32983d1d8b3f95f67"  # sha256("Admin1234")
#
#             reporting_val = (st.session_state.get("reporting_period") or "").strip()  # <-- whatever key you use
#             is_admin = hashlib.sha256(reporting_val.encode()).hexdigest() == ADMIN_HASH
#
#             # Persist (toggle live if they change the field)
#             prev = st.session_state.get("is_admin")
#             st.session_state.is_admin = bool(is_admin)
#             if is_admin and prev is not True:
#                 st.toast("🔓 Admin tools unlocked for this session.", icon="✅")
#             elif (prev is True) and not is_admin:
#                 st.toast("🔒 Admin tools locked.", icon="🔒")
#
#             st.session_state.full_dataset = st.session_state.df_traditional.copy()
#             st.session_state.df_traditional.rename(columns={
#                 "Coverage Snippet": "Snippet",
#                 "Content": "Snippet",
#                 "Network": "Type",
#                 "Title": "Headline"
#             }, inplace=True)
#
#
#             st.session_state.upload_step = True
#
#             st.rerun()