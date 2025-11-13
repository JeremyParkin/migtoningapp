# 7-Save_Load.py — Save / Load session for MIG Toning App

import io
import dill
import pandas as pd
import streamlit as st
from datetime import datetime
# import mig_functions as mig  # optional; uncomment if you use the shared sidebar

# --- Page setup (safe: 1x per page) ---
st.set_page_config(
    layout="wide",
    page_title="Save & Load",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
)
st.session_state.current_page = "Save & Load"
# mig.standard_sidebar()  # uncomment if you use the shared sidebar here

st.title("Save & Load")
st.divider()

# --- Define which DataFrames this app uses (idempotent) ---
default_df_names = ["df_traditional", "unique_stories", "full_dataset"]
st.session_state.setdefault("df_names", default_df_names)

# --- Keys we want to persist (non-DF scalars) ---
PERSIST_KEYS = [
    # high-level
    "client_name", "focus", "sentiment_type", "model_choice",
    "upload_step", "config_step", "toning_config_step", "counter",

    # prompt bits
    "pre_prompt", "post_prompt", "sentiment_instruction", "functions",
    "highlight_keyword", "highlight_regex_str",

    # UI defaults (Toning Config inputs)
    "ui_primary_names", "ui_alternate_names", "ui_spokespeople", "ui_products",
    "ui_sentiment_type", "ui_toning_rationale",

    # spot-check workflow
    "initial_ai_label", "spot_checked_groups", "accepted_initial", "spot_idx",

    # toning interface position pointers (add the one your page actually uses)
    "tone_idx", "toning_idx", "toning_focus_gid", "current_group_id", "toning_current_gid",

    # misc
    "uploaded_file_name", "current_page", "min_impressions", "min_domain_authority",
]

# --- Helpers to make sets pickle-friendly ---
def _to_plain(obj):
    if isinstance(obj, set):
        return {"__type__": "set", "data": list(obj)}
    return obj

def _from_plain(obj):
    if isinstance(obj, dict) and obj.get("__type__") == "set":
        return set(obj.get("data", []))
    return obj

# --- SAVE ---
def save_session_state():
    payload = {}

    # Persist scalars
    for k in PERSIST_KEYS:
        if k in st.session_state:
            payload[k] = _to_plain(st.session_state[k])

    # Persist df_names explicitly and the DataFrames themselves
    df_names = st.session_state.get("df_names", default_df_names)
    payload["df_names"] = df_names
    for name in df_names:
        if name in st.session_state and isinstance(st.session_state[name], pd.DataFrame):
            payload[name] = st.session_state[name]

    # Serialize
    blob = dill.dumps(payload)
    fname_client = (st.session_state.get("client_name") or "Session").strip()
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_name = f"{fname_client} - sentiment-session - {ts}.pkl"

    st.download_button(
        label="Download Session File",
        data=blob,
        file_name=file_name,
        mime="application/octet-stream",
        use_container_width=True,
    )

# --- LOAD ---
def load_session_state(uploaded_file):
    uploaded_file.seek(0)
    payload = dill.loads(uploaded_file.read())

    # Restore ALL non-DataFrame values (including strings)
    for k, v in payload.items():
        if k == "df_names":
            continue
        if isinstance(v, pd.DataFrame):
            continue
        st.session_state[k] = _from_plain(v)

    # Restore DataFrames
    restored_df_names = payload.get("df_names", default_df_names)
    real_restored = []
    for name in restored_df_names:
        df = payload.get(name)
        if isinstance(df, pd.DataFrame):
            st.session_state[name] = df
            real_restored.append(name)
    st.session_state["df_names"] = real_restored or default_df_names

    # Guard: ensure downstream-required columns exist
    for df_name in st.session_state.get("df_names", default_df_names):
        df = st.session_state.get(df_name)
        if isinstance(df, pd.DataFrame):
            for col in [
                "Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence",
                "AI Sentiment Rationale", "Translated Headline", "Translated Body"
            ]:
                if col not in df.columns:
                    df[col] = None

    st.success("Session state loaded successfully!")

# --- View: SAVE ---
st.header("Save")
if not st.session_state.get("upload_step"):
    st.info("Upload a CSV/XLSX to enable saving a meaningful session.")
else:
    st.write("Download a snapshot of your current session (data, prompts, and progress).")
    save_session_state()

st.divider()

# --- View: LOAD ---
st.header("Load")
st.write("Restore a previously downloaded session (.pkl).")
up = st.file_uploader("Restore a Previous Session", type="pkl", label_visibility="hidden")
if up is not None:
    load_session_state(up)
    # Optionally jump back to a page:
    # st.switch_page("pages/3-Toning_Interface.py")


# # 7-Save_Load.py  — Save / Load session for MIG Toning App
#
# import io
# import dill
# import pandas as pd
# import streamlit as st
# from datetime import datetime
#
# import mig_functions as mig  # optional; used for sidebar if you call it
#
# # --- Page setup (safe: 1x per page) ---
# st.set_page_config(
#     layout="wide",
#     page_title="Save & Load",
#     page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
# )
# st.session_state.current_page = "Save & Load"
# # mig.standard_sidebar()  # uncomment if you use the shared sidebar here
#
# # st.title("Save & Load")
# # st.divider()
#
# # --- Define which DataFrames this app uses (idempotent) ---
# default_df_names = ["df_traditional", "unique_stories", "full_dataset"]
# st.session_state.setdefault("df_names", default_df_names)
#
# # --- Keys we want to persist (non-DF scalars) ---
# PERSIST_KEYS = [
#     # high-level
#     "client_name", "focus", "sentiment_type", "model_choice",
#     "upload_step", "config_step", "toning_config_step",
#
#     # prompt bits
#     "pre_prompt", "post_prompt", "sentiment_instruction", "functions",
#     "highlight_keyword", "highlight_regex_str",
#
#     # UI defaults users typed
#     "ui_primary_names", "ui_alternate_names", "ui_spokespeople", "ui_products",
#     "ui_sentiment_type", "ui_toning_rationale",
#
#     # spot-check workflow state
#     "initial_ai_label", "spot_checked_groups", "accepted_initial",
#     "spot_idx",
#
#     # translations toggle columns will be in DFs, not here
#     # misc
#     "uploaded_file_name", "current_page", "min_impressions", "min_domain_authority",
# ]
#
# # --- Helpers to make sets JSON/pickle friendly ---
# def _to_plain(obj):
#     if isinstance(obj, set):
#         return {"__type__": "set", "data": list(obj)}
#     return obj
#
# def _from_plain(obj):
#     if isinstance(obj, dict) and obj.get("__type__") == "set":
#         return set(obj.get("data", []))
#     return obj
#
# # 7-Save_Load.py  — Save / Load session for MIG Toning App
#
# import io
# import dill
# import pandas as pd
# import streamlit as st
# from datetime import datetime
#
# import mig_functions as mig  # optional; used for sidebar if you call it
#
# # --- Page setup (safe: 1x per page) ---
# st.set_page_config(
#     layout="wide",
#     page_title="Save & Load",
#     page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
# )
# st.session_state.current_page = "Save & Load"
# # mig.standard_sidebar()  # uncomment if you use the shared sidebar here
#
# st.title("Save & Load")
# st.divider()
#
# # --- Define which DataFrames this app uses (idempotent) ---
# default_df_names = ["df_traditional", "unique_stories", "full_dataset"]
# st.session_state.setdefault("df_names", default_df_names)
#
# # --- Keys we want to persist (non-DF scalars) ---
# PERSIST_KEYS = [
#     # high-level
#     "client_name", "focus", "sentiment_type", "model_choice",
#     "upload_step", "config_step", "toning_config_step",
#
#     # prompt bits
#     "pre_prompt", "post_prompt", "sentiment_instruction", "functions",
#     "highlight_keyword", "highlight_regex_str",
#
#     # UI defaults users typed
#     "ui_primary_names", "ui_alternate_names", "ui_spokespeople", "ui_products",
#     "ui_sentiment_type", "ui_toning_rationale",
#
#     # spot-check workflow state
#     "initial_ai_label", "spot_checked_groups", "accepted_initial",
#     "spot_idx",
#
#     # translations toggle columns will be in DFs, not here
#     # misc
#     "uploaded_file_name", "current_page", "min_impressions", "min_domain_authority",
# ]
#
# # --- Helpers to make sets JSON/pickle friendly ---
# def _to_plain(obj):
#     if isinstance(obj, set):
#         return {"__type__": "set", "data": list(obj)}
#     return obj
#
# def _from_plain(obj):
#     if isinstance(obj, dict) and obj.get("__type__") == "set":
#         return set(obj.get("data", []))
#     return obj
#
# def save_session_state():
#     # Build a clean payload
#     payload = {}
#
#     # Persist scalars
#     for k in PERSIST_KEYS:
#         if k in st.session_state:
#             payload[k] = _to_plain(st.session_state[k])
#
#     # Persist df_names explicitly and the DataFrames themselves
#     df_names = st.session_state.get("df_names", default_df_names)
#     payload["df_names"] = df_names
#     for name in df_names:
#         if name in st.session_state and isinstance(st.session_state[name], pd.DataFrame):
#             payload[name] = st.session_state[name]
#
#     # Serialize
#     blob = dill.dumps(payload)
#     fname_client = (st.session_state.get("client_name") or "Session").strip()
#     ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
#     file_name = f"{fname_client} - sentiment-session - {ts}.pkl"
#
#     st.download_button(
#         label="Download Session File",
#         data=blob,
#         file_name=file_name,
#         mime="application/octet-stream",
#         use_container_width=True,
#     )
#
# def load_session_state(uploaded_file):
#     uploaded_file.seek(0)
#     payload = dill.loads(uploaded_file.read())
#
#     # Restore scalars first
#     for k, v in payload.items():
#         if k in ("df_names",):  # skip here; handle after
#             continue
#         if isinstance(v, pd.DataFrame):
#             continue
#         st.session_state[k] = _from_plain(v)
#
#     # DataFrames (from the list we saved)
#     restored_df_names = payload.get("df_names", default_df_names)
#     real_restored = []
#     for name in restored_df_names:
#         df = payload.get(name)
#         if isinstance(df, pd.DataFrame):
#             st.session_state[name] = df
#             real_restored.append(name)
#     st.session_state["df_names"] = real_restored or default_df_names
#
#     # If a save was taken pre-config, these may be missing; keep app robust
#     for df_name in st.session_state.get("df_names", default_df_names):
#         df = st.session_state.get(df_name)
#         if isinstance(df, pd.DataFrame):
#             # Ensure columns expected downstream exist
#             for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
#                         "Translated Headline", "Translated Body"]:
#                 if col not in df.columns:
#                     df[col] = None
#
#     st.success("Session state loaded successfully!")
#
# # --- View: SAVE ---
# st.header("Save")
# if not st.session_state.get("upload_step"):
#     st.info("Upload a CSV/XLSX to enable saving a meaningful session.")
# else:
#     st.write("Download a snapshot of your current session (data, prompts, and spot-check progress).")
#     save_session_state()
#
# st.divider()
#
# # --- View: LOAD ---
# st.header("Load")
# st.write("Restore a previously downloaded session (.pkl).")
# up = st.file_uploader("Restore a Previous Session", type="pkl", label_visibility="hidden")
# if up is not None:
#     load_session_state(up)
#     # Optional: jump users back to the spot-checks or config page after load
#     # st.switch_page("pages/6-Spot_Checks.py")
#
# def save_session_state():
#     # 1) Start with everything except the DataFrames listed in df_names
#     session_data = {k: v for k, v in st.session_state.items()
#                     if k not in st.session_state.df_names}
#
#     # 2) Make sure we explicitly keep toning position keys (whatever you use)
#     #    Add/adjust these names if your toning page uses different keys
#     for k in [
#         "tone_idx",                # common index var
#         "toning_idx",              # alt name, if used
#         "toning_focus_gid",        # preferred: stable Group ID
#         "current_group_id",        # some pages store this
#         "toning_current_gid"       # your own choice
#     ]:
#         if k in st.session_state:
#             session_data[k] = st.session_state[k]
#
#     # 3) Save df_names explicitly so LOAD can use it
#     session_data["df_names"] = st.session_state.df_names
#
#     # 4) Save the actual DataFrames
#     for df_name in st.session_state.df_names:
#         if df_name in st.session_state:
#             session_data[df_name] = st.session_state[df_name]
#
#     serialized_data = dill.dumps(session_data)
#     file_name = f"{st.session_state.client_name} - {dt_string}.pkl"
#     st.download_button(
#         label="Download Session File",
#         data=serialized_data,
#         file_name=file_name,
#         mime="application/octet-stream",
#     )
#
# # def save_session_state():
# #     # Build a clean payload
# #     payload = {}
# #
# #     # Persist scalars
# #     for k in PERSIST_KEYS:
# #         if k in st.session_state:
# #             payload[k] = _to_plain(st.session_state[k])
# #
# #     # Persist df_names explicitly and the DataFrames themselves
# #     df_names = st.session_state.get("df_names", default_df_names)
# #     payload["df_names"] = df_names
# #     for name in df_names:
# #         if name in st.session_state and isinstance(st.session_state[name], pd.DataFrame):
# #             payload[name] = st.session_state[name]
# #
# #     # Serialize
# #     blob = dill.dumps(payload)
# #     fname_client = (st.session_state.get("client_name") or "Session").strip()
# #     ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
# #     file_name = f"{fname_client} - sentiment-session - {ts}.pkl"
# #
# #     st.download_button(
# #         label="Download Session File",
# #         data=blob,
# #         file_name=file_name,
# #         mime="application/octet-stream",
# #         use_container_width=True,
# #     )
#
# def load_session_state(uploaded_file):
#     if uploaded_file is not None:
#         uploaded_file.seek(0)
#         session_data = dill.loads(uploaded_file.read())
#
#         # Restore ALL non-DataFrame values (do NOT skip strings)
#         for key, value in session_data.items():
#             if not isinstance(value, pd.DataFrame):
#                 st.session_state[key] = value
#
#         # Restore any DataFrame (and legacy CSV-string if you kept that path)
#         restored_df_names = []
#         for key, value in session_data.items():
#             if isinstance(value, pd.DataFrame):
#                 st.session_state[key] = value
#                 restored_df_names.append(key)
#             elif isinstance(value, str) and "\n" in value:
#                 # keep if you support legacy CSV-in-pkl
#                 df = pd.read_csv(io.StringIO(value))
#                 if 'Date' in df.columns:
#                     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#                 st.session_state[key] = df
#                 restored_df_names.append(key)
#
#         st.session_state.df_names = restored_df_names
#         st.session_state.pickle_load = True
#         st.success("Session state loaded successfully!")
#
#
# # def load_session_state(uploaded_file):
# #     uploaded_file.seek(0)
# #     payload = dill.loads(uploaded_file.read())
# #
# #     # Restore scalars first
# #     for k, v in payload.items():
# #         if k in ("df_names",):  # skip here; handle after
# #             continue
# #         if isinstance(v, pd.DataFrame):
# #             continue
# #         st.session_state[k] = _from_plain(v)
# #
# #     # DataFrames (from the list we saved)
# #     restored_df_names = payload.get("df_names", default_df_names)
# #     real_restored = []
# #     for name in restored_df_names:
# #         df = payload.get(name)
# #         if isinstance(df, pd.DataFrame):
# #             st.session_state[name] = df
# #             real_restored.append(name)
# #     st.session_state["df_names"] = real_restored or default_df_names
# #
# #     # If a save was taken pre-config, these may be missing; keep app robust
# #     for df_name in st.session_state.get("df_names", default_df_names):
# #         df = st.session_state.get(df_name)
# #         if isinstance(df, pd.DataFrame):
# #             # Ensure columns expected downstream exist
# #             for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
# #                         "Translated Headline", "Translated Body"]:
# #                 if col not in df.columns:
# #                     df[col] = None
# #
# #     st.success("Session state loaded successfully!")
#
# # --- View: SAVE ---
# st.header("Save")
# if not st.session_state.get("upload_step"):
#     st.info("Upload a CSV/XLSX to enable saving a meaningful session.")
# else:
#     st.write("Download a snapshot of your current session (data, prompts, and spot-check progress).")
#     save_session_state()
#
# st.divider()
#
# # --- View: LOAD ---
# st.header("Load")
# st.write("Restore a previously downloaded session (.pkl).")
# up = st.file_uploader("Restore a Previous Session", type="pkl", label_visibility="hidden")
# if up is not None:
#     load_session_state(up)
#     # Optional: jump users back to the spot-checks or config page after load
#     # st.switch_page("pages/6-Spot_Checks.py")
