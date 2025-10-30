# app.py
import streamlit as st
st.set_page_config(page_title="MIG Toning App", page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png", layout="wide")

import mig_functions as mig
mig.run_navigation(position="sidebar")
