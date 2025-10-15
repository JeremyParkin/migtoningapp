import streamlit as st
import pandas as pd
import mig_functions as mig


# Set Streamlit configuration
st.set_page_config(page_title="MIG Toning App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
                   layout="wide")

# Sidebar configuration
mig.standard_sidebar()

st.title("Getting Started")



# Initialize Session State Variables
string_vars = {'page': '1: Getting Started', 'sentiment_type': '3-way', 'client_name': '', 'focus': '',
               'model_choice': 'GPT-5-mini', 'counter': 0, 'analysis_note': '', 'group_ids':'',
               'sample_size': 0, 'highlight_keyword':'', 'current_page': 'Getting Started', 'min_impressions': 0, 'min_domain_authority': 0,
               'pre_prompt':'','post_prompt':'','functions':'','sentiment_examples':'', 'uploaded_file_name':''}

for key, value in string_vars.items():
    if key not in st.session_state:
        st.session_state[key] = value

df_vars = ['df_traditional', 'unique_stories', 'full_dataset']
for _ in df_vars:
    if _ not in st.session_state:
        st.session_state[_] = pd.DataFrame()



bool_vars = ['upload_step', 'config_step', 'sentiment_opinion', 'random_sample', 'toning_config_step', 'processing_started']
for _ in bool_vars:
    if _ not in st.session_state:
        st.session_state[_] = False

st.session_state.current_page = 'Getting Started'


if st.session_state.upload_step:
    st.success('File uploaded.')
    with st.expander('Uploaded File Preview'):
        st.dataframe(st.session_state.df_traditional)

    if st.button('Start Over?'):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()


if not st.session_state.upload_step:

    client = st.text_input('Client', placeholder='eg. Air Canada', key='client',
                           help='Required to build export file name.')
    focus = st.text_input('Reporting period or focus*', placeholder='eg. March 2025', key='period',
                          help='Required to build export file name.')
    uploaded_file = st.file_uploader(label='Upload your CSV or XLSX*', type=['csv', 'xlsx'],
                                     accept_multiple_files=False,
                                     )


    if not uploaded_file == None:
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Read the xlsx file
            excel_file = pd.ExcelFile(uploaded_file)
            # Get the sheet names
            sheet_names = excel_file.sheet_names
            # If there is more than one sheet, let the user select which one to use
            if len(sheet_names) > 1:

                sheet = st.selectbox('Select a sheet:', sheet_names)
                st.session_state.df_traditional = pd.read_excel(excel_file, sheet_name=sheet)
            else:
                st.session_state.df_traditional = pd.read_excel(excel_file)
        elif uploaded_file.type == 'text/csv':
            st.session_state.df_traditional = pd.read_csv(uploaded_file)


    submitted = st.button("Submit", type="primary")

    if submitted and (client == "" or focus == "" or uploaded_file is None):
        st.error('Missing required form inputs above.')

    elif submitted:
        with st.spinner("Converting file format."):

            st.session_state.uploaded_file_name = uploaded_file.name



            st.session_state.client_name = client
            st.session_state.focus = focus
            st.session_state.full_dataset = st.session_state.df_traditional.copy()
            st.session_state.df_traditional.rename(columns={
                "Coverage Snippet": "Snippet",
                "Content": "Snippet",
                "Network": "Type",
                "Title": "Headline"
            }, inplace=True)


            st.session_state.upload_step = True

            st.rerun()