# 5-Download.py

import io
import altair as alt
import pandas as pd
import streamlit as st
import mig_functions as mig

# ================== Page setup ==================
st.set_page_config(
    page_title="MIG Download",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
mig.standard_sidebar()
st.title("Download")

# Remember last page's download buffer only within this page
if st.session_state.get('current_page') != 'Download':
    st.session_state.download_data = None
st.session_state.current_page = 'Download'

# ================== Guards ==================
if not st.session_state.get('upload_step'):
    st.error('Please upload a CSV/XLSX before trying this step.')
    st.stop()
if not st.session_state.get('config_step'):
    st.error('Please run the configuration step before trying this step.')
    st.stop()

# ================== Helpers ==================
def ensure_columns():
    """Make sure we have the columns we expect in both DFs."""
    for df_name in ['df_traditional', 'unique_stories']:
        if df_name not in st.session_state:
            st.session_state[df_name] = pd.DataFrame()

    # Human label column
    if 'Assigned Sentiment' not in st.session_state.df_traditional.columns:
        st.session_state.df_traditional['Assigned Sentiment'] = pd.NA

    # AI output columns
    for col in ['AI Sentiment', 'AI Sentiment Confidence', 'AI Sentiment Rationale']:
        if col not in st.session_state.df_traditional.columns:
            st.session_state.df_traditional[col] = None
        if col not in st.session_state.unique_stories.columns:
            st.session_state.unique_stories[col] = None

    # Group ID column (needed for mapping)
    if 'Group ID' not in st.session_state.df_traditional.columns:
        st.session_state.df_traditional['Group ID'] = pd.NA
    if 'Group ID' not in st.session_state.unique_stories.columns:
        st.session_state.unique_stories['Group ID'] = pd.NA

def refresh_hybrid_sentiment():
    """Always recompute Hybrid Sentiment = Assigned Sentiment else AI Sentiment."""
    ensure_columns()

    st.session_state.df_traditional['Hybrid Sentiment'] = (
        st.session_state.df_traditional['Assigned Sentiment']
        .where(st.session_state.df_traditional['Assigned Sentiment'].notna(),
               st.session_state.df_traditional['AI Sentiment'])
    )

    # Mirror a 1-per-group hybrid label in unique_stories (first non-null seen in df_traditional)
    gid2hybrid = (
        st.session_state.df_traditional
        .dropna(subset=['Group ID'])
        .groupby('Group ID')['Hybrid Sentiment'].first()
    )
    st.session_state.unique_stories['Hybrid Sentiment'] = \
        st.session_state.unique_stories['Group ID'].map(gid2hybrid)

def label_order():
    """Return the correct label order from sentiment_type."""
    stype = st.session_state.get('sentiment_type', '3-way')
    if str(stype).strip().lower().startswith('5'):
        return ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL',
                'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT']
    return ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT']

def build_counts_from(series: pd.Series, ordered_labels: list) -> pd.DataFrame:
    """Build count/percentage table from a label series using ordered_labels."""
    series = series.dropna()
    if series.empty:
        return pd.DataFrame({'Sentiment': ordered_labels, 'Count': [0]*len(ordered_labels), 'Percentage': [0.0]*len(ordered_labels)})

    counts = series.value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    # Ensure all labels exist
    for s in ordered_labels:
        if s not in counts['Sentiment'].values:
            counts = pd.concat([counts, pd.DataFrame({'Sentiment':[s], 'Count':[0]})], ignore_index=True)
    counts['Sentiment'] = pd.Categorical(counts['Sentiment'], categories=ordered_labels, ordered=True)
    counts.sort_values('Sentiment', inplace=True)
    total = int(counts['Count'].sum()) or 1
    counts['Percentage'] = counts['Count'] / total
    return counts

def color_scale_for(ordered_labels):
    mapping = {
        "POSITIVE": "green",
        "NEUTRAL": "gold",
        "NEGATIVE": "crimson",
        "VERY POSITIVE": "darkgreen",
        "SOMEWHAT POSITIVE": "limegreen",
        "SOMEWHAT NEGATIVE": "coral",
        "VERY NEGATIVE": "maroon",
        "NOT RELEVANT": "dimgray",
    }
    return alt.Scale(domain=ordered_labels, range=[mapping.get(l, "grey") for l in ordered_labels])

def write_excel(traditional: pd.DataFrame, raw_data: pd.DataFrame) -> io.BytesIO:
    """Return an in-memory Excel file with CLEAN TRAD and RAW DATA sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
        workbook = writer.book
        number_format = workbook.add_format({'num_format': '#,##0'})
        currency_format = workbook.add_format({'num_format': '$#,##0'})

        # CLEAN TRAD
        df_out = traditional.copy()
        if 'Impressions' in df_out.columns:
            df_out = df_out.sort_values(by=['Impressions'], ascending=False)

        df_out.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
        ws1 = writer.sheets['CLEAN TRAD']
        ws1.set_tab_color('black')

        max_row, max_col = df_out.shape
        ws1.add_table(0, 0, max_row, max_col - 1, {'columns': [{'header': c} for c in df_out.columns]})
        ws1.set_default_row(22)

        # A few useful column widths if they exist
        def safe_set(col_name: str, width: int, fmt=None):
            if col_name in df_out.columns:
                idx = df_out.columns.get_loc(col_name)  # 0-based
                # Convert 0-based to Excel letters; handle up to Z
                if idx < 26:
                    letter = chr(ord('A') + idx)
                    ws1.set_column(f'{letter}:{letter}', width, fmt)

        safe_set('Date', 12)
        safe_set('Time', 12)
        safe_set('Time Zone', 12)
        safe_set('Author', 18)
        safe_set('Headline', 50)
        safe_set('Impressions', 12, number_format)
        safe_set('AVE', 12, currency_format)
        safe_set('Assigned Sentiment', 16)
        safe_set('AI Sentiment', 18)
        safe_set('AI Sentiment Confidence', 10)
        safe_set('Hybrid Sentiment', 18)
        safe_set('Group ID', 10)

        ws1.freeze_panes(1, 0)

        # RAW DATA
        if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
            raw_data.to_excel(writer, sheet_name='RAW DATA', startrow=1, header=False, index=False)
            ws2 = writer.sheets['RAW DATA']
            max_row2, max_col2 = raw_data.shape
            ws2.add_table(0, 0, max_row2, max_col2 - 1, {'columns': [{'header': c} for c in raw_data.columns]})
            ws2.set_default_row(22)
            ws2.freeze_panes(1, 0)

    output.seek(0)
    return output

# ================== Recompute Hybrid on load ==================
refresh_hybrid_sentiment()

# ================== Summary / Stats ==================
ordered_labels = label_order()

# Ensure categorical for plotting consistency
st.session_state.df_traditional['Hybrid Sentiment'] = pd.Categorical(
    st.session_state.df_traditional['Hybrid Sentiment'],
    categories=ordered_labels, ordered=True
)

hybrid_counts = build_counts_from(st.session_state.df_traditional['Hybrid Sentiment'], ordered_labels)

# Quick metrics
total_rows = len(st.session_state.df_traditional)
human_labeled = st.session_state.df_traditional['Assigned Sentiment'].notna().sum()
ai_filled = st.session_state.df_traditional['Hybrid Sentiment'].notna().sum() - human_labeled
st.caption(
    f"**Rows:** {total_rows} · **Human-labeled:** {human_labeled} · **AI-filled:** {max(ai_filled, 0)}"
)

# Stats section
st.subheader('Hybrid Sentiment Statistics')
col1, col2 = st.columns([3, 2], gap='large')

with col1:
    color_scale = color_scale_for(ordered_labels)

    base = alt.Chart(hybrid_counts).encode(
        y=alt.Y('Sentiment:N', sort=ordered_labels),
    )

    bar = base.mark_bar().encode(
        x=alt.X('Count:Q', title='Mentions'),
        color=alt.Color('Sentiment:N', scale=color_scale, legend=None),
        tooltip=['Sentiment', alt.Tooltip('Percentage:Q', format='.1%', title='Percent'), 'Count:Q']
    )

    text = base.mark_text(
        align='left', baseline='middle', dx=3, color='ivory'
    ).encode(
        x='Count:Q',
        text=alt.Text('Percentage:Q', format='.1%')
    )

    chart = (bar + text).properties(width=600, height=260).configure_view(strokeWidth=0).configure_axisLeft(labelLimit=200)
    st.altair_chart(chart, use_container_width=True)

with col2:
    tbl = hybrid_counts.copy()
    tbl['Percentage'] = (tbl['Percentage'] * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(tbl, hide_index=True)

# ================== Data preview ==================
with st.expander('View Processed Data (CLEAN TRAD source)'):
    st.dataframe(st.session_state.df_traditional)

# ================== Excel Export ==================
traditional = st.session_state.df_traditional
raw_data = st.session_state.get('full_dataset', pd.DataFrame())

if 'download_data' not in st.session_state:
    st.session_state.download_data = None

with st.form("download_form"):
    st.subheader("Generate your cleaned data workbook")
    submitted = st.form_submit_button("Build Excel", type="primary")
    if submitted:
        with st.spinner("Building workbook…"):
            st.session_state.download_data = write_excel(traditional, raw_data)

if st.session_state.download_data is not None:
    export_name = f"{st.session_state.client_name} - {st.session_state.focus} - MIG_Toned.xlsx"
    st.download_button('Download Excel', st.session_state.download_data, file_name=export_name, type="primary")



#
#
#
# import streamlit as st
# import pandas as pd
# import mig_functions as mig
# import io
# import altair as alt
#
#
# # Set Streamlit configuration
# st.set_page_config(page_title="MIG Bulk AI Analysis",
#                    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
#                    layout="wide")
#
#
# # Sidebar configuration
# mig.standard_sidebar()
#
# st.title("Download")
#
#
# if st.session_state.get('current_page') != 'Download':
#     st.session_state.download_data = None
#
#
# st.session_state.current_page = 'Download'
#
# def refresh_hybrid_sentiment():
#     """Always recompute Hybrid Sentiment = Assigned Sentiment else AI Sentiment."""
#     # Ensure columns exist
#     for col in ["Assigned Sentiment", "AI Sentiment"]:
#         if col not in st.session_state.df_traditional.columns:
#             st.session_state.df_traditional[col] = pd.NA
#
#     # Overwrite (fresh each load)
#     st.session_state.df_traditional["Hybrid Sentiment"] = (
#         st.session_state.df_traditional["Assigned Sentiment"]
#         .where(st.session_state.df_traditional["Assigned Sentiment"].notna(),
#                st.session_state.df_traditional["AI Sentiment"])
#     )
#
#     # Optional: keep a 1-row-per-group view in unique_stories too
#     if "unique_stories" in st.session_state:
#         if "Hybrid Sentiment" not in st.session_state.unique_stories.columns:
#             st.session_state.unique_stories["Hybrid Sentiment"] = None
#         # Map by Group ID (takes the first hybrid label found for that group)
#         gid2hybrid = (
#             st.session_state.df_traditional
#             .dropna(subset=["Group ID"])
#             .groupby("Group ID")["Hybrid Sentiment"].first()
#         )
#         st.session_state.unique_stories["Hybrid Sentiment"] = \
#             st.session_state.unique_stories["Group ID"].map(gid2hybrid)
#
#
# refresh_hybrid_sentiment()
#
#
# if not st.session_state.upload_step:
#     st.error('Please upload a CSV/XLSX before trying this step.')
# elif not st.session_state.config_step:
#     st.error('Please run the configuration step before trying this step.')
# else:
#     if 'AI Sentiment' in st.session_state.df_traditional.columns:
#
#         if pd.notna(st.session_state.df_traditional['AI Sentiment']).any():
#             # Count the frequency of each sentiment and calculate the percentage
#             sentiment_counts = st.session_state.df_traditional['AI Sentiment'].value_counts().reset_index()
#             sentiment_counts.columns = ['Sentiment', 'Count']
#             total = sentiment_counts['Count'].sum()
#             sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total)
#
#             if st.session_state.sentiment_type == '5-way':
#                 custom_order = ['VERY POSITIVE', 'SOMEWHAT POSITIVE', 'NEUTRAL', 'SOMEWHAT NEGATIVE', 'VERY NEGATIVE', 'NOT RELEVANT']
#             else:
#                 custom_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'NOT RELEVANT']
#
#             # Apply the custom order to 'Assigned Sentiment' in df_traditional
#             st.session_state.df_traditional['AI Sentiment'] = pd.Categorical(
#                 st.session_state.df_traditional['AI Sentiment'], categories=custom_order, ordered=True)
#
#
#             # Ensure all sentiments are in the dataframe (to avoid key errors in sorting)
#             for sentiment in custom_order:
#                 if sentiment not in sentiment_counts['Sentiment'].values:
#                     # Create a new DataFrame for the missing sentiment and concatenate it
#                     missing_sentiment_df = pd.DataFrame({'Sentiment': [sentiment], 'Count': [0], 'Percentage': [0.0]})
#                     sentiment_counts = pd.concat([sentiment_counts, missing_sentiment_df], ignore_index=True)
#
#
#
#             # Sort the dataframe based on the custom order
#             sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order, ordered=True)
#
#             st.subheader('Sentiment Statistics')
#
#             col1, col2 = st.columns([3,2], gap='large')
#             with col1:
#                 color_mapping = {
#                     "POSITIVE": "green",
#                     "NEUTRAL": "yellow",
#                     "NEGATIVE": "red",
#                     "VERY POSITIVE": "darkgreen",
#                     "SOMEWHAT POSITIVE": "limegreen",
#                     "SOMEWHAT NEGATIVE": "coral",
#                     "VERY NEGATIVE": "maroon",
#                     "NOT RELEVANT": "dimgray"
#                 }
#
#                 # Create the color domain and range from the sorted DataFrame
#                 color_domain = custom_order
#                 color_range = [color_mapping.get(sentiment, "grey") for sentiment in color_domain]
#
#                 # Create the color scale
#                 color_scale = alt.Scale(domain=color_domain, range=color_range)
#
#
#                 # Create the base chart for the horizontal bar chart
#                 base = alt.Chart(sentiment_counts).encode(
#                     y=alt.Y('Sentiment:N', sort=custom_order),
#                 )
#
#                 # Create the bar chart
#                 bar_chart = base.mark_bar().encode(
#                     x='Count:Q',
#                     color=alt.Color('Sentiment:N', scale=color_scale, legend=None),
#                     tooltip=['Sentiment', alt.Tooltip('Percentage', format='.1f', title='Percent'), 'Count']
#                 )
#
#                 # Create the text labels for the bars
#                 text = base.mark_text(
#                     align='left',
#                     baseline='middle',
#                     dx=3,  # Nudges text to the right so it doesn't appear on top of the bar
#                     color='ivory'  # Set text color to off-white
#                 ).encode(
#                     x=alt.X('Count:Q', axis=alt.Axis(title='Mentions')),
#                     text=alt.Text('Percentage:N', format='.1%')
#                 )
#
#                 # Combine the bar and text charts
#                 chart = (bar_chart + text).properties(
#                     # title='Sentiment Bar Chart',
#                     width=600,
#                     height=250
#                 ).configure_view(
#                     strokeWidth=0
#                 ).configure_axisLeft(
#                     labelLimit=180  # Increase label limit to accommodate longer labels
#                 )
#
#                 st.altair_chart(chart, use_container_width=True)
#
#
#
#             with col2:
#
#                 # Adding percentage column to the stats table
#                 sentiment_stats = sentiment_counts.copy()
#
#
#                 # Convert 'Sentiment' to a categorical type with the custom order
#                 sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=custom_order,
#                                                                ordered=True)
#
#                 # Sort 'sentiment_counts' by the categorical order
#                 sentiment_counts.sort_values(by='Sentiment', inplace=True)
#
#                 sentiment_counts['Percentage'] = (sentiment_counts['Percentage'] * 100).apply(lambda x: "{:.1f}%".format(x))
#
#
#                 # Display the table without the index
#                 st.dataframe(sentiment_counts, hide_index=True,)
#
#
#
#     with st.expander('View Processed Data'):
#         st.dataframe(st.session_state.df_traditional)
#
#
#     traditional = st.session_state.df_traditional
#
#
#     # Initialize a session state variable for the download link
#     if 'download_data' not in st.session_state:
#         st.session_state.download_data = None
#
#
#     with st.form("my_form_download"):
#         st.subheader("Generate your cleaned data workbook")
#         submitted = st.form_submit_button("Go!", type="primary")
#
#         if submitted:
#             with st.spinner('Building workbook now...'):
#                 output = io.BytesIO()
#                 writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd')
#
#                 workbook = writer.book
#
#                 # Add some cell formats.
#                 number_format = workbook.add_format({'num_format': '#,##0'})
#                 currency_format = workbook.add_format({'num_format': '$#,##0'})
#
#                 # if len(traditional) > 0:
#                 #     traditional = traditional.sort_values(by=['Impressions'], ascending=False)
#                 #     traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
#                 #     worksheet1 = writer.sheets['CLEAN TRAD']
#                 #     worksheet1.set_tab_color('black')
#                 #
#                 #     (max_row, max_col) = traditional.shape
#                 #     column_settings = [{'header': column} for column in traditional.columns]
#                 #     worksheet1.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})
#
#                 if len(traditional) > 0:
#                     if 'Impressions' in traditional.columns:
#                         traditional = traditional.sort_values(by=['Impressions'], ascending=False)
#
#                     traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
#                     worksheet1 = writer.sheets['CLEAN TRAD']
#                     worksheet1.set_tab_color('black')
#
#                     (max_row, max_col) = traditional.shape
#                     column_settings = [{'header': column} for column in traditional.columns]
#                     worksheet1.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})
#
#                     # Apply column-specific formats
#                     worksheet1.set_default_row(22)
#                     worksheet1.set_column('A:A', 12, None)  # date
#                     worksheet1.set_column('B:B', 12, None)  # time
#                     worksheet1.set_column('C:C', 12, None)  # timezone
#                     worksheet1.set_column('G:G', 12, None)  # author
#                     worksheet1.set_column('H:H', 40, None)  # headline
#                     worksheet1.set_column('Y:Y', 12, number_format)  # impressions
#                     worksheet1.set_column('X:X', 12, currency_format)  # AVE
#                     worksheet1.set_column('AH:AH', 12, None)  # Assigned Sentiment
#                     worksheet1.set_column('AI:AI', 12, None)  # Flagged
#                     worksheet1.set_column('AJ:AJ', 12, None)  # Group ID
#                     worksheet1.freeze_panes(1, 0)
#
#                     # Add another worksheet with st.session_state.full_dataset called 'RAW DATA'
#                     raw_data = st.session_state.full_dataset
#                     raw_data.to_excel(writer, sheet_name='RAW DATA', startrow=1, header=False, index=False)
#                     worksheet2 = writer.sheets['RAW DATA']
#                     # worksheet2.set_tab_color('grey')
#
#                     (max_row, max_col) = raw_data.shape
#                     column_settings = [{'header': column} for column in raw_data.columns]
#                     worksheet2.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})
#
#
#                     # Apply column-specific formats to the 'RAW DATA' worksheet
#                     # (Modify this part based on the structure of your raw data)
#                     worksheet2.set_default_row(22)
#                     worksheet2.set_column('A:A', 12, None)  # date
#                     worksheet2.set_column('B:B', 12, None)  # time
#                     worksheet2.set_column('C:C', 10, None)  # timezone
#                     worksheet2.set_column('G:G', 12, None)  # author
#                     worksheet2.set_column('H:H', 40, None)  # headline
#                     worksheet2.set_column('Y:Y', 12, number_format)  # impressions
#                     worksheet2.set_column('X:X', 12, currency_format)  # AVE
#                     worksheet2.freeze_panes(1, 0)
#
#
#                 workbook.close()
#                 output.seek(0)  # Important: move back to the beginning of the BytesIO object
#
#                 # Update the session state variable with the download data
#                 st.session_state.download_data = output
#
#     # Check if the download data is ready and display the download button
#     if st.session_state.download_data is not None:
#         export_name = f"{st.session_state.client_name} - {st.session_state.focus} - BulkAI.xlsx"
#         st.download_button('Download', st.session_state.download_data, file_name=export_name, type="primary")