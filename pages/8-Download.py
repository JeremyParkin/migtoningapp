# 8-Download.py

import io
import altair as alt
import pandas as pd
import streamlit as st
import mig_functions as mig

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="MIG Download",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)
# mig.standard_sidebar()
st.title("Download")

# Remember last page's download buffer only within this page
if st.session_state.get('current_page') != 'Download':
    st.session_state.download_data = None
st.session_state.current_page = 'Download'

# --- Validate required workflow steps ---
if not st.session_state.get('upload_step'):
    st.error('Please upload a CSV/XLSX before trying this step.')
    st.stop()
if not st.session_state.get('config_step'):
    st.error('Please run the configuration step before trying this step.')
    st.stop()

# --- Helper utilities ---
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

def refresh_hybrid_sentiment():
    """Recompute Hybrid Sentiment and place it immediately left of Assigned Sentiment in df_traditional."""
    ensure_columns()

    # 1) Recompute hybrid
    st.session_state.df_traditional['Hybrid Sentiment'] = (
        st.session_state.df_traditional['Assigned Sentiment']
        .where(st.session_state.df_traditional['Assigned Sentiment'].notna(),
               st.session_state.df_traditional['AI Sentiment'])
    )

    # 2) Reorder columns in-place: Hybrid just before Assigned
    cols = list(st.session_state.df_traditional.columns)
    if "Assigned Sentiment" in cols and "Hybrid Sentiment" in cols:
        # Move Hybrid to just before Assigned
        cols.remove("Hybrid Sentiment")
        insert_at = cols.index("Assigned Sentiment")
        cols.insert(insert_at, "Hybrid Sentiment")
        st.session_state.df_traditional = st.session_state.df_traditional[cols]

    # 3) Mirror one-per-group hybrid label in unique_stories
    gid2hybrid = (
        st.session_state.df_traditional
        .dropna(subset=['Group ID'])
        .groupby('Group ID')['Hybrid Sentiment'].first()
    )
    st.session_state.unique_stories['Hybrid Sentiment'] = \
        st.session_state.unique_stories['Group ID'].map(gid2hybrid)


# --- Recompute hybrid sentiment on load ---
refresh_hybrid_sentiment()

# --- Summary and statistics ---
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

# --- Data preview ---
with st.expander('View Processed Data (CLEAN TRAD source)'):
    st.dataframe(st.session_state.df_traditional)

# --- Excel export ---
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

