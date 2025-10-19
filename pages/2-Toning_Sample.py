import streamlit as st
import pandas as pd
import mig_functions as mig
import math
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import time


# --- Configure Streamlit page ---
st.set_page_config(page_title="MIG Toning App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
                   layout="wide")

# --- Record active page in session state ---
st.session_state.current_page = 'Toning Sample'

# --- Render standard sidebar ---
# mig.standard_sidebar()

# --- Initialize elapsed time tracker ---
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0

def normalize_text(text):
    """Normalise whitespace, casing, and punctuation for clustering."""
    text = str(text)  # Ensure string input
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Trim leading and trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Collapse repeated whitespace
    text = text.translate(str.maketrans('', '', string.punctuation))  # Optionally drop punctuation
    return text


def remove_extra_spaces(text):
    """Trim excess whitespace from text while preserving content."""
    text = str(text)  # Ensure string input
    text = text.strip()  # Remove leading and trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace repeated spaces with single spaces
    return text


def preprocess_online_news(df):
    """Group ONLINE/ONLINE_NEWS records by published date and headline."""
    # Resolve column naming differences across uploads
    date_column = 'Date' if 'Date' in df.columns else 'Published Date'
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    if date_column not in df.columns or 'Headline' not in df.columns:
        st.warning("Required columns for preprocessing (Date, Headline) are missing!")
        return df

    # Focus on ONLINE and ONLINE_NEWS records only
    online_df = df[df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])].copy()

    # Convert dates to datetime then back to a consistent string
    online_df[date_column] = pd.to_datetime(online_df[date_column], errors='coerce')
    online_df['Published Date'] = online_df[date_column].dt.strftime('%Y-%m-%d')

    # Deduplicate by date and headline
    grouped = online_df.groupby(['Published Date', 'Headline']).first().reset_index()

    # Merge cleaned online coverage with all remaining media types
    non_online_df = df[~df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])]
    preprocessed_df = pd.concat([grouped, non_online_df], ignore_index=True)

    return preprocessed_df



def cluster_similar_stories(df, similarity_threshold=0.85):
    """Cluster similar stories using agglomerative clustering."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet']).toarray()

    # Compute cosine distances
    cosine_distance_matrix = cosine_distances(tfidf_matrix)

    # Use agglomerative clustering with a distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Allow the algorithm to determine cluster count
        metric="precomputed",  # Provide the cosine distance matrix directly
        linkage="average",  # Average linkage performs well with cosine distance
        distance_threshold=1 - similarity_threshold  # Convert similarity to distance
    )
    cluster_labels = clustering.fit_predict(cosine_distance_matrix)

    # Persist cluster labels as a group identifier
    df['Group ID'] = cluster_labels
    return df



def cluster_by_media_type(df, similarity_threshold=0.92):
    """Cluster stories by media type while keeping group identifiers unique."""
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    # Process each media type independently
    unique_media_types = df[type_column].unique()

    clustered_frames = []
    group_id_offset = 0  # Offset to keep group identifiers unique across media types

    for media_type in unique_media_types:
        st.write(f"Processing media type: {media_type}")

        # Filter rows for the current media type
        media_df = df[df[type_column] == media_type].copy()

        if not media_df.empty:
            # Replace missing text fields with empty strings
            media_df['Headline'] = media_df['Headline'].fillna("")
            media_df['Snippet'] = media_df['Snippet'].fillna("")

            # Skip media types that do not contain meaningful text
            if media_df[['Headline', 'Snippet']].apply(lambda x: x.str.strip()).eq("").all(axis=None):
                st.warning(f"Skipping media type {media_type} due to missing headlines and snippets.")
                continue

            # Normalize and clean text for clustering
            media_df['Normalized Headline'] = media_df['Headline'].apply(normalize_text)
            media_df['Normalized Snippet'] = media_df['Snippet'].apply(normalize_text)

            if len(media_df) == 1:
                # Assign a unique group to single records
                media_df['Group ID'] = group_id_offset
                group_id_offset += 1
            else:
                # Cluster stories for this media type
                media_df = cluster_similar_stories(media_df, similarity_threshold=similarity_threshold)

                # Offset group identifiers to ensure global uniqueness
                media_df['Group ID'] += group_id_offset
                group_id_offset += media_df['Group ID'].max() + 1

            # Remove helper columns created for clustering
            normalized_columns = [col for col in ['Normalized Headline', 'Normalized Snippet'] if
                                  col in media_df.columns]
            media_df = media_df.drop(columns=normalized_columns, errors='ignore')

            clustered_frames.append(media_df)

    # Combine all clustered frames
    return pd.concat(clustered_frames, ignore_index=True) if clustered_frames else df



def assign_group_ids(duplicates):
    """Assign a group identifier to each article based on cluster membership."""
    group_id = 0
    group_ids = {}
    for i, similar_indices in duplicates.items():
        if i not in group_ids:
            group_ids[i] = group_id
            for index in similar_indices:
                group_ids[index] = group_id
            group_id += 1
    return group_ids


def identify_duplicates(cluster_labels):
    """Group article indices by shared cluster labels."""
    from collections import defaultdict
    duplicates = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        duplicates[label].append(idx)
    return duplicates


def clean_snippet(snippet):
    """Remove leading broadcast markers such as '>>>' or '>>'."""
    if snippet.startswith(">>>"):
        return snippet.replace(">>>", "", 1)
    if snippet.startswith(">>"):
        return snippet.replace(">>", "", 1)
    else:
        return snippet

# --- Render primary page title ---
st.title("Configuration")

# --- Guard against missing prerequisites ---
if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')


else:
    if not st.session_state.config_step:
        named_entity = st.session_state.client_name

        # Offer sampling strategies
        sampling_option = st.radio(
            'Sampling options:',
            ['Take a statistically significant sample', 'Set my own sample size', 'Use full data'],
            help="Choose how to sample your uploaded data set."
        )

        if sampling_option == 'Take a statistically significant sample':
            def calculate_sample_size(N, confidence_level=0.95, margin_of_error=0.05, p=0.5):
                # Use a 95% confidence interval
                Z = 1.96  # 95% confidence

                numerator = N * (Z ** 2) * p * (1 - p)
                denominator = (margin_of_error ** 2) * (N - 1) + (Z ** 2) * p * (1 - p)

                return math.ceil(numerator / denominator)


            population_size = len(st.session_state.full_dataset)
            st.session_state.sample_size = calculate_sample_size(population_size)
            st.write(f"Calculated sample size: {st.session_state.sample_size}")

        elif sampling_option == 'Set my own sample size':
            max_sample = len(st.session_state.full_dataset)

            custom_sample_size = st.number_input(
                "Enter your desired sample size:",
                min_value=1,
                max_value=max_sample,
                step=1,
                value=min(400, max_sample)
            )


            st.session_state.sample_size = int(custom_sample_size)

        else:
            st.session_state.sample_size = len(st.session_state.full_dataset)
            st.write(f"Full data size: {st.session_state.sample_size}")

        similarity_threshold = 0.93
        st.session_state.similarity_threshold = similarity_threshold

        if st.button("Save Configuration", type='primary'):
            start_time = time.time()
            st.session_state.config_step = True

            sample_size = st.session_state.sample_size

            # Apply the selected sampling strategy
            if sample_size < len(st.session_state.full_dataset):
                df = st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(drop=True)
            else:
                df = st.session_state.df_traditional.copy()

            st.write(f"Full data size: {len(st.session_state.df_traditional)}")
            st.write(f"Sample size used: {len(df)}")  # Confirm the sample size

            # Align column naming for snippet text
            if 'Coverage Snippet' in df.columns:
                df.rename(columns={'Coverage Snippet': 'Snippet'}, inplace=True)

            # Normalise and preprocess text fields
            df['Headline'] = df['Headline'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(clean_snippet)
            df['Normalized Headline'] = df['Headline'].apply(normalize_text)
            df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)

            # Cluster similar stories
            df = cluster_by_media_type(df, similarity_threshold=similarity_threshold)

            # Ensure group identifiers are present on the DataFrame
            df['Group ID'] = df['Group ID']

            # Remove temporary normalised columns
            df = df.drop(columns=['Normalized Headline', 'Normalized Snippet'], errors='ignore')

            # Persist updates to session state
            st.session_state.df_traditional = df.copy()

            # Calculate the size of each group
            group_counts = df.groupby('Group ID').size().reset_index(name='Group Count')

            # Build a unique-story view with counts
            unique_stories = df.groupby('Group ID').agg(lambda x: x.iloc[0]).reset_index()
            unique_stories_with_counts = unique_stories.merge(group_counts, on='Group ID')

            # Provide quick confirmation outputs
            st.write(f"Number of unique stories: {len(unique_stories_with_counts)}")
            st.write(unique_stories_with_counts.head())

            # Sort unique stories by descending group count
            unique_stories_sorted = unique_stories_with_counts.sort_values(by='Group Count',
                                                                           ascending=False).reset_index(drop=True)
            # Update session state
            st.session_state.unique_stories = unique_stories_sorted

            # Track the processing duration
            end_time = time.time()
            st.session_state.elapsed_time = end_time - start_time
            st.rerun()


    else:
        st.success('Configuration Completed!')
        st.write(f"Time taken: {st.session_state.elapsed_time:.2f} seconds")
        st.write(f"Full data size: {len(st.session_state.full_dataset)}")
        if 'sample_size' in st.session_state:
            st.write(f"Sample size used: {st.session_state.sample_size}")
        st.write(f"Unique stories in data: {len(st.session_state.unique_stories)}")
        st.dataframe(st.session_state.unique_stories)


        def reset_config():
            """Clear configuration-related state so the workflow can restart."""
            st.session_state.config_step = False
            st.session_state.sentiment_opinion = None
            st.session_state.random_sample = None
            st.session_state.similarity_threshold = None
            st.session_state.sentiment_instruction = None
            st.session_state.df_traditional = st.session_state.full_dataset.copy()
            st.session_state.counter = 0
            st.session_state.pop('unique_stories', None)


        # Offer a reset button to restart configuration
        if st.button("Reset Configuration"):
            reset_config()
            st.rerun()  # Rerun the script to reflect the reset state