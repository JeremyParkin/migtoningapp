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


# Set Streamlit configuration
st.set_page_config(page_title="MIG Toning App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
                   layout="wide")

# Set the current page in session state
st.session_state.current_page = 'Toning Sample'

# Sidebar configuration
mig.standard_sidebar()

# Initialize st.session_state.elapsed_time if it does not exist
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0

def normalize_text(text):
    """Convert to lowercase, remove extra spaces, remove punctuation, etc."""
    text = str(text)     # Convert to string in case the input is not a string
    text = text.lower()     # Convert to lowercase
    text = text.strip()     # Remove extra spaces from the beginning and end
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    text = text.translate(str.maketrans('', '', string.punctuation))     # Remove punctuation (optional)
    return text


def remove_extra_spaces(text):
    """Remove extra spaces from the beginning and end of a string and replace multiple spaces with a single space."""
    text = str(text)     # Convert to string in case the input is not a string
    text = text.strip()     # Remove extra spaces from the beginning and end
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    return text


def preprocess_online_news(df):
    """Pre-process ONLINE/ONLINE_NEWS articles by grouping by Date and Headline."""
    # Handle column name variations
    date_column = 'Date' if 'Date' in df.columns else 'Published Date'
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    if date_column not in df.columns or 'Headline' not in df.columns:
        st.warning("Required columns for preprocessing (Date, Headline) are missing!")
        return df

    # Filter only ONLINE and ONLINE_NEWS articles
    online_df = df[df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])].copy()

    # Ensure Date is in datetime format and extract year/month/day
    online_df[date_column] = pd.to_datetime(online_df[date_column], errors='coerce')
    online_df['Published Date'] = online_df[date_column].dt.strftime('%Y-%m-%d')

    # Group by Published Date and Headline
    grouped = online_df.groupby(['Published Date', 'Headline']).first().reset_index()

    # Merge grouped data back with non-online rows
    non_online_df = df[~df[type_column].isin(['ONLINE', 'ONLINE_NEWS'])]
    preprocessed_df = pd.concat([grouped, non_online_df], ignore_index=True)

    return preprocessed_df



def cluster_similar_stories(df, similarity_threshold=0.85):
    """Cluster similar stories using agglomerative clustering."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet']).toarray()

    # Compute cosine distances
    cosine_distance_matrix = cosine_distances(tfidf_matrix)

    # Use Agglomerative Clustering with a distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let the algorithm decide the number of clusters
        metric="precomputed",  # Use precomputed cosine distances
        linkage="average",  # Average linkage for cosine distances
        distance_threshold=1 - similarity_threshold  # Convert similarity to distance
    )
    cluster_labels = clustering.fit_predict(cosine_distance_matrix)

    # Add cluster labels as 'Group ID'
    df['Group ID'] = cluster_labels
    return df



def cluster_by_media_type(df, similarity_threshold=0.92):
    """Cluster stories by media type and ensure unique Group IDs across media types."""
    type_column = 'Media Type' if 'Media Type' in df.columns else 'Type'

    # Identify unique media types
    unique_media_types = df[type_column].unique()

    clustered_frames = []
    group_id_offset = 0  # Offset to ensure unique Group IDs across media types

    for media_type in unique_media_types:
        st.write(f"Processing media type: {media_type}")

        # Filter data for the current media type
        media_df = df[df[type_column] == media_type].copy()

        if not media_df.empty:
            # Fill missing Headline/Snippet with empty strings
            media_df['Headline'] = media_df['Headline'].fillna("")
            media_df['Snippet'] = media_df['Snippet'].fillna("")

            # Skip processing if all headlines and snippets are empty
            if media_df[['Headline', 'Snippet']].apply(lambda x: x.str.strip()).eq("").all(axis=None):
                st.warning(f"Skipping media type {media_type} due to missing headlines and snippets.")
                continue

            # Normalize and clean text
            media_df['Normalized Headline'] = media_df['Headline'].apply(normalize_text)
            media_df['Normalized Snippet'] = media_df['Snippet'].apply(normalize_text)

            if len(media_df) == 1:
                # Assign a unique Group ID to the single row
                media_df['Group ID'] = group_id_offset
                group_id_offset += 1
            else:
                # Cluster stories for this media type
                media_df = cluster_similar_stories(media_df, similarity_threshold=similarity_threshold)

                # Offset Group IDs to make them unique
                media_df['Group ID'] += group_id_offset
                group_id_offset += media_df['Group ID'].max() + 1

            # Drop normalized columns
            normalized_columns = [col for col in ['Normalized Headline', 'Normalized Snippet'] if
                                  col in media_df.columns]
            media_df = media_df.drop(columns=normalized_columns, errors='ignore')

            clustered_frames.append(media_df)

    # Combine all clustered frames
    return pd.concat(clustered_frames, ignore_index=True) if clustered_frames else df



def assign_group_ids(duplicates):
    """Assign a group ID to each article based on the similarity matrix."""
    group_id = 0
    group_ids = {}
    for i, similar_indices in duplicates.items():
        if i not in group_ids:
            group_ids[i] = group_id
            for index in similar_indices:
                group_ids[index] = group_id
            group_id += 1
    return group_ids


def identify_duplicates(cluster_labels):  ## refactored for agg clustering
    """Group articles based on cluster labels."""
    from collections import defaultdict
    duplicates = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        duplicates[label].append(idx)
    return duplicates


def clean_snippet(snippet):
    """Remove the '>>>' or '>>' from braodcast snippets."""
    if snippet.startswith(">>>"):
        return snippet.replace(">>>", "", 1)
    if snippet.startswith(">>"):
        return snippet.replace(">>", "", 1)
    else:
        return snippet

# Main title of the page
st.title("Configuration")

# Check if the upload step is completed
if not st.session_state.upload_step:
    st.error('Please upload a CSV/XLSX before trying this step.')


else:
    if not st.session_state.config_step:
        named_entity = st.session_state.client_name

        # Sampling options
        sampling_option = st.radio(
            'Sampling options:',
            ['Take a statistically significant sample', 'Set my own sample size', 'Use full data'],
            help="Choose how to sample your uploaded data set."
        )

        if sampling_option == 'Take a statistically significant sample':
            def calculate_sample_size(N, confidence_level=0.95, margin_of_error=0.05, p=0.5):
                # Z-score for 95% confidence level
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

            # Apply sampling
            if sample_size < len(st.session_state.full_dataset):
                df = st.session_state.df_traditional.sample(n=sample_size, random_state=1).reset_index(drop=True)
            else:
                df = st.session_state.df_traditional.copy()

            st.write(f"Full data size: {len(st.session_state.df_traditional)}")
            st.write(f"Sample size used: {len(df)}")  # Confirm the sample size

            # Check if 'Coverage Snippet' column exists and rename it to 'Snippet'
            if 'Coverage Snippet' in df.columns:
                df.rename(columns={'Coverage Snippet': 'Snippet'}, inplace=True)

            # Normalize and preprocess
            df['Headline'] = df['Headline'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
            df['Snippet'] = df['Snippet'].apply(clean_snippet)
            df['Normalized Headline'] = df['Headline'].apply(normalize_text)
            df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)

            # Cluster similar stories
            df = cluster_by_media_type(df, similarity_threshold=similarity_threshold)

            # Assign group IDs directly to the DataFrame
            df['Group ID'] = df['Group ID']

            # Drop the normalized columns
            df = df.drop(columns=['Normalized Headline', 'Normalized Snippet'], errors='ignore')

            # Update session state
            st.session_state.df_traditional = df.copy()

            # Calculate group counts
            group_counts = df.groupby('Group ID').size().reset_index(name='Group Count')

            # Group by 'Group ID' and calculate unique stories
            unique_stories = df.groupby('Group ID').agg(lambda x: x.iloc[0]).reset_index()
            unique_stories_with_counts = unique_stories.merge(group_counts, on='Group ID')

            # Debugging outputs
            st.write(f"Number of unique stories: {len(unique_stories_with_counts)}")  # Debugging output
            st.write(unique_stories_with_counts.head())  # Show the first few rows for verification

            # Sort unique stories by group count
            unique_stories_sorted = unique_stories_with_counts.sort_values(by='Group Count',
                                                                           ascending=False).reset_index(drop=True)
            # Update session state
            st.session_state.unique_stories = unique_stories_sorted

            # Time tracking
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
            """Reset the configuration step and related session state variables."""
            st.session_state.config_step = False
            st.session_state.sentiment_opinion = None
            st.session_state.random_sample = None
            st.session_state.similarity_threshold = None
            st.session_state.sentiment_instruction = None
            st.session_state.df_traditional = st.session_state.full_dataset.copy()
            st.session_state.counter = 0
            st.session_state.pop('unique_stories', None)


        # Add reset button
        if st.button("Reset Configuration"):
            reset_config()
            st.rerun()  # Rerun the script to reflect the reset state