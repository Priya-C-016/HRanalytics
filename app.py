import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

file_id = '1PYjw9lKCby0Kuj1d_VGFPYRIHFSBX7FM'


def download_file_from_google_drive():
    URL = f'https://drive.google.com/uc?id={file_id}'
    session = requests.Session()
    response = session.get(URL)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        st.error("Failed to download the file.")
        return None


file_data = download_file_from_google_drive()

if file_data:
    df = pd.read_csv(file_data)

    # Remove rows with NaN values in 'department' and 'education' columns for filtering
    df = df.dropna(subset=['department', 'education'])

    st.title("Employee Performance Analysis and HR Dashboard")
    st.subheader("Data Overview")
    st.write(df.head())

    st.sidebar.header("Filter Options")

    # Filter out 'NaN' from departments
    departments = ['All'] + list(df['department'].dropna().unique())
    selected_dept = st.sidebar.selectbox('Select Department', departments)

    # Filter out 'NaN' from education levels
    education_levels = df['education'].dropna().unique()
    selected_education = st.sidebar.selectbox('Select Education Level', education_levels)

    # Filter data based on selected options
    if selected_dept == 'All':
        filtered_df = df[df['education'] == selected_education]
    else:
        filtered_df = df[(df['department'] == selected_dept) & (df['education'] == selected_education)]

    st.subheader(f"Filtered Data - Department: {selected_dept}, Education: {selected_education}")
    st.write(filtered_df)

    st.subheader("Visualizations")

    # Gender Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='gender', ax=ax, palette='Set2')
    ax.set_title('Gender Distribution')
    st.pyplot(fig)

    # Employee Count by Department
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='department', ax=ax, palette='muted')
    ax.set_title('Employee Count by Department')
    st.pyplot(fig)

    # Employee Count by Region
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='region', ax=ax, palette='dark')
    ax.set_title('Employee Count by Region')
    st.pyplot(fig)

    # Average Number of Trainings by Department
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=filtered_df, x='department', y='no_of_trainings', ax=ax, palette='coolwarm')
    ax.set_title('Average Number of Trainings by Department')
    st.pyplot(fig)

    # Length of Service by Gender
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='gender', y='length_of_service', ax=ax, palette='pastel')
    ax.set_title('Length of Service by Gender')
    st.pyplot(fig)

    # Performance Rating vs Age
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='age', y='previous_year_rating', hue='gender', palette='Set1', ax=ax)
    ax.set_title('Performance Rating vs Age')
    st.pyplot(fig)

    # Line Chart: Performance Rating Trend Over Time (if there's a time column)
    if 'date' in df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        performance_over_time = filtered_df.groupby('date')['previous_year_rating'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=performance_over_time, x='date', y='previous_year_rating', ax=ax, marker='o', color='blue')
        ax.set_title('Performance Rating Trend Over Time')
        st.pyplot(fig)

    # Line Chart: Average Number of Trainings Over Time (grouped by date)
    if 'date' in df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        trainings_over_time = filtered_df.groupby('date')['no_of_trainings'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=trainings_over_time, x='date', y='no_of_trainings', ax=ax, marker='x', color='green')
        ax.set_title('Average Number of Trainings Over Time')
        st.pyplot(fig)

    st.subheader("Key Statistics")
    st.write(filtered_df.describe())

    st.subheader("Conclusion")
    st.write(
        "This dashboard provides insights into employee performance and HR metrics. You can filter data by department, education level, and other factors to focus on specific groups.")
else:
    st.error("Error downloading the file.")
