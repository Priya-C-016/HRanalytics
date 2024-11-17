import streamlit as st
import pandas as pd
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

# Download the CSV data
file_id = '1PYjw9lKCby0Kuj1d_VGFPYRIHFSBX7FM'
download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
output = 'employee_performance.csv'
gdown.download(download_url, output, quiet=False)

# Load the CSV data
df = pd.read_csv(output)

# Streamlit app title
st.title("Employee Performance Analysis and HR Dashboard")

# Show the first few rows of the data
st.subheader("Data Overview")
st.write(df.head())

# Sidebar for filtering options
st.sidebar.header("Filter Options")

# Filter by Department
departments = df['department'].unique()
selected_dept = st.sidebar.selectbox('Select Department', departments)

# Filter by Education Level
education_levels = df['education'].unique()
selected_education = st.sidebar.selectbox('Select Education Level', education_levels)

# Filter data based on selection
filtered_df = df[(df['department'] == selected_dept) & (df['education'] == selected_education)]

# Display filtered data
st.subheader(f"Filtered Data - {selected_dept} with Education {selected_education}")
st.write(filtered_df)

# Plotting graphs
st.subheader("Visualizations")

# 1. Distribution of Gender
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='gender', ax=ax, palette='Set2')
ax.set_title('Gender Distribution')
st.pyplot(fig)

# 2. Employee Count by Department
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='department', ax=ax, palette='muted')
ax.set_title('Employee Count by Department')
st.pyplot(fig)

# 3. Employee Count by Region
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='region', ax=ax, palette='dark')
ax.set_title('Employee Count by Region')
st.pyplot(fig)

# 4. Average Number of Trainings by Department
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='department', y='no_of_trainings', ax=ax, palette='coolwarm')
ax.set_title('Average Number of Trainings by Department')
st.pyplot(fig)

# 5. Length of Service by Gender
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='gender', y='length_of_service', ax=ax, palette='pastel')
ax.set_title('Length of Service by Gender')
st.pyplot(fig)

# Performance vs Age (assuming performance can be correlated with age or other features)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='previous_year_rating', hue='gender', palette='Set1', ax=ax)
ax.set_title('Performance Rating vs Age')
st.pyplot(fig)

# Show some stats
st.subheader("Key Statistics")
st.write(df.describe())

# Employee Search
st.sidebar.header("Search Employees")
employee_id = st.sidebar.text_input("Enter Employee ID")
if employee_id:
    employee_data = df[df['employee_id'].astype(str).str.contains(employee_id)]
    if not employee_data.empty:
        st.write(employee_data)
    else:
        st.write(f"No employees found with ID '{employee_id}'.")

# Conclusion message
st.subheader("Conclusion")
st.write("This dashboard provides insights into employee performance and HR metrics. You can filter data by department, education level, and other factors to focus on specific groups.")
