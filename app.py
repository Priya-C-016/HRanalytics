import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Advanced HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("📊 Advanced HR Analytics Dashboard")
st.markdown(
    """
    Analyze employee performance and HR metrics with interactive analytics, AI-powered predictions, and actionable insights. 
    You can also upload your own dataset to explore and visualize custom data.
    """
)

# Function to fetch data from API
@st.cache_data
def fetch_data():
    url = "https://jsonplaceholder.typicode.com/users"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to fetch data from the API.")
        return pd.DataFrame()

# Function to clean and format the address
def extract_address(row):
    address = row.get('address', {})
    street = address.get('street', '')
    suite = address.get('suite', '')
    city = address.get('city', '')
    zipcode = address.get('zipcode', '')
    return f"{street} {suite}, {city} {zipcode}"

def extract_company(row):
    company = row.get('company', {})
    bs = company.get('bs', '')
    catchPhrase = company.get('catchPhrase', '')
    name = company.get('name', '')
    return f"{bs} {catchPhrase} {name}"

# Preloaded Data
preloaded_df = fetch_data()
preloaded_df['address'] = preloaded_df.apply(extract_address, axis=1)
preloaded_df['company'] = preloaded_df.apply(extract_company, axis=1)

# Sidebar Data Options
st.sidebar.header("📤 Data Options")
data_source = st.sidebar.radio(
    "Select Data Source:",
    options=["Preloaded Data", "Upload Your Own Data"],
    index=0
)

custom_df = pd.DataFrame()  # Define custom_df to avoid NameError

if data_source == "Upload Your Own Data":

    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV format)", 
        type=["csv"]
    )
    if uploaded_file:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success("Custom data loaded successfully!")
            st.markdown("### 🔗 Data Preview")
            st.dataframe(custom_df.head())

             # Detect relevant columns dynamically
            employee_column = next((col for col in custom_df.columns if 'employee' in col.lower() or 'id' in col.lower()), None)
            salary_column = next((col for col in custom_df.columns if 'salary' in col.lower()), None)
            experience_column = next((col for col in custom_df.columns if 'experience' in col.lower() or 'length_of_service' in col.lower()), None)
            rating_column = next((col for col in custom_df.columns if 'rating' in col.lower() or 'performance' in col.lower()), None)
            
            # Calculate and display KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
             # Total Employees
            if employee_column:
                total_employees = len(custom_df[employee_column].dropna())
                kpi1.metric("Total Employees", total_employees)
            else:
                kpi1.metric("Total Employees", "N/A")
            
            # Average Salary
            if salary_column:
                avg_salary = custom_df[salary_column].mean()
                kpi2.metric("Avg Salary", f"${avg_salary:,.0f}")
            else:
                kpi2.metric("Avg Salary", "N/A")
            
            # Average Experience
            if experience_column:
                avg_experience = custom_df[experience_column].mean()
                kpi3.metric("Avg Experience (Years)", f"{avg_experience:.1f}")
            else:
                kpi3.metric("Avg Experience (Years)", "N/A")
            
            # Average Rating
            if rating_column:
                avg_rating = custom_df[rating_column].mean()
                kpi4.metric("Avg Rating", f"{avg_rating:.1f}")
            else:
                kpi4.metric("Avg Rating", "N/A")
        except Exception as e:
            st.error(f"Error loading data")
else:
    custom_df = preloaded_df.copy()
# Add Department-wise Gender Distribution
if 'department' in custom_df.columns and 'gender' in custom_df.columns:
    st.subheader("👥 Department-wise Gender Distribution")
    dept_gender_dist = px.histogram(
        custom_df,
        x='department',
        color='gender',
        title="Department-wise Gender Distribution",
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(dept_gender_dist)

# Add Age-wise Gender Distribution
if 'age' in custom_df.columns and 'gender' in custom_df.columns:
    st.subheader("👥 Age-wise Gender Distribution")
    age_gender_dist = px.histogram(
        custom_df,
        x='age',
        color='gender',
        title="Age-wise Gender Distribution",
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(age_gender_dist)
if not custom_df.empty:
    # Check if the dataset contains gender and employee ID columns
    gender_column = next((col for col in custom_df.columns if 'gender' in col.lower() or 'sex' in col.lower()), None)
    employee_id_column = next((col for col in custom_df.columns if 'employee' in col.lower() or 'id' in col.lower()), None)
    
    if gender_column and employee_id_column:
        # Calculate gender counts
        gender_counts = custom_df[gender_column].value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']  # Rename columns for clarity
        
        st.subheader("👥 Gender Distribution")
        
        # Display total counts of each gender
        st.write("### Gender Counts")
        for _, row in gender_counts.iterrows():
            gender_label = "Female" if row['gender'].lower().startswith('f') else "Male" if row['gender'].lower().startswith('m') else row['gender']
            st.write(f"{gender_label}: {row['count']}")
        
        # Visualize gender distribution
        fig = px.pie(
            gender_counts,
            names='gender',
            values='count',
            title="Gender Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig)
    if data_source=="Upload Your Own Data":    
        st.subheader("📊 Visualization Options")
    
    # Get a list of numerical columns
        numeric_columns = custom_df.select_dtypes(include=['int', 'float']).columns.tolist()
    
        if numeric_columns:
        # Allow user to select x-axis and y-axis columns
            x_axis = st.selectbox("Select X-axis", options=numeric_columns, key="x_axis")
            y_axis = st.selectbox("Select Y-axis", options=numeric_columns, key="y_axis")
        
        # Chart Type Selection
            chart_type = st.selectbox(
               "Select Chart Type",
               options=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"]
               )
        
        # Generate Visualization
            if st.button("Generate Chart"):
                if x_axis and y_axis:
                    if chart_type == "Scatter Plot":
                        fig = px.scatter(custom_df, x=x_axis, y=y_axis, title=f"{chart_type} of {y_axis} vs {x_axis}")
                        st.plotly_chart(fig)
                    elif chart_type == "Line Chart":
                        fig = px.line(custom_df, x=x_axis, y=y_axis, title=f"{chart_type} of {y_axis} vs {x_axis}")
                        st.plotly_chart(fig)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(custom_df, x=x_axis, y=y_axis, title=f"{chart_type} of {y_axis} vs {x_axis}")
                        st.plotly_chart(fig)
                    elif chart_type == "Histogram":
                        fig = px.histogram(custom_df, x=x_axis, nbins=30, title=f"Histogram of {x_axis}")
                        st.plotly_chart(fig)
                else:
                    st.warning("Please select both X-axis and Y-axis columns.")
        else:
            st.warning("Required columns (gender or employee ID) not found in the dataset.")

    
if custom_df.empty:
    st.error("No data available to analyze. Please upload a valid dataset.")
else:
    # Add synthetic columns for preloaded data
    if data_source == "Preloaded Data":
        custom_df['department'] = np.random.choice(['HR', 'Sales', 'IT', 'Finance', 'Operations'], size=len(custom_df))
        custom_df['gender'] = np.random.choice(['Male', 'Female'], size=len(custom_df))
        custom_df['age'] = np.random.randint(25, 60, size=len(custom_df))
        custom_df['length_of_service'] = np.random.randint(1, 20, size=len(custom_df))
        custom_df['no_of_trainings'] = np.random.randint(1, 10, size=len(custom_df))
        custom_df['previous_year_rating'] = np.random.randint(1, 6, size=len(custom_df))
        custom_df['salary'] = np.random.randint(30000, 150000, size=len(custom_df))
        custom_df['promoted'] = np.random.choice([0, 1], size=len(custom_df), p=[0.7, 0.3])  # Ensure diversity

    # Sidebar Filters
        st.sidebar.header("🔍 Filters")
        department_filter = st.sidebar.multiselect(
        "Select Department(s):",
        options=custom_df['department'].unique(),
        default=custom_df['department'].unique()
        )
        gender_filter = st.sidebar.radio(
        "Select Gender:",
        options=["All", "Male", "Female"],
        index=0
        )
        min_age, max_age = st.sidebar.slider(
        "Select Age Range:",
        min_value=int(custom_df['age'].min()),
        max_value=int(custom_df['age'].max()),
        value=(int(custom_df['age'].min()), int(custom_df['age'].max()))
        )

    # Apply Filters
        filtered_df = custom_df.copy()
        if department_filter:
           filtered_df = filtered_df[filtered_df['department'].isin(department_filter)]
        if gender_filter != "All":
           filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
        filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

    # Tabs for Navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Visualizations", "AI Predictions", "Heatmap", "Download Data"
        ])

    # Tab 1: Overview
        with tab1:
            st.subheader("📊 Key Performance Indicators (KPIs)")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Employees", len(filtered_df))
            kpi2.metric("Avg Salary", f"${filtered_df['salary'].mean():,.0f}")
            kpi3.metric("Promotion Rate", f"{(filtered_df['promoted'].mean() * 100):.1f}%")
            kpi4.metric("Avg Service (yrs)", f"{filtered_df['length_of_service'].mean():.1f}")
            st.markdown("### 🔗 Data Preview")
            st.dataframe(filtered_df.head())

    # Tab 2: Visualizations
        with tab2:
            st.subheader("📈 Visual Insights")
            if 'gender' in filtered_df.columns:
                gender_dist = px.pie(
                    filtered_df,
                    names='gender',
                    title="Gender Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(gender_dist)

            if 'department' in filtered_df.columns:
                dept_count = px.bar(
                    filtered_df,
                    x='department',
                    title="Employee Count by Department",
                    color='department',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(dept_count)

            if 'age' in filtered_df.columns and 'department' in filtered_df.columns:
                st.subheader("Age Distribution by Department")
                age_dept_chart = px.histogram(filtered_df, x='age', color='department', barmode='group', title="Age Distribution by Department")
                st.plotly_chart(age_dept_chart)

            if 'salary' in filtered_df.columns:
                salary_fig = px.violin(filtered_df, x='department', y='salary', box=True, title="Salary Distribution by Department",
                                        color='department')
                st.plotly_chart(salary_fig)

            if 'promoted' in filtered_df.columns and 'department' in filtered_df.columns:
                promo_by_dept = px.histogram(
                    filtered_df.groupby('department')['promoted'].mean().reset_index(),
                x='department',
                    y='promoted',
                    title="Promotion Rate by Department",
                    color='department'
                )
            st.plotly_chart(promo_by_dept)    
    # Tab 3: AI Predictions
        with tab3:
            st.subheader("🤖 AI-Powered Promotion Prediction")
            required_columns = ['age', 'length_of_service', 'previous_year_rating', 'no_of_trainings', 'promoted']
            if all(col in filtered_df.columns for col in required_columns):
                unique_classes=filtered_df['promoted'].nunique()
                if unique_classes > 1:
                    try:
                        X = filtered_df[required_columns[:-1]]
                        y = filtered_df['promoted']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        model = LogisticRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.markdown(f"**Model Accuracy:** {accuracy_score(y_test, y_pred) * 100:.2f}%")
  
                        age = st.number_input("Age", min_value=20, max_value=60, value=30)
                        service = st.number_input("Years of Service", min_value=1, max_value=20, value=5)
                        rating = st.slider("Previous Year Rating", 1, 5, 3)
                        trainings = st.number_input("Number of Trainings", min_value=1, max_value=10, value=2)
                        pred_input = [[age, service, rating, trainings]]
                        prediction = model.predict(pred_input)
                        st.success("The employee is likely to be promoted!" if prediction[0] else "The employee is not likely to be promoted.")
                    except Exception as e:
                         st.error(" ")
                else:
                    st.warning("The 'promoted' column contains only one unique value. AI predictions cannot be generated.")
            else:
                st.warning("AI predictions require specific columns in the dataset.")

    # Tab 4: Heatmap
        with tab4:
            st.subheader("📊 Correlation Heatmap")
            numeric_cols = filtered_df.select_dtypes(include=['int', 'float']).columns
            if len(numeric_cols) > 1:
               corr = filtered_df[numeric_cols].corr()
               fig, ax = plt.subplots(figsize=(8, 6))
               sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
               st.pyplot(fig)

    # Tab 5: Data Export
        with tab5:
            st.subheader("📥 Export Data")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='filtered_employee_data.csv',
            mime='text/csv'
            )
