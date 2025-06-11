import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest, f_regression
import io
from scipy import stats
from itertools import combinations


def setup_page():
    # Set page config
    st.set_page_config(page_title="Calorie Expenditure Prediction", layout="wide")
    
    # Apply dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Calorie Expenditure Prediction Analysis")
    st.markdown("""<hr style="border: 1px solid #333;">""", unsafe_allow_html=True)

# Load Data
def load_data():
    try:
        df = pd.read_parquet('calorie_expenditure.parquet', engine='fastparquet')
        return df
    except OSError:
        # Try with pyarrow engine if fastparquet fails
        df = pd.read_parquet('calorie_expenditure.parquet', engine='pyarrow')

    
    return clean_data(df)

def clean_data(df):
    """Clean and preprocess the raw data"""
    # Remove any duplicate rows
    df = df.drop_duplicates()
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert data types if needed
    numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'] 
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    return df

def show_data_overview(df):
    st.header(":blue[Dataset Overview]")
    
    # Show descriptive statistics for the dataset
    st.subheader(":green[Sample Data]")
    st.write("Here are the first few rows of the dataset to give you an overview of the data:")
    st.dataframe(df.head(), use_container_width=True)
    
    # Display dataset dimensions with descriptions
    st.subheader(":green[Dataset Size]")
    st.write("The dimensions of the dataset are:")
    shape_df = pd.DataFrame([
        {'Metric': 'Number of Rows', 'Value': df.shape[0]},
        {'Metric': 'Number of Columns', 'Value': df.shape[1]}
    ])
    st.dataframe(shape_df.set_index('Metric').T, use_container_width=True)
    
    # Show detailed column information with descriptions
    st.subheader(":green[Column Details]") 
    st.write("Information about each column in the dataset:")
    info_df = pd.DataFrame([
        {'Metric': 'Number of Valid Values'} | {col: df[col].count() for col in df.columns},
        {'Metric': 'Data Type'} | {col: df[col].dtype for col in df.columns}
    ])
    st.dataframe(info_df.set_index('Metric'), use_container_width=True)



def perform_eda(df):
    st.header(":blue[Exploratory Data Analysis]")
    
    # Numerical columns distribution
    st.subheader(":green[Numerical Features Distribution]")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Remove ID column if it exists
    if 'id' in num_cols:
        num_cols = num_cols.drop('id')

    plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot"])
    
    # Create two columns for side by side charts
    for i in range(0, len(num_cols), 2):
        col1, col2 = st.columns(2)
        
        # First chart
        with col1:
            with st.container():
                if plot_type == "Histogram":
                    mean = df[num_cols[i]].mean()
                    std = df[num_cols[i]].std()
                    fig1 = px.histogram(df, x=num_cols[i], 
                                      template="plotly_dark",
                                      title=f"Distribution of {num_cols[i]} (Mean: {mean:.2f}, Std: {std:.2f})",
                                      labels={num_cols[i]: num_cols[i]},
                                      color_discrete_sequence=['#1f77b4'])
                    fig1.update_layout(showlegend=True,
                                     xaxis_title=num_cols[i],
                                     yaxis_title="Count")
                else:
                    q1 = df[num_cols[i]].quantile(0.25)
                    q3 = df[num_cols[i]].quantile(0.75)
                    median = df[num_cols[i]].median()
                    fig1 = px.box(df, x=num_cols[i], 
                                 template="plotly_dark",
                                 title=f"Box Plot of {num_cols[i]} (Q1: {q1:.2f}, Median: {median:.2f}, Q3: {q3:.2f})",
                                 labels={num_cols[i]: num_cols[i]},
                                 color_discrete_sequence=['#1f77b4'])
                    fig1.update_layout(showlegend=True,
                                     xaxis_title=num_cols[i])
                with st.container(border=True):
                    st.plotly_chart(fig1)
        
        # Second chart (if available)
        with col2:
            if i + 1 < len(num_cols):  # Check if there's a second column
                with st.container():
                    if plot_type == "Histogram":
                        mean = df[num_cols[i+1]].mean()
                        std = df[num_cols[i+1]].std()
                        fig2 = px.histogram(df, x=num_cols[i+1], 
                                          template="plotly_dark",
                                          title=f"Distribution of {num_cols[i+1]} (Mean: {mean:.2f}, Std: {std:.2f})",
                                          labels={num_cols[i+1]: num_cols[i+1]},
                                          color_discrete_sequence=['#2ca02c'])
                        fig2.update_layout(showlegend=True,
                                         xaxis_title=num_cols[i+1],
                                         yaxis_title="Count")
                    else:
                        q1 = df[num_cols[i+1]].quantile(0.25)
                        q3 = df[num_cols[i+1]].quantile(0.75)
                        median = df[num_cols[i+1]].median()
                        fig2 = px.box(df, x=num_cols[i+1], 
                                     template="plotly_dark",
                                     title=f"Box Plot of {num_cols[i+1]} (Q1: {q1:.2f}, Median: {median:.2f}, Q3: {q3:.2f})",
                                     labels={num_cols[i+1]: num_cols[i+1]},
                                     color_discrete_sequence=['#2ca02c'])
                        fig2.update_layout(showlegend=True,
                                         xaxis_title=num_cols[i+1])
                    with st.container(border=True):
                        st.plotly_chart(fig2)



def perform_statistical_analysis(df):
    st.header(":blue[Statistical Analysis]")

    # Get only numeric columns for correlation and hypothesis testing
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Correlation Analysis
    st.subheader(":green[Correlation Analysis]")
    with st.container(border=True):
        corr = numeric_df.corr()
        
        # Heatmap showing strength and direction of relationships between variables
        fig1 = px.imshow(corr, 
                        template="plotly_dark",
                        title="Correlation Heatmap - Shows strength of relationships (-1 to +1)",
                        color_continuous_scale="RdBu",
                        width=1000,
                        height=800,
                        labels={"x": "Features", 
                               "y": "Features", 
                               "color": "Correlation",
                               "text": "Correlation Value"})
        fig1.update_traces(text=corr.round(2), texttemplate="%{text}")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Bar chart showing correlation coefficients for each pair
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'Pair': f"{corr.columns[i]} vs {corr.columns[j]}", 
                    'Correlation': corr.iloc[i,j]
                })
        corr_df = pd.DataFrame(corr_pairs)
        
        # Sort by absolute correlation value
        corr_df = corr_df.reindex(corr_df.Correlation.abs().sort_values(ascending=False).index)
        
        fig2 = px.bar(corr_df, 
                     x='Pair', 
                     y='Correlation',
                     title="Pairwise Correlations - Sorted by Strength",
                     template="plotly_dark",
                     width=800,
                     height=500,
                     color='Correlation',
                     color_continuous_scale=['red', 'lightgray', 'blue'],
                     text='Correlation')
        fig2.update_layout(xaxis_tickangle=45)
        fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Normality Testing
    st.subheader(":green[Normality Testing]")
    st.write("""
    The Shapiro-Wilk test checks if data follows a normal distribution:
    - Null hypothesis (H0): Data is normally distributed
    - Alternative hypothesis (H1): Data is not normally distributed
    - If p-value > 0.05: Data likely follows normal distribution
    - If p-value ≤ 0.05: Data likely does not follow normal distribution
    """)
    
    with st.container(border=True):
        
        results = []
        for col in numeric_df.columns:
            stat, p = stats.shapiro(numeric_df[col])
            results.append({
                'Variable': col,
                'Test Statistic': round(stat, 4),
                'p-value': round(p, 3),
                'Normal Distribution?': 'Yes (p>0.05)' if p > 0.05 else 'No (p≤0.05)'
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.T, use_container_width=True)

    # Cohort Analysis by Sex
    st.subheader(":green[Cohort Analysis by Sex]")
    st.write("""
    Mann-Whitney U test compares distributions between male and female groups:
    - Null hypothesis (H0): No significant difference between groups
    - Alternative hypothesis (H1): Significant difference exists between groups
    - If p-value > 0.05: No significant difference
    - If p-value ≤ 0.05: Significant difference exists
    """)
    
    with st.container(border=True):
        cohort_results = []
        for col in numeric_df.columns:
            male_data = df[df['Sex'] == 'male'][col]
            female_data = df[df['Sex'] == 'female'][col]
            
            # Skip if either group has no data
            if len(male_data) == 0 or len(female_data) == 0:
                continue
                
            # Calculate medians for each group
            male_median = male_data.median()
            female_median = female_data.median()
            
            # Perform Mann-Whitney U test
            stat, p = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
            
            cohort_results.append({
                'Variable': col,
                'Male Median': round(male_median, 2),
                'Female Median': round(female_median, 2),
                'Difference': round(male_median - female_median, 2),
                'p-value': round(p, 3),
                'Significant Difference?': 'Yes (p≤0.05)' if p <= 0.05 else 'No (p>0.05)'
            })
        
        cohort_df = pd.DataFrame(cohort_results)
        st.dataframe(cohort_df.T, use_container_width=True)

        
def preprocess_data(df, sample_size=10000, page="Data Preprocessing"):

    if page == "Data Preprocessing":
        st.header(":blue[Data Preprocessing]")

    # Reduce dataset to 1000 records
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    st.session_state['page'] = page

    # Split features and target
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    # Create preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Feature Selection
    X_processed = preprocessor.fit_transform(X)
    
    selector = SelectKBest(score_func=f_regression, k=5)
    X_selected = selector.fit_transform(X_processed, y)
    
    # Only display results if on preprocessing page
    if st.session_state.get('page') == "Data Preprocessing":
        st.write("Splitting data into features (X) and target variable (Calories)")
        st.write("Features (X):")
        st.dataframe(X.head())
        st.write("Target (y):")
        cols = st.columns([1,4])
        with cols[0]:
            st.dataframe(y.head())
        
        st.code("Creating preprocessing pipeline for numeric and categorical features")
        st.code(f"Numeric features: {numeric_features.tolist()}")
        st.code(f"Categorical features: {categorical_features.tolist()}")
        
        st.write("Numeric features will be standardized using StandardScaler")
        st.write("Categorical features will be one-hot encoded, dropping first category")
        
        st.markdown('---')

        st.header(":blue[Feature Selection]")
        st.write("Applying preprocessing transformations to features")
        processed_df = pd.DataFrame(X_processed)
        st.write("Processed features after scaling and encoding:")
        st.dataframe(processed_df.head())
        
        st.write("Selecting top 5 most important features using f_regression")
        selected_df = pd.DataFrame(X_selected)
        st.write("Final selected features:")
        st.dataframe(selected_df.head())
        
        selected_features = selector.get_support()
        st.code(f"Selected Features: {X.columns[selected_features].tolist()}")
    
    return X_selected, y, preprocessor, selector




def rmsle(y_true, y_pred):
    # Add small constant to avoid log(0)
    y_true = np.clip(y_true, 1e-15, None)
    y_pred = np.clip(y_pred, 1e-15, None)
    return np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))


def build_models(X_selected, y):
    st.header(":blue[Model Building]")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    
    results = {}
    best_score = float('inf')
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        st.subheader(f":green[Training {name}]")
        
        # Split training data into batches of 10
        n_batches = 10
        n_samples = X_train.shape[0]
        batch_size = n_samples // n_batches
        if n_samples % n_batches != 0:
            batch_size += 1  # Round up to ensure all samples are covered
        
        best_batch_model = None
        best_batch_rmsle = float('inf')
        best_batch_r2 = -float('inf')
        
        # Create columns for batch results
        cols = st.columns(5)  # Display 5 batches per row
        
        with st.container(border=True):
            for i in [0,1]:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Train model on batch
                batch_model = type(model)()  # Create new instance of same model type
                batch_model.fit(X_batch, y_batch)
                
                # Evaluate on test set
                y_pred = batch_model.predict(X_test)
                batch_rmsle = rmsle(y_test, y_pred)
                batch_r2 = r2_score(y_test, y_pred)
                
                # Display results in columns
                with cols[i % 5].container(border=True):
                    st.markdown(f"**Batch {i+1}/{n_batches}**")
                    st.markdown(f"- RMSLE: {batch_rmsle:.4f}")
                    st.markdown(f"- R² Score: {batch_r2:.4f}")
                
                # Update best batch model if better
                if batch_rmsle < best_batch_rmsle and batch_r2 > best_batch_r2:
                    best_batch_model = batch_model
                    best_batch_rmsle = batch_rmsle
                    best_batch_r2 = batch_r2
            
            # Use best batch model for final predictions
            y_pred = best_batch_model.predict(X_test)
            rmsle_score = rmsle(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            with st.container(border=True):
                st.markdown(f"#### Final {name} Results")
                st.markdown(f"- Best RMSLE: {rmsle_score:.4f}")
                st.markdown(f"- Best R² Score: {r2:.4f}")
                
                results[name] = {'RMSLE': rmsle_score, 'R2': r2}
                
                # Track best model based on both RMSLE and R2
                if rmsle_score < best_score:
                    best_score = rmsle_score
                    best_model = best_batch_model
                    best_model_name = name
                    st.markdown("🌟 **New overall best model!**")
    
    # Display results
    st.markdown('---')
    st.header(":blue[Model Performance Comparison]")


    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Plot results
    fig = go.Figure(data=[
        go.Bar(name='RMSLE', x=list(results.keys()), y=[r['RMSLE'] for r in results.values()]),
        go.Bar(name='R2', x=list(results.keys()), y=[r['R2'] for r in results.values()])
    ])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)
    
    # Display best model info
    st.subheader(":green[Best Performing Model]")

    with st.container(border=True):
        st.markdown(f"**Best model:** {best_model_name}")
        st.markdown(f"**RMSLE Score:** {best_score:.4f}")
        st.markdown(f"**R² Score:** {results[best_model_name]['R2']:.4f}")
    
    # Save best model in session state
    st.session_state['best_model'] = best_model
    st.session_state['best_model_name'] = best_model_name
    
    return models



def make_predictions(preprocessor, selector, models):
    st.header(":blue[Prediction on New Data]")
    
    uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")
    
    if uploaded_file is not None:
        pred_df = pd.read_csv(uploaded_file)
        
        X_new = preprocessor.transform(pred_df)
        X_new_selected = selector.transform(X_new)
        
        best_model = st.session_state['best_model']
        predictions = best_model.predict(X_new_selected)
        
        pred_df['Predicted_Calories'] = predictions

        pred_df = pred_df[['id', 'Predicted_Calories']].rename(columns={'Predicted_Calories': 'Calories'})
        
        st.write("Predictions:")
        col1, col2 = st.columns([1,4])
        with col1:
            st.dataframe(pred_df, use_container_width=True)
        with col2:
            st.write("Summary Statistics:")
            st.write(f"Number of Predictions: {len(pred_df)}")
            st.write(f"Average Predicted Calories: {pred_df['Calories'].mean():.2f}")
            st.write(f"Min Predicted Calories: {pred_df['Calories'].min():.2f}")
            st.write(f"Max Predicted Calories: {pred_df['Calories'].max():.2f}")
        
        # Download option
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )



def main():
    setup_page()
    with st.spinner('Loading data...'):
        df = load_data()
    
    page = st.sidebar.radio("Modules", 
        ["Data Overview", 
         "EDA", 
         "Statistical Analysis",
         "Data Preprocessing",
         "Model Building",
         "Prediction"
         ])
    
    if page == "Data Overview":
        with st.spinner('Generating data overview...'):
            show_data_overview(df)
    elif page == "EDA":
        with st.spinner('Performing exploratory data analysis...'):
            perform_eda(df)
            clean_data(df)
    elif page == "Statistical Analysis":
        with st.spinner('Running statistical analysis...'):
            perform_statistical_analysis(df)
            if 'statistical_analysis' not in st.session_state:
                st.session_state['statistical_analysis'] = True
    elif page == "Data Preprocessing":
        if 'statistical_analysis' not in st.session_state:
            st.error("Please complete Statistical Analysis first")
        else:
            with st.spinner('Preprocessing data...'):
                X_selected, y, preprocessor, selector = preprocess_data(df, sample_size=10000000, page="Data Preprocessing")
                st.session_state['X_selected'] = X_selected
                st.session_state['y'] = y
                st.session_state['preprocessor'] = preprocessor
                st.session_state['selector'] = selector
    elif page == "Model Building":
        if 'preprocessor' not in st.session_state or 'selector' not in st.session_state:
            st.error("Please complete Data Preprocessing first")
        else:
            with st.spinner('Building and training models...'):
                models = build_models(st.session_state['X_selected'], st.session_state['y'])
                st.session_state['models'] = models
                
    elif page == "Prediction":
        if not all(key in st.session_state for key in ['preprocessor', 'selector', 'models']):
            st.error("Please complete Data Preprocessing and Model Building first")
        else:
            with st.spinner('Making predictions...'):
                # Get the best model based on R2 score
                make_predictions(st.session_state['preprocessor'], st.session_state['selector'], st.session_state['best_model'])

if __name__ == "__main__":
    main()
