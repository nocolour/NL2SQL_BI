import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import sqlite3
import datetime
from io import StringIO
import re
import time

# Configure the app
st.set_page_config(
    page_title="Python BI System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'connection_string' not in st.session_state:
    st.session_state.connection_string = ""
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'available_tables' not in st.session_state:
    st.session_state.available_tables = []

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Define functions for data operations
def connect_to_database(connection_type, params):
    """Connect to a database using the provided parameters"""
    try:
        if connection_type == "SQLite":
            conn = sqlite3.connect(params['database_path'])
            st.session_state.db_connection = conn
            return conn
        elif connection_type == "MySQL":
            connection_string = f"mysql+pymysql://{params['username']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
            engine = create_engine(connection_string)
            conn = engine.connect()
            st.session_state.db_connection = engine
            return engine
        elif connection_type == "PostgreSQL":
            connection_string = f"postgresql://{params['username']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
            engine = create_engine(connection_string)
            conn = engine.connect()
            st.session_state.db_connection = engine
            return engine
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def load_sample_data():
    """Load sample datasets for demo purposes"""
    dataset_name = st.selectbox(
        "Choose a sample dataset:", 
        ["Sales Data", "Customer Data", "Product Data", "Marketing Campaign Data"]
    )
    
    if dataset_name == "Sales Data":
        # Generate sample sales data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = {
            'date': np.random.choice(dates, size=1000),
            'product_id': np.random.randint(1, 21, size=1000),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Office'], size=1000),
            'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=1000),
            'sales_amount': np.random.uniform(10, 1000, size=1000).round(2),
            'quantity': np.random.randint(1, 11, size=1000),
            'customer_id': np.random.randint(1, 101, size=1000)
        }
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['sales_cost'] = df['sales_amount'] * 0.6  # 60% cost
        df['profit'] = df['sales_amount'] - df['sales_cost']
        
        return df
    
    elif dataset_name == "Customer Data":
        # Generate sample customer data
        np.random.seed(43)
        data = {
            'customer_id': list(range(1, 101)),
            'customer_name': [f"Customer {i}" for i in range(1, 101)],
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], size=100),
            'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan'], size=100),
            'registration_date': pd.date_range(start='2020-01-01', periods=100, freq='3D'),
            'age': np.random.randint(18, 80, size=100),
            'total_spent': np.random.uniform(100, 10000, size=100).round(2),
            'orders_count': np.random.randint(1, 30, size=100)
        }
        
        df = pd.DataFrame(data)
        df['average_order_value'] = (df['total_spent'] / df['orders_count']).round(2)
        
        return df
    
    elif dataset_name == "Product Data":
        # Generate sample product data
        np.random.seed(44)
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Office']
        subcategories = {
            'Electronics': ['Phones', 'Laptops', 'Tablets', 'TVs', 'Audio'],
            'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
            'Food': ['Dairy', 'Meat', 'Produce', 'Bakery', 'Beverages'],
            'Home': ['Furniture', 'Decor', 'Kitchen', 'Bathroom', 'Bedding'],
            'Office': ['Stationery', 'Furniture', 'Equipment', 'Storage', 'Supplies']
        }
        
        data = {
            'product_id': list(range(1, 101)),
            'product_name': [f"Product {i}" for i in range(1, 101)],
            'category': np.random.choice(categories, size=100)
        }
        
        df = pd.DataFrame(data)
        
        # Add subcategory based on category
        df['subcategory'] = df.apply(lambda x: np.random.choice(subcategories[x['category']]), axis=1)
        
        # Add other product attributes
        df['price'] = np.random.uniform(10, 1000, size=100).round(2)
        df['cost'] = (df['price'] * np.random.uniform(0.3, 0.7, size=100)).round(2)
        df['launch_date'] = pd.date_range(start='2019-01-01', periods=100, freq='7D')
        df['stock_quantity'] = np.random.randint(0, 500, size=100)
        df['rating'] = np.random.uniform(1, 5, size=100).round(1)
        
        return df
    
    elif dataset_name == "Marketing Campaign Data":
        # Generate sample marketing campaign data
        np.random.seed(45)
        campaign_types = ['Email', 'Social Media', 'Display Ads', 'Search Ads', 'Content Marketing']
        
        data = {
            'campaign_id': list(range(1, 51)),
            'campaign_name': [f"Campaign {i}" for i in range(1, 51)],
            'campaign_type': np.random.choice(campaign_types, size=50),
            'start_date': pd.date_range(start='2023-01-01', periods=50, freq='7D'),
            'budget': np.random.uniform(1000, 50000, size=50).round(2),
            'impressions': np.random.randint(5000, 500000, size=50),
            'clicks': [],
            'conversions': [],
            'revenue': []
        }
        
        # Calculate related metrics with realistic conversion rates
        for i in range(50):
            clicks = int(data['impressions'][i] * np.random.uniform(0.01, 0.1))  # 1-10% CTR
            data['clicks'].append(clicks)
            
            conversions = int(clicks * np.random.uniform(0.02, 0.15))  # 2-15% conversion rate
            data['conversions'].append(conversions)
            
            avg_order = np.random.uniform(50, 200)
            data['revenue'].append(round(conversions * avg_order, 2))
        
        df = pd.DataFrame(data)
        
        # Add end date (7-30 days after start)
        df['duration_days'] = np.random.randint(7, 31, size=50)
        df['end_date'] = df['start_date'] + pd.to_timedelta(df['duration_days'], unit='D')
        
        # Calculate derived metrics
        df['ctr'] = (df['clicks'] / df['impressions'] * 100).round(2)
        df['conversion_rate'] = (df['conversions'] / df['clicks'] * 100).round(2)
        df['cpa'] = (df['budget'] / df['conversions']).round(2)
        df['roi'] = ((df['revenue'] - df['budget']) / df['budget'] * 100).round(2)
        
        return df

def nl_to_sql(question, available_tables, sample_data=None):
    """
    A simplified natural language to SQL converter
    For a real-world application, you would use an LLM or a more sophisticated NLP approach
    """
    # This is a very simple pattern matching approach, not for production use
    question = question.lower()
    
    # If we have sample data, we can use it to match column names
    if sample_data is not None:
        columns = list(sample_data.columns)
    else:
        columns = []
    
    # Simple pattern matching for common questions
    if "average" in question or "mean" in question:
        # Look for a column to average
        for col in columns:
            if col.lower() in question:
                return f"SELECT AVG({col}) FROM {available_tables[0]}"
    
    elif "sum" in question or "total" in question:
        # Look for a column to sum
        for col in columns:
            if col.lower() in question:
                return f"SELECT SUM({col}) FROM {available_tables[0]}"
    
    elif "count" in question:
        return f"SELECT COUNT(*) FROM {available_tables[0]}"
    
    elif "maximum" in question or "highest" in question or "max" in question:
        # Look for a column to find maximum
        for col in columns:
            if col.lower() in question:
                return f"SELECT MAX({col}) FROM {available_tables[0]}"
    
    elif "minimum" in question or "lowest" in question or "min" in question:
        # Look for a column to find minimum
        for col in columns:
            if col.lower() in question:
                return f"SELECT MIN({col}) FROM {available_tables[0]}"
    
    elif "group by" in question:
        group_col = None
        agg_col = None
        
        # Look for column to group by
        for col in columns:
            if col.lower() in question:
                if group_col is None:
                    group_col = col
                else:
                    agg_col = col
                    break
        
        if group_col and agg_col:
            return f"SELECT {group_col}, SUM({agg_col}) FROM {available_tables[0]} GROUP BY {group_col}"
    
    # Default to a simple select all
    return f"SELECT * FROM {available_tables[0]} LIMIT 100"

def execute_sql(conn, query):
    """Execute a SQL query and return the results as a DataFrame"""
    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def generate_simple_dashboard(df):
    """Generate a simple dashboard from the provided DataFrame"""
    st.markdown('<p class="sub-header">Dashboard Overview</p>', unsafe_allow_html=True)
    
    if df is None or df.empty:
        st.warning("No data available for dashboard generation.")
        return
    
    # Layout with 4 metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Determine which metrics to display based on the columns available
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Metric 1: Record count
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metric 2: Sum of first numeric column (if available)
    with col2:
        if numeric_cols:
            sum_val = df[numeric_cols[0]].sum()
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{sum_val:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Sum of {numeric_cols[0]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">No numeric data available</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Metric 3: Average of second numeric column (if available)
    with col3:
        if len(numeric_cols) > 1:
            avg_val = df[numeric_cols[1]].mean()
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_val:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Average {numeric_cols[1]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">No additional metrics</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Metric 4: Date range (if date column available)
    with col4:
        if date_cols:
            min_date = df[date_cols[0]].min().strftime('%Y-%m-%d')
            max_date = df[date_cols[0]].max().strftime('%Y-%m-%d')
            date_range = f"{min_date} to {max_date}"
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{date_range}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Date Range</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">No date data available</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two column layout for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<p class="sub-header">Data Distribution</p>', unsafe_allow_html=True)
        
        if numeric_cols:
            # Let user select a column to visualize
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            
            # Create distribution chart with Plotly
            fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution of {selected_col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution chart.")
    
    with chart_col2:
        st.markdown('<p class="sub-header">Data Trends</p>', unsafe_allow_html=True)
        
        # Check if we have date and numeric columns for a line chart
        if date_cols and numeric_cols:
            date_col = st.selectbox("Select date column:", date_cols)
            value_col = st.selectbox("Select value column:", numeric_cols)
            
            # Aggregate by date for the line chart
            df_agg = df.groupby(pd.Grouper(key=date_col, freq='M')).agg({value_col: 'sum'}).reset_index()
            
            # Create line chart with Plotly
            fig = px.line(df_agg, x=date_col, y=value_col, 
                          title=f"Trend of {value_col} over time",
                          markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 2:
            # If no date column, create a scatter plot of two numeric columns
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            fig = px.scatter(df, x=x_col, y=y_col, 
                             title=f"Relationship between {x_col} and {y_col}",
                             trendline="ols")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for trend visualization.")
    
    # Add correlation heatmap if we have multiple numeric columns
    if len(numeric_cols) > 2:
        st.markdown('<p class="sub-header">Correlation Analysis</p>', unsafe_allow_html=True)
        
        # Calculate correlation matrix
        corr = df[numeric_cols].corr()
        
        # Create heatmap with Plotly
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                         title="Correlation Heatmap of Numeric Variables")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add categorical data breakdown if available
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.markdown('<p class="sub-header">Categorical Breakdown</p>', unsafe_allow_html=True)
        
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            selected_cat = st.selectbox("Select category:", categorical_cols)
            
            # Count values in category
            cat_counts = df[selected_cat].value_counts().reset_index()
            cat_counts.columns = [selected_cat, 'Count']
            
            # Create bar chart
            fig = px.bar(cat_counts, x=selected_cat, y='Count', 
                         title=f"Count by {selected_cat}",
                         color='Count', text_auto=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with cat_col2:
            if numeric_cols:
                value_col = st.selectbox("Select value column for breakdown:", numeric_cols)
                
                # Group by category and sum the value
                agg_df = df.groupby(selected_cat)[value_col].sum().reset_index()
                
                # Create pie chart
                fig = px.pie(agg_df, values=value_col, names=selected_cat, 
                             title=f"{value_col} by {selected_cat}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for category breakdown.")

def generate_default_queries(df):
    """Generate a list of default SQL queries based on the DataFrame structure"""
    if df is None or df.empty:
        return []
    
    queries = []
    table_name = "current_data"
    columns = df.columns.tolist()
    
    # Get numeric and date columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Basic queries
    queries.append(f"SELECT * FROM {table_name} LIMIT 10")
    queries.append(f"SELECT COUNT(*) FROM {table_name}")
    
    # Aggregate queries if we have numeric columns
    if numeric_cols:
        for col in numeric_cols[:2]:  # Only use first two numeric columns to avoid too many queries
            queries.append(f"SELECT SUM({col}) FROM {table_name}")
            queries.append(f"SELECT AVG({col}) FROM {table_name}")
            queries.append(f"SELECT MIN({col}), MAX({col}) FROM {table_name}")
    
    # Date-based queries
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        queries.append(f"SELECT {date_col}, SUM({num_col}) FROM {table_name} GROUP BY {date_col}")
        
        # If it's a proper date, add more complex date queries
        queries.append(f"SELECT STRFTIME('%Y-%m', {date_col}) as month, SUM({num_col}) FROM {table_name} GROUP BY month")
    
    # Categorical breakdowns
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        queries.append(f"SELECT {cat_col}, SUM({num_col}) FROM {table_name} GROUP BY {cat_col} ORDER BY SUM({num_col}) DESC")
        queries.append(f"SELECT {cat_col}, COUNT(*) FROM {table_name} GROUP BY {cat_col}")
    
    return queries

def create_time_series_forecast(df, date_col, value_col, periods=30):
    """Create a simple time series forecast using moving averages"""
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return None, None
    
    # Ensure the date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df = df.sort_values(by=date_col)
    
    # Resample to daily frequency and fill missing values
    df_daily = df.set_index(date_col)[[value_col]].resample('D').mean()
    df_daily = df_daily.fillna(method='ffill')
    
    # Calculate the rolling mean (simple forecasting method)
    window_size = min(30, len(df_daily) // 3)  # Use 1/3 of the data points as window size
    if window_size < 2:
        window_size = 2
    
    df_daily['rolling_mean'] = df_daily[value_col].rolling(window=window_size).mean()
    
    # Create forecast dates
    last_date = df_daily.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    
    # Create forecast values (using the last rolling mean)
    last_mean = df_daily['rolling_mean'].dropna().iloc[-1]
    
    # Add some random variation for a more realistic forecast
    np.random.seed(42)
    forecast_values = last_mean + (np.random.normal(0, 0.1, size=periods) * last_mean)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        date_col: future_dates,
        value_col: forecast_values
    })
    
    # Prepare historical data for plotting
    historical_df = df_daily.reset_index()
    historical_df = historical_df.rename(columns={historical_df.index.name: date_col})
    
    return historical_df, forecast_df

def perform_regression_analysis(df, x_col, y_col):
    """Perform a simple regression analysis between two variables"""
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    
    # Prepare data
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Prepare results
    results = {
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'rmse': rmse,
        'equation': f"{y_col} = {model.coef_[0]:.4f} * {x_col} + {model.intercept_:.4f}",
        'x': X.flatten(),
        'y': y,
        'y_pred': y_pred
    }
    
    return results

def perform_cluster_analysis(df, columns, n_clusters=3):
    """Perform basic cluster analysis on the data"""
    if df is None or df.empty:
        return None
    
    # Ensure all selected columns exist in the dataframe
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        return None
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = df[valid_columns].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster centers in original scale
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    # Create cluster centers dataframe
    centers_df = pd.DataFrame(centers_original, columns=valid_columns)
    centers_df['cluster'] = range(n_clusters)
    
    # Calculate basic cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        **{col: ['mean', 'min', 'max'] for col in valid_columns},
        'cluster': 'count'
    })
    
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={'cluster_count': 'count'})
    
    return {
        'df_with_clusters': df,
        'cluster_centers': centers_df,
        'cluster_stats': cluster_stats,
        'columns_used': valid_columns
    }

# Main application function
def main():
    # Sidebar navigation
    st.sidebar.markdown("# Python BI System")
    
    # Navigation options
    nav_options = [
        "Home", 
        "Data Connection", 
        "Data Exploration", 
        "Dashboard", 
        "Natural Language Queries",
        "Advanced Analytics"
    ]
    
    page = st.sidebar.radio("Navigation", nav_options)
    
    # Home page
    if page == "Home":
        st.markdown('<h1 class="main-header">Python Business Intelligence System</h1>', unsafe_allow_html=True)
        
        st.markdown('''
        Welcome to the Python BI System! This application demonstrates key capabilities 
        of a business intelligence platform built with Python and Streamlit.
        
        ### Features:
        - **Data Connections**: Connect to databases or load sample data
        - **Data Exploration**: Explore and analyze your data
        - **Interactive Dashboards**: Visualize key metrics and trends
        - **Natural Language Queries**: Ask questions in plain English
        - **Advanced Analytics**: Perform statistical analysis and forecasting
        
        ### Getting Started:
        1. Navigate to the "Data Connection" page to load data
        2. Explore your data with built-in analysis tools
        3. Generate dashboards automatically
        4. Use natural language to query your data
        5. Apply advanced analytics for deeper insights
        ''')
        
        # Display a sample visualization
        st.markdown('<p class="sub-header">Sample Visualization</p>', unsafe_allow_html=True)
        
        # Create a sample chart
        np.random.seed(0)
        
        # Sample time series data
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        values = np.cumsum(np.random.randn(90)) + 100
        
        # Create a dataframe
        df = pd.DataFrame({'date': dates, 'value': values})
        
        # Create a line chart with Plotly
        fig = px.line(df, x='date', y='value', title='Sample Time Series Data')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("👈 Use the navigation panel on the left to explore the application!")
    
    # Data Connection page
    elif page == "Data Connection":
        st.markdown('<h1 class="main-header">Data Connection</h1>', unsafe_allow_html=True)
        
        connection_tabs = st.tabs(["Sample Data", "Database Connection", "File Upload"])
        
        # Sample Data Tab
        with connection_tabs[0]:
            st.markdown('<p class="sub-header">Load Sample Data</p>', unsafe_allow_html=True)
            st.markdown('''
            Load pre-generated sample datasets to explore the BI system capabilities.
            These datasets simulate real-world business data like sales, customers, 
            products, and marketing campaigns.
            ''')
            
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.available_tables = ["sample_data"]
                    
                    st.success(f"Sample data loaded successfully! {len(df)} rows and {len(df.columns)} columns.")
                    st.dataframe(df.head())
        
        # Database Connection Tab
        with connection_tabs[1]:
            st.markdown('<p class="sub-header">Connect to Database</p>', unsafe_allow_html=True)
            
            st.markdown('''
            Connect to various database types to analyze your data.
            Supported database types include SQLite, MySQL, and PostgreSQL.
            ''')
            
            connection_type = st.selectbox(
                "Select database type:", 
                ["SQLite", "MySQL", "PostgreSQL"]
            )
            
            if connection_type == "SQLite":
                database_path = st.text_input("Database file path:", "example.db")
                
                params = {
                    'database_path': database_path
                }
            else:
                col1, col2 = st.columns(2)
                with col1:
                    host = st.text_input("Host:", "localhost")
                    username = st.text_input("Username:", "root")
                    database = st.text_input("Database name:")
                
                with col2:
                    port = st.text_input("Port:", "3306" if connection_type == "MySQL" else "5432")
                    password = st.text_input("Password:", type="password")
                
                params = {
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'database': database
                }
            
            if st.button("Connect to Database"):
                if connection_type == "SQLite" and not params['database_path']:
                    st.error("Please provide a database file path.")
                elif connection_type != "SQLite" and not params['database']:
                    st.error("Please provide a database name.")
                else:
                    with st.spinner("Connecting to database..."):
                        conn = connect_to_database(connection_type, params)
                        
                        if conn:
                            st.success("Connected to database successfully!")
                            
                            # Get list of tables
                            if connection_type == "SQLite":
                                tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                            else:
                                tables_query = "SHOW TABLES;"
                            
                            try:
                                if connection_type == "SQLite":
                                    cursor = conn.cursor()
                                    cursor.execute(tables_query)
                                    tables = [row[0] for row in cursor.fetchall()]
                                    cursor.close()
                                else:
                                    tables = pd.read_sql(tables_query, conn)
                                    tables = tables.iloc[:, 0].tolist()
                                
                                st.session_state.available_tables = tables
                                
                                if tables:
                                    st.write("Available tables:")
                                    for table in tables:
                                        st.write(f"- {table}")
                                    
                                    selected_table = st.selectbox("Select a table to load:", tables)
                                    
                                    if st.button("Load Table Data"):
                                        with st.spinner("Loading table data..."):
                                            df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 1000", conn)
                                            st.session_state.data = df
                                            
                                            st.success(f"Data loaded successfully! {len(df)} rows and {len(df.columns)} columns.")
                                            st.dataframe(df.head())
                                else:
                                    st.warning("No tables found in the database.")
                            except Exception as e:
                                st.error(f"Error loading tables: {str(e)}")
                        else:
                            st.error("Failed to connect to database.")
        
        # File Upload Tab
        with connection_tabs[2]:
            st.markdown('<p class="sub-header">Upload Data File</p>', unsafe_allow_html=True)
            
            st.markdown('''
            Upload a CSV or Excel file to analyze your data.
            ''')
            
            uploaded_file = st.file_uploader("Choose a file:", type=["csv", "xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"File uploaded successfully! {len(df)} rows and {len(df.columns)} columns.")
                    
                    # Date column detection and conversion
                    date_columns = st.multiselect(
                        "Select any date columns to convert:",
                        df.columns.tolist()
                    )
                    
                    if date_columns:
                        for col in date_columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    st.session_state.data = df
                    st.session_state.available_tables = ["uploaded_data"]
                    
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    # Data Exploration page
    elif page == "Data Exploration":
        st.markdown('<h1 class="main-header">Data Exploration</h1>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("No data loaded. Please go to Data Connection page to load data.")
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.available_tables = ["sample_data"]
                    st.success("Sample data loaded successfully!")
                    st.experimental_rerun()
        else:
            df = st.session_state.data
            
            # Key metrics
            st.markdown('<p class="sub-header">Data Overview</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            # Create tabs for different exploration views
            explore_tabs = st.tabs(["Data Viewer", "Statistics", "Visualizations", "SQL Query"])
            
            # Data Viewer Tab
            with explore_tabs[0]:
                st.markdown('<p class="sub-header">Data Preview</p>', unsafe_allow_html=True)
                
                # Filters
                st.markdown("#### Apply Filters")
                
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Add column filters
                filtered_df = df.copy()
                
                # Add categorical filters
                if categorical_cols:
                    st.markdown("**Categorical Filters:**")
                    cat_filters = st.multiselect("Select categorical columns to filter:", categorical_cols)
                    
                    for col in cat_filters:
                        unique_values = df[col].unique()
                        if len(unique_values) <= 30:  # Only show if not too many values
                            selected_values = st.multiselect(
                                f"Select values for {col}:",
                                unique_values.tolist(),
                                default=unique_values.tolist()
                            )
                            if selected_values:
                                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                
                # Add numeric filters
                if numeric_cols:
                    st.markdown("**Numeric Filters:**")
                    num_filters = st.multiselect("Select numeric columns to filter:", numeric_cols)
                    
                    for col in num_filters:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        
                        range_values = st.slider(
                            f"Range for {col}:",
                            min_val, max_val,
                            (min_val, max_val)
                        )
                        
                        filtered_df = filtered_df[(filtered_df[col] >= range_values[0]) & 
                                                 (filtered_df[col] <= range_values[1])]
                
                # Show the filtered dataframe
                st.markdown(f"**Filtered Data: {len(filtered_df):,} rows**")
                st.dataframe(filtered_df)
            
            # Statistics Tab
            with explore_tabs[1]:
                st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)
                
                # Summary statistics for numeric columns
                if not df.select_dtypes(include=['number']).empty:
                    st.markdown("#### Numeric Statistics")
                    st.dataframe(df.describe())
                
                # Summary for categorical columns
                if not df.select_dtypes(include=['object', 'category']).empty:
                    st.markdown("#### Categorical Distributions")
                    
                    # Let user select a categorical column
                    cat_col = st.selectbox(
                        "Select a categorical column:",
                        df.select_dtypes(include=['object', 'category']).columns.tolist()
                    )
                    
                    # Show value counts
                    value_counts = df[cat_col].value_counts().reset_index()
                    value_counts.columns = [cat_col, 'Count']
                    
                    # Add percentage
                    value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
                    
                    st.dataframe(value_counts)
                    
                    # Visualize as bar chart
                    fig = px.bar(value_counts, x=cat_col, y='Count', text='Percentage',
                                 title=f"Distribution of {cat_col}")
                    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                if len(df.select_dtypes(include=['number']).columns) > 1:
                    st.markdown("#### Correlation Analysis")
                    
                    # Calculate correlations
                    corr = df.select_dtypes(include=['number']).corr()
                    
                    # Show correlation matrix
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                    title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Missing value analysis
                st.markdown("#### Missing Values Analysis")
                
                # Calculate missing values
                missing_values = df.isnull().sum().reset_index()
                missing_values.columns = ['Column', 'Missing Values']
                missing_values['Percentage'] = (missing_values['Missing Values'] / len(df) * 100).round(2)
                
                # Filter to only show columns with missing values
                missing_values = missing_values[missing_values['Missing Values'] > 0]
                
                if not missing_values.empty:
                    st.dataframe(missing_values)
                    
                    # Visualize missing values
                    fig = px.bar(missing_values, x='Column', y='Percentage',
                                 title="Percentage of Missing Values by Column")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values found in the dataset!")
            
            # Visualizations Tab
            with explore_tabs[2]:
                st.markdown('<p class="sub-header">Data Visualization</p>', unsafe_allow_html=True)
                
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Histogram", "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Pie Chart", "Heatmap"]
                )
                
                # Histogram
                if viz_type == "Histogram":
                    if not df.select_dtypes(include=['number']).empty:
                        num_col = st.selectbox("Select column for histogram:", 
                                              df.select_dtypes(include=['number']).columns.tolist())
                        
                        bins = st.slider("Number of bins:", 5, 100, 20)
                        
                        fig = px.histogram(df, x=num_col, nbins=bins, 
                                          title=f"Histogram of {num_col}", 
                                          marginal="box")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns available for histogram.")
                
                # Scatter Plot
                elif viz_type == "Scatter Plot":
                    if len(df.select_dtypes(include=['number']).columns) > 1:
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        x_col = st.selectbox("Select X-axis column:", num_cols)
                        y_col = st.selectbox("Select Y-axis column:", [col for col in num_cols if col != x_col])
                        
                        color_option = st.checkbox("Add color dimension")
                        if color_option:
                            color_col = st.selectbox("Select color column:", df.columns.tolist())
                            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                            title=f"{y_col} vs {x_col} by {color_col}")
                        else:
                            fig = px.scatter(df, x=x_col, y=y_col,
                                            title=f"{y_col} vs {x_col}")
                            
                        add_trendline = st.checkbox("Add trendline")
                        if add_trendline:
                            fig.update_traces(mode='markers')
                            fig.update_layout(shapes=[
                                dict(
                                    type='line',
                                    xref='x', yref='y',
                                    x0=df[x_col].min(), y0=df[y_col].min(),
                                    x1=df[x_col].max(), y1=df[y_col].max(),
                                    line=dict(color='red', width=2, dash='dash')
                                )
                            ])
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least two numeric columns for scatter plot.")
                
                # Bar Chart
                elif viz_type == "Bar Chart":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_cols and df.select_dtypes(include=['number']).columns.tolist():
                        x_col = st.selectbox("Select categorical column (X-axis):", categorical_cols)
                        y_col = st.selectbox("Select numeric column (Y-axis):", 
                                            df.select_dtypes(include=['number']).columns.tolist())
                        
                        # Limit to top N categories if there are too many
                        top_n = st.slider("Show top N categories:", 5, 50, 10)
                        
                        # Group by the categorical column and calculate the sum
                        grouped_df = df.groupby(x_col)[y_col].sum().reset_index()
                        grouped_df = grouped_df.sort_values(by=y_col, ascending=False).head(top_n)
                        
                        fig = px.bar(grouped_df, x=x_col, y=y_col,
                                     title=f"Sum of {y_col} by {x_col} (Top {top_n})")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need categorical and numeric columns for bar chart.")
                
                # Line Chart
                elif viz_type == "Line Chart":
                    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                    
                    if date_cols and df.select_dtypes(include=['number']).columns.tolist():
                        x_col = st.selectbox("Select date column (X-axis):", date_cols)
                        y_col = st.selectbox("Select numeric column (Y-axis):", 
                                            df.select_dtypes(include=['number']).columns.tolist())
                        
                        # Group by option
                        group_option = st.checkbox("Group by category")
                        
                        if group_option and categorical_cols:
                            group_col = st.selectbox("Select grouping column:", categorical_cols)
                            
                            # Resample by day, week, month, quarter, or year
                            freq = st.selectbox("Select time frequency:", 
                                               ["Day", "Week", "Month", "Quarter", "Year"])
                            
                            freq_map = {
                                "Day": "D",
                                "Week": "W",
                                "Month": "M",
                                "Quarter": "Q",
                                "Year": "Y"
                            }
                            
                            # Group by date and category
                            df_copy = df.copy()
                            df_copy[x_col] = pd.to_datetime(df_copy[x_col])
                            df_copy = df_copy.set_index(x_col)
                            
                            grouped = df_copy.groupby([pd.Grouper(freq=freq_map[freq]), group_col])[y_col].sum().reset_index()
                            
                            fig = px.line(grouped, x=x_col, y=y_col, color=group_col,
                                         title=f"{y_col} over time by {group_col} ({freq})")
                        else:
                            # Simple time series without grouping
                            df_copy = df.copy()
                            df_copy[x_col] = pd.to_datetime(df_copy[x_col])
                            df_copy = df_copy.sort_values(by=x_col)
                            
                            fig = px.line(df_copy, x=x_col, y=y_col,
                                         title=f"{y_col} over time")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need date and numeric columns for line chart.")
                
                # Box Plot
                elif viz_type == "Box Plot":
                    if df.select_dtypes(include=['number']).columns.tolist():
                        y_col = st.selectbox("Select numeric column (Y-axis):", 
                                            df.select_dtypes(include=['number']).columns.tolist())
                        
                        group_option = st.checkbox("Group by category")
                        
                        if group_option and categorical_cols:
                            x_col = st.selectbox("Select categorical column (X-axis):", categorical_cols)
                            
                            # Limit to top N categories if there are too many
                            if df[x_col].nunique() > 10:
                                top_n = st.slider("Show top N categories:", 3, 20, 5)
                                top_categories = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(top_n).index.tolist()
                                filtered_df = df[df[x_col].isin(top_categories)]
                            else:
                                filtered_df = df
                            
                            fig = px.box(filtered_df, x=x_col, y=y_col,
                                        title=f"Distribution of {y_col} by {x_col}")
                        else:
                            fig = px.box(df, y=y_col,
                                        title=f"Distribution of {y_col}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need numeric columns for box plot.")
                
                # Pie Chart
                elif viz_type == "Pie Chart":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_cols:
                        names_col = st.selectbox("Select categorical column (names):", categorical_cols)
                        
                        if df.select_dtypes(include=['number']).columns.tolist():
                            values_col = st.selectbox("Select numeric column (values):", 
                                                    df.select_dtypes(include=['number']).columns.tolist())
                            
                            # Group by the categorical column and sum the values
                            grouped_df = df.groupby(names_col)[values_col].sum().reset_index()
                            
                            # Limit to top N categories if there are too many
                            if len(grouped_df) > 10:
                                top_n = st.slider("Show top N categories:", 3, 15, 5)
                                grouped_df = grouped_df.sort_values(by=values_col, ascending=False).head(top_n)
                                grouped_df.loc[len(grouped_df)] = ["Others", df[values_col].sum() - grouped_df[values_col].sum()]
                            
                            fig = px.pie(grouped_df, names=names_col, values=values_col,
                                        title=f"Distribution of {values_col} by {names_col}")
                        else:
                            # Count-based pie chart
                            counts = df[names_col].value_counts().reset_index()
                            counts.columns = [names_col, 'count']
                            
                            # Limit to top N categories if there are too many
                            if len(counts) > 10:
                                top_n = st.slider("Show top N categories:", 3, 15, 5)
                                top_counts = counts.head(top_n)
                                top_counts.loc[len(top_counts)] = ["Others", counts['count'].sum() - top_counts['count'].sum()]
                                counts = top_counts
                            
                            fig = px.pie(counts, names=names_col, values='count',
                                        title=f"Distribution of records by {names_col}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need categorical columns for pie chart.")
                
                # Heatmap
                elif viz_type == "Heatmap":
                    if len(df.select_dtypes(include=['number']).columns) > 1:
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        # Allow selecting columns for heatmap
                        selected_cols = st.multiselect(
                            "Select numeric columns for heatmap:",
                            numeric_cols,
                            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                        )
                        
                        if selected_cols:
                            # Calculate correlation matrix
                            corr = df[selected_cols].corr()
                            
                            fig = px.imshow(corr, text_auto=True, aspect="auto",
                                           title="Correlation Heatmap",
                                           color_continuous_scale="RdBu_r")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please select at least two columns for the heatmap.")
                    else:
                        st.warning("Need at least two numeric columns for heatmap.")
            
            # SQL Query Tab
            with explore_tabs[3]:
                st.markdown('<p class="sub-header">SQL Query</p>', unsafe_allow_html=True)
                
                st.markdown('''
                Write SQL queries to explore your data. Note that for the sample data or uploaded files,
                the queries are executed against a temporary in-memory SQLite database.
                ''')
                
                # Get default queries based on the dataframe structure
                default_queries = generate_default_queries(df)
                
                query_to_run = st.selectbox(
                    "Select a query template or write your own:",
                    ["Write your own query"] + default_queries
                )
                
                if query_to_run == "Write your own query":
                    sql_query = st.text_area("Enter SQL query:", height=150)
                else:
                    sql_query = query_to_run
                    st.code(sql_query, language="sql")
                
                if st.button("Run Query"):
                    if not sql_query:
                        st.warning("Please enter a SQL query.")
                    else:
                        with st.spinner("Executing query..."):
                            try:
                                # For sample data or uploaded files, create a temporary SQLite database
                                if st.session_state.db_connection is None:
                                    conn = sqlite3.connect(":memory:")
                                    df.to_sql("current_data", conn, if_exists="replace", index=False)
                                else:
                                    conn = st.session_state.db_connection
                                
                                # Execute query and get results
                                result = execute_sql(conn, sql_query)
                                
                                if result is not None:
                                    st.success(f"Query executed successfully! {len(result)} rows returned.")
                                    st.dataframe(result)
                                    
                                    # Add query to history
                                    if sql_query not in st.session_state.query_history:
                                        st.session_state.query_history.append(sql_query)
                                        
                                    # Option to download results
                                    csv = result.to_csv(index=False)
                                    st.download_button(
                                        label="Download Results as CSV",
                                        data=csv,
                                        file_name="query_results.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.error("Query execution failed.")
                            except Exception as e:
                                st.error(f"Error executing query: {str(e)}")
                
                # Show query history
                if st.session_state.query_history:
                    st.markdown("#### Query History")
                    for i, query in enumerate(st.session_state.query_history):
                        with st.expander(f"Query {i+1}"):
                            st.code(query, language="sql")
    
    # Dashboard page
    elif page == "Dashboard":
        st.markdown('<h1 class="main-header">Interactive Dashboard</h1>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("No data loaded. Please go to Data Connection page to load data.")
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.available_tables = ["sample_data"]
                    st.success("Sample data loaded successfully!")
                    st.experimental_rerun()
        else:
            df = st.session_state.data
            
            # Dashboard settings
            st.markdown('<p class="sub-header">Dashboard Settings</p>', unsafe_allow_html=True)
            
            # Get date columns if any
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            # Allow date filtering if date columns exist
            if date_cols:
                date_col = st.selectbox("Select date column for filtering:", date_cols)
                
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                
                date_range = st.date_input(
                    "Select date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = df[(df[date_col].dt.date >= start_date) & 
                                    (df[date_col].dt.date <= end_date)]
                else:
                    filtered_df = df
            else:
                filtered_df = df
            
            # Generate the dashboard
            generate_simple_dashboard(filtered_df)
            
            # Download dashboard
            st.markdown('<p class="sub-header">Export Dashboard</p>', unsafe_allow_html=True)
            
            st.markdown('''
            Export the dashboard as an HTML file to share with others.
            ''')
            
            if st.button("Prepare Dashboard Export"):
                with st.spinner("Preparing dashboard for export..."):
                    time.sleep(1)  # Simulate processing time
                    st.success("Dashboard prepared for export!")
                    
                    st.download_button(
                        label="Download Dashboard HTML",
                        data="<html><body><h1>Dashboard Export</h1><p>This is a placeholder for a dashboard export feature.</p></body></html>",
                        file_name="dashboard_export.html",
                        mime="text/html"
                    )
    
    # Natural Language Queries page
    elif page == "Natural Language Queries":
        st.markdown('<h1 class="main-header">Natural Language Queries</h1>', unsafe_allow_html=True)
        
        st.markdown('''
        Ask questions about your data in plain English, and the system will convert them to SQL
        and return the results. This simplified NL2SQL system demonstrates how AI can make 
        data analysis more accessible to non-technical users.
        
        **Example questions you can ask:**
        - What is the average sales amount?
        - Show me the total revenue by region
        - Count orders by product category
        - What is the maximum price in the product data?
        ''')
        
        if st.session_state.data is None:
            st.warning("No data loaded. Please go to Data Connection page to load data.")
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.available_tables = ["sample_data"]
                    st.success("Sample data loaded successfully!")
                    st.experimental_rerun()
        else:
            df = st.session_state.data
            
            # Natural language query input
            nl_query = st.text_input("Ask a question about your data:", 
                                    placeholder="e.g., What is the average sales amount?")
            
            if st.button("Execute Query"):
                if not nl_query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Processing your question..."):
                        # Convert natural language to SQL
                        sql_query = nl_to_sql(nl_query, st.session_state.available_tables, df)
                        
                        st.markdown('<p class="sub-header">Generated SQL Query</p>', unsafe_allow_html=True)
                        st.code(sql_query, language="sql")
                        
                        try:
                            # For sample data or uploaded files, create a temporary SQLite database
                            if st.session_state.db_connection is None:
                                conn = sqlite3.connect(":memory:")
                                df.to_sql("current_data", conn, if_exists="replace", index=False)
                            else:
                                conn = st.session_state.db_connection
                            
                            # Execute the query
                            result = execute_sql(conn, sql_query)
                            
                            if result is not None:
                                st.markdown('<p class="sub-header">Query Results</p>', unsafe_allow_html=True)
                                st.dataframe(result)
                                
                                # Generate a simple visualization of the results
                                if not result.empty:
                                    st.markdown('<p class="sub-header">Visualization</p>', unsafe_allow_html=True)
                                    
                                    if len(result.columns) == 1:
                                        # Single value result
                                        st.metric("Result", f"{result.iloc[0, 0]:,}")
                                    elif len(result.columns) == 2:
                                        # Two columns - likely a category and a value
                                        if result.dtypes.iloc[1].kind in 'ifc':  # If second column is numeric
                                            fig = px.bar(result, x=result.columns[0], y=result.columns[1],
                                                        title="Query Results")
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.dataframe(result)
                                    else:
                                        # Multiple columns - show a table
                                        st.dataframe(result)
                            else:
                                st.error("Query execution failed.")
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
            
            # Example queries
            with st.expander("View Example Questions"):
                st.markdown('''
                Here are some example questions you can ask about your data:
                
                **For Sales Data:**
                - What is the total sales amount?
                - Show me average profit by product category
                - Which region has the highest sales?
                - Count sales by month
                
                **For Customer Data:**
                - Who are our top 10 customers by total spent?
                - What is the average age of our customers?
                - How many customers are in each segment?
                - What is the distribution of customers by country?
                
                **For Product Data:**
                - What is the average price by category?
                - Which products have the highest profit margin?
                - How many products do we have in each category?
                - What is our total inventory value?
                
                **For Marketing Campaign Data:**
                - Which campaign type has the highest ROI?
                - What is the average conversion rate across campaigns?
                - Show me the total impressions by campaign type
                - Which campaigns had negative ROI?
                ''')
    
    # Advanced Analytics page
    elif page == "Advanced Analytics":
        st.markdown('<h1 class="main-header">Advanced Analytics</h1>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("No data loaded. Please go to Data Connection page to load data.")
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.available_tables = ["sample_data"]
                    st.success("Sample data loaded successfully!")
                    st.experimental_rerun()
        else:
            df = st.session_state.data
            
            # Create tabs for different analysis types
            analysis_tabs = st.tabs(["Time Series Analysis", "Regression Analysis", "Cluster Analysis"])
            
            # Time Series Analysis Tab
            with analysis_tabs[0]:
                st.markdown('<p class="sub-header">Time Series Analysis & Forecasting</p>', unsafe_allow_html=True)
                
                # Get date columns
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                
                if date_cols and df.select_dtypes(include=['number']).columns.tolist():
                    date_col = st.selectbox("Select date column:", date_cols)
                    value_col = st.selectbox("Select value column to forecast:", 
                                            df.select_dtypes(include=['number']).columns.tolist())
                    
                    forecast_periods = st.slider("Forecast periods ahead:", 7, 90, 30)
                    
                    if st.button("Generate Forecast"):
                        with st.spinner("Generating forecast..."):
                            historical_df, forecast_df = create_time_series_forecast(
                                df, date_col, value_col, periods=forecast_periods
                            )
                            
                            if historical_df is not None and forecast_df is not None:
                                # Create the plot
                                fig = go.Figure()
                                
                                # Add historical data
                                fig.add_trace(go.Scatter(
                                    x=historical_df[date_col], 
                                    y=historical_df[value_col],
                                    mode='lines',
                                    name='Historical Data',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast data
                                fig.add_trace(go.Scatter(
                                    x=forecast_df[date_col], 
                                    y=forecast_df[value_col],
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"Time Series Forecast: {value_col}",
                                    xaxis_title=date_col,
                                    yaxis_title=value_col,
                                    hovermode="x unified"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show forecast data
                                st.markdown("#### Forecast Data")
                                st.dataframe(forecast_df)
                                
                                # Add forecast download option
                                csv = forecast_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Forecast Data",
                                    data=csv,
                                    file_name="forecast_data.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Could not generate forecast with the selected columns.")
                else:
                    st.warning("Need both date and numeric columns for time series analysis.")
                    
                    if not date_cols:
                        st.info("Tip: Go to the Data Connection page and specify date columns when loading data.")
            
            # Regression Analysis Tab
            with analysis_tabs[1]:
                st.markdown('<p class="sub-header">Regression Analysis</p>', unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select independent variable (X):", numeric_cols)
                    y_col = st.selectbox("Select dependent variable (Y):", 
                                        [col for col in numeric_cols if col != x_col])
                    
                    if st.button("Run Regression Analysis"):
                        with st.spinner("Analyzing data..."):
                            results = perform_regression_analysis(df, x_col, y_col)
                            
                            if results:
                                # Create scatter plot with regression line
                                fig = go.Figure()
                                
                                # Add scatter points
                                fig.add_trace(go.Scatter(
                                    x=results['x'], 
                                    y=results['y'],
                                    mode='markers',
                                    name='Data Points',
                                    marker=dict(color='blue', opacity=0.6)
                                ))
                                
                                # Add regression line
                                fig.add_trace(go.Scatter(
                                    x=results['x'], 
                                    y=results['y_pred'],
                                    mode='lines',
                                    name='Regression Line',
                                    line=dict(color='red')
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"Regression Analysis: {y_col} vs {x_col}",
                                    xaxis_title=x_col,
                                    yaxis_title=y_col,
                                    hovermode="closest"
                                )
                                
                                # Add regression equation annotation
                                fig.add_annotation(
                                    xref="paper", yref="paper",
                                    x=0.02, y=0.98,
                                    text=f"Equation: {results['equation']}<br>R²: {results['r2']:.4f}<br>RMSE: {results['rmse']:.4f}",
                                    showarrow=False,
                                    font=dict(size=14),
                                    bgcolor="rgba(255, 255, 255, 0.8)",
                                    bordercolor="black",
                                    borderwidth=1
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show regression details
                                st.markdown("#### Regression Statistics")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("R-squared (R²)", f"{results['r2']:.4f}")
                                    st.metric("Slope (Coefficient)", f"{results['coefficient']:.4f}")
                                
                                with col2:
                                    st.metric("RMSE", f"{results['rmse']:.4f}")
                                    st.metric("Intercept", f"{results['intercept']:.4f}")
                                
                                st.markdown(f"**Regression Equation**: {results['equation']}")
                                
                                # Interpretation
                                interpretation = f"""
                                ### Interpretation
                                
                                - The R-squared value of {results['r2']:.4f} indicates that approximately {results['r2']*100:.1f}% of the 
                                  variation in {y_col} can be explained by {x_col}.
                                
                                - The coefficient of {results['coefficient']:.4f} means that for each one-unit increase in {x_col}, 
                                  {y_col} changes by {results['coefficient']:.4f} units.
                                
                                - The intercept of {results['intercept']:.4f} represents the expected value of {y_col} when {x_col} is zero.
                                
                                - The Root Mean Square Error (RMSE) of {results['rmse']:.4f} indicates the average deviation of predictions from actual values.
                                """
                                
                                st.markdown(interpretation)
                            else:
                                st.error("Could not perform regression analysis with the selected columns.")
                else:
                    st.warning("Need at least two numeric columns for regression analysis.")
            
            # Cluster Analysis Tab
            with analysis_tabs[2]:
                st.markdown('<p class="sub-header">Cluster Analysis</p>', unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    selected_cols = st.multiselect(
                        "Select numeric columns for clustering:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
                    )
                    
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    
                    if selected_cols and st.button("Run Cluster Analysis"):
                        with st.spinner("Clustering data..."):
                            cluster_results = perform_cluster_analysis(df, selected_cols, n_clusters)
                            
                            if cluster_results:
                                # Show cluster assignments
                                st.markdown("#### Cluster Assignments")
                                
                                cluster_df = cluster_results['df_with_clusters']
                                st.dataframe(cluster_df[selected_cols + ['cluster']].head(20))
                                
                                # Show cluster statistics
                                st.markdown("#### Cluster Statistics")
                                st.dataframe(cluster_results['cluster_stats'])
                                
                                # Visualize clusters (2D or 3D plot)
                                st.markdown("#### Cluster Visualization")
                                
                                if len(selected_cols) >= 2:
                                    # Create 2D scatter plot for the first two selected columns
                                    x_col = selected_cols[0]
                                    y_col = selected_cols[1]
                                    
                                    fig = px.scatter(
                                        cluster_df, 
                                        x=x_col, 
                                        y=y_col, 
                                        color="cluster",
                                        title=f"Cluster Analysis: {y_col} vs {x_col}"
                                    )
                                    
                                    # Add cluster centers
                                    for i, row in cluster_results['cluster_centers'].iterrows():
                                        fig.add_trace(go.Scatter(
                                            x=[row[x_col]], 
                                            y=[row[y_col]],
                                            mode='markers',
                                            marker=dict(
                                                symbol='star',
                                                size=15,
                                                color=i,
                                                line=dict(color='black', width=2)
                                            ),
                                            name=f"Cluster {i} Center"
                                        ))
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # If we have 3 or more columns, add 3D visualization option
                                    if len(selected_cols) >= 3:
                                        show_3d = st.checkbox("Show 3D visualization")
                                        
                                        if show_3d:
                                            z_col = selected_cols[2]
                                            
                                            fig_3d = px.scatter_3d(
                                                cluster_df, 
                                                x=x_col, 
                                                y=y_col, 
                                                z=z_col,
                                                color="cluster",
                                                title=f"3D Cluster Analysis: {x_col}, {y_col}, {z_col}"
                                            )
                                            
                                            # Add cluster centers
                                            for i, row in cluster_results['cluster_centers'].iterrows():
                                                fig_3d.add_trace(go.Scatter3d(
                                                    x=[row[x_col]], 
                                                    y=[row[y_col]],
                                                    z=[row[z_col]],
                                                    mode='markers',
                                                    marker=dict(
                                                        symbol='diamond',
                                                        size=8,
                                                        color=i,
                                                        line=dict(color='black', width=2)
                                                    ),
                                                    name=f"Cluster {i} Center"
                                                ))
                                            
                                            st.plotly_chart(fig_3d, use_container_width=True)
                                
                                # Interpretation
                                st.markdown("#### Cluster Interpretation")
                                
                                # Calculate key characteristics of each cluster
                                interpretation = "### Key Characteristics by Cluster\n\n"
                                
                                for cluster_id in range(n_clusters):
                                    interpretation += f"**Cluster {cluster_id}**\n"
                                    interpretation += f"- Size: {(cluster_df['cluster'] == cluster_id).sum()} records ({(cluster_df['cluster'] == cluster_id).sum() / len(cluster_df) * 100:.1f}% of data)\n"
                                    
                                    # Get the key features of this cluster (highest/lowest values)
                                    for col in selected_cols:
                                        cluster_mean = cluster_results['cluster_stats'][f"{col}_mean"][cluster_id]
                                        overall_mean = df[col].mean()
                                        
                                        if cluster_mean > overall_mean * 1.2:  # At least 20% higher than overall mean
                                            interpretation += f"- High {col}: {cluster_mean:.2f} (overall avg: {overall_mean:.2f})\n"
                                        elif cluster_mean < overall_mean * 0.8:  # At least 20% lower than overall mean
                                            interpretation += f"- Low {col}: {cluster_mean:.2f} (overall avg: {overall_mean:.2f})\n"
                                    
                                    interpretation += "\n"
                                
                                st.markdown(interpretation)
                                
                                # Allow downloading cluster results
                                csv = cluster_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Clustered Data",
                                    data=csv,
                                    file_name="clustered_data.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Could not perform cluster analysis with the selected columns.")
                    else:
                        st.info("Select at least two numeric columns and click 'Run Cluster Analysis'.")
                else:
                    st.warning("Need at least two numeric columns for cluster analysis.")

# Run the main application
if __name__ == "__main__":
    main()
