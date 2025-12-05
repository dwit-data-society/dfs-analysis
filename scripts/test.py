import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Deerwalk Food System Analytics",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }

    /* Section headers */
    h1, h2, h3 {
        color: #1f1f1f;
        font-weight: 600;
    }

    /* Cards and containers */
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
class Config:
    """Application configuration"""
    DATA_PATH = Path('../data/orders.csv')
    DEFAULT_TOP_N = 10
    ENCODING = 'utf-8-sig'

    # Column names
    COL_ITEM_NAME = 'item_name'
    COL_TOTAL_PRICE = 'total_price'
    COL_STATUS = 'status'
    COL_CREATED_AT = 'created_at'
    COL_UPDATED_AT = 'updated_at'
    COL_MENU_ID = 'menu_id'
    COL_ORDER_ID = 'order_id'

    # Status indicating successful delivery
    DELIVERED_STATUSES = ['delivered', 'completed', 'success']

    # Menu lifecycle thresholds (days)
    NEW_ITEM_DAYS = 30
    ESTABLISHED_ITEM_DAYS = 90


class DataLoader:
    """Handles data loading and preprocessing"""

    @staticmethod
    def clean_csv_line(line: str) -> list[str]:
        """
        Clean and parse a CSV line with embedded semicolons.

        Args:
            line: Raw CSV line

        Returns:
            List of cleaned field values
        """
        # Strip outer quotes and whitespace
        line = line.strip().strip('"')

        # Fix known data quality issues
        line = line.replace('Chicken mo;mo', 'Chicken momo')

        # Split on semicolons
        parts = line.split(';')

        # Clean each part - remove surrounding quotes and double-quote escapes
        cleaned_parts = [p.strip('"').replace('""', '"') for p in parts]

        return cleaned_parts

    @staticmethod
    def load_orders(file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and parse orders CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with orders data, or None if loading fails
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                st.error(f"Data file not found: {file_path}")
                return None

            rows = []
            with open(file_path, 'r', encoding=Config.ENCODING) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        cleaned_parts = DataLoader.clean_csv_line(line)
                        rows.append(cleaned_parts)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

            if len(rows) < 2:
                logger.error("CSV file has insufficient data")
                st.error("CSV file appears to be empty or invalid")
                return None

            # Create DataFrame
            df = pd.DataFrame(rows[1:], columns=rows[0])

            # Validate required columns
            required_cols = [Config.COL_ITEM_NAME, Config.COL_TOTAL_PRICE, Config.COL_STATUS]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None

            # Convert numeric columns
            df[Config.COL_TOTAL_PRICE] = pd.to_numeric(
                df[Config.COL_TOTAL_PRICE],
                errors='coerce'
            )

            # Convert datetime columns
            if Config.COL_CREATED_AT in df.columns:
                df[Config.COL_CREATED_AT] = pd.to_datetime(
                    df[Config.COL_CREATED_AT],
                    errors='coerce'
                )

            if Config.COL_UPDATED_AT in df.columns:
                df[Config.COL_UPDATED_AT] = pd.to_datetime(
                    df[Config.COL_UPDATED_AT],
                    errors='coerce'
                )

            # Remove rows with invalid prices
            invalid_prices = df[Config.COL_TOTAL_PRICE].isna().sum()
            if invalid_prices > 0:
                logger.warning(f"Removed {invalid_prices} rows with invalid prices")
                df = df.dropna(subset=[Config.COL_TOTAL_PRICE])

            logger.info(f"Successfully loaded {len(df)} orders")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            st.error(f"Failed to load data: {str(e)}")
            return None


class DataAnalyzer:
    """Handles data analysis operations"""

    @staticmethod
    def calculate_revenue_per_item(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total revenue per item.

        Args:
            df: Orders DataFrame

        Returns:
            DataFrame with item and total_revenue columns
        """
        revenue = df.groupby(Config.COL_ITEM_NAME)[Config.COL_TOTAL_PRICE].sum().reset_index()
        revenue.columns = ['item', 'total_revenue']
        return revenue

    @staticmethod
    def get_top_n_items(revenue_df: pd.DataFrame, n: int = Config.DEFAULT_TOP_N) -> pd.DataFrame:
        """
        Get top N items by revenue.

        Args:
            revenue_df: Revenue DataFrame
            n: Number of top items to return

        Returns:
            DataFrame with top N items
        """
        return revenue_df.sort_values(by='total_revenue', ascending=False).head(n)

    @staticmethod
    def get_status_counts(df: pd.DataFrame) -> pd.Series:
        """
        Get order status counts.

        Args:
            df: Orders DataFrame

        Returns:
            Series with status counts
        """
        return df[Config.COL_STATUS].value_counts()

    @staticmethod
    def calculate_fulfillment_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Calculate order fulfillment time statistics.

        Args:
            df: Orders DataFrame

        Returns:
            Tuple of (DataFrame with fulfillment times, dict with statistics)
        """
        if Config.COL_CREATED_AT not in df.columns or Config.COL_UPDATED_AT not in df.columns:
            return pd.DataFrame(), {}

        # Calculate fulfillment time
        valid_times = df[
            df[Config.COL_CREATED_AT].notna() &
            df[Config.COL_UPDATED_AT].notna()
        ].copy()

        valid_times['fulfillment_time'] = (
            valid_times[Config.COL_UPDATED_AT] - valid_times[Config.COL_CREATED_AT]
        )

        # Convert to hours
        valid_times['fulfillment_hours'] = valid_times['fulfillment_time'].dt.total_seconds() / 3600

        # Remove negative times (data quality issues)
        valid_times = valid_times[valid_times['fulfillment_hours'] >= 0]

        stats = {
            'avg_hours': valid_times['fulfillment_hours'].mean(),
            'median_hours': valid_times['fulfillment_hours'].median(),
            'min_hours': valid_times['fulfillment_hours'].min(),
            'max_hours': valid_times['fulfillment_hours'].max(),
            'total_orders': len(valid_times)
        }

        return valid_times, stats

    @staticmethod
    def calculate_delivery_success_rate(df: pd.DataFrame) -> dict:
        """
        Calculate delivery success rate.

        Args:
            df: Orders DataFrame

        Returns:
            Dictionary with success metrics
        """
        total_orders = len(df)

        # Check for delivered orders (case-insensitive)
        delivered_orders = df[
            df[Config.COL_STATUS].str.lower().isin(Config.DELIVERED_STATUSES)
        ].shape[0]

        success_rate = (delivered_orders / total_orders * 100) if total_orders > 0 else 0

        return {
            'total_orders': total_orders,
            'delivered_orders': delivered_orders,
            'success_rate': success_rate,
            'failed_orders': total_orders - delivered_orders
        }

    @staticmethod
    def analyze_peak_times(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze peak order times by hour and day.

        Args:
            df: Orders DataFrame

        Returns:
            Tuple of (hourly orders DataFrame, daily orders DataFrame)
        """
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()

        # Extract hour and day
        valid_orders['hour'] = valid_orders[Config.COL_CREATED_AT].dt.hour
        valid_orders['day_of_week'] = valid_orders[Config.COL_CREATED_AT].dt.day_name()
        valid_orders['day_num'] = valid_orders[Config.COL_CREATED_AT].dt.dayofweek

        # Hourly analysis
        hourly = valid_orders.groupby('hour').size().reset_index(name='order_count')

        # Daily analysis (ordered by day of week)
        daily = valid_orders.groupby(['day_num', 'day_of_week']).size().reset_index(name='order_count')
        daily = daily.sort_values('day_num')

        return hourly, daily

    @staticmethod
    def analyze_order_trends(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze order volume trends over time.

        Args:
            df: Orders DataFrame

        Returns:
            DataFrame with daily order trends
        """
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()

        # Group by date
        valid_orders['date'] = valid_orders[Config.COL_CREATED_AT].dt.date

        trends = valid_orders.groupby('date').agg({
            Config.COL_TOTAL_PRICE: ['count', 'sum', 'mean']
        }).reset_index()

        trends.columns = ['date', 'order_count', 'total_revenue', 'avg_order_value']
        trends['date'] = pd.to_datetime(trends['date'])

        return trends

    @staticmethod
    def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze monthly sales trends.

        Args:
            df: Orders DataFrame

        Returns:
            DataFrame with monthly trends
        """
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()

        # Extract year-month
        valid_orders['year_month'] = valid_orders[Config.COL_CREATED_AT].dt.to_period('M')

        monthly = valid_orders.groupby('year_month').agg({
            Config.COL_TOTAL_PRICE: ['count', 'sum', 'mean']
        }).reset_index()

        monthly.columns = ['year_month', 'order_count', 'total_revenue', 'avg_order_value']
        monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

        return monthly

    @staticmethod
    def calculate_revenue_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate month-over-month revenue growth rate.

        Args:
            monthly_df: Monthly trends DataFrame

        Returns:
            DataFrame with growth metrics
        """
        if monthly_df.empty:
            return pd.DataFrame()

        growth_df = monthly_df.copy()

        # Calculate month-over-month changes
        growth_df['revenue_change'] = growth_df['total_revenue'].diff()
        growth_df['revenue_growth_rate'] = (growth_df['total_revenue'].pct_change() * 100)
        growth_df['order_count_change'] = growth_df['order_count'].diff()
        growth_df['order_growth_rate'] = (growth_df['order_count'].pct_change() * 100)

        return growth_df

    @staticmethod
    def analyze_menu_categories(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze menu category performance.

        Args:
            df: Orders DataFrame with menu_id

        Returns:
            DataFrame with category performance metrics
        """
        if Config.COL_MENU_ID not in df.columns:
            return pd.DataFrame()

        category_stats = df.groupby(Config.COL_MENU_ID).agg({
            Config.COL_TOTAL_PRICE: ['sum', 'mean', 'count'],
            Config.COL_ITEM_NAME: 'nunique'
        }).reset_index()

        category_stats.columns = ['menu_id', 'total_revenue', 'avg_order_value', 'order_count', 'unique_items']

        # Calculate revenue per item
        category_stats['revenue_per_item'] = category_stats['total_revenue'] / category_stats['unique_items']

        # Sort by total revenue
        category_stats = category_stats.sort_values('total_revenue', ascending=False)

        return category_stats

    @staticmethod
    def find_cross_selling_opportunities(df: pd.DataFrame, min_support: int = 3) -> pd.DataFrame:
        """
        Find items frequently ordered together.

        Args:
            df: Orders DataFrame with order_id
            min_support: Minimum number of co-occurrences to include

        Returns:
            DataFrame with item pairs and their co-occurrence count
        """
        if Config.COL_ORDER_ID not in df.columns or Config.COL_ITEM_NAME not in df.columns:
            return pd.DataFrame()

        # Group items by order
        orders_items = df.groupby(Config.COL_ORDER_ID)[Config.COL_ITEM_NAME].apply(list).reset_index()

        # Find item pairs
        from itertools import combinations

        pairs = []
        for items in orders_items[Config.COL_ITEM_NAME]:
            if len(items) > 1:
                # Get all unique pairs
                unique_items = list(set(items))
                for item1, item2 in combinations(sorted(unique_items), 2):
                    pairs.append((item1, item2))

        if not pairs:
            return pd.DataFrame()

        # Count pairs
        pairs_df = pd.DataFrame(pairs, columns=['item_1', 'item_2'])
        pair_counts = pairs_df.groupby(['item_1', 'item_2']).size().reset_index(name='co_occurrence_count')

        # Filter by minimum support
        pair_counts = pair_counts[pair_counts['co_occurrence_count'] >= min_support]

        # Calculate lift (how much more likely items are ordered together vs independently)
        total_orders = df[Config.COL_ORDER_ID].nunique()
        item_counts = df.groupby(Config.COL_ITEM_NAME)[Config.COL_ORDER_ID].nunique().to_dict()

        pair_counts['item_1_frequency'] = pair_counts['item_1'].map(item_counts) / total_orders
        pair_counts['item_2_frequency'] = pair_counts['item_2'].map(item_counts) / total_orders
        pair_counts['pair_frequency'] = pair_counts['co_occurrence_count'] / total_orders
        pair_counts['lift'] = pair_counts['pair_frequency'] / (pair_counts['item_1_frequency'] * pair_counts['item_2_frequency'])

        # Sort by co-occurrence count and lift
        pair_counts = pair_counts.sort_values(['co_occurrence_count', 'lift'], ascending=[False, False])

        return pair_counts

    @staticmethod
    def analyze_menu_item_lifecycle(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Analyze menu items by their lifecycle stage (new vs established).

        Args:
            df: Orders DataFrame

        Returns:
            Tuple of (DataFrame with item lifecycle info, dict with summary stats)
        """
        if Config.COL_CREATED_AT not in df.columns or Config.COL_ITEM_NAME not in df.columns:
            return pd.DataFrame(), {}

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()

        # Get first and last order date for each item
        item_dates = valid_orders.groupby(Config.COL_ITEM_NAME)[Config.COL_CREATED_AT].agg(['min', 'max', 'count']).reset_index()
        item_dates.columns = ['item_name', 'first_order', 'last_order', 'total_orders']

        # Calculate days since first order
        latest_date = valid_orders[Config.COL_CREATED_AT].max()
        item_dates['days_on_menu'] = (latest_date - item_dates['first_order']).dt.days
        item_dates['days_since_last_order'] = (latest_date - item_dates['last_order']).dt.days

        # Calculate revenue per item
        item_revenue = valid_orders.groupby(Config.COL_ITEM_NAME)[Config.COL_TOTAL_PRICE].sum().reset_index()
        item_revenue.columns = ['item_name', 'total_revenue']

        item_dates = item_dates.merge(item_revenue, on='item_name')

        # Classify items
        def classify_item(row):
            if row['days_on_menu'] <= Config.NEW_ITEM_DAYS:
                return 'New'
            elif row['days_on_menu'] <= Config.ESTABLISHED_ITEM_DAYS:
                return 'Growing'
            else:
                return 'Established'

        item_dates['lifecycle_stage'] = item_dates.apply(classify_item, axis=1)

        # Add performance metrics
        item_dates['avg_revenue_per_day'] = item_dates['total_revenue'] / item_dates['days_on_menu']
        item_dates['orders_per_day'] = item_dates['total_orders'] / item_dates['days_on_menu']

        # Summary statistics
        summary = {
            'new_items': len(item_dates[item_dates['lifecycle_stage'] == 'New']),
            'growing_items': len(item_dates[item_dates['lifecycle_stage'] == 'Growing']),
            'established_items': len(item_dates[item_dates['lifecycle_stage'] == 'Established']),
            'total_items': len(item_dates)
        }

        return item_dates, summary


class Dashboard:
    """Streamlit dashboard renderer"""

    # Color palette for consistent styling
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#3b82f6'
    }

    @staticmethod
    def render_header():
        """Render dashboard header"""
        st.markdown('<h1 class="main-title">🍜 Deerwalk Food System Analytics</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Comprehensive insights into food ordering and delivery performance</p>', unsafe_allow_html=True)

        # Quick stats banner
        current_date = datetime.now().strftime("%B %d, %Y")
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                        border-radius: 10px; margin-bottom: 2rem;'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                    📅 Dashboard Updated: {current_date}
                </p>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_metrics(df: pd.DataFrame, delivery_stats: dict):
        """Render key metrics"""
        st.markdown("### 📊 Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Orders",
                f"{len(df):,}",
                help="Total number of orders in the system"
            )

        with col2:
            total_revenue = df[Config.COL_TOTAL_PRICE].sum()
            st.metric(
                "Total Revenue",
                f"रु{total_revenue:,.2f}",
                help="Total revenue generated"
            )

        with col3:
            avg_order = df[Config.COL_TOTAL_PRICE].mean()
            st.metric(
                "Average Order Value",
                f"रु{avg_order:.2f}",
                help="Average value per order"
            )

        with col4:
            success_rate = delivery_stats.get('success_rate', 0)
            delta_color = "normal" if success_rate >= 90 else "inverse"
            st.metric(
                "Delivery Success Rate",
                f"{success_rate:.1f}%",
                delta=f"{success_rate - 100:.1f}%" if success_rate < 100 else "Perfect!",
                help="Percentage of successfully delivered orders"
            )

    @staticmethod
    def render_top_items_chart(top_items: pd.DataFrame, n: int):
        """Render top items revenue chart"""
        st.markdown(f"### 🏆 Top {n} Items by Revenue")

        if top_items.empty:
            st.warning("No data available for top items")
            return

        # Create plotly bar chart for better aesthetics
        fig = px.bar(
            top_items,
            x='total_revenue',
            y='item',
            orientation='h',
            labels={'total_revenue': 'Revenue (रु)', 'item': 'Menu Item'},
            color='total_revenue',
            color_continuous_scale='Purples'
        )

        fig.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Data Table"):
            display_df = top_items.copy()
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_status_chart(status_counts: pd.Series):
        """Render order status chart"""
        st.markdown("### 📦 Order Status Distribution")

        if status_counts.empty:
            st.warning("No data available for order status")
            return

        # Create a pie chart for status
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.sequential.Purples_r
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )

        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Status Breakdown"):
            status_df = pd.DataFrame({
                'Status': status_counts.index,
                'Count': status_counts.values,
                'Percentage': (status_counts.values / status_counts.sum() * 100).round(2)
            })
            status_df['Percentage'] = status_df['Percentage'].apply(lambda x: f"{x}%")
            st.dataframe(status_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_fulfillment_analysis(fulfillment_df: pd.DataFrame, stats: dict):
        """Render fulfillment time analysis"""
        st.markdown("### ⏱️ Order Fulfillment Time Analysis")

        if fulfillment_df.empty or not stats:
            st.info("⚠️ Fulfillment time data not available. Requires 'created_at' and 'updated_at' columns.")
            return

        # Display stats with color-coded metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("⚡ Avg Fulfillment", f"{stats['avg_hours']:.1f} hrs")
        with col2:
            st.metric("📊 Median Time", f"{stats['median_hours']:.1f} hrs")
        with col3:
            st.metric("🚀 Fastest", f"{stats['min_hours']:.1f} hrs")
        with col4:
            st.metric("🐌 Slowest", f"{stats['max_hours']:.1f} hrs")

        # Histogram with custom colors
        fig = px.histogram(
            fulfillment_df,
            x='fulfillment_hours',
            nbins=30,
            labels={'fulfillment_hours': 'Fulfillment Time (hours)', 'count': 'Number of Orders'},
            color_discrete_sequence=['#667eea']
        )

        fig.update_layout(
            title='Distribution of Fulfillment Times',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_peak_times_analysis(hourly: pd.DataFrame, daily: pd.DataFrame):
        """Render peak times analysis"""
        st.markdown("### 🕐 Peak Order Times")

        if hourly.empty or daily.empty:
            st.info("⚠️ Peak times data not available. Requires 'created_at' column.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Orders by Hour of Day**")
            fig = px.bar(
                hourly,
                x='hour',
                y='order_count',
                labels={'hour': 'Hour of Day', 'order_count': 'Number of Orders'},
                color='order_count',
                color_continuous_scale='Purples'
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Orders by Day of Week**")
            fig = px.bar(
                daily,
                x='day_of_week',
                y='order_count',
                labels={'day_of_week': 'Day of Week', 'order_count': 'Number of Orders'},
                color='order_count',
                color_continuous_scale='Purples'
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_trends_analysis(trends: pd.DataFrame):
        """Render order volume trends"""
        st.markdown("### 📈 Daily Trends")

        if trends.empty:
            st.info("⚠️ Trends data not available. Requires 'created_at' column.")
            return

        # Order count over time
        fig = px.area(
            trends,
            x='date',
            y='order_count',
            labels={'date': 'Date', 'order_count': 'Number of Orders'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            title='Daily Order Volume',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Revenue over time
        fig = px.area(
            trends,
            x='date',
            y='total_revenue',
            labels={'date': 'Date', 'total_revenue': 'Revenue (रु)'},
            color_discrete_sequence=['#764ba2']
        )
        fig.update_layout(
            title='Daily Revenue',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Average order value over time
        fig = px.line(
            trends,
            x='date',
            y='avg_order_value',
            labels={'date': 'Date', 'avg_order_value': 'Avg Order Value (रु)'},
            color_discrete_sequence=['#10b981']
        )
        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            title='Average Order Value Trend',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show summary stats
        with st.expander("📊 View Trend Statistics"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Daily Orders", f"{trends['order_count'].mean():.0f}")
            with col2:
                st.metric("Peak Day Orders", f"{trends['order_count'].max():.0f}")
            with col3:
                st.metric("Avg Daily Revenue", f"रु{trends['total_revenue'].mean():,.2f}")

    @staticmethod
    def render_sales_revenue_analysis(monthly_df: pd.DataFrame, growth_df: pd.DataFrame):
        """Render comprehensive sales & revenue analysis"""
        st.markdown("## 💰 Sales & Revenue Analysis")

        if monthly_df.empty:
            st.info("⚠️ Monthly sales data not available. Requires 'created_at' column.")
            return

        # Monthly revenue trend
        st.markdown("### Monthly Revenue Trend")
        fig = px.line(
            monthly_df,
            x='year_month',
            y='total_revenue',
            markers=True,
            labels={'year_month': 'Month', 'total_revenue': 'Revenue (रु)'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Peak sales months
        st.markdown("### 🏅 Peak Sales Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 5 Months by Revenue**")
            top_months = monthly_df.nlargest(5, 'total_revenue').copy()
            top_months['year_month'] = top_months['year_month'].dt.strftime('%B %Y')
            top_months['total_revenue'] = top_months['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
            st.dataframe(
                top_months[['year_month', 'total_revenue', 'order_count']].rename(
                    columns={'year_month': 'Month', 'total_revenue': 'Revenue', 'order_count': 'Orders'}
                ),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("**Top 5 Months by Order Volume**")
            top_order_months = monthly_df.nlargest(5, 'order_count').copy()
            top_order_months['year_month'] = top_order_months['year_month'].dt.strftime('%B %Y')
            top_order_months['total_revenue'] = top_order_months['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
            st.dataframe(
                top_order_months[['year_month', 'order_count', 'total_revenue']].rename(
                    columns={'year_month': 'Month', 'order_count': 'Orders', 'total_revenue': 'Revenue'}
                ),
                use_container_width=True,
                hide_index=True
            )

        # Revenue growth analysis
        if not growth_df.empty:
            st.markdown("### 📊 Revenue Growth Rate")

            # Metrics
            recent_growth = growth_df['revenue_growth_rate'].iloc[-1] if len(growth_df) > 1 else 0
            avg_growth = growth_df['revenue_growth_rate'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Latest Month Growth",
                    f"{recent_growth:+.1f}%",
                    delta=f"{recent_growth:.1f}%"
                )
            with col2:
                st.metric(
                    "Average Growth Rate",
                    f"{avg_growth:+.1f}%"
                )
            with col3:
                positive_months = (growth_df['revenue_growth_rate'] > 0).sum()
                total_months = len(growth_df) - 1
                st.metric(
                    "Positive Growth Months",
                    f"{positive_months}/{total_months}"
                )

            # Growth rate chart with improved styling
            fig = go.Figure()

            colors = ['#10b981' if x > 0 else '#ef4444' for x in growth_df['revenue_growth_rate']]

            fig.add_trace(go.Bar(
                x=growth_df['year_month'],
                y=growth_df['revenue_growth_rate'],
                name='Revenue Growth %',
                marker_color=colors
            ))

            fig.update_layout(
                title='Month-over-Month Revenue Growth Rate',
                xaxis_title='Month',
                yaxis_title='Growth Rate (%)',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Combined revenue and growth
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=growth_df['year_month'],
                y=growth_df['total_revenue'],
                name='Revenue',
                yaxis='y',
                marker_color='#667eea',
                opacity=0.7
            ))

            fig.add_trace(go.Scatter(
                x=growth_df['year_month'],
                y=growth_df['revenue_growth_rate'],
                name='Growth Rate %',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title='Revenue & Growth Rate Combined',
                xaxis_title='Month',
                yaxis=dict(title='Revenue (रु)', side='left'),
                yaxis2=dict(title='Growth Rate (%)', side='right', overlaying='y'),
                hovermode='x unified',
                legend=dict(x=0, y=1.1, orientation='h'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Growth data table
            with st.expander("📋 View Monthly Growth Details"):
                display_growth = growth_df.copy()
                display_growth['year_month'] = display_growth['year_month'].dt.strftime('%B %Y')
                display_growth['total_revenue'] = display_growth['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
                display_growth['revenue_change'] = display_growth['revenue_change'].apply(
                    lambda x: f"रु{x:,.2f}" if pd.notna(x) else "N/A"
                )
                display_growth['revenue_growth_rate'] = display_growth['revenue_growth_rate'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
                )

                st.dataframe(
                    display_growth[['year_month', 'total_revenue', 'order_count', 'revenue_change', 'revenue_growth_rate']].rename(
                        columns={
                            'year_month': 'Month',
                            'total_revenue': 'Revenue',
                            'order_count': 'Orders',
                            'revenue_change': 'Revenue Change',
                            'revenue_growth_rate': 'Growth Rate'
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    @staticmethod
    def render_menu_categories_analysis(categories_df: pd.DataFrame):
        """Render menu category performance analysis"""
        st.subheader("🍽️ Menu Category Performance")

        if categories_df.empty:
            st.info("Category data not available. Requires 'menu_id' column.")
            return

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Categories", f"{len(categories_df)}")
        with col2:
            top_category_revenue = categories_df.iloc[0]['total_revenue']
            st.metric("Top Category Revenue", f"रु{top_category_revenue:,.2f}")
        with col3:
            avg_category_revenue = categories_df['total_revenue'].mean()
            st.metric("Avg Category Revenue", f"रु{avg_category_revenue:,.2f}")

        # Revenue by category
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                categories_df.head(10),
                x='menu_id',
                y='total_revenue',
                title='Top 10 Categories by Revenue',
                labels={'menu_id': 'Category ID', 'total_revenue': 'Revenue (रु)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                categories_df,
                x='unique_items',
                y='total_revenue',
                size='order_count',
                hover_data=['menu_id'],
                title='Items vs Revenue (bubble size = orders)',
                labels={'unique_items': 'Number of Items', 'total_revenue': 'Revenue (रु)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("View Category Details"):
            display_df = categories_df.copy()
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
            display_df['avg_order_value'] = display_df['avg_order_value'].apply(lambda x: f"रु{x:.2f}")
            display_df['revenue_per_item'] = display_df['revenue_per_item'].apply(lambda x: f"रु{x:,.2f}")
            st.dataframe(display_df, use_container_width=True)

    @staticmethod
    def render_cross_selling_analysis(pairs_df: pd.DataFrame):
        """Render cross-selling opportunities analysis"""
        st.subheader("🔗 Cross-Selling Opportunities")

        if pairs_df.empty:
            st.info("Cross-selling data not available. Requires 'order_id' column and multiple items per order.")
            return

        # Top pairs
        top_pairs = pairs_df.head(15).copy()
        top_pairs['item_pair'] = top_pairs['item_1'] + ' + ' + top_pairs['item_2']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Frequently Ordered Together**")
            fig = px.bar(
                top_pairs.head(10),
                x='co_occurrence_count',
                y='item_pair',
                orientation='h',
                labels={'co_occurrence_count': 'Times Ordered Together', 'item_pair': 'Item Pair'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Highest Lift (Strongest Association)**")
            top_lift = pairs_df.nlargest(10, 'lift').copy()
            top_lift['item_pair'] = top_lift['item_1'] + ' + ' + top_lift['item_2']

            fig = px.bar(
                top_lift,
                x='lift',
                y='item_pair',
                orientation='h',
                labels={'lift': 'Lift Score', 'item_pair': 'Item Pair'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.markdown("""
        **Understanding the metrics:**
        - **Co-occurrence Count**: How many times these items were ordered together
        - **Lift**: How much more likely items are ordered together vs independently (>1 = positive correlation)
        """)

        # Data table
        with st.expander("View All Cross-Selling Pairs"):
            display_df = pairs_df.copy()
            display_df['lift'] = display_df['lift'].apply(lambda x: f"{x:.2f}")
            display_df = display_df[['item_1', 'item_2', 'co_occurrence_count', 'lift']]
            st.dataframe(display_df, use_container_width=True)

    @staticmethod
    def render_menu_lifecycle_analysis(lifecycle_df: pd.DataFrame, summary: dict):
        """Render menu item lifecycle analysis"""
        st.subheader("🌱 Menu Item Lifecycle Analysis")

        if lifecycle_df.empty or not summary:
            st.info("Lifecycle data not available. Requires 'created_at' column.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("New Items", summary['new_items'], help=f"On menu ≤ {Config.NEW_ITEM_DAYS} days")
        with col2:
            st.metric("Growing Items", summary['growing_items'], help=f"{Config.NEW_ITEM_DAYS}-{Config.ESTABLISHED_ITEM_DAYS} days")
        with col3:
            st.metric("Established Items", summary['established_items'], help=f"> {Config.ESTABLISHED_ITEM_DAYS} days")
        with col4:
            st.metric("Total Items", summary['total_items'])

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Lifecycle stage distribution
            stage_counts = lifecycle_df['lifecycle_stage'].value_counts()
            fig = px.pie(
                values=stage_counts.values,
                names=stage_counts.index,
                title='Items by Lifecycle Stage'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Revenue by lifecycle stage
            stage_revenue = lifecycle_df.groupby('lifecycle_stage')['total_revenue'].sum()
            fig = px.bar(
                x=stage_revenue.index,
                y=stage_revenue.values,
                labels={'x': 'Lifecycle Stage', 'y': 'Total Revenue (रु)'},
                title='Revenue by Lifecycle Stage'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Performance scatter plot
        fig = px.scatter(
            lifecycle_df,
            x='days_on_menu',
            y='total_revenue',
            color='lifecycle_stage',
            size='total_orders',
            hover_data=['item_name', 'orders_per_day'],
            title='Item Performance Over Time',
            labels={'days_on_menu': 'Days on Menu', 'total_revenue': 'Total Revenue (रु)'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top performers by stage
        st.markdown("**Top Performers by Lifecycle Stage**")

        tab1, tab2, tab3 = st.tabs(["New Items", "Growing Items", "Established Items"])

        with tab1:
            new_items = lifecycle_df[lifecycle_df['lifecycle_stage'] == 'New'].nlargest(10, 'total_revenue')
            if not new_items.empty:
                display_df = new_items[['item_name', 'total_revenue', 'total_orders', 'days_on_menu', 'orders_per_day']].copy()
                display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
                display_df['orders_per_day'] = display_df['orders_per_day'].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No new items found")

        with tab2:
            growing_items = lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Growing'].nlargest(10, 'total_revenue')
            if not growing_items.empty:
                display_df = growing_items[['item_name', 'total_revenue', 'total_orders', 'days_on_menu', 'orders_per_day']].copy()
                display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
                display_df['orders_per_day'] = display_df['orders_per_day'].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No growing items found")

        with tab3:
            established_items = lifecycle_df[lifecycle_df['lifecycle_stage'] == 'Established'].nlargest(10, 'total_revenue')
            if not established_items.empty:
                display_df = established_items[['item_name', 'total_revenue', 'total_orders', 'days_on_menu', 'orders_per_day']].copy()
                display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"रु{x:,.2f}")
                display_df['orders_per_day'] = display_df['orders_per_day'].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No established items found")


def main():
    """Main application entry point"""

    # Render header
    Dashboard.render_header()

    # Sidebar controls with better styling
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Settings")
        st.markdown("---")

        top_n = st.slider(
            "🏆 Top items to display",
            min_value=5,
            max_value=20,
            value=Config.DEFAULT_TOP_N,
            step=1
        )

        min_support = st.slider(
            "🔗 Min co-occurrences",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Minimum times items must be ordered together for cross-selling analysis"
        )

        st.markdown("---")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("""
            <div style='padding: 1rem; background: #f8f9fa; border-radius: 5px; margin-top: 2rem;'>
                <h4 style='margin: 0 0 0.5rem 0; font-size: 0.9rem;'>📌 About</h4>
                <p style='margin: 0; font-size: 0.8rem; color: #666;'>
                    Deerwalk Food System Analytics Dashboard provides comprehensive insights
                    into ordering patterns, revenue trends, and operational metrics.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        orders_df = DataLoader.load_orders(Config.DATA_PATH)

    if orders_df is None or orders_df.empty:
        st.error("Unable to load orders data. Please check the data file and try again.")
        st.stop()

    # Analyze data
    revenue_per_item = DataAnalyzer.calculate_revenue_per_item(orders_df)
    top_items = DataAnalyzer.get_top_n_items(revenue_per_item, n=top_n)
    status_counts = DataAnalyzer.get_status_counts(orders_df)
    delivery_stats = DataAnalyzer.calculate_delivery_success_rate(orders_df)
    fulfillment_df, fulfillment_stats = DataAnalyzer.calculate_fulfillment_time(orders_df)
    hourly_orders, daily_orders = DataAnalyzer.analyze_peak_times(orders_df)
    trends = DataAnalyzer.analyze_order_trends(orders_df)
    monthly_trends = DataAnalyzer.analyze_monthly_trends(orders_df)
    growth_df = DataAnalyzer.calculate_revenue_growth(monthly_trends)
    categories_df = DataAnalyzer.analyze_menu_categories(orders_df)
    cross_sell_df = DataAnalyzer.find_cross_selling_opportunities(orders_df, min_support=min_support)
    lifecycle_df, lifecycle_summary = DataAnalyzer.analyze_menu_item_lifecycle(orders_df)

    # Display metrics
    Dashboard.render_metrics(orders_df, delivery_stats)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sales & Revenue Analysis (comprehensive section)
    Dashboard.render_sales_revenue_analysis(monthly_trends, growth_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render visualizations - Original charts
    st.markdown("## 📊 Revenue & Status Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        Dashboard.render_top_items_chart(top_items, top_n)

    with col2:
        Dashboard.render_status_chart(status_counts)

    st.markdown("<br>", unsafe_allow_html=True)

    # Operational metrics
    st.markdown("## ⚡ Operational Metrics")
    Dashboard.render_fulfillment_analysis(fulfillment_df, fulfillment_stats)

    st.markdown("<br>", unsafe_allow_html=True)

    Dashboard.render_peak_times_analysis(hourly_orders, daily_orders)

    st.markdown("<br>", unsafe_allow_html=True)

    Dashboard.render_trends_analysis(trends)

    st.markdown("<br>", unsafe_allow_html=True)

    # Menu insights section
    st.markdown("## 🔍 Menu Intelligence")

    Dashboard.render_menu_categories_analysis(categories_df)

    st.markdown("<br>", unsafe_allow_html=True)

    Dashboard.render_cross_selling_analysis(cross_sell_df)

    st.markdown("<br>", unsafe_allow_html=True)

    Dashboard.render_menu_lifecycle_analysis(lifecycle_df, lifecycle_summary)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                    border-radius: 10px; margin-top: 3rem;'>
            <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                🍜 Deerwalk Food System Analytics Dashboard | Built with Streamlit & Python
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
