import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "orders.csv"

# Page config
st.set_page_config(page_title="Deerwalk Food System Insights", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Merriweather:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Merriweather', serif;
        font-weight: 700;
        color: #010003;
    }

    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .stMarkdown {
        font-size: 16px;
        line-height: 1.7;
        color: #2c3e50;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 4px solid #0F7F98;
        margin: 1rem 0;
    }

    .insight-box {
        background: #f0f7f8;
        padding: 1.2rem;
        border-radius: 4px;
        border-left: 3px solid #498F8C;
        margin: 1.5rem 0;
        font-style: italic;
        color: #0E4E4A;
    }

    .section-header {
        border-bottom: 2px solid #0F7F98;
        padding-bottom: 0.5rem;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
    }

    .subtitle {
        color: #6c757d;
        font-size: 14px;
        font-weight: 300;
        margin-top: -10px;
    }

    .narrative-text {
        color: #2c3e50;
        font-size: 16px;
        line-height: 1.8;
        text-align: justify;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Color palette
COLORS = {
    'primary': '#0F7F98',
    'secondary': '#498F8C',
    'accent': '#B0C5C3',
    'dark': '#0E4E4A',
    'neutral': '#010003',
    'background': '#f8f9fa',
    'diverging_pos': '#0F7F98',
    'diverging_neg': '#d97575'
}

# Load data with caching
@st.cache_data
def load_data():
    """Load and preprocess orders data—ultra-optimized for Streamlit Cloud."""
    try:
        # Read entire file at once
        with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Split by newline
        lines = content.split('\n')
        
        # Parse header
        header_line = lines[0].strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[1:-1]
        header_line = header_line.replace('""', '"')
        columns = [col.strip().replace('"', '') for col in header_line.split(';')]
        
        # Batch parse data: split all non-header lines at once
        data_lines = [l.strip() for l in lines[1:] if l.strip()]
        
        data = []
        for line in data_lines:
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            
            # Fast split on semicolon
            values = [v.strip().replace('"', '') for v in line.split(';')]
            if len(values) == len(columns):
                data.append(values)
        
        # Create DataFrame in one go (faster than row-by-row)
        orders = pd.DataFrame(data, columns=columns)
        del data  # Free memory immediately
        
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        st.stop()

    # Replace NULL strings with NaN (vectorized)
    for col in orders.columns:
        orders[col] = orders[col].replace(['NULL', 'null', 'Null', 'None', 'none'], np.nan)

    # Validate and create critical columns
    critical_columns = ['created_at', 'updated_at', 'deleted_at', 'item_name', 'status', 'total_price', 'quantity', 'id']
    for col in critical_columns:
        if col not in orders.columns:
            if col == 'deleted_at':
                orders[col] = np.nan
            elif col == 'status':
                orders[col] = 'Unknown'

    # Early filtering: remove deleted records
    mask = orders['deleted_at'].isna()
    orders = orders[mask].copy()

    # Type conversions (pandas is optimized for this)
    orders['created_at'] = pd.to_datetime(orders['created_at'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    orders['updated_at'] = pd.to_datetime(orders['updated_at'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    orders['total_price'] = pd.to_numeric(orders['total_price'], errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce')

    # Drop invalid rows early
    orders = orders.dropna(subset=['created_at', 'updated_at', 'total_price'], how='any')

    # Derived columns (all vectorized, no loops)
    orders['fulfillment_mins'] = (orders['updated_at'] - orders['created_at']).dt.total_seconds() / 60
    orders['date'] = orders['created_at'].dt.date
    orders['hour'] = orders['created_at'].dt.hour
    orders['day_of_week'] = orders['created_at'].dt.day_name()
    orders['month'] = orders['created_at'].dt.to_period('M').astype(str)
    orders['week'] = orders['created_at'].dt.to_period('W').astype(str)

    # Price bins (vectorized)
    orders['price_category'] = pd.cut(
        orders['total_price'],
        bins=[0, 50, 100, 200, float('inf')],
        labels=['Budget (<50)', 'Mid (50-100)', 'Premium (100-200)', 'Luxury (200+)'],
        include_lowest=True
    )

    return orders

@st.cache_resource
def calculate_churn_rate(df, period_days=30):
    if df.empty or 'user_wallet_id' not in df.columns:
        return None, None, None

    max_date = df['created_at'].max()
    period_start = max_date - timedelta(days=period_days)

    pre_period = df[df['created_at'] < period_start]
    active_before = set(pre_period['user_wallet_id'].unique())

    during_period = df[(df['created_at'] >= period_start) & (df['created_at'] <= max_date)]
    active_during = set(during_period['user_wallet_id'].unique())

    lost_customers = active_before - active_during
    churn_rate = (len(lost_customers) / len(active_before) * 100) if len(active_before) > 0 else 0

    customer_activity = df.groupby(df['created_at'].dt.to_period('W'))['user_wallet_id'].nunique().reset_index()
    customer_activity.columns = ['week', 'active_customers']
    customer_activity['week'] = customer_activity['week'].astype(str)

    return churn_rate, len(lost_customers), len(active_before), customer_activity

@st.cache_resource
def forecast_revenue(df, forecast_days=30):
    if df.empty or len(df) < 14:
        return None, None

    daily_rev = df.groupby('date')['total_price'].sum().reset_index()
    daily_rev = daily_rev.sort_values('date')
    daily_rev['ma7'] = daily_rev['total_price'].rolling(window=7, min_periods=1).mean()

    daily_rev['day_num'] = range(len(daily_rev))
    if len(daily_rev) >= 7:
        recent_data = daily_rev.tail(14)
        slope = (recent_data['ma7'].iloc[-1] - recent_data['ma7'].iloc[0]) / len(recent_data)
    else:
        slope = 0

    last_date = daily_rev['date'].max()
    last_value = daily_rev['ma7'].iloc[-1]

    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast_values = [last_value + (slope * i) for i in range(1, forecast_days + 1)]
    forecast_values = [max(0, v) for v in forecast_values]

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast_values
    })

    return daily_rev, forecast_df

@st.cache_resource
def calculate_cancellation_metrics(df):
    if df.empty:
        return None

    cancelled_statuses = ['Cancelled', 'Canceled', 'cancelled', 'canceled', 'Missed']
    df['is_cancelled'] = df['status'].isin(cancelled_statuses)

    total_orders = len(df)
    cancelled_orders = df['is_cancelled'].sum()
    cancellation_rate = (cancelled_orders / total_orders * 100) if total_orders > 0 else 0

    hourly_cancel = df.groupby('hour').agg({
        'id': 'count',
        'is_cancelled': 'sum'
    }).reset_index()
    hourly_cancel.columns = ['hour', 'total', 'cancelled']
    hourly_cancel['cancel_rate'] = (hourly_cancel['cancelled'] / hourly_cancel['total'] * 100)

    item_cancel = df.groupby('item_name').agg({
        'id': 'count',
        'is_cancelled': 'sum'
    }).reset_index()
    item_cancel.columns = ['item_name', 'total', 'cancelled']
    item_cancel['cancel_rate'] = (item_cancel['cancelled'] / item_cancel['total'] * 100)
    item_cancel = item_cancel[item_cancel['total'] >= 5].sort_values('cancel_rate', ascending=False).head(10)

    return {
        'total_rate': cancellation_rate,
        'total_cancelled': cancelled_orders,
        'total_orders': total_orders,
        'hourly': hourly_cancel,
        'by_item': item_cancel
    }

@st.cache_resource
def calculate_revenue_concentration(df):
    if df.empty or 'user_wallet_id' not in df.columns:
        return None

    customer_revenue = df.groupby('user_wallet_id')['total_price'].sum().reset_index()
    customer_revenue.columns = ['customer_id', 'revenue']
    customer_revenue = customer_revenue.sort_values('revenue', ascending=True)

    customer_revenue['cumulative_customers'] = np.arange(1, len(customer_revenue) + 1) / len(customer_revenue) * 100
    customer_revenue['cumulative_revenue'] = customer_revenue['revenue'].cumsum() / customer_revenue['revenue'].sum() * 100

    n = len(customer_revenue)
    revenue_sorted = customer_revenue['revenue'].values
    cumsum = np.cumsum(revenue_sorted)
    gini = (2 * np.sum((np.arange(1, n + 1) * revenue_sorted))) / (n * np.sum(revenue_sorted)) - (n + 1) / n

    total_revenue = customer_revenue['revenue'].sum()
    top_10_pct = int(np.ceil(len(customer_revenue) * 0.1))
    top_20_pct = int(np.ceil(len(customer_revenue) * 0.2))

    top_10_revenue = customer_revenue.nlargest(top_10_pct, 'revenue')['revenue'].sum()
    top_20_revenue = customer_revenue.nlargest(top_20_pct, 'revenue')['revenue'].sum()

    top_10_contribution = (top_10_revenue / total_revenue * 100) if total_revenue > 0 else 0
    top_20_contribution = (top_20_revenue / total_revenue * 100) if total_revenue > 0 else 0

    customer_revenue['tier'] = pd.cut(
        customer_revenue['revenue'],
        bins=[0, customer_revenue['revenue'].quantile(0.5),
              customer_revenue['revenue'].quantile(0.8),
              customer_revenue['revenue'].quantile(0.95),
              float('inf')],
        labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
    )

    tier_stats = customer_revenue.groupby('tier').agg({
        'customer_id': 'count',
        'revenue': 'sum'
    }).reset_index()
    tier_stats.columns = ['tier', 'customers', 'revenue']
    tier_stats['revenue_pct'] = (tier_stats['revenue'] / total_revenue * 100)
    tier_stats['customer_pct'] = (tier_stats['customers'] / len(customer_revenue) * 100)

    return {
        'lorenz_data': customer_revenue,
        'gini': gini,
        'top_10_contribution': top_10_contribution,
        'top_20_contribution': top_20_contribution,
        'tier_stats': tier_stats,
        'total_customers': len(customer_revenue)
    }

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.title("Dashboard Filters")

if not df.empty and df['created_at'].notna().any():
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['created_at'].dt.date >= start_date) & (df['created_at'].dt.date <= end_date)
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
else:
    st.warning("No valid date data found.")
    filtered_df = df.copy()

if 'item_name' in df.columns:
    all_items = ['All Items'] + sorted(df['item_name'].dropna().unique().tolist())
    selected_items = st.sidebar.multiselect("Filter by Items", all_items, default=['All Items'])

    if 'All Items' not in selected_items and selected_items:
        filtered_df = filtered_df[filtered_df['item_name'].isin(selected_items)]

if 'status' in df.columns:
    status_filter = st.sidebar.multiselect("Order Status", df['status'].dropna().unique(), default=df['status'].dropna().unique())
    filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]

st.sidebar.markdown("---")
comparison_mode = st.sidebar.checkbox("Enable Period Comparison", value=False)

if comparison_mode:
    period_days = st.sidebar.slider("Compare last N days", 7, 90, 30)

# Main title
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Deerwalk Food System Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 18px;'>Unpacking Canteen Sales</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p class='narrative-text'>The data collection for this report started on October 14, 2024 achieving 15.6 thousand on the first day and ended on November 3, 2025, achieving 20 thousand as daily revenue. The Deerwalk Foods System has a steady flow of revenue as there is no continuous downward slope anywhere.</p>", unsafe_allow_html=True)

# Executive Summary
st.markdown("<div class='section-header'><h2>Executive Summary</h2></div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_df['total_price'].sum()
total_orders = len(filtered_df)
avg_order_value = filtered_df['total_price'].mean()
delivery_rate = (filtered_df['status'] == 'Delivered').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0

with col1:
    st.metric("Total Revenue", f"NPR {total_revenue:,.0f}")
with col2:
    st.metric("Total Orders", f"{total_orders:,}")
with col3:
    st.metric("Avg Order Value", f"NPR {avg_order_value:.2f}")
with col4:
    st.metric("Delivery Success", f"{delivery_rate:.1f}%")

missed_orders = filtered_df[filtered_df['status'] == 'Missed']
missed_pct = len(missed_orders) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0

st.markdown(f"""
<div class='insight-box'>
<strong>Key Finding:</strong> The canteen processed {total_orders:,} orders generating NPR {total_revenue:,.0f} in revenue.
However, {len(missed_orders)} orders ({missed_pct:.1f}%) were missed, representing NPR {missed_orders['total_price'].sum():,.0f}
in lost revenue. Addressing operational bottlenecks could unlock significant value.
</div>
""", unsafe_allow_html=True)

# Revenue Trends
st.markdown("<div class='section-header'><h2>1. Revenue Performance & Growth Trajectory</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding revenue patterns reveals peak periods and growth opportunities</p>", unsafe_allow_html=True)

if not filtered_df.empty:
    daily_revenue = filtered_df.groupby('date').agg({
        'total_price': 'sum',
        'id': 'count'
    }).reset_index()
    daily_revenue.columns = ['date', 'revenue', 'orders']

    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Scatter(
        x=daily_revenue['date'],
        y=daily_revenue['revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(15, 127, 152, 0.1)'
    ))

    daily_revenue['ma7'] = daily_revenue['revenue'].rolling(window=7, min_periods=1).mean()
    fig_revenue.add_trace(go.Scatter(
        x=daily_revenue['date'],
        y=daily_revenue['ma7'],
        mode='lines',
        name='7-Day Average',
        line=dict(color=COLORS['dark'], width=2, dash='dash')
    ))

    fig_revenue.update_layout(
        title="Daily Revenue Trends",
        xaxis_title="",
        yaxis_title="Revenue (NPR)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Roboto', size=12, color=COLORS['neutral']),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5)
    )

    st.plotly_chart(fig_revenue, use_container_width=True)

    if len(daily_revenue) > 7:
        recent_avg = daily_revenue.tail(7)['revenue'].mean()
        previous_avg = daily_revenue.iloc[-14:-7]['revenue'].mean() if len(daily_revenue) > 14 else daily_revenue.head(7)['revenue'].mean()
        growth_rate = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0

        growth_color = COLORS['diverging_pos'] if growth_rate > 0 else COLORS['diverging_neg']
        st.markdown(f"""
        <div class='insight-box'>
        <strong>Growth Insight:</strong> Week-over-week revenue is {'up' if growth_rate > 0 else 'down'} by
        <span style='color: {growth_color}; font-weight: bold;'>{abs(growth_rate):.1f}%</span>.
        The 7-day moving average shows {'positive momentum' if growth_rate > 0 else 'concerning declining trends'},
        suggesting {'sustained operational improvements' if growth_rate > 0 else 'the need for strategic intervention'}.
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No revenue data available for the selected range.")

st.markdown("<p class='narrative-text'>According to data and analysis from Data Society, the highest revenue earned in a single day was 72.5 thousand on November 28, 2024 and the lowest recorded revenue was on October 12, 2025 with NPR 60.</p>", unsafe_allow_html=True)

# Product Performance
st.markdown("<div class='section-header'><h2>2. Menu Item Performance Matrix</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Which items drive revenue, and which underperform?</p>", unsafe_allow_html=True)

st.markdown("<p class='narrative-text'><strong>Plain rice full shows up undisputed as the most revenue generating item.</strong> Followed by Chicken curry which pairs well with the rice. The Chicken momo has done considerable well despite being one of the items higher in the expense scale. The momo items also show up in the top 5 revenue generating items.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

top_items = pd.DataFrame()
bottom_items = pd.DataFrame()

if not filtered_df.empty:
    with col1:
        top_items = filtered_df.groupby('item_name').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        top_items.columns = ['item_name', 'revenue', 'orders']
        top_items = top_items.sort_values('revenue', ascending=False).head(10)

        fig_top = go.Figure(go.Bar(
            x=top_items['revenue'],
            y=top_items['item_name'],
            orientation='h',
            marker=dict(color=COLORS['primary']),
            hovertemplate='<b>%{y}</b><br>NPR %{x:,.0f}<extra></extra>'
        ))

        fig_top.update_layout(
            title="Top 10 Items by Revenue",
            xaxis_title="Revenue (NPR)",
            yaxis_title="",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            yaxis=dict(showgrid=False, autorange="reversed"),
            margin=dict(r=20, l=10, t=50, b=50)
        )

        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        bottom_items = filtered_df.groupby('item_name').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        bottom_items.columns = ['item_name', 'revenue', 'orders']
        bottom_items = bottom_items[bottom_items['orders'] >= 5]

        if len(bottom_items) > 0:
            bottom_items = bottom_items.sort_values('revenue', ascending=True).head(10)

            fig_bottom = go.Figure(go.Bar(
                x=bottom_items['revenue'],
                y=bottom_items['item_name'],
                orientation='h',
                marker=dict(color=COLORS['diverging_neg']),
                hovertemplate='<b>%{y}</b><br>NPR %{x:,.0f}<extra></extra>'
            ))

            fig_bottom.update_layout(
                title="Bottom 10 Items by Revenue (min 5 orders)",
                xaxis_title="Revenue (NPR)",
                yaxis_title="",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
                yaxis=dict(showgrid=False, autorange="reversed"),
                margin=dict(r=20, l=10, t=50, b=50)
            )

            st.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No items with at least 5 orders to display in bottom performers.")

    if len(top_items) > 0:
        top_item = top_items.iloc[0]
        if len(bottom_items) > 0:
            bottom_item = bottom_items.iloc[0]
            st.markdown(f"""
            <div class='insight-box'>
            <strong>Strategic Recommendation:</strong> <em>{top_item['item_name']}</em> dominates with NPR {top_item['revenue']:,.0f} in revenue
            ({top_item['orders']} orders). Consider featuring this prominently in promotions. Conversely, <em>{bottom_item['item_name']}</em>
            generates only NPR {bottom_item['revenue']:,.0f}. Evaluate whether to discontinue or reposition this item.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='insight-box'>
            <strong>Strategic Recommendation:</strong> <em>{top_item['item_name']}</em> dominates with NPR {top_item['revenue']:,.0f} in revenue
            ({top_item['orders']} orders). Consider featuring this prominently in promotions.
            </div>
            """, unsafe_allow_html=True)

st.markdown("<p class='narrative-text'>There were some items that did not sell well due to unpopularity among customers, with Aloo tarkari being the most ignored. It is closely followed by the newer items on the menu which are baked goods: Cheese Danish and Mini Chocolate Doughnut. However, for these items there is the excuse that they were kept on the menu far fewer times than the Aloo tarkari.</p>", unsafe_allow_html=True)

# Demand Patterns
st.markdown("<div class='section-header'><h2>3. Temporal Demand Patterns</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>When do customers order? Understanding peak hours guides staffing and inventory decisions</p>", unsafe_allow_html=True)

st.markdown("<p class='narrative-text'>The peak time when the most orders arrive in the system is at 16:00 (4 PM) when all the people working in the Deerwalk premises take a break from work. We can deduce that the time after 2 PM is when most of the orders start to arrive. In accordance to the days, all days of the week except Saturday and Sunday show a similar amount of orders coming in.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
hourly = pd.DataFrame()

if not filtered_df.empty:
    with col1:
        hourly = filtered_df.groupby('hour').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        hourly.columns = ['hour', 'revenue', 'orders']

        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly['hour'],
            y=hourly['orders'],
            name='Orders',
            marker=dict(color=COLORS['primary'])
        ))

        fig_hourly.update_layout(
            title="Orders by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Number of Orders",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False, dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        st.plotly_chart(fig_hourly, use_container_width=True)

    with col2:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = filtered_df.groupby('day_of_week').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        daily.columns = ['day', 'revenue', 'orders']

        daily['day'] = pd.Categorical(daily['day'], categories=day_order, ordered=True)
        daily = daily.sort_values('day')

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily['day'],
            y=daily['orders'],
            marker=dict(color=COLORS['secondary'])
        ))

        fig_daily.update_layout(
            title="Orders by Day of Week",
            xaxis_title="",
            yaxis_title="Number of Orders",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        st.plotly_chart(fig_daily, use_container_width=True)

    if len(hourly) > 0:
        peak_hour = hourly.loc[hourly['orders'].idxmax(), 'hour']
        peak_orders = hourly.loc[hourly['orders'].idxmax(), 'orders']
        off_peak = hourly[hourly['hour'].isin([7, 8, 20, 21, 22])]['orders'].sum()

        st.markdown(f"""
        <div class='insight-box'>
        <strong>Operating Hours Analysis:</strong> Peak demand occurs at {int(peak_hour)}:00 with {peak_orders} orders.
        However, early morning (7-8 AM) and evening (8-10 PM) slots show {'minimal' if off_peak < peak_orders * 0.1 else 'moderate'} activity.
        {'Consider extending hours with targeted promotions to capture untapped demand.' if off_peak > 0 else 'Current hours appear optimal for demand patterns.'}
        </div>
        """, unsafe_allow_html=True)

# Operational Efficiency
st.markdown("<div class='section-header'><h2>4. Operational Efficiency Metrics</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Speed and reliability define customer satisfaction</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

if not filtered_df.empty:
    with col1:
        delivered_orders = filtered_df[filtered_df['status'] == 'Delivered']

        if len(delivered_orders) > 0:
            fig_fulfill = go.Figure()
            fig_fulfill.add_trace(go.Histogram(
                x=delivered_orders['fulfillment_mins'],
                nbinsx=30,
                marker=dict(color=COLORS['primary'], line=dict(color='white', width=1))
            ))

            median_time = delivered_orders['fulfillment_mins'].median()
            fig_fulfill.add_vline(x=median_time, line_dash="dash", line_color=COLORS['dark'],
                                  annotation_text=f"Median: {median_time:.1f} min")

            fig_fulfill.update_layout(
                title="Order Fulfillment Time Distribution",
                xaxis_title="Minutes",
                yaxis_title="Number of Orders",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef')
            )

            st.plotly_chart(fig_fulfill, use_container_width=True)
        else:
            st.info("No delivered orders to analyze fulfillment time.")

    with col2:
        hourly_success = filtered_df.groupby('hour').agg({
            'id': 'count',
            'status': lambda x: (x == 'Delivered').sum()
        }).reset_index()
        hourly_success.columns = ['hour', 'total', 'delivered']
        hourly_success['success_rate'] = (hourly_success['delivered'] / hourly_success['total'] * 100)

        fig_success = go.Figure()
        fig_success.add_trace(go.Scatter(
            x=hourly_success['hour'],
            y=hourly_success['success_rate'],
            mode='lines+markers',
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(size=8, color=COLORS['secondary'])
        ))

        fig_success.add_hline(y=95, line_dash="dash", line_color=COLORS['diverging_neg'],
                             annotation_text="95% Target")

        fig_success.update_layout(
            title="Delivery Success Rate by Hour",
            xaxis_title="Hour",
            yaxis_title="Success Rate (%)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False, dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100])
        )

        st.plotly_chart(fig_success, use_container_width=True)

    if len(delivered_orders) > 0:
        avg_fulfill = delivered_orders['fulfillment_mins'].mean()
        median_time = delivered_orders['fulfillment_mins'].median()
        slow_orders = len(delivered_orders[delivered_orders['fulfillment_mins'] > 30])

        st.markdown(f"""
        <div class='insight-box'>
        <strong>Service Speed Analysis:</strong> Average fulfillment time is {avg_fulfill:.1f} minutes (median: {median_time:.1f} min).
        {slow_orders} orders ({slow_orders/len(delivered_orders)*100:.1f}%) took over 30 minutes, indicating potential kitchen bottlenecks.
        Streamlining preparation for high-volume items could significantly improve customer satisfaction.
        </div>
        """, unsafe_allow_html=True)

# Price Point Analysis
st.markdown("<div class='section-header'><h2>5. Price Point & Customer Behavior</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding willingness to pay and order patterns</p>", unsafe_allow_html=True)

st.markdown("<p class='narrative-text'>It seems that people show more willingness to pay in the medium range, meaning from 50-100 NPR for items. After 100 rupees, customers find it harder to pay for food. However, although people buy items in the medium price range most frequently, those who do purchase premium-priced items contribute more to total revenue themselves.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

if not filtered_df.empty:
    with col1:
        price_dist = filtered_df.groupby('price_category').agg({
            'id': 'count',
            'total_price': 'sum'
        }).reset_index()
        price_dist.columns = ['category', 'orders', 'revenue']

        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(
            x=price_dist['category'],
            y=price_dist['orders'],
            marker=dict(color=[COLORS['primary'], COLORS['secondary'], COLORS['dark'], COLORS['accent']]),
            text=price_dist['orders'],
            textposition='auto'
        ))

        fig_price.update_layout(
            title="Order Distribution by Price Category",
            xaxis_title="",
            yaxis_title="Number of Orders",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        monthly_price = filtered_df.groupby(['month', 'price_category']).agg({
            'total_price': 'sum'
        }).reset_index()

        fig_monthly_price = go.Figure()

        for cat in price_dist['category']:
            cat_data = monthly_price[monthly_price['price_category'] == cat]
            fig_monthly_price.add_trace(go.Scatter(
                x=cat_data['month'],
                y=cat_data['total_price'],
                mode='lines+markers',
                name=str(cat),
                stackgroup='one'
            ))

        fig_monthly_price.update_layout(
            title="Revenue Composition Over Time",
            xaxis_title="",
            yaxis_title="Revenue (NPR)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            hovermode='x unified',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        st.plotly_chart(fig_monthly_price, use_container_width=True)

# Customer Churn Analysis
st.markdown("<div class='section-header'><h2>6. Customer Retention & Churn Analysis</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding customer loyalty and retention patterns</p>", unsafe_allow_html=True)

st.markdown("<p class='narrative-text'>This analysis provides information about how many customers are stopping use of the system, possibly going outside for better establishments. There's a real problem here as our system is showing 50% of people lost in the first 30 days.</p>", unsafe_allow_html=True)

if not filtered_df.empty and 'user_wallet_id' in filtered_df.columns:
    col1, col2 = st.columns(2)

    with col1:
        churn_30, lost_30, total_30, activity_30 = calculate_churn_rate(filtered_df, 30)
        churn_60, lost_60, total_60, activity_60 = calculate_churn_rate(filtered_df, 60)

        st.markdown(f"""
        <div class='metric-card'>
        <h4 style='color: {COLORS['primary']}; margin-top: 0;'>Churn Rate Metrics</h4>
        <p style='font-size: 18px; margin: 10px 0;'>
            <strong>30-Day Churn:</strong>
            <span style='color: {COLORS['diverging_neg'] if churn_30 > 20 else COLORS['diverging_pos']}; font-size: 24px; font-weight: bold;'>
                {churn_30:.1f}%
            </span>
        </p>
        <p style='font-size: 14px; color: #6c757d;'>{lost_30} of {total_30} customers lost</p>
        <hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'>
        <p style='font-size: 18px; margin: 10px 0;'>
            <strong>60-Day Churn:</strong>
            <span style='color: {COLORS['diverging_neg'] if churn_60 > 30 else COLORS['diverging_pos']}; font-size: 24px; font-weight: bold;'>
                {churn_60:.1f}%
            </span>
        </p>
        <p style='font-size: 14px; color: #6c757d;'>{lost_60} of {total_60} customers lost</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if activity_30 is not None and not activity_30.empty:
            fig_churn = go.Figure()
            fig_churn.add_trace(go.Scatter(
                x=activity_30['week'],
                y=activity_30['active_customers'],
                mode='lines+markers',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8, color=COLORS['primary']),
                fill='tozeroy',
                fillcolor='rgba(15, 127, 152, 0.1)'
            ))

            fig_churn.update_layout(
                title="Active Customers per Week",
                xaxis_title="Week",
                yaxis_title="Active Customers",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef')
            )

            st.plotly_chart(fig_churn, use_container_width=True)

    churn_status = "concerning" if churn_30 > 25 else "moderate" if churn_30 > 15 else "healthy"
    st.markdown(f"""
    <div class='insight-box'>
    <strong>Retention Analysis:</strong> The 30-day churn rate of {churn_30:.1f}% is {churn_status}.
    {"This suggests significant customer attrition—investigate menu satisfaction, pricing, and service quality." if churn_30 > 25 else
     "While manageable, there's room to improve retention through loyalty programs or personalized offers." if churn_30 > 15 else
     "Customer retention is strong, but continuous engagement initiatives will maintain this momentum."}
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Customer ID data not available for churn analysis.")

st.markdown("<p class='narrative-text'>The line graph shows that the most active users in the span of the data taken were between the dates of July 21, 2025 and September 7, 2025, with the number of most active customers ever recorded being 302. After the active period there has been a sharp decline sadly.</p>", unsafe_allow_html=True)

# Revenue Forecasting
st.markdown("<div class='section-header'><h2>7. Revenue Forecasting & Projections</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Data-driven predictions for planning and budgeting</p>", unsafe_allow_html=True)

if not filtered_df.empty:
    historical_data, forecast_data = forecast_revenue(filtered_df, forecast_days=30)

    if historical_data is not None and forecast_data is not None:
        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['total_price'],
            mode='lines',
            name='Actual Revenue',
            line=dict(color=COLORS['primary'], width=2),
            opacity=0.6
        ))

        fig_forecast.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['ma7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color=COLORS['dark'], width=3)
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines',
            name='30-Day Forecast',
            line=dict(color=COLORS['diverging_pos'], width=3, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(15, 127, 152, 0.1)'
        ))

        fig_forecast.update_layout(
            title="Revenue Forecast (Next 30 Days)",
            xaxis_title="Date",
            yaxis_title="Revenue (NPR)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        total_forecast = forecast_data['forecast'].sum()
        daily_avg_forecast = forecast_data['forecast'].mean()
        historical_avg = historical_data.tail(30)['total_price'].mean()
        change_pct = ((daily_avg_forecast - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0

        with col1:
            st.metric("30-Day Forecast", f"NPR {total_forecast:,.0f}",
                     delta=f"{change_pct:+.1f}% vs last 30 days")
        with col2:
            st.metric("Daily Average (Forecast)", f"NPR {daily_avg_forecast:,.0f}")
        with col3:
            st.metric("Daily Average (Historical)", f"NPR {historical_avg:,.0f}")

        st.markdown(f"""
        <div class='insight-box'>
        <strong>Forecast Insight:</strong> Based on recent trends, daily revenue is projected to
        {'increase' if change_pct > 0 else 'decrease'} by {abs(change_pct):.1f}% over the next 30 days.
        {"This positive trajectory suggests operational improvements are taking effect." if change_pct > 0 else
         "This decline warrants immediate attention—consider promotional campaigns or menu refresh."}
        Total forecasted revenue for the next month: NPR {total_forecast:,.0f}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data for revenue forecasting (minimum 14 days required).")
else:
    st.info("No data available for forecasting.")

st.markdown("<p class='narrative-text'>The trends from October 2024 to November 2025 show a certain pattern from which Data Society conducted predictions for planning and budgeting. According to those predictions, the daily revenue is estimated to increase in the coming days.</p>", unsafe_allow_html=True)

# Order Cancellation Analysis
st.markdown("<div class='section-header'><h2>8. Order Cancellation & Fulfillment Friction</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Identifying pain points in the ordering process</p>", unsafe_allow_html=True)

st.markdown("<p class='narrative-text'>The system at Deerwalk boasts an overall cancellation rate of 1.6% of orders. The chart alongside shows that the time when most orders were cancelled was 2 o'clock. So, we can infer that 2 o'clock is when something happens that causes people to cancel their orders.</p>", unsafe_allow_html=True)

if not filtered_df.empty:
    cancel_metrics = calculate_cancellation_metrics(filtered_df)

    if cancel_metrics:
        col1, col2 = st.columns([1, 2])

        with col1:
            cancel_rate = cancel_metrics['total_rate']
            cancel_color = COLORS['diverging_neg'] if cancel_rate > 10 else COLORS['primary'] if cancel_rate > 5 else COLORS['diverging_pos']

            st.markdown(f"""
            <div class='metric-card'>
            <h4 style='color: {COLORS['primary']}; margin-top: 0;'>Cancellation Metrics</h4>
            <p style='font-size: 18px; margin: 10px 0;'>
                <strong>Overall Rate:</strong>
                <span style='color: {cancel_color}; font-size: 32px; font-weight: bold;'>
                    {cancel_rate:.1f}%
                </span>
            </p>
            <p style='font-size: 14px; color: #6c757d;'>
                {cancel_metrics['total_cancelled']} of {cancel_metrics['total_orders']} orders cancelled/missed
            </p>
            <hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'>
            <p style='font-size: 14px;'>
                <strong>Lost Revenue:</strong><br>
                NPR {filtered_df[filtered_df['status'].isin(['Cancelled', 'Canceled', 'cancelled', 'canceled', 'Missed'])]['total_price'].sum():,.0f}
            </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig_cancel = go.Figure()

            fig_cancel.add_trace(go.Bar(
                x=cancel_metrics['hourly']['hour'],
                y=cancel_metrics['hourly']['cancel_rate'],
                marker=dict(color=COLORS['diverging_neg']),
                name='Cancellation Rate'
            ))

            fig_cancel.add_hline(y=5, line_dash="dash", line_color=COLORS['dark'],
                                annotation_text="5% Threshold")

            fig_cancel.update_layout(
                title="Cancellation Rate by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Cancellation Rate (%)",
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=False, dtick=1),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 10])
            )

            st.plotly_chart(fig_cancel, use_container_width=True)

        if not cancel_metrics['by_item'].empty:
            st.markdown("<h4 style='margin-top: 2rem;'>Items with Highest Cancellation Rates</h4>", unsafe_allow_html=True)

            fig_item_cancel = go.Figure(go.Bar(
                x=cancel_metrics['by_item']['cancel_rate'],
                y=cancel_metrics['by_item']['item_name'],
                orientation='h',
                marker=dict(color=COLORS['diverging_neg']),
                text=cancel_metrics['by_item']['cancel_rate'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ))

            fig_item_cancel.update_layout(
                title="Top 10 Items by Cancellation Rate (min 5 orders)",
                xaxis_title="Cancellation Rate (%)",
                yaxis_title="",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
                yaxis=dict(showgrid=False, autorange="reversed")
            )

            st.plotly_chart(fig_item_cancel, use_container_width=True)

        peak_cancel_hour = cancel_metrics['hourly'].loc[cancel_metrics['hourly']['cancel_rate'].idxmax(), 'hour']
        peak_cancel_rate = cancel_metrics['hourly'].loc[cancel_metrics['hourly']['cancel_rate'].idxmax(), 'cancel_rate']

        st.markdown(f"""
        <div class='insight-box'>
        <strong>Friction Points:</strong> The cancellation rate of {cancel_rate:.1f}% represents
        NPR {filtered_df[filtered_df['status'].isin(['Cancelled', 'Canceled', 'cancelled', 'canceled', 'Missed'])]['total_price'].sum():,.0f} in lost revenue.
        Peak cancellations occur at {int(peak_cancel_hour)}:00 ({peak_cancel_rate:.1f}% rate), suggesting
        {"capacity issues during rush hours." if peak_cancel_rate > 10 else "manageable operational flow."}
        {"High-cancellation items may indicate inventory issues, long preparation times, or quality concerns—investigate root causes immediately." if not cancel_metrics['by_item'].empty else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No cancellation data available.")
else:
    st.info("No data available for cancellation analysis.")

st.markdown("<p class='narrative-text'>Here the item with the highest cancellation rate was Kadai chicken with a whopping 34.1% of its total orders cancelled. Following it is the Mini Chocolate Doughnut which coincidentally was also one of the least ordered items, making it a top contender for items to be discontinued.</p>", unsafe_allow_html=True)

# Revenue Concentration Analysis
st.markdown("<div class='section-header'><h2>9. Revenue Concentration & Customer Value Distribution</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding revenue dependency and customer segmentation</p>", unsafe_allow_html=True)

if not filtered_df.empty and 'user_wallet_id' in filtered_df.columns:
    concentration_data = calculate_revenue_concentration(filtered_df)

    if concentration_data:
        gini = concentration_data['gini']

        if gini < 0.3:
            gini_status = "very low concentration (highly egalitarian)"
            gini_color = COLORS['diverging_pos']
            risk_level = "Low Risk"
        elif gini < 0.5:
            gini_status = "moderate concentration (balanced)"
            gini_color = COLORS['primary']
            risk_level = "Medium Risk"
        elif gini < 0.7:
            gini_status = "high concentration"
            gini_color = COLORS['secondary']
            risk_level = "Medium-High Risk"
        else:
            gini_status = "very high concentration (heavily skewed)"
            gini_color = COLORS['diverging_neg']
            risk_level = "High Risk"

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
            <h4 style='color: {COLORS['primary']}; margin-top: 0;'>Concentration Metrics</h4>
            <p style='font-size: 16px; margin: 10px 0;'>
                <strong>Gini Coefficient:</strong>
            </p>
            <p style='font-size: 48px; margin: 5px 0; font-weight: bold; color: {gini_color};'>
                {gini:.3f}
            </p>
            <p style='font-size: 13px; color: #6c757d; margin-bottom: 15px;'>
                {gini_status}
            </p>
            <hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'>
            <p style='font-size: 14px; margin: 8px 0;'>
                <strong>Top 10% Customers:</strong><br>
                <span style='font-size: 20px; color: {COLORS['dark']}; font-weight: bold;'>
                    {concentration_data['top_10_contribution']:.1f}%
                </span> of revenue
            </p>
            <p style='font-size: 14px; margin: 8px 0;'>
                <strong>Top 20% Customers:</strong><br>
                <span style='font-size: 20px; color: {COLORS['dark']}; font-weight: bold;'>
                    {concentration_data['top_20_contribution']:.1f}%
                </span> of revenue
            </p>
            <hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'>
            <p style='font-size: 14px;'>
                <strong>Risk Exposure:</strong> <span style='color: {gini_color}; font-weight: bold;'>{risk_level}</span>
            </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            lorenz_data = concentration_data['lorenz_data']

            fig_lorenz = go.Figure()

            fig_lorenz.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Perfect Equality',
                line=dict(color='#dee2e6', width=2, dash='dash'),
                showlegend=True
            ))

            fig_lorenz.add_trace(go.Scatter(
                x=np.concatenate([[0], lorenz_data['cumulative_customers'].values]),
                y=np.concatenate([[0], lorenz_data['cumulative_revenue'].values]),
                mode='lines',
                name='Actual Distribution',
                line=dict(color=COLORS['primary'], width=3),
                fill='tonexty',
                fillcolor='rgba(15, 127, 152, 0.2)'
            ))

            fig_lorenz.add_annotation(
                x=10, y=concentration_data['top_10_contribution'],
                text=f"Top 10%: {concentration_data['top_10_contribution']:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLORS['dark'],
                font=dict(size=11, color=COLORS['dark']),
                bgcolor='white',
                bordercolor=COLORS['dark'],
                borderwidth=1
            )

            fig_lorenz.update_layout(
                title="Lorenz Curve - Revenue Distribution",
                xaxis_title="Cumulative % of Customers",
                yaxis_title="Cumulative % of Revenue",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100]),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_lorenz, use_container_width=True)

        st.markdown("<h4 style='margin-top: 2rem;'>Customer Value Segmentation</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            tier_stats = concentration_data['tier_stats']

            fig_tier_count = go.Figure()
            fig_tier_count.add_trace(go.Bar(
                x=tier_stats['tier'],
                y=tier_stats['customer_pct'],
                marker=dict(color=[COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['dark']]),
                text=tier_stats['customer_pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ))

            fig_tier_count.update_layout(
                title="Customer Distribution by Value Tier",
                xaxis_title="",
                yaxis_title="% of Customers",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef')
            )

            st.plotly_chart(fig_tier_count, use_container_width=True)

        with col2:
            fig_tier_revenue = go.Figure()
            fig_tier_revenue.add_trace(go.Bar(
                x=tier_stats['tier'],
                y=tier_stats['revenue_pct'],
                marker=dict(color=[COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['dark']]),
                text=tier_stats['revenue_pct'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto'
            ))

            fig_tier_revenue.update_layout(
                title="Revenue Contribution by Value Tier",
                xaxis_title="",
                yaxis_title="% of Revenue",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#e9ecef')
            )

            st.plotly_chart(fig_tier_revenue, use_container_width=True)

        pareto_ratio = concentration_data['top_20_contribution']

        st.markdown(f"""
        <div class='insight-box'>
        <strong>Revenue Concentration Analysis:</strong> With a Gini coefficient of {gini:.3f}, the canteen exhibits {gini_status}.
        The top 10% of customers account for {concentration_data['top_10_contribution']:.1f}% of revenue,
        while the top 20% contribute {pareto_ratio:.1f}% {"(close to the Pareto principle)" if 75 <= pareto_ratio <= 85 else ""}.
        <br><br>
        {"<strong>Strategic Implication:</strong> High concentration means significant revenue risk if top customers churn. Implement VIP retention programs, personalized engagement, and regular satisfaction checks for high-value customers. Simultaneously, develop strategies to upgrade medium-value customers." if gini > 0.5 else
         "<strong>Strategic Implication:</strong> Balanced revenue distribution reduces risk exposure. Focus on maintaining this healthy diversity while identifying opportunities to increase overall spending across all segments through targeted promotions and menu optimization." if gini > 0.3 else
         "<strong>Strategic Implication:</strong> Extremely even distribution suggests either early-stage operations or highly transactional customer base. Consider implementing loyalty programs to identify and nurture potential high-value customers."}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Customer ID data not available for concentration analysis.")
else:
    st.info("Customer data required for revenue concentration analysis.")

# Final Recommendations
st.markdown("<div class='section-header'><h2>Strategic Recommendations</h2></div>", unsafe_allow_html=True)

rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)

with rec_col1:
    st.markdown(f"""
    <div class='metric-card'>
    <h4 style='color: {COLORS['primary']}; margin-top: 0;'>Menu Optimization</h4>
    <ul style='font-size: 14px;'>
        <li>Promote top 3 items aggressively</li>
        <li>Consider discontinuing items with <5 orders/week</li>
        <li>Bundle popular items for increased AOV</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    peak_hour_val = int(hourly.loc[hourly['orders'].idxmax(), 'hour']) if not hourly.empty else 12
    st.markdown(f"""
    <div class='metric-card'>
    <h4 style='color: {COLORS['secondary']}; margin-top: 0;'>Operating Hours</h4>
    <ul style='font-size: 14px;'>
        <li>Extend evening service with light menu</li>
        <li>Staff appropriately for {peak_hour_val}:00 rush</li>
        <li>Introduce breakfast deals for 7-9 AM</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col3:
    st.markdown(f"""
    <div class='metric-card'>
    <h4 style='color: {COLORS['dark']}; margin-top: 0;'>Service Efficiency</h4>
    <ul style='font-size: 14px;'>
        <li>Target <20 min fulfillment for all orders</li>
        <li>Investigate causes of {len(missed_orders)} missed orders</li>
        <li>Pre-prepare high-volume items during peak hours</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col4:
    st.markdown(f"""
    <div class='metric-card'>
    <h4 style='color: {COLORS['accent']}; margin-top: 0;'>Customer Retention</h4>
    <ul style='font-size: 14px;'>
        <li>Implement VIP program for high-value customers</li>
        <li>Target churned customers with re-engagement offers</li>
        <li>Upgrade medium-value customers through loyalty rewards</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6c757d; font-size: 12px; padding: 2rem 0;'>
    <p>Dashboard generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    <p>Analyzing {len(filtered_df):,} orders • NPR {filtered_df['total_price'].sum():,.0f} in revenue</p>
    <p style='margin-top: 1rem;'>Analysis conducted by Data Society for Deerwalk Foods System</p>
</div>
""", unsafe_allow_html=True)
