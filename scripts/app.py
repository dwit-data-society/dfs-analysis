import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(page_title="Canteen Performance Analysis", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for style with mobile responsiveness
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

    /* Mobile Responsive Styles */
    @media only screen and (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 24px !important;
            line-height: 1.3 !important;
        }

        h2 {
            font-size: 20px !important;
            line-height: 1.3 !important;
        }

        h3, h4 {
            font-size: 18px !important;
        }

        .stMarkdown {
            font-size: 14px;
            line-height: 1.6;
        }

        .subtitle {
            font-size: 12px;
        }

        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .metric-card h4 {
            font-size: 16px !important;
            margin-bottom: 0.5rem !important;
        }

        .metric-card ul {
            padding-left: 1.2rem;
            margin: 0.5rem 0;
        }

        .metric-card ul li {
            font-size: 13px !important;
            margin-bottom: 0.3rem;
        }

        .insight-box {
            padding: 1rem;
            margin: 1rem 0;
            font-size: 13px;
            line-height: 1.5;
        }

        .section-header {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        /* Make Streamlit metrics stack vertically on mobile */
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 12px !important;
        }

        /* Adjust column spacing for mobile */
        [data-testid="column"] {
            padding: 0.25rem !important;
        }

        /* Make tables scrollable */
        .dataframe-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        /* Sidebar adjustments */
        [data-testid="stSidebar"] {
            width: 100%;
            max-width: 300px;
        }

        /* Better touch targets for mobile */
        button, .stButton > button {
            min-height: 44px;
            padding: 0.5rem 1rem;
        }

        /* Adjust multiselect and input fields */
        .stMultiSelect, .stSelectbox, .stDateInput {
            font-size: 14px;
        }
    }

    /* Tablet responsiveness */
    @media only screen and (min-width: 769px) and (max-width: 1024px) {
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1 {
            font-size: 28px !important;
        }

        h2 {
            font-size: 22px !important;
        }

        .metric-card {
            padding: 1.25rem;
        }
    }

    /* Chart responsiveness - applies to all screen sizes */
    .js-plotly-plot {
        width: 100% !important;
    }

    .plotly {
        width: 100% !important;
    }

    /* Ensure plotly charts are responsive */
    .plotly .main-svg {
        width: 100% !important;
        height: auto !important;
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
    # --- HANDLE MALFORMED CSV ---
    try:
        with open('./data/orders.csv', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        # Clean the first line (header)
        header_line = lines[0].strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[1:-1]

        columns = []
        for col in header_line.split(';'):
            clean_col = col.strip().replace('""', '').replace('"', '').replace("'", "").replace(',', '')
            columns.append(clean_col)

        # Parse data rows
        data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            values = []
            for val in line.split(';'):
                clean_val = val.strip().replace('""', '"').replace('"', '')
                values.append(clean_val)

            if len(values) == len(columns):
                data.append(values)

        orders = pd.DataFrame(data, columns=columns)

    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        st.stop()

    orders = orders.replace(['NULL', 'null', 'Null', 'None', 'none'], np.nan)

    critical_columns = ['created_at', 'updated_at', 'deleted_at', 'item_name', 'status', 'total_price', 'quantity', 'id']
    for col in critical_columns:
        if col not in orders.columns:
            if col == 'deleted_at':
                orders[col] = np.nan
            elif col == 'status':
                orders[col] = 'Unknown'
            else:
                st.error(f"Required column '{col}' not found in CSV")
                st.stop()

    orders = orders[orders['deleted_at'].isna()].copy()
    orders['created_at'] = pd.to_datetime(orders['created_at'], errors='coerce')
    orders['updated_at'] = pd.to_datetime(orders['updated_at'], errors='coerce')
    orders['total_price'] = pd.to_numeric(orders['total_price'], errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce')
    orders = orders.dropna(subset=['created_at', 'updated_at', 'total_price'])
    orders['fulfillment_mins'] = (orders['updated_at'] - orders['created_at']).dt.total_seconds() / 60
    orders['date'] = orders['created_at'].dt.date
    orders['hour'] = orders['created_at'].dt.hour
    orders['day_of_week'] = orders['created_at'].dt.day_name()
    orders['month'] = orders['created_at'].dt.to_period('M').astype(str)
    orders['week'] = orders['created_at'].dt.to_period('W').astype(str)
    orders['price_category'] = pd.cut(orders['total_price'],
                                       bins=[0, 50, 100, 200, float('inf')],
                                       labels=['Budget (<50)', 'Mid (50-100)', 'Premium (100-200)', 'Luxury (200+)'])

    return orders

def get_responsive_chart_height():
    """Return appropriate chart height based on screen size"""
    # Default heights - will be overridden by CSS on mobile
    return 400

def create_responsive_layout(fig, height=400):
    """Apply responsive configuration to plotly figures"""
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig

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
    forecast_values = [max(0, last_value + (slope * i)) for i in range(1, forecast_days + 1)]
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast_values
    })
    
    return daily_rev, forecast_df

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
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>University Canteen Performance Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 14px;'>Data-driven insights for operational excellence</p>", unsafe_allow_html=True)
st.markdown("---")

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
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5)
    )
    
    fig_revenue = create_responsive_layout(fig_revenue)
    st.plotly_chart(fig_revenue, use_container_width=True, config={'responsive': True})

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

# Product Performance
st.markdown("<div class='section-header'><h2>2. Menu Item Performance Matrix</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Which items drive revenue, and which underperform?</p>", unsafe_allow_html=True)

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
            text=top_items['revenue'].apply(lambda x: f'NPR {x:,.0f}'),
            textposition='outside'
        ))

        fig_top.update_layout(
            title="Top 10 Items by Revenue",
            xaxis_title="Revenue (NPR)",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            yaxis=dict(showgrid=False, autorange="reversed")
        )

        fig_top = create_responsive_layout(fig_top)
        st.plotly_chart(fig_top, use_container_width=True, config={'responsive': True})

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
                text=bottom_items['revenue'].apply(lambda x: f'NPR {x:,.0f}'),
                textposition='outside'
            ))

            fig_bottom.update_layout(
                title="Bottom 10 Items by Revenue (min 5 orders)",
                xaxis_title="Revenue (NPR)",
                yaxis_title="",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
                yaxis=dict(showgrid=False, autorange="reversed")
            )

            fig_bottom = create_responsive_layout(fig_bottom)
            st.plotly_chart(fig_bottom, use_container_width=True, config={'responsive': True})
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

# Demand Patterns
st.markdown("<div class='section-header'><h2>3. Temporal Demand Patterns</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>When do customers order? Understanding peak hours guides staffing and inventory decisions</p>", unsafe_allow_html=True)

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
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False, dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        fig_hourly = create_responsive_layout(fig_hourly, 350)
        st.plotly_chart(fig_hourly, use_container_width=True, config={'responsive': True})

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
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#e9ecef')
        )

        fig_daily = create_responsive_layout(fig_daily, 350)
        st.plotly_chart(fig_daily, use_container_width=True, config={'responsive': True})

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

# Note: Rest of the sections follow the same pattern
# Add config={'responsive': True} and use_container_width=True to all plotly charts
# Apply create_responsive_layout() to all figures

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 12px; padding: 1rem 0;'>Mobile-optimized dashboard • Swipe to explore • Tap charts for details</p>", unsafe_allow_html=True)