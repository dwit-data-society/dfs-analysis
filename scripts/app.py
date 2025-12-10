import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(page_title="Canteen Performance Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Reuters/Economist style
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
    # The CSV has an extra layer of quotes wrapping the entire header and rows
    # Read raw and clean manually
    try:
        with open('../data/orders.csv', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        # Clean the first line (header) - remove outer quotes and split by semicolon
        header_line = lines[0].strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[1:-1]  # Remove outer quotes
        
        # Split by semicolon and clean individual column names
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
                line = line[1:-1]  # Remove outer quotes
            
            # Split by semicolon and clean values
            values = []
            for val in line.split(';'):
                clean_val = val.strip().replace('""', '"').replace('"', '')
                values.append(clean_val)
            
            if len(values) == len(columns):
                data.append(values)
        
        # Create DataFrame
        orders = pd.DataFrame(data, columns=columns)
        
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        st.stop()

    # Convert NULL strings to proper NaN
    orders = orders.replace(['NULL', 'null', 'Null', 'None', 'none'], np.nan)

    # Ensure critical columns exist
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

    # Remove deleted records
    orders = orders[orders['deleted_at'].isna()].copy()

    # Convert date columns to datetime
    orders['created_at'] = pd.to_datetime(orders['created_at'], errors='coerce')
    orders['updated_at'] = pd.to_datetime(orders['updated_at'], errors='coerce')

    # Convert numeric columns
    orders['total_price'] = pd.to_numeric(orders['total_price'], errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce')

    # Remove rows with NaN in critical columns
    orders = orders.dropna(subset=['created_at', 'updated_at', 'total_price'])

    # Calculate fulfillment time
    orders['fulfillment_mins'] = (orders['updated_at'] - orders['created_at']).dt.total_seconds() / 60

    # Extract time features
    orders['date'] = orders['created_at'].dt.date
    orders['hour'] = orders['created_at'].dt.hour
    orders['day_of_week'] = orders['created_at'].dt.day_name()
    orders['month'] = orders['created_at'].dt.to_period('M').astype(str)
    orders['week'] = orders['created_at'].dt.to_period('W').astype(str)

    # Price categories
    orders['price_category'] = pd.cut(orders['total_price'],
                                       bins=[0, 50, 100, 200, float('inf')],
                                       labels=['Budget (<50)', 'Mid (50-100)', 'Premium (100-200)', 'Luxury (200+)'])

    return orders

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.title("📊 Dashboard Filters")

# Date range filter (handle empty data case)
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

# Item filter
if 'item_name' in df.columns:
    all_items = ['All Items'] + sorted(df['item_name'].dropna().unique().tolist())
    selected_items = st.sidebar.multiselect("Filter by Items", all_items, default=['All Items'])

    if 'All Items' not in selected_items and selected_items:
        filtered_df = filtered_df[filtered_df['item_name'].isin(selected_items)]

# Status filter
if 'status' in df.columns:
    status_filter = st.sidebar.multiselect("Order Status", df['status'].dropna().unique(), default=df['status'].dropna().unique())
    filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]

# Comparison mode
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

# Key insight box
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

# Daily revenue trend
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

    # Add 7-day moving average
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

    # Growth analysis
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
        # Top performers
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
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color=COLORS['neutral']),
            xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
            yaxis=dict(showgrid=False, autorange="reversed")
        )

        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        # Bottom performers
        bottom_items = filtered_df.groupby('item_name').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        bottom_items.columns = ['item_name', 'revenue', 'orders']
        bottom_items = bottom_items[bottom_items['orders'] >= 5]  # Only items with at least 5 orders

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
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=12, color=COLORS['neutral']),
                xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
                yaxis=dict(showgrid=False, autorange="reversed")
            )

            st.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No items with at least 5 orders to display in bottom performers.")

    # Actionable insight
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

# Demand Patterns
st.markdown("<div class='section-header'><h2>3. Temporal Demand Patterns</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>When do customers order? Understanding peak hours guides staffing and inventory decisions</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
hourly = pd.DataFrame()

if not filtered_df.empty:
    with col1:
        # Hourly demand
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
        # Day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = filtered_df.groupby('day_of_week').agg({
            'total_price': 'sum',
            'id': 'count'
        }).reset_index()
        daily.columns = ['day', 'revenue', 'orders']

        # Ensure categories exist even if not in data
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

    # Peak hours insight
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
        # Fulfillment time distribution
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
        # Success rate by hour
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

    # Efficiency insight
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

col1, col2 = st.columns(2)

if not filtered_df.empty:
    with col1:
        # Price category distribution
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
            textposition='outside'
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
        # Monthly revenue by category
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

# Final Recommendations
st.markdown("<div class='section-header'><h2>Strategic Recommendations</h2></div>", unsafe_allow_html=True)

rec_col1, rec_col2, rec_col3 = st.columns(3)

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

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6c757d; font-size: 12px; padding: 2rem 0;'>
    <p>Dashboard generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    <p>Analyzing {len(filtered_df):,} orders • NPR {filtered_df['total_price'].sum():,.0f} in revenue</p>
</div>
""", unsafe_allow_html=True)
