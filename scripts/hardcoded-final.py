import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Import all hardcoded values
from hardcoded import *

# Page config
st.set_page_config(page_title="Deerwalk Food System Insights", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Merriweather:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    h1, h2, h3 { font-family: 'Merriweather', serif; font-weight: 700; color: #010003; }
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    .stMarkdown { font-size: 16px; line-height: 1.7; color: #2c3e50; }
    .metric-card { background: #f8f9fa; padding: 1.5rem; border-radius: 4px; border-left: 4px solid #0F7F98; margin: 1rem 0; }
    .insight-box { background: #f0f7f8; padding: 1.2rem; border-radius: 4px; border-left: 3px solid #498F8C; margin: 1.5rem 0; font-style: italic; color: #0E4E4A; }
    .section-header { border-bottom: 2px solid #0F7F98; padding-bottom: 0.5rem; margin-top: 3rem; margin-bottom: 1.5rem; }
    .subtitle { color: #6c757d; font-size: 14px; font-weight: 300; margin-top: -10px; }
    .narrative-text { color: #2c3e50; font-size: 16px; line-height: 1.8; text-align: justify; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Colors
COLORS = {'primary': '#0F7F98', 'secondary': '#498F8C', 'accent': '#B0C5C3', 'dark': '#0E4E4A', 'neutral': '#010003', 'diverging_pos': '#0F7F98', 'diverging_neg': '#d97575'}

# Title
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Deerwalk Food System Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 18px;'>Unpacking Canteen Sales</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p class='narrative-text'>The data collection for this report started on October 14, 2024 achieving 15.6 thousand on the first day and ended on November 3, 2025, achieving 20 thousand as daily revenue. The Deerwalk Foods System has a steady flow of revenue as there is no continuous downward slope anywhere.</p>", unsafe_allow_html=True)

# Executive Summary
st.markdown("<div class='section-header'><h2>Executive Summary</h2></div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenue", f"NPR {TOTAL_REVENUE:,.0f}")
with col2:
    st.metric("Total Orders", f"{TOTAL_ORDERS:,}")
with col3:
    st.metric("Avg Order Value", f"NPR {AVG_ORDER_VALUE:.2f}")
with col4:
    st.metric("Delivery Success", f"{DELIVERY_RATE:.1f}%")

st.markdown(f"""<div class='insight-box'><strong>Key Finding:</strong> The canteen processed {TOTAL_ORDERS:,} orders generating NPR {TOTAL_REVENUE:,.0f} in revenue. However, {MISSED_ORDERS_COUNT} orders ({MISSED_ORDERS_PCT:.1f}%) were missed, representing NPR {MISSED_REVENUE:,.0f} in lost revenue. Addressing operational bottlenecks could unlock significant value.</div>""", unsafe_allow_html=True)

# 1. Revenue Trends
st.markdown("<div class='section-header'><h2>1. Revenue Performance & Growth Trajectory</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding revenue patterns reveals peak periods and growth opportunities</p>", unsafe_allow_html=True)

fig_revenue = go.Figure()
fig_revenue.add_trace(go.Scatter(x=DAILY_REVENUE_DATES, y=DAILY_REVENUE_VALUES, mode='lines', name='Daily Revenue', line=dict(color=COLORS['primary'], width=2.5), fill='tozeroy', fillcolor='rgba(15, 127, 152, 0.1)'))
fig_revenue.add_trace(go.Scatter(x=DAILY_REVENUE_DATES, y=DAILY_REVENUE_MA7, mode='lines', name='7-Day Average', line=dict(color=COLORS['dark'], width=2, dash='dash')))
fig_revenue.update_layout(title="Daily Revenue Trends", xaxis_title="", yaxis_title="Revenue (NPR)", hovermode='x unified', plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5), yaxis=dict(showgrid=True, gridcolor='#e9ecef', gridwidth=0.5))
st.plotly_chart(fig_revenue, use_container_width=True)

growth_color = COLORS['diverging_pos'] if GROWTH_RATE > 0 else COLORS['diverging_neg']
st.markdown(f"""<div class='insight-box'><strong>Growth Insight:</strong> Week-over-week revenue is {'up' if GROWTH_RATE > 0 else 'down'} by <span style='color: {growth_color}; font-weight: bold;'>{abs(GROWTH_RATE):.1f}%</span>. The 7-day moving average shows {'positive momentum' if GROWTH_RATE > 0 else 'concerning declining trends'}, suggesting {'sustained operational improvements' if GROWTH_RATE > 0 else 'the need for strategic intervention'}.</div>""", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>According to data and analysis from Data Society, the highest revenue earned in a single day was 72.5 thousand on November 28, 2024 and the lowest recorded revenue was on October 12, 2025 with NPR 60.</p>", unsafe_allow_html=True)

# 2. Product Performance
st.markdown("<div class='section-header'><h2>2. Menu Item Performance Matrix</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Which items drive revenue, and which underperform?</p>", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'><strong>Plain rice full shows up undisputed as the most revenue generating item.</strong> Followed by Chicken curry which pairs well with the rice. The Chicken momo has done considerable well despite being one of the items higher in the expense scale. The momo items also show up in the top 5 revenue generating items.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_top = go.Figure(go.Bar(x=TOP_ITEMS_REVENUE, y=TOP_ITEMS_NAMES, orientation='h', marker=dict(color=COLORS['primary']), hovertemplate='<b>%{y}</b><br>NPR %{x:,.0f}<extra></extra>'))
    fig_top.update_layout(title="Top 10 Items by Revenue", xaxis_title="Revenue (NPR)", yaxis_title="", height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=True, gridcolor='#e9ecef'), yaxis=dict(showgrid=False, autorange="reversed"), margin=dict(r=20, l=10, t=50, b=50))
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    fig_bottom = go.Figure(go.Bar(x=BOTTOM_ITEMS_REVENUE, y=BOTTOM_ITEMS_NAMES, orientation='h', marker=dict(color=COLORS['diverging_neg']), hovertemplate='<b>%{y}</b><br>NPR %{x:,.0f}<extra></extra>'))
    fig_bottom.update_layout(title="Bottom 10 Items by Revenue (min 5 orders)", xaxis_title="Revenue (NPR)", yaxis_title="", height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=True, gridcolor='#e9ecef'), yaxis=dict(showgrid=False, autorange="reversed"), margin=dict(r=20, l=10, t=50, b=50))
    st.plotly_chart(fig_bottom, use_container_width=True)

st.markdown(f"""<div class='insight-box'><strong>Strategic Recommendation:</strong> <em>{TOP_ITEMS_NAMES[0]}</em> dominates with NPR {TOP_ITEMS_REVENUE[0]:,.0f} in revenue ({TOP_ITEMS_ORDERS[0]} orders). Consider featuring this prominently in promotions. Conversely, <em>{BOTTOM_ITEMS_NAMES[0]}</em> generates only NPR {BOTTOM_ITEMS_REVENUE[0]:,.0f}. Evaluate whether to discontinue or reposition this item.</div>""", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>There were some items that did not sell well due to unpopularity among customers, with Aloo tarkari being the most ignored. It is closely followed by the newer items on the menu which are baked goods: Cheese Danish and Mini Chocolate Doughnut. However, for these items there is the excuse that they were kept on the menu far fewer times than the Aloo tarkari.</p>", unsafe_allow_html=True)

# 3. Demand Patterns
st.markdown("<div class='section-header'><h2>3. Temporal Demand Patterns</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>When do customers order? Understanding peak hours guides staffing and inventory decisions</p>", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>The peak time when the most orders arrive in the system is at 16:00 (4 PM) when all the people working in the Deerwalk premises take a break from work. We can deduce that the time after 2 PM is when most of the orders start to arrive. In accordance to the days, all days of the week except Saturday and Sunday show a similar amount of orders coming in.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(x=HOURLY_HOURS, y=HOURLY_ORDERS, name='Orders', marker=dict(color=COLORS['primary'])))
    fig_hourly.update_layout(title="Orders by Hour of Day", xaxis_title="Hour", yaxis_title="Number of Orders", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False, dtick=1), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=DAILY_DAYS, y=DAILY_ORDERS, marker=dict(color=COLORS['secondary'])))
    fig_daily.update_layout(title="Orders by Day of Week", xaxis_title="", yaxis_title="Number of Orders", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown(f"""<div class='insight-box'><strong>Operating Hours Analysis:</strong> Peak demand occurs at {PEAK_HOUR}:00 with {PEAK_HOUR_ORDERS} orders. However, early morning (7-8 AM) and evening (8-10 PM) slots show minimal activity. Consider extending hours with targeted promotions to capture untapped demand.</div>""", unsafe_allow_html=True)

# 4. Operational Efficiency
st.markdown("<div class='section-header'><h2>4. Operational Efficiency Metrics</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Speed and reliability define customer satisfaction</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_fulfill = go.Figure()
    fig_fulfill.add_trace(go.Histogram(x=FULFILLMENT_TIMES_SAMPLE, nbinsx=30, marker=dict(color=COLORS['primary'], line=dict(color='white', width=1))))
    fig_fulfill.add_vline(x=MEDIAN_FULFILLMENT, line_dash="dash", line_color=COLORS['dark'], annotation_text=f"Median: {MEDIAN_FULFILLMENT:.1f} min")
    fig_fulfill.update_layout(title="Order Fulfillment Time Distribution", xaxis_title="Minutes", yaxis_title="Number of Orders", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=True, gridcolor='#e9ecef'), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_fulfill, use_container_width=True)

with col2:
    fig_success = go.Figure()
    fig_success.add_trace(go.Scatter(x=SUCCESS_RATE_HOURS, y=SUCCESS_RATE_VALUES, mode='lines+markers', line=dict(color=COLORS['secondary'], width=3), marker=dict(size=8, color=COLORS['secondary'])))
    fig_success.add_hline(y=95, line_dash="dash", line_color=COLORS['diverging_neg'], annotation_text="95% Target")
    fig_success.update_layout(title="Delivery Success Rate by Hour", xaxis_title="Hour", yaxis_title="Success Rate (%)", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False, dtick=1), yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100]))
    st.plotly_chart(fig_success, use_container_width=True)

st.markdown(f"""<div class='insight-box'><strong>Service Speed Analysis:</strong> Average fulfillment time is {AVG_FULFILLMENT:.1f} minutes (median: {MEDIAN_FULFILLMENT:.1f} min). {SLOW_ORDERS} orders ({SLOW_ORDERS_PCT:.1f}%) took over 30 minutes, indicating potential kitchen bottlenecks. Streamlining preparation for high-volume items could significantly improve customer satisfaction.</div>""", unsafe_allow_html=True)

# 5. Price Point Analysis
st.markdown("<div class='section-header'><h2>5. Price Point & Customer Behavior</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding willingness to pay and order patterns</p>", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>It seems that people show more willingness to pay in the medium range, meaning from 50-100 NPR for items. After 100 rupees, customers find it harder to pay for food. However, although people buy items in the medium price range most frequently, those who do purchase premium-priced items contribute more to total revenue themselves.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig_price = go.Figure()
    fig_price.add_trace(go.Bar(x=PRICE_CATEGORIES, y=PRICE_ORDERS, marker=dict(color=[COLORS['primary'], COLORS['secondary'], COLORS['dark'], COLORS['accent']]), text=PRICE_ORDERS, textposition='auto'))
    fig_price.update_layout(title="Order Distribution by Price Category", xaxis_title="", yaxis_title="Number of Orders", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    fig_monthly_price = go.Figure()
    fig_monthly_price.add_trace(go.Scatter(x=MONTHLY_PRICE_MONTHS, y=MONTHLY_PRICE_BUDGET_UNDER50, mode='lines+markers', name='Budget (<50)', stackgroup='one'))
    fig_monthly_price.add_trace(go.Scatter(x=MONTHLY_PRICE_MONTHS, y=MONTHLY_PRICE_MID_50_100, mode='lines+markers', name='Mid (50-100)', stackgroup='one'))
    fig_monthly_price.add_trace(go.Scatter(x=MONTHLY_PRICE_MONTHS, y=MONTHLY_PRICE_PREMIUM_100_200, mode='lines+markers', name='Premium (100-200)', stackgroup='one'))
    fig_monthly_price.add_trace(go.Scatter(x=MONTHLY_PRICE_MONTHS, y=MONTHLY_PRICE_LUXURY_200_PLUS, mode='lines+markers', name='Luxury (200+)', stackgroup='one'))
    fig_monthly_price.update_layout(title="Revenue Composition Over Time", xaxis_title="", yaxis_title="Revenue (NPR)", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), hovermode='x unified', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_monthly_price, use_container_width=True)

# 6. Customer Churn Analysis
st.markdown("<div class='section-header'><h2>6. Customer Retention & Churn Analysis</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding customer loyalty and retention patterns</p>", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>This analysis provides information about how many customers are stopping use of the system, possibly going outside for better establishments. There's a real problem here as our system is showing 50% of people lost in the first 30 days.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['primary']}; margin-top: 0;'>Churn Rate Metrics</h4><p style='font-size: 18px; margin: 10px 0;'><strong>30-Day Churn:</strong> <span style='color: {COLORS['diverging_neg'] if CHURN_30_RATE > 20 else COLORS['diverging_pos']}; font-size: 24px; font-weight: bold;'>{CHURN_30_RATE:.1f}%</span></p><p style='font-size: 14px; color: #6c757d;'>{CHURN_30_LOST} of {CHURN_30_TOTAL} customers lost</p><hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'><p style='font-size: 18px; margin: 10px 0;'><strong>60-Day Churn:</strong> <span style='color: {COLORS['diverging_neg'] if CHURN_60_RATE > 30 else COLORS['diverging_pos']}; font-size: 24px; font-weight: bold;'>{CHURN_60_RATE:.1f}%</span></p><p style='font-size: 14px; color: #6c757d;'>{CHURN_60_LOST} of {CHURN_60_TOTAL} customers lost</p></div>""", unsafe_allow_html=True)

with col2:
    fig_churn = go.Figure()
    fig_churn.add_trace(go.Scatter(x=ACTIVITY_WEEKS, y=ACTIVITY_CUSTOMERS, mode='lines+markers', line=dict(color=COLORS['primary'], width=3), marker=dict(size=8, color=COLORS['primary']), fill='tozeroy', fillcolor='rgba(15, 127, 152, 0.1)'))
    fig_churn.update_layout(title="Active Customers per Week", xaxis_title="Week", yaxis_title="Active Customers", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_churn, use_container_width=True)

churn_status = "concerning" if CHURN_30_RATE > 25 else "moderate" if CHURN_30_RATE > 15 else "healthy"
st.markdown(f"""<div class='insight-box'><strong>Retention Analysis:</strong> The 30-day churn rate of {CHURN_30_RATE:.1f}% is {churn_status}. {"This suggests significant customer attrition—investigate menu satisfaction, pricing, and service quality." if CHURN_30_RATE > 25 else "While manageable, there's room to improve retention through loyalty programs or personalized offers." if CHURN_30_RATE > 15 else "Customer retention is strong, but continuous engagement initiatives will maintain this momentum."}</div>""", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>The line graph shows that the most active users in the span of the data taken were between the dates of July 21, 2025 and September 7, 2025, with the number of most active customers ever recorded being 302. After the active period there has been a sharp decline sadly.</p>", unsafe_allow_html=True)

# 7. Revenue Forecasting
st.markdown("<div class='section-header'><h2>7. Revenue Forecasting & Projections</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Data-driven predictions for planning and budgeting</p>", unsafe_allow_html=True)

recent_revenue = DAILY_REVENUE_VALUES[-14:]
recent_ma7 = DAILY_REVENUE_MA7[-14:]
slope = (recent_ma7[-1] - recent_ma7[0]) / 14
last_value = DAILY_REVENUE_MA7[-1]
forecast_values = [max(0, last_value + (slope * i)) for i in range(1, 31)]
last_date = datetime.strptime(DAILY_REVENUE_DATES[-1], '%Y-%m-%d')
forecast_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=DAILY_REVENUE_DATES, y=DAILY_REVENUE_VALUES, mode='lines', name='Actual Revenue', line=dict(color=COLORS['primary'], width=2), opacity=0.6))
fig_forecast.add_trace(go.Scatter(x=DAILY_REVENUE_DATES, y=DAILY_REVENUE_MA7, mode='lines', name='7-Day Average', line=dict(color=COLORS['dark'], width=3)))
fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='30-Day Forecast', line=dict(color=COLORS['diverging_pos'], width=3, dash='dash'), fill='tozeroy', fillcolor='rgba(15, 127, 152, 0.1)'))
fig_forecast.update_layout(title="Revenue Forecast (Next 30 Days)", xaxis_title="Date", yaxis_title="Revenue (NPR)", height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), hovermode='x unified', xaxis=dict(showgrid=True, gridcolor='#e9ecef'), yaxis=dict(showgrid=True, gridcolor='#e9ecef'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_forecast, use_container_width=True)

col1, col2, col3 = st.columns(3)
total_forecast = sum(forecast_values)
daily_avg_forecast = np.mean(forecast_values)
historical_avg = np.mean(DAILY_REVENUE_VALUES[-30:])
change_pct = ((daily_avg_forecast - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0

with col1:
    st.metric("30-Day Forecast", f"NPR {total_forecast:,.0f}", delta=f"{change_pct:+.1f}% vs last 30 days")
with col2:
    st.metric("Daily Average (Forecast)", f"NPR {daily_avg_forecast:,.0f}")
with col3:
    st.metric("Daily Average (Historical)", f"NPR {historical_avg:,.0f}")

st.markdown(f"""<div class='insight-box'><strong>Forecast Insight:</strong> Based on recent trends, daily revenue is projected to {'increase' if change_pct > 0 else 'decrease'} by {abs(change_pct):.1f}% over the next 30 days. {"This positive trajectory suggests operational improvements are taking effect." if change_pct > 0 else "This decline warrants immediate attention—consider promotional campaigns or menu refresh."} Total forecasted revenue for the next month: NPR {total_forecast:,.0f}.</div>""", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>The trends from October 2024 to November 2025 show a certain pattern from which Data Society conducted predictions for planning and budgeting. According to those predictions, the daily revenue is estimated to increase in the coming days.</p>", unsafe_allow_html=True)

# 8. Order Cancellation Analysis
st.markdown("<div class='section-header'><h2>8. Order Cancellation & Fulfillment Friction</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Identifying pain points in the ordering process</p>", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>The system at Deerwalk boasts an overall cancellation rate of 1.6% of orders. The chart alongside shows that the time when most orders were cancelled was 2 o'clock. So, we can infer that 2 o'clock is when something happens that causes people to cancel their orders.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    cancel_color = COLORS['diverging_neg'] if CANCEL_RATE > 10 else COLORS['primary'] if CANCEL_RATE > 5 else COLORS['diverging_pos']
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['primary']}; margin-top: 0;'>Cancellation Metrics</h4><p style='font-size: 18px; margin: 10px 0;'><strong>Overall Rate:</strong> <span style='color: {cancel_color}; font-size: 32px; font-weight: bold;'>{CANCEL_RATE:.1f}%</span></p><p style='font-size: 14px; color: #6c757d;'>{CANCEL_COUNT} of {TOTAL_ORDERS} orders cancelled/missed</p><hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'><p style='font-size: 14px;'><strong>Lost Revenue:</strong><br>NPR {CANCEL_LOST_REVENUE:,.0f}</p></div>""", unsafe_allow_html=True)

with col2:
    fig_cancel = go.Figure()
    fig_cancel.add_trace(go.Bar(x=CANCEL_HOURS, y=CANCEL_RATES, marker=dict(color=COLORS['diverging_neg']), name='Cancellation Rate'))
    fig_cancel.add_hline(y=5, line_dash="dash", line_color=COLORS['dark'], annotation_text="5% Threshold")
    fig_cancel.update_layout(title="Cancellation Rate by Hour", xaxis_title="Hour of Day", yaxis_title="Cancellation Rate (%)", height=300, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False, dtick=1), yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 10]))
    st.plotly_chart(fig_cancel, use_container_width=True)

st.markdown("<h4 style='margin-top: 2rem;'>Items with Highest Cancellation Rates</h4>", unsafe_allow_html=True)
fig_item_cancel = go.Figure(go.Bar(x=CANCEL_ITEMS_RATES, y=CANCEL_ITEMS, orientation='h', marker=dict(color=COLORS['diverging_neg']), text=[f'{x:.1f}%' for x in CANCEL_ITEMS_RATES], textposition='auto'))
fig_item_cancel.update_layout(title="Top 10 Items by Cancellation Rate (min 5 orders)", xaxis_title="Cancellation Rate (%)", yaxis_title="", height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=True, gridcolor='#e9ecef'), yaxis=dict(showgrid=False, autorange="reversed"))
st.plotly_chart(fig_item_cancel, use_container_width=True)

st.markdown(f"""<div class='insight-box'><strong>Friction Points:</strong> The cancellation rate of {CANCEL_RATE:.1f}% represents NPR {CANCEL_LOST_REVENUE:,.0f} in lost revenue. Peak cancellations occur at {PEAK_CANCEL_HOUR}:00 ({PEAK_CANCEL_RATE:.1f}% rate), suggesting {"capacity issues during rush hours." if PEAK_CANCEL_RATE > 10 else "manageable operational flow."} High-cancellation items may indicate inventory issues, long preparation times, or quality concerns—investigate root causes immediately.</div>""", unsafe_allow_html=True)
st.markdown("<p class='narrative-text'>Here the item with the highest cancellation rate was Kadai chicken with a whopping 34.1% of its total orders cancelled. Following it is the Mini Chocolate Doughnut which coincidentally was also one of the least ordered items, making it a top contender for items to be discontinued.</p>", unsafe_allow_html=True)

# 9. Revenue Concentration Analysis
st.markdown("<div class='section-header'><h2>9. Revenue Concentration & Customer Value Distribution</h2></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Understanding revenue dependency and customer segmentation</p>", unsafe_allow_html=True)

gini = GINI_COEFFICIENT
if gini < 0.3:
    gini_status, gini_color, risk_level = "very low concentration (highly egalitarian)", COLORS['diverging_pos'], "Low Risk"
elif gini < 0.5:
    gini_status, gini_color, risk_level = "moderate concentration (balanced)", COLORS['primary'], "Medium Risk"
elif gini < 0.7:
    gini_status, gini_color, risk_level = "high concentration", COLORS['secondary'], "Medium-High Risk"
else:
    gini_status, gini_color, risk_level = "very high concentration (heavily skewed)", COLORS['diverging_neg'], "High Risk"

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['primary']}; margin-top: 0;'>Concentration Metrics</h4><p style='font-size: 16px; margin: 10px 0;'><strong>Gini Coefficient:</strong></p><p style='font-size: 48px; margin: 5px 0; font-weight: bold; color: {gini_color};'>{gini:.3f}</p><p style='font-size: 13px; color: #6c757d; margin-bottom: 15px;'>{gini_status}</p><hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'><p style='font-size: 14px; margin: 8px 0;'><strong>Top 10% Customers:</strong><br><span style='font-size: 20px; color: {COLORS['dark']}; font-weight: bold;'>{TOP_10_CONTRIBUTION:.1f}%</span> of revenue</p><p style='font-size: 14px; margin: 8px 0;'><strong>Top 20% Customers:</strong><br><span style='font-size: 20px; color: {COLORS['dark']}; font-weight: bold;'>{TOP_20_CONTRIBUTION:.1f}%</span> of revenue</p><hr style='margin: 15px 0; border: none; border-top: 1px solid #dee2e6;'><p style='font-size: 14px;'><strong>Risk Exposure:</strong> <span style='color: {gini_color}; font-weight: bold;'>{risk_level}</span></p></div>""", unsafe_allow_html=True)

with col2:
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect Equality', line=dict(color='#dee2e6', width=2, dash='dash'), showlegend=True))
    fig_lorenz.add_trace(go.Scatter(x=np.concatenate([[0], LORENZ_CUSTOMERS]), y=np.concatenate([[0], LORENZ_REVENUE]), mode='lines', name='Actual Distribution', line=dict(color=COLORS['primary'], width=3), fill='tonexty', fillcolor='rgba(15, 127, 152, 0.2)'))
    fig_lorenz.add_annotation(x=10, y=TOP_10_CONTRIBUTION, text=f"Top 10%: {TOP_10_CONTRIBUTION:.1f}%", showarrow=True, arrowhead=2, arrowcolor=COLORS['dark'], font=dict(size=11, color=COLORS['dark']), bgcolor='white', bordercolor=COLORS['dark'], borderwidth=1)
    fig_lorenz.update_layout(title="Lorenz Curve - Revenue Distribution", xaxis_title="Cumulative % of Customers", yaxis_title="Cumulative % of Revenue", height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100]), yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 100]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_lorenz, use_container_width=True)

st.markdown("<h4 style='margin-top: 2rem;'>Customer Value Segmentation</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    fig_tier_count = go.Figure()
    fig_tier_count.add_trace(go.Bar(x=TIER_NAMES, y=TIER_CUSTOMER_PCT, marker=dict(color=[COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['dark']]), text=[f'{x:.1f}%' for x in TIER_CUSTOMER_PCT], textposition='auto'))
    fig_tier_count.update_layout(title="Customer Distribution by Value Tier", xaxis_title="", yaxis_title="% of Customers", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_tier_count, use_container_width=True)

with col2:
    fig_tier_revenue = go.Figure()
    fig_tier_revenue.add_trace(go.Bar(x=TIER_NAMES, y=TIER_REVENUE_PCT, marker=dict(color=[COLORS['accent'], COLORS['secondary'], COLORS['primary'], COLORS['dark']]), text=[f'{x:.1f}%' for x in TIER_REVENUE_PCT], textposition='auto'))
    fig_tier_revenue.update_layout(title="Revenue Contribution by Value Tier", xaxis_title="", yaxis_title="% of Revenue", height=350, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Roboto', size=12, color=COLORS['neutral']), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e9ecef'))
    st.plotly_chart(fig_tier_revenue, use_container_width=True)

pareto_ratio = TOP_20_CONTRIBUTION
st.markdown(f"""<div class='insight-box'><strong>Revenue Concentration Analysis:</strong> With a Gini coefficient of {gini:.3f}, the canteen exhibits {gini_status}. The top 10% of customers account for {TOP_10_CONTRIBUTION:.1f}% of revenue, while the top 20% contribute {pareto_ratio:.1f}% {"(close to the Pareto principle)" if 75 <= pareto_ratio <= 85 else ""}.<br><br>{"<strong>Strategic Implication:</strong> High concentration means significant revenue risk if top customers churn. Implement VIP retention programs, personalized engagement, and regular satisfaction checks for high-value customers. Simultaneously, develop strategies to upgrade medium-value customers." if gini > 0.5 else "<strong>Strategic Implication:</strong> Balanced revenue distribution reduces risk exposure. Focus on maintaining this healthy diversity while identifying opportunities to increase overall spending across all segments through targeted promotions and menu optimization." if gini > 0.3 else "<strong>Strategic Implication:</strong> Extremely even distribution suggests either early-stage operations or highly transactional customer base. Consider implementing loyalty programs to identify and nurture potential high-value customers."}</div>""", unsafe_allow_html=True)

# Strategic Recommendations
st.markdown("<div class='section-header'><h2>Strategic Recommendations</h2></div>", unsafe_allow_html=True)
rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)

with rec_col1:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['primary']}; margin-top: 0;'>Menu Optimization</h4><ul style='font-size: 14px;'><li>Promote top 3 items aggressively</li><li>Consider discontinuing items with <5 orders/week</li><li>Bundle popular items for increased AOV</li></ul></div>""", unsafe_allow_html=True)

with rec_col2:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['secondary']}; margin-top: 0;'>Operating Hours</h4><ul style='font-size: 14px;'><li>Extend evening service with light menu</li><li>Staff appropriately for {PEAK_HOUR}:00 rush</li><li>Introduce breakfast deals for 7-9 AM</li></ul></div>""", unsafe_allow_html=True)

with rec_col3:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['dark']}; margin-top: 0;'>Service Efficiency</h4><ul style='font-size: 14px;'><li>Target <20 min fulfillment for all orders</li><li>Investigate causes of {MISSED_ORDERS_COUNT} missed orders</li><li>Pre-prepare high-volume items during peak hours</li></ul></div>""", unsafe_allow_html=True)

with rec_col4:
    st.markdown(f"""<div class='metric-card'><h4 style='color: {COLORS['accent']}; margin-top: 0;'>Customer Retention</h4><ul style='font-size: 14px;'><li>Implement VIP program for high-value customers</li><li>Target churned customers with re-engagement offers</li><li>Upgrade medium-value customers through loyalty rewards</li></ul></div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""<div style='text-align: center; color: #6c757d; font-size: 12px; padding: 2rem 0;'><p>Dashboard generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p><p>Analyzing {TOTAL_ORDERS:,} orders • NPR {TOTAL_REVENUE:,.0f} in revenue</p><p style='margin-top: 1rem;'>Analysis conducted by Data Society for Deerwalk Foods System</p></div>""", unsafe_allow_html=True)
