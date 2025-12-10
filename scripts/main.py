"""
Ok.
so this is a test file to test the submodules.
you can either delete the app.py or migrate this code to app.py.
I leave this to you r reliable hands Arun
Deerwalk Food System Analytics Dashboard
Main application entry point
"""

import logging

import streamlit as st
from analytics.analyzers import DataAnalyzer
from config import Config
from UI.components import UIComponents
from UI.styles import CUSTOM_CSS
from utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Deerwalk Food System Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application entry point"""

    # Render header
    UIComponents.render_header()

    # Sidebar controls
    top_n, min_support = UIComponents.render_sidebar(
        top_n=Config.DEFAULT_TOP_N, min_support=3
    )

    # Load data
    with st.spinner("Loading data..."):
        orders_df = DataLoader.load_orders(Config.DATA_PATH)

    if orders_df is None or orders_df.empty:
        st.error(
            "Unable to load orders data. Please check the data file and try again."
        )
        st.stop()

    # Analyze data
    revenue_per_item = DataAnalyzer.calculate_revenue_per_item(orders_df)
    top_items = DataAnalyzer.get_top_n_items(revenue_per_item, n=top_n)
    status_counts = DataAnalyzer.get_status_counts(orders_df)
    delivery_stats = DataAnalyzer.calculate_delivery_success_rate(orders_df)
    fulfillment_df, fulfillment_stats = DataAnalyzer.calculate_fulfillment_time(
        orders_df
    )
    hourly_orders, daily_orders = DataAnalyzer.analyze_peak_times(orders_df)
    trends = DataAnalyzer.analyze_order_trends(orders_df)
    monthly_trends = DataAnalyzer.analyze_monthly_trends(orders_df)
    growth_df = DataAnalyzer.calculate_revenue_growth(monthly_trends)
    categories_df = DataAnalyzer.analyze_menu_categories(orders_df)
    cross_sell_df = DataAnalyzer.find_cross_selling_opportunities(
        orders_df, min_support=min_support
    )
    lifecycle_df, lifecycle_summary = DataAnalyzer.analyze_menu_item_lifecycle(
        orders_df
    )

    # Display KPIs
    UIComponents.render_metrics(orders_df, delivery_stats)
    st.markdown("<br>", unsafe_allow_html=True)

    # Sales & Revenue Analysis
    UIComponents.render_sales_revenue_section(monthly_trends, growth_df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue & Status Overview
    st.markdown("## Revenue & Status Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        UIComponents.render_top_items_section(top_items, top_n)
    with col2:
        UIComponents.render_status_section(status_counts)
    st.markdown("<br>", unsafe_allow_html=True)

    # Operational Metrics
    st.markdown("## Operational Metrics")
    UIComponents.render_fulfillment_section(fulfillment_df, fulfillment_stats)
    st.markdown("<br>", unsafe_allow_html=True)

    UIComponents.render_peak_times_section(hourly_orders, daily_orders)
    st.markdown("<br>", unsafe_allow_html=True)

    UIComponents.render_trends_section(trends)
    st.markdown("<br>", unsafe_allow_html=True)

    # Menu Intelligence
    st.markdown("## Menu Intelligence")
    UIComponents.render_menu_categories_section(categories_df)
    st.markdown("<br>", unsafe_allow_html=True)

    UIComponents.render_cross_selling_section(cross_sell_df)
    st.markdown("<br>", unsafe_allow_html=True)

    UIComponents.render_lifecycle_section(lifecycle_df, lifecycle_summary)

    # Footer
    UIComponents.render_footer()


if __name__ == "__main__":
    main()
