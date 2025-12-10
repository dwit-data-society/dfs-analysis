"""Reusable UI components"""

from datetime import datetime

import pandas as pd
import streamlit as st
from config import Config
from visualizations.charts import Charts
from visualizations.tables import Tables


class UIComponents:
    """Reusable UI components for the dashboard"""

    @staticmethod
    def render_header():
        """Render dashboard header"""
        st.markdown(
            '<h1 class="main-title"> Deerwalk Food System Analytics</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="subtitle">Comprehensive insights into food ordering and delivery performance</p>',
            unsafe_allow_html=True,
        )

        current_date = datetime.now().strftime("%B %d, %Y")
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                        border-radius: 10px; margin-bottom: 2rem;'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                     Dashboard Updated: {current_date}
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_metrics(df: pd.DataFrame, delivery_stats: dict):
        """Render key metrics"""
        st.markdown("###  Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Orders",
                f"{len(df):,}",
                help="Total number of orders in the system",
            )

        with col2:
            total_revenue = df[Config.COL_TOTAL_PRICE].sum()
            st.metric(
                "Total Revenue",
                f"रु{total_revenue:,.2f}",
                help="Total revenue generated",
            )

        with col3:
            avg_order = df[Config.COL_TOTAL_PRICE].mean()
            st.metric(
                "Average Order Value",
                f"रु{avg_order:.2f}",
                help="Average value per order",
            )

        with col4:
            success_rate = delivery_stats.get("success_rate", 0)
            st.metric(
                "Delivery Success Rate",
                f"{success_rate:.1f}%",
                delta=f"{success_rate - 100:.1f}%"
                if success_rate < 100
                else "Perfect!",
                help="Percentage of successfully delivered orders",
            )

    @staticmethod
    def render_sidebar(top_n: int, min_support: int):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("##  Dashboard Settings")
            st.markdown("---")

            top_n = st.slider(
                " Top items to display",
                min_value=5,
                max_value=20,
                value=top_n,
                step=1,
            )

            min_support = st.slider(
                " Min co-occurrences",
                min_value=2,
                max_value=10,
                value=min_support,
                step=1,
                help="Minimum times items must be ordered together for cross-selling analysis",
            )

            st.markdown("---")

            if st.button(" Refresh Data"):
                st.cache_data.clear()
                st.rerun()

            st.markdown("---")
            st.markdown(
                """
                <div style='padding: 1rem; background: #f8f9fa; border-radius: 5px; margin-top: 2rem;'>
                    <h4 style='margin: 0 0 0.5rem 0; font-size: 0.9rem;'>📌 About</h4>
                    <p style='margin: 0; font-size: 0.8rem; color: #666;'>
                        Deerwalk Food System Analytics Dashboard provides comprehensive insights
                        into ordering patterns, revenue trends, and operational metrics.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        return top_n, min_support

    @staticmethod
    def render_footer():
        """Render dashboard footer"""
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                        border-radius: 10px; margin-top: 3rem;'>
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                     Deerwalk Food System Analytics Dashboard | Built with Streamlit & Python
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_top_items_section(top_items: pd.DataFrame, n: int):
        """Render top items section with chart and table"""
        st.markdown(f"###  Top {n} Items by Revenue")
        Charts.render_top_items_chart(top_items, n)

        with st.expander(" View Data Table"):
            Tables.render_top_items_table(top_items)

    @staticmethod
    def render_status_section(status_counts: pd.Series):
        """Render status section with chart and table"""
        st.markdown("###  Order Status Distribution")
        Charts.render_status_pie_chart(status_counts)

        with st.expander(" View Status Breakdown"):
            Tables.render_status_breakdown_table(status_counts)

    @staticmethod
    def render_fulfillment_section(fulfillment_df: pd.DataFrame, stats: dict):
        """Render fulfillment analysis section"""
        st.markdown("### Order Fulfillment Time Analysis")

        if fulfillment_df.empty or not stats:
            st.info(
                " Fulfillment time data not available. Requires 'created_at' and 'updated_at' columns."
            )
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" Avg Fulfillment", f"{stats['avg_hours']:.1f} hrs")
        with col2:
            st.metric(" Median Time", f"{stats['median_hours']:.1f} hrs")
        with col3:
            st.metric(" Fastest", f"{stats['min_hours']:.1f} hrs")
        with col4:
            st.metric(" Slowest", f"{stats['max_hours']:.1f} hrs")

        Charts.render_fulfillment_histogram(fulfillment_df)

    @staticmethod
    def render_peak_times_section(hourly: pd.DataFrame, daily: pd.DataFrame):
        """Render peak times section"""
        st.markdown("###  Peak Order Times")

        if hourly.empty or daily.empty:
            st.info(" Peak times data not available. Requires 'created_at' column.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Orders by Hour of Day**")
            Charts.render_hourly_orders_chart(hourly)
        with col2:
            st.markdown("**Orders by Day of Week**")
            Charts.render_daily_orders_chart(daily)

    @staticmethod
    def render_trends_section(trends: pd.DataFrame):
        """Render trends section"""
        st.markdown("###  Daily Trends")

        if trends.empty:
            st.info(" Trends data not available. Requires 'created_at' column.")
            return

        Charts.render_daily_order_volume(trends)
        Charts.render_daily_revenue(trends)
        Charts.render_avg_order_value(trends)

        with st.expander(" View Trend Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Daily Orders", f"{trends['order_count'].mean():.0f}")
            with col2:
                st.metric("Peak Day Orders", f"{trends['order_count'].max():.0f}")
            with col3:
                st.metric(
                    "Avg Daily Revenue", f"रु{trends['total_revenue'].mean():,.2f}"
                )

    @staticmethod
    def render_sales_revenue_section(monthly_df: pd.DataFrame, growth_df: pd.DataFrame):
        """Render sales & revenue analysis section"""
        st.markdown("##  Sales & Revenue Analysis")

        if monthly_df.empty:
            st.info(" Monthly sales data not available. Requires 'created_at' column.")
            return

        st.markdown("### Monthly Revenue Trend")
        Charts.render_monthly_revenue_trend(monthly_df)

        st.markdown("###  Peak Sales Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Months by Revenue**")
            Tables.render_top_months_revenue_table(monthly_df)
        with col2:
            st.markdown("**Top 5 Months by Order Volume**")
            Tables.render_top_months_orders_table(monthly_df)

        if not growth_df.empty:
            st.markdown("###  Revenue Growth Rate")

            recent_growth = (
                growth_df["revenue_growth_rate"].iloc[-1] if len(growth_df) > 1 else 0
            )
            avg_growth = growth_df["revenue_growth_rate"].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Latest Month Growth",
                    f"{recent_growth:+.1f}%",
                    delta=f"{recent_growth:.1f}%",
                )
            with col2:
                st.metric("Average Growth Rate", f"{avg_growth:+.1f}%")
            with col3:
                positive_months = (growth_df["revenue_growth_rate"] > 0).sum()
                total_months = len(growth_df) - 1
                st.metric("Positive Growth Months", f"{positive_months}/{total_months}")

            Charts.render_revenue_growth_rate(growth_df)
            Charts.render_revenue_and_growth_combined(growth_df)

            with st.expander(" View Monthly Growth Details"):
                Tables.render_growth_details_table(growth_df)

    @staticmethod
    def render_menu_categories_section(categories_df: pd.DataFrame):
        """Render menu categories section"""
        st.subheader(" Menu Category Performance")

        if categories_df.empty:
            st.info("Category data not available. Requires 'menu_id' column.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Categories", f"{len(categories_df)}")
        with col2:
            top_category_revenue = categories_df.iloc[0]["total_revenue"]
            st.metric("Top Category Revenue", f"रु{top_category_revenue:,.2f}")
        with col3:
            avg_category_revenue = categories_df["total_revenue"].mean()
            st.metric("Avg Category Revenue", f"रु{avg_category_revenue:,.2f}")

        col1, col2 = st.columns(2)
        with col1:
            Charts.render_category_revenue_bar(categories_df)
        with col2:
            Charts.render_category_scatter(categories_df)

        with st.expander("View Category Details"):
            Tables.render_category_details_table(categories_df)

    @staticmethod
    def render_cross_selling_section(pairs_df: pd.DataFrame):
        """Render cross-selling section"""
        st.subheader(" Cross-Selling Opportunities")

        if pairs_df.empty:
            st.info(
                "Cross-selling data not available. Requires 'order_id' column and multiple items per order."
            )
            return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Frequently Ordered Together**")
            Charts.render_cross_sell_frequency(pairs_df)
        with col2:
            st.markdown("**Highest Lift (Strongest Association)**")
            Charts.render_cross_sell_lift(pairs_df)

        st.markdown("""
        **Understanding the metrics:**
        - **Co-occurrence Count**: How many times these items were ordered together
        - **Lift**: How much more likely items are ordered together vs independently (>1 = positive correlation)
        """)

        with st.expander("View All Cross-Selling Pairs"):
            Tables.render_cross_sell_pairs_table(pairs_df)

    @staticmethod
    def render_lifecycle_section(lifecycle_df: pd.DataFrame, summary: dict):
        """Render menu lifecycle section"""
        st.subheader(" Menu Item Lifecycle Analysis")

        if lifecycle_df.empty or not summary:
            st.info("Lifecycle data not available. Requires 'created_at' column.")
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "New Items",
                summary["new_items"],
                help=f"On menu ≤ {Config.NEW_ITEM_DAYS} days",
            )
        with col2:
            st.metric(
                "Growing Items",
                summary["growing_items"],
                help=f"{Config.NEW_ITEM_DAYS}-{Config.ESTABLISHED_ITEM_DAYS} days",
            )
        with col3:
            st.metric(
                "Established Items",
                summary["established_items"],
                help=f"> {Config.ESTABLISHED_ITEM_DAYS} days",
            )
        with col4:
            st.metric("Total Items", summary["total_items"])

        col1, col2 = st.columns(2)
        with col1:
            Charts.render_lifecycle_pie(lifecycle_df)
        with col2:
            Charts.render_lifecycle_revenue_bar(lifecycle_df)

        Charts.render_lifecycle_scatter(lifecycle_df)

        st.markdown("**Top Performers by Lifecycle Stage**")
        tab1, tab2, tab3 = st.tabs(["New Items", "Growing Items", "Established Items"])

        with tab1:
            Tables.render_lifecycle_stage_table(lifecycle_df, "New")
        with tab2:
            Tables.render_lifecycle_stage_table(lifecycle_df, "Growing")
        with tab3:
            Tables.render_lifecycle_stage_table(lifecycle_df, "Established")
