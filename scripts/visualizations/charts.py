"""Chart visualization functions"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import Colors


class Charts:
    """Chart rendering functions"""

    @staticmethod
    def render_top_items_chart(top_items: pd.DataFrame, n: int):
        """Render top items revenue bar chart"""
        if top_items.empty:
            st.warning("No data available for top items")
            return

        fig = px.bar(
            top_items,
            x="total_revenue",
            y="item",
            orientation="h",
            labels={"total_revenue": "Revenue (रु)", "item": "Menu Item"},
            color="total_revenue",
            color_continuous_scale="Purples",
        )

        fig.update_layout(
            showlegend=False,
            height=400,
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_status_pie_chart(status_counts: pd.Series):
        """Render order status pie chart"""
        if status_counts.empty:
            st.warning("No data available for order status")
            return

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_sequence=px.colors.sequential.Purples_r,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )

        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_fulfillment_histogram(fulfillment_df: pd.DataFrame):
        """Render fulfillment time histogram"""
        if fulfillment_df.empty:
            return

        fig = px.histogram(
            fulfillment_df,
            x="fulfillment_hours",
            nbins=30,
            labels={
                "fulfillment_hours": "Fulfillment Time (hours)",
                "count": "Number of Orders",
            },
            color_discrete_sequence=[Colors.PRIMARY],
        )

        fig.update_layout(
            title="Distribution of Fulfillment Times",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_hourly_orders_chart(hourly: pd.DataFrame):
        """Render hourly orders bar chart"""
        fig = px.bar(
            hourly,
            x="hour",
            y="order_count",
            labels={"hour": "Hour of Day", "order_count": "Number of Orders"},
            color="order_count",
            color_continuous_scale="Purples",
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_daily_orders_chart(daily: pd.DataFrame):
        """Render daily orders bar chart"""
        fig = px.bar(
            daily,
            x="day_of_week",
            y="order_count",
            labels={"day_of_week": "Day of Week", "order_count": "Number of Orders"},
            color="order_count",
            color_continuous_scale="Purples",
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_daily_order_volume(trends: pd.DataFrame):
        """Render daily order volume area chart"""
        fig = px.area(
            trends,
            x="date",
            y="order_count",
            labels={"date": "Date", "order_count": "Number of Orders"},
            color_discrete_sequence=[Colors.PRIMARY],
        )
        fig.update_layout(
            title="Daily Order Volume",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_daily_revenue(trends: pd.DataFrame):
        """Render daily revenue area chart"""
        fig = px.area(
            trends,
            x="date",
            y="total_revenue",
            labels={"date": "Date", "total_revenue": "Revenue (रु)"},
            color_discrete_sequence=[Colors.SECONDARY],
        )
        fig.update_layout(
            title="Daily Revenue",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_avg_order_value(trends: pd.DataFrame):
        """Render average order value line chart"""
        fig = px.line(
            trends,
            x="date",
            y="avg_order_value",
            labels={"date": "Date", "avg_order_value": "Avg Order Value (रु)"},
            color_discrete_sequence=[Colors.SUCCESS],
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(
            title="Average Order Value Trend",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_monthly_revenue_trend(monthly_df: pd.DataFrame):
        """Render monthly revenue trend line chart"""
        fig = px.line(
            monthly_df,
            x="year_month",
            y="total_revenue",
            markers=True,
            labels={"year_month": "Month", "total_revenue": "Revenue (रु)"},
            color_discrete_sequence=[Colors.PRIMARY],
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_revenue_growth_rate(growth_df: pd.DataFrame):
        """Render revenue growth rate bar chart"""
        colors = [
            Colors.SUCCESS if x > 0 else Colors.DANGER
            for x in growth_df["revenue_growth_rate"]
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=growth_df["year_month"],
                y=growth_df["revenue_growth_rate"],
                name="Revenue Growth %",
                marker_color=colors,
            )
        )

        fig.update_layout(
            title="Month-over-Month Revenue Growth Rate",
            xaxis_title="Month",
            yaxis_title="Growth Rate (%)",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_revenue_and_growth_combined(growth_df: pd.DataFrame):
        """Render combined revenue and growth rate chart"""
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=growth_df["year_month"],
                y=growth_df["total_revenue"],
                name="Revenue",
                yaxis="y",
                marker_color=Colors.PRIMARY,
                opacity=0.7,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=growth_df["year_month"],
                y=growth_df["revenue_growth_rate"],
                name="Growth Rate %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=Colors.WARNING, width=3),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Revenue & Growth Rate Combined",
            xaxis_title="Month",
            yaxis=dict(title="Revenue (रु)", side="left"),
            yaxis2=dict(title="Growth Rate (%)", side="right", overlaying="y"),
            hovermode="x unified",
            legend=dict(x=0, y=1.1, orientation="h"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_category_revenue_bar(categories_df: pd.DataFrame):
        """Render category revenue bar chart"""
        fig = px.bar(
            categories_df.head(10),
            x="menu_id",
            y="total_revenue",
            title="Top 10 Categories by Revenue",
            labels={"menu_id": "Category ID", "total_revenue": "Revenue (रु)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_category_scatter(categories_df: pd.DataFrame):
        """Render category items vs revenue scatter plot"""
        fig = px.scatter(
            categories_df,
            x="unique_items",
            y="total_revenue",
            size="order_count",
            hover_data=["menu_id"],
            title="Items vs Revenue (bubble size = orders)",
            labels={"unique_items": "Number of Items", "total_revenue": "Revenue (रु)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_cross_sell_frequency(pairs_df: pd.DataFrame):
        """Render cross-selling frequency bar chart"""
        top_pairs = pairs_df.head(10).copy()
        top_pairs["item_pair"] = top_pairs["item_1"] + " + " + top_pairs["item_2"]

        fig = px.bar(
            top_pairs,
            x="co_occurrence_count",
            y="item_pair",
            orientation="h",
            labels={
                "co_occurrence_count": "Times Ordered Together",
                "item_pair": "Item Pair",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_cross_sell_lift(pairs_df: pd.DataFrame):
        """Render cross-selling lift bar chart"""
        top_lift = pairs_df.nlargest(10, "lift").copy()
        top_lift["item_pair"] = top_lift["item_1"] + " + " + top_lift["item_2"]

        fig = px.bar(
            top_lift,
            x="lift",
            y="item_pair",
            orientation="h",
            labels={"lift": "Lift Score", "item_pair": "Item Pair"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_lifecycle_pie(lifecycle_df: pd.DataFrame):
        """Render lifecycle stage pie chart"""
        stage_counts = lifecycle_df["lifecycle_stage"].value_counts()
        fig = px.pie(
            values=stage_counts.values,
            names=stage_counts.index,
            title="Items by Lifecycle Stage",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_lifecycle_revenue_bar(lifecycle_df: pd.DataFrame):
        """Render lifecycle revenue bar chart"""
        stage_revenue = lifecycle_df.groupby("lifecycle_stage")["total_revenue"].sum()
        fig = px.bar(
            x=stage_revenue.index,
            y=stage_revenue.values,
            labels={"x": "Lifecycle Stage", "y": "Total Revenue (रु)"},
            title="Revenue by Lifecycle Stage",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_lifecycle_scatter(lifecycle_df: pd.DataFrame):
        """Render lifecycle performance scatter plot"""
        fig = px.scatter(
            lifecycle_df,
            x="days_on_menu",
            y="total_revenue",
            color="lifecycle_stage",
            size="total_orders",
            hover_data=["item_name", "orders_per_day"],
            title="Item Performance Over Time",
            labels={
                "days_on_menu": "Days on Menu",
                "total_revenue": "Total Revenue (रु)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
