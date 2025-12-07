"""Table rendering functions"""

import pandas as pd
import streamlit as st


class Tables:
    """Table rendering functions"""

    @staticmethod
    def render_top_items_table(top_items: pd.DataFrame):
        """Render top items data table"""
        display_df = top_items.copy()
        display_df["total_revenue"] = display_df["total_revenue"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_status_breakdown_table(status_counts: pd.Series):
        """Render status breakdown table"""
        status_df = pd.DataFrame(
            {
                "Status": status_counts.index,
                "Count": status_counts.values,
                "Percentage": (status_counts.values / status_counts.sum() * 100).round(
                    2
                ),
            }
        )
        status_df["Percentage"] = status_df["Percentage"].apply(lambda x: f"{x}%")
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_top_months_revenue_table(monthly_df: pd.DataFrame):
        """Render top months by revenue table"""
        top_months = monthly_df.nlargest(5, "total_revenue").copy()
        top_months["year_month"] = top_months["year_month"].dt.strftime("%B %Y")
        top_months["total_revenue"] = top_months["total_revenue"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        st.dataframe(
            top_months[["year_month", "total_revenue", "order_count"]].rename(
                columns={
                    "year_month": "Month",
                    "total_revenue": "Revenue",
                    "order_count": "Orders",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def render_top_months_orders_table(monthly_df: pd.DataFrame):
        """Render top months by order volume table"""
        top_order_months = monthly_df.nlargest(5, "order_count").copy()
        top_order_months["year_month"] = top_order_months["year_month"].dt.strftime(
            "%B %Y"
        )
        top_order_months["total_revenue"] = top_order_months["total_revenue"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        st.dataframe(
            top_order_months[["year_month", "order_count", "total_revenue"]].rename(
                columns={
                    "year_month": "Month",
                    "order_count": "Orders",
                    "total_revenue": "Revenue",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def render_growth_details_table(growth_df: pd.DataFrame):
        """Render monthly growth details table"""
        display_growth = growth_df.copy()
        display_growth["year_month"] = display_growth["year_month"].dt.strftime("%B %Y")
        display_growth["total_revenue"] = display_growth["total_revenue"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        display_growth["revenue_change"] = display_growth["revenue_change"].apply(
            lambda x: f"रु{x:,.2f}" if pd.notna(x) else "N/A"
        )
        display_growth["revenue_growth_rate"] = display_growth[
            "revenue_growth_rate"
        ].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

        st.dataframe(
            display_growth[
                [
                    "year_month",
                    "total_revenue",
                    "order_count",
                    "revenue_change",
                    "revenue_growth_rate",
                ]
            ].rename(
                columns={
                    "year_month": "Month",
                    "total_revenue": "Revenue",
                    "order_count": "Orders",
                    "revenue_change": "Revenue Change",
                    "revenue_growth_rate": "Growth Rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def render_category_details_table(categories_df: pd.DataFrame):
        """Render category details table"""
        display_df = categories_df.copy()
        display_df["total_revenue"] = display_df["total_revenue"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        display_df["avg_order_value"] = display_df["avg_order_value"].apply(
            lambda x: f"रु{x:.2f}"
        )
        display_df["revenue_per_item"] = display_df["revenue_per_item"].apply(
            lambda x: f"रु{x:,.2f}"
        )
        st.dataframe(display_df, use_container_width=True)

    @staticmethod
    def render_cross_sell_pairs_table(pairs_df: pd.DataFrame):
        """Render cross-selling pairs table"""
        display_df = pairs_df.copy()
        display_df["lift"] = display_df["lift"].apply(lambda x: f"{x:.2f}")
        display_df = display_df[["item_1", "item_2", "co_occurrence_count", "lift"]]
        st.dataframe(display_df, use_container_width=True)

    @staticmethod
    def render_lifecycle_stage_table(lifecycle_df: pd.DataFrame, stage: str):
        """Render lifecycle stage items table"""
        stage_items = lifecycle_df[lifecycle_df["lifecycle_stage"] == stage].nlargest(
            10, "total_revenue"
        )
        if not stage_items.empty:
            display_df = stage_items[
                [
                    "item_name",
                    "total_revenue",
                    "total_orders",
                    "days_on_menu",
                    "orders_per_day",
                ]
            ].copy()
            display_df["total_revenue"] = display_df["total_revenue"].apply(
                lambda x: f"रु{x:,.2f}"
            )
            display_df["orders_per_day"] = display_df["orders_per_day"].apply(
                lambda x: f"{x:.2f}"
            )
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info(f"No {stage.lower()} items found")
