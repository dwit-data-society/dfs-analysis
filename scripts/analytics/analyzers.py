"""Data analysis functions"""

from itertools import combinations
from typing import Tuple

import pandas as pd
from config import Config


class DataAnalyzer:
    """Handles data analysis operations"""

    @staticmethod
    def calculate_revenue_per_item(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate total revenue per item"""
        revenue = (
            df.groupby(Config.COL_ITEM_NAME)[Config.COL_TOTAL_PRICE].sum().reset_index()
        )
        revenue.columns = ["item", "total_revenue"]
        return revenue

    @staticmethod
    def get_top_n_items(
        revenue_df: pd.DataFrame, n: int = Config.DEFAULT_TOP_N
    ) -> pd.DataFrame:
        """Get top N items by revenue"""
        return revenue_df.sort_values(by="total_revenue", ascending=False).head(n)

    @staticmethod
    def get_status_counts(df: pd.DataFrame) -> pd.Series:
        """Get order status counts"""
        return df[Config.COL_STATUS].value_counts()

    @staticmethod
    def calculate_fulfillment_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Calculate order fulfillment time statistics"""
        if (
            Config.COL_CREATED_AT not in df.columns
            or Config.COL_UPDATED_AT not in df.columns
        ):
            return pd.DataFrame(), {}

        valid_times = df[
            df[Config.COL_CREATED_AT].notna() & df[Config.COL_UPDATED_AT].notna()
        ].copy()

        valid_times["fulfillment_time"] = (
            valid_times[Config.COL_UPDATED_AT] - valid_times[Config.COL_CREATED_AT]
        )
        valid_times["fulfillment_hours"] = (
            valid_times["fulfillment_time"].dt.total_seconds() / 3600
        )
        valid_times = valid_times[valid_times["fulfillment_hours"] >= 0]

        stats = {
            "avg_hours": valid_times["fulfillment_hours"].mean(),
            "median_hours": valid_times["fulfillment_hours"].median(),
            "min_hours": valid_times["fulfillment_hours"].min(),
            "max_hours": valid_times["fulfillment_hours"].max(),
            "total_orders": len(valid_times),
        }

        return valid_times, stats

    @staticmethod
    def calculate_delivery_success_rate(df: pd.DataFrame) -> dict:
        """Calculate delivery success rate"""
        total_orders = len(df)
        delivered_orders = df[
            df[Config.COL_STATUS].str.lower().isin(Config.DELIVERED_STATUSES)
        ].shape[0]
        success_rate = (
            (delivered_orders / total_orders * 100) if total_orders > 0 else 0
        )

        return {
            "total_orders": total_orders,
            "delivered_orders": delivered_orders,
            "success_rate": success_rate,
            "failed_orders": total_orders - delivered_orders,
        }

    @staticmethod
    def analyze_peak_times(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze peak order times by hour and day"""
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()
        valid_orders["hour"] = valid_orders[Config.COL_CREATED_AT].dt.hour
        valid_orders["day_of_week"] = valid_orders[Config.COL_CREATED_AT].dt.day_name()
        valid_orders["day_num"] = valid_orders[Config.COL_CREATED_AT].dt.dayofweek

        hourly = valid_orders.groupby("hour").size().reset_index(name="order_count")
        daily = (
            valid_orders.groupby(["day_num", "day_of_week"])
            .size()
            .reset_index(name="order_count")
        )
        daily = daily.sort_values("day_num")

        return hourly, daily

    @staticmethod
    def analyze_order_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze order volume trends over time"""
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()
        valid_orders["date"] = valid_orders[Config.COL_CREATED_AT].dt.date

        trends = (
            valid_orders.groupby("date")
            .agg({Config.COL_TOTAL_PRICE: ["count", "sum", "mean"]})
            .reset_index()
        )

        trends.columns = ["date", "order_count", "total_revenue", "avg_order_value"]
        trends["date"] = pd.to_datetime(trends["date"])

        return trends

    @staticmethod
    def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze monthly sales trends"""
        if Config.COL_CREATED_AT not in df.columns:
            return pd.DataFrame()

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()
        valid_orders["year_month"] = valid_orders[Config.COL_CREATED_AT].dt.to_period(
            "M"
        )

        monthly = (
            valid_orders.groupby("year_month")
            .agg({Config.COL_TOTAL_PRICE: ["count", "sum", "mean"]})
            .reset_index()
        )

        monthly.columns = [
            "year_month",
            "order_count",
            "total_revenue",
            "avg_order_value",
        ]
        monthly["year_month"] = monthly["year_month"].dt.to_timestamp()

        return monthly

    @staticmethod
    def calculate_revenue_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate month-over-month revenue growth rate"""
        if monthly_df.empty:
            return pd.DataFrame()

        growth_df = monthly_df.copy()
        growth_df["revenue_change"] = growth_df["total_revenue"].diff()
        growth_df["revenue_growth_rate"] = growth_df["total_revenue"].pct_change() * 100
        growth_df["order_count_change"] = growth_df["order_count"].diff()
        growth_df["order_growth_rate"] = growth_df["order_count"].pct_change() * 100

        return growth_df

    @staticmethod
    def analyze_menu_categories(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze menu category performance"""
        if Config.COL_MENU_ID not in df.columns:
            return pd.DataFrame()

        category_stats = (
            df.groupby(Config.COL_MENU_ID)
            .agg(
                {
                    Config.COL_TOTAL_PRICE: ["sum", "mean", "count"],
                    Config.COL_ITEM_NAME: "nunique",
                }
            )
            .reset_index()
        )

        category_stats.columns = [
            "menu_id",
            "total_revenue",
            "avg_order_value",
            "order_count",
            "unique_items",
        ]
        category_stats["revenue_per_item"] = (
            category_stats["total_revenue"] / category_stats["unique_items"]
        )
        category_stats = category_stats.sort_values("total_revenue", ascending=False)

        return category_stats

    @staticmethod
    def find_cross_selling_opportunities(
        df: pd.DataFrame, min_support: int = 3
    ) -> pd.DataFrame:
        """Find items frequently ordered together"""
        if (
            Config.COL_ORDER_ID not in df.columns
            or Config.COL_ITEM_NAME not in df.columns
        ):
            return pd.DataFrame()

        orders_items = (
            df.groupby(Config.COL_ORDER_ID)[Config.COL_ITEM_NAME]
            .apply(list)
            .reset_index()
        )

        pairs = []
        for items in orders_items[Config.COL_ITEM_NAME]:
            if len(items) > 1:
                unique_items = list(set(items))
                for item1, item2 in combinations(sorted(unique_items), 2):
                    pairs.append((item1, item2))

        if not pairs:
            return pd.DataFrame()

        pairs_df = pd.DataFrame(pairs, columns=["item_1", "item_2"])
        pair_counts = (
            pairs_df.groupby(["item_1", "item_2"])
            .size()
            .reset_index(name="co_occurrence_count")
        )
        pair_counts = pair_counts[pair_counts["co_occurrence_count"] >= min_support]

        total_orders = df[Config.COL_ORDER_ID].nunique()
        item_counts = (
            df.groupby(Config.COL_ITEM_NAME)[Config.COL_ORDER_ID].nunique().to_dict()
        )

        pair_counts["item_1_frequency"] = (
            pair_counts["item_1"].map(item_counts) / total_orders
        )
        pair_counts["item_2_frequency"] = (
            pair_counts["item_2"].map(item_counts) / total_orders
        )
        pair_counts["pair_frequency"] = (
            pair_counts["co_occurrence_count"] / total_orders
        )
        pair_counts["lift"] = pair_counts["pair_frequency"] / (
            pair_counts["item_1_frequency"] * pair_counts["item_2_frequency"]
        )
        pair_counts = pair_counts.sort_values(
            ["co_occurrence_count", "lift"], ascending=[False, False]
        )

        return pair_counts

    @staticmethod
    def analyze_menu_item_lifecycle(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Analyze menu items by their lifecycle stage"""
        if (
            Config.COL_CREATED_AT not in df.columns
            or Config.COL_ITEM_NAME not in df.columns
        ):
            return pd.DataFrame(), {}

        valid_orders = df[df[Config.COL_CREATED_AT].notna()].copy()

        item_dates = (
            valid_orders.groupby(Config.COL_ITEM_NAME)[Config.COL_CREATED_AT]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        item_dates.columns = ["item_name", "first_order", "last_order", "total_orders"]

        latest_date = valid_orders[Config.COL_CREATED_AT].max()
        item_dates["days_on_menu"] = (latest_date - item_dates["first_order"]).dt.days
        item_dates["days_since_last_order"] = (
            latest_date - item_dates["last_order"]
        ).dt.days

        item_revenue = (
            valid_orders.groupby(Config.COL_ITEM_NAME)[Config.COL_TOTAL_PRICE]
            .sum()
            .reset_index()
        )
        item_revenue.columns = ["item_name", "total_revenue"]
        item_dates = item_dates.merge(item_revenue, on="item_name")

        def classify_item(row):
            if row["days_on_menu"] <= Config.NEW_ITEM_DAYS:
                return "New"
            elif row["days_on_menu"] <= Config.ESTABLISHED_ITEM_DAYS:
                return "Growing"
            else:
                return "Established"

        item_dates["lifecycle_stage"] = item_dates.apply(classify_item, axis=1)
        item_dates["avg_revenue_per_day"] = (
            item_dates["total_revenue"] / item_dates["days_on_menu"]
        )
        item_dates["orders_per_day"] = (
            item_dates["total_orders"] / item_dates["days_on_menu"]
        )

        summary = {
            "new_items": len(item_dates[item_dates["lifecycle_stage"] == "New"]),
            "growing_items": len(
                item_dates[item_dates["lifecycle_stage"] == "Growing"]
            ),
            "established_items": len(
                item_dates[item_dates["lifecycle_stage"] == "Established"]
            ),
            "total_items": len(item_dates),
        }

        return item_dates, summary
