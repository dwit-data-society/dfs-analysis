"""
Run this script locally to extract all computed values for hardcoding.
Run: python hardcoded_values.py > computed_values.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Adjust this path to match your data location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "orders.csv"

def load_data():
    """Load and preprocess orders data"""
    try:
        with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        lines = content.split('\n')
        header_line = lines[0].strip()
        if header_line.startswith('"') and header_line.endswith('"'):
            header_line = header_line[1:-1]
        header_line = header_line.replace('""', '"')
        columns = [col.strip().replace('"', '') for col in header_line.split(';')]

        data_lines = [l.strip() for l in lines[1:] if l.strip()]
        data = []
        for line in data_lines:
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            values = [v.strip().replace('"', '') for v in line.split(';')]
            if len(values) == len(columns):
                data.append(values)

        orders = pd.DataFrame(data, columns=columns)
        del data

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

    for col in orders.columns:
        orders[col] = orders[col].replace(['NULL', 'null', 'Null', 'None', 'none'], np.nan)

    critical_columns = ['created_at', 'updated_at', 'deleted_at', 'item_name', 'status', 'total_price', 'quantity', 'id']
    for col in critical_columns:
        if col not in orders.columns:
            if col == 'deleted_at':
                orders[col] = np.nan
            elif col == 'status':
                orders[col] = 'Unknown'

    mask = orders['deleted_at'].isna()
    orders = orders[mask].copy()

    orders['created_at'] = pd.to_datetime(orders['created_at'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    orders['updated_at'] = pd.to_datetime(orders['updated_at'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    orders['total_price'] = pd.to_numeric(orders['total_price'], errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce')

    orders = orders.dropna(subset=['created_at', 'updated_at', 'total_price'], how='any')

    orders['fulfillment_mins'] = (orders['updated_at'] - orders['created_at']).dt.total_seconds() / 60
    orders['date'] = orders['created_at'].dt.date
    orders['hour'] = orders['created_at'].dt.hour
    orders['day_of_week'] = orders['created_at'].dt.day_name()
    orders['month'] = orders['created_at'].dt.to_period('M').astype(str)
    orders['week'] = orders['created_at'].dt.to_period('W').astype(str)

    orders['price_category'] = pd.cut(
        orders['total_price'],
        bins=[0, 50, 100, 200, float('inf')],
        labels=['Budget (<50)', 'Mid (50-100)', 'Premium (100-200)', 'Luxury (200+)'],
        include_lowest=True
    )

    return orders

# Load data
print("Loading data...")
df = load_data()

if df is None or df.empty:
    print("ERROR: Could not load data!")
    exit(1)

print(f"Loaded {len(df)} orders successfully!\n")
print("="*80)
print("# COPY EVERYTHING BELOW THIS LINE INTO hardcoded_values.py")
print("="*80)
print()

print(f"# Generated on: {datetime.now()}")
print(f"# Total records processed: {len(df)}")
print()

print("# ===== BASIC METRICS =====")
print(f"TOTAL_REVENUE = {df['total_price'].sum()}")
print(f"TOTAL_ORDERS = {len(df)}")
print(f"AVG_ORDER_VALUE = {df['total_price'].mean()}")
print(f"DELIVERY_RATE = {(df['status'] == 'Delivered').sum() / len(df) * 100}")
print()

print("# ===== MISSED ORDERS =====")
missed = df[df['status'] == 'Missed']
print(f"MISSED_ORDERS_COUNT = {len(missed)}")
print(f"MISSED_ORDERS_PCT = {len(missed) / len(df) * 100}")
print(f"MISSED_REVENUE = {missed['total_price'].sum()}")
print()

print("# ===== DAILY REVENUE =====")
daily_revenue = df.groupby('date').agg({
    'total_price': 'sum',
    'id': 'count'
}).reset_index()
daily_revenue.columns = ['date', 'revenue', 'orders']
daily_revenue['ma7'] = daily_revenue['revenue'].rolling(window=7, min_periods=1).mean()

print(f"DAILY_REVENUE_DATES = {[str(d) for d in daily_revenue['date'].tolist()]}")
print(f"DAILY_REVENUE_VALUES = {daily_revenue['revenue'].round(2).tolist()}")
print(f"DAILY_REVENUE_MA7 = {daily_revenue['ma7'].round(2).tolist()}")
print(f"DAILY_REVENUE_ORDERS = {daily_revenue['orders'].tolist()}")
print()

# Growth rate
if len(daily_revenue) > 7:
    recent_avg = daily_revenue.tail(7)['revenue'].mean()
    previous_avg = daily_revenue.iloc[-14:-7]['revenue'].mean() if len(daily_revenue) > 14 else daily_revenue.head(7)['revenue'].mean()
    growth_rate = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
    print(f"GROWTH_RATE = {growth_rate}")
    print()

print("# ===== TOP ITEMS =====")
top_items = df.groupby('item_name').agg({
    'total_price': 'sum',
    'id': 'count'
}).reset_index()
top_items.columns = ['item_name', 'revenue', 'orders']
top_items = top_items.sort_values('revenue', ascending=False).head(10)
print(f"TOP_ITEMS_NAMES = {top_items['item_name'].tolist()}")
print(f"TOP_ITEMS_REVENUE = {top_items['revenue'].round(2).tolist()}")
print(f"TOP_ITEMS_ORDERS = {top_items['orders'].tolist()}")
print()

print("# ===== BOTTOM ITEMS =====")
bottom_items = df.groupby('item_name').agg({
    'total_price': 'sum',
    'id': 'count'
}).reset_index()
bottom_items.columns = ['item_name', 'revenue', 'orders']
bottom_items = bottom_items[bottom_items['orders'] >= 5].sort_values('revenue', ascending=True).head(10)
print(f"BOTTOM_ITEMS_NAMES = {bottom_items['item_name'].tolist()}")
print(f"BOTTOM_ITEMS_REVENUE = {bottom_items['revenue'].round(2).tolist()}")
print(f"BOTTOM_ITEMS_ORDERS = {bottom_items['orders'].tolist()}")
print()

print("# ===== HOURLY PATTERNS =====")
hourly = df.groupby('hour').agg({
    'total_price': 'sum',
    'id': 'count'
}).reset_index()
hourly.columns = ['hour', 'revenue', 'orders']
print(f"HOURLY_HOURS = {hourly['hour'].tolist()}")
print(f"HOURLY_REVENUE = {hourly['revenue'].round(2).tolist()}")
print(f"HOURLY_ORDERS = {hourly['orders'].tolist()}")
print(f"PEAK_HOUR = {int(hourly.loc[hourly['orders'].idxmax(), 'hour'])}")
print(f"PEAK_HOUR_ORDERS = {int(hourly.loc[hourly['orders'].idxmax(), 'orders'])}")
print()

print("# ===== DAILY PATTERNS =====")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = df.groupby('day_of_week').agg({
    'total_price': 'sum',
    'id': 'count'
}).reset_index()
daily.columns = ['day', 'revenue', 'orders']
daily['day'] = pd.Categorical(daily['day'], categories=day_order, ordered=True)
daily = daily.sort_values('day')
print(f"DAILY_DAYS = {[str(d) for d in daily['day'].tolist()]}")
print(f"DAILY_REVENUE = {daily['revenue'].round(2).tolist()}")
print(f"DAILY_ORDERS = {daily['orders'].tolist()}")
print()

print("# ===== FULFILLMENT TIME =====")
delivered = df[df['status'] == 'Delivered']
if len(delivered) > 0:
    # Sample 1000 fulfillment times for the histogram (to keep data size manageable)
    sample_size = min(1000, len(delivered))
    fulfillment_sample = delivered['fulfillment_mins'].sample(sample_size, random_state=42).tolist()
    print(f"FULFILLMENT_TIMES_SAMPLE = {[round(x, 2) for x in fulfillment_sample]}")
    print(f"AVG_FULFILLMENT = {delivered['fulfillment_mins'].mean()}")
    print(f"MEDIAN_FULFILLMENT = {delivered['fulfillment_mins'].median()}")
    print(f"SLOW_ORDERS = {len(delivered[delivered['fulfillment_mins'] > 30])}")
    print(f"SLOW_ORDERS_PCT = {len(delivered[delivered['fulfillment_mins'] > 30]) / len(delivered) * 100}")
print()

print("# ===== SUCCESS RATE BY HOUR =====")
hourly_success = df.groupby('hour').agg({
    'id': 'count',
    'status': lambda x: (x == 'Delivered').sum()
}).reset_index()
hourly_success.columns = ['hour', 'total', 'delivered']
hourly_success['success_rate'] = (hourly_success['delivered'] / hourly_success['total'] * 100)
print(f"SUCCESS_RATE_HOURS = {hourly_success['hour'].tolist()}")
print(f"SUCCESS_RATE_VALUES = {hourly_success['success_rate'].round(2).tolist()}")
print()

print("# ===== PRICE CATEGORIES =====")
price_dist = df.groupby('price_category', observed=True).agg({
    'id': 'count',
    'total_price': 'sum'
}).reset_index()
price_dist.columns = ['category', 'orders', 'revenue']
print(f"PRICE_CATEGORIES = {[str(c) for c in price_dist['category'].tolist()]}")
print(f"PRICE_ORDERS = {price_dist['orders'].tolist()}")
print(f"PRICE_REVENUE = {price_dist['revenue'].round(2).tolist()}")
print()

print("# ===== MONTHLY PRICE TRENDS =====")
monthly_price = df.groupby(['month', 'price_category'], observed=True).agg({
    'total_price': 'sum'
}).reset_index()
print(f"MONTHLY_PRICE_MONTHS = {monthly_price['month'].unique().tolist()}")
for cat in price_dist['category']:
    cat_data = monthly_price[monthly_price['price_category'] == cat]
    print(f"MONTHLY_PRICE_{str(cat).upper().replace(' ', '_').replace('(', '').replace(')', '').replace('<', 'UNDER').replace('-', '_')} = {cat_data['total_price'].round(2).tolist()}")
print()

print("# ===== CHURN METRICS =====")
if 'user_wallet_id' in df.columns:
    max_date = df['created_at'].max()

    # 30-day churn
    period_start = max_date - timedelta(days=30)
    pre_period = df[df['created_at'] < period_start]
    active_before = set(pre_period['user_wallet_id'].unique())
    during_period = df[(df['created_at'] >= period_start) & (df['created_at'] <= max_date)]
    active_during = set(during_period['user_wallet_id'].unique())
    lost_customers = active_before - active_during
    churn_30 = (len(lost_customers) / len(active_before) * 100) if len(active_before) > 0 else 0

    print(f"CHURN_30_RATE = {churn_30}")
    print(f"CHURN_30_LOST = {len(lost_customers)}")
    print(f"CHURN_30_TOTAL = {len(active_before)}")

    # 60-day churn
    period_start_60 = max_date - timedelta(days=60)
    pre_period_60 = df[df['created_at'] < period_start_60]
    active_before_60 = set(pre_period_60['user_wallet_id'].unique())
    during_period_60 = df[(df['created_at'] >= period_start_60) & (df['created_at'] <= max_date)]
    active_during_60 = set(during_period_60['user_wallet_id'].unique())
    lost_customers_60 = active_before_60 - active_during_60
    churn_60 = (len(lost_customers_60) / len(active_before_60) * 100) if len(active_before_60) > 0 else 0

    print(f"CHURN_60_RATE = {churn_60}")
    print(f"CHURN_60_LOST = {len(lost_customers_60)}")
    print(f"CHURN_60_TOTAL = {len(active_before_60)}")

    # Weekly activity
    customer_activity = df.groupby(df['created_at'].dt.to_period('W'), observed=True)['user_wallet_id'].nunique().reset_index()
    customer_activity.columns = ['week', 'active_customers']
    customer_activity['week'] = customer_activity['week'].astype(str)
    print(f"ACTIVITY_WEEKS = {customer_activity['week'].tolist()}")
    print(f"ACTIVITY_CUSTOMERS = {customer_activity['active_customers'].tolist()}")
print()

print("# ===== CANCELLATION METRICS =====")
cancelled_statuses = ['Cancelled', 'Canceled', 'cancelled', 'canceled', 'Missed']
is_cancelled = df['status'].isin(cancelled_statuses)
cancel_rate = (is_cancelled.sum() / len(df) * 100)
print(f"CANCEL_RATE = {cancel_rate}")
print(f"CANCEL_COUNT = {int(is_cancelled.sum())}")
print(f"CANCEL_LOST_REVENUE = {df[is_cancelled]['total_price'].sum()}")
print()

print("# ===== CANCELLATION BY HOUR =====")
hourly_cancel = df.groupby('hour', observed=True).agg({
    'id': 'count',
    'status': lambda x: x.isin(cancelled_statuses).sum()
}).reset_index()
hourly_cancel.columns = ['hour', 'total', 'cancelled']
hourly_cancel['cancel_rate'] = (hourly_cancel['cancelled'] / hourly_cancel['total'] * 100)
print(f"CANCEL_HOURS = {hourly_cancel['hour'].tolist()}")
print(f"CANCEL_RATES = {hourly_cancel['cancel_rate'].round(2).tolist()}")
print(f"PEAK_CANCEL_HOUR = {int(hourly_cancel.loc[hourly_cancel['cancel_rate'].idxmax(), 'hour'])}")
print(f"PEAK_CANCEL_RATE = {hourly_cancel['cancel_rate'].max()}")
print()

print("# ===== CANCELLATION BY ITEM =====")
item_cancel = df.groupby('item_name', observed=True).agg({
    'id': 'count',
    'status': lambda x: x.isin(cancelled_statuses).sum()
}).reset_index()
item_cancel.columns = ['item_name', 'total', 'cancelled']
item_cancel['cancel_rate'] = (item_cancel['cancelled'] / item_cancel['total'] * 100)
item_cancel = item_cancel[item_cancel['total'] >= 5].sort_values('cancel_rate', ascending=False).head(10)
print(f"CANCEL_ITEMS = {item_cancel['item_name'].tolist()}")
print(f"CANCEL_ITEMS_RATES = {item_cancel['cancel_rate'].round(2).tolist()}")
print(f"CANCEL_ITEMS_TOTAL = {item_cancel['total'].tolist()}")
print()

print("# ===== REVENUE CONCENTRATION =====")
if 'user_wallet_id' in df.columns:
    customer_revenue = df.groupby('user_wallet_id')['total_price'].sum().reset_index()
    customer_revenue.columns = ['customer_id', 'revenue']
    customer_revenue = customer_revenue.sort_values('revenue', ascending=True)

    # Gini coefficient
    n = len(customer_revenue)
    revenue_sorted = customer_revenue['revenue'].values
    gini = (2 * np.sum((np.arange(1, n + 1) * revenue_sorted))) / (n * np.sum(revenue_sorted)) - (n + 1) / n

    total_revenue = customer_revenue['revenue'].sum()
    top_10_pct = int(np.ceil(len(customer_revenue) * 0.1))
    top_20_pct = int(np.ceil(len(customer_revenue) * 0.2))

    top_10_revenue = customer_revenue.nlargest(top_10_pct, 'revenue')['revenue'].sum()
    top_20_revenue = customer_revenue.nlargest(top_20_pct, 'revenue')['revenue'].sum()

    top_10_contribution = (top_10_revenue / total_revenue * 100)
    top_20_contribution = (top_20_revenue / total_revenue * 100)

    print(f"GINI_COEFFICIENT = {gini}")
    print(f"TOP_10_CONTRIBUTION = {top_10_contribution}")
    print(f"TOP_20_CONTRIBUTION = {top_20_contribution}")
    print(f"TOTAL_CUSTOMERS = {len(customer_revenue)}")

    # Customer tiers
    customer_revenue['tier'] = pd.cut(
        customer_revenue['revenue'],
        bins=[0, customer_revenue['revenue'].quantile(0.5),
              customer_revenue['revenue'].quantile(0.8),
              customer_revenue['revenue'].quantile(0.95),
              float('inf')],
        labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
    )

    tier_stats = customer_revenue.groupby('tier', observed=True).agg({
        'customer_id': 'count',
        'revenue': 'sum'
    }).reset_index()
    tier_stats.columns = ['tier', 'customers', 'revenue']
    tier_stats['revenue_pct'] = (tier_stats['revenue'] / total_revenue * 100)
    tier_stats['customer_pct'] = (tier_stats['customers'] / len(customer_revenue) * 100)

    print(f"TIER_NAMES = {[str(t) for t in tier_stats['tier'].tolist()]}")
    print(f"TIER_CUSTOMERS = {tier_stats['customers'].tolist()}")
    print(f"TIER_REVENUE_PCT = {tier_stats['revenue_pct'].round(2).tolist()}")
    print(f"TIER_CUSTOMER_PCT = {tier_stats['customer_pct'].round(2).tolist()}")

    # Lorenz curve data (sample 100 points for visualization)
    sample_indices = np.linspace(0, len(customer_revenue)-1, min(100, len(customer_revenue)), dtype=int)
    lorenz_sample = customer_revenue.iloc[sample_indices].copy()
    lorenz_sample['cumulative_customers'] = np.arange(1, len(lorenz_sample) + 1) / len(customer_revenue) * 100
    lorenz_sample['cumulative_revenue'] = lorenz_sample['revenue'].cumsum() / total_revenue * 100

    print(f"LORENZ_CUSTOMERS = {lorenz_sample['cumulative_customers'].round(2).tolist()}")
    print(f"LORENZ_REVENUE = {lorenz_sample['cumulative_revenue'].round(2).tolist()}")

# ===== FULFILLMENT TIME HISTOGRAM DATA =====
delivered_orders = df[df['status'] == 'Delivered']
hist_counts, hist_bins = np.histogram(delivered_orders['fulfillment_mins'], bins=30)
print("\n# ===== FULFILLMENT HISTOGRAM =====")
print("FULFILLMENT_HIST_BINS =", hist_bins.tolist())
print("FULFILLMENT_HIST_COUNTS =", hist_counts.tolist())

# ===== LORENZ CURVE COMPLETE DATA =====
customer_revenue = df.groupby('user_wallet_id')['total_price'].sum().reset_index()
customer_revenue = customer_revenue.sort_values('total_price', ascending=True)
customer_revenue['cumulative_customers'] = np.arange(1, len(customer_revenue) + 1) / len(customer_revenue) * 100
customer_revenue['cumulative_revenue'] = customer_revenue['total_price'].cumsum() / customer_revenue['total_price'].sum() * 100

# Sample every Nth point to get ~200-300 points for smooth curve
step = max(1, len(customer_revenue) // 200)
lorenz_customers = customer_revenue['cumulative_customers'].iloc[::step].tolist()
lorenz_revenue = customer_revenue['cumulative_revenue'].iloc[::step].tolist()

print("\n# ===== LORENZ CURVE COMPLETE =====")
print("LORENZ_CUSTOMERS_FULL =", lorenz_customers)
print("LORENZ_REVENUE_FULL =", lorenz_revenue)

print()
print("="*80)
print("# END OF HARDCODED VALUES")
print("="*80)
