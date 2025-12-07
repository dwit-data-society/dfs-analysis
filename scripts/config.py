"""Configuration settings for the Deerwalk Food System Analytics Dashboard"""

import os
from pathlib import Path


class Config:
    """Application configuration"""

    # Data settings - try multiple possible locations
    _possible_paths = [
        Path("data/orders.csv"),  # Same level as main.py
        Path("../data/orders.csv"),  # One level up
        Path("./data/orders.csv"),  # Current directory
        Path(__file__).parent / "data" / "orders.csv",  # Relative to config.py
    ]

    # Find the first path that exists
    DATA_PATH = None
    for path in _possible_paths:
        if path.exists():
            DATA_PATH = path
            break

    # If none found, default to the first one
    if DATA_PATH is None:
        DATA_PATH = _possible_paths[0]

    ENCODING = "utf-8-sig"

    # Display settings
    DEFAULT_TOP_N = 10

    # Column names
    COL_ITEM_NAME = "item_name"
    COL_TOTAL_PRICE = "total_price"
    COL_STATUS = "status"
    COL_CREATED_AT = "created_at"
    COL_UPDATED_AT = "updated_at"
    COL_MENU_ID = "menu_id"
    COL_ORDER_ID = "order_id"

    # Status indicating successful delivery
    DELIVERED_STATUSES = ["delivered", "completed", "success"]

    # Menu lifecycle thresholds (days)
    NEW_ITEM_DAYS = 30
    ESTABLISHED_ITEM_DAYS = 90


class Colors:
    """Color palette for consistent styling"""

    PRIMARY = "#667eea"
    SECONDARY = "#764ba2"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    INFO = "#3b82f6"
