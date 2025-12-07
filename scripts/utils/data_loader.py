"""Data loading and preprocessing utilities"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing"""

    @staticmethod
    def clean_csv_line(line: str) -> list[str]:
        """
        Clean and parse a CSV line with embedded semicolons.

        Args:
            line: Raw CSV line

        Returns:
            List of cleaned field values
        """
        # Strip outer quotes and whitespace
        line = line.strip().strip('"')

        # Fix known data quality issues
        line = line.replace("Chicken mo;mo", "Chicken momo")

        # Split on semicolons
        parts = line.split(";")

        # Clean each part - remove surrounding quotes and double-quote escapes
        cleaned_parts = [p.strip('"').replace('""', '"') for p in parts]

        return cleaned_parts

    @staticmethod
    def load_orders(file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and parse orders CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with orders data, or None if loading fails
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                st.error(f"Data file not found: {file_path}")
                return None

            rows = []
            with open(file_path, "r", encoding=Config.ENCODING) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        cleaned_parts = DataLoader.clean_csv_line(line)
                        rows.append(cleaned_parts)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

            if len(rows) < 2:
                logger.error("CSV file has insufficient data")
                st.error("CSV file appears to be empty or invalid")
                return None

            # Create DataFrame
            df = pd.DataFrame(rows[1:], columns=rows[0])

            # Validate required columns
            required_cols = [
                Config.COL_ITEM_NAME,
                Config.COL_TOTAL_PRICE,
                Config.COL_STATUS,
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None

            # Convert numeric columns
            df[Config.COL_TOTAL_PRICE] = pd.to_numeric(
                df[Config.COL_TOTAL_PRICE], errors="coerce"
            )

            # Convert datetime columns
            if Config.COL_CREATED_AT in df.columns:
                df[Config.COL_CREATED_AT] = pd.to_datetime(
                    df[Config.COL_CREATED_AT], errors="coerce"
                )

            if Config.COL_UPDATED_AT in df.columns:
                df[Config.COL_UPDATED_AT] = pd.to_datetime(
                    df[Config.COL_UPDATED_AT], errors="coerce"
                )

            # Remove rows with invalid prices
            invalid_prices = df[Config.COL_TOTAL_PRICE].isna().sum()
            if invalid_prices > 0:
                logger.warning(f"Removed {invalid_prices} rows with invalid prices")
                df = df.dropna(subset=[Config.COL_TOTAL_PRICE])

            logger.info(f"Successfully loaded {len(df)} orders")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            st.error(f"Failed to load data: {str(e)}")
            return None
