"""
Retail Sales Study - Initial Data Inspection
Author: B.Tech Data Science Student
Description: This script performs initial exploratory inspection on the 
             raw Superstore sales dataset (data/superstore_sales.csv) without
             modifying the underlying data.
"""

import os
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("RETAIL SALES STUDY - INITIAL DATA INSPECTION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Load data/superstore_sales.csv
    # -------------------------------------------------------------
    # Robust path handling to allow running from project root or src/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "superstore_sales.csv")

    if not os.path.exists(file_path):
        # Fallback to direct relative path if run from root
        file_path = os.path.join("data", "superstore_sales.csv")

    print(f"\n[1] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print("Dataset successfully loaded into memory.")

    # -------------------------------------------------------------
    # 2. Display dataset shape
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2] DATASET SHAPE")
    print("=" * 60)
    rows, cols = df.shape
    print(f"Total Rows (Records) : {rows:,}")
    print(f"Total Columns        : {cols}")

    # -------------------------------------------------------------
    # 3. Display column names
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3] COLUMN NAMES")
    print("=" * 60)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")

    # -------------------------------------------------------------
    # 4. Display first 5 rows
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[4] FIRST 5 ROWS")
    print("=" * 60)
    # Set display options to show all columns clearly in console
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(5))

    # -------------------------------------------------------------
    # 5. Display data types
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5] DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    # -------------------------------------------------------------
    # 6. Display missing values
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[6] MISSING VALUES (NULL COUNT)")
    print("=" * 60)
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage (%)': missing_percent.round(2)
    })
    print(missing_summary)

    # -------------------------------------------------------------
    # 7. Display duplicate count
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[7] DUPLICATE ROWS COUNT")
    print("=" * 60)
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")

    # -------------------------------------------------------------
    # 8. Display descriptive statistics
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[8] DESCRIPTIVE STATISTICS (NUMERICAL)")
    print("=" * 60)
    print(df.describe())

    print("\n" + "-" * 60)
    print("[8b] SUMMARY STATISTICS (CATEGORICAL / OBJECT COLUMNS)")
    print("-" * 60)
    print(df.describe(include=['object', 'string']))

    # -------------------------------------------------------------
    # 9. Check unique values of Category, Sub-Category, Region, Segment
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[9] UNIQUE VALUES OF KEY DIMENSIONS")
    print("=" * 60)
    target_columns = ["Category", "Sub-Category", "Region", "Segment"]
    for col in target_columns:
        if col in df.columns:
            unique_vals = df[col].unique()
            print(f"\nDimension: '{col}' ({len(unique_vals)} unique values)")
            print(f"  Values: {list(unique_vals)}")
        else:
            print(f"\nDimension: '{col}' not found in dataset.")

    # -------------------------------------------------------------
    # 10. Check the minimum and maximum Order Date
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[10] ORDER DATE RANGE")
    print("=" * 60)
    # Parse dates using dayfirst=True since raw dates follow DD/MM/YYYY format
    parsed_dates = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    min_date = parsed_dates.min()
    max_date = parsed_dates.max()
    print(f"Earliest Order Date (Min) : {min_date.strftime('%d-%b-%Y')}")
    print(f"Latest Order Date (Max)   : {max_date.strftime('%d-%b-%Y')}")
    print(f"Date Range Span           : {(max_date - min_date).days} days")

    # -------------------------------------------------------------
    # 11. Check minimum, maximum, and total Sales
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[11] SALES METRICS (MIN, MAX, TOTAL)")
    print("=" * 60)
    min_sales = df["Sales"].min()
    max_sales = df["Sales"].max()
    total_sales = df["Sales"].sum()
    mean_sales = df["Sales"].mean()
    median_sales = df["Sales"].median()

    print(f"Minimum Sales Value       : ${min_sales:,.2f}")
    print(f"Maximum Sales Value       : ${max_sales:,.2f}")
    print(f"Total Sales (Sum)         : ${total_sales:,.2f}")
    print(f"Average Sales (Mean)      : ${mean_sales:,.2f}")
    print(f"Median Sales (50th pct)   : ${median_sales:,.2f}")

    print("\n" + "=" * 60)
    print("DATA INSPECTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
