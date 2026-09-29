"""
Retail Sales Study - Data Cleaning & Preprocessing Script
Author: B.Tech Data Science Student
Description: This script cleans the raw Superstore sales dataset (data/superstore_sales.csv),
             performs datetime parsing with day-first interpretation, handles missing postal
             codes using a documented numeric placeholder (0), derives calendar features
             for time-trend analysis, and saves the cleaned dataset to data/cleaned_superstore_sales.csv.
"""

import os
import pandas as pd
import numpy as np

def main():
    print("=" * 65)
    print("RETAIL SALES STUDY - DATA CLEANING & PREPROCESSING")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Load data/superstore_sales.csv using pandas
    # -------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "superstore_sales.csv")

    if not os.path.exists(input_path):
        input_path = os.path.join("data", "superstore_sales.csv")

    print(f"\n[1] Loading raw dataset from: {input_path}")
    raw_df = pd.read_csv(input_path)

    # Capture initial statistics before any modifications
    orig_shape = raw_df.shape
    missing_before = raw_df.isnull().sum()
    duplicates_before = raw_df.duplicated().sum()

    # -------------------------------------------------------------
    # 2. Create an explicit copy so the original dataset is never modified
    # -------------------------------------------------------------
    df_clean = raw_df.copy()
    print("[2] Created working copy of dataset (raw data preserved).")

    # -------------------------------------------------------------
    # 3. Convert Order Date to datetime using day-first interpretation
    # -------------------------------------------------------------
    print("[3] Converting 'Order Date' to datetime (dayfirst=True)...")
    df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"], dayfirst=True)

    # -------------------------------------------------------------
    # 4. Convert Ship Date to datetime using day-first interpretation
    # -------------------------------------------------------------
    print("[4] Converting 'Ship Date' to datetime (dayfirst=True)...")
    df_clean["Ship Date"] = pd.to_datetime(df_clean["Ship Date"], dayfirst=True)

    # -------------------------------------------------------------
    # 5. Handle missing Postal Code values without deleting records
    #    Treatment: The 11 missing records all correspond to Burlington,
    #    Vermont. To retain all sales transactions without introducing bias,
    #    we fill NaN values with numeric placeholder 0 (representing 'Unknown / Not Provided').
    #    We then cast Postal Code to integer so it is stored cleanly.
    # -------------------------------------------------------------
    print("[5] Handling missing 'Postal Code' values...")
    missing_postal_count = df_clean["Postal Code"].isnull().sum()
    print(f"    - Found {missing_postal_count} missing Postal Code values.")
    print("    - Applying documented numeric placeholder: 0 (Unknown / Not Provided).")
    df_clean["Postal Code"] = df_clean["Postal Code"].fillna(0).astype(int)

    # -------------------------------------------------------------
    # 6. Check for duplicate rows and remove only if any are found
    # -------------------------------------------------------------
    print(f"[6] Checking for duplicate records: {duplicates_before} duplicates found.")
    if duplicates_before > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"    - Removed {duplicates_before} duplicate rows.")
    else:
        print("    - No duplicate rows present. No records removed.")

    # -------------------------------------------------------------
    # 7. Check for missing values after cleaning
    # -------------------------------------------------------------
    missing_after = df_clean.isnull().sum()
    duplicates_after = df_clean.duplicated().sum()

    # -------------------------------------------------------------
    # 8. Create new calendar columns derived from Order Date
    # -------------------------------------------------------------
    print("[8] Engineering time features from 'Order Date'...")
    df_clean["Year"] = df_clean["Order Date"].dt.year
    df_clean["Month"] = df_clean["Order Date"].dt.month_name()
    df_clean["Month Number"] = df_clean["Order Date"].dt.month
    df_clean["Quarter"] = "Q" + df_clean["Order Date"].dt.quarter.astype(str)
    df_clean["Year-Month"] = df_clean["Order Date"].dt.strftime("%Y-%m")

    new_columns = ["Year", "Month", "Month Number", "Quarter", "Year-Month"]
    print(f"    - Created {len(new_columns)} new columns: {', '.join(new_columns)}")

    # -------------------------------------------------------------
    # 9. Verify original columns are retained
    # -------------------------------------------------------------
    for col in raw_df.columns:
        assert col in df_clean.columns, f"Original column {col} is missing!"

    # -------------------------------------------------------------
    # 10. Sort cleaned dataframe chronologically by Order Date
    # -------------------------------------------------------------
    print("[10] Sorting cleaned dataframe chronologically by 'Order Date'...")
    df_clean = df_clean.sort_values(by="Order Date", ascending=True).reset_index(drop=True)

    # -------------------------------------------------------------
    # 11. Save cleaned dataset to data/cleaned_superstore_sales.csv
    # -------------------------------------------------------------
    output_path = os.path.join(base_dir, "data", "cleaned_superstore_sales.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"[11] Saved cleaned dataset to: {output_path}")

    # Also save a copy in outputs/cleaned_data/ for structured outputs
    outputs_clean_path = os.path.join(base_dir, "outputs", "cleaned_data", "cleaned_superstore_sales.csv")
    os.makedirs(os.path.dirname(outputs_clean_path), exist_ok=True)
    df_clean.to_csv(outputs_clean_path, index=False)
    print(f"     (Also archived copy in: {outputs_clean_path})")

    # -------------------------------------------------------------
    # 12. Display summary inspection report
    # -------------------------------------------------------------
    cleaned_shape = df_clean.shape
    min_date = df_clean["Order Date"].min().strftime("%d-%b-%Y")
    max_date = df_clean["Order Date"].max().strftime("%d-%b-%Y")
    total_sales = df_clean["Sales"].sum()

    print("\n" + "=" * 65)
    print("DATA CLEANING SUMMARY REPORT")
    print("=" * 65)
    print(f"Original Shape           : {orig_shape[0]:,} rows x {orig_shape[1]} columns")
    print(f"Cleaned Shape            : {cleaned_shape[0]:,} rows x {cleaned_shape[1]} columns")
    print(f"\nDuplicate Count Before   : {duplicates_before}")
    print(f"Duplicate Count After    : {duplicates_after}")
    print(f"\nOrder Date Range         : {min_date} to {max_date}")
    print(f"Total Sales Amount       : ${total_sales:,.2f}")
    print(f"\nNewly Created Columns    : {new_columns}")

    print("\nMissing Values Before Cleaning:")
    missing_before_filtered = missing_before[missing_before > 0]
    if len(missing_before_filtered) == 0:
        print("  None")
    else:
        for col, count in missing_before_filtered.items():
            print(f"  - {col}: {count} missing")

    print("\nMissing Values After Cleaning:")
    missing_after_filtered = missing_after[missing_after > 0]
    if len(missing_after_filtered) == 0:
        print("  None (0 missing values across all columns)")
    else:
        for col, count in missing_after_filtered.items():
            print(f"  - {col}: {count} missing")

    print("\n" + "=" * 65)
    print("CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
