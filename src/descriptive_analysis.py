"""
Retail Sales Study - Descriptive Analytics Script
Author: B.Tech Data Science Student
Description: This script performs comprehensive descriptive analysis on the
             cleaned Superstore retail sales dataset (data/cleaned_superstore_sales.csv).
             It analyzes products, categories, sub-categories, regions, segments,
             and temporal trends without using machine learning or predictions.
"""

import os
import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("RETAIL SALES STUDY - DESCRIPTIVE ANALYSIS REPORT")
    print("=" * 80)

    # -------------------------------------------------------------
    # Load Cleaned Dataset
    # -------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "cleaned_superstore_sales.csv")

    if not os.path.exists(file_path):
        file_path = os.path.join("data", "cleaned_superstore_sales.csv")

    print(f"\nLoading cleaned dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data successfully loaded. Records: {len(df):,}, Columns: {len(df.columns)}")

    # =============================================================
    # 1. OVERALL SALES SUMMARY
    # =============================================================
    print("\n" + "=" * 80)
    print("1. OVERALL SALES SUMMARY")
    print("=" * 80)

    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    median_sales = df["Sales"].median()
    min_sales = df["Sales"].min()
    max_sales = df["Sales"].max()

    num_transactions = len(df)
    num_orders = df["Order ID"].nunique()
    num_customers = df["Customer ID"].nunique()
    num_products = df["Product ID"].nunique()
    num_categories = df["Category"].nunique()
    num_subcategories = df["Sub-Category"].nunique()
    num_regions = df["Region"].nunique()
    num_segments = df["Segment"].nunique()

    print(f"Total Sales                     : ${total_sales:,.2f}")
    print(f"Average Sales per Transaction   : ${avg_sales:,.2f}")
    print(f"Median Sales per Transaction    : ${median_sales:,.2f}")
    print(f"Minimum Sales in a Transaction  : ${min_sales:,.2f}")
    print(f"Maximum Sales in a Transaction  : ${max_sales:,.2f}")
    print("-" * 50)
    print(f"Number of Transactions (Rows)   : {num_transactions:,}")
    print(f"Number of Unique Orders         : {num_orders:,}")
    print(f"Number of Unique Customers      : {num_customers:,}")
    print(f"Number of Unique Products       : {num_products:,}")
    print(f"Number of Categories            : {num_categories}")
    print(f"Number of Sub-Categories        : {num_subcategories}")
    print(f"Number of Regions               : {num_regions}")
    print(f"Number of Segments              : {num_segments}")

    # =============================================================
    # 2. PRODUCT ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("2. PRODUCT ANALYSIS")
    print("=" * 80)

    # Group by Product Name to compute total sales and transaction count
    product_stats = df.groupby("Product Name").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count")
    ).reset_index()

    # Sort descending by Total Sales
    product_stats_sorted = product_stats.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    # Top 10 products
    top_10_products = product_stats_sorted.head(10).copy()
    top_10_sales_sum = top_10_products["Total_Sales"].sum()
    top_10_pct = (top_10_sales_sum / total_sales) * 100

    print("--- TOP 10 PRODUCTS BY TOTAL SALES ---")
    top_10_display = top_10_products.copy()
    top_10_display["Rank"] = range(1, 11)
    top_10_display["Sales Contribution (%)"] = (top_10_display["Total_Sales"] / total_sales * 100).map("{:.2f}%".format)
    top_10_display["Total Sales"] = top_10_display["Total_Sales"].map("${:,.2f}".format)
    top_10_display.rename(columns={"Product Name": "Product Name", "Transaction_Count": "Transactions"}, inplace=True)
    cols_order = ["Rank", "Product Name", "Total Sales", "Transactions", "Sales Contribution (%)"]
    print(top_10_display[cols_order].to_string(index=False))

    print(f"\nCumulative Sales of Top 10 Products : ${top_10_sales_sum:,.2f}")
    print(f"Sales Contribution of Top 10 Products: {top_10_pct:.2f}% of overall sales")

    # Bottom 10 products (displayed in descending order of Sales among the lowest 10)
    bottom_10_products = product_stats_sorted.tail(10).copy()
    print("\n--- BOTTOM 10 PRODUCTS BY TOTAL SALES (Descending Order) ---")
    bottom_10_display = bottom_10_products.copy()
    bottom_10_display["Rank"] = range(len(product_stats_sorted) - 9, len(product_stats_sorted) + 1)
    bottom_10_display["Sales Contribution (%)"] = (bottom_10_display["Total_Sales"] / total_sales * 100).map("{:.4f}%".format)
    bottom_10_display["Total Sales"] = bottom_10_display["Total_Sales"].map("${:,.2f}".format)
    bottom_10_display.rename(columns={"Product Name": "Product Name", "Transaction_Count": "Transactions"}, inplace=True)
    print(bottom_10_display[["Rank", "Product Name", "Total Sales", "Transactions", "Sales Contribution (%)"]].to_string(index=False))

    # =============================================================
    # 3. CATEGORY ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("3. CATEGORY ANALYSIS")
    print("=" * 80)

    cat_analysis = df.groupby("Category").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count"),
        Average_Sales=("Sales", "mean")
    ).reset_index()

    # Calculate percentage contribution
    cat_analysis["Sales_Percentage"] = (cat_analysis["Total_Sales"] / total_sales) * 100
    cat_analysis = cat_analysis.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    # Format for clean display
    cat_display = cat_analysis.copy()
    cat_display["Total Sales"] = cat_display["Total_Sales"].map("${:,.2f}".format)
    cat_display["Average Sales"] = cat_display["Average_Sales"].map("${:,.2f}".format)
    cat_display["Sales Percentage"] = cat_display["Sales_Percentage"].map("{:.2f}%".format)
    cat_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    print(cat_display[["Category", "Total Sales", "Transactions", "Average Sales", "Sales Percentage"]].to_string(index=False))

    # =============================================================
    # 4. SUB-CATEGORY ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("4. SUB-CATEGORY ANALYSIS")
    print("=" * 80)

    subcat_analysis = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count"),
        Average_Sales=("Sales", "mean")
    ).reset_index()

    subcat_analysis["Sales_Percentage"] = (subcat_analysis["Total_Sales"] / total_sales) * 100
    subcat_analysis = subcat_analysis.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    subcat_display = subcat_analysis.copy()
    subcat_display["Total Sales"] = subcat_display["Total_Sales"].map("${:,.2f}".format)
    subcat_display["Average Sales"] = subcat_display["Average_Sales"].map("${:,.2f}".format)
    subcat_display["Sales Percentage"] = subcat_display["Sales_Percentage"].map("{:.2f}%".format)
    subcat_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    print(subcat_display[["Sub-Category", "Total Sales", "Transactions", "Average Sales", "Sales Percentage"]].to_string(index=False))

    highest_subcat = subcat_analysis.iloc[0]
    lowest_subcat = subcat_analysis.iloc[-1]
    print(f"\nHighest-selling sub-category : {highest_subcat['Sub-Category']} (${highest_subcat['Total_Sales']:,.2f})")
    print(f"Lowest-selling sub-category  : {lowest_subcat['Sub-Category']} (${lowest_subcat['Total_Sales']:,.2f})")

    # =============================================================
    # 5. REGIONAL ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("5. REGIONAL ANALYSIS")
    print("=" * 80)

    reg_analysis = df.groupby("Region").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count"),
        Average_Sales=("Sales", "mean")
    ).reset_index()

    reg_analysis["Sales_Percentage"] = (reg_analysis["Total_Sales"] / total_sales) * 100
    reg_analysis = reg_analysis.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    reg_display = reg_analysis.copy()
    reg_display["Total Sales"] = reg_display["Total_Sales"].map("${:,.2f}".format)
    reg_display["Average Sales"] = reg_display["Average_Sales"].map("${:,.2f}".format)
    reg_display["Sales Percentage"] = reg_display["Sales_Percentage"].map("{:.2f}%".format)
    reg_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    print(reg_display[["Region", "Total Sales", "Transactions", "Average Sales", "Sales Percentage"]].to_string(index=False))

    highest_region = reg_analysis.iloc[0]
    lowest_region = reg_analysis.iloc[-1]
    print(f"\nRegion with highest total sales : {highest_region['Region']} (${highest_region['Total_Sales']:,.2f})")
    print(f"Region with lowest total sales  : {lowest_region['Region']} (${lowest_region['Total_Sales']:,.2f})")

    # =============================================================
    # 6. SEGMENT ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("6. SEGMENT ANALYSIS")
    print("=" * 80)

    seg_analysis = df.groupby("Segment").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count"),
        Average_Sales=("Sales", "mean")
    ).reset_index()

    seg_analysis["Sales_Percentage"] = (seg_analysis["Total_Sales"] / total_sales) * 100
    seg_analysis = seg_analysis.sort_values(by="Total_Sales", ascending=False).reset_index(drop=True)

    seg_display = seg_analysis.copy()
    seg_display["Total Sales"] = seg_display["Total_Sales"].map("${:,.2f}".format)
    seg_display["Average Sales"] = seg_display["Average_Sales"].map("${:,.2f}".format)
    seg_display["Sales Percentage"] = seg_display["Sales_Percentage"].map("{:.2f}%".format)
    seg_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    print(seg_display[["Segment", "Total Sales", "Transactions", "Average Sales", "Sales Percentage"]].to_string(index=False))

    # =============================================================
    # 7. YEARLY SALES ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("7. YEARLY SALES ANALYSIS")
    print("=" * 80)

    yearly_analysis = df.groupby("Year").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count")
    ).reset_index().sort_values(by="Year", ascending=True).reset_index(drop=True)

    # Calculate year-over-year percentage change
    yearly_analysis["YoY_Change_Pct"] = yearly_analysis["Total_Sales"].pct_change() * 100

    yearly_display = yearly_analysis.copy()
    yearly_display["Total Sales"] = yearly_display["Total_Sales"].map("${:,.2f}".format)
    yearly_display.rename(columns={"Transaction_Count": "Transaction Count"}, inplace=True)
    yearly_display["YoY Growth (%)"] = yearly_display["YoY_Change_Pct"].map(
        lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A (Base Year)"
    )

    print(yearly_display[["Year", "Total Sales", "Transaction Count", "YoY Growth (%)"]].to_string(index=False))

    # =============================================================
    # 8. MONTHLY SALES ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("8. MONTHLY SALES ANALYSIS (AGGREGATED CALENDAR MONTHS)")
    print("=" * 80)

    monthly_analysis = df.groupby(["Month Number", "Month"]).agg(
        Total_Sales=("Sales", "sum")
    ).reset_index().sort_values(by="Month Number", ascending=True).reset_index(drop=True)

    monthly_analysis["Sales_Percentage"] = (monthly_analysis["Total_Sales"] / total_sales) * 100

    monthly_display = monthly_analysis.copy()
    monthly_display["Total Sales"] = monthly_display["Total_Sales"].map("${:,.2f}".format)
    monthly_display["Sales Percentage"] = monthly_display["Sales_Percentage"].map("{:.2f}%".format)

    print(monthly_display[["Month Number", "Month", "Total Sales", "Sales Percentage"]].to_string(index=False))

    # =============================================================
    # 9. YEAR-MONTH SALES TREND
    # =============================================================
    print("\n" + "=" * 80)
    print("9. YEAR-MONTH SALES TREND (COMPLETE 48-MONTH TIME SERIES)")
    print("=" * 80)

    ym_trend = df.groupby("Year-Month").agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count")
    ).reset_index().sort_values(by="Year-Month", ascending=True).reset_index(drop=True)

    ym_display = ym_trend.copy()
    ym_display["Total Sales"] = ym_display["Total_Sales"].map("${:,.2f}".format)
    ym_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    # Print full time series in two side-by-side columns if needed or complete list
    print(ym_display[["Year-Month", "Total Sales", "Transactions"]].to_string(index=False))

    # =============================================================
    # 10. QUARTERLY SALES ANALYSIS
    # =============================================================
    print("\n" + "=" * 80)
    print("10. QUARTERLY SALES ANALYSIS (YEAR + QUARTER)")
    print("=" * 80)

    quarterly_analysis = df.groupby(["Year", "Quarter"]).agg(
        Total_Sales=("Sales", "sum"),
        Transaction_Count=("Sales", "count")
    ).reset_index().sort_values(by=["Year", "Quarter"], ascending=True).reset_index(drop=True)

    quarterly_display = quarterly_analysis.copy()
    quarterly_display["Total Sales"] = quarterly_display["Total_Sales"].map("${:,.2f}".format)
    quarterly_display.rename(columns={"Transaction_Count": "Transactions"}, inplace=True)

    print(quarterly_display[["Year", "Quarter", "Total Sales", "Transactions"]].to_string(index=False))

    # =============================================================
    # 11. CATEGORY BY REGION (CROSS-TABULATION)
    # =============================================================
    print("\n" + "=" * 80)
    print("11. CATEGORY BY REGION (CROSS-TABULATION - TOTAL SALES)")
    print("=" * 80)

    pivot_region_cat = pd.pivot_table(
        df,
        values="Sales",
        index="Region",
        columns="Category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    )

    # Format numbers for readability
    pivot_formatted = pivot_region_cat.map("${:,.2f}".format)
    print(pivot_formatted.to_string())

    # =============================================================
    # 12. SUMMARY FINDINGS (FACTUAL OBSERVATIONS)
    # =============================================================
    print("\n" + "=" * 80)
    print("12. SUMMARY FINDINGS (FACTUAL OBSERVATIONS)")
    print("=" * 80)

    # Automatically derive factual observations from computed metrics
    top_cat = cat_analysis.iloc[0]
    top_reg = reg_analysis.iloc[0]
    lowest_reg = reg_analysis.iloc[-1]
    regional_diff = top_reg["Total_Sales"] - lowest_reg["Total_Sales"]

    top_product = product_stats_sorted.iloc[0]
    top_year = yearly_analysis.sort_values(by="Total_Sales", ascending=False).iloc[0]
    top_month = monthly_analysis.sort_values(by="Total_Sales", ascending=False).iloc[0]
    top_quarter = quarterly_analysis.sort_values(by="Total_Sales", ascending=False).iloc[0]

    print("Key factual observations derived from data analysis:")
    print(f"1. Category with highest total sales : {top_cat['Category']} "
          f"(${top_cat['Total_Sales']:,.2f}, representing {top_cat['Sales_Percentage']:.2f}% of total sales)")

    print(f"2. Region with highest total sales   : {top_reg['Region']} "
          f"(${top_reg['Total_Sales']:,.2f}, representing {top_reg['Sales_Percentage']:.2f}% of total sales)")

    print(f"3. Highest-selling sub-category      : {highest_subcat['Sub-Category']} "
          f"(${highest_subcat['Total_Sales']:,.2f}, representing {highest_subcat['Sales_Percentage']:.2f}% of total sales)")

    print(f"4. Product with highest total sales  : {top_product['Product Name']} "
          f"(${top_product['Total_Sales']:,.2f} across {top_product['Transaction_Count']} transactions)")

    print(f"5. Year with highest total sales     : {int(top_year['Year'])} "
          f"(${top_year['Total_Sales']:,.2f}, with {int(top_year['Transaction_Count']):,} transactions)")

    print(f"6. Month with highest total sales    : {top_month['Month']} (Month {top_month['Month Number']}) "
          f"(${top_month['Total_Sales']:,.2f}, representing {top_month['Sales_Percentage']:.2f}% of aggregate annual sales)")

    print(f"7. Quarter with highest total sales  : {int(top_quarter['Year'])} {top_quarter['Quarter']} "
          f"(${top_quarter['Total_Sales']:,.2f}, with {top_quarter['Transaction_Count']:,} transactions)")

    print(f"8. Regional sales disparity          : Difference between highest ({top_reg['Region']}) "
          f"and lowest ({lowest_reg['Region']}) regions is ${regional_diff:,.2f}")

    print("\n" + "=" * 80)
    print("DESCRIPTIVE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
