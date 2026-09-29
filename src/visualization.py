"""
Retail Sales Study - Visualization Generation Script
Author: B.Tech Data Science Student
Description: Generates 12 high-resolution, publication-quality visualizations
             illustrating product, category, regional, segment, and time-series
             sales trends from data/cleaned_superstore_sales.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def setup_plot_style():
    """Configures consistent professional styling across all charts."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300
    })

def format_currency_axis(ax, axis='y'):
    """Formats the specified axis numbers into clean currency notation ($K / $M)."""
    formatter = ticker.FuncFormatter(lambda x, pos: f"${x*1e-3:.0f}K" if x < 1e6 else f"${x*1e-6:.2f}M")
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

def main():
    setup_plot_style()

    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "cleaned_superstore_sales.csv")
    output_dir = os.path.join(base_dir, "outputs", "charts")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading cleaned data from: {data_path}")
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    generated_files = []

    # Palette definition for corporate consistency
    primary_color = "#2b5c8f"
    accent_palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    # =============================================================
    # 1. CATEGORY SALES
    # =============================================================
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(cat_sales["Category"], cat_sales["Sales"], color=accent_palette[:3], width=0.55, edgecolor="none")
    ax.set_title("Total Sales by Product Category", pad=15)
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales (USD)")
    format_currency_axis(ax, 'y')

    # Add direct data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

    f1 = os.path.join(output_dir, "01_sales_by_category.png")
    plt.tight_layout()
    plt.savefig(f1, dpi=300)
    plt.close()
    generated_files.append(f1)

    # =============================================================
    # 2. REGION SALES
    # =============================================================
    reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(reg_sales["Region"], reg_sales["Sales"], color=accent_palette, width=0.55)
    ax.set_title("Total Sales by Geographic Region", pad=15)
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales (USD)")
    format_currency_axis(ax, 'y')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')

    f2 = os.path.join(output_dir, "02_sales_by_region.png")
    plt.tight_layout()
    plt.savefig(f2, dpi=300)
    plt.close()
    generated_files.append(f2)

    # =============================================================
    # 3. SUB-CATEGORY SALES
    # =============================================================
    subcat_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).reset_index()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(subcat_sales["Sub-Category"], subcat_sales["Sales"], color="#3b6998", height=0.65)
    ax.set_title("Total Sales by Sub-Category (17 Sub-Categories)", pad=15)
    ax.set_xlabel("Total Sales (USD)")
    ax.set_ylabel("Sub-Category")
    format_currency_axis(ax, 'x')

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f" ${width:,.0f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=8.5)

    f3 = os.path.join(output_dir, "03_sales_by_subcategory.png")
    plt.tight_layout()
    plt.savefig(f3, dpi=300)
    plt.close()
    generated_files.append(f3)

    # =============================================================
    # 4. TOP 10 PRODUCTS
    # =============================================================
    prod_sales = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    # Sort ascending for horizontal bar chart display so highest appears at the top
    prod_sales = prod_sales.sort_values(by="Sales", ascending=True).reset_index(drop=True)

    # Truncate long product names for clean visual presentation
    short_names = [name if len(name) <= 40 else name[:37] + "..." for name in prod_sales["Product Name"]]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax.barh(short_names, prod_sales["Sales"], color="#2c7bb6", height=0.6)
    ax.set_title("Top 10 Products by Total Sales", pad=15)
    ax.set_xlabel("Total Sales (USD)")
    ax.set_ylabel("Product Name")
    format_currency_axis(ax, 'x')

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f" ${width:,.0f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='semibold')

    f4 = os.path.join(output_dir, "04_top_10_products.png")
    plt.tight_layout()
    plt.savefig(f4, dpi=300)
    plt.close()
    generated_files.append(f4)

    # =============================================================
    # 5. YEARLY SALES TREND
    # =============================================================
    yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(yearly_sales["Year"], yearly_sales["Sales"], marker="o", color=primary_color,
            linewidth=2.5, markersize=8)
    ax.set_title("Annual Sales Trend (2015 – 2018)", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales (USD)")
    ax.set_xticks(yearly_sales["Year"])
    format_currency_axis(ax, 'y')

    for _, row in yearly_sales.iterrows():
        ax.annotate(f"${row['Sales']:,.0f}",
                    xy=(row["Year"], row["Sales"]),
                    xytext=(0, 9), textcoords="offset points",
                    ha='center', fontsize=9.5, fontweight='semibold')

    f5 = os.path.join(output_dir, "05_yearly_sales_trend.png")
    plt.tight_layout()
    plt.savefig(f5, dpi=300)
    plt.close()
    generated_files.append(f5)

    # =============================================================
    # 6. MONTHLY SALES PATTERN
    # =============================================================
    monthly_sales = df.groupby(["Month Number", "Month"])["Sales"].sum().reset_index().sort_values(by="Month Number")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(monthly_sales["Month"], monthly_sales["Sales"], color="#4575b4", width=0.6)
    ax.set_title("Aggregated Monthly Sales Pattern (January – December)", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales (USD)")
    plt.xticks(rotation=35, ha='right')
    format_currency_axis(ax, 'y')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height*1e-3:.0f}K",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    f6 = os.path.join(output_dir, "06_monthly_sales_pattern.png")
    plt.tight_layout()
    plt.savefig(f6, dpi=300)
    plt.close()
    generated_files.append(f6)

    # =============================================================
    # 7. COMPLETE MONTHLY TIME SERIES (48 MONTHS)
    # =============================================================
    ym_sales = df.groupby("Year-Month")["Sales"].sum().reset_index().sort_values(by="Year-Month")
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.plot(ym_sales["Year-Month"], ym_sales["Sales"], marker=".", color=primary_color, linewidth=1.8, markersize=6)
    ax.set_title("Complete 48-Month Sales Trend (2015-01 to 2018-12)", pad=15)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Monthly Sales (USD)")
    plt.xticks(rotation=90, fontsize=8)
    format_currency_axis(ax, 'y')

    f7 = os.path.join(output_dir, "07_48_month_sales_trend.png")
    plt.tight_layout()
    plt.savefig(f7, dpi=300)
    plt.close()
    generated_files.append(f7)

    # =============================================================
    # 8. QUARTERLY SALES TREND
    # =============================================================
    quarterly_sales = df.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
    quarterly_sales["Year-Quarter"] = quarterly_sales["Year"].astype(str) + " " + quarterly_sales["Quarter"]
    quarterly_sales = quarterly_sales.sort_values(by=["Year", "Quarter"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(quarterly_sales["Year-Quarter"], quarterly_sales["Sales"], marker="s", color="#1f78b4", linewidth=2.2, markersize=7)
    ax.set_title("Quarterly Sales Performance (16 Quarters)", pad=15)
    ax.set_xlabel("Year & Quarter")
    ax.set_ylabel("Quarterly Sales (USD)")
    plt.xticks(rotation=45, ha='right')
    format_currency_axis(ax, 'y')

    for _, row in quarterly_sales.iterrows():
        ax.annotate(f"${row['Sales']*1e-3:.0f}K",
                    xy=(row["Year-Quarter"], row["Sales"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', fontsize=8.5)

    f8 = os.path.join(output_dir, "08_quarterly_sales_trend.png")
    plt.tight_layout()
    plt.savefig(f8, dpi=300)
    plt.close()
    generated_files.append(f8)

    # =============================================================
    # 9. CUSTOMER SEGMENT SALES
    # =============================================================
    seg_sales = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bars = ax.bar(seg_sales["Segment"], seg_sales["Sales"], color=["#1b9e77", "#d95f02", "#7570b3"], width=0.5)
    ax.set_title("Total Sales by Customer Segment", pad=15)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Total Sales (USD)")
    format_currency_axis(ax, 'y')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"${height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='semibold')

    f9 = os.path.join(output_dir, "09_sales_by_segment.png")
    plt.tight_layout()
    plt.savefig(f9, dpi=300)
    plt.close()
    generated_files.append(f9)

    # =============================================================
    # 10. CATEGORY BY REGION (GROUPED BAR CHART)
    # =============================================================
    pivot_cr = df.groupby(["Region", "Category"])["Sales"].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    pivot_cr.plot(kind="bar", ax=ax, width=0.75, colormap="tab10", edgecolor="none")
    ax.set_title("Sales by Category Across Regions", pad=15)
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales (USD)")
    plt.xticks(rotation=0)
    format_currency_axis(ax, 'y')
    ax.legend(title="Category", frameon=True)

    f10 = os.path.join(output_dir, "10_category_region_comparison.png")
    plt.tight_layout()
    plt.savefig(f10, dpi=300)
    plt.close()
    generated_files.append(f10)

    # =============================================================
    # 11. SALES DISTRIBUTION
    # =============================================================
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    # Filter transactions under $1,500 to clearly visualize the high-density distribution
    # which captures 97% of all retail transactions
    threshold = 1500
    subset_sales = df[df["Sales"] <= threshold]["Sales"]

    sns.histplot(subset_sales, bins=50, kde=True, color="#2b5c8f", ax=ax, edgecolor="white")
    ax.set_title(f"Distribution of Transaction Sales (Up to ${threshold:,})", pad=15)
    ax.set_xlabel("Transaction Sales Amount (USD)")
    ax.set_ylabel("Frequency (Number of Transactions)")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    # Annotate summary text box
    ax.text(0.65, 0.75,
            f"Median: ${df['Sales'].median():.2f}\nMean: ${df['Sales'].mean():.2f}\nCap: 97.4% of Records",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    f11 = os.path.join(output_dir, "11_sales_distribution.png")
    plt.tight_layout()
    plt.savefig(f11, dpi=300)
    plt.close()
    generated_files.append(f11)

    # =============================================================
    # 12. YEARLY CATEGORY TREND
    # =============================================================
    cat_yearly = df.groupby(["Year", "Category"])["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for idx, (category, group) in enumerate(cat_yearly.groupby("Category")):
        ax.plot(group["Year"], group["Sales"], marker="o", linewidth=2.2,
                label=category, color=accent_palette[idx % len(accent_palette)])

    ax.set_title("Yearly Sales Trends by Category (2015 – 2018)", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales (USD)")
    ax.set_xticks(cat_yearly["Year"].unique())
    format_currency_axis(ax, 'y')
    ax.legend(title="Category", frameon=True)

    f12 = os.path.join(output_dir, "12_yearly_category_trend.png")
    plt.tight_layout()
    plt.savefig(f12, dpi=300)
    plt.close()
    generated_files.append(f12)

    # =============================================================
    # COMPLETION NOTICE
    # =============================================================
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETED")
    print("=" * 60)
    for idx, f in enumerate(generated_files, start=1):
        rel_path = os.path.relpath(f, base_dir)
        print(f"{idx:2d}. {rel_path}")

if __name__ == "__main__":
    main()
