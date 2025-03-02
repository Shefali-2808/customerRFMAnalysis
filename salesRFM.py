import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure pandas display options
pd.set_option('display.max_rows', 100)

# Load dataset
DATA_PATH = './sales_data_sample.csv'
data = pd.read_csv(DATA_PATH, encoding='unicode_escape')

# Function to plot missing values
def plot_missing_values(df, title='Missing Values in Dataset'):
    missing_values = df.isnull().sum() / len(df) * 100
    missing_values = missing_values[missing_values > 0].reset_index()
    missing_values.columns = ['Feature', 'Missing Percentage']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_values, x='Feature', y='Missing Percentage')
    plt.title(title, fontsize=20)
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('% of Missing Values', fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/missing_values.png', dpi=300)
    plt.show()

# Function to summarize dataset
def summarize_data():
    print("Dataset Preview:\n", data.head())
    print("\nDataset Statistics:\n", data.describe())
    print(f'\nDataset contains {data.shape[0]} rows and {data.shape[1]} columns.')

# Function to plot number of customers per country
def plot_customers_by_country():
    customer_counts = data.groupby('COUNTRY')['CUSTOMERNAME'].nunique().reset_index()
    customer_counts.columns = ['Country', 'Customer Count']
    customer_counts = customer_counts.sort_values(by='Customer Count', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=customer_counts, x='Country', y='Customer Count')
    plt.title('Number of Customers by Country', fontsize=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/sales_by_country.png', dpi=300)
    plt.show()

# Function to plot top customers by sales
def plot_top_customers_by_sales():
    top_customers = (data.groupby(['CUSTOMERNAME', 'COUNTRY'])['SALES']
                        .sum()
                        .reset_index()
                        .sort_values(by='SALES', ascending=False)
                        .head(5))

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_customers, x='CUSTOMERNAME', y='SALES', hue='COUNTRY')
    plt.title('Top 5 Customers by Sales', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/top_customers_by_sales.png', dpi=300)
    plt.show()

# Function to plot total sales by country
def plot_sales_by_country():
    sales_by_country = (data.groupby('COUNTRY')['SALES']
                          .sum()
                          .reset_index()
                          .sort_values(by='SALES', ascending=False))

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_by_country.head(10), x='COUNTRY', y='SALES')
    plt.title('Total Sales by Country', fontsize=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot shipping status distribution
def plot_shipping_status():
    shipping_status = data['STATUS'].value_counts().reset_index()
    shipping_status.columns = ['Status', 'Count']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=shipping_status, x='Status', y='Count')
    plt.title('Shipping Status Distribution', fontsize=20)
    plt.yscale('log')  # Log scale to visualize differences better
    plt.tight_layout()
    plt.savefig('plots/shipping_status.png', dpi=300)
    plt.show()

# Function to analyze total and cancelled orders per country
def plot_orders_by_country():
    cancelled_orders = data[data['STATUS'] == 'Cancelled'].groupby('COUNTRY')['ORDERNUMBER'].count().reset_index()
    cancelled_orders.columns = ['Country', 'Cancelled Orders']

    total_orders = data.groupby('COUNTRY')['ORDERNUMBER'].count().reset_index()
    total_orders.columns = ['Country', 'Total Orders']

    merged_orders = cancelled_orders.merge(total_orders, on='Country', how='outer').fillna(0)
    melted_orders = merged_orders.melt(id_vars=['Country'], value_vars=['Total Orders', 'Cancelled Orders'],
                                       var_name='Order Type', value_name='Count')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_orders, x='Country', y='Count', hue='Order Type')
    plt.title('Total and Cancelled Orders by Country', fontsize=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/orders_by_country.png', dpi=300)
    plt.show()

# Function to calculate customer recency
def calculate_customer_recency():
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'])
    recency = data.groupby('CUSTOMERNAME')['ORDERDATE'].max().reset_index()
    recency.columns = ['Customer Name', 'Last Purchase Date']
    recency['Recency (days)'] = (data['ORDERDATE'].max() - recency['Last Purchase Date']).dt.days

    plt.figure(figsize=(12, 6))
    sns.histplot(data=recency, x='Recency (days)', bins=30)
    plt.title('Customer Recency Distribution', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/recency_distribution.png', dpi=300)
    plt.show()

    return recency

# Function to calculate customer frequency
def calculate_customer_frequency():
    frequency = data.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique().reset_index()
    frequency.columns = ['Customer Name', 'Frequency']

    plt.figure(figsize=(12, 6))
    sns.histplot(data=frequency, x='Frequency', bins=30)
    plt.title('Customer Frequency Distribution', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/frequency_distribution.png', dpi=300)
    plt.show()

    return frequency

# Function to calculate customer monetary value
def calculate_customer_monetary_value():
    monetary_value = data.groupby('CUSTOMERNAME')['SALES'].sum().reset_index()
    monetary_value.columns = ['Customer Name', 'Monetary Value']

    plt.figure(figsize=(12, 6))
    sns.histplot(data=monetary_value, x='Monetary Value', bins=30)
    plt.title('Customer Monetary Value Distribution', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/monetary_distribution.png', dpi=300)
    plt.show()

    return monetary_value

# Function to create an RFM dataframe
def create_rfm_dataframe():
    recency = calculate_customer_recency()
    frequency = calculate_customer_frequency()
    monetary_value = calculate_customer_monetary_value()

    rfm_df = recency.merge(frequency, on='Customer Name').merge(monetary_value, on='Customer Name')
    rfm_numeric = rfm_df[['Recency (days)', 'Frequency', 'Monetary Value']]

    # Create pairplot
    sns.pairplot(rfm_numeric, diag_kind='kde')  # kde for smoothed diagonal plots
    plt.suptitle('RFM Pairplot', y=1.02, fontsize=16)

    # Save figure
    plt.savefig('plots/rfm_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    return rfm_df

# Main function
def main():
    summarize_data()
    plot_sales_by_country()
    plot_missing_values(data)
    plot_customers_by_country()
    plot_top_customers_by_sales()
    plot_shipping_status()
    plot_orders_by_country()
    create_rfm_dataframe()

if __name__ == "__main__":
    main()
