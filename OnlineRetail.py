import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data= pd.read_csv('online.csv', encoding='unicode_escape')
#print(data)
#print(data.head())
description = data.describe()
#print(description)
#print(data.shape)
print(data.info())
#print(data.columns)#use info intead
#print(data.isnull() .sum())
df_null = round(100*(data.isnull().sum())/len(data), 2)
#print(df_null)
#data= data.drop(['StockCode'], axis =1)
data['CustomerId']=data['CustomerID'].astype(str)
#print(data.info())
data['Amount']=data['Quantity']*data['UnitPrice']
#print(data.info())
#print(data.head)
data_monitoring=data.groupby('CustomerId')['Amount'].sum()
data_monitoring = data_monitoring.reset_index()
#print(data_monitoring.head())
#print(data.head)
data['MostSold']=data['Description']*data['Quantity']
data_most_sold=data.groupby('Description')['Quantity'].sum()
data_most_sold = data_most_sold.reset_index().sort_values(by='Quantity', ascending=False)
#print(data_most_sold)
#print(data.columns)

data['RegionMost']=data['Quantity']*data['Country']
data_most_region=data.groupby('Country')['Quantity'].sum()
data_most_region = data_most_region.reset_index()
#print(data_most_region.head())

#Frequently sold

data['Frequent']=data['Description']*data['Quantity']
data_most_frequent=data.groupby('Description')['InvoiceNo'].count()
data_most_frequent = data_most_frequent.sort_values()
#print(data_most_frequent)

data['lastMonth']=data['Quantity']*data['InvoiceDate']
data_most_frequent=data.groupby('Description')['InvoiceNo'].count()
data_most_frequent = data_most_frequent.sort_values()
#print(data_most_frequent)

# Filter data for the last month
data['InvoiceDate'] =pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
#print(data['InvoiceDate'])
max_date = max(data['InvoiceDate'])
print(max_date)

min_date = min(data['InvoiceDate'])
print(min_date)

#total number of days
diff=(max_date-min_date)
print(diff)

#Transaction for the las 30 days

# Compute last transaction date to get the recency of customers

# Assuming data is a DataFrame and 'InvoiceDate' is a datetime column
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Get the last date in the dataset
last_date = data['InvoiceDate'].max()

# Calculate the first date of the last month
first_date_last_month = last_date.replace(day=1) - pd.DateOffset(months=1)

# Filter data for the last month
last_month_data = data[(data['InvoiceDate'] >= first_date_last_month) & (data['InvoiceDate'] < last_date.replace(day=1))]

# Calculate TotalSales for each row
last_month_data['TotalSales'] = last_month_data['UnitPrice'] * last_month_data['Quantity']

# Sum the TotalSales for the last month
total_sales_last_month = last_month_data['TotalSales'].sum()

print('The total sales for thr last month are:', total_sales_last_month)





'''last = data.groupby('CustomerID')['Diff'].min()
last  = last.reset_index()
print(last.head())

#mean_date = data['InvoiceDate'].mean()
#print(mean_date)'''

'''start_date = max_date - pd.DateOffset(days=30)
print(start_date)
print(data.columns)'''

'''
data['TotalSales']=data['UnitPrice']*data['Quantity']
print(data.info())
# Filter data for the specific date and onwards for that month
#filtered_data = df[(df['InvoiceDate'] >= specific_date) & (df['InvoiceDate'].dt.month == specific_date.month)]
data_last_month_sales=data[data['InvoiceDate'] >= ('2011-11-09 12:50:00')],data.groupby('InvoiceDate')['TotalSales'].sum()
#data_most_region = data_most_region.reset_index()
#data_last_month_sales = data['TotalSales'].groupby(data['InvoiceDate'].dt.to_period('M')).sum()
data_last_month_sales = data.groupby('InvoiceDate')['TotalSales'].sum()
data_last_month_sales=data_last_month_sales.sum()

print(data_last_month_sales)
'''

''''
# Assuming 'data' is a pandas DataFrame containing 'UnitPrice', 'Quantity', and 'InvoiceDate' columns

# 1. Calculate total sales for each invoice date
data['TotalSales'] = data['UnitPrice'] * data['Quantity']

# 2. Get today's date and extract last month's year and month

today = date.today()
last_month = today - timedelta(days=today.day - 1)  # Adjust for current day within month

# 3. Filter data for last month using boolean indexing
last_month_mask = (data['InvoiceDate'].dt.year == last_month.year) & (data['InvoiceDate'].dt.month == last_month.month)
filtered_data = data[last_month_mask]

# 4. Get total sales for last month
total_sales_last_month = filtered_data['TotalSales'].sum()

print("Total sales for last month:", total_sales_last_month)
'''

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
features = ['Quantity', 'UnitPrice', 'CustomerID']
data = data.dropna(subset=features)
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

kmeans = KMeans(n_clusters=4, max_iter=50, random_state=42)
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=50, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_k = 5  # Based on the analysis
kmeans = KMeans(n_clusters=optimal_k, max_iter=50, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original data
data['Cluster'] = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print("For n_clusters={0}, the silhouette score is {1}".format(n_clusters, silhouette_avg))

