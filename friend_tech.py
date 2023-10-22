# In[ ]:


# Importing Packages
from flipside import Flipside
import pandas as pd
import seaborn as sns
import numpy as np
import time
import requests
import time
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[ ]:


# Flipside api key 

sdk_api = "ADD_YOUR_API_KEY_HERE"

flipside = Flipside(sdk_api, "https://api-v2.flipsidecrypto.xyz")


# In[ ]:


# TVL

sql = """
WITH daily_traces AS (
  SELECT
    DATE_TRUNC('day', TO_TIMESTAMP(BLOCK_TIMESTAMP)) AS date,
    SUM(CASE WHEN TO_ADDRESS = LOWER('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4') THEN ETH_VALUE ELSE 0 END) AS daily_inflow,
    SUM(CASE WHEN FROM_ADDRESS = LOWER('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4') THEN ETH_VALUE ELSE 0 END) AS daily_outflow
  FROM base.core.fact_traces
  WHERE (TO_ADDRESS = LOWER('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4') OR FROM_ADDRESS = LOWER('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4')) AND TX_STATUS = 'SUCCESS'
  GROUP BY date
),
cumulative_traces AS (
  SELECT
    date,
    SUM(daily_inflow) OVER (ORDER BY date) AS cumulative_inflow,
    SUM(daily_outflow) OVER (ORDER BY date) AS cumulative_outflow
  FROM daily_traces
)
SELECT
  dt.date,
  dt.daily_inflow AS inflow,
  dt.daily_outflow AS outflow,
  (dt.daily_inflow - dt.daily_outflow) AS net_inflow,
  ct.cumulative_inflow - ct.cumulative_outflow AS cumulative_balance
FROM daily_traces dt
JOIN cumulative_traces ct ON dt.date = ct.date
ORDER BY dt.date;
"""

query1 = flipside.query(sql)

tvl = pd.DataFrame(query1.records)
tvl['date'] = pd.to_datetime(tvl['date'])


# In[ ]:


#Making bar chart for TVL change daily and area chart to show TVL each day on same plot

fig, ax1 = plt.subplots(figsize=(14, 7))


ax1.fill_between(tvl['date'], tvl['cumulative_balance'], label='Cumulative Balance', color='#6b6b6b', alpha=0.3)
ax1.set_xlabel('date')
ax1.set_ylabel('Cumulative Balance')

ax1.set_yticks(range(0, int(max(tvl['cumulative_balance'])) + 1, 2500))


positive_inflow = tvl[tvl['net_inflow'] > 0]
negative_inflow = tvl[tvl['net_inflow'] < 0]

cmap = plt.get_cmap('magma')

bar_colors_positive = cmap(0.2)  
bar_colors_negative = cmap(0.6)  


# Create bar charts for positive and negative net inflow on the right y-axis (y2)
ax2 = ax1.twinx()
ax2.bar(positive_inflow['date'], positive_inflow['net_inflow'], label='Positive Inflow', color=bar_colors_positive, alpha=0.7)
ax2.bar(negative_inflow['date'], negative_inflow['net_inflow'], label='Negative Inflow', color=bar_colors_negative, alpha=0.7)
ax2.set_ylabel('Net Inflow')

min_net_inflow = min(tvl['net_inflow'])
max_net_inflow = max(tvl['net_inflow'])

tick_spacing = 500

y2_ticks = range(int(min_net_inflow // tick_spacing) * tick_spacing, int(max_net_inflow // tick_spacing) * tick_spacing + tick_spacing, tick_spacing)

ax2.set_yticks(y2_ticks)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Net Inflow and Cumulative Balance Over Time')

date_labels = tvl['date'].dt.strftime('%d-%b')
ax1.set_xticks(tvl['date'])
ax1.set_xticklabels(date_labels, rotation=90)

plt.tight_layout()

#plt.savefig('tvl_per_day.png', dpi=300)

plt.show()


# In[ ]:


# Transactions

sql = """
SELECT DATE_TRUNC('day',BLOCK_TIMESTAMP) AS DATE, STATUS, 
Count(*) as Tx , SUM(Count(*)) OVER (ORDER BY DATE_TRUNC('day',BLOCK_TIMESTAMP)) AS Cumulative_Tx
from base.core.fact_transactions
where TO_ADDRESS = lower('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4')
GROUP BY 1,2
ORDER BY 1 ASC
"""

query2 = flipside.query(sql)
txn = pd.DataFrame(query2.records)
txn['date'] = pd.to_datetime(txn['date'])


# In[ ]:


# making status values a column 

pivot_df = txn.pivot_table(index='date', columns='status', values='tx', aggfunc='sum').fillna(0)
pivot_df.reset_index(inplace=True)
pivot_df['date'] = pd.to_datetime(pivot_df['date'])
pivot_df['day'] = pd.to_datetime(pivot_df['date']).dt.strftime('%A')


# In[ ]:


#Making Bar chart for transactions per day with status stacked on eachother

pivot_df = pivot_df.sort_values(by='date')
cmap = plt.get_cmap('magma')
colors = [cmap(0.6), cmap(0.3)]  

# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
pivot_df[['SUCCESS', 'FAIL']].plot(kind='bar', stacked=True, ax=ax, color=colors)
# Add labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Tx')
ax.set_title('Transcations per Day')

ax.legend(title='Status')

date_labels = pivot_df['date'].dt.strftime('%d-%b')
ax.set_xticklabels(date_labels)

# Set y-axis ticks with a spacing of 50,000
ax.set_yticks(range(0, 550000, 50000))

# Can add shading for FT week too if needed

# alternate_shading = False
# for i in range(len(pivot_df)):
#     if pivot_df['day'].iloc[i] == 'Friday':
#         if alternate_shading:
#             if i + 6 < len(pivot_df):
#                 ax.axvspan(i, i + 6, color='lightgray', alpha=0.5)
#             alternate_shading = not alternate_shading
#         else:
#             alternate_shading = not alternate_shading

plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
#plt.savefig('tx_per_day.png', dpi=300)

plt.show()


# In[ ]:


# Transaction Heat Map.

sql = """
SELECT DATE_TRUNC('hour', BLOCK_TIMESTAMP) AS HOUR,
       COUNT(*) AS Tx
FROM base.core.fact_transactions
WHERE TO_ADDRESS = lower('0xCF205808Ed36593aa40a44F10c7f7C2F67d4A4d4')
GROUP BY 1
ORDER BY 1 ASC
"""

query3 = flipside.query(sql)
txn_heat = pd.DataFrame(query3.records)
txn_heat['hour'] = pd.to_datetime(txn_heat['hour'])


# In[ ]:


# Adding EST timezone to data as EST provides more visible trend in data.

txn_heat['hour'] = pd.to_datetime(txn_heat['hour'], utc=True)  

est_timezone = pytz.timezone('US/Eastern')
txn_heat['hour_est'] = txn_heat['hour'].dt.tz_convert(est_timezone)


txn_heat['hour'] = pd.to_datetime(txn_heat['hour'])
txn_heat['dayOfweek'] = txn_heat['hour'].dt.day_name()
txn_heat['hourOfday'] = txn_heat['hour'].dt.hour

txn_heat['hour_est'] = pd.to_datetime(txn_heat['hour_est'])
txn_heat['dayOfweek_est'] = txn_heat['hour_est'].dt.day_name()
txn_heat['hourOfday_est'] = txn_heat['hour_est'].dt.hour


# In[ ]:


#following is using UTC time zone to make EST one replace hour with hour_est, hourOfday with hourOfday_EST
# & dayOfweek with dayOfweek_est

#grouping data by day and week and counting occurrences (would be used to normalize data )

# Group the data by day of the week and hour of the day and count occurrences
day_hour_counts = txn_heat.groupby(['dayOfweek', 'hourOfday']).agg({'tx': 'sum', 'hour': 'count'}).reset_index()
day_hour_counts.columns = ['dayOfweek', 'hourOfday', 'TotalTX', 'Occurrence']

day_hour_counts['Normalized_TX'] = (day_hour_counts['TotalTX'] / day_hour_counts['Occurrence']) * day_hour_counts['Occurrence'].max()

pivot_data = day_hour_counts.pivot_table(values='Normalized_TX', index='hourOfday', columns='dayOfweek', aggfunc='sum', fill_value=0)

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data[days_order]

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, cmap='magma', annot=True, fmt='.2f', linewidths=1)

plt.title('Normalized Transaction Sum UTC')
plt.xlabel('Day of the Week')
plt.ylabel('hour of the Day')

#plt.savefig('heat_utc_normal.png', dpi=300)

# Show the heatmap
plt.show()


# In[ ]:


# Trading Volume Bar Chart & More

sql = """
SELECT
    DATE_TRUNC('day', BLOCK_TIMESTAMP) AS date,
    COUNT(*) AS txn,
    SUM(decoded_log:ethAmount::NUMBER / 1e18) AS volume,
    AVG(decoded_log:ethAmount::NUMBER / 1e18) AS avg_volume,
    SUM(CASE WHEN decoded_log:isBuy::string = true THEN decoded_log:ethAmount::NUMBER / 1e18 ELSE 0 END) as buy_vol,
    SUM(CASE WHEN decoded_log:isBuy::string = true THEN 1 ELSE 0 END) as buy_count,
    AVG(CASE WHEN decoded_log:isBuy::string = true THEN decoded_log:ethAmount::NUMBER / 1e18 ELSE NULL END) as avg_buy_vol,
    SUM(CASE WHEN decoded_log:isBuy::string = false THEN decoded_log:ethAmount::NUMBER / 1e18 ELSE 0 END) as sell_vol,
    SUM(CASE WHEN decoded_log:isBuy::string = false THEN 1 ELSE 0 END) as sell_count,
    AVG(CASE WHEN decoded_log:isBuy::string = false THEN decoded_log:ethAmount::NUMBER / 1e18 ELSE NULL END) as avg_sell_vol
FROM
    base.core.fact_decoded_event_logs 
WHERE
    contract_address = '0xcf205808ed36593aa40a44f10c7f7c2f67d4a4d4'
    AND block_timestamp > '2023-08-05'
GROUP BY 1
ORDER BY date;

"""

query4 = flipside.query(sql)

vol = pd.DataFrame(query4.records)
vol['date'] = pd.to_datetime(vol['date'])


# In[ ]:


# Making Plot with line chart for average volume per transactions and bars showing daily tradind volume
# here only successful transactions are used

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(vol['date'], vol['avg_volume'], label='Average txn Amount', color='black')

ax1.set_xlabel('date')
ax1.set_ylabel('Average Transaction Amount', color='black')

ax1.set_ylim(0, max(vol['avg_volume']) + 0.01)  # Adjust the upper limit slightly

ax1.tick_params(axis='y', labelcolor='black')

# Format x-axis labels as '%d-%b'
date_labels = vol['date'].dt.strftime('%d-%b')
ax1.set_xticks(vol['date'])
ax1.set_xticklabels(date_labels, rotation=90, ha='right')

ax2 = ax1.twinx()

ax2.bar(vol['date'], vol['volume'], label='volume', color=plt.get_cmap('magma')(0.2), alpha=0.7)

ax2.set_ylabel('volume', color='black')

ax2.set_ylim(0, max(vol['volume']) + 250)  # Adjust the upper limit slightly for 'txn'

ax2.tick_params(axis='y', labelcolor='black')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')

plt.title('Average Transaction Amount and Transaction Over Time')

plt.tight_layout()
#plt.savefig('txnsize.png', dpi=300)

plt.show()


# In[ ]:


# Line chart for average trading volume per transactions and same for buys and sells
fig, ax = plt.subplots(figsize=(12, 6))

cmap = plt.get_cmap('magma')
colors = [cmap(i) for i in [1, 0.3, 0.6]]  

# Plot lines for 'avg_volume', 'avg_buy_vol', and 'avg_sell_vol'
x = np.arange(len(vol))
for i, column in enumerate(['avg_volume', 'avg_buy_vol', 'avg_sell_vol']):
    if column == 'avg_volume':
        label = 'Average Volume per Transaction'
    elif column == 'avg_buy_vol':
        label = 'Average Buy per Transaction'
    elif column == 'avg_sell_vol':
        label = 'Average Sell per Transaction'
    ax.plot(x, vol[column], label=label, color=colors[i])


ax.set_xlabel('date')
ax.set_ylabel('Value')
ax.set_title('Line Chart for Volume per Transaction, Volume per Buy and Volume per Sell')

ax.set_xticks(x)
date_labels = pd.to_datetime(vol['date']).dt.strftime('%d-%b')
ax.set_xticklabels(date_labels, rotation=90, ha='right')

ax.set_yticks(np.arange(0, 0.15, 0.01))

ax.legend()

plt.tight_layout()

#plt.savefig('txntype.png', dpi=300)

plt.show()


# In[ ]:


# Stacked bar chart to see how what % of daily transactions were buys or sells 
# same can be made for trading volume distribution between buys and sells by using buy_vol & sell_vol

vol['TOTAL_COUNT'] = vol['buy_count'] + vol['sell_count']

# Calculate percentages for BUY_VOL and SELL_VOL
vol['buy_count_%'] = (vol['buy_count'] / vol['TOTAL_COUNT']) * 100
vol['sell_count_%'] = (vol['sell_count'] / vol['TOTAL_COUNT']) * 100

cmap = plt.get_cmap('magma')
colors = [cmap(0.3), cmap(0.6)]

ax = vol[['buy_count_%', 'sell_count_%']].plot(kind='area', stacked=True, title='100% Stacked Area Chart', figsize=(12, 6), color=colors, alpha=0.7)

ax.axhline(y=50, color='black', linestyle='--', label='50%', alpha=0.8)


ax.set_ylabel('Percent (%)')
ax.margins(0, 0)  # Set margins to avoid "whitespace"

ax.set_yticks(range(0, 101, 10))

date_labels = pd.to_datetime(vol['date']).dt.strftime('%d-%b')
ax.set_xticks(range(len(date_labels)))
ax.set_xticklabels(date_labels, rotation=90, ha='right')

plt.tight_layout()
#plt.savefig('stackedtx.png', dpi=300)
plt.show()


# In[ ]:


# Trading Volume Heat Map.

sql = """
SELECT
    DATE_TRUNC('hour', BLOCK_TIMESTAMP) AS hour,
    SUM(decoded_log:ethAmount::NUMBER / 1e18) AS volume,
    SUM(decoded_log:protocolEthAmount::NUMBER / 1e18) AS fees,
    SUM(SUM(decoded_log:ethAmount::NUMBER / 1e18)) OVER (ORDER BY HOUR) AS cumulative_volume,
    SUM(SUM(decoded_log:protocolEthAmount::NUMBER / 1e18)) OVER (ORDER BY HOUR) AS cumulative_fees
FROM
    base.core.fact_decoded_event_logs
WHERE
    contract_address = '0xcf205808ed36593aa40a44f10c7f7c2f67d4a4d4'
    AND block_timestamp > '2023-08-05'
GROUP BY hour
ORDER BY hour;
"""

query5 = flipside.query(sql)


# In[ ]:


vol_heat = pd.DataFrame(query5.records)
vol_heat['hour'] = pd.to_datetime(vol_heat['hour'])


# In[ ]:


# Adding EST timezone to data as EST provides more visible trend in data.

vol_heat['hour'] = pd.to_datetime(vol_heat['hour'], utc=True)  

est_timezone = pytz.timezone('US/Eastern')
vol_heat['hour_est'] = vol_heat['hour'].dt.tz_convert(est_timezone)


vol_heat['hour'] = pd.to_datetime(vol_heat['hour'])
vol_heat['dayOfweek'] = vol_heat['hour'].dt.day_name()
vol_heat['hourOfday'] = vol_heat['hour'].dt.hour

vol_heat['hour_est'] = pd.to_datetime(vol_heat['hour_est'])
vol_heat['dayOfweek_est'] = vol_heat['hour_est'].dt.day_name()
vol_heat['hourOfday_est'] = vol_heat['hour_est'].dt.hour


# In[ ]:


#following is using UTC time zone to make EST one replace hour with hour_est, hourOfday with hourOfday_EST
# & dayOfweek with dayOfweek_est

#grouping data by day and week and counting occurrences (would be used to normalize data )

# Group the data by day of the week and hour of the day and count occurrences
day_hour_counts = vol_heat.groupby(['dayOfweek', 'hourOfday']).agg({'volume': 'sum', 'hour': 'count'}).reset_index()
day_hour_counts.columns = ['dayOfweek', 'hourOfday', 'TotalVolume', 'Occurrence']

day_hour_counts['Normalized_Volume'] = (day_hour_counts['TotalVolume'] / day_hour_counts['Occurrence']) * day_hour_counts['Occurrence'].max()

# Pivot the data to create a pivot table with normalized transaction data
pivot_data = day_hour_counts.pivot_table(values='Normalized_Volume', index='hourOfday', columns='dayOfweek', aggfunc='sum', fill_value=0)

# Reorder columns to have days in the desired order (e.g., Monday to Sunday)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data[days_order]

# Create the heatmap with normalized data
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, cmap='magma', annot=True, fmt='.2f', linewidths=1)

plt.title('Normalized Volume UTC')
plt.xlabel('Day of the Week')
plt.ylabel('hour of the Day')

#plt.savefig('heat_utc_vol.png', dpi=300)

# Show the heatmap
plt.show()