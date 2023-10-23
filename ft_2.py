# In[1]:


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


# In[2]:


# Flipside api key 

sdk_api = "ADD_YOUR_API_KEY_HERE"

flipside = Flipside(sdk_api, "https://api-v2.flipsidecrypto.xyz")


# In[3]:


# Volume & Fees

sql = """
SELECT
    DATE_TRUNC('day', BLOCK_TIMESTAMP) AS date,
    SUM(decoded_log:ethAmount::NUMBER / 1e18) AS volume,
    SUM(decoded_log:protocolEthAmount::NUMBER / 1e18) AS fees,
    SUM(SUM(decoded_log:ethAmount::NUMBER / 1e18)) OVER (ORDER BY date) AS cumulative_volume,
    SUM(SUM(decoded_log:protocolEthAmount::NUMBER / 1e18)) OVER (ORDER BY date) AS cumulative_fees
FROM
    base.core.fact_decoded_event_logs
WHERE
    contract_address = '0xcf205808ed36593aa40a44f10c7f7c2f67d4a4d4'
    AND block_timestamp > '2023-08-05'
GROUP BY date
ORDER BY date;
"""

query1 = flipside.query(sql)

week = pd.DataFrame(query1.records)
week['date'] = pd.to_datetime(week['date'])


# In[4]:


week['date'] = pd.to_datetime(week['date'])

# Add a new column 'DAY_OF_WEEK' with the day names
week['Day_of_Week'] = week['date'].dt.strftime('%A')


# In[5]:


# Find the start date of the first reporting week (Friday)
start_date = week[week['Day_of_Week'] == 'Friday']['date'].min()

# Define a function to calculate the reporting week number
def calculate_rep_week(row):
    delta = row['date'] - start_date
    week_number = 1 + delta.days // 7
    return f'Week {week_number}'

# Add the 'rep_week' column
week['rep_week'] = week.apply(calculate_rep_week, axis=1)

week.loc[week['date'] == '2023-08-10', 'rep_week'] = 'Week 1'


# In[6]:


week1 = week.groupby('rep_week', as_index=False).agg({
    'volume': 'sum',
    'fees': 'sum'
})

week1['fees'] = week1['fees'].round(2)
week1['volume'] = week1['volume'].round(2)


# In[7]:


week1['week_number'] = week1['rep_week'].str.extract(r'(\d+)').astype(int)
week1 = week1.sort_values(by='week_number')

# Drop the temporary 'week_number' column
week1 = week1.drop(columns='week_number')
week1 = week1.reset_index(drop=True)


# In[8]:


# Making Bar chart for weekly trading Volume, same can be made for fees by replacing volume with fees

rep_week = week1['rep_week']
volume = week1['volume']

# Get the "magma" color map
cmap = plt.get_cmap('magma')

colors = cmap(0.5)

plt.figure(figsize=(10, 6))
bars = plt.bar(rep_week, volume, color=colors)

plt.ylabel('Volume')
plt.title('Volume (in ETH) by Friend Tech Week')


for bar, value in zip(bars, volume):
    plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom')

plt.yticks(np.arange(0, max(volume) + 1, 10000))


# Save the plot as an image file (e.g., PNG)
#plt.savefig('vol_week.png', dpi=300)

# Show the plot
plt.show()


# In[9]:


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

query2 = flipside.query(sql)

tvl_week = pd.DataFrame(query2.records)
tvl_week['date'] = pd.to_datetime(tvl_week['date'])


# In[10]:


tvl_week = tvl_week[tvl_week['date'] != '2023-08-10 00:00:00.000']

tvl_week['date'] = pd.to_datetime(tvl_week['date'])

# Filter by Thursday and rename the column
thursdays = tvl_week[tvl_week['date'].dt.day_name() == 'Thursday']
thursdays = thursdays.rename(columns={'cumulative_balance': 'TVL'})

# Reset the index (if needed)
thursdays.reset_index(drop=True, inplace=True)

thursdays = thursdays[['TVL']]

# Add a new column for rep_week
thursdays['rep_week'] = 'Week ' + (thursdays.index + 1).astype(str)


# In[11]:


# Making Bar chart for what TVL was at weekly FT points snapshot

rep_week = thursdays['rep_week']
TVL = thursdays['TVL']

TVL_increase_percentage = [0] + [(TVL[i] - TVL[i-1]) / TVL[i-1] * 100 for i in range(1, len(TVL))]


# Get the "magma" color map
cmap = plt.get_cmap('magma')

colors = cmap(0.5)

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(rep_week, TVL, color=colors)

# Add labels and title
plt.ylabel('TVL')
plt.title('TVL (in ETH) by Friend Tech Week')


# Add data labels on top of the bars with TVL and percentage increase
for i, bar in enumerate(bars):
    value = TVL[i]
    increase_percentage = TVL_increase_percentage[i]
    label = f'{value:.2f}\n(+{increase_percentage:.2f}%)' if i > 0 else f'{value:.2f}'
    plt.text(bar.get_x() + bar.get_width() / 2, value, label, ha='center', va='bottom')

plt.yticks(np.arange(0, max(TVL) + 5000, 2500))


#plt.savefig('tvl_week.png', dpi=300)

plt.show()