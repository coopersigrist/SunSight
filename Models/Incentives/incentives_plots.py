import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

parquet_df = pd.read_parquet('data_AK.parquet')
x_cost = parquet_df.head(1)['incentive_cost_x_1']
s = parquet_df.loc[0, 'incentive_cost_x_1']  # get a single cell (string)
s_clean = s.split("[",1)[1].rsplit("]",1)[0]  # remove brackets
arr = np.fromstring(s_clean, sep=" ")
print(arr)
#y_cost = parquet_df.head(1)['incentive_cost_y_1']
#print(data)
print(type(parquet_df.loc[0, 'incentive_cost_x_1']))
x = np.sort(x_cost)

#calculate CDF values
y = 1. * np.arange(len(x_cost)) / (len(x_cost) - 1)
plt.plot(x,y)
plt.show()
df = pd.read_csv('/Users/asitaram/Documents/GitHub/Untitled/SunSight/AK_small_results_9.csv')
df_2 = pd.read_csv('output_9.csv')
data = np.array(df['payback_periods'].values).flatten()
#print(data)
x = np.sort(data)

#calculate CDF values
y = 1. * np.arange(len(data)) / (len(data) - 1)

#plot CDF
plt.plot(x, y)
plt.title('AK payback CDF')
plt.xlabel('Payback Period (Years)')
plt.ylabel('CDF')
plt.vlines(df_2[df_2['State code'] == 'AK']['payback_period_status_quo'].values[0], 0, 1, color='red',label='status quo adoption (2022)')
plt.vlines(df_2[df_2['State code'] == 'AK']['payback_period_first_year_cutoff'].values[0], 0, 1, color='orange',label='no prior incentive (just adoption rate)')
plt.legend()
plt.show()
