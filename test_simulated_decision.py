
"""
Test script for simulated_decision.py
"""
from pickle import Unpickler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('/Users/asitaram/Documents/GitHub/Untitled/SunSight/agent_df_base_com_vt_revised.pkl', 'rb') as f:
    file = pd.read_pickle(f)
print(file.info())
print(file.head(1))
#file = pickle.Unpickler('/Users/asitaram/Documents/GitHub/Untitled/SunSight/agent_df_base_com_vt_revised.pkl').load()
'''df = pd.read_parquet("data.parquet")
df1 = pd.read_parquet("data_2.parquet")
df2 = pd.read_parquet("data_3.parquet")
df3 = pd.read_parquet("data_4.parquet")
df4 = pd.read_parquet("data_5.parquet")
df5= pd.read_parquet("data_6.parquet")
df5= pd.read_parquet("data_7.parquet")
df5= pd.read_parquet("data_8.parquet")
df5= pd.read_parquet("data_9.parquet")'''
print('here')
#parquet_files = ['/Users/asitaram/Documents/GitHub/Untitled/SunSight/data.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_2.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_3.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_4.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_5.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_6.parquet', 'data_7.parquet','data_8.parquet', 'data_9.parquet', 'data_10.parquet','data_11.parquet', 'data_12.parquet', 'data_13.parquet' ]
parquet_files=['data_10.parquet','data_11.parquet', 'data_12.parquet', 'data_13.parquet']
parquet_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
parquet_df = parquet_df.drop_duplicates(subset=["zip"], keep="first")
print(parquet_df['state_abbr'].unique())
print('LENGTH')
print(len(parquet_df))   
print(parquet_df.info())
state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0])
state_install_costs_adjusted_for_size = (panel_size_watts/7000) * state_install_costs
state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])
parquet_df['ov_multiplier'] = parquet_df['']
#print(parquet_df[parquet_df['state_abbr'] == 'AK'].head(2)['phi'].values.flatten())
#print(parquet_df.iloc[0]['zip']) 
#print(np.array_equal(parquet_df.iloc[2]['npv_vals'], parquet_df.iloc[3]['npv_vals'])) # show first few rows
'''print(df.info())
print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())'''

#small_parq = pd.DataFrame('data_15.parquet')

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print('in here')
    from SunSight.Models.Incentives.simulated_decision import SolarAdoptionModelZipCode
    
    # Test with a sample zipcode
    print("Testing SolarAdoptionModelZipCode...")
    
    # Use a zipcode that might exist in your data
    test_zipcode = 10001  # New York zipcode
    
    model = SolarAdoptionModelZipCode(test_zipcode)
    print(f"Created model for zipcode: {test_zipcode}")
    
    # Try to generate agents
    try:
        model.generate_agents()
        print(f"Generated {len(model.agents)} agents")
        
        # Test getting discount rates
        model.get_all_needed_discount_rates()
        print("Successfully calculated needed discount rates")
        
        # Test stepping the model
        model.step()
        print("Successfully stepped the model")
        
    except Exception as e:
        print(f"Error during agent generation: {e}")
        print("This might be due to missing data files or incorrect file paths")
    
    print("Test completed!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"Unexpected error: {e}") 