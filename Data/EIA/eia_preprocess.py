import pandas as pd
import numpy as np

eia_df = pd.read_csv('small_scale_solar_by_state_by_month.csv')

def conv_strings_to_floats(lst):
    new = [float(s.replace(",", "")) if type(s) == str else s for s in lst]
    return np.array(new)

for key in ['Residential_cap', 'Residential_gen','Commercial_cap','Commercial_gen','Industrial_cap','Industrial_gen', 'Total_cap', 'Total_gen']:
    eia_df[key] = conv_strings_to_floats(eia_df[key])

eia_df.to_csv('small_scale_solar_by_state_by_month.csv', index=False)

jan_df = eia_df[eia_df['Month'] == 1]
jan_2024_df = jan_df[jan_df['Year'] == 2024]
jan_2025_df = jan_df[jan_df['Year'] == 2025]


comp_df = pd.DataFrame(jan_2024_df['State'])
for key in ['Residential_cap', 'Residential_gen','Commercial_cap','Commercial_gen','Industrial_cap','Industrial_gen', 'Total_cap', 'Total_gen']:
    comp_df[key + "_24"] = jan_2024_df[key].values
    comp_df[key + "_25"] = jan_2025_df[key].values
    

totals = {'24_res-cap': np.sum(jan_2024_df['Residential_cap']), '24_res-gen': np.sum(jan_2024_df['Residential_gen']), '25_res-cap': np.sum(jan_2025_df['Residential_cap']), '25_res-gen': np.sum(jan_2025_df['Residential_gen'])}

comp_df['Residential_cap_prop_24'] = jan_2024_df['Residential_cap'].values / totals['24_res-cap']
comp_df['Residential_gen_prop_24'] = jan_2024_df['Residential_gen'].values / totals['24_res-gen']
comp_df['Residential_cap_prop_25'] = jan_2025_df['Residential_cap'].values / totals['25_res-cap']
comp_df['Residential_gen_prop_25'] = jan_2025_df['Residential_gen'].values / totals['25_res-gen']

comp_df['Residential_added_cap'] = comp_df['Residential_cap_25'] - comp_df['Residential_cap_24']
comp_df['Residential_added_gen'] = comp_df['Residential_gen_25'] - comp_df['Residential_gen_24']

comp_df['prop_cap_added'] = comp_df['Residential_added_cap'] / np.sum(comp_df['Residential_added_cap'])

comp_df.to_csv('jan_24_25_by_state.csv', index=False)



