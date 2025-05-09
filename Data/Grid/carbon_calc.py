import pandas as pd


# Get emissions per Grid
emissions_by_grid = pd.read_csv('emissions_by_grid.csv')
emissions_by_grid_dict = {grid: co2 for grid,co2 in zip(emissions_by_grid['Grid'],emissions_by_grid['CO₂e_nb'])} # CO2e_nb is nonbaseline CO2 emissions + carbon equiv emissions of other greanhouse gasses

# Get the grids used by each ZIP
grid_by_zip = pd.read_csv('grid_by_zip.csv')
grid_by_zip_dict = {zip_code: grid for zip_code, grid in zip(grid_by_zip['ZIP (character)'],grid_by_zip['eGRID Subregion #1'])}


# Load Solar Data, will use some features to calc emissions, will also only use the zips from it
df = pd.read_csv('Clean_Data/solar_by_zip.csv')[['region_name', 'yearly_sunlight_kwh_total','carbon_offset_metric_tons']]
df = df[df['region_name'].isin(grid_by_zip_dict.keys())]

# Find grid of each ZIP (does not account for zips with more than one grid, defaults to primary)
df['Grid'] = [grid_by_zip_dict[zip] for zip in df['region_name']]

# Calc Carbon emission (+ other emission equivelant) per MWh for each zip
df['CO2e_nb_kg_per_MWh'] = [float(emissions_by_grid_dict[grid].replace(',', "")) * 0.4535924 for grid in df['Grid']] 

# Calc total possible offset (+ equivelant) if all possible panels built (based off yearly sunlight)
df['carbon_offset_metric_tons'] = df['CO2e_nb_kg_per_MWh'] * df['yearly_sunlight_kwh_total'] * 0.000001



# Calc Energy Gen by Zip
df.to_csv('emissions_by_zip.csv', index=False)