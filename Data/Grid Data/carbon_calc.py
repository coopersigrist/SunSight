import pandas as pd


# Get emissions per Grid
emissions_by_grid = pd.read_csv('emissions_by_grid.csv')
emissions_by_grid_dict = {grid: co2 for grid,co2 in zip(emissions_by_grid['Grid'],emissions_by_grid['CO₂e_nb'])} # CO2e_nb is nonbaseline CO2 emissions + carbon equiv emissions of other greanhouse gasses

# Get the grids used by each ZIP
grid_by_zip = pd.read_csv('grid_by_zip.csv')
grid_by_zip_dict = {zip_code: grid for zip_code, grid in zip(grid_by_zip['ZIP (character)'],grid_by_zip['eGRID Subregion #1'])}


# Get Solar Data
solar_df = pd.read_csv('../solar_by_zip.csv')[['region_name', 'count_qualified', 'yearly_sunlight_kwh_total','carbon_offset_metric_tons']]
solar_df = solar_df[solar_df['region_name'].isin(grid_by_zip_dict.keys())]
solar_df['Grid'] = [grid_by_zip_dict[zip] for zip in solar_df['region_name']]
solar_df['CO2e_nb_kg_per_MWh'] = [float(emissions_by_grid_dict[grid].replace(',', "")) * 0.4535924 for grid in solar_df['Grid']] 
solar_df['carbon_offset_metric_tons_new'] = solar_df['CO2e_nb_kg_per_MWh'] * solar_df['yearly_sunlight_kwh_total'] * 0.000001
solar_df['prop_diff'] = solar_df['carbon_offset_metric_tons_new'] / solar_df['carbon_offset_metric_tons']


# Calc Energy Gen by Zip
solar_df.to_csv('emissions_by_zip.csv', index=False)