# Directory manipulation for relative notebook imports
import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)

# Relative import for relevant methods
from Visualization.plot_util import *
from Data.data_load_util import *
from Simulation.projections_util import *

# Data loading (including projections)
zips_df, state_df, pos_df = make_dataset(granularity='both', remove_outliers=False, load_dir_prefix='../Data/')
projections = create_projections(zips_df, state_df, n_panels=2000000, save="../Simulation/Projection_Saves/projections_2mil.pkl", load="../Simulation/Projection_Saves/projections_2mil.pkl")

plot_projections(projections[0:2], objective="Carbon Offset", save_dir_prefix="../", save_name='CarbonOffset_2mil')
plot_projections(projections[0:2], objective="Energy Potential", save_dir_prefix="../", save_name='EnergyPot_2mil')
# plot_projections(projections, objective="Income Equity", save_dir_prefix="../", save_name='IncomeEq_2mil')

# State map of carbon offset per panel added
plot_state_map(state_df, key='carbon_offset_kg_per_panel', fill_color='Blues', legend_name="Carbon Offset (Kg Per Panel)", save_dir_prefix="../Visualization/", show=False)
# plot_state_map(state_df, key='panel_utilization', fill_color='Blues', legend_name="Realized Potential", save_dir_prefix="../Visualization/", show=False)
# plot_state_map(state_df, key='Adjusted Payback Period (Years, under energy generation assumptions)', fill_color='Blues', legend_name="Adjusted Payback Period", save_dir_prefix="../Visualization/", show=False)


# geo_plot(zips_df['yearly_sunlight_kwh_kw_threshold_avg'], 'mint_r', "Yearly Average Sunlight", pos_df, save_dir_prefix="../Visualization/")
# geo_plot(zips_df['Median_income'], 'mint_r', "Median income", pos_df, save_dir_prefix="../Visualization/")
# geo_plot(zips_df['carbon_offset_kg_per_panel'], 'mint_r', "Carbon Offset (Kg) Per Panel", pos_df, save_dir_prefix="../Visualization/")
# geo_plot(zips_df['hispanic_prop'], 'mint_r', "Hispanic Population Proportion", pos_df, save_dir_prefix="../Visualization/")
# geo_plot(zips_df['black_prop'], 'mint_r', "Black Population Proportion", pos_df, save_dir_prefix="../Visualization/")
# geo_plot(zips_df['black_prop'] + zips_df['hispanic_prop'], 'mint_r', "Black or Hispanic Population Proportion", pos_df, save_dir_prefix="../Visualization/")

