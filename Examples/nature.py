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
# projections = create_projections(zips_df, state_df, n_panels=2000000, save="../Simulation/Projection_Saves/projections_2mil.pkl", load="../Simulation/Projection_Saves/projections_2mil.pkl")

# State map of carbon offset per panel added
plot_state_map(state_df, key='carbon_offset_metric_tons_per_panel', fill_color='Blues', legend_name="Carbon Offset Metric Tons Per Panel", save_dir_prefix="../Visualization/")

# geo_plot(zips_df['carbon_offset_metric_tons_per_panel'], 'mint_r', "Carbon Offset Per Panel", pos_df)

