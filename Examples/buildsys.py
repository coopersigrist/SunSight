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

ks = state_df[state_df['State'] == 'Kansas']

stats = ['Adjusted Payback Period (Years, under energy generation assumptions)', 'Solar_prop', 'carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'existing_installs_count_per_capita', 'panel_utilization', 'Republican_prop']
for stat in stats:
    print(stat, ":" , ks[stat])
    all_states = state_df[stat].values
    print(sum([state > ks[stat] for state in all_states]), "states are higher")

