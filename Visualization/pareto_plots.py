from plot_util import *
# from data_load_util import *
from projections_util import *
from tqdm import tqdm

n = 10000
objs = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity']

combined_df = combined_df = make_dataset(remove_outliers=True)
projections, picked = create_projections(combined_df, n=n, load=False, metric='carbon_offset_metric_tons_per_panel', save=False)

sq_evals = {'Carbon Offset': create_continued_projection(combined_df, n=n, metric='carbon_offset_metric_tons_per_panel')[-1], 
            'Energy Generation': create_continued_projection(combined_df, n=n, metric='yearly_sunlight_kwh_kw_threshold_avg')[-1], 
            'Racial Equity': calc_equity(combined_df, type='racial'), 
            'Income Equity': calc_equity(combined_df, type='income'),
            'label':"Status Quo"}

new_df = df_with_updated_picks(combined_df, picked['Round Robin'])
rr_evals = {'Carbon Offset': calc_obj_by_picked(combined_df, list(picked['Round Robin']), metric='carbon_offset_metric_tons_per_panel', cull=True), 
            'Energy Generation': calc_obj_by_picked(combined_df, list(picked['Round Robin'].values), metric='yearly_sunlight_kwh_kw_threshold_avg', cull=True), 
            'Racial Equity': calc_equity(new_df, type='racial'), 
            'Income Equity': calc_equity(new_df, type='income'),
            'label': "Round Robin"}

linear_evals_df = linear_weighted_gridsearch(combined_df, load='Projection_Data/weighted_gridsearch_'+str(n)+'.csv')

create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Energy Generation', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Racial Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Energy Generation', 'Racial Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Energy Generation', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Racial Equity', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)