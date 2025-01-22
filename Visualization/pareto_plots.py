from plot_util import *
# from data_load_util import *
from projections_util import *
from tqdm import tqdm

n = 10000
objs = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity']

combined_df = combined_df = make_dataset(remove_outliers=True)
projections = create_projections(combined_df, n_panels=n)

sq_evals = {'Carbon Offset': projections[0].objective_projections['Carbon Offset'][n], 
            'Energy Generation': projections[0].objective_projections['Energy Potential'][n], 
            'Racial Equity': projections[0].objective_projections['Racial Equity'][n], 
            'Income Equity': projections[0].objective_projections['Income Equity'][n],
            'label':"Status Quo"}

max_rr_placed = max(projections[-1].objective_projections['Carbon Offset'].keys())

rr_evals = {'Carbon Offset': projections[-1].objective_projections['Carbon Offset'][max_rr_placed], 
            'Energy Generation': projections[-1].objective_projections['Energy Potential'][max_rr_placed], 
            'Racial Equity': projections[-1].objective_projections['Racial Equity'][max_rr_placed], 
            'Income Equity': projections[-1].objective_projections['Income Equity'][max_rr_placed],
            'label': "Round Robin"}

objectives = create_paper_objectives()
linear_evals_df = linear_weighted_gridsearch(combined_df, load='Projection_Data/weighted_gridsearch_'+str(n)+'.csv', )

create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Energy Potential', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Racial Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Energy Potential', 'Racial Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Energy Potential', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)
create_pareto_front_plots(linear_evals_df, 'Racial Equity', 'Income Equity', fit=2, others=[sq_evals, rr_evals], scale=sq_evals)