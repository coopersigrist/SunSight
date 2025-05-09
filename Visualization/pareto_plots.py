from plot_util import *
# from data_load_util import *
from SunSight.Simulation.projections_util import *
from tqdm import tqdm

n_panels = 2000000
n_samples=7
objs = ['Carbon Offset', 'Energy Generation', 'Racial Equity', 'Income Equity']

combined_df = combined_df = make_dataset(remove_outliers=True)
projections = create_projections(combined_df, n_panels=n_panels, save='Projection_Data/all_projections_'+str(n_panels)+'_panels.pkl', load='Projection_Data/all_projections_'+str(n_panels)+'_panels.pkl')

sq_evals = {'Carbon Offset': projections[0].objective_projections['Carbon Offset'][n_panels], 
            'Energy Potential': projections[0].objective_projections['Energy Potential'][n_panels], 
            'Racial Equity': projections[0].objective_projections['Racial Equity'][n_panels], 
            'Income Equity': projections[0].objective_projections['Income Equity'][n_panels],
            'label':"Status Quo"}

max_rr_placed = max(projections[-1].objective_projections['Carbon Offset'].keys())

rr_evals = {'Carbon Offset': projections[-1].objective_projections['Carbon Offset'][max_rr_placed], 
            'Energy Potential': projections[-1].objective_projections['Energy Potential'][max_rr_placed], 
            'Racial Equity': projections[-1].objective_projections['Racial Equity'][max_rr_placed], 
            'Income Equity': projections[-1].objective_projections['Income Equity'][max_rr_placed],
            'label': "Round Robin",
            'color': "salmon"}

neat_lex = {'Carbon Offset': 1.61 * sq_evals['Carbon Offset'], 
            'Energy Potential': 1 * sq_evals['Energy Potential'], 
            'Racial Equity': 1.23 * sq_evals['Racial Equity'], 
            'Income Equity': 1.41 * sq_evals['Income Equity'],
            'label': "NEAT-Lexicase",
            'color': "paleturquoise"}

neat_tournament = {'Carbon Offset': 1.69 * sq_evals['Carbon Offset'], 
            'Energy Potential': 0.97 * sq_evals['Energy Potential'], 
            'Racial Equity': 1.25 * sq_evals['Racial Equity'], 
            'Income Equity': 1.33 * sq_evals['Income Equity'],
            'label': "NEAT-Tournament (ours)",
            'color': "palegreen"}



print(rr_evals['Income Equity'])
print("Beginning Linear Gridsearch")

objectives = create_paper_objectives()
linear_evals_df = linear_weighted_gridsearch(combined_df, n_panels=n_panels, 
                                      attributes=['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop', 'Median_income'], 
                                      objectives=objectives, max_weights=np.array([2,2,2,-2]), n_samples=n_samples, 
                                      save="Linear_Weight_Data/weighted_gridsearch_"+str(n_samples)+"_samples_"+str(n_panels)+"_panels.csv",
                                      load="Linear_Weight_Data/weighted_gridsearch_"+str(n_samples)+"_samples_"+str(n_panels)+"_panels.csv")

others = [rr_evals, neat_lex]

create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Energy Potential', fit=2, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_CO_EP.csv")
# create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Racial Equity', fit=1, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_CO_RE.csv")
# create_pareto_front_plots(linear_evals_df, 'Carbon Offset', 'Income Equity', fit=1, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_CO_IE.csv")
# create_pareto_front_plots(linear_evals_df, 'Energy Potential', 'Racial Equity', fit=1, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_EP_RE.csv")
# create_pareto_front_plots(linear_evals_df, 'Energy Potential', 'Income Equity', fit=1, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_EP_IE.csv")
# create_pareto_front_plots(linear_evals_df, 'Racial Equity', 'Income Equity', fit=1, others=others, scale=sq_evals, load='Linear_Weight_Data/Pareto_opt/Pareto_opt_'+str(n_panels)+"_RE_IE.csv")