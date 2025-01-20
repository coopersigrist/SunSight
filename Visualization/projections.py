from plot_util import *
from data_load_util import *
from projections_util import *
from tqdm import tqdm
from joblib import Parallel, delayed

combined_df = make_dataset(remove_outliers=True)
# max_num_added = 1850000
print(combined_df)
objectives = ['carbon_offset_kg_per_panel','energy_generation_per_panel','black_prop','Median_income','yearly_sunlight_kwh_kw_threshold_avg','carbon_offset_metric_tons_per_panel']
n = normalise_df(combined_df, objectives)

max_num_added = 100000
Energy_projections, Energy_picked = create_projections(combined_df, n=max_num_added, load=False, metric='energy_generation_per_panel')
Carbon_offset_projections, Carbon_offset_picked = create_projections(combined_df, n=max_num_added, load=False, metric='carbon_offset_kg_per_panel')
Income_equity_projections, Income_equity_picked = create_projections(combined_df, n=max_num_added, load=False, metric='Median_income')
Racial_equity_projections, Racial_equity_picked = create_projections(combined_df, n=max_num_added, load=False, metric='black_prop')

print(Carbon_offset_projections['Status-Quo'])
panel_estimations_by_year = [("Net-Zero" , 479000 * 3), ("  2030  ", 479000 * 1), ("  2034  ", 479000 * 2)]
 
def plot_projections(projections, panel_estimations=None, net_zero_horizontal=False, interval=1, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], upper_bound='Greedy Carbon Offset', ylabel=None):
    # print(plt.style.available)
    plt.style.use("seaborn-v0_8")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}
    print(projections)

    matplotlib.rc('font', **font)

    if net_zero_horizontal:
        two_mill_continued = np.array(projections['Status-Quo'])[479000 * 3]

    keys = projections.keys()
    x = np.arange((len(projections[keys[0]]) // interval) + 1) * interval

    if panel_estimations is not None:
        for label, value in panel_estimations:
            plt.vlines(value, np.array(projections[upper_bound])[-1]/18, np.array(projections[upper_bound])[-1], colors='darkgray' , linestyles='dashed', linewidth=2, alpha=0.7)
            plt.text(value - len(projections[upper_bound])/23, np.array(projections[upper_bound])[-1]/80, label, alpha=0.7, fontsize=25)

    if net_zero_horizontal:
        plt.hlines(two_mill_continued, 0, len(projections[upper_bound]), colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)
        plt.text(0, two_mill_continued*1.1, "Continued trend at\nNet-zero prediction", alpha=0.95, fontsize=18, color='black')

    for key,fmt in zip(keys,fmts):
        plt.plot(x, np.array(projections[key])[0::interval], fmt, label=key, linewidth=3, markersize=8, alpha=0.9)


    plt.locator_params(axis='x', nbins=8) 
    plt.locator_params(axis='y', nbins=8) 
    plt.yticks(fontsize=fontsize/(1.2))
    plt.xticks(fontsize=fontsize/(1.2))

    print("percent difference between continued and Carbon-efficient:", projections['Round Robin'].values[-1] / projections['Carbon-Efficient'].values[-1] )
    print("percent difference between continued and racially-aware:", projections['Racial-Equity-Aware'].values[-1] / projections['Status-Quo'].values[-1])
    
    for i, elem in enumerate(projections['Round Robin'].values):
        if net_zero_horizontal and elem > two_mill_continued:
            print("number of panels for net zero round robin:", i)
            print("percentage relative to status-quo:", i/(479000 * 3))
            break

    

    plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/1.5)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.tight_layout()
    plt.show()

# Plots a map of where the zip_codes picked are located
def plot_picked(combined_df, picked, metric, title=""):

    if metric is None:
        region_list = list(combined_df['region_name'])
        occurence_counts = picked.value_counts()
        times_picked = np.zeros_like(combined_df['region_name'])
        for pick in picked.unique():
            times_picked[region_list.index(pick)] += occurence_counts[pick]
        
        combined_df['times_picked'] = times_picked
        metric ='times_picked'

        dot_size_scale = (40 * times_picked[combined_df['times_picked']>0]/ (np.max(combined_df['times_picked'][combined_df['times_picked']>0]))) + 40     
    picked = picked.astype(str)

    geo_plot(combined_df['times_picked'][combined_df['times_picked']>0], color_scale='agsunset', title=title, edf=combined_df[combined_df['times_picked']>0], zipcodes=picked.unique(), colorbar_label="", size=dot_size_scale)

# Creates a DF with updated values of existing installs, carbon offset potential(along with per panel), and realized potential
# After a set of picks (zip codes with a panel placed in them)
def df_with_updated_picks(combined_df, picks, load=None, save=None):

    # if load is not None and exists(load):
    #     return pd.read_csv(load)

    new_df = combined_df
    new_co = np.array(new_df['carbon_offset_metric_tons'])
    new_existing = np.array(new_df['existing_installs_count'])

    for pick in tqdm(picks):
        index = list(new_df['region_name']).index(pick)
        new_co[index] -= new_df['carbon_offset_metric_tons_per_panel'][index]
        new_existing[index] += 1
    
    # print('carbon offset difference:', np.sum(new_df['carbon_offset_metric_tons'] - new_co))
    # new_df['carbon_offset_metric_tons'] = new_co
    # new_df['carbon_offset_kg'] = new_co * 1000
    print('Number install change:', np.sum(new_existing - new_df['existing_installs_count']) )
    new_df['existing_installs_count'] = new_existing
    new_df['existing_installs_count_per_capita'] = new_existing / new_df['Total_Population']
    new_df['panel_utilization'] = new_existing / new_df['number_of_panels_total']

    # if save is not None:
    #     new_df.to_csv(save, index=False)

    return new_df

#sped up version 
# def df_with_updated_picks(combined_df, picks, load=None, save=None):
#     # if load is not None and exists(load):
#     #     return pd.read_csv(load)

#     new_df = combined_df.copy()
#     new_co = new_df['carbon_offset_metric_tons'].to_numpy()
#     new_existing = new_df['existing_installs_count'].to_numpy()

#     # Create a mapping from region names to indices for faster access
#     # region_index_map = {region: index for index, region in enumerate(new_df['region_name'])}
#     # Update carbon offset and existing installs using vectorized operations
#     for pick in tqdm(picks):
#         # if pick in region_index_map:  # Check if the region exists in the map
#         index = list(new_df['region_name']).index(pick)
#         new_co[index] -= new_df['carbon_offset_metric_tons_per_panel'].iloc[index]
#         new_existing[index] += 1
#         # else:
#         #     print("no")

#     print('Number install change:', np.sum(new_existing - new_df['existing_installs_count']))
    
#     # Update DataFrame columns with new values
#     new_df['existing_installs_count'] = new_existing
#     new_df['existing_installs_count_per_capita'] = new_existing / new_df['Total_Population']
#     new_df['panel_utilization'] = new_existing / new_df['number_of_panels_total']

#     # if save is not None:
#     #     new_df.to_csv(save)
    
#     return new_df

def plot_demo_state_stats(new_df,save="/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/data_by_state_proj.csv"):
    # state_df = load_state_data(new_df, load=None, save=save)

    hatches=['o','o','o','o','o','x','x','x','x','x']
    annotate = False
    type = 'paper'
    stacked = False

    # bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income','Republican'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    # bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)
    # bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_kg", type=type, stacked=stacked, ylabel="Carbon Offset Potential (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True) 

    hatches=['o','o','o','o','x','x','x','x']

    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True) 
    # bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="carbon_offset_kg", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Potential Carbon Offset (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    # bar_plot_demo_split(new_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)

# plot_projections(Carbon_offset_projections, interval=100, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
# plot_projections(Energy_projections, interval=100, upper_bound='Energy-Gen', ylabel="Energy Gen (kW/h)")
# plot_projections(Income_equity_projections, interval=100, upper_bound='Income-Equity', ylabel="Income Equity")
# plot_projections(Racial_equity_projections, interval = 100, upper_bound='Black-Prop', ylabel="Racial Equity")


# plot_projections(Carbon_offset_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
# plot_projections(Energy_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Energy-Efficient', ylabel="Additional Energy Capacity (kWh)")

# print(Energy_picked[''])

# for key in ['Energy-Efficient', 'Carbon-Efficient', 'Racial-Equity-Aware', 'Income-Equity-Aware', 'Round Robin']:
#     plot_picked(combined_df, Energy_picked[key], None, title="")

# quit()

def weighted_proj_heatmap(combined_df, npanels = 10000, metric='carbon_offset_kg_per_panel', objectives=['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop']):
    weight_starts = [0.0, 0.0]
    weight_ends = [5,1.5]
    number_of_samples = 12 #should be called resolution, "calling it samples is dumb" - cooper
    if metric == 'carbon_offset_kg_per_panel':
        statquoproj = Carbon_offset_projections['Status-Quo'].values[-1]
        title = "Carbon Offset Potential after Adding " + str(npanels) + " Panels"
        filename = 'normweighted_map_carbonoffset_' + str(number_of_samples) +'res_'+str(npanels)+"panels_"+str(weight_ends[0])+str(weight_ends[1])+"w"
    else:
        statquoproj = Energy_projections['Status-Quo'].values[-1]
        title = "Energy Potential after Adding " + str(npanels) + " Panels"
        filename = 'normweighted_map_energy_' + str(number_of_samples) +'res_'+str(npanels)+"panels_"+str(weight_ends[0])+str(weight_ends[1])+"w"
    weighted_proj_array = create_many_weighted(combined_df, n=npanels, objectives=objectives, weight_starts=weight_starts, weight_ends=weight_ends, number_of_samples=number_of_samples, metric=metric,
                                               save='/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/' + filename, load='/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/' + filename+'.npy')
    print(weighted_proj_array)

    print(weighted_proj_array[:,:,-1])
    # print(Energy_projections['Status-Quo'])
    # print(np.round(np.arange(weight_starts[1],weight_ends[1], (weight_ends[1] - weight_starts[1])/number_of_samples), 2))
    ax = sns.heatmap(weighted_proj_array[:,:,-1], 
                                xticklabels=np.round(np.arange(weight_starts[1],weight_ends[1], (weight_ends[1] - weight_starts[1])/number_of_samples), 1), 
                                yticklabels=np.round(np.arange(weight_starts[0],weight_ends[0], (weight_ends[0] - weight_starts[0])/number_of_samples), 1),
                                annot=weighted_proj_array[:,:,-1]/statquoproj)
    ax.set_ylabel("Energy Potential Weight (Relative to Carbon Offset per Panel)")
    ax.set_xlabel("Black Prop Weight (Relative to Carbon Offset per Panel)")
    plt.title(title)
    plt.show()

# weighted_proj_heatmap(combined_df,npanels=10000,metric='carbon_offset_kg_per_panel')
# weighted_proj_heatmap(combined_df, npanels = 10000, metric='energy_generation_per_panel')

def dominates(sol1, sol2):
    """Check if sol1 dominates sol2."""
    return all(s1 >= s2 for s1, s2 in zip(sol1, sol2)) and any(s1 > s2 for s1, s2 in zip(sol1, sol2))


# def grid_search(combined_df, npanels, metrics, objectives, save = None, load = None):
#     """Perform grid search to find Pareto-optimal solutions."""
#     weight_starts = [0.0, 0.0, 0.0, 0.0]
#     weight_ends = [2.0, 2.0, 2.0, 2.0]
#     number_of_samples = 10
#     filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS" + str(number_of_samples)
#     results = {}

#     for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
#         for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):
#             for k, weight3 in enumerate(np.arange(weight_starts[2], weight_ends[2], (weight_ends[2] - weight_starts[2]) / number_of_samples)):
#                 for l, weight4 in enumerate(np.arange(weight_starts[3], weight_ends[3], (weight_ends[3] - weight_starts[3]) / number_of_samples)):

#                     key = ' '.join(str(x) for x in [weight1,weight2,weight3,weight4])
#                     results[key] = []

#     for metric in metrics:
#         print("current metric is" + metric)
#         for weights, projectvalue in results.items():
#             print("creating projection array for weights "+ weights)
#             weightarr = list(map(float, weights.split(" ")))
#             projection,picked = create_weighted_proj(combined_df, n=npanels, objectives=objectives, weights=weightarr, metric=metric)
#             if metric == 'black_prop' or metric == 'Median_income':
#                 new_picks_df = df_with_updated_picks(combined_df, picked)
#                 # print(new_picks_df)
#                 key = "panel_utilization"
#                 if metric == 'black_prop':
#                     demo = 'black_prop'
#                 else:
#                     demo = 'Median_income'
#                 median = np.median(new_picks_df[demo].values)
#                 low_avg = np.mean(new_picks_df[new_picks_df[demo] < median][key].values)
#                 high_avg = np.mean(new_picks_df[new_picks_df[demo] >= median][key].values)
#                 equity_score = np.abs(1-np.abs(high_avg-low_avg))
#                 projectvalue.append(equity_score)
#                 continue
#             if metric == 'carbon_offset_kg_per_panel':
#                 projectvalue.append(projection[-1]/Carbon_offset_projections['Status-Quo'].values[-1])
#                 continue
#             if metric == 'energy_generation_per_panel':
#                 projectvalue.append(projection[-1]/Energy_projections['Status-Quo'].values[-1])
#                 continue
#     data = []
#     for key, values in results.items():
#         weights = list(map(float, key.split()))
#         row = weights + values
#         data.append(row)
    
#     columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    
#     df = pd.DataFrame(data, columns=columns)
    
#     # Save the DataFrame to CSV
#     df.to_csv(filename, index=False)

#     return df
def grid_search(combined_df, npanels, metrics, objectives, save=None, load=None):
    """Perform grid search to find Pareto-optimal solutions."""
    weight_starts = [0.0, 0.0, 0.0, 0.0]
    weight_ends = [2.0, 2.0, 2.0, 2.0]
    number_of_samples = 6
    filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS" + str(number_of_samples)
    
    # Precompute grid of weights
    weight_values = np.linspace(weight_starts[0], weight_ends[0], number_of_samples)
    weight_grid = np.array(np.meshgrid(weight_values, weight_values, weight_values, weight_values)).T.reshape(-1, 4)

    results = {tuple(weights): [] for weights in weight_grid}

    # Preload projections
    co_baseline = Carbon_offset_projections['Status-Quo'].values[-1]
    energy_baseline = Energy_projections['Status-Quo'].values[-1]

    def process_metric(metric):
        """Process a single metric."""
        local_results = {}
        for weights in weight_grid:
            weights_tuple = tuple(weights)
            if weights[0]==weights[1]:
                continue
            if weights[0]==weights[1] and weights[1]==weights[2] and weights[2]==weights[3] and weights[0]!=1:
                continue
            projection, picked = create_weighted_proj(combined_df, n=npanels, objectives=objectives, weights=weights, metric=metric)
            if metric in ['black_prop', 'Median_income']:
                new_picks_df = df_with_updated_picks(combined_df, picked)
                key = "panel_utilization"
                demo = 'black_prop' if metric == 'black_prop' else 'Median_income'
                median = np.median(new_picks_df[demo].values)
                # print(new_picks_df[new_picks_df[demo] < median][key].values)
                low_avg = np.mean(new_picks_df[new_picks_df[demo] < median][key].values)
                high_avg = np.mean(new_picks_df[new_picks_df[demo] >= median][key].values)
                true_avg = np.mean(new_picks_df[key].values)
                #1 - ((1 - (1 - H-L)) / avg) = 1 - (H-L)/avg
                equity_score = np.abs(1 - (np.abs(high_avg - low_avg)/true_avg))
                print(true_avg)
                print("wah")
                print(equity_score)
                local_results[weights_tuple] = equity_score
            elif metric == 'carbon_offset_kg_per_panel':
                local_results[weights_tuple] = projection[-1] / co_baseline
            elif metric == 'energy_generation_per_panel':
                local_results[weights_tuple] = projection[-1] / energy_baseline
        return metric, local_results

    # Parallelise metric computation
    # metrics_results = Parallel(n_jobs=-1)(delayed(process_metric)(metric) for metric in metrics)
    metrics_results = process_metric("black_prop")
    # Combine results
    for metric, metric_results in metrics_results:
        for weights_tuple, value in metric_results.items():
            results[weights_tuple].append(value)

    # Create DataFrame
    data = [list(weights) + values for weights, values in results.items()]
    columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

    return df
def pareto_calc(df= pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS7")):  
    pareto_opt_solutions = []
    filename = "/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GSParetoOptSoln7"
    for index, row in df.iterrows():
        add = True
        for i, optsoln in enumerate(pareto_opt_solutions):
            # print(row[4:])
            # print(optsoln[4:])
            if dominates(row[4:],optsoln[4:]):
                del(pareto_opt_solutions[i])
            elif dominates(optsoln, row): 
                add = False
                break
        if add == True:
            pareto_opt_solutions.append(row)
        # print(pareto_opt_solutions)
    columns = ['w1', 'w2', 'w3', 'w4', 'CO (prop to SQ)', 'Energy gen (prop to SQ)', 'Income_EQ', 'racial_EQ']
    new_df = pd.DataFrame(pareto_opt_solutions, columns=columns)
    new_df.to_csv(filename, index=False)
    return df

def plot_pareto(pareto_df, solution_df, rr_co, rr_eng, x_metric, y_metric,x_rr, y_rr):
    """Plotting pareto front"""
    plt.figure(figsize=(10, 6))

    plt.scatter(solution_df[x_metric], solution_df[y_metric], label='All Points', alpha=0.5)
    plt.scatter(pareto_df[x_metric], pareto_df[y_metric], color='red', label='Pareto Optimal', s=80)
    print("wah")
    print(rr_co)
    plt.scatter(rr_co.values[-1]/Carbon_offset_projections['Status-Quo'].values[-1], rr_eng.values[-1]/Energy_projections['Status-Quo'].values[-1], color='green', marker='x', label='Round Robin Results', s=100)

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title('Pareto Front Visualization')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# metrics = ['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop','Median_income']
# objectives = ['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop','Median_income']
# df = grid_search(combined_df, npanels=max_num_added, metrics = metrics, objectives = objectives)
# print(df)
pareto_set = pareto_calc()
print(pareto_set)

# pareto_df = pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GSParetoOptSoln5")
# solution_df = pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/GS5")
# plot_pareto(pareto_df, solution_df, rr_co = Carbon_offset_projections['Round Robin'], rr_eng = Energy_projections['Round Robin'], x_metric= "CO (prop to SQ)", y_metric="Energy gen (prop to SQ)", x_rr ="carbon_offset_kg_per_panel", y_rr =  "energy_generation_per_panel")
# print("done")
# plot_projections(Carbon_offset_projections, interval=100, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
# plot_projections(Energy_projections, interval=100, upper_bound='Energy-Gen', ylabel="Energy Gen (kW/h)")

# coprojections, copicked = create_weighted_proj(combined_df,1000, objectives = objectives,weights = soln, metric = 'carbon_offset_kg_per_panel')
# engprojections, engpicked = create_weighted_proj(combined_df,1000, objectives = objectives,weights = soln, metric = 'energy_generation_per_panel')
# plot_projections(coprojections, interval=100, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
# plot_projections(engprojections, interval=100, upper_bound='Energy-Gen', ylabel="Energy Gen (kW/h)")

# co_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Carbon Offset'], load='Projection_Data/df_greedy_co.csv', save='Projection_Data/df_greedy_co.csv')
# round_robin_df = df_with_updated_picks(combined_df, Energy_picked['Round Robin Policy'], load='Projection_Data/df_greedy_rrtest_rr.csv', save='Projection_Data/df_greedy_rrtest_rr.csv')
# energy_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Average Sun'], load='Projection_Data/df_greedy_sun.csv', save='Projection_Data/df_greedy_sun.csv')
# black_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Black Proportion'], load='Projection_Data/df_greedy_black.csv', save='Projection_Data/df_greedy_black.csv')
# blackprojections, blackpicked = create_weighted_proj(combined_df,1000, objectives = objectives,weights = soln, metric = 'black_prop')
# incomeprojections, incomepicked = create_weighted_proj(combined_df,1000, objectives = objectives,weights = soln, metric = 'Median_income')

# blackweighted_df = df_with_updated_picks(combined_df, Racial_equity_picked['Weighted Greedy'], load='Projection_Data/df_greedy_weighted_gs.csv', save='/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/df_greedy_weighted_gs.csv')

# # # plot_demo_state_stats(round_robin_df, save="Projection_Data/data_by_state_proj_greedy_round_robink.csv")
# plot_demo_state_stats(blackweighted_df, save=None)

# incomeweighted_df = df_with_updated_picks(combined_df, Income_equity_picked['Weighted Greedy'], load='Projection_Data/df_greedy_weighted_gs.csv', save='/Users/mimilertsaroj/Desktop/SunSight/Visualization/Projection_Data/df_greedy_weighted_gs.csv')

# # plot_demo_state_stats(round_robin_df, save="Projection_Data/data_by_state_proj_greedy_round_robink.csv")
# plot_demo_state_stats(blackweighted_df, save=None)



quit()