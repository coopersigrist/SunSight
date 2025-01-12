from data_load_util import *
from tqdm import tqdm


# Creates a projection of carbon offset if the current ratio of panel locations remain the same 
# allowing partial placement of panels in zips and not accounting in the filling of zip codes.
def create_continued_projection(combined_df, n=1000, metric='carbon_offset_metric_tons'):
    total_panels = np.sum(combined_df['existing_installs_count'])
    # print("total, current existing panels:", total_panels)
    panel_percentage = combined_df['existing_installs_count'] / total_panels
    ratiod_carbon_offset_per_panel = np.sum(panel_percentage * combined_df[metric])
    return np.arange(n+1) * ratiod_carbon_offset_per_panel

# Greedily adds 1-> n solar panels to zips which maximize the sort_by metric until no more can be added
# Returns the Carbon offset for each amount of panels added
def create_greedy_projection(combined_df, n=1000, sort_by='carbon_offset_metric_tons_per_panel', ascending=False, metric='carbon_offset_metric_tons_per_panel', record=True):
    sorted_combined_df = combined_df.sort_values(sort_by, ascending=ascending, inplace=False, ignore_index=True)
    projection = np.zeros(n+1)
    greedy_best_not_filled_index = 0
    existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]
    i = 0

    if record:
        picked = [sorted_combined_df['region_name'][greedy_best_not_filled_index]]

    while (i < n):
        if existing_count >= sorted_combined_df['count_qualified'][greedy_best_not_filled_index]:
            greedy_best_not_filled_index += 1
            existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]

        else:
            projection[i+1] = projection[i] + sorted_combined_df[metric][greedy_best_not_filled_index]
            existing_count += 1
            i += 1
            if record:
                picked.append(sorted_combined_df['region_name'][greedy_best_not_filled_index])
    
    return projection, picked

# Creates a projection which decides each placement alternating between different policies
def create_round_robin_projection(projection_list, picked_list):
    n = len(projection_list[0])
    number_of_projections = len(projection_list)
    projection = np.zeros(n)
    picked = [picked_list[0][0]]
    for i in range(1, n):
        chosen_projection = projection_list[i % number_of_projections]
        projection[i] = projection[i-1] + (chosen_projection[i] - chosen_projection[i-1])
        picked.append(picked_list[i % number_of_projections][i])
    return projection, picked

# Creates the projection of a policy which weighs multiple different factors (objectives)
# and greedily chooses zips based on the weighted total of proportions to national avg. 
def create_weighted_proj(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weights=[1], metric='carbon_offset_metric_tons_per_panel'):

    new_df = combined_df
    new_df['weighted_combo_metric'] = combined_df[objectives[0]] * 0
    norm_df = normalise_df(combined_df,objectives)
    for weight, obj in zip(weights,objectives):
        # new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + (combined_df[obj] / np.mean(combined_df[obj])) * weight
        #lambda x: (x-combined_df[obj].min())/(combined_df[obj].max()-combined_df[obj].min()))
        new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + (norm_df[obj] ) * weight

    return create_greedy_projection(combined_df=new_df, n=n, sort_by='weighted_combo_metric', metric=metric)

# Creates a projection of the carbon offset if we place panels to normalize the panel utilization along the given "demographic"
# I.e. if we no correlation between the demographic and the panel utilization and only fous on that, how Carbon would we offset
# TODO
def create_pop_demo_normalizing_projection(combined_df, n=1000, demographic="black_prop", metric='carbon_offset_metric_tons_per_panel'):
    pass

# Creates a projection of carbon offset for adding solar panels to random zipcodes
# The zipcode is randomly chosen for each panel, up to n panels
def create_random_proj(combined_df, n=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n+1)
    picks = np.random.randint(0, len(combined_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):

        while math.isnan(combined_df[metric][pick]):
            pick = np.random.randint(0, len(combined_df[metric]))
        projection[i+1] = projection[i] + combined_df[metric][pick]

    return projection

# Places a panel at a zip code
def place_panel(new_df, zip, metric):

    zip_mask = new_df['region_name'] == zip
    new_df['existing_installs_count'] += zip_mask 

    return new_df[zip_mask][metric].values[0]

# Places a single panel using the lexicase strategy, parameters are ordered by lexicase random choice
def place_panel_lexicase(new_df, demographics, inverses, dfs, thresholds, metric):
    
    initial_df = dfs[demographics[0]] # Take DF which is sorted by the chosen primary feature
    initial_df = initial_df[initial_df['existing_installs_count'] < initial_df['count_qualified'] ] # Removes full (max number of panels reached) zips
    df = initial_df[abs(initial_df[demographics[0]] - initial_df[demographics[0]][0]) < thresholds[0]] # Remove all zips not within the threshold for that feature
    for demo, threshold, inverse in zip(demographics[1:], thresholds[1:], inverses[1:]):
        if len(df) == 1:
            chosen_zip = df['region_name'][0]
            return place_panel(new_df, chosen_zip, metric), chosen_zip
        
        # Resort and remove zips outside of threhold for following demographics
        df = df.sort_values(demo, ascending=inverse)
        df = initial_df[abs(initial_df[demo] - initial_df[demo][0]) <= threshold] 

    chosen_zip = np.random.choice(df['region_name'])
    return place_panel(new_df, chosen_zip, metric), chosen_zip

# Creates an entire projection based on lexicase choosing strategy over given demographics
def create_lexicase_proj(combined_df, n=1000, demographics=["black_prop", "carbon_offset_metric_tons_per_panel", "yearly_sunlight_kwh_kw_threshold_avg"], inverses=[False, False, False], thresholds=[0,0,0], metric="carbon_offset_metric_tons_per_panel"):
    
    proj = np.zeros(n+1)
    chosen_zips = np.zeros(n)
    sorted_dfs = {}
    new_df = pd.DataFrame.copy(combined_df, deep=True)
    
    for demo,inverse,threshold in zip(demographics,inverses, thresholds):
        sorted_dfs[demo] = new_df.sort_values(demo, ascending=inverse) # Create sorted dfs for each of the demographics to save on compute and space
        sorted_dfs[demo] = sorted_dfs[demo] [abs(sorted_dfs[demo][demo] - sorted_dfs[demo][demo][0]) <= threshold] # Remove all elements that aren't within the threshold for each demo metric
    
    print("Creating Lexicase Projection")

    # Each loop places one panel
    for i in tqdm(range(n)):
        # Chooses the random order of the metric testing
        order = np.arange(0, len(demographics))
        np.random.shuffle(order)
        ordered_demos = [demographics[i] for i in order]
        ordered_thresholds = [thresholds[i] for i in order] 
        ordered_inverses = [inverses[i] for i in order] 

        # Places a single panel
        metric_change, chosen_zip = place_panel_lexicase(new_df, demographics=ordered_demos, inverses=ordered_inverses, dfs=sorted_dfs, thresholds=ordered_thresholds, metric=metric)
        proj[i+1] = proj[i] + metric_change
        chosen_zips[i] = chosen_zip

    return proj, chosen_zips

# Creates multiple different projections and returns them
def create_projections(combined_df, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True):

    ## TODO remove rrtest (just for a new version of round robin)
    # if load and exists("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_weighted_"+metric+".csv") and exists("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_weighted_picked.csv"):
        # return pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_weighted_"+metric+".csv"), pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_weighted_picked.csv")
    
    picked = pd.DataFrame()
    proj = pd.DataFrame()
    print("Creating Continued Projection")
    proj['Status-Quo'] = create_continued_projection(combined_df, n, metric)
    print("Creating Greedy Carbon Offset Projection")
    proj['Carbon-Efficient'], picked['Carbon-Efficient'] = create_greedy_projection(combined_df, n, sort_by='carbon_offset_metric_tons_per_panel', metric=metric)
    print("Creating Greedy Average Sun Projection")
    proj['Energy-Efficient'], picked['Energy-Efficient'] = create_greedy_projection(combined_df, n, sort_by='yearly_sunlight_kwh_kw_threshold_avg', metric=metric)
    print("Creating Greedy Black Proportion Projection")
    proj['Racial-Equity-Aware'], picked['Racial-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='black_prop', metric=metric)
    print("Creating Greedy Low Median Income Projection")
    proj['Income-Equity-Aware'], picked['Income-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='Median_income', ascending=True, metric=metric)

    # proj['Lexicase'], _ = create_lexicase_proj(combined_df, n, demographics=['carbon_offset_metric_tons_per_panel','yearly_sunlight_kwh_kw_threshold_avg','black_prop', 'Median_income'], inverses=[False,False,False,True], thresholds=[10000,10000,10000,10000], metric=metric)

    print("Creating Round Robin Projection")
    proj['Round Robin'], picked['Round Robin'] = create_round_robin_projection(projection_list=
                                                                                                   [proj['Carbon-Efficient'], proj['Energy-Efficient'], proj['Racial-Equity-Aware'], proj['Income-Equity-Aware']],
                                                                                                   picked_list=
                                                                                                   [picked['Carbon-Efficient'], picked['Energy-Efficient'], picked['Racial-Equity-Aware'], picked['Income-Equity-Aware']])

    print("Creating Weighted Greedy Projection")
    proj['Weighted Greedy'], picked['Weighted Greedy'] = create_weighted_proj(combined_df, n, ['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop'], [0.4,1.6,1.6,1.2000000000000002], metric=metric)

    # uniform_samples = 10

    # print("Creating uniform random projection with", uniform_samples, "samples")

    # proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] = np.zeros(n+1)
    # for i in range(uniform_samples):
    #     proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] += create_random_proj(combined_df, n)/uniform_samples
    
    ## TODO remove rrtest (just for a new version of round robin)
    # if save:
    #     proj.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_"+metric+".csv",index=False)
    #     picked.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/projections_picked.csv", index=False)

    return proj, picked

# Searches over many different weight settings, with the first weight being set permenantly to 1 and the other two being set proportionally
# Returns a 2d array of projections (i.e. 3d array)

# def create_many_weighted(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weight_starts=[], weight_ends=[], number_of_samples=1, metric='carbon_offset_metric_tons_per_panel', save=None, load=None):

#     if exists(load):
#        return np.load(load)

#     all_projections = np.zeros((number_of_samples,number_of_samples,n+1))

#     for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
#         for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):

#             print("weighted proj number:", (i*number_of_samples + j))
            
#             all_projections[i][j],_ = create_weighted_proj(combined_df, n=n, objectives=objectives, weights=[1, weight1, weight2], metric=metric)
#             print(all_projections[i][j])
#     print("all proj")
#     print(all_projections)
#     if save is not None:
#         np.save(save, all_projections)

#     return all_projections
def create_many_weighted(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weight_starts=[], weight_ends=[], number_of_samples=1, metric='carbon_offset_metric_tons_per_panel', save=None, load=None):

    # if load is not None:
    #     if exists(load):
    #         return np.load(load)

    all_projections = {}
    #carbon offset, energy generation, black prop, median income
    for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
        for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):
            for k, weight3 in enumerate(np.arange(weight_starts[2], weight_ends[2], (weight_ends[2] - weight_starts[2]) / number_of_samples)):
                for l, weight4 in enumerate(np.arange(weight_starts[3], weight_ends[3], (weight_ends[3] - weight_starts[3]) / number_of_samples)):
            
                    print("weighted proj number:", (i*number_of_samples + j))
                    valuekey,_ = create_weighted_proj(combined_df, n=n, objectives=objectives, weights=[weight1, weight2, weight3, weight4], metric=metric)
                    key =str(valuekey[-1])
                    
                    all_projections[key] = [weight1,weight2,weight3,weight4]
            
    if save is not None:
        np.save(save, all_projections)
    print(all_projections)
    return all_projections