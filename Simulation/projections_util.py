
# from data_manager import DataManager
# from data_load_util import *
from tqdm import tqdm
import matplotlib
import pickle
import os
import pandas as pd
import numpy as np
import math
from os.path import exists


# Projection Object that will be used to store the projections of different solar siting strategies
class Projection():

    def __init__(self, objective_projections: dict, panel_placements: dict, name:str):
        '''
        objective_projections: A dictionary, keyed by the name of objectives: strings, to dictionaries which are key'd by
            a number, int, of panels placed to the score on the corresponding objective. e.g.  objective_projections['Carbon Offset'] would be a dict like:
            {5: 2.5, 100: 80, ...} where after placing 5 panels 2.5 metric tons of carbon are predicted to be offset by this strategy, self.

        panel_placements: A dictionary key'd by zip codes, strings, to number of panels placed in that ZIP code, int, by this strategy, self.

        '''

        self.objective_projections = objective_projections
        self.panel_placements = panel_placements
        self.name = name

    def __str__(self):
        return "<Projection Object> of type: " + self.name
    
    def add_proj_to_plot(self, ax, objective: str, **kwargs):
        # Takes a matplotlib Axes object, ax, and adds the projection of a given objective to it
        objective_proj = self.objective_projections[objective]
        return ax.plot(objective_proj.keys(), objective_proj.values(), label=self.name, **kwargs)
    
class Objective():

    def __init__(self, name, func, **func_kwargs):
        '''
        name: A string which denotes which objective this is, this will be used in plotting and keying dictionaries
        func: This is the calculation of the objective given a full DF of all ZIP codes, and a dictionary of picked panels (see panel_placements in Projection class)
              Generally wrapped as a specific call (see self.calc) 
        '''
        self.name = name
        self.func = func
        self.func_kwargs = func_kwargs

    def calc(self, zip_df, panel_placements):
        # Wraps the given func with specific inputs (i.e. racial vs Income equity both use the calc_equity func)
        return self.func(zip_df, panel_placements, **self.func_kwargs)

# Creates a DF with updated values of existing installs, carbon offset potential(along with per panel), and realized potential
# After a set of picks (zip codes with a panel placed in them)
def updated_df_with_picks(zip_df:pd.DataFrame, placed_panels:dict):
    new_df = zip_df.copy(deep=True)
    new_existing = np.array(new_df['existing_installs_count'])

    for zip in placed_panels:
        index = list(new_df['region_name']).index(zip)
        new_existing[index] += placed_panels[zip]
    

    # print('Number install change:', np.sum(new_existing - new_df['existing_installs_count']) )
    new_df['existing_installs_count'] = new_existing
    new_df['existing_installs_count_per_capita'] = new_existing / new_df['Total_Population']
    new_df['panel_utilization'] = new_existing / new_df['number_of_panels_total']

    return new_df

#Calculates the equity of a given panel distribution, by default does racial equity over realized_potential
def calc_equity(zip_df, placed_panels, type="racial", by='panel_utilization'):

    if type == 'racial':
        metric = 'black_prop'
    elif type == 'income':
        metric = 'Median_income'
    else:
        print("Invalid type for equity calculation, defaulting to black_prop")
        metric = 'black_prop'

    zip_df = updated_df_with_picks(zip_df, placed_panels)
    
    metric_median = np.median(zip_df[metric])
    high_avg = np.mean(zip_df[zip_df[metric] > metric_median]['panel_utilization'].values)
    low_avg = np.mean(zip_df[zip_df[metric] < metric_median]['panel_utilization'].values)
    avg = np.mean(zip_df['panel_utilization'].values)

    return 2 - np.abs(high_avg-low_avg)/avg

# Calculates amount of a given metric (per panel metric) gained by placing panels according to picked starting with zip_df
def calc_obj_by_picked(zip_df, placed_panels, metric='carbon_offset_metric_tons_per_panel', cull=True):

    if cull:
        culled_df = zip_df[zip_df['region_name'].isin(placed_panels)]
    else:
        culled_df = zip_df
    
    total = 0
    for _, row in culled_df.iterrows():
        # print(placed_panels.keys())
        zip = row['region_name']
        total += row[metric] * placed_panels[zip]
    
    return total

# initializes a projection dict for each objective from a list of objectives
def init_objective_projs(zip_df, objectives:list[Objective]):

    placed_panel_init = {zip_code:0 for zip_code in zip_df['region_name']}
    objectives_proj = {obj.name : {0: obj.calc(zip_df, placed_panel_init)} for obj in objectives}

    return objectives_proj
    

# Creates a projection of carbon offset if the current ratio of panel locations remain the same 
# allowing partial placement of panels in zips and not accounting in the filling of zip codes.
def create_status_quo_projection(zip_df, n_panels:int=1000, objectives:list[Objective]=[], intervals=10):

    total_panels = np.sum(zip_df['existing_installs_count'])
    total_prop = n_panels / total_panels
    # zip_df['panel_prop'] = zip_df['existing_installs_count'] / total_panels

    objective_projections = {obj.name : {} for obj in objectives}

    panel_placements = {row['region_name']: row['existing_installs_count']*total_prop for _,row in zip_df.iterrows()}
    # interval_size = n_panels/intervals
    for obj in objectives:
        obj_val = obj.calc(zip_df, panel_placements=panel_placements)
        obj_ratio = obj_val/n_panels

        # Shortcut for calcing progressively increasing projections quickly, doesnt work for equity
        if obj.name not in ['Racial Equity', 'Income Equity']:
            objective_projections[obj.name] = { n:float(n*obj_ratio) for n in range(0, n_panels+1)}
        else:
            print(obj.name, obj_val)
            objective_projections[obj.name] = { n:float(obj_val) for n in range(0, n_panels+1)}

    sq_projection = Projection(objective_projections, panel_placements, name="Staus Quo")

    return sq_projection

# Creates a projection based on the ongoing installation data from SEIA TODO
def create_future_estimate_projection(zip_df, state_df, n_panels:int=1000, objectives:list[Objective]=[], intervals=10):
    '''
    This estimate of future installations first divides the potential added panels into proportions added to each State based on EIA data of added Capacity by State (see Data/EIA/details.md)
    Next the potential added panels are proportioned out further inside the states based on the existing install counts from project sunroof (see Data/Sunroof/details.md) 

    '''

    # Overall ratio of the panels added to each ZIP
    placement_ratio = {}

    for state in state_df['State']:

        # Calculates relevant proportions for current 'state'
        state_zip_df = zip_df[zip_df['state_name'] == state]
        state_prop_installs = float(state_df[state_df['State'] == state]['prop_cap_added'].values)
        zip_prop_installs = state_zip_df['existing_installs_count'] / np.sum(state_zip_df['existing_installs_count'])

        # Creates a list of tuples zip, estimated panel addition prop for current 'state'
        state_added_installs = zip(state_zip_df['region_name'], state_prop_installs * zip_prop_installs) 
        placement_ratio.update(dict(state_added_installs))

    objective_projections = init_objective_projs(zip_df,objectives)
    
    # Calculates a new batch of added panels for each of the intervals (1/interval of n_panels)
    for interval in tqdm(range(intervals - 1)):
        placed_panels = {zip_code:(placement_ratio[zip_code]*n_panels* (interval+1)/intervals) for zip_code in placement_ratio}
        for obj in objectives:
            objective_projections[obj.name].update({(n_panels) * (interval+1)/intervals : obj.calc(zip_df, placed_panels)})
    
    estimated_projection = Projection(objective_projections, placed_panels, name="Estimated Future Installations")

    return estimated_projection
    
# Greedily adds 1-> n solar panels to zips which maximize the sort_by metric until no more can be added
# Returns the Carbon offset for each amount of panels added
def create_greedy_projection(zip_df, n_panels=1000, sort_by='carbon_offset_metric_tons_per_panel', ascending=False, objectives:list[Objective]=[], name="Greedy"):
    
    # Initialize the projections dictionary
    projections = init_objective_projs(zip_df,objectives)

    # Sorts the combined DF by a given value (must be a col in zip_df)
    sorted_zip_df = zip_df.sort_values(sort_by, ascending=ascending, inplace=False, ignore_index=True)

    # greedy_best_not_filled is which index of the sorted array we will pick next, i is a counter
    greedy_best_not_filled_index = 0
    i = 0

    panel_placements = dict()

    while (i < n_panels):

        # calculates the amount that can be added to the chosen ZIP -- can only add up to n-i so only n panels are placed
        amount_to_add = min(n_panels-i, sorted_zip_df['count_qualified'][greedy_best_not_filled_index])
        zip = sorted_zip_df['region_name'][greedy_best_not_filled_index]
        
        # Updates our counter for check if we passed n
        i += amount_to_add

        # Update the dict of which zips were picked and how much
        panel_placements[zip] = amount_to_add

        # Calculates the value of each objective after placing all possible panels in the ZIP
        # Each objective function must take the zip_df and the picked dict only
        for objective in objectives:
            projections[objective.name][i] = objective.calc(zip_df, panel_placements)
        
        greedy_best_not_filled_index += 1

    greedy_proj = Projection(objective_projections=projections, panel_placements=panel_placements, name=name)

    return greedy_proj

# Given a panel_placements dict, gets the ZIP code of the first n placed panel.
def get_zips_of_first_nth_panels(n_panels:int, panel_placements:dict) -> dict:
    
    partial_panel_placements = dict()

    total = 0
    for val in panel_placements.values():
        total += val

    panel_counter = 0
    for zip in panel_placements:
        if panel_placements[zip] + panel_counter < n_panels:
            partial_panel_placements[zip] = panel_placements[zip]
            panel_counter += panel_placements[zip]
        else:
            partial_panel_placements[zip] = n_panels - panel_counter
            return partial_panel_placements
    
    raise ValueError("Tried to get zip of panel number "+ str(n_panels) + " but there were not that many placed panels in the given dict") 

# Makes a round robin projection (without placements) for a single objective 
def make_rr_proj_from_projs(zip_df:pd.DataFrame, panel_placements:dict, objectives:list[Objective]) -> dict:

    # setup
    rr_projection = {obj.name : dict() for obj in objectives}
    counted_placements = dict()

    panel_total = 0 
    for zip in panel_placements:
        counted_placements.update({zip:panel_placements[zip]}) # Adds a new element (zip: #panels) to the rr projection dict
        panel_total += panel_placements[zip]

        # updates each rr projection (key'd by obj.name) with a #panels : objective score after that many panels
        for obj in objectives:
            rr_projection[obj.name].update({panel_total : obj.calc(zip_df, counted_placements)})

    return rr_projection

# Creates a projection which decides each placement alternating between different policies
def create_round_robin_projection(zip_df, n_panels=1000, projections:list[Projection]=[], objectives:list[Objective]=[]):
    # Creates a Projection object via the round robin startegy over a give list of projections
    panel_placements = {}
    for i in range(len(projections)):

        # small issue here -- could over palce panels in a zip if multiple choose the same ZIP, TODO fix this.
        partial_panel_placement = get_zips_of_first_nth_panels(n_panels//len(projections), projections[i%len(projections)].panel_placements)

        for zip in partial_panel_placement:
            if zip in panel_placements:
                panel_placements[zip] += partial_panel_placement[zip]
            else:
                panel_placements[zip] = partial_panel_placement[zip]

    # Creates the actual projection for each objective
    objective_projections = make_rr_proj_from_projs(zip_df, panel_placements, objectives)
    
    rr_projection = Projection(objective_projections, panel_placements, name="Round Robin")

    return rr_projection

# Creates the projection of a policy which weighs multiple different factors (objectives)
# and greedily chooses zips based on the weighted total of proportions to national avg. 
def create_weighted_proj(zip_df, n_panels=1000, attributes=['carbon_offset_metric_tons_per_panel'], weights=[1], objectives:list[Objective]=[], scale='normalize'):

    new_df = zip_df
    new_df['weighted_combo_metric'] = zip_df[attributes[0]] * 0

    for weight, obj in zip(weights,attributes):
        if scale == 'avg':
            new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + (zip_df[obj] / np.mean(zip_df[obj])) * weight
        else:
            new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + ((zip_df[obj] - np.min(zip_df[obj])) / (np.max(zip_df[obj]) - np.min(zip_df[obj]))) * weight

    return create_greedy_projection(zip_df=new_df, n_panels=n_panels, sort_by='weighted_combo_metric', objectives=objectives)

# Creates the projection of a policy where the value function is determined by a NEAT model
def create_neat_proj(data_manager, n_panels=1000, model = None, objectives:list[Objective]=[], save=None, load=None):
    #load
    if load is not None and os.path.exists(load):
        print("Loading from previous calculations...")
        with open(load, 'rb') as dir:
            return pickle.load(dir)
    
    new_df = data_manager.combined_df.copy(deep=True)
    new_df['value'] = 0
    zip_values = model.run_network(data_manager)
    new_df['value'] = new_df['region_name'].map(zip_values)

    proj = create_greedy_projection(zip_df=new_df, n_panels=n_panels, sort_by='value', objectives=objectives, name="NEAT Model")
    #save
    if save is not None:
        with open(save, 'wb') as dir:
            pickle.dump(proj, dir, pickle.HIGHEST_PROTOCOL)

    return proj


# Creates a projection of carbon offset for adding solar panels to random zipcodes
# The zipcode is randomly chosen for each panel, up to n panels
''' TODO REFACTOR (low prio)'''
def create_random_proj(zip_df, n_panels=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n_panels+1)
    picks = np.random.randint(0, len(zip_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):

        while math.isnan(zip_df[metric][pick]):
            pick = np.random.randint(0, len(zip_df[metric]))
        projection[i+1] = projection[i] + zip_df[metric][pick]

    return projection

# Creates multiple different projections and returns them
def create_projections(zip_df:pd.DataFrame, state_df:pd.DataFrame = None, n_panels:int=1000, objectives='paper', save=None, load=None) -> list[Projection]:

    if load is not None and os.path.exists(load):
        print("Loading from previous simulation...")
        with open(load, 'rb') as dir:
            return pickle.load(dir)

    if objectives == 'paper':
        objectives = create_paper_objectives()

    proj = []
    print("Creating Status-Quo Projection")
    proj.append(create_status_quo_projection(zip_df, n_panels, objectives=objectives))
    print("Creating Continued Projection")
    proj.append(create_future_estimate_projection(zip_df, state_df, n_panels, objectives=objectives))
    print("Creating Greedy Carbon Offset Projection")
    proj.append(create_greedy_projection(zip_df, n_panels, sort_by='carbon_offset_metric_tons_per_panel', objectives=objectives, name="Carbon Aware"))
    print("Creating Greedy Average Sun Projection")
    proj.append(create_greedy_projection(zip_df, n_panels, sort_by='yearly_sunlight_kwh_kw_threshold_avg', objectives=objectives, name="Energy Aware"))
    print("Creating Greedy Black Proportion Projection")
    proj.append(create_greedy_projection(zip_df, n_panels, sort_by='black_prop', objectives=objectives, name="Racial-Equity Aware"))
    print("Creating Greedy Low Median Income Projection")
    proj.append(create_greedy_projection(zip_df, n_panels, sort_by='Median_income', ascending=True, objectives=objectives, name="Income-Equity Aware"))

    # print("Creating Round Robin Projection")
    # proj.append(create_round_robin_projection(zip_df, projections=proj[1:5],
    #                                                     n_panels=n_panels,
    #                                                     objectives=objectives))


    if save is not None:
        with open(save, 'wb') as dir:
            pickle.dump(proj, dir, pickle.HIGHEST_PROTOCOL)

    return proj

# Creates a list of the four objectives used in the paper
def create_paper_objectives() -> list[Objective]:
    carbon_offset = Objective(name = "Carbon Offset", func=calc_obj_by_picked, metric="carbon_offset_metric_tons_per_panel")
    energy_pot = Objective(name = "Energy Potential", func=calc_obj_by_picked, metric="yearly_sunlight_kwh_kw_threshold_avg")
    racial_equity = Objective(name = "Racial Equity", func=calc_equity, type="racial")
    income_equity = Objective(name = "Income Equity", func=calc_equity, type="income")

    return [carbon_offset, energy_pot, racial_equity,income_equity]

#just create equity objectives
def create_equity_objectives() -> list[Objective]:
    racial_equity = Objective(name = "Racial Equity", func=calc_equity, type="racial")
    income_equity = Objective(name = "Income Equity", func=calc_equity, type="income")

    return [racial_equity,income_equity]

# does a gridsearch over a possible range of weight values for a list of attributes and calculates the score on multiple objectives, storing them in a CSV if save is given
''' TODO TEST '''
def linear_weighted_gridsearch(zip_df:pd.DataFrame, n_panels:int=1000, attributes:list[str]=[], max_weights=np.ones(4), n_samples:int=10, objectives:list[Objective]=[], save=None, load=None):
    
    if load is not None and exists(load):
       return pd.read_csv(load)
    
    all_scores = np.zeros((n_samples**(len(attributes)), len(attributes) + len(objectives)))
    weights = np.zeros(len(attributes))
    i = 0

    for i in tqdm(range(n_samples**(len(attributes)))):
        panel_placements = create_weighted_proj(zip_df, n_panels=n_panels, attributes=attributes, objectives=objectives, weights=weights).panel_placements
        obj_scores = [obj.calc(zip_df, panel_placements) for obj in objectives]

        all_scores[i] = np.append(weights, obj_scores)

        # Increment weights
        for j in range(len(weights)-1, -1, -1):
            if abs(weights[j]) < abs(max_weights[j] - max_weights[j] / n_samples):
                weights[j] += max_weights[j] / n_samples
                break
            else:
                weights[j] = 0

    scores_df = pd.DataFrame()

    for i, key in enumerate(attributes + [obj.name for obj in objectives]):
        scores_df[key] = all_scores.T[i]
    print(scores_df)

    if save is not None:   
        scores_df.to_csv(save, header=attributes+[obj.name for obj in objectives], index=False)

    return scores_df

# if __name__ == '__main__':
#     # TEST
#     zip_df = make_dataset(remove_outliers=True)
#     objectives = create_paper_objectives()
#     n_panels = 10000
#     test = linear_weighted_gridsearch(zip_df, n_panels=n_panels, 
#                                       attributes=['yearly_sunlight_kwh_kw_threshold_avg', 'carbon_offset_metric_tons_per_panel', 'black_prop', 'Median_income'], 
#                                       objectives=objectives, max_weights=np.array([2,2,-2,2]), n_samples=5, 
#                                       save="Projection_Data/weighted_gridsearch_"+str(n_panels)+".csv",
#                                       load="Projection_Data/weighted_gridsearch_"+str(n_panels)+".csv")

#     print(test)
