from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import warnings

# from projections_util import create_paper_objectives


#data manager
class DataManager:
    def __init__(self, combined_df,
                 fields=['Median_income', 'carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'realized_potential_percent', 'black_prop'], train=False):
        #Note: remove Republican_prop
        #fields needs to consider energy efficiency, equity, and carbon efficiency
        self.combined_df = combined_df
        # self.state_df = state_df
        self.num_zips = len(self.combined_df)
        self.num_train_zips = self.num_zips # number of training points
        self.fields = fields
        self.train = train
        self.synthesize_df()

        # self.objectives = create_paper_objectives()

        #save percentile thresholds
        self.racial_thresholds = None
        self.income_thresholds = None

        #for polynomial basis expansion for network inputs (unused)
        self.poly = PolynomialFeatures(degree=1)

    def synthesize_df(self): #create a full df per zip code
        #change the Washington, D.C. key in state_df to be compatible with combined_df
        # self.state_df.loc[47, 'State'] = "District of Columbia"

        #Merge political preference into combined_df
        # republian_prop = self.state_df[['State', 'Republican_prop']]
        # self.combined_df = self.combined_df.merge(republian_prop, left_on='state_name', right_on='State', how='left')

        #only use desired fields
        # self.combined_df = self.combined_df[self.fields]

        #normalize all inputs to [0,1] in new df
        self.normalized_df = (self.combined_df[self.fields] - self.combined_df[self.fields].min()) / (self.combined_df[self.fields].max() - self.combined_df[self.fields].min())
        # self.normalized_df['State'] = self.combined_df['State']

        #add existing installs count as not normalized
        self.normalized_df['existing_installs_count'] = self.combined_df['existing_installs_count'] 

        #by default set the training set to all data
        self.train_df = self.normalized_df
        self.test_df = pd.DataFrame()

        #k_fold validation stuff
        self.k_folds = []
        self.fold_num = 0
    
    #train-test split; call this once
    def train_test_split_wrapper(self, test_size=0.2, random_state=69):
        self.train_df, self.test_df = train_test_split(self.normalized_df, test_size=test_size, random_state=random_state)

    #generate k folds of data from the train data; call this once
    def generate_folds(self, k=5, random_state=69):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        self.k_folds = []

        for train_idx, test_idx in skf.split(self.train_df, self.train_df['State']):
            self.k_folds.append((train_idx, test_idx))

    #get the indices of a specific fold
    def get_fold_indices(self, num=None):
        #handle case where k folds haven't been set
        if len(self.k_folds) == 0:
            # warnings.warn("No folds has been set. Run generate_folds() before calling get_fold_indices().")
            return range(self.num_zips), []
        
        if num == None:
            num = self.fold_num #default to the currently selected fold
        
        return self.k_folds[num] #returns ([train indices], [test indices])
    
    #select a new fold
    def set_fold(self, num):
        self.fold_num = num


    #get all data associated with a zip code index
    def get_zip(self, ind, train=True):
        if train == False:
            return self.normalized_df.loc[ind, self.fields].tolist()
        else:
            return self.train_df.iloc[ind][self.fields].tolist()
        
    #return feature-engineered vector for a zip code index
    def network_inputs(self, id, train=True):
        #perform polynomial basis expansion
        inputs = self.poly.fit_transform([self.get_zip(id, train)])[0]
        return inputs.tolist()
    
    # #return feature-engineered vector from a zip code TODO
    # def network_inputs(self, zip_code, train=True):
    #     #perform polynomial basis expansion
    #     inputs = self.poly.fit_transform([self.get_zip(id, train)])[0]
    #     return inputs.tolist()
    
    #score the zip order based on various metrics
    # def score(self, zip_order, mode = "energy_generation", n = 1000, record=False, train=True):
    #    #use plot_projections_util objectives
    #    for obj in self.objectives:
    #        if obj.name == mode:
    #            return obj.calc(self.combined_df, zip_order)
