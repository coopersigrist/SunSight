import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import pickle



#NOTE: all these objectives are HARDCODED
#TODO: generalize objectives
class MilpModel:
    def __init__(self, weights=[1,1,1,1]):
        self.weights = weights
        
    def set_weights(self, weights):
        self.weights = weights

    def get_placements(self, data_manager, objectives=None, num_panels=1000000):
        zips_df = data_manager.combined_df.copy(deep=True)
        num_zips = len(zips_df)

        #initialize constants for the objectives
        #get energy potential
        energy_potential_by_zip = data_manager.normalized_df['yearly_sunlight_kwh_kw_threshold_avg'].values
        carbon_offset_by_zip = data_manager.normalized_df['carbon_offset_metric_tons_per_panel'].values

        #high black prop flag
        black_prop_median = zips_df['black_prop'].median()
        zips_df['black_prop_flag'] = 0
        zips_df.loc[zips_df['black_prop'] > black_prop_median, 'black_prop_flag'] = 1
        high_black_prop_flag = zips_df['black_prop_flag'].values

        #high income flag
        income_median = zips_df['Median_income'].median()
        zips_df['income_flag'] = 0
        zips_df.loc[zips_df['Median_income'] > income_median, 'income_flag'] = 1
        high_income_flag = zips_df['income_flag'].values




        #optimize num panels in each zip
        zip_placements = [self.weights[0] * energy_potential_by_zip[i]/num_panels + 
                    self.weights[1] * carbon_offset_by_zip[i]/num_panels for i in range(num_zips)]

        zip_placement_bounds = data_manager.combined_df['count_qualified'].values.tolist()
        #aux vars for abs val when calculating equity
        auxilliary_vars = [-self.weights[2]/num_panels, -self.weights[3]/num_panels]

        total_objective = -np.array(zip_placements + auxilliary_vars)

        # Variable bound
        bounds = Bounds([0 for i in range(num_zips + 2)], zip_placement_bounds+ [np.inf, np.inf])

        # Constraints matrix and bounds
        A = np.array([
            [1 for i in range(num_zips)] + [0, 0], # sum of total panels is num_panels
            
            [-2 * high_black_prop_flag[i] for i in range(num_zips)] + [1, 0], #racial aux constraint
            [2 * high_black_prop_flag[i] for i in range(num_zips)] + [1, 0], #racial aux constraint

            [-2 * high_income_flag[i] for i in range(num_zips)] + [0, 1], #equity aux constraint
            [2 * high_income_flag[i] for i in range(num_zips)] + [0, 1], #equity aux constraint
        ])
        lb = np.array([num_panels, -num_panels, num_panels, -num_panels, num_panels])
        ub = np.array([num_panels, np.inf, np.inf, np.inf, np.inf])
        constraints = LinearConstraint(A, lb, ub)

        # Integer constraint
        integrality = np.array([1 for i in range(num_zips)] + [0, 0])  # 1 = integer, 0 = continuous

        # Solve MILP
        res = milp(c=total_objective, constraints=constraints, integrality=integrality, bounds=bounds)

        # Output result
        print("Status:", res.message)
        print("Objective value:", res.fun)
        # print("x =", res.x)
        panel_placements = {zips_df['region_name'][i]: res.x[i] for i in range(num_zips)}
        return panel_placements