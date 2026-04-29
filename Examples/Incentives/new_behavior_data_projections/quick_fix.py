import pandas as pd
import numpy as np

placements = pd.read_csv("panels_-2000.csv")

placed = np.array(placements['panels'])
dic = {'panels_'+str(year)+'_years_-2000_incentive' : placed[(10194 * (year -1)) : (10194 * year)] for year in [1,2,3,4,5]}
new_df = pd.DataFrame(dic)
new_df['zip'] = placements['zip'].unique()
new_df.to_csv('-2000_projection.csv', index=False)



# for year in [2,3,4,5]:
#     new_df['panels_'+str(year)+'_years_-2000_incentive'] = placements['panels'][(10194 * (year -1)) : (10194 * year)]
#     print(sum(placements[placements['year'] == year]['panels']))
#     print(sum(new_df['panels_'+str(year)+'_years_-2000_incentive']))


# new_df.to_csv('-2000_projection.csv', index=False)
