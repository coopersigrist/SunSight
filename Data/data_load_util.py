import pandas as pd
from os.path import exists
import numpy as np
import pgeocode
from functools import reduce

# Simple convertion from string with commas to float -- used by EIA data load
def conv_strings_to_floats(lst):
    new = [float(s.replace(",", "")) if type(s) == str else s for s in lst]
    return np.array(new)

# Loads Solar data for given zipcodes, also cleans and calculates new values
def load_solar_data(zip_codes=None, load_dir="Clean_Data/sunroof_by_zip.csv", save=False):

    # If we have already cleaned data then we load that instead of processing
    if load_dir is not None and exists(load_dir):
        return pd.read_csv(load_dir)

    # Load the entire dataset
    df = pd.read_csv('Sunroof/solar_by_zip.csv')

    # Remove all Zipcodes not in the given list
    if zip_codes is not None:
        zip_codes = list(map(int, zip_codes))
        df = df[df["region_name"].isin(zip_codes)]

    # Remove duplicates and unused data features
    df = df.drop_duplicates(subset=['region_name'], keep='first')
    df = df.sort_values('region_name')
    df = df[['region_name','state_name','yearly_sunlight_kwh_kw_threshold_avg','yearly_sunlight_kwh_total','existing_installs_count','percent_covered','count_qualified', 'number_of_panels_total']]

    # We have to scale by "percent covered" as that is the percent of the zipcode area that has data, but census dat attempts to cover 100% of population
    # df['count_qualified'] *= (100/ df['percent_covered']) 
    # df['existing_installs_count'] *= (100/ df['percent_covered']) 

    df['estimated_max_panels'] = df['yearly_sunlight_kwh_total'] / df['yearly_sunlight_kwh_kw_threshold_avg']

    if save:
        df.to_csv(load_dir, index=False)

    return df

# Loads eGRID data for given zipcodes
def load_grid_data(zip_codes=None, load_dir="Clean_Data/grid_by_zip.csv", save=False):

    # If we have already cleaned data then we load that instead of processing
    if load_dir is not None and exists(load_dir):
        return pd.read_csv(load_dir)

    # Load all census data
    df = pd.read_csv('Grid/emissions_by_zip.csv')

    # Removes all data not in the zip_codes list
    if zip_codes is not None:
        zip_codes = list(map(int, zip_codes))
        df = df[df["region_name"].isin(zip_codes)]

    # Also removes duplicates (which happen for some reason even when there are no duplicates in zips)
    df = df.drop_duplicates(subset=['region_name'], keep='first')
    df = df.sort_values('region_name')

    if save:
        df.to_csv(load_dir, index=False)

    return df

# Loads Cenus data for given zipcodes, also cleans and calculates new values
def load_census_data(zip_codes=None, load_dir="Clean_Data/census_by_zip.csv", save=False):

    # If we have already cleaned data then we load that instead of processing
    if load_dir is not None and exists(load_dir):
        return pd.read_csv(load_dir)

    # Load all census data
    df = pd.read_csv('Census/census_by_zip.csv')

    # Removes bad data, should be already removed from the zip.csv, but this to be certain.
    mask = df['Median_income'] <= 0
    df = df[~mask]

    # Removes all data not in the zip_codes list
    if zip_codes is not None:
        zip_codes = list(map(int, zip_codes))
        df = df[df["zcta"].isin(zip_codes)]

    # Also removes duplicates (which happen for some reason even when there are no duplicates in zips)
    df = df.drop_duplicates(subset=['zcta'], keep='first')
    df = df.sort_values('zcta')
    df.rename(columns={'zcta':'zip'}, inplace=True)

    if save:
        df.to_csv(load_dir, index=False)

    return df

def load_state_energy_dat(keys= ['Clean', 'Bioenergy', 'Coal','Gas','Fossil','Solar','Hydro','Nuclear','Total Generation'], load_dir="Clean_Data/energy_stats_by_state.csv", total=True, save=True):

    if load_dir is not None and exists(load_dir):
        df = pd.read_csv(load_dir) 
        return df
    
    df = pd.read_csv('Grid/energy_stats_by_state.csv') 
    solar_data = df[['State', 'State code', 'Variable', 'Value', 'Category']]

    # Mask out Puerto Rico (not enough other data)
    mask = solar_data['State'].isin(["Puerto Rico", "Washington, D.C."])
    df = solar_data[~mask]

    if not total:
        df = df[~(df['State'] == "US Total")]

    # This can change but for now we only care about generation
    mask2 = df['Category'] == 'Electricity generation'
    df = df[mask2]
    state_list = df['State'].unique()
    state_code_list = df['State code'].unique()

    # Types of energy generation that we will load
    energy_list = keys 

    new_df_dict = {'State' : state_list, "State code" : state_code_list}
    new_df = pd.DataFrame(new_df_dict)

    # This all reformats the data to have only a single row per state
    for state in state_list:
        mask = df['State'] == state
        temp_df = df[mask]

        for var in energy_list:
            if var not in new_df_dict.keys():
                new_df_dict[var] = []

            if var not in temp_df['Variable'].values:
                new_df_dict[var].append(0)
            else:
                mask_var = temp_df['Variable'] == var
                temp2_df = temp_df[mask_var]
                val = temp2_df["Value"].values[0]
                new_df_dict[var].append(val)

    for key in new_df_dict.keys():
        new_df[key] = new_df_dict[key]
    
    for key in keys:
        new_df[key+'_prop'] = new_df[key] / new_df['Total Generation']

    if save:
        new_df.to_csv(load_dir, index=False)

    return new_df

def load_election_data(load_dir="Election/election_by_state_cleaner.csv", year=2020):

    if exists(load_dir):
        df = pd.read_csv(load_dir)
        return df 

    df = pd.read_csv('Election/election_by_state.csv') 

    
    df = df[df['year'] == year]
    demo_df = df[df['party_simplified'] == "DEMOCRAT"]
    rep_df = df[df['party_simplified'] == "REPUBLICAN"]

    new_df = pd.DataFrame()

    new_df['state'] = df['state'].unique()
    new_df['Democrat'] = demo_df["candidatevotes"].values
    new_df['Republican'] = rep_df["candidatevotes"].values
    new_df['Total'] = demo_df["totalvotes"].values
    new_df["Democrat_prop"] = new_df['Democrat']/ new_df['Total']
    new_df["Republican_prop"] = new_df['Republican']/ new_df['Total']

    new_df.to_csv(load_dir, index=False)

    return new_df
    
def load_eia_installations_data(load_dir="Clean_Data/installs_by_state.csv", save=True, print_stats=False):
    '''
    Loads the most recent years EIA data for added Capacity and Generation by State
    This has a known issue with AL data (see EIA/details.md)
    '''
    if exists(load_dir):
        df = pd.read_csv(load_dir)
        return df 

    eia_df = pd.read_csv('EIA/small_scale_solar_by_state_by_month.csv')

    for key in ['Residential_cap', 'Residential_gen','Commercial_cap','Commercial_gen','Industrial_cap','Industrial_gen', 'Total_cap', 'Total_gen']:
        eia_df[key] = conv_strings_to_floats(eia_df[key])

    eia_df.to_csv('small_scale_solar_by_state_by_month.csv', index=False)

    eia_df = eia_df[eia_df['State'] != 'DC']

    jan_df = eia_df[eia_df['Month'] == 1]
    jan_2024_df = jan_df[jan_df['Year'] == 2024]
    jan_2025_df = jan_df[jan_df['Year'] == 2025]


    comp_df = pd.DataFrame(jan_2024_df['State'])
    for key in ['Residential_cap', 'Residential_gen','Commercial_cap','Commercial_gen','Industrial_cap','Industrial_gen', 'Total_cap', 'Total_gen']:
        comp_df[key + "_24"] = jan_2024_df[key].values
        comp_df[key + "_25"] = jan_2025_df[key].values
        

    totals = {'24_res-cap': np.sum(jan_2024_df['Residential_cap']), '24_res-gen': np.sum(jan_2024_df['Residential_gen']), '25_res-cap': np.sum(jan_2025_df['Residential_cap']), '25_res-gen': np.sum(jan_2025_df['Residential_gen'])}

    comp_df['Residential_cap_prop_24'] = jan_2024_df['Residential_cap'].values / totals['24_res-cap']
    comp_df['Residential_gen_prop_24'] = jan_2024_df['Residential_gen'].values / totals['24_res-gen']
    comp_df['Residential_cap_prop_25'] = jan_2025_df['Residential_cap'].values / totals['25_res-cap']
    comp_df['Residential_gen_prop_25'] = jan_2025_df['Residential_gen'].values / totals['25_res-gen']

    comp_df['Residential_added_cap'] = comp_df['Residential_cap_25'] - comp_df['Residential_cap_24']
    comp_df['Residential_added_gen'] = comp_df['Residential_gen_25'] - comp_df['Residential_gen_24']

    comp_df['prop_cap_added'] = comp_df['Residential_added_cap'] / np.sum(comp_df['Residential_added_cap'])

    if save:
        comp_df.to_csv(load_dir, index=False)

    if print_stats:
        print("total small-scale solar capacity in Jan 2024:", round(np.sum(comp_df['Residential_cap_24']),1), "(estimated panels:", round(np.sum(comp_df['Residential_cap_24'])*4000,0), ")")
        print("total small-scale solar capacity in Jan 2025:", round(np.sum(comp_df['Residential_cap_25']),1), "(estimated panels:", round(np.sum(comp_df['Residential_cap_25'])*4000,0), ")")
        print("total small-scale solar capacity added in 2024:", np.sum(comp_df['Residential_cap_25']) - np.sum(comp_df['Residential_cap_24']), "(estimated panels:", (np.sum(comp_df['Residential_cap_25']) - np.sum(comp_df['Residential_cap_24']))*4000, ")")

    return comp_df

def stats_by_state(df, key, state):
    '''
    calculates the mean, std, and median of a particular coloumn of df (denoted by "key")
    does this only for rows from the given state
    '''

    df = df.dropna(axis=0)

    df = df[df['state_name'] == state] 
    # df = df[df[key].notna()]
    vals = df[key].values 

    stats = {'state_name' : [state], 'mean' : [np.mean(vals)], 'std': [np.std(vals)], 'median' : [np.median(vals)], 'sum': [np.sum(vals)]}


    return pd.DataFrame(stats)

def stats_for_states(df, key):
    '''
    Calculates the mean, std, and median of the key col of df
    outputs a df witheach row corresponding to a state and cols : mean, std, median
    '''

    print("calculating statistics of states on:", key)

    pr_mask = df['state_name'].isin(['Aguadilla', 'Arecibo', 'Dorado', 'Hormigueros', 'Moca', 'Mayagüez', 'Ponce',
    'Canóvanas', 'Corozal', 'San Juan', 'Toa Baja', 'Toa Alta', 'Bayamón', 'Cataño',
    'Guaynabo', 'Trujillo Alto', 'Carolina','District of Columbia'])

    # Remove Puerto Rico Data as well as Washington DC
    states = df[~pr_mask]['state_name'].unique()
    states = np.sort(states)

    stats = stats_by_state(df, key, states[0])

    for state in states[1:]:
        stats = pd.concat([stats, stats_by_state(df, key, state)])

    stats = stats[stats['mean'] != 0]

    return stats

# Creates a csv of all zip codes which are in each dataset
def get_clean_zips(load_dir="Clean_Data/zips.csv"):
    if exists(load_dir):
        zips = pd.read_csv(load_dir,dtype=str) 
        zips = zips.drop_duplicates(subset=['zip'], keep='first')
        return zips['zip'].values
    else:
        solar_zips = load_solar_data(load_dir=None)['region_name'].values
        census_zips = load_census_data(load_dir=None)['zip'].values
        grid_zips = load_grid_data(load_dir=None)['region_name'].values

        # Gets only the zips in all data
        zips = reduce(np.intersect1d, (solar_zips, census_zips, grid_zips))

        zip_df = pd.DataFrame()
        zip_df['zip'] = zips

        # Saves zips
        zip_df.to_csv(load_dir, index=False)

        return zips

# Loads both the census and solar data across all zips and returns both dfs, it is necessary to have already created the solar_by_zip and census_by_zip data under the Data folder though
def load_data(dir='Clean_Data/'):
    print("Loading Data...")

    zip_codes = get_clean_zips() 

    print("loading Solar Data")
    solar_df = load_solar_data(zip_codes, save=True)
    print("Loading Census Data")
    census_df = load_census_data(zip_codes, save=True)
    print("Loading Grid Data")
    grid_df = load_grid_data(zip_codes, save=True)

    nomi = pgeocode.Nominatim('us')

    print("Creating Lat and Long for each zip")
    edf = pd.DataFrame()
    edf['Latitude'] = (nomi.query_postal_code(zip_codes).latitude)
    edf['Longitude'] = (nomi.query_postal_code(zip_codes).longitude)
    edf['zip_code'] = zip_codes

    return zip_codes, solar_df, census_df, grid_df, edf

def make_state_dataset(df, energy_keys=None, stats_keys=None, load_dir="Clean_Data/data_by_state.csv", agg='mean'):
    
    if exists(load_dir):
        return pd.read_csv(load_dir)
    
    if energy_keys == None:
        energy_keys = ['Clean', 'Bioenergy', 'Coal','Gas','Fossil','Solar','Hydro','Nuclear','Wind','Other Renewables','Other Fossil','Total Generation']
        stats_keys= ["Total_Population","total_households","Median_income","per_capita_income","households_below_poverty_line", "yearly_sunlight_kwh_kw_threshold_avg", "existing_installs_count", "carbon_offset_metric_tons", "carbon_offset_metric_tons_per_panel","carbon_offset_metric_tons_per_capita" , 'existing_installs_count_per_capita',  "existing_installs_count_per_capita", "panel_utilization", 'carbon_offset_kg_per_panel','carbon_offset_kg']
    
    election_df = load_election_data().drop('state', axis=1)
    energy_df = load_state_energy_dat(keys=energy_keys, total=False)
    incentives_df = pd.read_csv("Incentives/incentives_by_state.csv").drop('State', axis=1)
    installs_df = load_eia_installations_data().drop('State', axis=1)
    stats_df = pd.DataFrame()

    for key in stats_keys:
        # Returns just the means over the ZIPs of a state, change mean to std or median to change
        vals = stats_for_states(df=df.drop(['Latitude', 'Longitude'], axis=1), key=key)[agg].values
        stats_df[key] = vals
    
    for demo in ['black', 'white', 'asian', 'hispanic', 'native']:
        stats_df[demo+'_population'] = stats_for_states(df=df.drop(['Latitude', 'Longitude'], axis=1), key=demo+'_population')['sum'].values
        stats_df[demo+'_prop'] = stats_df[demo+'_population'] / np.sum(df['Total_Population'])

    combined_state_df = pd.concat([energy_df, election_df, stats_df, incentives_df], axis=1)




    combined_state_df['estimated_install_count'] = combined_state_df['Solar'] * 1000 / combined_state_df['yearly_sunlight_kwh_kw_threshold_avg']

    # Reorder to state code since installs_df is sorted by code, then resort to state name
    combined_state_df.sort_values('State code')
    combined_state_df = pd.concat([combined_state_df, installs_df], axis=1).sort_values('State')

    if load_dir is not None:
        combined_state_df.to_csv(load_dir, index=False)

    return combined_state_df

# This makes the full combined df for zip granularity, mostly this just calls load_data, but then removes outliers if desired and adds some new terms
def make_zip_dataset(remove_outliers=True, load_dir='Clean_Data/data_by_zip.csv', save=True):

    if exists(load_dir):
        return pd.read_csv(load_dir)

    # Loads all zipcode data
    _, solar_df, census_df, grid_df, pos_df = load_data()

    combined_df = pd.concat([solar_df, census_df, grid_df, pos_df], axis=1)

    if remove_outliers:
        print("Removing Outliers")

        print("zips before removing outliers:", len(combined_df))

        # Remove outliers for carbon offset (4 outliers in this case)
        mask = combined_df['carbon_offset_metric_tons'] < 50 * ( combined_df['Total_Population'])
        combined_df = combined_df[mask]

        # Removing outliers for existing install counts too large (~90)
        mask = combined_df['existing_installs_count'] < 600
        combined_df = combined_df[mask]

        # Removing outlier for no installations
        mask = combined_df['existing_installs_count'] > 0
        combined_df = combined_df[mask]

        # Removing outliers of too high cabon per panel
        mask = (combined_df['carbon_offset_metric_tons'] / (combined_df['number_of_panels_total']) ) > 0.05
        combined_df = combined_df[mask]

        print("zips after removing outliers:", len(combined_df))

    # Metrics used for data analysis
    combined_df['panel_utilization'] = (combined_df['existing_installs_count'] / combined_df['number_of_panels_total'])
    combined_df['existing_installs_count_per_capita'] = (combined_df['existing_installs_count'] / combined_df['Total_Population'])

    combined_df['carbon_offset_metric_tons_per_panel'] = (combined_df['carbon_offset_metric_tons'] / (combined_df['number_of_panels_total']) )
    combined_df['carbon_offset_metric_tons_per_capita'] = combined_df['carbon_offset_metric_tons']/ combined_df['Total_Population']
    combined_df['carbon_offset_kg'] = combined_df['carbon_offset_metric_tons'] * 1000
    combined_df['carbon_offset_kg_per_panel'] = combined_df['carbon_offset_metric_tons_per_panel'] * 1000

    combined_df['asian_prop'] = (combined_df['asian_population'].values / combined_df['Total_Population'].values)
    combined_df['white_prop'] = (combined_df['white_population'].values / combined_df['Total_Population'].values)
    combined_df['black_prop'] = (combined_df['black_population'].values / combined_df['Total_Population'].values)
    combined_df['hispanic_prop'] = (combined_df['hispanic_population'].values / combined_df['Total_Population'].values)

    combined_df['percent_below_poverty_line'] = combined_df['households_below_poverty_line'] / combined_df['total_households']


    if save:
        combined_df.to_csv(load_dir, index=False)

    return combined_df.reset_index(drop=True)

def make_dataset(granularity='zip', remove_outliers=False, save=True, load_dir_prefix=''):

    zip_codes = get_clean_zips(load_dir=load_dir_prefix+"Clean_Data/zips.csv")

    nomi = pgeocode.Nominatim('us')

    # Positional dat for plotting maps
    edf = pd.DataFrame()
    edf['Latitude'] = (nomi.query_postal_code(zip_codes).latitude)
    edf['Longitude'] = (nomi.query_postal_code(zip_codes).longitude)
    edf['zip_code'] = zip_codes

    # Create the Zip-level data
    zip_data = make_zip_dataset(remove_outliers=remove_outliers, save=save, load_dir=load_dir_prefix+'Clean_Data/data_by_zip.csv')

    if granularity == 'zip':
        return  zip_data, edf
    
    state_data = make_state_dataset(zip_data,load_dir=load_dir_prefix+'Clean_Data/data_by_state.csv')
    if granularity == 'state':
        return state_data
    
    if granularity == 'both':
        return zip_data, state_data, edf
    

if __name__ == '__main__':
    zips_df, state_df, pos_df = make_dataset(granularity='both', remove_outliers=False, save=True)
    sum_df = make_state_dataset(zips_df, None, None, "Clean_data/data_by_state_sum.csv", agg="sum")
    # print(state_df['estimated_install_count'] - sum_df['existing_installs_count'])