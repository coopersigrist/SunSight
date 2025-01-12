import pandas as pd
from os.path import exists
import numpy as np
import json
import pgeocode
import math


# This code is used to combine the Json of solar installation sizes into a total sq footage
def combine_counts(solar_size_json):
    counts = []
    for zip in solar_size_json:
        count = 0
        if type(zip) == str:
            lzip = json.loads(zip)
            for elem in lzip:
                count += elem[0] * elem[1]
        counts.append(count)

    return counts

# Loads Solar data for given zipcodes, also cleans and calculates new values
def load_solar_dat(zip_codes, load_dir="/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/solar_zip_usable.csv"):

    # If we have already cleaned data then we load that instead of processing
    if load_dir is not None and exists(load_dir):
        return pd.read_csv(load_dir)

    zip_codes = list(map(int, zip_codes))
    # df = pd.read_csv('solar_zip_usable.csv')
    df = pd.read_csv('Data/solar_by_zip.csv')
    df = df[df["region_name"].isin(zip_codes)]
    df = df.drop_duplicates(subset=['region_name'], keep='first')
    df = df[['region_name','state_name','yearly_sunlight_kwh_kw_threshold_avg','existing_installs_count','percent_covered','carbon_offset_metric_tons','count_qualified','number_of_panels_total','install_size_kw_buckets_json']]
    solar_size_json = df['install_size_kw_buckets_json']

    # Potential solar panels are saved as a json of different sizes, we use combine counts to get a single square-footage number of potential solar panel area
    counts = combine_counts(solar_size_json.values)
    df['square_footage'] = counts

    # We have to scale by "percent covered" as that is the percent of the zipcode area that has data, but census dat attempts to cover 100% of population
    df['number_of_panels_total'] *= (100/ df['percent_covered']) 
    df['square_footage'] *= (100/ df['percent_covered']) 
    df['carbon_offset_metric_tons'] *= (100/ df['percent_covered']) 
    df['existing_installs_count'] *= (100/ df['percent_covered']) 

    # This metric for solar potential is somewhat arbitrary, it is simply the avg amount of solar energy produced if all possibe solar panels were built
    df['solar_potential'] = df['square_footage'] * df['yearly_sunlight_kwh_kw_threshold_avg']
    
    df.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/solar_zip_usable.csv", index=False)

    return df

def stats_by_state(df, key, state):
    '''
    calculates the mean, std, and median of a particular coloumn of df (denoted by "key")
    does this only for rows from the given state
    '''

    df = df.dropna(axis=0)
    df = df[df['state_name'] == state] 
    # df = df[df[key].notna()]
    vals = df[key].values 
    if key in ['solar_utilization', 'carbon_offset_metric_tons','existing_install_count']:
        vals /= df['Total_Population']

    stats = {'state_name' : [state], 'mean' : [np.mean(vals)], 'std': [np.std(vals)], 'median' : [np.median(vals)]}

    return pd.DataFrame(stats)

def stats_for_states(df, key):
    '''
    Calculates the mean, std, and median of the key col of df
    outputs a df witheach row corresponding to a state and cols : mean, std, median
    '''

    print("calculating statistics of states on:", key)


    pr_mask = df['state_name'].isin(['Aguadilla', 'Arecibo', 'Dorado', 'Hormigueros', 'Moca', 'Mayagüez', 'Ponce',
    'Canóvanas', 'Corozal', 'San Juan', 'Toa Baja', 'Toa Alta', 'Bayamón', 'Cataño',
    'Guaynabo', 'Trujillo Alto', 'Carolina'])

    # GROSS DUMB code to make rows align for the combination data by state df
    states = df[~pr_mask]['state_name'].unique()
    states[states == 'District of Columbia'] = 'Washington, D.C.'
    states = np.sort(states)
    states[states == 'Washington, D.C.'] = 'District of Columbia'

    stats = stats_by_state(df, key, states[0])

    for state in states[1:]:
        stats = pd.concat([stats, stats_by_state(df, key, state)])

    stats = stats[stats['mean'] != 0]

    return stats

# Loads Cenus data for given zipcodes, also cleans and calculates new values
def load_census_dat(zip_codes, load_dir="/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/census_zip_usable.csv"):

    # If we have already cleaned data then we load that instead of processing
    if load_dir is not None and exists(load_dir):
        return pd.read_csv(load_dir)

    zip_codes = list(map(int, zip_codes))
    df = pd.read_csv('Data/census_by_zip.csv')
    df = df[df["zcta"].isin(zip_codes)]

    # Removes bad data, should be already removed from the zip.csv, but this to be certain.
    mask = df['Median_income'] <= 0
    df = df[~mask]

    # Also removes duplicates (which happen for some reason even when there are no duplicates in zips)
    df = df.drop_duplicates(subset=['zcta'], keep='first')
    df = df.sort_values('zcta')
    df.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/census_zip_usable.csv", index=False)

    return df

def load_state_energy_dat(keys= ['Clean', 'Bioenergy', 'Coal','Gas','Fossil','Solar','Hydro','Nuclear','Total Generation'], load=True, total=True):

    if exists("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/state_energy_usable.csv") and load:
        df = pd.read_csv('/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/state_energy_usable.csv') 
        return df
    
    df = pd.read_csv('Data/energy_stats_by_state.csv') 
    solar_data = df[['State', 'State code', 'Variable', 'Value', 'Category']]

    # Mask out Puerto Rico (not enough other data)
    mask = solar_data['State'].isin(["Puerto Rico"])
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

    new_df.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/state_energy_usable.csv", index=False)

    return new_df

def load_election_data(load=True):

    if load:
        df = pd.read_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/election_by_state.csv")
        return df 

    df = pd.read_csv('Data/election_by_state.csv') 

    df = df[df['year'] == 2020]
    demo_df = df[df['party_simplified'] == "DEMOCRAT"]
    rep_df = df[df['party_simplified'] == "REPUBLICAN"]

    new_df = pd.DataFrame()

    new_df['state'] = df['state'].unique()
    new_df['Democrat'] = demo_df["candidatevotes"].values
    new_df['Republican'] = rep_df["candidatevotes"].values
    new_df['Total'] = demo_df["totalvotes"].values
    new_df["Democrat_prop"] = new_df['Democrat']/ new_df['Total']
    new_df["Republican_prop"] = new_df['Republican']/ new_df['Total']

    new_df.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/election_by_state.csv", index=False)

    return new_df

def load_state_data(df, energy_keys=['Clean', 'Bioenergy', 'Coal','Gas','Fossil','Solar','Hydro','Nuclear','Wind','Other Renewables','Other Fossil','Total Generation'], stats_keys=["Total_Population","total_households","Median_income","per_capita_income","households_below_poverty_line","black_population","white_population","asian_population","native_population", "black_prop","white_prop", "asian_prop","yearly_sunlight_kwh_kw_threshold_avg", "existing_installs_count", "carbon_offset_metric_tons", "carbon_offset_metric_tons_per_panel","carbon_offset_metric_tons_per_capita" , 'existing_installs_count_per_capita',  "existing_installs_count_per_capita", "panel_utilization", "realized_potential_percent", 'carbon_offset_kg_per_panel','carbon_offset_kg'], load="/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/data_by_state.csv", save="/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/data_by_state.csv"):
    
    if load is not None and exists(load):
        return pd.read_csv(load)
    
    election_df = load_election_data().drop('state', axis=1)
    energy_df = load_state_energy_dat(keys=energy_keys, load=False, total=False)
    stats_df = pd.DataFrame()

    for key in stats_keys:
        vals = stats_for_states(df=df, key=key)['mean'].values
        stats_df[key] = vals

    combined_state_df = pd.concat([energy_df, election_df, stats_df], axis=1)
    if save is not None:
        combined_state_df.to_csv(save, index=False)

    combined_state_df = combined_state_df[combined_state_df['State'] != "Washington, D.C."]

    return combined_state_df

def get_clean_zips():
    if exists("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/zips_usable.csv"):
        zips = pd.read_csv('/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/zips_usable.csv',dtype=str) 
        zips = zips.drop_duplicates(subset=['zcta'], keep='first')
        return zips['zcta'].values
    else:
        zips = pd.read_csv('Data/zips.csv',dtype=str) 
        zips = zips.drop_duplicates(subset=['zcta'], keep='first')
        zip_codes = zips['zcta'].values
        solar_df = load_solar_dat(zip_codes)
        census_df = load_census_dat(zip_codes, None)

        # Remove all zips not in solar data
        z_temp = zips[zips['zcta'].isin( solar_df['region_name'].astype(str).str.zfill(5))]
        # Remove all zips not in census data (including median income outliers)
        z_temp2 = z_temp[z_temp['zcta'].isin(census_df['zcta'].astype(str).str.zfill(5))]

        # Save this new zip list
        z_temp2.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/zips_usable.csv", index=False)

        return z_temp2['zcta'].values
# Loads both the census and solar data across all zips and returns both dfs, it is necessary to have already created the solar_by_zip and census_by_zip data under the Data folder though

def load_data():
    print("Loading Data")
    # Loads Zip Codes from Data Folder
    # zips = pd.read_csv('../Data/zips.csv',dtype=str) 
    # zips = zips.drop_duplicates(subset=['zcta'], keep='first')
    # zip_codes = zips['zcta'].values

    zip_codes = get_clean_zips() 

    print("number of zip codes:", len(zip_codes))
    solar_df = load_solar_dat(zip_codes)
    print("number of zip codes with solar data:", len(solar_df))
    census_df = load_census_dat(zip_codes)
    print("number of zip codes with census data:", len(census_df))

    solar_df['solar_potential_per_capita'] = solar_df['solar_potential'] / census_df['Total_Population']
    solar_df = solar_df.sort_values('region_name')
    solar_df.to_csv("/Users/mimilertsaroj/Desktop/SunSight/Visualization/Clean_Data/solar_zip_usable.csv", index=False)

    nomi = pgeocode.Nominatim('us')

    edf = pd.DataFrame()
    edf['Latitude'] = (nomi.query_postal_code(zip_codes).latitude)
    edf['Longitude'] = (nomi.query_postal_code(zip_codes).longitude)
    edf['zip_code'] = zip_codes

    return zip_codes, solar_df, census_df, edf

# This makes the full combined df for zip granularity, mostly this just calls load_data, but then removes outliers if desired and adds some new terms
def make_dataset(remove_outliers=True):

    zip_codes, solar_df, census_df, pos_df = load_data()

    combined_df = pd.concat([solar_df, census_df, pos_df], axis=1)

    if remove_outliers:
        print("Removing Outliers")

        # Remove outliers for carbon offset (4 outliers in this case)
        mask = combined_df['carbon_offset_metric_tons'] < 50 * ( combined_df['Total_Population'])
        combined_df = combined_df[mask]

        # Removing outliers for existing install counts (~90)
        mask = combined_df['existing_installs_count'] < 600
        combined_df = combined_df[mask]

        mask = combined_df['existing_installs_count'] > 0
        combined_df = combined_df[mask]

        # mask = combined_df['count_qualified'] > 0
        # combined_df = combined_df[mask]

        # mask = combined_df['state_name'] != 'California'
        # combined_df = combined_df[mask]

        print("zips after removing outliers:", len(combined_df))

    # Current working metric of "solar utilization", should be ~ current carbon offset
    combined_df['solar_utilization'] = (combined_df['existing_installs_count'] / combined_df['number_of_panels_total']) * combined_df['carbon_offset_metric_tons']
    combined_df['panel_utilization'] = (combined_df['existing_installs_count'] / combined_df['number_of_panels_total'])
    combined_df['realized_potential_percent'] = (combined_df['existing_installs_count'] / combined_df['number_of_panels_total']) * 100
    combined_df['energy_generation_per_panel'] = combined_df['yearly_sunlight_kwh_kw_threshold_avg'] * 0.4
    combined_df['existing_installs_count_per_capita'] = (combined_df['existing_installs_count'] / combined_df['Total_Population'])

    avg_panel_util = np.mean(combined_df['panel_utilization'])
    combined_df['panel_util_relative'] = (combined_df['panel_utilization']/ avg_panel_util) - 1

    combined_df['carbon_offset_metric_tons_per_panel'] = (combined_df['carbon_offset_metric_tons'] / (combined_df['number_of_panels_total'] - combined_df['existing_installs_count'] ) )
    combined_df['carbon_offset_metric_tons_per_capita'] = combined_df['carbon_offset_metric_tons']/ combined_df['Total_Population']
    combined_df['carbon_offset_kg_per_panel'] = combined_df['carbon_offset_metric_tons_per_panel'] * 1000

    mask = combined_df['carbon_offset_metric_tons_per_panel'] > 50/1000
    combined_df = combined_df[mask]

    asian_prop = (combined_df['asian_population'].values / combined_df['Total_Population'].values)
    white_prop = (combined_df['white_population'].values / combined_df['Total_Population'].values)
    black_prop = (combined_df['black_population'].values / combined_df['Total_Population'].values)

    combined_df['asian_prop'] = asian_prop 
    combined_df['white_prop'] = white_prop 
    combined_df['black_prop'] = black_prop

    combined_df['percent_below_poverty_line'] = combined_df['households_below_poverty_line'] / combined_df['total_households']

    # combined_df['zips'] = zip_codes

    return combined_df.reset_index(drop=True)

def normalise_df(combined_df,objectives):
    # print(combined_df[metric])
    norm_df = combined_df.copy()
    for ob in objectives:
        max = norm_df[ob].max()
        min = norm_df[ob].min()
        norm_df[ob] = norm_df[ob].apply(lambda x: (x-min)/(max-min))
    return norm_df

