# Required Libraries
from census import Census
from us import states
import os
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression #, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# from geopy.geocoders import Nominatim # For address to lat/long
from uszipcode import SearchEngine
from tqdm import tqdm

from os.path import exists

from urllib.request import urlopen
import urllib
import ast
import numpy as np

def conv_strings_to_floats(lst):
    new = [float(s.replace(",", "")) if type(s) == str else s for s in lst]
    return np.array(new)
state_to_region_division = {
    # Northeast
    'CT': ('Northeast', 'New England'),
    'ME': ('Northeast', 'New England'),
    'MA': ('Northeast', 'New England'),
    'NH': ('Northeast', 'New England'),
    'RI': ('Northeast', 'New England'),
    'VT': ('Northeast', 'New England'),
    'NJ': ('Northeast', 'Middle Atlantic'),
    'NY': ('Northeast', 'Middle Atlantic'),
    'PA': ('Northeast', 'Middle Atlantic'),

    # Midwest
    'IN': ('Midwest', 'East North Central'),
    'IL': ('Midwest', 'East North Central'),
    'MI': ('Midwest', 'East North Central'),
    'OH': ('Midwest', 'East North Central'),
    'WI': ('Midwest', 'East North Central'),
    'IA': ('Midwest', 'West North Central'),
    'KS': ('Midwest', 'West North Central'),
    'MN': ('Midwest', 'West North Central'),
    'MO': ('Midwest', 'West North Central'),
    'NE': ('Midwest', 'West North Central'),
    'ND': ('Midwest', 'West North Central'),
    'SD': ('Midwest', 'West North Central'),

    # South
    'DE': ('South', 'South Atlantic'),
    'FL': ('South', 'South Atlantic'),
    'GA': ('South', 'South Atlantic'),
    'MD': ('South', 'South Atlantic'),
    'NC': ('South', 'South Atlantic'),
    'SC': ('South', 'South Atlantic'),
    'VA': ('South', 'South Atlantic'),
    'DC': ('South', 'South Atlantic'),
    'WV': ('South', 'South Atlantic'),
    'AL': ('South', 'East South Central'),
    'KY': ('South', 'East South Central'),
    'MS': ('South', 'East South Central'),
    'TN': ('South', 'East South Central'),
    'AR': ('South', 'West South Central'),
    'LA': ('South', 'West South Central'),
    'OK': ('South', 'West South Central'),
    'TX': ('South', 'West South Central'),

    # West
    'AZ': ('West', 'Mountain'),
    'CO': ('West', 'Mountain'),
    'ID': ('West', 'Mountain'),
    'MT': ('West', 'Mountain'),
    'NV': ('West', 'Mountain'),
    'NM': ('West', 'Mountain'),
    'UT': ('West', 'Mountain'),
    'WY': ('West', 'Mountain'),
    'AK': ('West', 'Pacific'),
    'CA': ('West', 'Pacific'),
    'HI': ('West', 'Pacific'),
    'OR': ('West', 'Pacific'),
    'WA': ('West', 'Pacific')
}

state_abbr_to_state_full = state_map = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico"
}

#connecting the zipcode to region/division
def zip_to_region_division(zipcode, search_engine):
    zipcode_info = search_engine.by_zipcode(zipcode)
    state_abbr = zipcode_info.state_abbr

    if not state_abbr:
        return None, None, None  

    region, division = state_to_region_division.get(state_abbr, (None, None))
    return state_abbr, region, division

def state_abbr_to_state_full_func(state_abbr):
    return state_abbr_to_state_full[state_abbr]
def state_abbr_to_region(state_abbr):
    region, division = state_to_region_division.get(state_abbr, (None, None))
    return region, division
# Downloads csv's of the project sunroof data at different granularities and saves them
def project_sunroof_scrape():

    df_state = pd.read_csv('https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-state.csv')
    df_city = pd.read_csv('https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-city.csv')
    df_zip = pd.read_csv('https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-postal_code.csv')

    df_state.to_csv('../Sunroof/solar_by_state.csv', index=False)  
    df_city.to_csv('../Sunroof/solar_by_city.csv', index=False)  
    df_zip.to_csv('../Sunroof/solar_by_zip.csv', index=False)  


def get_census_info_by_zip_codes(code_dict, save_dir="../Census/census_by_zip.csv"):

    # Code dict must not be empty
    if code_dict is None:
        print("bad call to census dataset, no code dictionary")
        return -1

    # Queries the ACS5 dataset by URL -- formats for usage
    code_keys = str(code_dict.keys())
    url = "https://api.census.gov/data/2022/acs/acs5?get="+code_keys +"&for="
    ZCTA = 'zip code tabulation area'
    url = url + urllib.parse.quote(ZCTA) + ":*"
    url = url.replace("dict_keys(", "").replace(")", "").replace("[", "").replace("]", "").replace("'","").replace(" ","")

    # Gets Bytes from the URL, converts to str, removes null elements and then converts to list
    f = urlopen(url)
    myfile = f.read().decode("utf-8").replace("null","-1")
    res = ast.literal_eval(myfile)

    # Converts that list into a DF and then renames the columns
    df = pd.DataFrame(res[1:],columns=res[0])
    df = df.rename(columns=code_dict)
    df = df.rename(columns={'zip code tabulation area':'zcta'})

    #adds state_abbr, region, and division to the df
    search_engine = SearchEngine()
    

    df['state_abbr'] = df['zcta'].apply(lambda x: zip_to_region_division(x, search_engine)[0])
    df['region'] = df['zcta'].apply(lambda x: zip_to_region_division(x, search_engine)[1])
    df['division'] = df['zcta'].apply(lambda x: zip_to_region_division(x, search_engine)[2])

    # Saves the Census data 
    df.to_csv(save_dir, index=False)

    return df
# BY BUILDING (lat/long) SOLAR DATA
def sunroof_by_coord(lat=0,long=0,label='test',API_key=''):
    
    # Example lat and long
    lat = 37.4450
    long = -122.1390

    link = 'https://solar.googleapis.com/v1/buildingInsights:findClosest?location.latitude=' +str(lat)+'&location.longitude='+str(long)+'&requiredQuality=HIGH&key='+str(API_key)
    df = pd.read_json(link).drop('roofSegmentStats').drop('solarPanelConfigs').drop('financialAnalyses').drop('solarPanels').to_csv('Data/test/'+label+".csv")

    print(df)







