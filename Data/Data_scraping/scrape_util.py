# Required Libraries
from census import Census
from us import states

import pandas as pd
import csv

# from geopy.geocoders import Nominatim # For address to lat/long
# from uszipcode import SearchEngine
from tqdm import tqdm

from os.path import exists

from urllib.request import urlopen
import urllib
import ast

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







