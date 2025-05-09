from scrape_util import *

# Define which codes to use and what they are here, find the full instructions for these codes here:
# https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html
code_dict = {'B01003_001E': 'Total_Population',
                    'B11001_001E': 'total_households',
                    'B19013_001E': 'Median_income',
                    'B19301_001E': 'per_capita_income',
                    'B17001_002E': 'households_below_poverty_line', 
                    'B02001_003E': 'black_population',
                    'B02001_002E': 'white_population',  
                    'B02001_005E' : 'asian_population', 
                    'B02001_004E': 'native_population',}

# Downloads and save census data by ZIP code -- downloads features listed in the code dict above
# census_df = get_census_info_by_zip_codes(save_dir="../Census/census_by_zip",code_dict=code_dict)

# Downloads and saves the project sunroof data at zip, city, and state granularity
project_sunroof_scrape()