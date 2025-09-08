from Data.Data_scraping.scrape_util import *

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
                    'B02001_004E': 'native_population',
                    'B03003_003E' : 'hispanic_population'}

code_dict_new = {
    'B19001_002E':'Income: < $10,000',
    'B19001_003E':'Income: $10,000-$14,999',
    'B19001_004E':'Income: $15,000-$19,999',
    'B19001_005E': 'Income: $20,000-$24,999',
    'B19001_006E': 'Income: $25,000-$29,999',
    'B19001_007E': 'Income: $30,000-$34,999',
    'B19001_008E': 'Income: $35,000-$39,999',
    'B19001_009E': 'Income: $40,000-$44,999',
    'B19001_010E': 'Income: $45,000-$49,999',
    'B19001_011E': 'Income: $50,000-$59,999',
    'B19001_012E': 'Income: $60,000-$74,999',
    'B19001_013E': 'Income: $75,000-$99,999',
    'B19001_014E':'Income: $100,000-$124,999',
    'B19001_015E': 'Income: $125,000-$149,999',
    'B19001_016E':'Income: $150,000-$199,999',
    'B19001_017E':'Income: $200,000+',


    'B11001_001E': 'total_households',
    'B19013_001E': 'Median_income',
    'B19301_001E': 'per_capita_income',
    'B17001_002E': 'households_below_poverty_line', 
    'B02001_003E': 'black_population',
    'B02001_002E': 'white_population',  
    'B02001_005E' : 'asian_population', 
    'B02001_004E': 'native_population',
    'B03003_003E' : 'hispanic_population'
}

# Downloads and save census data by ZIP code -- downloads features listed in the code dict above
#census_df = get_census_info_by_zip_codes(save_dir="../Census/census_by_zip_w_hisp.csv",code_dict=code_dict)
census_df = get_census_info_by_zip_codes(save_dir="census_by_zip.csv",code_dict=code_dict_new)
# Downloads and saves the project sunroof data at zip, city, and state granularity
# project_sunroof_scrape()