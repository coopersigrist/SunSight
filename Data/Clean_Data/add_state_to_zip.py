import pandas as pd

zips_df = pd.read_csv('data_by_zip.csv')
state_df = pd.read_csv('data_by_state.csv')


# Creates a df with zip and state code columns
zips = zips_df['zip']
state_names = zips_df['state_name']
#TODO reverse this dictionary
state_abbr_to_full = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
        'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        # Including DC and territories for completeness
        "District of Columbia": "DC", "American Samoa": "AS", "Guam": "GU",
        "Northern Mariana Islands": "MP", "Puerto Rico": "PR", "U.S. Virgin Islands": "VI"
    }

#replace state name for all zipcodes under 1000 with Puerto Rico
state_names = ['Puerto Rico' if int(z) < 1000 else name for z, name in zip(zips, state_names)]

state_code = [state_abbr_to_full[name] for name in state_names]

output_df = pd.DataFrame({'zip': zips, 'state_name': state_names, 'state_code': state_code})
output_df.to_csv('zips_2.csv', index=False)
