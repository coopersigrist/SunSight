from cgitb import reset
import pandas as pd
import math
from uszipcode import SearchEngine
import numpy as np
from scipy.stats import truncnorm
import ast
from time import time
'''
This is the list of decision-making variables for our agents to make decisions on

'''

#relevant_dataframes 
sunroof_df = pd.read_csv('../../Data/Sunroof/solar_by_zip.csv')
sunroof_state_df = pd.read_csv('../../Data/Sunroof/solar_by_state.csv')
elec_cost_state_df = pd.read_csv('../../Data/Grid/elec_rate_states_2022.csv')
elec_cost_df = pd.read_csv('../../Data/Grid/elec_rate_zipcodes_2022.csv')
energy_expenditure_df = pd.read_csv('../../Data/EIA/eia_household_elec_consumption.csv')
state_data_df = pd.read_csv('../../Data/Clean_Data/data_by_state_sum.csv')
solar_by_state_df = pd.read_csv('../../Data/Sunroof/solar_by_state.csv')

DISCOUNT_RATE = 0.05

search = SearchEngine()
'''def sample_from_normal_with_mean_rse(mean, rse, n_samples):
    n_samples = int(round(n_samples*1000000, 0))
    se = rse * mean
    # Truncate at 0 to avoid negatives
    lower, upper = se, np.inf
    a, b = (lower - mean) / se, (upper - mean) / se
    samples = truncnorm(a, b, loc=mean, scale=se).rvs(n_samples)
    return samples '''

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
#connecting the zipcode to region/division

def state_abbr_to_state_full_func(state_abbr):
    state_abbr_to_full = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 
        'MO': 'Missouri', 	'MT': 'Montana', 	'NE': 'Nebraska', 	'NV': 'Nevada',
        'NH': 'New Hampshire', 	'NJ': 'New Jersey', 	'NM': 'New Mexico', 	'NY': 'New York',
        'NC': 'North Carolina', 	'ND': 'North Dakota', 	'OH': 'Ohio', 	'OK': 'Oklahoma',
        'OR': 'Oregon', 	'PA': 'Pennsylvania', 	'RI': 'Rhode Island',
        'SC': 	'South Carolina', 	'SD': 	'South Dakota',
        'TN': 	'Tennessee', 	'TX': 	'Texas',
        'UT': 	'Utah', 	'VT': 	'Vermont',
        'VA': 	'Virginia', 	'WA': 	'Washington',
        # Including DC and territories for completeness
        "DC": "District of Columbia", "AS": "American Samoa", "GU": "Guam",
        "MP": "Northern Mariana Islands", "PR": "Puerto Rico", "VI": "U.S. Virgin Islands"
    }
    return state_abbr_to_full.get(state_abbr, None)

def state_abbr_to_region(state_abbr):    
    return state_to_region_division.get(state_abbr, (None, None))

def zip_to_region_division(zipcode, search_engine):
    zipcode_info = search_engine.by_zipcode(zipcode)
    state_abbr = zipcode_info.state_abbr

    if not state_abbr:
        return None, None, None  

    region, division = state_to_region_division.get(state_abbr, (None, None))
    return state_abbr, region, division

def sample_from_normal_with_mean_rse(mean, rse_percent, n_samples):
    sd = (rse_percent/100) * mean
    a, b = (0 - mean)/sd, np.inf
    return truncnorm(a, b, loc=mean, scale=sd).rvs(int(n_samples*1000))

def get_real_discount_rate(electric_growth_rate, federal_funds_rate, inflation_rate):
    #assume an increase electric prices of 10% --> https://www.masslive.com/news/2022/10/power-planning-westfield-holyoke-other-municipal-utilities-prepare-for-long-expensive-and-uncertain-winter-ahead.html#:~:text=Holyoke%20Gas%20%26%20Electric%20is%20looking,James%20Lavelle%2C%20HG%26E's%20general%20manager.
    #assume inflation rate of avg across 2018 to 2023 https://www.usinflationcalculator.com/inflation/current-inflation-rates/#
    #assume federal funds rate of 5%
    return (1+federal_funds_rate +inflation_rate)/(1 + electric_growth_rate) -1

def npv_of_energy_savings(electric_cost, electric_consumption, energy_ratio, real_discount_rate, payback_period):
    npv=0
    for i in range(payback_period):
        price_of_electricity = electric_cost * (math.e ** (-real_discount_rate * i))
        price_of_electricity_consumed = price_of_electricity * electric_consumption
        #print(f'electricity consumed {price_of_electricity_consumed}') 
        #print(f'energy ration {energy_ratio}')
        #print(price_of_electricity_consumed* energy_ratio)                              
        npv += price_of_electricity_consumed
    return npv* energy_ratio

def get_payment_stats(installation_cost, income, energy_burden, proportion_offset):
    '''
    inputs:
    max_payback: maximum cash inflow per year from energy savings
    installation_cost: upfront payment toward solar installation, including any incentives
    income: annual income of the household
    energy_burden: proportion of income spent on energy annually before solar installation
    proportion_offset: proportion of energy consumption offset by solar installation 

    returns:
    down_payment: upfront payment toward solar installation
    loan: amount financed through a loan
    yearly_savings: amount saved from energy bill each year, paid toward loan if needed

    ASSUMPTIONS: 
    Buyer does not pay more than their current electric bill per year.
    therefor down_payment/ yearly payment is capped at energy_burden * income * proportion -- i.e. yearly bill remains constant after solar installation
    '''
    yearly_savings = energy_burden * income * proportion_offset # This is the amount saved from your energy bill each year, paid toward loan if needed
    loan = max(installation_cost - yearly_savings, 0)

    return loan, yearly_savings

def payback_period_of_energy_savings(installation_cost, income, energy_burden, proportion_offset, interest_rate=0):

    '''
    inputs:
    max_payback: maximum cash inflow per year from energy savings
    investment: upfront payment toward solar installation, including any incentives
    income: annual income of the household
    energy_burden: proportion of income spent on energy annually before solar installation
    proportion_offset: proportion of energy consumption offset by solar installation
    
    returns: 
    payback_period: number of years to pay back installation cost
    '''

    loan, yearly_savings = get_payment_stats(installation_cost, income, energy_burden, proportion_offset)

    if loan == 0 or interest_rate == 0:
        payback_period = installation_cost / yearly_savings
    
    else :
        payback_period = -(math.log(1 - (interest_rate * loan / yearly_savings)) / math.log(1 + interest_rate)) + 1 # time to pay off loan and down payment

    return payback_period

def incentive_for_target_payback(payback_target, installation_cost, energy_burden, income, proportion_offset, interest_rate=0):
    '''
    inputs:
    max_payback: maximum cash inflow per year from energy savings
    real_discount_rate: real discount rate (as a decimal)
    actual_payback: expected payback period in years
    payback_target: desired payback period in years
    installation_cost: upfront payment toward solar installation, including any incentives
    income: annual income of the household
    energy_burden: proportion of income spent on energy annually before solar installation
    proportion_offset: proportion of energy consumption offset by solar installation

    returns: required incentive amount to achieve target payback period
    '''

    loan, yearly_savings = get_payment_stats(installation_cost, income, energy_burden, proportion_offset)

    if interest_rate == 0:
        needed_incentive = installation_cost - (yearly_savings * payback_target)
    else:
        max_loan = (1 - math.exp((1-payback_target)*math.log(1+interest_rate))) * yearly_savings/interest_rate
        needed_incentive = (loan - max_loan)

    return needed_incentive

def npv_of_elec_consumption_only(electric_cost, electric_consumption, real_discount_rate, payback_period):
    npv=0
    for i in range(payback_period):
        price_of_electricity = electric_cost * (math.e ** (-real_discount_rate * i))
        price_of_electricity_consumed = price_of_electricity * electric_consumption
        print(f'electricity consumed {price_of_electricity_consumed}')                          
        npv += price_of_electricity_consumed
    return npv

def make_installation_right_sized_cost(agent, proportion_of_install_energy_consumption_covered):
    energy_consumption = agent.energy_expenditure * proportion_of_install_energy_consumption_covered
    energy_consumption_2 = agent.energy_expenditure[0]
    zipcode = agent.zipcode
    state_full_name = state_abbr_to_state_full_func(zipcode)
    offset = solar_by_state_df[solar_by_state_df['region_name'] == state_full_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]
    solar_right_sized = (energy_consumption / offset) *1000
    solar_right_sized_2 = ((energy_consumption_2 / offset) *1000)
    #state, region, division = zip_to_region_division(zipcode, search)
    state_install_costs = float(state_data_df[state_data_df['State code'] == zipcode]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0])
    state_install_costs_adjusted_for_size = (solar_right_sized/7000) * state_install_costs
    state_install_costs_2 = (solar_right_sized_2 /7000)* state_install_costs
    return state_install_costs_adjusted_for_size

def get_agent_energy_ratio(agent):
    energy_consumption = agent.energy_expenditure
    offset = agent.solar_offset_per_home
    return offset/energy_consumption

def get_agent_energy_burden(agent):
    return (agent.energy_expenditure *agent.elec_cost)/agent.income

def get_agent_npv(agent, real_discount_rate, payback_period):
    #print('in the npv func')
    zip = agent.zipcode
    income = agent.income
    energy_consumption = agent.energy_expenditure
    #print(f'energy consumption {energy_consumption}')
    electricity_dollar_per_kwh_start = agent.elec_cost
    installation_cost = agent.installation_cost
    #offset = solar_potential_to_cost_offset(sunroof_df, zip) #* electricity_dollar_per_kwh_start
    offset = agent.solar_offset_per_home
    #proportion_offset = agent.energy_ratio
    proportion_offset = offset/energy_consumption
    #print(f'proportion offset {energy_consumption}')
    npv_val = npv_of_energy_savings(electricity_dollar_per_kwh_start, energy_consumption, proportion_offset, real_discount_rate, payback_period)
    #print(f'npv_val {npv_val}')
    return npv_val

def get_agent_payback_period(agent, average_energy_burden, average_energy_ratio):

    installation_cost = agent.installation_cost
    payback_period = payback_period_of_energy_savings(installation_cost, agent.income, average_energy_burden, average_energy_ratio)

    return payback_period

def get_agent_target_incentive(agent, payback_target, average_energy_burden, average_energy_ratio):

    installation_cost = agent.installation_cost
    needed_incentive = incentive_for_target_payback(payback_target, installation_cost, average_energy_burden, agent.income, average_energy_ratio)

    return needed_incentive

def get_elec_cost(zipcode, is_state=False):
    elec_cost = elec_cost_df[elec_cost_df['zip'] == zipcode]
    if elec_cost.empty:
        if not is_state:
            state_abbr, region, division = zip_to_region_division(zipcode, search)
        else:
            state_abbr = zipcode
        state_name = state_abbr_to_state_full_func(state_abbr)
        elec_cost = elec_cost_state_df[elec_cost_state_df['State'] == state_name]
        elec_cost = elec_cost['Average Price (cents/kWh)'].values/100
        # print(f'state_res_rate is {elec_cost}')
        return elec_cost
    else:
        elec_cost = elec_cost['res_rate'].values
        # print(f'zip_res_rate is {elec_cost}')
        if not isinstance(elec_cost, float):
            return elec_cost[0]
        return elec_cost
    #except (KeyError, IndexError):
     #   return 0.12  # Default electricity rate

def build_energy_distributions(zipcode, is_state=False):

    if not is_state:
        _, region, _ = zip_to_region_division(zipcode, search)
        filtered_df = energy_expenditure_df[energy_expenditure_df['Region'] == region]
    else:
        region, _ = state_abbr_to_region(zipcode)
        filtered_df = energy_expenditure_df[energy_expenditure_df['Region'] == region]

    dists = []
    for bracket in ['Less than $5,000', '$5,000 to $9,999', '$10,000 to $19,999', '$20,000 to $39,999', '$40,000 to $59,999', '$60,000 to $99,999', '$100,000 to $149,999', '$150,000 or more']:
        mean = filtered_df[f'{bracket}_elec'].iloc[0]
        rse = filtered_df[f'{bracket}_RSE'].iloc[0]
        n_samples = filtered_df[f'{bracket}_count_millions'].iloc[0]
        dists.append(sample_from_normal_with_mean_rse(mean, rse, n_samples))

    return dists

def get_energy_expenditure(dist_array, income, elec_cost):
    #print('getting energy expenditure')
    ratio = income/elec_cost
    if income < 5000:
        #print(income)
        #print(dist_arr)
        dist_arr = dist_array[0]
        dist_arr = [x for x in dist_arr if x <= ratio]
        if len(dist_arr) == 0:
            #print('income too low')
            return income/elec_cost
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 10000:
        #dist_arr = dist_array[1]* elec_cost
        dist_arr = dist_array[1]
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        #draw = np.random.choice(dist_arr, size=1)
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 20000:
        dist_arr = dist_array[2]
        #dist_arr = dist_array[2]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 40000:
        dist_arr = dist_array[3]
        #dist_arr = dist_array[3]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 60000:
        dist_arr = dist_array[4]
        #dist_arr = dist_array[4]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 100000:
        dist_arr = dist_array[5]
        #dist_arr = dist_array[5]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 150000:
        dist_arr = dist_array[6]
        #dist_arr = dist_array[6]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    elif income < 200000:
        dist_arr = dist_array[7]
        #dist_arr = dist_array[7]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw
    else:
        dist_arr = dist_array[7]
        #dist_arr = dist_array[7]* elec_cost
        dist_arr = [x for x in dist_arr if x <= ratio]
        #dist_arr = np.array(dist_arr)/elec_cost
        draw = np.random.choice(dist_arr, size=1)
        return draw

def solar_potential_to_cost_offset(zip, solar_pv_sizing_kw, is_state=False):
    offset = sunroof_df[sunroof_df['region_name'] == zip]
    if offset.empty:
        if not is_state:
            state_abbr, _, _ = zip_to_region_division(zip, search)
            
        else:
            state_abbr = zip
        state_full_name = state_abbr_to_state_full_func(state_abbr)
        offset = sunroof_state_df[sunroof_state_df['region_name'] == state_full_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]*solar_pv_sizing_kw
        # print(f'state offset is {offset}')
        return offset
    else:
        offset = offset['yearly_sunlight_kwh_kw_threshold_avg'].values[0]* solar_pv_sizing_kw
        # print(f'zip offset is {offset}')
        return offset

def get_peer_influence(zip):
    try:
        frac_adopted = sunroof_df[sunroof_df['region_name']== zip]['existing_installs_count'].values[0] / sunroof_df[sunroof_df['region_name']== zip]['count_qualified'].values[0] 
        return frac_adopted
    except (KeyError, IndexError, ZeroDivisionError):
        return 0  # Default if not found or division by zero
    
def get_response(agent, weights):
    RoI = get_acceptance(agent, DISCOUNT_RATE, agent.true_option_value)
    peer_influence = get_peer_influence(agent.zipcode)
    response = weights[0] * RoI + weights[1] * peer_influence 
    if response >= weights[0]:
        return 1
    else:
        return 0

def get_energy_expenditure_vectorized(dist_array, incomes, elec_cost):
    
    '''
    Sample from income-based energy expenditure distributions.

    Parameters:
    dist_array: List of 8 arrays, each containing sampled energy expenditures for income brackets:
                [<5000, 5000-9999, 10000-19999, 20000-39999, 40000-59999, 60000-99999, 100000-149999, 150000+]
    incomes: Array-like of household incomes.
    elec_cost: Electricity cost in $/kWh.

    returns:
    expenditures: Array of sampled energy expenditures corresponding to each income.
    '''

    incomes = np.asarray(incomes, dtype=float)
    ratios = incomes / elec_cost  

    # Define bins once
    bins = np.array([5000, 10000, 20000, 40000, 60000, 100000, 150000, 200000])
    idxs = np.searchsorted(bins, incomes, side="right")
    idxs = np.clip(idxs, 0, len(dist_array) - 1)

    expenditures = np.empty_like(incomes)

    # Iterate over each income bracket to sample expenditures
    for i in range(len(dist_array)):
        mask = idxs == i
        if not np.any(mask):
            continue

        arr = np.asarray(dist_array[i])
        for j in np.where(mask)[0]:
            valid = arr[arr <= ratios[j]]
            if valid.size == 0:
                # fallback: spend at most income/elec_cost
                expenditures[j] = incomes[j]/elec_cost
            else:
                draw_ratio = np.random.choice(valid)
                exp = draw_ratio #* elec_cost 
                # Ensure constraint: expenditure <= income
                expenditures[j] = min(exp, incomes[j])

    return expenditures

if __name__ == "__main__":
    print(payback_period_of_energy_savings(15100, 75000, 0.03, 0.5)) # No subsidy
    print(payback_period_of_energy_savings(10600, 75000, 0.03, 0.5)) # With subsidy

    print(incentive_for_target_payback(10, 15100, 0.03, 100000, 0.5)) # No subsidy
    print(incentive_for_target_payback(10, 10600, 0.03, 100000, 0.5)) # With subsidy

    for i in range(15):
        print("needed incentive for", i, "year payback:", incentive_for_target_payback(i, 15100, 0.03, 100000, 0.5))