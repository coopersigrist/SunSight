from cgitb import reset
import pandas as pd
import math
from uszipcode import SearchEngine
from Data.Data_scraping.scrape_util import zip_to_region_division, state_abbr_to_state_full_func, state_abbr_to_region
import numpy as np
from scipy.stats import truncnorm
import ast
'''
This is the list of decision-making variables for our agents to make decisions on

'''

#relevant_dataframes 
sunroof_df = pd.read_csv('Data/Sunroof/solar_by_zip.csv')
sunroof_state_df = pd.read_csv('Data/Sunroof/solar_by_state.csv')
elec_cost_state_df = pd.read_csv('Data/Grid/elec_rate_states_2022.csv')
elec_cost_df = pd.read_csv('Data/Grid/elec_rate_zipcodes_2022.csv')
energy_expenditure_df = pd.read_csv('Data/EIA/eia_household_elec_consumption.csv')
state_data_df = pd.read_csv('/Users/asitaram/Documents/GitHub/Untitled/SunSight/Data/Clean_Data/data_by_state_sum.csv')
solar_by_state_df = pd.read_csv('Data/Sunroof/solar_by_state.csv')

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

def sample_from_normal_with_mean_rse(mean, rse_percent, n_samples):
    sd = (rse_percent/100) * mean
    a, b = (0 - mean)/sd, np.inf
    return truncnorm(a, b, loc=mean, scale=sd).rvs(int(n_samples*1000000))
def get_real_discount_rate(electric_growth_rate, federal_funds_rate, inflation_rate):
    #assume an increase electric prices of 10% --> https://www.masslive.com/news/2022/10/power-planning-westfield-holyoke-other-municipal-utilities-prepare-for-long-expensive-and-uncertain-winter-ahead.html#:~:text=Holyoke%20Gas%20%26%20Electric%20is%20looking,James%20Lavelle%2C%20HG%26E's%20general%20manager.
    #assume inflation rate of avg across 2018 to 2023 https://www.usinflationcalculator.com/inflation/current-inflation-rates/#
    #assume federal funds rate of 5%
    return (1+federal_funds_rate +inflation_rate)/(1 + electric_growth_rate) -1


#not including installation cost
#parameters that will vary are -
# electricity consumption
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

def payback_period_of_energy_savings(max_payback, real_discount_rate, investment, income, energy_burden, proportion_offset):
    #if home can pay upfront without exceeding existing electricity costs, it should
    #simple payback
    if investment/income <= energy_burden:
        #print('in here')
        down_payment = investment
        remaining_investment = 0
        yearly_payment_to_investment = 0
    #if the home cannot outright afford the installation, they their current energy bill as a downpayment
    else :
        down_payment = energy_burden * income * proportion_offset
        remaining_investment = investment - down_payment
        yearly_payment_to_investment = energy_burden * income * proportion_offset
        #homes will pay at most the energy burden they were paying previously per year to finance the solar panels
        #remaining_investment_per_year = remaining_investment/down_payment
        #print(f'downpayment {down_payment}')
        #print(f'remaining_investment {remaining_investment}')
        #print(f'yearly payment to investment {yearly_payment_to_investment}')
        #print(f'discount rate {real_discount_rate}')
        #print(f'remaining_investment per year {remaining_investment_per_year}')
    cash_inflow = max_payback
    cumulative_gains = -remaining_investment
    year = 0
    '''
    cash_outflow = min(yearly_payment_to_investment, remaining_investment)
    remaining_investment -= cash_outflow
    discounted_cash_inflow =(cash_inflow  - cash_outflow)
    return (cumulative_gains *-1)/discounted_cash_inflow
    '''
    while year >=0:
        cash_outflow = min(yearly_payment_to_investment, remaining_investment)
        remaining_investment -= cash_outflow
        #remaining_investment= remaining_investment * (1.05)/ ((1 + real_discount_rate) ** year)

        discounted_cash_inflow =(cash_inflow  - cash_outflow)  #/ ((1 + real_discount_rate) ** year) 
        #print(f' discounted_cash_inflow {discounted_cash_inflow}')
        cumulative_gains += discounted_cash_inflow 
        year += 1
        if cumulative_gains >= 0:
            excess = cumulative_gains 
            #print(f'excess {excess}')
            fraction_of_year = 1 - (excess / discounted_cash_inflow)
            #print(f'years to payback {year + fraction_of_year}')
            return year + fraction_of_year
    
def incentive_for_target_payback(max_payment, real_discount_rate, actual_payback, payback_target, investment_original, energy_burden, income, proportion_offset):
    years_to_make_up = actual_payback - payback_target
    print(years_to_make_up)
    return years_to_make_up * energy_burden *income *proportion_offset
    ''''C = energy_burden * income * proportion_offset
    if payback_target< 1:
        cumulative =  C * payback_target
        #print(investment_original)
        return investment_original-cumulative
    year_floor = int(payback_target)
    fraction = payback_target - year_floor
    
    cumulative = 0
    for t in range(year_floor):
        cumulative += C #/ ((1 + real_discount_rate) ** year_floor)
    
    # Add fraction of last year
    cumulative += fraction * (C) #/ ((1 + real_discount_rate) ** year_floor))
    return investment_original - cumulative'''
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
    #print(f'energy_consumpotion {energy_consumption}')
    #print(f'solar sizing {solar_right_sized}')
    #print(f'silar sizing 2 {solar_right_sized_2}')
    #print(f'state install cost adjusted for size{state_install_costs_adjusted_for_size}')
    #print(f'state install cost adjusted for size{state_install_costs_2}')
    #print(state_install_costs_2 > state_install_costs_adjusted_for_size)
    return state_install_costs_adjusted_for_size
def adjusted_installation_cost(installation_cost, rebate):
    return installation_cost - rebate

def get_agent_energy_ratio(agent):
    energy_consumption = agent.energy_expenditure
    offset = agent.solar_offset_per_home
    return offset/energy_consumption

'''def get_energy_burden_from_income(income, bracket_dict):
        for (low, high), avg_burden in bracket_dict.items():
            if low <= income <= high:
                return avg_burden
        return None   '''
def get_agent_energy_burden(agent):
    return (agent.energy_expenditure[0] *agent.elec_cost)/agent.income
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

def get_agent_payback_period(agent, real_discount_rate, average_energy_burden, average_energy_ratio):
    energy_consumption = agent.energy_expenditure
    electricity_dollar_per_kwh_start = agent.elec_cost
    

    #offset = solar_potential_to_cost_offset(sunroof_df, zip) #* electricity_dollar_per_kwh_start
    offset = agent.solar_offset_per_home
    energy_burden = (energy_consumption*electricity_dollar_per_kwh_start)/agent.income
    #installation_cost = make_installation_right_sized_cost(agent, 1)
    installation_cost = agent.installation_cost
    #proportion_offset = agent.energy_ratio
    proportion_offset = offset/energy_consumption
    #print(f'agent income {agent.income}')
    #print(f'energy proportion of income {energy_burden}')
    #print(f'install proportion of income {installation_cost/agent.income}')
    #print(f'average energy ratio {average_energy_ratio}')
    max_savings_per_year = (energy_consumption*electricity_dollar_per_kwh_start)* average_energy_ratio #*proportion_offset
    #print(f'install size {energy_consumption*electricity_dollar_per_kwh_start }')
    payback_period = payback_period_of_energy_savings(max_savings_per_year, real_discount_rate, installation_cost, agent.income,average_energy_burden, average_energy_ratio)
    #print(f'npv_val {npv_val}')
    return payback_period

def get_agent_target_incentive(agent, real_discount_rate, payback_target, average_energy_burden, average_energy_ratio):
    energy_consumption = agent.energy_expenditure
    electricity_dollar_per_kwh_start = agent.elec_cost
    installation_cost = agent.installation_cost
    #print(f'installation cost {installation_cost}')
    #offset = solar_potential_to_cost_offset(sunroof_df, zip) #* electricity_dollar_per_kwh_start
    offset = agent.solar_offset_per_home
    #proportion_offset = agent.energy_ratio
    proportion_offset = offset/energy_consumption

    #ratio_of_savings = ((energy_consumption*electricity_dollar_per_kwh_start) * proportion_offset)/agent.income
    #print(f'proportion offset {energy_consumption}')
    max_savings_per_year = (energy_consumption*electricity_dollar_per_kwh_start)
    #payback_period = incentive_for_target_payback(max_savings_per_year, real_discount_rate, payback_target, installation_cost, agent.income,average_energy_burden, average_energy_ratio)
    payback_period = incentive_for_target_payback(max_savings_per_year, real_discount_rate, agent.payback_period, payback_target, installation_cost, average_energy_burden, agent.income, average_energy_ratio)
    #print(f'npv_val {npv_val}')
    return payback_period

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
        print(f'state_res_rate is {elec_cost}')
        return elec_cost
    else:
        elec_cost = elec_cost['res_rate'].values
        print(f'zip_res_rate is {elec_cost}')
        if not isinstance(elec_cost, float):
            return elec_cost[0]
        return elec_cost
    #except (KeyError, IndexError):
     #   return 0.12  # Default electricity rate

def build_energy_distributions(zipcode, is_state=False):
    if not is_state:
        state_abbr, region, division = zip_to_region_division(zipcode, search)
        filtered_df = energy_expenditure_df[energy_expenditure_df['Region'] == region]
    else:
        region, division = state_abbr_to_region(zipcode)
        filtered_df = energy_expenditure_df[energy_expenditure_df['Region'] == region]

    #build all distributions:
    mean_1 = filtered_df['Less than $5,000_elec'].iloc[0]
    rse_1 = filtered_df['Less than $5,000_RSE'].iloc[0]
    n_samples_1 = filtered_df['Less than $5,000_count_millions'].iloc[0]
    dist_1 = sample_from_normal_with_mean_rse(mean_1, rse_1, n_samples_1)

    mean_2 = filtered_df['$5,000 to $9,999_elec'].iloc[0]
    rse_2 = filtered_df['$5,000 to $9,999_RSE'].iloc[0]
    n_samples_2 = filtered_df['$5,000 to $9,999_count_millions'].iloc[0]
    dist_2 = sample_from_normal_with_mean_rse(mean_2, rse_2, n_samples_2)
    print(f'dist_2 {dist_2}')

    mean_3 = filtered_df['$10,000 to $19,999_elec'].iloc[0]
    rse_3 = filtered_df['$10,000 to $19,999_RSE'].iloc[0]
    n_samples_3 = filtered_df['$10,000 to $19,999_count_millions'].iloc[0]
    dist_3 = sample_from_normal_with_mean_rse(mean_3, rse_3, n_samples_3)

    mean_4 = filtered_df['$20,000 to $39,999_elec'].iloc[0]
    rse_4 = filtered_df['$20,000 to $39,999_RSE'].iloc[0]
    n_samples_4 = filtered_df['$20,000 to $39,999_count_millions'].iloc[0]
    dist_4 = sample_from_normal_with_mean_rse(mean_4, rse_4, n_samples_4)

    mean_5 = filtered_df['$40,000 to $59,999_elec'].iloc[0]
    rse_5 = filtered_df['$40,000 to $59,999_RSE'].iloc[0]
    n_samples_5 = filtered_df['$40,000 to $59,999_count_millions'].iloc[0]
    dist_5 = sample_from_normal_with_mean_rse(mean_5, rse_5, n_samples_5)

    mean_6 = filtered_df['$60,000 to $99,999_elec'].iloc[0]
    rse_6 = filtered_df['$60,000 to $99,999_RSE'].iloc[0]
    n_samples_6 = filtered_df['$60,000 to $99,999_count_millions'].iloc[0]
    dist_6 = sample_from_normal_with_mean_rse(mean_6, rse_6, n_samples_6)

    mean_7 = filtered_df['$100,000 to $149,999_elec'].iloc[0]
    rse_7 = filtered_df['$100,000 to $149,999_RSE'].iloc[0]
    n_samples_7 = filtered_df['$100,000 to $149,999_count_millions'].iloc[0]
    dist_7 = sample_from_normal_with_mean_rse(mean_7, rse_7, n_samples_7)

    mean_8 = filtered_df['$150,000 or more_elec'].iloc[0]
    rse_8 = filtered_df['$150,000 or more_RSE'].iloc[0]
    n_samples_8 = filtered_df['$150,000 or more_count_millions'].iloc[0]
    dist_8 = sample_from_normal_with_mean_rse(mean_8, rse_8, n_samples_8)
    return [dist_1, dist_2, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8]
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
            state_abbr, region, division =zip_to_region_division(zip, search)
            
        else:
            state_abbr = zip
        state_full_name = state_abbr_to_state_full_func(state_abbr)
        offset = sunroof_state_df[sunroof_state_df['region_name'] == state_full_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]*solar_pv_sizing_kw
        print(f'state offset is {offset}')
        return offset
    else:
        offset = offset['yearly_sunlight_kwh_kw_threshold_avg'].values[0]* solar_pv_sizing_kw
        print(f'zip offset is {offset}')
        return offset

#the decision to get solar from an NPV perspective is modeled similarly to Crago et.al --> the NPV of savings must be 1.6X the investment cost 
def get_acceptance(agent, real_discount_rate, option_value):
    zip = agent.zipcode
    income = agent.income
    energy_consumption = agent.energy_expenditure
    electricity_dollar_per_kwh_start = agent.elec_cost
    installation_cost = agent.installation_cost 
    offset = solar_potential_to_cost_offset(sunroof_df, zip) * electricity_dollar_per_kwh_start
    proportion_offset = offset/energy_consumption
    npv_val = npv(electricity_dollar_per_kwh_start, energy_consumption, proportion_offset, installation_cost, real_discount_rate)
    if income < installation_cost:
        return 0
    elif npv_val > option_value * installation_cost:
        return 1
    else:
        return 0

def get_peer_influence(zip):
    try:
        frac_adopted = sunroof_df[sunroof_df['region_name']== zip]['existing_installs_count'].values[0] / sunroof_df[sunroof_df['region_name']== zip]['count_qualified'].values[0] 
        return frac_adopted
    except (KeyError, IndexError, ZeroDivisionError):
        return 0  # Default if not found or division by zero

def get_needed_option_value(agent):
    income = agent.income
    energy_consumption = agent.energy_expenditure
    electricity_dollar_per_kwh_start = agent.elec_cost
    installation_cost = agent.installation_cost 
    offset = solar_potential_to_cost_offset(sunroof_df, agent.zipcode) #* electricity_dollar_per_kwh_start
    proportion_offset = offset/energy_consumption
    npv_val = npv(electricity_dollar_per_kwh_start, energy_consumption, proportion_offset, installation_cost, DISCOUNT_RATE)
    needed_option_value = npv_val/installation_cost
    return needed_option_value, npv_val
    
def get_response(agent, weights):
    RoI = get_acceptance(agent, DISCOUNT_RATE, agent.true_option_value)
    peer_influence = get_peer_influence(agent.zipcode)
    response = weights[0] * RoI + weights[1] * peer_influence 
    if response >= weights[0]:
        return 1
    else:
        return 0


def get_energy_expenditure_vectorized(dist_array, incomes, elec_cost):
    
    '''if not isinstance(elec_cost, float):
        print(f'weird elec cost {elec_cost}')
        if isinstance(elec_cost, list):
            print('weird list')
            elec_cost = elec_cost[0]
        if isinstance(elec_cost, str):
            print('weird string')
            elec_cost = ast.literal_eval(elec_cost)
            if isinstance(elec_cost, list):
                elec_cost = elec_cost[0]'''

    incomes = np.asarray(incomes, dtype=float)
    ratios = incomes / elec_cost  

    # Define bins once
    bins = np.array([5000, 10000, 20000, 40000, 60000, 100000, 150000, 200000])
    idxs = np.searchsorted(bins, incomes, side="right")
    idxs = np.clip(idxs, 0, len(dist_array) - 1)

    expenditures = np.empty_like(incomes)

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
