from Models.Incentives.acceptance_vars import adjusted_installation_cost, sunroof_df
from Models.Incentives.simulated_decision import *
#from SunSight.Data.Incentives.Data_Cleaning_Scripts.data_cleaning_incentives import *
from Data.Data_scraping.scrape_util import *
from Models.Incentives.simulated_decision import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import os
import json
eia_df = pd.read_csv('Data/Clean_Data/dec_cap_per_year_per_state.csv')
census_df = pd.read_csv('Data/Incentives/census_by_zip_complex.csv')
state_data_df = pd.read_csv('/Users/asitaram/Documents/GitHub/Untitled/SunSight/Data/Clean_Data/data_by_state_sum.csv')
solar_by_zip_df = pd.read_csv('Data/Clean_Data/sunroof_by_zip.csv')
solar_by_state_df = pd.read_csv('Data/Sunroof/solar_by_state.csv')
output_5_df = pd.read_csv('output_9_1.csv')
incentives = pd.read_csv('Data/Incentives/incentives_by_state.csv')

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import log_loss
search_engine = SearchEngine()
import us


def get_state_full_name(abbr):
    state = us.states.lookup(abbr)
    if state:
        return state.name
    else:
        return None
def add_row(df, new_row):
    df = pd.concat([df, new_row], ignore_index=True)
    return df

def option_val_multiplier_per_state(panel_size_watts=6637.375):
    census_by_state = census_df.groupby(['state_abbr'])
    df = pd.DataFrame()
    for state, group in census_by_state:
        if state[0] == 'DC' or state[0] == 'PR':
            print('skipping dc, pr')
            continue
        print(state)
        print(state[0])
        total_households = state_data_df[state_data_df['State code'] == state[0]]['total_households'].values
        state_name = get_state_full_name(state[0])
        #state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0])
        #state_install_costs_adjusted_for_size = (panel_size_watts/7000) * state_install_costs
        #state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])

        state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0]) *1.3
        state_install_costs_adjusted_for_size = (panel_size_watts/7000) * state_install_costs
        #state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])

        #payback_period = int(state_data_df[state_data_df['State code'] == state[0]]['Adjusted Payback Period (Years, under energy generation assumptions)'].values[0])
        payback_period = 25
        state_existing_cost = state_install_costs_adjusted_for_size #- state_existing_incentive
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        estimated_max_panels = (solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values[0] * 1000 / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values[0] * 1000) /panel_size_watts
        unallowed_years = incentives[incentives['State'] ==state[0]]['Year in affect']
        print(unallowed_years)
        if not unallowed_years.empty:
            eia_df_state = eia_df_state[eia_df_state['Year'].astype(str) not in unallowed_years]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')
        #print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)
        

        state_full_name = state_abbr_to_state_full_func(state[0])
        offset = solar_by_state_df[solar_by_state_df['region_name'] == state_full_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]
        elec_cost = av.get_elec_cost(state[0], is_state=True)

        print(f'Estimated panels installed in state per year {estimated_panels_installed_in_state_per_year}')
        print(f'Estimated prop panels installed in state per year {estimated_panels_installed_in_state_per_year/total_households}')
        print(f'Estimated panels installed already {estimated_panels_installed_in_2022}')
        print(f'Estimated Max panels {estimated_max_panels}')
        print(f'Prop of households with panels installed {estimated_panels_installed_in_2022/total_households}')
        print(f'total_households {total_households}')
        model = SolarAdoptionModelState(state[0])
        #setting no additional incentive
        model.set_incentive_offered_function(0)
        model.set_installation_cost(state_install_costs_adjusted_for_size)
        model.set_payback_period(payback_period)
        model.set_solar_offset_per_home(panel_size_watts/1000)
        print('generating models')
        model.generate_agents()
        model.get_agent_average_bracket_energy_burden()
        model.get_agent_average_bracket_energy_ratio()
        print(f'{model.agent_average_bracket_energy_burden}')
        with open(f'agent_average_bracket_energy_burden_{state[0]}.json', 'w') as f:
            converted = {f"{k[0]}-{k[1]}": v for k, v in model.agent_average_bracket_energy_burden.items()}
            json.dump(converted, f)
        with open(f'agent_average_bracket_energy_ratio_{state[0]}.json', 'w') as g:
            converted1 = {f"{k[0]}-{k[1]}": v for k, v in model.agent_average_bracket_energy_ratio.items()}
            json.dump(dict(converted1), g)
        #json.dumps(model.agent_average_bracket_energy_burden, open(f'agent_average_bracket_energy_burden_{state[0]}.json', 'w'))
        model.calculate_all_agent_paybacks()
        npv_list =model.get_all_npvs()
        payback_period_list = model.get_all_paybacks()
        print(payback_period_list)
        energy_ratio_list = model.get_all_energy_ratios()
        income_list = model.get_all_incomes()
        # for each state:
        # V = h*I
        # V/I = energy_consumption* energy_ratio/investment 
        # (prop_energy_savings) * energy_consumption = investment *option_value_multiplier 
        # (prop energy_savings) *energy_consumption / investment  = option_value_multiplier
        # (prop_energy_savings)/option_value_multiplier * energy_consumption = investment
        # energy_consumption/investment = option_value_multiplier/prop_energy_savings
        energy_consumptions = model.get_all_energy_consumptions()
        small_df = pd.DataFrame(dict({'income': income_list,'energy_consumptions':energy_consumptions, 'energy_ratios': energy_ratio_list, 'npv': npv_list, 'payback_periods': payback_period_list}))
        small_df['ov_multiplier'] = small_df['npv']/state_install_costs_adjusted_for_size
        small_df['portion_of_income_spent_on_electricity'] = (small_df['energy_consumptions'] *elec_cost)/small_df['income']
        small_df['npv_energy_consumptions'] = small_df['npv']/small_df['energy_ratios']
        small_df['npv_actual'] = small_df['npv'] - state_existing_cost
        small_df['installation_cost'] = state_install_costs_adjusted_for_size
        small_df['solar_installation_right_sized'] = (small_df['energy_consumptions'] / offset) *1000
        small_df['installation_cost_right_sized'] = small_df['solar_installation_right_sized']/7000 * state_install_costs #- state_existing_incentive
        small_df['npv_actual_right_sized'] =  small_df['npv_energy_consumptions'] - small_df['installation_cost_right_sized']
        small_df['payback_period'] = small_df['installation_cost_right_sized']/small_df['npv_energy_consumptions']
        small_df['ov_multiplier'] = (small_df['npv_energy_consumptions'] * small_df['energy_ratios'])/state_existing_cost
        small_df['multiplier'] = small_df['npv_energy_consumptions']/small_df['installation_cost_right_sized']
        small_df['multiplier_right_sized'] = small_df['npv_energy_consumptions']/small_df['installation_cost_right_sized']
        small_df['ov_multiplier_right_sized'] = (small_df['npv_energy_consumptions'])/small_df['installation_cost_right_sized']
        small_df['ov_multiplier_check'] = small_df['multiplier']/small_df['energy_ratios']
        small_df = small_df.sort_values(['payback_periods'], ascending=[True])

        small_df.to_csv(state[0]+'_small_results_10.csv')
        prop_adopted_in_state = (estimated_panels_installed_in_2022)  /total_households
        prop_adopted_in_state_quantified = int(prop_adopted_in_state * len(small_df))
        prop_annual_adopted_in_state = int(estimated_panels_installed_in_state_per_year/total_households *len(small_df))

        row = small_df.iloc[prop_adopted_in_state_quantified]
        row2 = small_df.iloc[prop_adopted_in_state_quantified + prop_annual_adopted_in_state]
        row3 = small_df.iloc[prop_annual_adopted_in_state]
        #min_coeff = np.percentile(small_df['energy_consumptions'], (1-prop_adopted_in_state)*100 )
        #perc_value = small_df["energy_consumptions"].quantile(1-prop_adopted_in_state)
        #row = small_df.iloc[(small_df["energy_consumptions"] - perc_value).abs().argsort()[0]]

        
        #min_energy_ratio =energy_ratio_list, (1-prop_adopted_in_state)*100
        #min_energy_consumption = energy_ratio_list, (1-prop_adopted_in_state)*100
        #min_ov_multiplier = (state_existing_cost /min_energy_consumption)
        #min_energy_expenditure = 1/energy_ratio_list * solar_by_state_df_for_state
        new_row = pd.DataFrame([{'State code':state[0], 'portion_of_income_spent_on_electricity': row['portion_of_income_spent_on_electricity'][0], 'income': row['income'], 'npv_per_install': row['npv_actual'], 'npv_right_size': row['npv_actual_right_sized'][0],'right_sized_multiplier': row['multiplier_right_sized'][0], 'ov_per_install':row['multiplier']/row['energy_ratios'], 'needed_phi_per_install':row['energy_ratios'], 'energy_consumption_kwh':row['energy_consumptions'][0], 'prop_adopted_status_quo':prop_adopted_in_state, 'prop_adopted_per_year_average': (estimated_panels_installed_in_state_per_year/total_households)[0],  'median_npv': np.median(npv_list), 'median_elec_consumption': np.median(energy_consumptions), 'median_energy_ratio': np.median(energy_ratio_list),  'cutoff_energy_ratio': row['energy_ratios'], 'elec_cost': elec_cost, 'payback_period_status_quo': row['payback_periods'], 'payback_period_first_year_cutoff': row3['payback_periods'], 'payback_period_new_year_added': row2['payback_periods']}])
        data = np.array(payback_period_list).flatten()
        print(data)
        x = np.sort(data)

        #calculate CDF values
        y = 1. * np.arange(len(data)) / (len(data) - 1)

        #plot CDF
        plt.plot(x, y)
        plt.hlines(prop_adopted_in_state, np.min(x), np.max(x))
        plt.vlines(row['payback_periods'], 0, 1)
        #plt.show()
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("output_10.csv", index=False, header=new_row.columns)



    return

def get_more_cutoff_periods_all_states(list_years_adopted, panel_size_watts=6637.375):
    census_by_state = census_df.groupby(['state_abbr'])
    
    df = pd.read_csv('output_9.csv')
    payback_1 = []
    payback_2 = []
    payback_3 = []
    payback_4 = []
    payback_5 = []
    for state, group in census_by_state:
        if state[0] == 'DC' or state[0] == 'PR':
            print('skipping dc, pr')
            continue
        print(state)
        print(state[0])
        #yearly_adoption_rate = df[df['['prop_adopted_per_year_average']
        total_households = state_data_df[state_data_df['State code'] == state[0]]['total_households'].values
        state_name = get_state_full_name(state[0])
        #total_households = state_data_df[state_data_df['State code'] == state[0]]['total_households'].values
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        estimated_max_panels = (solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values[0] * 1000 / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values[0] * 1000) /panel_size_watts
        unallowed_years = incentives[incentives['State'] ==state[0]]['Year in affect']
        print(unallowed_years)
        if not unallowed_years.empty:
            eia_df_state = eia_df_state[eia_df_state['Year'].astype(str) not in unallowed_years]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)
        state_df_path = f'{state[0]}_small_results_9.csv'
        state_df = pd.read_csv(state_df_path)
        small_list = []
        for i in list_years_adopted:
            panels_for_cutoff = int((estimated_panels_installed_in_state_per_year*i)/total_households *len(state_df))
            row = state_df.iloc[panels_for_cutoff]
            small_list.append(row['payback_periods'])
        payback_1.append(small_list[0])
        payback_2.append(small_list[1])
        payback_3.append(small_list[2])
        payback_4.append(small_list[3])
        payback_5.append(small_list[4])
    df['payback_1'] = payback_1
    df['payback_2'] = payback_2
    df['payback_3'] = payback_3
    df['payback_4'] = payback_4
    df['payback_5'] = payback_5
    df.to_csv('output_9_1.csv')

def fix_dfs(dir_dfs, state_abbr):
    '''DEPACATED'''
    payback_periods = output_5_df['payback']
    with open(f'agent_average_bracket_energy_burden_{state_abbr}.json', 'r') as f:
        print('here')
        average_energy_burden = json.load(f)
    print('here')
    with open(f'agent_average_bracket_energy_ratio_{state_abbr}.json', 'r') as g:
        average_energy_ratio = json.load(g)
    average_energy_burden = {tuple(map(int, k.split('-'))): v for k, v in average_energy_burden.items()}
    average_energy_ratio = {tuple(map(int, k.split('-'))): v for k, v in average_energy_ratio.items()}
    for i in os.listdir(dir_dfs):
        df = pd.read_csv(i)
        elec_cost = av.get_elec_cost(i)
        # years_to_make_up * energy_burden *income *proportion_offset

        # energy_burdern * income = energy_consumption * elec_cost / income * income 
        df['max_savings_status_quo'] = df['max_install_cost_status_quo']/ (status_quo_payback - df['payback_period']) * average_energy_ratio 
        df['yearly_pay'] = df['max_install_cost_1']/(needed_payback_1 - df['payback_period']) 

        df['max_savings_2'] = df['max_install_cost_2']/(needed_payback_2 - df['payback_period']) * average_energy_ratio
        df['max_savings_3'] = df['max_install_cost_3']/(needed_payback_3 - df['payback_period']) * average_energy_ratio
        df['max_savings_4'] = df['max_install_cost_4']/(needed_payback_4 - df['payback_period']) * average_energy_ratio
        df['max_savings_5'] = df['max_install_cost_5']/(needed_payback_5 - df['payback_period']) * average_energy_ratio

        df['max_install_cost_status_quo'] = df['max_install_cost_status_quo'] - (status_quo_payback - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio 
        df['max_install_cost_1'] = df['max_install_cost_1'] - (needed_payback_1 - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio
        df['max_install_cost_2'] = df['max_install_cost_2'] - (needed_payback_2 - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio
        df['max_install_cost_3'] = df['max_install_cost_3'] - (needed_payback_3 - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio
        df['max_install_cost_4'] = df['max_install_cost_4'] - (needed_payback_4 - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio
        df['max_install_cost_5'] = df['max_install_cost_5'] - (needed_payback_5 - df['payback_period']) * df['energy_consumption'] * elec_cost * average_energy_ratio
        df['max_install_cost_status_quo'] = df['max_install_cost_status_quo'] - df['installation_cost']
        elec_cost = av.get_elec_cost(state_abbr)
        needed_payback_1 = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_1'].iloc[0]
        needed_payback_2 = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_2'].iloc[0]
        needed_payback_3 = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_3'].iloc[0]
        needed_payback_4 = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_4'].iloc[0]
        needed_payback_5 = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_5'].iloc[0]
        status_quo_payback = output_5_df.loc[output_5_df['State code'] == state_abbr, 'payback_period_status_quo'].iloc[0]
def just_generate_agents(panel_size_watts=6637.375):
    estimated_option_values  = []
    #df_og = pd.read_parquet('data_11.parquet')
    
    census_by_state = census_df.groupby(['state_abbr'])
    for state, group in census_by_state:
        
        
        df_distributions = pd.DataFrame(columns=['state_abbr', 'zip' , 'energy_expenditures', 'incomes', 'ov_right_size_mult_x', 'ov_right_size_mult_y'])
        print(state)
        if state[0] == 'DC' or state[0] == 'PR' or state[0] == 'AL':
            print('skipping dc, pr')
            continue
        directory = str(state[0])
        if not os.path.exists(directory):
            os.makedirs(directory)
        total_households = state_data_df[state_data_df['State code'] == state[0]]['total_households'].values
        state_name = get_state_full_name(state[0])
        #if state[0] not in ['AL', 'AK', 'AZ', 'AR', 'AS', 'CA', 'CO','CT', 'DE', 'FL', 'GA']:
        #if state[0] not in ['HI' , 'ID', 'IL', 'IN' ,'IA', 'KS' ,'KY' ,'LA' ,'ME', 'MD', 'MA', 'MI', 'MN']:
        #if state[0] not in ['MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH']:
        #if state[0] not in ['OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'TT', 'UT', 'VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI']:
            #print(f'skipping state:{state} ')
            #continue
        #print(state[0])
        
        #total_households = state_data_df['total_households']
        #state_name = get_state_full_name(state[0])
        state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0]) *1.3
        state_install_costs_adjusted_for_size = (panel_size_watts/7000) * state_install_costs
        state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])
        #payback_period = int(state_data_df[state_data_df['State code'] == state[0]]['Adjusted Payback Period (Years, under energy generation assumptions)'].values[0])
        payback_period = 25
        state_existing_cost = state_install_costs_adjusted_for_size #- state_existing_incentive
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        estimated_max_panels = (solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values[0] * 1000 / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values[0] * 1000) /panel_size_watts
        unallowed_years = incentives[incentives['State'] ==state[0]]['Year in affect']
        print(unallowed_years)
        if not unallowed_years.empty:
            eia_df_state = eia_df_state[eia_df_state['Year'].astype(str) not in unallowed_years]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')
        #print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)

        ''' state_existing_cost = state_install_costs_adjusted_for_size - state_existing_incentive
        print(f'State existing cost {state_existing_cost}')
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')'''
        print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        #estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)
        #estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        #estimated_max_panels = solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values
        print(f'Estimated panels installed in state per year {estimated_panels_installed_in_state_per_year}')
        print(f'Estimated panels installed already {estimated_panels_installed_in_2022}')
        print(f'Prop of maximum panels installed {estimated_panels_installed_in_2022/estimated_max_panels}')
        #eia_df_zip_average_prop_cap_added = np.average(eia_df_state['prop_cap_added'])
        #solar_by_state_df_state = solar_by_state_df[solar_by_state_df['state_name'] == get_state_full_name(state[0])]
        #solar_by_state_df_state = solar_by_state_df_state['existing_installs_count'].iloc[0]
        #prop_already_installed_in_state =  eia_df_zip_average_prop_cap_added * prop_already_installed_in_zip
        #eia_df_zip_prop_adopted = eia_df_zip['prop_cap_added'].iloc[0]

        with open(f'agent_average_bracket_energy_burden_{state[0]}.json', 'r') as f:
            print('here')
            average_energy_burden = json.load(f)
        print('here')
        with open(f'agent_average_bracket_energy_ratio_{state[0]}.json', 'r') as g:
            average_energy_ratio = json.load(g)
        average_energy_burden = {tuple(map(int, k.split('-'))): v for k, v in average_energy_burden.items()}
        average_energy_ratio = {tuple(map(int, k.split('-'))): v for k, v in average_energy_ratio.items()}
        large_npv_list = []
        state_model_list = []
        
        prop_adopted_in_state = (estimated_panels_installed_in_2022)  /estimated_max_panels
        counter = 0
        print('generating modeling:')
        print(f'Payback period: {payback_period}')
        print(f"Installation costs: {state_install_costs_adjusted_for_size}")
        print(f"Incentives: {state_existing_incentive}")
        print(f"Maximum estiamted panels: {estimated_max_panels}")
        print(f"prop estiamted panels: {prop_adopted_in_state}")
        #ov_mult = float(ast.literal_eval(output_5_df.loc[output_5_df['State code'] == state[0], 'right_sized_multiplier'].iloc[0])[0])
        #ov_mult = output_5_df.loc[output_5_df['State code'] == state[0], 'ov_per_install'].iloc[0] 
        needed_phi = output_5_df.loc[output_5_df['State code'] == state[0], 'cutoff_energy_ratio'].iloc[0]
        needed_payback = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_period_first_year_cutoff'].iloc[0]
        needed_payback_1 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_1'].iloc[0]
        needed_payback_2 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_2'].iloc[0]
        needed_payback_3 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_3'].iloc[0]
        needed_payback_4 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_4'].iloc[0]
        needed_payback_5 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_5'].iloc[0]
        status_quo_payback = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_period_status_quo'].iloc[0]
        state_name = get_state_full_name(state[0])
        for index, row in group.iterrows():
            print('up here')
            zip = row['zip']
            print(zip)
            if os.path.exists(directory+f'/{zip}.csv'):
                print('skipping zip')
                continue

            
            #if zip in df_og['zip'].tolist():
            #    print('already processed zip')
            #    continue
            model = SolarAdoptionModelZipCode(zip)
            #model.set_incentive_offered_function(state_existing_incentive)
            #model.set_installation_cost(state_install_costs_adjusted_for_size)
            #model.set_payback_period(payback_period)
            model.set_incentive_offered_function(0)
            model.set_installation_cost(state_install_costs_adjusted_for_size)
            model.set_payback_period(payback_period)
            model.agent_average_bracket_energy_burden = average_energy_burden
            model.agent_average_bracket_energy_ratio = average_energy_ratio
            model.set_solar_offset_per_home(panel_size_watts/1000)
            
            #npv_list =model.get_all_npvs()
            #payback_period_list = model.get_all_paybacks()
            #print(payback_period_list)
            #energy_ratio_list = model.get_all_energy_ratios()
            #income_list = model.get_all_incomes()
            
            #model.payback_cutoff = needed_payback
            #model.set_solar_offset_per_home(av.solar_potential_to_cost_offset(zip, panel_size_watts/1000))
            print('generating models')
            print(f'needed payback {needed_payback}')
            model.generate_agents()
            model.calculate_all_agent_paybacks()
            payback_list = np.array(model.get_all_paybacks())
            income_list = np.array(model.get_all_incomes())
            energy_consumptions = model.get_all_energy_consumptions()
            phi_list = model.get_all_energy_ratios()
            #model.get_agent_average_bracket_energy_burden()
            #model.calculate_all_agent_paybacks()

            
            #npv =model.get_all_npvs()
            #if len(npv) == [] or len(npv) <= 0:
            #    npv = np.nan
            #else:
            #    npv = npv[0]

            

            model.set_installation_cost(state_install_costs_adjusted_for_size*0.7)
            model.calculate_all_agent_paybacks()
            status_quo_paybacks = np.array(model.get_all_paybacks())
            model.calculate_all_agent_target_incentives(status_quo_payback)
            required_incentives_status_quo=  model.get_all_required_incentives()
            #large_npv_list.append(npv)
            counter +=1
            print(f'NPV calculated for zip {counter} / {len(group)}')
            #print(f'npv is {npv}')

            offset = solar_by_state_df[solar_by_state_df['region_name'] == state_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]
            small_df = pd.DataFrame(dict({'energy_consumptions':energy_consumptions, 'energy_ratios': phi_list, 'income': income_list}))
            #small_df['ov_multiplier'] = small_df['npv']/state_existing_cost
            #small_df['npv_energy_consumptions'] = small_df['npv']/small_df['energy_ratios']
            #small_df['npv_actual'] = small_df['npv'] - state_existing_cost
            small_df['installation_cost'] = state_existing_cost
            #small_df['solar_installation_right_sized'] = (small_df['energy_consumptions'] / offset) *1000
            #small_df['solar_installation_right_sized'] = np.where(small_df['solar_installation_right_sized'] > 50000,50000, small_df['solar_installation_right_sized'] )
            #small_df['installation_cost_right_sized'] = small_df['solar_installation_right_sized']/7000 * state_install_costs # - state_existing_incentive
            #small_df['npv_actual_right_sized'] =  small_df['npv_energy_consumptions'] - small_df['installation_cost_right_sized']
            #small_df['ov_multiplier'] = (small_df['npv_energy_consumptions'] * small_df['energy_ratios'])/state_existing_cost
            #small_df['multiplier'] = small_df['npv_energy_consumptions']/state_existing_cost
            #small_df['multiplier_right_sized'] = small_df['npv_energy_consumptions']/small_df['installation_cost_right_sized']
            #small_df['ov_multiplier_right_sized'] = (small_df['npv_energy_consumptions'])/small_df['installation_cost_right_sized']
            #small_df['payback_periods'] = model.get_all_paybacks() #av.payback_period_of_energy_savings(model.elec_cost, energy_consumptions, energy_ratio, real_discount_rate, payback_period, investment)
            #small_df['max_install_cost'] = (small_df['npv_energy_consumptions'] * needed_phi) / ov_mult
            small_df['payback_period'] = payback_list
            small_df['payback_period_status_quo'] = status_quo_paybacks
            small_df['elec_cost'] = elec_cost
        

            small_df.to_csv(directory+f'/{zip}.csv')
            
            #small_df['max_install_cost_status_quo'] = np.where(small_df['max_install_cost_status_quo'] <= 0,0, small_df['max_install_cost_status_quo'] )
            #small_df['max_install_cost_1'] = np.where(small_df['max_install_cost_1'] <= 0,0, small_df['max_install_cost_1'] )
            '''
            small_df['max_install_cost_2'] = np.where(small_df['max_install_cost_2'] <= 0,0, small_df['max_install_cost_2'] )
            small_df['max_install_cost_3'] = np.where(small_df['max_install_cost_3'] <= 0,0, small_df['max_install_cost_3'] )
            small_df['max_install_cost_4'] = np.where(small_df['max_install_cost_4'] <= 0,0, small_df['max_install_cost_4'] )
            small_df['max_install_cost_5'] = np.where(small_df['max_install_cost_5'] <= 0,0, small_df['max_install_cost_5'] )
            '''
            #print(np.median(small_df['max_install_cost']))

            #sort data
            #data = small_df['max_install_cost_1'].to_numpy().flatten()
            #x = np.sort(data)
            #calculate CDF values
            #y = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_2'].to_numpy().flatten()
            #x_2 = np.sort(data)
            #calculate CDF values
            #y_2 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_3'].to_numpy().flatten()
            #x_3 = np.sort(data)
            #calculate CDF values
            #y_3 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_4'].to_numpy().flatten()
            #x_4 = np.sort(data)
            #calculate CDF values
            #y_4 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_5'].to_numpy().flatten()
            #x_5 = np.sort(data)
            #calculate CDF values
            #y_5 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_status_quo'].to_numpy().flatten()
            #x_sq = np.sort(data)
            #calculate CDF values
            #y_sq = 1. * np.arange(len(data)) / (len(data) - 1)

            #plot CDF
            #plt.plot(x, y)
            #plt.show()

            #print(x)
            '''
            x_log= np.log(x).reshape(-1, 1)
            log_regression_model = LinearRegression()
            log_regression_model.fit(x_log, y)
            params = LinearRegression.get_params(log_regression_model)
            y_pred = log_regression_model.predict(x_log)
            print(params)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            '''

            #print(f"Model: y = {a:.3f} + {b:.3f} * ln(x)")
            #print("MSE:", mse)
            #print("RMSE:", rmse)
            #print("R²:", r2)

            #print(phi_list)
            #print(large_npv_list)
            #print(f'OV  {np.median(np.array(large_npv_list)/state_existing_cost)}')
            #new_df_dist_row =pd.DataFrame([{'state_abbr':state[0], 'zip':zip , 'phi':phi_list,'energy_expenditures':energy_consumptions, 'incomes':incomes , 'incentive_cost_x_1':str(np.array(x).flatten()), 'incentiv_cost_y_1': str(np.array(y).flatten()), 'incentive_cost_x_2':str(np.array(x_2).flatten()), 'incentive_cost_y_2': str(np.array(y_2).flatten()), 'incentive_cost_x_3':str(np.array(x_3).flatten()), 'incentive_cost_y_3': str(np.array(y_3).flatten()), 'incentive_cost_x_4':str(np.array(x_4).flatten()), 'incentive_cost_y_4': str(np.array(y_4).flatten()), 'incentive_cost_x_5':str(np.array(x_5).flatten()), 'incentive_y_5': str(np.array(y_5).flatten()), 'incentive_cost_x_sq':str(np.array(x_sq).flatten()), 'incentive_cost_y_sq': str(np.array(y_sq).flatten())}])
            #new_df_dist_row = pd.DataFrame([{'state_abbr':state[0], 'zip': zip, 'phi': phi_list, 'npv_val':npv}])
            #df_distributions = pd.concat([df_distributions, new_df_dist_row], ignore_index=True)
            #df_distributions.to_parquet(f"data_{state[0]}.parquet", index=False)
            #state_model_list.append(model)
        #large_npv_list = np.array(large_npv_list).flatten()
        #print(f'number of homes {len(large_npv_list)}')
        #print(f'prop adopted in state {prop_adopted_in_state}')
        #max_npv=np.percentile(large_npv_list, (1-prop_adopted_in_state)*100)
        #print(f'max npv {max_npv}')
        #needed_option_value = max_npv/state_existing_cost
        #print(f'needed option value {needed_option_value}')
        #estimated_option_values.append(needed_option_value)
        #new_row = pd.DataFrame([{'State':state[0], 'needed_option_value_multiplier': needed_option_value, 'max_npv': max_npv, 'prop_adopted_status_quo':prop_adopted_in_state, 'median_npv': np.median(npv_list)}])
        #df = pd.concat([df, new_row], ignore_index=True)
        #df.to_csv("output_5.csv", index=False, header=['State', 'needed_option_value_multiplier', 'max_npv', 'prop_adopted_status_quo', 'median_npv'])

def generate_state_estimated_option_values_by_state(panel_size_watts=6637.375):
    estimated_option_values  = []
    #df_og = pd.read_parquet('data_11.parquet')
    
    census_by_state = census_df.groupby(['state_abbr'])
    for state, group in census_by_state:
        
        
        df_distributions = pd.DataFrame(columns=['state_abbr', 'zip' , 'energy_expenditures', 'incomes', 'ov_right_size_mult_x', 'ov_right_size_mult_y'])
        print(state)
        if state[0] == 'DC' or state[0] == 'PR' or state[0] == 'AL':
            print('skipping dc, pr')
            continue
        directory = str(state[0])
        if not os.path.exists(directory):
            os.makedirs(directory)
        total_households = state_data_df[state_data_df['State code'] == state[0]]['total_households'].values
        state_name = get_state_full_name(state[0])
        #if state[0] not in ['AL', 'AK', 'AZ', 'AR', 'AS', 'CA', 'CO','CT', 'DE', 'FL', 'GA']:
        #if state[0] not in ['HI' , 'ID', 'IL', 'IN' ,'IA', 'KS' ,'KY' ,'LA' ,'ME', 'MD', 'MA', 'MI', 'MN']:
        #if state[0] not in ['MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH']:
        #if state[0] not in ['OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'TT', 'UT', 'VT', 'VA', 'VI', 'WA', 'WV', 'WI', 'WY']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI']:
            #print(f'skipping state:{state} ')
            #continue
        #print(state[0])
        
        #total_households = state_data_df['total_households']
        #state_name = get_state_full_name(state[0])
        state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0]) *1.3
        state_install_costs_adjusted_for_size = (panel_size_watts/7000) * state_install_costs
        state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])
        #payback_period = int(state_data_df[state_data_df['State code'] == state[0]]['Adjusted Payback Period (Years, under energy generation assumptions)'].values[0])
        payback_period = 25
        state_existing_cost = state_install_costs_adjusted_for_size #- state_existing_incentive
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        estimated_max_panels = (solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values[0] * 1000 / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values[0] * 1000) /panel_size_watts
        unallowed_years = incentives[incentives['State'] ==state[0]]['Year in affect']
        print(unallowed_years)
        if not unallowed_years.empty:
            eia_df_state = eia_df_state[eia_df_state['Year'].astype(str) not in unallowed_years]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')
        #print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)

        ''' state_existing_cost = state_install_costs_adjusted_for_size - state_existing_incentive
        print(f'State existing cost {state_existing_cost}')
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')'''
        print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        #estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)
        #estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        #estimated_max_panels = solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values
        print(f'Estimated panels installed in state per year {estimated_panels_installed_in_state_per_year}')
        print(f'Estimated panels installed already {estimated_panels_installed_in_2022}')
        print(f'Prop of maximum panels installed {estimated_panels_installed_in_2022/estimated_max_panels}')
        #eia_df_zip_average_prop_cap_added = np.average(eia_df_state['prop_cap_added'])
        #solar_by_state_df_state = solar_by_state_df[solar_by_state_df['state_name'] == get_state_full_name(state[0])]
        #solar_by_state_df_state = solar_by_state_df_state['existing_installs_count'].iloc[0]
        #prop_already_installed_in_state =  eia_df_zip_average_prop_cap_added * prop_already_installed_in_zip
        #eia_df_zip_prop_adopted = eia_df_zip['prop_cap_added'].iloc[0]

        with open(f'agent_average_bracket_energy_burden_{state[0]}.json', 'r') as f:
            print('here')
            average_energy_burden = json.load(f)
        print('here')
        with open(f'agent_average_bracket_energy_ratio_{state[0]}.json', 'r') as g:
            average_energy_ratio = json.load(g)
        average_energy_burden = {tuple(map(int, k.split('-'))): v for k, v in average_energy_burden.items()}
        average_energy_ratio = {tuple(map(int, k.split('-'))): v for k, v in average_energy_ratio.items()}
        large_npv_list = []
        state_model_list = []
        
        prop_adopted_in_state = (estimated_panels_installed_in_2022)  /estimated_max_panels
        counter = 0
        print('generating modeling:')
        print(f'Payback period: {payback_period}')
        print(f"Installation costs: {state_install_costs_adjusted_for_size}")
        print(f"Incentives: {state_existing_incentive}")
        print(f"Maximum estiamted panels: {estimated_max_panels}")
        print(f"prop estiamted panels: {prop_adopted_in_state}")
        #ov_mult = float(ast.literal_eval(output_5_df.loc[output_5_df['State code'] == state[0], 'right_sized_multiplier'].iloc[0])[0])
        #ov_mult = output_5_df.loc[output_5_df['State code'] == state[0], 'ov_per_install'].iloc[0] 
        needed_phi = output_5_df.loc[output_5_df['State code'] == state[0], 'cutoff_energy_ratio'].iloc[0]
        needed_payback = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_period_first_year_cutoff'].iloc[0]
        needed_payback_1 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_1'].iloc[0]
        needed_payback_2 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_2'].iloc[0]
        needed_payback_3 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_3'].iloc[0]
        needed_payback_4 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_4'].iloc[0]
        needed_payback_5 = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_5'].iloc[0]
        status_quo_payback = output_5_df.loc[output_5_df['State code'] == state[0], 'payback_period_status_quo'].iloc[0]
        state_name = get_state_full_name(state[0])
        for index, row in group.iterrows():
            print('up here')
            zip = row['zip']
            print(zip)
            if os.path.exists(directory+f'/{zip}.csv'):
                print('skipping zip')
                continue

            
            #if zip in df_og['zip'].tolist():
            #    print('already processed zip')
            #    continue
            model = SolarAdoptionModelZipCode(zip)
            #model.set_incentive_offered_function(state_existing_incentive)
            #model.set_installation_cost(state_install_costs_adjusted_for_size)
            #model.set_payback_period(payback_period)
            model.set_incentive_offered_function(0)
            model.set_installation_cost(state_install_costs_adjusted_for_size)
            model.set_payback_period(payback_period)
            model.agent_average_bracket_energy_burden = average_energy_burden
            model.agent_average_bracket_energy_ratio = average_energy_ratio
            model.set_solar_offset_per_home(panel_size_watts/1000)
            
            #npv_list =model.get_all_npvs()
            #payback_period_list = model.get_all_paybacks()
            #print(payback_period_list)
            #energy_ratio_list = model.get_all_energy_ratios()
            #income_list = model.get_all_incomes()
            
            #model.payback_cutoff = needed_payback
            #model.set_solar_offset_per_home(av.solar_potential_to_cost_offset(zip, panel_size_watts/1000))
            print('generating models')
            print(f'needed payback {needed_payback}')
            model.generate_agents()
            model.calculate_all_agent_paybacks()
            payback_list = np.array(model.get_all_paybacks())
            #model.get_agent_average_bracket_energy_burden()
            #model.calculate_all_agent_paybacks()

            
            #npv =model.get_all_npvs()
            #if len(npv) == [] or len(npv) <= 0:
            #    npv = np.nan
            #else:
            #    npv = npv[0]
            phi_list = model.get_all_energy_ratios()

            
            required_incentives_1=  model.calculate_all_agent_target_incentives(needed_payback_1)
            print(required_incentives_1)

            
            required_incentives_2= model.calculate_all_agent_target_incentives(needed_payback_2)

            print(required_incentives_1!=required_incentives_2)
            
            required_incentives_3= model.calculate_all_agent_target_incentives(needed_payback_3)

            
            required_incentives_4=  model.calculate_all_agent_target_incentives(needed_payback_4)

            
            required_incentives_5=  model.calculate_all_agent_target_incentives(needed_payback_5)

            model.set_installation_cost(state_install_costs_adjusted_for_size*0.7)
            model.calculate_all_agent_paybacks()
            model.calculate_all_agent_target_incentives(status_quo_payback)
            required_incentives_status_quo=  model.get_all_required_incentives()
            energy_consumptions = model.get_all_energy_consumptions()
            incomes = model.get_all_incomes()
            #large_npv_list.append(npv)
            counter +=1
            print(f'NPV calculated for zip {counter} / {len(group)}')
            #print(f'npv is {npv}')

            offset = solar_by_state_df[solar_by_state_df['region_name'] == state_name]['yearly_sunlight_kwh_kw_threshold_avg'].values[0]
            small_df = pd.DataFrame(dict({'energy_consumptions':energy_consumptions, 'energy_ratios': phi_list}))
            #small_df['ov_multiplier'] = small_df['npv']/state_existing_cost
            #small_df['npv_energy_consumptions'] = small_df['npv']/small_df['energy_ratios']
            #small_df['npv_actual'] = small_df['npv'] - state_existing_cost
            small_df['installation_cost'] = state_existing_cost
            #small_df['solar_installation_right_sized'] = (small_df['energy_consumptions'] / offset) *1000
            #small_df['solar_installation_right_sized'] = np.where(small_df['solar_installation_right_sized'] > 50000,50000, small_df['solar_installation_right_sized'] )
            #small_df['installation_cost_right_sized'] = small_df['solar_installation_right_sized']/7000 * state_install_costs # - state_existing_incentive
            #small_df['npv_actual_right_sized'] =  small_df['npv_energy_consumptions'] - small_df['installation_cost_right_sized']
            #small_df['ov_multiplier'] = (small_df['npv_energy_consumptions'] * small_df['energy_ratios'])/state_existing_cost
            #small_df['multiplier'] = small_df['npv_energy_consumptions']/state_existing_cost
            #small_df['multiplier_right_sized'] = small_df['npv_energy_consumptions']/small_df['installation_cost_right_sized']
            #small_df['ov_multiplier_right_sized'] = (small_df['npv_energy_consumptions'])/small_df['installation_cost_right_sized']
            print(f'needed_phi {needed_phi}')
            #small_df['payback_periods'] = model.get_all_paybacks() #av.payback_period_of_energy_savings(model.elec_cost, energy_consumptions, energy_ratio, real_discount_rate, payback_period, investment)
            #small_df['max_install_cost'] = (small_df['npv_energy_consumptions'] * needed_phi) / ov_mult
            small_df['payback_period'] = payback_list
            small_df['max_install_cost_status_quo'] = required_incentives_status_quo
            small_df['max_install_cost_1'] = required_incentives_1
            small_df['max_install_cost_2'] = required_incentives_2
            small_df['max_install_cost_3'] = required_incentives_3
            small_df['max_install_cost_4'] = required_incentives_4
            small_df['max_install_cost_5'] = required_incentives_5

            small_df.to_csv(directory+f'/{zip}.csv')
            
            #small_df['max_install_cost_status_quo'] = np.where(small_df['max_install_cost_status_quo'] <= 0,0, small_df['max_install_cost_status_quo'] )
            #small_df['max_install_cost_1'] = np.where(small_df['max_install_cost_1'] <= 0,0, small_df['max_install_cost_1'] )
            '''
            small_df['max_install_cost_2'] = np.where(small_df['max_install_cost_2'] <= 0,0, small_df['max_install_cost_2'] )
            small_df['max_install_cost_3'] = np.where(small_df['max_install_cost_3'] <= 0,0, small_df['max_install_cost_3'] )
            small_df['max_install_cost_4'] = np.where(small_df['max_install_cost_4'] <= 0,0, small_df['max_install_cost_4'] )
            small_df['max_install_cost_5'] = np.where(small_df['max_install_cost_5'] <= 0,0, small_df['max_install_cost_5'] )
            '''
            #print(np.median(small_df['max_install_cost']))

            #sort data
            #data = small_df['max_install_cost_1'].to_numpy().flatten()
            #x = np.sort(data)
            #calculate CDF values
            #y = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_2'].to_numpy().flatten()
            #x_2 = np.sort(data)
            #calculate CDF values
            #y_2 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_3'].to_numpy().flatten()
            #x_3 = np.sort(data)
            #calculate CDF values
            #y_3 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_4'].to_numpy().flatten()
            #x_4 = np.sort(data)
            #calculate CDF values
            #y_4 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_5'].to_numpy().flatten()
            #x_5 = np.sort(data)
            #calculate CDF values
            #y_5 = 1. * np.arange(len(data)) / (len(data) - 1)

            #data = small_df['max_install_cost_status_quo'].to_numpy().flatten()
            #x_sq = np.sort(data)
            #calculate CDF values
            #y_sq = 1. * np.arange(len(data)) / (len(data) - 1)

            #plot CDF
            #plt.plot(x, y)
            #plt.show()

            #print(x)
            '''
            x_log= np.log(x).reshape(-1, 1)
            log_regression_model = LinearRegression()
            log_regression_model.fit(x_log, y)
            params = LinearRegression.get_params(log_regression_model)
            y_pred = log_regression_model.predict(x_log)
            print(params)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            '''

            #print(f"Model: y = {a:.3f} + {b:.3f} * ln(x)")
            #print("MSE:", mse)
            #print("RMSE:", rmse)
            #print("R²:", r2)

            #print(phi_list)
            #print(large_npv_list)
            #print(f'OV  {np.median(np.array(large_npv_list)/state_existing_cost)}')
            #new_df_dist_row =pd.DataFrame([{'state_abbr':state[0], 'zip':zip , 'phi':phi_list,'energy_expenditures':energy_consumptions, 'incomes':incomes , 'incentive_cost_x_1':str(np.array(x).flatten()), 'incentiv_cost_y_1': str(np.array(y).flatten()), 'incentive_cost_x_2':str(np.array(x_2).flatten()), 'incentive_cost_y_2': str(np.array(y_2).flatten()), 'incentive_cost_x_3':str(np.array(x_3).flatten()), 'incentive_cost_y_3': str(np.array(y_3).flatten()), 'incentive_cost_x_4':str(np.array(x_4).flatten()), 'incentive_cost_y_4': str(np.array(y_4).flatten()), 'incentive_cost_x_5':str(np.array(x_5).flatten()), 'incentive_y_5': str(np.array(y_5).flatten()), 'incentive_cost_x_sq':str(np.array(x_sq).flatten()), 'incentive_cost_y_sq': str(np.array(y_sq).flatten())}])
            #new_df_dist_row = pd.DataFrame([{'state_abbr':state[0], 'zip': zip, 'phi': phi_list, 'npv_val':npv}])
            #df_distributions = pd.concat([df_distributions, new_df_dist_row], ignore_index=True)
            #df_distributions.to_parquet(f"data_{state[0]}.parquet", index=False)
            #state_model_list.append(model)
        #large_npv_list = np.array(large_npv_list).flatten()
        #print(f'number of homes {len(large_npv_list)}')
        #print(f'prop adopted in state {prop_adopted_in_state}')
        #max_npv=np.percentile(large_npv_list, (1-prop_adopted_in_state)*100)
        #print(f'max npv {max_npv}')
        #needed_option_value = max_npv/state_existing_cost
        #print(f'needed option value {needed_option_value}')
        #estimated_option_values.append(needed_option_value)
        #new_row = pd.DataFrame([{'State':state[0], 'needed_option_value_multiplier': needed_option_value, 'max_npv': max_npv, 'prop_adopted_status_quo':prop_adopted_in_state, 'median_npv': np.median(npv_list)}])
        #df = pd.concat([df, new_row], ignore_index=True)
        #df.to_csv("output_5.csv", index=False, header=['State', 'needed_option_value_multiplier', 'max_npv', 'prop_adopted_status_quo', 'median_npv'])
def generate_state_estimated_option_values(save_dir ='Data/Incentives/needed_option_values_by_state.csv', panel_size_watts=250):
    estimated_option_values  = []

    df = pd.DataFrame(columns=['State', 'needed_option_value_multiplier', 'max_npv', 'prop_adopted_status_quo', 'median_npv'])
    df_distributions = pd.DataFrame(columns=['state_abbr', 'zip', 'npv_vals'])
    census_by_state = census_df.groupby(['state_abbr'])
    for state, group in census_by_state:
        print(state)
        if state[0] in ['AZ','DE','FL','NC','ND','MA','AR','SC','CA']:
        #if state[0] in ['AZ' , 'CA', 'DE', 'FL' ,'NC', 'ND' ,'MA' ,'AR' ,'SC', 'AK', 'AL', 'DC', 'DE', 'FL', 'DE' 'FL' 'NC' 'MA' 'AK' 'AL', 'AZ', 'DE', 'FL', 'NC' ,'ND', 'MA', 'AK', 'AL' ,'AR', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'DC']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ']:
        #if state[0] in ['AK', 'AL', 'AZ', 'AR', 'AS', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI','MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI']:
            print(f'skipping state:{state} ')
            continue
        parquet_files = ['/Users/asitaram/Documents/GitHub/Untitled/SunSight/data.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_2.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_3.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_4.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_5.parquet', '/Users/asitaram/Documents/GitHub/Untitled/SunSight/data_6.parquet']
        parquet_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        print(state[0])
        
        total_households = state_data_df['total_households']
        state_name = get_state_full_name(state[0])
        state_install_costs = float(state_data_df[state_data_df['State code'] == state[0]]['Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)'].values[0])
        state_install_costs_adjusted_for_size = (panel_size_watts/700) * state_install_costs
        state_existing_incentive = float(state_data_df[state_data_df['State code'] == state[0]]['Numeric state-level upfront incentive'].values[0])
        payback_period = int(state_data_df[state_data_df['State code'] == state[0]]['Adjusted Payback Period (Years, under energy generation assumptions)'].values[0])
        state_existing_cost = state_install_costs_adjusted_for_size - state_existing_incentive
        eia_df_state = eia_df[eia_df['State'] == state[0]]
        eia_df_state['Yearly_Diff'] = np.where(eia_df_state['Yearly_Diff'] < 0, 0, eia_df_state['Yearly_Diff'])
        eia_df_state_avg_yearly_cap_added = np.average(eia_df_state['Yearly_Diff'].to_numpy())
        print('yearly cap added')
        print(eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values )
        estimated_panels_installed_in_state_per_year = round((eia_df_state_avg_yearly_cap_added*1000000) / (panel_size_watts), 0)
        estimated_panels_installed_in_2022 = round(((eia_df_state[eia_df_state['Year'] == 2022]['Residential_cap'].values[0]*1000000) / (panel_size_watts)), 0)
        solar_by_state_df_for_state = solar_by_state_df[solar_by_state_df['state_name'] == state_name]
        estimated_max_panels = solar_by_state_df_for_state['yearly_sunlight_kwh_total'].values / solar_by_state_df_for_state['yearly_sunlight_kwh_kw_threshold_avg'].values
        print(f'Estimated panels installed in state per year {estimated_panels_installed_in_state_per_year}')
        print(f'Estimated panels installed already {estimated_panels_installed_in_2022}')
        print(f'Prop of maximum panels installed {estimated_panels_installed_in_2022/estimated_max_panels}')
        #eia_df_zip_average_prop_cap_added = np.average(eia_df_state['prop_cap_added'])
        #solar_by_state_df_state = solar_by_state_df[solar_by_state_df['state_name'] == get_state_full_name(state[0])]

        #solar_by_state_df_state = solar_by_state_df_state['existing_installs_count'].iloc[0]
        #prop_already_installed_in_state =  eia_df_zip_average_prop_cap_added * prop_already_installed_in_zip
        #eia_df_zip_prop_adopted = eia_df_zip['prop_cap_added'].iloc[0]
        large_npv_list = []
        state_model_list = []
        
        prop_adopted_in_state = (estimated_panels_installed_in_state_per_year)  /estimated_max_panels
        counter = 0
        print('generating modeling:')
        print(f'Payback period: {payback_period}')
        print(f"Installation costs: {state_install_costs_adjusted_for_size}")
        print(f"Incentives: {state_existing_incentive}")
        print(f"Maximum estiamted panels: {estimated_max_panels}")
        print(f"prop estiamted panels: {prop_adopted_in_state}")
        for index, row in group.iterrows():
            zip = row['zip']
            #elec_cost = av.get_elec_cost(zip)
            #npv = av.npv_of_energy_savings(elec_cost, electric_consumption, energy_ratio, real_discount_rate, payback_period)
            if zip in parquet_df['zip'].tolist():
                print('already processed zip')
                continue
            print(zip)
            model = SolarAdoptionModelZipCode(zip)
            model.set_incentive_offered_function(state_existing_incentive)
            model.set_installation_cost(state_install_costs_adjusted_for_size)
            model.set_payback_period(payback_period)
            model.set_solar_offset_per_home(av.solar_potential_to_cost_offset(zip))
            print('generating models')
            model.generate_agents()
            npv_list =model.get_all_npvs()
            print(npv_list)
            #npv_list = np.array(npv_list).flatten()
            #large_npv_list = np.concatenate(large_npv_list, npv_list)
            large_npv_list += npv_list
            #large_npv_list = np.array(large_npv_list).flatten()
            #print('NPV LIST')
            #print(npv_list)
            counter +=1
            print(f'NPV calculated for zip {counter} / {len(group)}')
            print(f'average npv {np.mean(npv_list)}')
            #print(large_npv_list)
            #print(f'OV  {np.median(np.array(large_npv_list)/state_existing_cost)}')
            new_df_dist_row = pd.DataFrame([{'state_abbr':state[0], 'zip': zip, 'npv_vals':npv_list}])
            df_distributions = pd.concat([df_distributions, new_df_dist_row], ignore_index=True)
            df_distributions.to_parquet("data_15.parquet", index=False)
            # #print(npv_list)
            state_model_list.append(model)
        #large_npv_list = np.array(large_npv_list).flatten()
        #print(f'number of homes {len(large_npv_list)}')
        #print(f'prop adopted in state {prop_adopted_in_state}')
        #max_npv=np.percentile(large_npv_list, (1-prop_adopted_in_state)*100)
        #print(f'max npv {max_npv}')
        #needed_option_value = max_npv/state_existing_cost
        #print(f'needed option value {needed_option_value}')
        #estimated_option_values.append(needed_option_value)
        #new_row = pd.DataFrame([{'State':state[0], 'needed_option_value_multiplier': needed_option_value, 'max_npv': max_npv, 'prop_adopted_status_quo':prop_adopted_in_state, 'median_npv': np.median(npv_list)}])
        #df = pd.concat([df, new_row], ignore_index=True)
        #df.to_csv("output_5.csv", index=False, header=['State', 'needed_option_value_multiplier', 'max_npv', 'prop_adopted_status_quo', 'median_npv'])
        
    return estimated_option_values
               
def generate_option_value_training_data(parquet_files):
    # V = coeff * npv consumption 
    # V = phi * npv_of_elec_consumption
    # I * ov_multiplier = phi * npv_of_elec_consumption
    # ov_multiplier/phi = npv_of_elec_consumption/I
    if 'npv_vals' in parquet_files.columns:
        return

    return

def train_nn():
    return

def predict_zip_option_value():
    return


'''
def get_zip_status_quos(save_dir='Data/Incentives/needed_option_values.csv'):
    needed_option_values = []
    for i in range(len(census_df)):
        zipcode = census_df.iloc[i]['zip']
        state_abbr, region, division = zip_to_region_division(zipcode, search_engine)
        if state_abbr == 'PR':
            continue
        print(f'loading model for zipcode: {zipcode}, state: {state_abbr}')
        model = SolarAdoptionModelZipCode(zipcode)
        model.generate_agents()
        print('model loaded')
        model.get_all_needed_option_values()
        npv_list =model.get_all_npvs()
        eia_df_zip = eia_df[eia_df['State'] == state_abbr]
        if eia_df_zip.empty:
            needed_option_values.append(-1)
            continue
        print(state_abbr)
        print(eia_df_zip)
        eia_df_zip_prop_adopted = eia_df_zip['prop_cap_added'].iloc[0]
        print(f'eia_df_zip_prop_adopted: {eia_df_zip_prop_adopted}')
        if eia_df_zip_prop_adopted < 0:
            eia_df_zip_prop_adopted = 0
        solar_by_state_df_state = solar_by_state_df[solar_by_state_df['state_name'] == get_state_full_name(state_abbr)]
        print(f'solar_by_state_df_state: {solar_by_state_df_state}')
        solar_by_state_df_state = solar_by_state_df_state['existing_installs_count'].iloc[0]
        if zipcode in solar_by_zip_df['region_name'].values:
            solar_by_zip_df_zip = solar_by_zip_df[solar_by_zip_df['region_name'] == zipcode]['existing_installs_count'].iloc[0]
        else:
            solar_by_zip_df_zip = 0
        print(f'solar_by_zip_df_zip: {solar_by_zip_df_zip}')
        print(f'solar_by_state_df_state: {solar_by_state_df_state}')
        prop_already_installed_in_zip = solar_by_zip_df_zip / solar_by_state_df_state
        #assuming that new adoptions are proportional to the prop of panels already installed in the zipcode
        prop_adopted_in_zip = eia_df_zip_prop_adopted * prop_already_installed_in_zip
        if prop_adopted_in_zip <= 0:
            #assuming that the status quo option value is 1 for states with no adoption info
            needed_option_value = -1
            needed_option_values.append(needed_option_value)
        else:
            min_npv=np.percentile(npv_list, 100*prop_adopted_in_zip)
            print(f"min_npv: {min_npv}")
            needed_option_value = min_npv/INSTALLATION_COST
            needed_option_values.append(needed_option_value)
            print(f"needed_option_value: {needed_option_value}")
        save_df = pd.DataFrame({
            'zipcode': [zipcode],
            'prop_adopted_in_zip': [prop_adopted_in_zip],
            'solar_by_state_df_state': [solar_by_state_df_state],
            'solar_by_zip_df_zip': [solar_by_zip_df_zip],
            'needed_option_value': [needed_option_value]
        })
        save_df.to_csv(save_dir, mode='a', header=not os.path.exists(save_dir), index=False)
    return needed_option_values




'''
if __name__ == "__main__":
    option_val_multiplier_per_state(panel_size_watts=6637.375)
    #get_more_cutoff_periods_all_states([1,2,3,4,5], panel_size_watts=6637.375)
    #generate_state_estimated_option_values_by_state(panel_size_watts=6637.375)
    #needed_option_values =generate_state_estimated_option_values_no_agents()
    #print(needed_option_values)
