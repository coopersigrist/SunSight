#from agent import House
from mesa import Model
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import acceptance_vars as av
from tqdm import tqdm
import random
from time import time

# Load census data globally
census_df = pd.read_csv('../../Data/Incentives/census_by_zip_complex.csv')

'''

Agent Decision-Making Rules:
0. Calculated as w_npv * npv_var + w_peer * peer_var
1. If the home cannot incur the initial startup cost, it will not accept the incentive (even if ROI is high enough)
2. Modeling NPV as in Crago.et.al, but with a discount rate of 0.2 (pessimistic economy as in prior work (Sitaraman et.al) - we can change this to a more optimisitc value)
(If NPV * OPTION_VALUE >= INSTALLATION_COST, then the agent will adopt)
3. Modeling just as in "Data-driven agent-based modeling, with application to rooftop solar" (frac adopted in zipcode) (except doing this differently - acutal panels installed / max estimated panels)


This specific model deals with a statitc peer influcence component (that is, just the peer influence that we start out with)
'''
INSTALLATION_COST = 25000 #note: we can change the initial investment cost to something standard for some experiments. otherwise, it is equal to the investment cost given existing incentives
def sample_within_bracket(low, high, count, method='uniform'):
    if method == 'uniform':
        return np.random.uniform(low, high, size=count)
    elif method == 'normal':
        mu = (low + high) / 2
        sigma = (high - low) / 6  
        samples = np.random.normal(mu, sigma, size=count * 2)
        samples = samples[(samples >= low) & (samples <= high)]
        if len(samples) >= count:
            return samples[:count]
        else:
            raise ValueError("Low/High too close ")
    else:
        raise ValueError("Unsupported method")

def sample_from_normal_with_mean_rse(mean, rse, n_samples, count):
    se = rse * mean
    samples = np.random.normal(loc=mean, scale=se, size=n_samples)
    draws = np.random.choice(samples, size=count)
    return draws

class Home(Agent):
    '''
    This model just does a simple accept-reject with two factors:
    1. is the ROI (return on investment). This modeled as in Crago et.al, if ROI *1.6 >= initial installation cost, npv_var =1. Additionally, if the home cannot afford the initial
    start-up cost, the agent cannot accept the adoption
    2. The other is peer-influence (do others in the zipcode/do my neighbors have solar?). This is calculated using the density of solar in the zipcode based on sunroof data. In this model speciifcally, the agent peer influence is a constant 
    and does not change across other agents in the zipcode. 
    3.
    '''
    def __init__(self, zipcode, income, elec_cost, energy_expenditure, has_solar, solar_offset_per_home, installation_cost, model):
        super().__init__(model)
        self.zipcode = zipcode
        self.income = income
        self.elec_cost = elec_cost
        self.energy_expenditure = energy_expenditure
        self.has_solar = has_solar
        self.npv = None
        self.response = None
        self.needed_option_value = None
        self.true_option_value = 1.6
        self.installation_cost = installation_cost
        self.solar_offset_per_home = solar_offset_per_home
        self.energy_ratio = None
        self.payback_period = None
        self.required_incentive = None
        self.discount_rate = None
        self.percent_loan_amount = None
        self.energy_burden = None

    def describe(self):
        """Return a detailed description of the agent"""
        description = {
            'agent_id': self.unique_id,
            'zipcode': self.zipcode,
            'income': f"${self.income:,.2f}",
            'electricity_cost': f"${self.elec_cost:.3f}/kWh",
            'annual_energy_expenditure': f"{self.energy_expenditure:,.0f} kWh",
            'has_solar': self.has_solar,
            'installation_cost': f"${self.installation_cost:,.2f}",
            'true_option_value': self.true_option_value,
            'can_afford_installation': self.income >= self.installation_cost,
            'income_to_installation_ratio': f"{self.income/self.installation_cost:.2f}",
            'npv': self.npv,
            'needed_option_value': self.needed_option_value,
            'response': self.response
        }
        print(description)
       

    def apply_dollar_value_incentive(self, incentive_value):
        self.npv = self.npv - incentive_value
        self.needed_option_value = self.npv/self.installation_cost
        self.response = av.get_response(self, [1,0])
       

    def get_needed_option_value(self):
        self.needed_option_value, self.npv = av.get_needed_option_value(self)
        #print(self.needed_option_value)
  

    def accept_or_reject(self):
        #setting weights for now to completely disregard peer effect
        self.response = av.get_response(self, [1,0])
    
        #print(self.response)
    
    def calculate_energy_ratio(self):
        self.energy_ratio = av.get_agent_energy_ratio(self)

    def change_true_option_value(self, new_option_value):
        self.true_option_value = new_option_value

    def calculate_npv(self, discount_rate, payback_period):
        self.npv = av.get_agent_npv(self, discount_rate, payback_period)

class SolarAdoptionModelZipCode(Model):
    def __init__(self, zipcode):
        super().__init__()
        self.zipcode = zipcode
        self.installation_cost = None
        self.incentive_offered_function = 0
        self.payback_period = 0
        self.schedule = RandomActivation(self)
        self.option_value = None
        self.discount_rate = av.DISCOUNT_RATE
        self.solar_offset_per_home = None
        self.npvs = []
        self.agent_average_bracket_energy_burden = None
        self.agent_average_bracket_energy_ratio = None
        
    
    def set_installation_cost(self, installation_cost):
        self.installation_cost = installation_cost
       
    
    def set_payback_period(self, payback_period):
        self.payback_period = payback_period
       

    def set_incentive_offered_function(self, incentive_offered_function):
        self.incentive_offered_function = incentive_offered_function
      
    def set_solar_offset_per_home(self, solar_pv_sizing_kw):
        solar_offset = av.solar_potential_to_cost_offset(self.zipcode, solar_pv_sizing_kw)
        self.solar_offset_per_home  = solar_offset

    def set_option_value(self, option_value):
        self.option_value = option_value

    def generate_agents(self):

        elec_cost = av.get_elec_cost(self.zipcode)
        zipcode_row = census_df[census_df['zcta'] == self.zipcode]
        if zipcode_row.empty:
            print(f"No data found for zipcode {self.zipcode}")
            return

        total_pop = zipcode_row['total_households'].iloc[0]
        brackets = [
            ('Income: < $10,000', (1000, 9999)),
            ('Income: $10,000-$14,999', (10000, 14999)),
            ('Income: $15,000-$19,999', (15000, 19999)),
            ('Income: $20,000-$24,999', (20000, 24999)),
            ('Income: $25,000-$29,999', (25000, 29999)),
            ('Income: $30,000-$34,999', (30000, 34999)),
            ('Income: $35,000-$39,999', (35000, 39999)),
            ('Income: $40,000-$44,999', (40000, 44999)),
            ('Income: $45,000-$49,999', (45000, 49999)),
            ('Income: $50,000-$59,999', (50000, 59999)),
            ('Income: $60,000-$74,999', (60000, 74999)),
            ('Income: $75,000-$99,999', (75000, 99999)),
            ('Income: $100,000-$124,999', (100000, 124999)),
            ('Income: $125,000-$149,999', (125000, 149999)),
            ('Income: $150,000-$199,999', (150000, 199999)),
            ('Income: $200,000+', (200000, 1000000))
        ]
        bracket_count = {}
        for col, rng in brackets:
            bracket_count[rng] = zipcode_row[col].iloc[0] / total_pop

        synthetic_income = []
        dist_array = av.build_energy_distributions(self.zipcode) # TODO Speed up

        for (low, high), count in bracket_count.items():
            if count > 0:
                samples = sample_within_bracket(low, high, round(count * 1000), 'uniform')
                synthetic_income.extend(samples)


        expenditures = av.get_energy_expenditure_vectorized(dist_array, synthetic_income, elec_cost)

        # Create agents with the synthetic income data
        for i in range(len(synthetic_income)):
            income_val = synthetic_income[i]
            energy_exp = expenditures[i]
            agent = Home(
                zipcode=self.zipcode,
                income=income_val,
                elec_cost=elec_cost,
                energy_expenditure=energy_exp,
                has_solar=False,
                solar_offset_per_home=self.solar_offset_per_home,
                installation_cost=self.installation_cost,
                model=self,
            )
            agent.energy_ratio = av.get_agent_energy_ratio(agent)
            agent.energy_burden = av.get_agent_energy_burden(agent)
            self.schedule.add(agent)

        self.get_agent_average_bracket_energy_burden()
        self.get_agent_average_bracket_energy_ratio()

    
    def get_neighbors_in_zip(self, zipcode):
        return [a for a in self.schedule.agents if a.zipcode == zipcode]
    
    def step(self):
        self.schedule.step()

    def apply_incentive(self, incentive_value):
        for agent in self.schedule.agents:
            if self.incentive_offered_function is not None:
                incentive_value = self.incentive_offered_function(agent)
                agent.apply_dollar_value_incentive(incentive_value)
    
    def get_all_needed_option_values(self):
        for agent in self.schedule.agents:
            agent.get_needed_option_value()
            #print(agent.needed_option_value)

    def describe_all_agents(self):
        """Return descriptions of all agents in the model"""
        agent_descriptions = []
        for agent in self.schedule.agents:
            agent_descriptions.append(agent.describe())
        return agent_descriptions

    def percent_agents_adopting(self):
        can_afford = sum(1 for agent in self.schedule.agents if agent.npv/self.installation_cost >= option_value)
        return can_afford/len(self.schedule.agents)
    
    def get_all_npvs(self):
        npvs = [agent.npv for agent in self.schedule.agents]
        return npvs
    def get_all_energy_ratios(self):
        energy_ratios = [agent.energy_ratio for agent in self.schedule.agents]
        return energy_ratios
    
    def get_agent_average_bracket_energy_burden(self):
        brackets = [
            (1000, 9999), (10000, 14999), (15000, 19999), (20000, 24999),
            (25000, 29999), (30000, 34999), (35000, 39999), (40000, 44999),
            (45000, 49999), (50000, 59999), (60000, 74999), (75000, 99999),
            (100000, 124999), (125000, 149999), (150000, 199999), (200000, 10000000)
        ]
        avg_elec_income = {}
        for low, high in brackets:
            bracket_agents = [a for a in self.schedule.agents if low <= a.income < high]
            if bracket_agents:
                avg_elec_income[(low, high)] = np.average([agent.energy_burden for agent in bracket_agents])
            else:
                avg_elec_income[(low, high)] = 0
        self.agent_average_bracket_energy_burden = avg_elec_income
        
    def get_agent_average_bracket_energy_ratio(self):
        brackets = [
            (1000, 9999), (10000, 14999), (15000, 19999), (20000, 24999),
            (25000, 29999), (30000, 34999), (35000, 39999), (40000, 44999),
            (45000, 49999), (50000, 59999), (60000, 74999), (75000, 99999),
            (100000, 124999), (125000, 149999), (150000, 199999), (200000, 10000000)
        ]
        avg_elec_income = {}
        for low, high in brackets:
            bracket_agents = [a for a in self.schedule.agents if low <= a.income < high]
            if bracket_agents:
                avg_elec_income[(low, high)] = np.average([agent.energy_ratio for agent in bracket_agents])
            else:
                avg_elec_income[(low, high)] = 0
        self.agent_average_bracket_energy_ratio = avg_elec_income

    def get_all_energy_consumptions(self):
        energy_expenditures = [agent.energy_expenditure for agent in self.schedule.agents]
        return energy_expenditures
    def get_all_required_incentives(self):
        energy_ratios = [agent.required_incentive for agent in self.schedule.agents]
        return energy_ratios
    def get_all_incomes(self):
        incomes = [agent.income for agent in self.schedule.agents]
        return incomes
    def get_energy_burden_from_income(self, income):
        #print('energy burden time')
        for (low, high), avg_burden in self.agent_average_bracket_energy_burden.items():
            if low <= income <= high:
                return avg_burden
        return None

    def get_energy_ratio_from_income(self, income):
        #print('energy burden time')
        for (low, high), avg_burden in self.agent_average_bracket_energy_ratio.items():
            if low <= income <= high:
                return avg_burden
        return None 

    def get_all_paybacks(self):
        payback_periods = [agent.payback_period for agent in self.schedule.agents]
        return payback_periods

    def calculate_all_agent_paybacks(self):
        for a in range(len(self.schedule.agents)):
            agent = self.schedule.agents[a]
            agent_bracket = self.get_energy_burden_from_income(agent.income)
            agent_ratio = self.get_energy_ratio_from_income(agent.income)
            agent.payback_period = av.get_agent_payback_period(agent, agent_bracket, agent_ratio)

    def calculate_all_agent_target_incentives(self, cutoff):
        list_agents = []
        for a in range(len(self.schedule.agents)):
            agent = self.schedule.agents[a]
            agent_bracket = self.get_energy_burden_from_income(agent.income)
            agent_ratio = self.get_energy_ratio_from_income(agent.income)
            target_incentive = av.get_agent_target_incentive(agent, cutoff, agent_bracket,agent_ratio)
            agent.required_incentive = target_incentive
            list_agents.append(target_incentive)
        return list_agents
    
    def print_agent_summary(self):
        """Print a summary of all agents"""
        print(f"\n=== Agent Summary for Zipcode {self.zipcode} ===")
        print(f"Total agents: {len(self.schedule.agents)}")
        
        if len(self.schedule.agents) > 0:
            incomes = [agent.income for agent in self.schedule.agents]
            energy_expenditures = [agent.energy_expenditure for agent in self.schedule.agents]
            npvs = [agent.npv for agent in self.schedule.agents]
            print(f"Average income: ${np.mean(incomes):,.2f}")
            print(f"Median income: ${np.median(incomes):,.2f}")
            print(f"Min income: ${min(incomes):,.2f}")
            print(f"Max income: ${max(incomes):,.2f}")
            print(f"Average NPV: {np.mean(npvs):,.2f}")
            print(f"Median NPV: {np.median(npvs):,.2f}")
            print(f"Min NPV: {min(npvs):,.2f}")
            print(f"Max NPV: {max(npvs):,.2f}")
            
            print(f"Average energy expenditure: {np.mean(energy_expenditures):,.0f} kWh")
            print(f"Median energy expenditure: {np.median(energy_expenditures):,.0f} kWh")
            
            can_afford = sum(1 for agent in self.schedule.agents if agent.income >= agent.installation_cost)
            print(f"Agents who can afford installation: {can_afford}/{len(self.schedule.agents)} ({can_afford/len(self.schedule.agents)*100:.1f}%)")

            can_afford = sum(1 for agent in self.schedule.agents if agent.npv/self.installation_cost >= 1)
            print(f" Percent of Agengs adoption (assuming a option value of 1): {can_afford}/{len(self.schedule.agents)} ({can_afford/len(self.schedule.agents)*100:.1f}%)")

            can_afford = sum(1 for agent in self.schedule.agents if agent.npv/self.installation_cost >= 1.6)
            print(f"Percent of Agents adopting (assuming a option value of 1.6): {can_afford}/{len(self.schedule.agents)} ({can_afford/len(self.schedule.agents)*100:.1f}%)")
        
        print("=" * 50)

class SolarAdoptionModelState(Model):
    def __init__(self, state):
        super().__init__()
        self.num_agents = None
        self.state = state
        self.installation_cost = None
        self.incentive_offered_function = 0
        self.payback_period = 0
        self.schedule = RandomActivation(self)
        self.option_value = None
        self.discount_rate = av.DISCOUNT_RATE
        self.solar_offset_per_home = None
        self.npvs = []
        self.payback_cutoff = None
        self.agent_average_bracket_energy_burden = None
        self.agent_average_bracket_energy_ratio = None
    
    def set_installation_cost(self, installation_cost):
        self.installation_cost = installation_cost
       
    def set_num_agents(self, num_agents):
        self.num_agents = num_agents

    def set_payback_period(self, payback_period):
        self.payback_period = payback_period
       
    def set_incentive_offered_function(self, incentive_offered_function):
        self.incentive_offered_function = incentive_offered_function
      
    def set_solar_offset_per_home(self, solar_pv_sizing_kw):
        solar_offset = av.solar_potential_to_cost_offset(self.state, solar_pv_sizing_kw, is_state=True)
        self.solar_offset_per_home  = solar_offset

    def set_option_value(self, option_value):
        self.option_value = option_value

    def generate_agents(self):
        #print('generating agents')
        
        #getting electric costs
        elec_cost = av.get_elec_cost(self.state, is_state=True)

        # getting synthetic income values state
        state_rows= census_df[census_df['state_abbr'] == self.state]
            
        total_pop = state_rows['total_households'].sum()
        total_pop_bracket_1 = state_rows['Income: < $10,000'].sum()
        total_pop_bracket_2 = state_rows['Income: $10,000-$14,999'].sum()
        total_pop_bracket_3 = state_rows['Income: $15,000-$19,999'].sum()
        total_pop_bracket_4 = state_rows['Income: $20,000-$24,999'].sum()
        total_pop_bracket_5 = state_rows['Income: $25,000-$29,999'].sum()
        total_pop_bracket_6 = state_rows['Income: $30,000-$34,999'].sum()
        total_pop_bracket_7 = state_rows['Income: $35,000-$39,999'].sum()
        total_pop_bracket_8 = state_rows['Income: $40,000-$44,999'].sum()
        total_pop_bracket_9 = state_rows['Income: $45,000-$49,999'].sum()
        total_pop_bracket_10 = state_rows['Income: $50,000-$59,999'].sum()
        total_pop_bracket_11 = state_rows['Income: $60,000-$74,999'].sum()
        total_pop_bracket_12 = state_rows['Income: $75,000-$99,999'].sum()
        total_pop_bracket_13 = state_rows['Income: $100,000-$124,999'].sum()
        total_pop_bracket_14 = state_rows['Income: $125,000-$149,999'].sum()
        total_pop_bracket_15 = state_rows['Income: $150,000-$199,999'].sum()
        total_pop_bracket_16 = state_rows['Income: $200,000+'].sum()
        print('set population')
        bracket_count = dict({
            (1000,9999): total_pop_bracket_1/total_pop,
            (10000,14999): total_pop_bracket_2/total_pop,
            (15000, 19999): total_pop_bracket_3/total_pop,
            (20000, 24999): total_pop_bracket_4/total_pop,
            (25000, 29999): total_pop_bracket_5/total_pop,
            (30000, 34999): total_pop_bracket_6/total_pop,
            (35000, 39999): total_pop_bracket_7/total_pop,
            (40000, 44999): total_pop_bracket_8/total_pop,
            (45000, 49999): total_pop_bracket_9/total_pop,
            (50000, 59999):total_pop_bracket_10/total_pop,
            (60000, 74999): total_pop_bracket_11/total_pop,
            (75000, 99999): total_pop_bracket_12/total_pop,
            (100000, 124999):total_pop_bracket_13/total_pop,
            (125000, 149999):total_pop_bracket_14/total_pop,
            (150000, 199999):total_pop_bracket_15/total_pop,
            #arbitrary upper limit here - because we do need an upper limit
            (200000,10000000):total_pop_bracket_16/total_pop
        })

        synthetic_income = []

        dist_array = av.build_energy_distributions(self.state, is_state=True)
        #print('got energy distribition')
        for (low, high), count in bracket_count.items():
            #print(count)
            if count*1000 > 0:  # Only sample if there are households in this bracket
                samples = sample_within_bracket(low, high, int(count*1000), 'uniform')
                synthetic_income.extend(samples)
        #print('generated synethic incomes')
        #print(synthetic_income)
        # Create agents with the synthetic income data
        
        for income_val in synthetic_income:
            #print('in loop')
            #print(energy_expenditure)
            energy_exp = av.get_energy_expenditure(dist_array, income_val, elec_cost)
            agent = Home(
                zipcode=self.state,
                income=income_val,
                elec_cost=elec_cost,
                energy_expenditure=energy_exp,
                has_solar=False,
                solar_offset_per_home = self.solar_offset_per_home,
                installation_cost = self.installation_cost,
                model=self,
                
            )
            
            agent.energy_ratio = av.get_agent_energy_ratio(agent)
            agent.npv= av.get_agent_npv(agent, self.discount_rate, self.payback_period)
            agent.energy_burden = av.get_agent_energy_burden(agent)
            
            #print('npv')
            #print(agent.energy_ratio)
            self.schedule.add(agent)
        self.get_agent_average_bracket_energy_burden()
        self.get_agent_average_bracket_energy_ratio()
    
    def get_neighbors_in_zip(self, zipcode):
        return [a for a in self.schedule.agents if a.zipcode == zipcode]
    
    def step(self):
        self.schedule.step()
    
    def get_agent_average_bracket_energy_burden(self):
        print('getting energy burden buckets')
        brackets = [
            (1000, 9999), (10000, 14999), (15000, 19999), (20000, 24999),
            (25000, 29999), (30000, 34999), (35000, 39999), (40000, 44999),
            (45000, 49999), (50000, 59999), (60000, 74999), (75000, 99999),
            (100000, 124999), (125000, 149999), (150000, 199999), (200000, 10000000)
        ]
        avg_elec_income = {}
        for low, high in brackets:
            bracket_agents = [a for a in self.schedule.agents if low <= a.income < high]
            if bracket_agents:
                avg_elec_income[(low, high)] = np.average([agent.energy_burden for agent in bracket_agents])
            else:
                avg_elec_income[(low, high)] = 0
        self.agent_average_bracket_energy_burden = avg_elec_income
        
    def get_agent_average_bracket_energy_ratio(self):
        print('getting energy ratio buckets')
        brackets = [
            (1000, 9999), (10000, 14999), (15000, 19999), (20000, 24999),
            (25000, 29999), (30000, 34999), (35000, 39999), (40000, 44999),
            (45000, 49999), (50000, 59999), (60000, 74999), (75000, 99999),
            (100000, 124999), (125000, 149999), (150000, 199999), (200000, 10000000)
        ]
        avg_elec_income = {}
        for low, high in brackets:
            bracket_agents = [a for a in self.schedule.agents if low <= a.income < high]
            if bracket_agents:
                avg_elec_income[(low, high)] = np.average([agent.energy_ratio for agent in bracket_agents])
            else:
                avg_elec_income[(low, high)] = 0
        self.agent_average_bracket_energy_ratio = avg_elec_income

    def get_energy_burden_from_income(self, income):
        #print('energy burden time')
        for (low, high), avg_burden in self.agent_average_bracket_energy_burden.items():
            if low <= income <= high:
                return avg_burden
        return None

    def get_energy_ratio_from_income(self, income):
        #print('energy burden time')
        for (low, high), avg_burden in self.agent_average_bracket_energy_ratio.items():
            if low <= income <= high:
                return avg_burden
        return None     

    def apply_incentive(self, incentive_value):
        for agent in self.schedule.agents:
            if self.incentive_offered_function is not None:
                incentive_value = self.incentive_offered_function(agent)
                agent.apply_dollar_value_incentive(incentive_value)
    
    def get_all_needed_option_values(self):
        for agent in self.schedule.agents:
            agent.get_needed_option_value()
            #print(agent.needed_option_value)

    def describe_all_agents(self):
        """Return descriptions of all agents in the model"""
        agent_descriptions = []
        for agent in self.schedule.agents:
            agent_descriptions.append(agent.describe())
        return agent_descriptions

    def percent_agents_adopting(self):
        can_afford = sum(1 for agent in self.schedule.agents if agent.npv/self.installation_cost >= option_value)
        return can_afford/len(self.schedule.agents)
    
    def get_all_npvs(self):
        #print(agent)
        npvs = [agent.npv for agent in self.schedule.agents]
        return npvs
    def get_all_energy_ratios(self):
        energy_ratios = [agent.energy_ratio for agent in self.schedule.agents]
        return energy_ratios

    def get_all_paybacks(self):
        energy_ratios = [agent.payback_period for agent in self.schedule.agents]
        return energy_ratios

    def get_all_energy_consumptions(self):
        #print('in here')
        energy_expenditures = [agent.energy_expenditure for agent in self.schedule.agents]
        #print(energy_expenditures)
        return energy_expenditures
    
    def get_all_incomes(self):
        incomes = [agent.income for agent in self.schedule.agents]
        return incomes
    
    def calculate_all_agent_paybacks(self):
        for agent in self.schedule.agents:
            agent_bracket = self.get_energy_burden_from_income(agent.income)
            agent_ratio = self.get_energy_ratio_from_income(agent.income)
            agent.payback_period = av.get_agent_payback_period(agent, agent_bracket, agent_ratio)


if __name__ == "__main__":

    zips = pd.read_csv('../../Data/Clean_Data/zips.csv')['zip']
    zips = [z for z in zips if z in census_df['zcta'].values]

    quit() # Remove when ready to run

    for zip in zips:
        model = SolarAdoptionModelZipCode(zip)
        model.set_installation_cost(25000) # TODO need to set this by state
        model.set_solar_offset_per_home(solar_pv_sizing_kw=6.6)
        model.generate_agents()
        payback_period = model.get_all_paybacks()
        required_incentives = model.get_all_required_incentives() # TODO Get year cutoff by state and by year

        # TODO Save payback period, required incentives and install cost to a CSV
        # Need to calc the year cutoff for each state






