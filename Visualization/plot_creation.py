from plot_util import *
from data_load_util import *

import scipy.stats as sp

scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

Hot_color_scale = ["rgb(255, 0, 0)","rgb(255, 111, 0)","rgb(255, 234, 0)","rgb(151, 255, 0)","rgb(44, 255, 150)","rgb(0, 152, 255)","rgb(0, 25, 255)","rgb(0, 0, 200)","rgb(150,0,90)"]
scl =["rgb(150,0,90)","rgb(0, 0, 200)","rgb(0, 25, 255)","rgb(0, 152, 255)","rgb(44, 255, 150)","rgb(151, 255, 0)","rgb(255, 234, 0)","rgb(255, 111, 0)","rgb(255, 0, 0)"]

# color scales : "hot", "deep", 'rainbow'    ## Add _r to reverse

'''
Full list of color scales:
'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
'ylorrd'
'''

combined_df = make_dataset(remove_outliers=False)

pop_bins_quartile = q_binning(combined_df['Total_Population'].values, 'Total_Population', q=4, legible_label="Population")
white_prop_bins_quartile = q_binning(combined_df['white_prop'].values, 'white_prop', q=4, legible_label="White Proportion")
asain_prop_bins_quartile = q_binning(combined_df['asian_prop'].values, 'asian_prop', q=4, legible_label="Asian Proportion")
black_prop_bins_med = q_binning(combined_df['black_prop'].values, 'black_prop', q=2, legible_label="Black Proportion")
black_prop_bins_quartile = q_binning(combined_df['black_prop'].values, 'black_prop', q=4, legible_label="Black Proportion")
income_bins_quartile = q_binning(combined_df['Median_income'].values, 'Median_income', q=4, legible_label="Median Income")
co_bins_quartile = q_binning(combined_df['carbon_offset_metric_tons'].values, 'carbon_offset_metric_tons', q=4, legible_label="Carbon Offset" )
co_per_bins_quartile = q_binning(combined_df['carbon_offset_metric_tons_per_panel'].values, 'carbon_offset_metric_tons_per_panel',q=4, legible_label="Carbon Offset Per Panel")

# combined_df.to_csv('Clean_Data/data_by_zip.csv')

# # Increase Font size
# font = {'family' : 'DejaVu Sans',
#     'weight' : 'bold',
#     'size'   : 20}

# matplotlib.rc('font', **font)


# mask = combined_df['existing_installs_count_per_capita'] < 0.06
# combined_df = combined_df[mask]

state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")

# Noman requested plot (Sunlight vs Carbon Offset)
# scatter_plot(combined_df['carbon_offset_metric_tons_per_panel'] * 1000, combined_df['yearly_sunlight_kwh_kw_threshold_avg']*0.4, None, xlabel="Carbon Offset Potential (Kg / Panel)", ylabel="Energy Generation Potential (kWh/panel)", alpha=0.8, avgs=True, c=combined_df['panel_utilization'], cmap='inferno', color='', title="")

# Carbon offset map
# geo_plot(combined_df['carbon_offset_kg_per_panel'] ,'rainbow', "Carbon offset (kg) per panel", edf=combined_df)

# Main/ Intro Plot 
# complex_scatter(combined_df=combined_df, x=combined_df['carbon_offset_metric_tons'] *1000, y=combined_df['existing_installs_count'], xlabel="Carbon Offset Potential (Kg)", ylabel="Existing Installs", title="", bins=co_bins_quartile, fit=[2], legend=True, square=True, fontsize=20)


# print("CO/panel var:", np.std(combined_df['carbon_offset_metric_tons_per_panel'] *1000))
# print("Energy Generation Potential (kWh/panel) var:", np.std(combined_df['yearly_sunlight_kwh_kw_threshold_avg']*0.4 ))

# print(sp.pearsonr(combined_df['carbon_offset_metric_tons_per_panel'] *1000, combined_df['yearly_sunlight_kwh_kw_threshold_avg']*0.4 )) 


# bar_plot_demo_split(state_df, demos=["black_prop", "Median_income", "Republican_prop"], key="carbon_offset_metric_tons")
# bar_plot_demo_split(state_df, demos=["black_prop", "Median_income", "Republican_prop"], key="existing_installs_count_per_capita")
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "asian_prop", "Median_income", "Republican_prop"], key="carbon_offset_metric_tons_per_panel", type="percent", stacked=True, xticks=['Black', 'White','Asian', 'Median income', 'Republican'], ylabel="Carbon Offset Per Capita (Percent above average)", title="")
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "asian_prop", "Median_income", "Republican_prop"], key="existing_installs_count_per_capita", type="percent", stacked=True,  xticks=['Black', 'White','Asian', 'Median income', 'Republican'], ylabel="Existing Installs Per Capita (Percent above average)", title="")
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "asian_prop", "Median_income", "Republican_prop"], key="panel_utilization", type="percent", stacked=True,  xticks=['Black', 'White','Asian', 'Median income', 'Republican'], ylabel="Panel Utilization (Percent above average)", title="")
# bar_plot_demo_split(state_df, demos=["black_prop", "Median_income", "Republican_prop"], key="carbon_offset_metric_tons", type="diff")
# bar_plot_demo_split(state_df, demos=["black_prop", "Median_income", "Republican_prop"], key="existing_installs_count_per_capita",type="diff")

hatches=['o','o','o','o','o','x','x','x','x','x']
annotate = False

# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_metric_tons_per_panel", type="percent", stacked=True, ylabel="Carbon Offset Potential (% Avg)", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate)
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income','Republican'] , type="percent", stacked=True, ylabel="Realized Potential (% Avg)", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate, legend=False)
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type="percent", stacked=True, ylabel="Existing Installs Per Capita (% Avg)", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate,  legend=False)

# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_kg_per_panel", type="paper", stacked=False, ylabel="Carbon Offset Potential (x average)", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate)
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="realized_potential_percent", xticks=['Black', 'White', 'Asian','Income','Republican'] , type="paper", stacked=False, ylabel="Realized Potential (x average)", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate, legend=True)
# bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type="paper", stacked=False, ylabel="Existing Installs Per Capita", title="", hatches=['o','o','o','o','o','x','x','x','x','x'], annotate=annotate,  legend=False)

# hatches = None
# annotate = True

# bar_plot_demo_split(combined_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="carbon_offset_kg_per_panel", type="percent", stacked=True, ylabel="Carbon Offset Potential (kg / Panel)", title="", hatches=hatches, annotate=annotate)
# bar_plot_demo_split(combined_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="realized_potential_percent", xticks=['Black', 'White', 'Asian','Income'] , type="percent", stacked=True, ylabel="Realized Potential (%)", title="", hatches=hatches, annotate=annotate, legend=False)
# bar_plot_demo_split(combined_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="existing_installs_count_per_capita", type="percent", stacked=True, ylabel="Existing Installs Per Capita", title="", hatches=hatches, annotate=annotate,  legend=False)

# quit()

# for key in ['carbon_offset_metric_tons_per_panel', 'carbon_offset_metric_tons']:
    # state_stats = stats_for_states(combined_df, key)
    # plot_state_stats(state_df, states=None, key=key, sort_by='std')

# log_solar_pot = np.log((solar_df['solar_potential_per_capita'].values + 0.001))
# log_solar_util = np.log((solar_df['solar_potential_per_capita'].values * solar_df['existing_installs_count']) + 0.001)

# geo_plot(log_solar_pot,'hot', "log solar potential per capita", pos_df)
# geo_plot(census_df['Median_income'], 'mint', "Median income", pos_df)
# geo_plot(log_solar_util,'hot', "log solar utilization", pos_df)
# geo_plot(np.log(asian_prop),'rainbow', "Log Asian Population Proportion", pos_df)
# geo_plot(white_prop,'rainbow', "Log White Population Proportion", pos_df)
# geo_plot(np.log(black_prop),'rainbow', "Log Black Population Proportion", pos_df)
# geo_plot(np.log((combined_df['carbon_offset_metric_tons']/ combined_df['Total_Population'].values) + 0.001) ,'rainbow', "Log Carbon offset per capita", pos_df)
# geo_plot(combined_df['percent_covered'] ,'rainbow', "Carbon offset", pos_df)
# geo_plot(np.log(combined_df['panel_util_relative']) ,'rainbow', "Log Panel Utilization relative to national average", pos_df)


# Xs = [(asian_prop, "Proportional of people which are asian"), (white_prop, "Proportional of people which are white"), (black_prop, "Proportional of people which are black"), (census_df['Median_income'].values, "Median Income")]
# Ys = [(np.log(solar_df['solar_potential_per_capita'].values), "Log Solar Potential Per Capita"), ((solar_df['carbon_offset_metric_tons']/ census_df['Total_Population'].values), "Carbon Offset if all panels built (metric tons) per capita")]

# for x, xname in Xs:
#     for y, yname in Ys:
#         scatter_plot(x, y, xlabel=xname, ylabel=yname, fit=[1,5,10])

# for key in ['carbon_offset_metric_tons_per_panel', 'Median_income', 'solar_potential_per_capita', 'solar_utilization', 'panel_utilization']:

# # Example Geo Plot (map of us)

# geo_plot(combined_df['existing_installs_count'] ,'rainbow', "Existing Installs", pos_df)
# geo_plot(combined_df['number_of_panels_total'] ,'rainbow', "Panel Total", pos_df)
# geo_plot(combined_df['panel_utilization'] ,'rainbow', "Panel Util", pos_df)
# geo_plot(np.log(combined_df['solar_potential_per_capita']) ,'rainbow', "Log Solar potential per capita", pos_df)

#### MAIN PLOTS FOR PAPER ############

print("Plotting")

##### INTRO PLOTS #########

# Demonstrates there is an issue of panel locations
# states = ['California']
# combined_df = combined_df[combined_df['state_name'].isin(states)]
# scatter_plot(x=np.log(combined_df['carbon_offset_metric_tons']), y=np.log(combined_df['existing_installs_count']), xlabel="Log Potential carbon offset", ylabel="Log Existing Panel Count", title=None, fit=[1,2], log=False, color="red")
# co_bins_quartile = q_binning(combined_df['carbon_offset_metric_tons'].values, 'carbon_offset_metric_tons', q=2, legible_label="Carbon Offset" )
# co_per_bins_quartile = q_binning(combined_df['carbon_offset_metric_tons_per_panel'].values, 'carbon_offset_metric_tons_per_panel',q=4, legible_label="Carbon Offset Per Panel")
# complex_scatter(combined_df=combined_df, x=combined_df['carbon_offset_metric_tons'], y=combined_df['existing_installs_count'], xlabel="Potential carbon offset (Metric Tons)", ylabel="Existing Installed Panels", title=None, bins=co_bins_quartile, fit=[2], legend=True)
# scatter_plot(x=combined_df['carbon_offset_metric_tons_per_panel'], y=combined_df['existing_installs_count_per_capita'], xlabel="Potential carbon offset per panel", ylabel="Existing Panel Count Per Capita", title=None, fit=[2], log=False, color="red")

# Shows where we should put panels
# geo_plot(np.log(combined_df['carbon_offset_metric_tons'] * combined_df['existing_installs_count']) ,'rainbow', "Carbon offset Per Capita", pos_df)

#######################################

##### EXEMPLAR STATE CHOOSING #######

# Exemplar states carbon offset to demo why we picked them


# for key in ['carbon_offset_metric_tons_per_panel', 'carbon_offset_metric_tons']:
#     state_stats = pd.concat([stats_for_states(combined_df, key),state_df['State code']])
#     plot_state_stats(state_stats, states=None, key=key, sort_by='mean')



# Plot just the clean vs fossil generation of each state


# state_df['Non_solar_clean_prop'] = state_df['Clean_prop'] - state_df['Solar_prop']
# state_bar_plot(state_df, states=None, keys=['Solar_prop', 'Non_solar_clean_prop','Fossil_prop'], sort_by="Clean_prop", legend_loc='upper center', fontsize=15, ylabel="", title='', colors=['green', 'blue', 'red'])
# # Plot full breakdown of energy gen by state
# state_bar_plot(state_df,states=None, keys=['Solar_prop', 'Bioenergy_prop', 'Coal_prop','Gas_prop','Hydro_prop','Nuclear_prop','Wind_prop', 'Other Renewables_prop', 'Other Fossil_prop'], sort_by="Solar_prop",legend_loc='right',fontsize=40, ylabel="")

exemplar_states = ['Texas', 'California', 'Mississippi', 'Delaware', 'Massachusetts', 'US Total']
# exemplar_states = ['California', 'Florida', 'Vermont', 'Texas']

##################################################


# Exemplar states carbon offset to demo why we picked them
# for key in ['carbon_offset_metric_tons_per_panel', 'panel_utilization', 'Median_income']:
#     state_bar_plot(state_df, states=exemplar_states, keys=[key], ylabel=key, title="By state stats")


# Plot just the clean vs fossil generation of each state
# state_bar_plot(state_df, states=exemplar_states, keys=['Clean_prop','Fossil_prop'], sort_by="Clean_prop", legend_loc='right', fontsize=40, ylabel="")

# Plot full breakdown of energy gen by state
# state_bar_plot(state_df,states=exemplar_states, keys=['Solar_prop', 'Bioenergy_prop', 'Coal_prop','Gas_prop','Hydro_prop','Nuclear_prop','Wind_prop', 'Other Renewables_prop', 'Other Fossil_prop'], sort_by="Solar_prop",legend_loc='right',fontsize=40, ylabel="")


# state_bar_plot(state_df, states=exemplar_states, keys=['carbon_offset_metric_tons_per_panel', 'existing_installs_count_per_capita', 'Median_income'], sort_by='carbon_offset_metric_tons_per_panel', ylabel=key, title="By state stats")

# no_dc = state_df['State code'].isin(["DC", "HI"])

# state_df_no_dc = state_df[~no_dc]

# plot_state_map(state_df, key='carbon_offset_metric_tons_per_panel', fill_color='Greens', legend_name="Carbon Offset Per Panel")
# plot_state_map(state_df, key='Fossil_prop')
# plot_state_map(state_df[no_dc], key='Democrat_prop')
# plot_state_map(state_df_no_dc, key='panel_utilization', fill_color='OrRd', legend_name="Panel Utilization")
# plot_state_map(state_df, key='Republican_prop', legend_name="Republican Voter Proportion")
# plot_state_map(state_df, key='Median_income', legend_name="Median Income") 
# plot_state_map(state_df, key='black_prop', legend_name="Black Population Proportion")
# plot_state_map(state_df, key='white_prop')  
# plot_state_map(state_df, key='carbon_offset_metric_tons')
# plot_state_map(state_df[no_dc], key='existing_installs_count')
# plot_state_map(state_df, key='yearly_sunlight_kwh_kw_threshold_avg', fill_color='Greens', legend_name="Yearly Average Sunlight")
# plot_state_map(state_df, key='Clean_prop', fill_color='Greens', legend_name="Clean Energy Gen Proportion")
# plot_state_map(state_df, key='Solar_prop')

# Supporting plots (shows energy generation Splits)
# state_bar_plot(state_df,states=exemplar_states, keys=['Clean_prop','Fossil_prop'], sort_by="Clean_prop")
# energy_gen_bar_plot(state_energy_df,states=exemplar_states, keys=['Solar_prop', 'Bioenergy_prop', 'Coal_prop','Gas_prop','Hydro_prop','Nuclear_prop','Wind_prop', 'Other Renewables_prop', 'Other Fossil_prop'], sort_by="Solar_prop")


################# END OF VERY IMPORTANT PLOTS ######################################################3

# Scatter Plot exmple

# scatter_plot(x=combined_df['households_below_poverty_line'].values, y=np.log(combined_df['panel_utilization']), xlabel="Percent below poverty line", ylabel="panel utilization", title=None,fit=[1])
# scatter_plot(x=combined_df['black_prop'], y=combined_df['panel_utilization'], xlabel="Black proportion of pop", ylabel='Existing installs', fit=[1])
# scatter_plot(x=combined_df['black_prop'], y=combined_df['carbon_offset_metric_tons'], xlabel="Black proportion of pop", ylabel='Carbon Offset (metric tons)', fit=[1])

fontsize = 25

# Change Font size
font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

matplotlib.rc('font', **font)


exem_state_df = state_df[state_df['State'].isin(exemplar_states)]

df = state_df
# scatter_plot(x=df['Republican_prop'], y=df['carbon_offset_metric_tons'],xlabel="proportion republican voter", ylabel='Carbon offset (metric tons) per capita', title="Republican proportion vs Carbon offset" ,fit=[1],color="blue", alpha=1)

# for df in [exem_state_df]:
    # scatter_plot(x=df['Republican_prop'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="proportion republican voter", ylabel='Carbon offset per panel', title="Republican proportion vs Carbon offset" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Republican_prop'], y=df['panel_utilization'], texts=df['State code'], xlabel="proportion republican voter", ylabel='Panel Utilization', fit=[1],color="blue",alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Median_income'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="Median income", ylabel='Carbon offset (metric tons) per panel', title="Median income vs Carbon offset" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Median_income'], y=df['panel_utilization'], texts=df['State code'], xlabel="Median income", ylabel='Panel Utilization', fit=[1],color="blue",alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['black_prop'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="Black population proportion", ylabel='Carbon offset (metric tons) per panel', title="Black proportion vs Carbon offset" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['black_prop'], y=df['panel_utilization'], texts=df['State code'], xlabel="Black population proportion", ylabel='Panel Utilization', fit=[1],color="blue",alpha=0.1, fontsize=fontsize)

    # scatter_plot(x=df['Republican_prop'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="", ylabel='', title="" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Republican_prop'], y=df['panel_utilization'], texts=df['State code'],xlabel="", ylabel='', title="" , fit=[1],color="blue",alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Median_income'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="", ylabel='', title="" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['Median_income'], y=df['panel_utilization'], texts=df['State code'], xlabel="", ylabel='', title="" ,fit=[1],color="blue",alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['black_prop'], y=df['carbon_offset_metric_tons_per_panel'], texts=df['State code'],xlabel="", ylabel='', title="" ,fit=[1],color="blue", alpha=0.1, fontsize=fontsize)
    # scatter_plot(x=df['black_prop'], y=df['panel_utilization'], texts=df['State code'],xlabel="", ylabel='', title="" , fit=[1],color="blue",alpha=0.1, fontsize=fontsize)

# scatter_plot(x=combined_df['carbon_offset_metric_tons_per_capita'], y=np.log(combined_df['Median_income']), xlabel="Carbon Offset per capita", ylabel='Log Median Income', fit=[2])
# scatter_plot(x=combined_df['households_below_poverty_line'], y=np.log(combined_df['Median_income']), xlabel="Households below poverty line", ylabel='Log Median Income', fit=[2])


# Complex scatter plot example with separation for total pop and racial proportions separated by quartiles:
# This section just sets up the bins for each



# carbon_offset_outlier_removal = [('carbon_offset_metric_tons_per_capita', (0.01, 200), "carbon offset per capita below 200 and above 0", 'blue')]

# Because we want to run over each one of these binnings we concat them
# bins_list = [pop_bins_quartile, white_prop_bins_quartile, asain_prop_bins_quartile, black_prop_bins_quartile, income_bins_quartile, co_bins_quartile]

# bins_list = [black_prop_bins_quartile]


# Then run them together (fit here is 1 giving a linear fit)
# for bins in bins_list:
#     complex_scatter(combined_df=combined_df, x=combined_df['carbon_offset_metric_tons'], y=combined_df['existing_installs_count_per_capita'], xlabel="Potential carbon offset (Metric Tons)", ylabel="Existing Installed Panels", title=None, bins=bins, fit=[2])
#     complex_scatter(combined_df=combined_df, x=combined_df['Median_income'], y=combined_df['existing_installs_count'], xlabel="Median Income", ylabel="Existing Installed Panels", title=None, bins=bins, fit=[2])
#     # complex_scatter(combined_df=combined_df, x=combined_df['percent_below_poverty_line'].values, y=combined_df['panel_utilization'], xlabel="Percent below poverty line", ylabel="panel utilization", title=None, bins= bins, fit=[4])
#     # complex_scatter(combined_df=combined_df, x=combined_df['percent_below_poverty_line'].values, y=combined_df['carbon_offset_metric_tons_per_panel'].values, xlabel="Percent below pverty line", ylabel="Carbon offset per panel", title=None, bins=bins, fit=[4])
#     complex_scatter(combined_df=combined_df, x=combined_df['carbon_offset_metric_tons_per_panel'], y=combined_df['existing_installs_count_per_capita'], xlabel="Potential carbon offset (Metric Tons) per panel", ylabel="Existing Installed Panels", title=None, bins=bins, fit=[2], legend=True)