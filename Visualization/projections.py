from plot_util import *
from data_load_util import *
from projections_util import *
from tqdm import tqdm

 
''' TODO TEST '''
def plot_projections(projections:list[Projection], objective:str="Carbon Offset", panel_estimations=None, net_zero_horizontal=False, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], ylabel=None, **kwargs):

    # Some default sizing and styling
    plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}
    matplotlib.rc('font', **font)

    # Adds a horizontal line at the point where Status-Quo is expected to be when net-zero carbon emissions (479000 * 3 panels) is reached.
    if net_zero_horizontal and 'Status-Quo' in projections:
        two_mill_continued = np.array(projections['Status-Quo'])[479000 * 3]

    ax = plt.subplot()
    for projection, marker in zip(projections, fmts):
        projection.add_proj_to_plot(ax=ax, objective=objective, **kwargs)

    plt.locator_params(axis='x', nbins=8) 
    plt.locator_params(axis='y', nbins=8) 
    plt.yticks(fontsize=fontsize/(1.2))
    plt.xticks(fontsize=fontsize/(1.2))

    # get ranges of the plots axes
    xmin, xmax, ymin, ymax = plt.axis()

    ''' TODO test'''
    if panel_estimations is not None:
        for label, value in panel_estimations:
            plt.vlines(value, ymin+ymax/18, ymax, colors='darkgray' , linestyles='dashed', linewidth=2, alpha=0.7)
            plt.text(value - (xmax-xmin)/23, ymin + ymax/80, label, alpha=0.7, fontsize=25)
    
    if net_zero_horizontal:
        plt.hlines(two_mill_continued, 0, xmax, colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)
        plt.text(0, two_mill_continued*1.1, "Continued trend at\nNet-zero prediction", alpha=0.95, fontsize=18, color='black')

    

    plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)
    if ylabel is None:
        plt.ylabel(objective, fontsize=fontsize, labelpad=20)
    else:
        plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)

    plt.legend(fontsize=fontsize/1.5)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.tight_layout()
    plt.show()

# Plots a map of where the zip_codes picked are located
''' TODO REFACTOR '''
def plot_picked(combined_df, picked, metric, title=""):

    if metric is None:
        region_list = list(combined_df['region_name'])
        occurence_counts = picked.value_counts()
        times_picked = np.zeros_like(combined_df['region_name'])
        for pick in picked.unique():
            times_picked[region_list.index(pick)] += occurence_counts[pick]
        
        combined_df['times_picked'] = times_picked
        metric ='times_picked'

        dot_size_scale = (40 * times_picked[combined_df['times_picked']>0]/ (np.max(combined_df['times_picked'][combined_df['times_picked']>0]))) + 40     
    picked = picked.astype(str)

    geo_plot(combined_df['times_picked'][combined_df['times_picked']>0], color_scale='agsunset', title=title, edf=combined_df[combined_df['times_picked']>0], zipcodes=picked.unique(), colorbar_label="", size=dot_size_scale)

def plot_demo_state_stats(new_df,save="Clean_Data/data_by_state_proj.csv"):
    state_df = load_state_data(new_df, load=None, save=save)

    hatches=['o','o','o','o','o','x','x','x','x','x']
    annotate = False
    type = 'paper'
    stacked = False

    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income','Republican'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_kg", type=type, stacked=stacked, ylabel="Carbon Offset Potential (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True) 

    hatches=['o','o','o','o','x','x','x','x']

    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True) 
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="carbon_offset_kg", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Potential Carbon Offset (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)

'''TODO REFACTOR (low prio)'''
def weighted_proj_heatmap(combined_df, metric='carbon_offset_kg_per_panel', objectives=['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop']):
    weight_starts = [0.0, 0.0]
    weight_ends = [0.5,1.5]
    number_of_samples = 5
    weighted_proj_array = create_many_weighted(combined_df, n=1850000, objectives=objectives, weight_starts=weight_starts, weight_ends=weight_ends, number_of_samples=5, metric=metric,
                                               save='Projection_Data/weighted_map_5_energy', load='Projection_Data/weighted_map_5_energy.npy')


    ax = sns.heatmap(weighted_proj_array[:,:,-1], xticklabels=np.round(np.arange(weight_starts[0],weight_ends[0], (weight_ends[0] - weight_starts[0])/number_of_samples), 1), yticklabels=np.round(np.arange(weight_starts[1],weight_ends[1], (weight_ends[1] - weight_starts[1])/number_of_samples), 1))
    ax.set_xlabel("Energy Potential Weight")
    ax.set_ylabel("Black Prop Weight")
    plt.show()


if __name__ == '__main__':
    combined_df = make_dataset(remove_outliers=True)
    max_num_added = 2000000
    projection_list = create_projections(combined_df, n_panels=max_num_added, save='Projection_Data/2mill_No_lex.pkl', load='Projection_Data/2mill_No_lex.pkl')
    panel_estimations_by_year = [("Net-Zero" , 479000 * 3), ("  2030  ", 479000 * 1), ("  2034  ", 479000 * 2)]

    plot_projections(projections=projection_list, panel_estimations=panel_estimations_by_year)