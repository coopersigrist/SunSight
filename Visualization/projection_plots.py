import seaborn as sns
import matplotlib.pyplot as plt

from Visualization.plot_util import *
from Data.data_load_util import *
from Simulation.projections_util import Projection, Objective, create_paper_objectives, create_equity_objectives

#ratio comparison projection
def plot_comparison_ratio(base_projection:Projection, comparison_projection:Projection, base_key, comparison_key, objectives:list[Objective] = create_paper_objectives(), interval = 10000, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"]):
    
    # plt.style.use("seaborn")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)

    #calculate ratios between the base and the comparison
    ratios = pd.DataFrame()
    comp = pd.DataFrame(comparison_projection.objective_projections)
    base = pd.DataFrame(base_projection.objective_projections)

    #interpolate x-values based on interval
    new_x = np.arange(0, int(comp.index.max()) + 1, interval)
    #compared projection
    comp=comp.sort_index()
    comp_interp = pd.DataFrame(index=new_x)
    for objective in objectives:
        objective_name = objective.name
        comp_interp[objective_name] = np.interp(new_x, comp.index, comp[objective_name])
    comp = comp_interp


    #base projection
    base = base.sort_index()
    base_interp = pd.DataFrame(index=new_x)
    for objective in objectives:
        objective_name = objective.name
        base_interp[objective_name] = np.interp(new_x, base.index, base[objective_name])
    base = base_interp

    #calculate ratios
    for objective in objectives:
        objective_name = objective.name
        ratios[objective_name] = comp[objective_name]/base[objective_name]
    
    ratios = ratios.fillna(1) #remove all NAN items

    #plot ratios
    # x = np.arange(math.ceil(len(ratios[objectives[0]]) / interval)) * interval
    for key, fmt in zip([objectives.name for objectives in objectives], fmts):
        #use x values from the ratios list
        plt.plot(ratios.index.tolist(), np.array(ratios[key]), fmt, label=key, linewidth=3, markersize=8, alpha=0.9)
        plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)

    #show baseline
    plt.hlines(1, 0, ratios.index.tolist()[-1], colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)

    # plt.ylim(0, 2) #set the range of the plots
    plt.ylabel(f"Ratio to {base_key}", fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/2)
    # plt.title(f"Performance of {comparison_key}")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.tight_layout()
    plt.show()


#bar graph version of ratio comparison, compares many different projections
def plot_bar_comparison_ratio(base_projection:Projection, all_projections:list[Projection], objectives:list[Objective] = create_paper_objectives(), panel_count = 1000000, fontsize = 15):
    #get the last value for all objectives for all methods
    results = [] #ex: array of [lexicase results, tournament results etc.]
    base = pd.DataFrame(base_projection.objective_projections)

    for projection in all_projections:
        result = [] #array of [CO, EG, RE, IE]
        proj = pd.DataFrame(projection.objective_projections)
        for objective in objectives:
            objective_name = objective.name
            result.append(proj[objective_name][panel_count] / base[objective_name][panel_count]) #get the ratio to the base key
        results.append(result)

    # Configuration for the bar graph
    x = np.arange(len(objectives)) # X positions for the groups
    width = 0.1  # Width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add bars for each method
    for i, proj in enumerate(all_projections):
        ax.bar(x + i * width, results[i], width, label=proj.name)
    
    #show baseline
    plt.axhline(y=1, color='b', linestyle='--', linewidth=1) 

    # Add labels, title, and legend
    # ax.set_xlabel('Objectives')
    ax.set_ylabel(f'Fitness Ratio to {base_projection.name}', fontsize=fontsize)
    # ax.set_title('Fitness of Selection Methods for all Objectives')
    ax.set_xticks(x + width*(len(all_projections)-1)/2)
    ax.set_xticklabels([objectives.name for objectives in objectives], fontsize=fontsize/1.5)
    ax.legend(fontsize=fontsize/1.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    # plt.ylim(0, 2) #set the y axis bounds

    # Show the plot
    plt.tight_layout()
    plt.show()

#bar graph equity comparison over panel counts
def plot_equity_comparison(projection:Projection, objectives:list[Objective] = create_equity_objectives(), panel_counts = [0,100000,1000000], fontsize = 15):
    results = [] #arr of equity objective values for different panel counts of a single projection

    proj = pd.DataFrame(projection.objective_projections)

    #interpolate x-values based on given panel counts
    new_x = np.array(panel_counts)
    proj = proj.sort_index()
    proj_interp = pd.DataFrame(index=new_x)
    for objective in objectives:
        objective_name = objective.name
        proj_interp[objective_name] = np.interp(new_x, proj.index, proj[objective_name])
    proj = proj_interp

    #put results into an array
    for panel_count in panel_counts:
        result = [] #array of [CO, EG, RE, IE]
        for objective in objectives:
            objective_name = objective.name
            result.append(proj[objective_name][panel_count])
        results.append(result)

    # Configuration for the bar graph
    x = np.arange(len(objectives)) # X positions for the groups
    width = 0.1  # Width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add bars for each method
    for i, panel_count in enumerate(panel_counts):
        ax.bar(x + i * width, results[i], width, label=f"{panel_count} panels")
    
    #show baseline
    plt.axhline(y=1, color='b', linestyle='--', linewidth=1) 

    # Add labels, title, and legend
    # ax.set_xlabel('Objectives')
    ax.set_ylabel(f'Equity Value', fontsize=fontsize)
    # ax.set_title('Fitness of Selection Methods for all Objectives')
    ax.set_xticks(x + width*(len(panel_counts)-1)/2)
    ax.set_xticklabels([objectives.name for objectives in objectives], fontsize=fontsize/1.5)
    ax.legend(fontsize=fontsize/1.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    # plt.ylim(0, 2) #set the y axis bounds

    # Show the plot
    plt.tight_layout()
    plt.show()
