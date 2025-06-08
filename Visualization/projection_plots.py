import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler

from Visualization.plot_util import *
from Data.data_load_util import *
from Simulation.projections_util import Projection, Objective, create_paper_objectives, create_equity_objectives

#plot multiple projections over a single objective
def plot_projections(projections:list[Projection], objective:str="Carbon Offset", interval=100000, panel_estimations=None, net_zero_horizontal=False, fontsize=20, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], ylabel=None, color_palette = sns.color_palette("Set2"), **kwargs):
    # Some default sizing and styling
    # print(plt.style.available)
    # plt.style.use('seaborn-v0_8')
    # font = {'family' : 'DejaVu Sans',
    # 'weight' : 'bold',
    # 'size'   : fontsize}
    # matplotlib.rc('font', **font)
    plt.style.use("seaborn-v0_8")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)
    plt.figure()

    # Adds a horizontal line at the point where Status-Quo is expected to be when net-zero carbon emissions (479000 * 3 panels) is reached.
    if net_zero_horizontal and 'Status-Quo' in projections:
        two_mill_continued = np.array(projections['Status-Quo'])[479000 * 3]

    interp_projections = {projection.name: projection.interpolateObjectiveProjections(interval=interval, return_df=False) for projection in projections}

    ax = plt.subplot()
    ax.set_prop_cycle(cycler(color=color_palette)) #color palette

    for projection, fmt in zip(projections, fmts):
        interp_projection = interp_projections[projection.name][objective]
        ax.plot(interp_projection.keys(), interp_projection.values(), fmt, label=projection.name, linewidth=3, markersize=8, alpha=0.9, **kwargs)
        # projection.add_proj_to_plot(ax=ax, objective=objective, linewidth=3, markersize=8, alpha=0.9, fmt=marker, **kwargs)

    # plt.locator_params(axis='x', nbins=8) 
    # plt.locator_params(axis='y', nbins=8) 
    # plt.yticks(fontsize=fontsize/(1.2))
    # plt.xticks(fontsize=fontsize/(1.2))

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
    plt.xticks(fontsize=fontsize/1.5)
    plt.yticks(fontsize=fontsize/1.5)

    plt.legend(fontsize=fontsize/1.5)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.tight_layout()
    plt.show()

#ratio comparison projection
def plot_comparison_ratio(base_projection:Projection, comparison_projection:Projection, base_key, comparison_key, objectives:list[Objective] = create_paper_objectives(), interval = 100000, fontsize=20, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p"], color_palette = sns.color_palette("deep")):
    
    plt.style.use("seaborn-v0_8")
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : fontsize}

    matplotlib.rc('font', **font)
    plt.figure()

    #calculate ratios between the base and the comparison
    ratios = pd.DataFrame()
    comp = comparison_projection.interpolateObjectiveProjections(interval=interval)
    base = base_projection.interpolateObjectiveProjections(interval=interval)

    #calculate ratios
    for objective in objectives:
        objective_name = objective.name
        ratios[objective_name] = comp[objective_name]/base[objective_name]
    
    ratios = ratios.fillna(1) #remove all NAN items

    #plot ratios
    ax = plt.subplot()
    ax.set_prop_cycle(cycler(color=color_palette)) #color palette

    for key, fmt in zip([objectives.name for objectives in objectives], fmts):
        #use x values from the ratios list
        ax.plot(ratios.index.tolist(), np.array(ratios[key]), fmt, label=key, linewidth=3, markersize=8, alpha=0.9)
    plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)

    #show baseline
    plt.hlines(1, 0, ratios.index.tolist()[-1], colors='black' , linestyles='dashed', linewidth=2, alpha=0.5)

    # plt.ylim(0, 2) #set the range of the plots
    plt.ylabel(f"Ratio to {base_key}", fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/2)
    # plt.title(f"Performance of {comparison_key}")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    #       ncol=1, shadow=True, fontsize=fontsize/1.4)
    plt.xticks(fontsize=fontsize/1.5)
    plt.yticks(fontsize=fontsize/1.5)
    plt.tight_layout()
    plt.show()


#bar graph version of ratio comparison, compares many different projections
def plot_bar_comparison_ratio(base_projection:Projection, all_projections:list[Projection], objectives:list[Objective] = create_paper_objectives(), panel_count = 1000000, fontsize = 15, color_palette = sns.color_palette("muted")):
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
    ax.set_prop_cycle(cycler(color=color_palette)) #color palette

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
def plot_equity_comparison(projection:Projection, objectives:list[Objective] = create_equity_objectives(), panel_counts = [0,100000,1000000], fontsize = 15, color_palette = sns.color_palette("Greens")):
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
    ax.set_prop_cycle(cycler(color=color_palette)) #color palette

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
