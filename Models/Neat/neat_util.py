# cd Visualization
# NEW NEAT UTIL
from datetime import datetime
import os
import pickle
import neat
import numpy as np
from Data.data_manager import DataManager
from Simulation.projections_util import Objective, create_neat_proj, create_paper_objectives
from Models.Neat.selection_util import TournamentReproduction, FitnessPropReproduction
from Data.data_load_util import make_dataset
from tqdm import tqdm
from .saving_util import *


class NeatModel():
    # TODO Move to neat Util
    def __init__(self, model):
        self.model = model #this model should be a neat-python model
    
    '''run the NEAT model given a DataManager as input
    Returns a dictionary with zip codes and scores
    {zip code: score}
    '''
    def run_network(self, data_manager: DataManager):
        zip_outputs = {}
        #TODO:FIX
        indices = range(data_manager.num_zips)

        for i in indices:
            score = self.model.activate(data_manager.network_inputs(i))
            zip_code = data_manager.combined_df.loc[i, 'region_name'] #find zip code from index
            zip_outputs[zip_code] = score
        return zip_outputs

class Evaluation():
    '''
    Genome Evaluation schemes like lexicase and weighted sum; works in tandem with Reproduction schemes
    '''
    def __init__(self, combined_df, data_manager, objectives, num_panels, weights=[1,1,1,1]):
        self.objectives = objectives
        self.num_panels = num_panels
        self.weights = weights
        self.overall_threshold = 0.3
        self.combined_df = combined_df
        self.data_manager = data_manager

    #NOTE: may be unnecessary with projections now up and running
    #score a simulation and record the cumulative score across all metrics
    def score_assignment(self, info, objective_ind, n=1):
        #change this to match the new refactor!!
        '''
        for the objective determined by metric_ind, rank a zip order
        
        '''
        #info[1] is projection objective scores, info[2] is cumulative score
        objective_name = self.objectives[objective_ind].name
        score = info[1][objective_name][self.num_panels]

        #TODO: find out what the type of "score" is... it should just be a number but apparently not!??
        # print(objective_name, score)

        #self.objectives[objective_ind].calc(self.combined_df, info[1])#data_manager.score(info[1], self.objectives[metric_ind], n, train=True)

        #placed_panels is an {zip code: panel count} we can't just feed it the straight up zip code
        # print("metric:", eval_metric,"score:",score)
        info[2] += score * self.weights[objective_ind]
        return score
    
    def eval(self, genomes, config):
        pass

class LexicaseEval(Evaluation):
    def eval(self, genomes, config):
        step_threshold = self.overall_threshold ** (1/sum(self.weights))
        best_score = 0 #record best score
        
        #get zip orders for all genomes by running each NN once
        genome_info = [] #stores genome, zip_order, and cumulative score
        for genome_id, genome in tqdm(genomes):
            genome.fitness = 0 #set all fitness to a minimum initially
                    
            model = NeatModel(neat.nn.FeedForwardNetwork.create(genome, config))
            projection = create_neat_proj(self.data_manager, self.num_panels, model, self.objectives)
            #objective projections is projection.objective_projections; is a dictionary of {objective name: objective score}

            genome_info.append([genome, projection.objective_projections, 0]) #genome pointer, zip_order, and cumulative score

        #lexicase: evaluate based on all metrics in random order
        objective_inds = np.random.permutation(len(self.objectives))
        for i in objective_inds:
            obj = self.objectives[i]
            metric_weight = self.weights[i]
            num_panels = self.num_panels #np.random.randint(NUM_PANELS_LOWER, NUM_PANELS_UPPER) #pick a random number of panels to evaluate for each objective for each generation
            
            #sort genomes based on the objective
            genome_info.sort(key = lambda info: self.score_assignment(info, i, num_panels), reverse=True)
            
            #naturally select the genome list by the step_threshold
            cutoff = np.ceil((step_threshold**metric_weight * len(genome_info))).astype(int)
            genome_info = genome_info[0:cutoff]

            #update ranking data for tiebreakers, a number closer to 0 is better (deprecated)
            # for i in range(len(genome_info)):
            #     genome_info[i][2] -= i
            
        #set tie-breaker fitness for final survivors
        final_threshold = np.ceil(len(genomes) * self.overall_threshold).astype(int)
        genome_info.sort(key = lambda info: info[2], reverse=True)

        for genome, zip_order, score in genome_info[0:final_threshold]:
            #genome.fitness = score
            genome.fitness = 1 #set fitness to 1 for the chosen genomes

            #find the best performing score in this generation
            if genome.fitness > best_score:
                best_score = genome.fitness

#fitness proportion TODO
class WeightedSumEval(Evaluation):
    def eval(self, genomes, config):

        best_score = 0 #record best score
        
        #get zip orders for all genomes by running each NN once
        genome_info = [] #stores genome, zip_order, and cumulative score
        for genome_id, genome in tqdm(genomes):
            genome.fitness = 0 #set all fitness to a 0 initially
            
            model = NeatModel(neat.nn.FeedForwardNetwork.create(genome, config))
            zip_order = model.run_network(data_manager)
            genome_info.append([genome, zip_order, 0]) #genome pointer, zip_order, and cumulative score

        #fitness prop: evaluate based on sum of all metrics (weighted)
        indices = np.arange(len(self.objectives))
        num_panels = self.num_panels #np.random.randint(NUM_PANELS_LOWER, NUM_PANELS_UPPER) #pick a random number of panels to evaluate for each generation

        for i in indices:
            for j in range(len(genome_info)):
                self.score_assignment(genome_info[j], i, num_panels)
            
        #set the fitnesses for all genomes
        for genome, zip_order, score in genome_info:
            genome.fitness = score #note: this score can go up to NUM_PANELS * len(self.objectives)

            #find the best performing score in this generation
            if genome.fitness > best_score:
                best_score = genome.fitness




class NeatTrainer():
    '''
    Wrapper class for all the training stuff of a NEAT
    '''
    def __init__(self, pop_size=20, num_generations=20, objectives:list[Objective]=[], overall_threshold=0.3):
        #constants
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.objectives = objectives #lexicase metrics to evaluate
        self.overall_threshold = overall_threshold #what fraction of TOTAL population reproduces, makes sure this matches 'survival_threshold' in neat-config
        # self.step_threshold = self.overall_threshold ** (1/sum(self.weights)) #calculate what fraction survives after each metric is applied sequentially

    def run(self, config_file, selection_method, reproduction_method=neat.DefaultReproduction, checkpoint=0):

        print("loading configuration...")
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, reproduction_method,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        #edit config and set global vars based on additional specifications
        # NUM_PANELS = panels

        # OVERALL_THRESHOLD = threshold #what fraction of TOTAL population reproduces, makes sure this matches 'survival_threshold' in neat-config
        # STEP_THRESHOLD = OVERALL_THRESHOLD ** (1/sum(METRIC_WEIGHTS))
        config.reproduction_config.survival_threshold = self.overall_threshold

        # POP_SIZE = pop_size
        config.pop_size = self.pop_size

        # NUM_GENERATIONS = self.num_generations

        # Create the population, which is the top-level object for a NEAT run.
        print("creating population...") #WARNING: population takes a LONG time to create
        if checkpoint == 0:
            p = neat.Population(config)
        else:
            p = neat.Checkpointer.restore_checkpoint(f'Neat/checkpoints/neat-checkpoint-{checkpoint}')

        # Add a stdout reporter to show progress in the terminal.
        print("setting reporters...")
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        # p.add_reporter(neat.Checkpointer(time_interval_seconds=1200, filename_prefix='Neat/checkpoints/neat-checkpoint-'))

        # Run for up to 300 generations.
        print("training model...")
        
        winner = p.run(selection_method, self.num_generations)

        # Display the winning genome.
        # print('\nBest genome:\n{!s}'.format(winner))

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        return winner_net



#random selection
# def eval_genomes_random(genomes, config):
#     for genome_id, genome in tqdm(genomes):
#         genome.fitness = 1 #set all fitness to a 1 and let the model do its thing
        
#do a single training run


#do a full K-fold runs
def K_fold_run(config_path, data_manager, selection_method, reproduction_method=neat.DefaultReproduction, k=5):
    data_manager.generate_folds(k)

    scores = []
    networks = []
    for i in range(k):
        print(f"running fold {i}")
        data_manager.set_fold(i)
        winner_net = run(config_path, selection_method, reproduction_method)
        networks.append(winner_net)

        #evaluate network on test set
        cv_order = run_network(winner_net, data_manager, cross_val=True)

        IE = data_manager.score(cv_order, 'income_equity', NUM_PANELS, train=True)
        RE = data_manager.score(cv_order, 'racial_equity', NUM_PANELS, train=True)
        CO = data_manager.score(cv_order, 'carbon_offset', NUM_PANELS, train=True)
        EG = data_manager.score(cv_order, 'energy_generation', NUM_PANELS, train=True)

        scores.append([IE, RE, CO, EG])
    #average scores
    scores_np = np.array(scores)
    return networks[0], np.mean(scores_np, axis=0)
    
#load datasets
# print("Loading data_manager for NEAT")
# combined_df = make_dataset(remove_outliers=True)
# state_df = load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
# data_manager = DataManager(combined_df, state_df)


if __name__=="__main__":
    print("Running")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Neat/neat-config')

    combined_df = make_dataset(remove_outliers=True)
    state_df = None #load_state_data(combined_df, load="Clean_Data/data_by_state.csv")
    data_manager = DataManager(combined_df, state_df)

    objectives = create_paper_objectives()
    trainer = NeatTrainer(pop_size = 5, num_generations=2, objectives=objectives, overall_threshold=0.3)
    lexicase = LexicaseEval(combined_df, objectives, 10000, [1,1,1,1])

    network = trainer.run(config_path, lexicase.eval)
    save_model(network, None, model_name="NEAT_model_lexicase.pkl")

    # k_folds = 5

    # #run lexicase
    # lexi_network, lexi_results = K_fold_run(config_path, data_manager, eval_genomes_lexicase, k=k_folds)
    # save_model(lexi_network, lexi_results, model_name="NEAT_model_lexicase.pkl", results_name="lexicase_results.pkl")
    
    # #run fitness prop
    # fp_network, fp_results = K_fold_run(config_path, data_manager, eval_genomes_weighted_sum, reproduction_method=FitnessPropReproduction, k=k_folds)
    # save_model(fp_network, fp_results, model_name="NEAT_model_fitness_prop.pkl", results_name="fitness_prop_results.pkl")

    # #tournament selection
    # tourney_network, tourney_results = K_fold_run(config_path, data_manager, eval_genomes_weighted_sum, reproduction_method=TournamentReproduction, k=k_folds)
    # save_model(tourney_network, model_name="NEAT_model_tournament.pkl", results_name="tournament_results.pkl")

    # #run random selection
    # rand_network, rand_results = K_fold_run(config_path, data_manager, eval_genomes_random, k=k_folds)
    # save_model(rand_network, model_name="NEAT_model_random.pkl")

    # #print results
    # result_metrics = ['income_equity', 'racial_equity', 'carbon_offset', 'energy_generation']
    # for i, res in enumerate(lexi_results):
    #     print(f"Lexicase {result_metrics[i]}", res)
    
    # for i, res in enumerate(fp_results):
    #     print(f"Fitness prop {result_metrics[i]}", res)
        
    # for i, res in enumerate(tourney_results):
    #     print(f"Tourney {result_metrics[i]}", res)

    # for i, res in enumerate(rand_results):
    #     print(f"Random {result_metrics[i]}", res)

    # for i, res in enumerate(rand_results):
    #     print(f"Random {result_metrics[i]}", res)
    







    # data_manager.train_test_split(test_size = 0.0) #no test set for now
    # data_manager.generate_folds(5)
    # data_manager.set_fold(0)

    # print("Running Genetic Algorithm")
    # winner_net = run(config_path)

    # #evaluate winner against the test set
    # cv_order = run_network(winner_net, data_manager, cross_val=True)

    # print("Carbon Offset score: ",data_manager.score(cv_order, 'carbon_offset', NUM_PANELS, train=True))

    # print("Energy Generation score: ",data_manager.score(cv_order, 'energy_generation', NUM_PANELS, train=True))
    
    # #save output into a pickle
    # with open('Neat/models/NEAT_model.pkl', 'wb') as f:
    #     pickle.dump(winner_net, f)

    # #save the best scores
    # with open('Neat/models/fitness_data.pkl', 'wb') as f:
    #     pickle.dump(best_scores, f)