from neat import DefaultReproduction
import random
import math

import numpy as np
from neat.math_util import mean

class FitnessPropReproduction(DefaultReproduction):
    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.k = 16 #number of genomes per tournament

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes using Fitness Proportion
        The only change is the selection within a species: choose the parents based on a softmax distribution of the fitnesses
        """
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            # repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
            #                              len(old_members)))
            # # Use at least two parents no matter what the threshold fraction result is.
            # repro_cutoff = max(repro_cutoff, 2)
            # old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                weights = [m[1].fitness for m in old_members]
                parent1_id, parent1 = random.choices(old_members, weights=weights)[0]
                parent2_id, parent2 = random.choices(old_members, weights=weights)[0]

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                # TODO: if config.genome_config.feed_forward, no cycles should exist
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population

class TournamentReproduction(DefaultReproduction):
    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.k = 16 #number of genomes per tournament

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes using Tournament Selection
        Much of this is taken from the parent reproduce function
        """


        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                # all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []
        #create the new population by crossover
        #NOTE: do not care about species, only the tourney winners
        
        spawn = pop_size #number of things to spawn in the new generation
        new_population = {}
        species.species = {}
        all_genomes = []
        for s in remaining_species:

            #record all genomes in the species in the form (id, genome)
            old_members = list(s.members.items())
            all_genomes.extend(old_members)
            
            species.species[s.key] = s
            s.members = {}
            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                # Sort members in order of descending fitness. (only if elitism is on)
                old_members.sort(reverse=True, key=lambda x: x[1].fitness)
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

        #run tournaments of size k to pick parents
        winners = []
        k = min(self.k, len(all_genomes)) #tournament can only be as big as the population
        for tournament in range(2*spawn):
            #choose k genomes
            indices = np.random.permutation(len(all_genomes))
            competitors = [all_genomes[j] for j in indices[:k]]
            winners.append(max(competitors, key=lambda x: x[1].fitness))
            
            # print("competitor fitness: ", [c[1].fitness for c in competitors])
            # print("winner fitness:", winners[-1][1].fitness)


        # Choose parents based on tournament winners
        while spawn > 0:
            spawn -= 1

            parent1_id, parent1 = winners[spawn*2]
            parent2_id, parent2 = winners[spawn*2 + 1]

            # Note that if the parents are not distinct, crossover will produce a
            # genetically identical clone of the parent (but with a different ID).
            gid = next(self.genome_indexer)
            child = config.genome_type(gid)
            child.configure_crossover(parent1, parent2, config.genome_config)
            child.mutate(config.genome_config)
            # TODO: if config.genome_config.feed_forward, no cycles should exist
            new_population[gid] = child
            self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
