import random
import numpy
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Define your graph and number of partitions here
# G = nx.karate_club_graph()
G = nx.random_partition_graph([10, 10, 10], 0.25, 0.01)
k = 3


# Fitness Evaluation
def evaluate(individual):
    communities_as_list = decode_individual(individual)
    return nx.algorithms.community.modularity(G, communities_as_list),


# Decode individual to communities
def decode_individual(individual):
    communities = {i: set() for i in range(k)}
    for node, community in enumerate(individual):
        communities[community].add(node)
    communities_as_list = [nodes for nodes in communities.values()]
    return communities_as_list


# Plot Graph Function
def plot_graph(G, communities_as_list, generation, show=False):
    partition_colors = ['red', 'blue', 'green', 'yellow', 'purple']
    partition = {node: com for com, nodes in enumerate(communities_as_list) for node in nodes}
    color_map = [partition_colors[partition[node] % len(partition_colors)] for node in G]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color=color_map, edge_color='gray', with_labels=True, node_size=100)
    plt.title(f"Generation {generation}")
    if show:
        plt.show()
    else:
        plt.savefig(f"graph_generation_{generation}.png")
    plt.close()


# Set up the GA components
def setup_toolbox(G, k):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, k - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(G.nodes()))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


# Custom Genetic Algorithm
def custom_eaSimple(G, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    pop = toolbox.population(n=50)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(pop)

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Optional: Save or show graph of the best individual
        if gen % 10 == 0:  # Adjust this to control frequency
            best_ind = tools.selBest(pop, k=1)[0]
            plot_graph(G, decode_individual(best_ind), gen)

    return pop, logbook


toolbox = setup_toolbox(G, k)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)

pop, log = custom_eaSimple(G, toolbox, 0.5, 0.2, 100, stats=stats)
best_ind = tools.selBest(pop, k=1)[0]
plot_graph(G, decode_individual(best_ind), "Final", show=True)
