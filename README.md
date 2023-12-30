# Graph Partition using Genetic Algorithm

This script represents a solution to the graph partitioning problem using a Genetic Algorithm (GA) approach. The concept of graph partitioning involves dividing a graph into parts or partitions where the number of edges that cut across the partitions are minimized.

## Description

The program is written in Python and makes use of the following libraries:

* random: For generating random numbers
* numpy: For scientific computing
* networkx: For creation and manipulation of complex networks
* matplotlib: For data visualization
* deap: For evolutionary computation

The GA is used to evolve a solution to the problem by progressively improving candidate solutions according to a fitness function. In this case, the fitness function is community modularity which helps to evaluate the quality of the partition.

## Setup

After defining the graph and the number of partitions, we set up the GA components using DEAP framework. We define the individual and the fitness evaluation function. The individual is defined as a list where the index represents the node and the value represents the community to which the node belongs. The fitness evaluation function measures the community modularity. We also define helper functions for decoding an individual to communities and for plotting the graph.

The core of the genetic algorithm lies in the custom_eaSimple function where the GA is applied on the graph. The GA flow includes initialization of population, evaluation of fitness, selection of offspring, crossover, mutation, evaluation of new offspring, updating the hall of fame, and replacing the current population with the new generation.

The GA is run across multiple generations where for each generation, the offspring undergo the process of selection, crossover, and mutation, and the best individuals are updated.

### How to Run


To use the script, you need to have Python installed along with the necessary packages (numpy, networkx, matplotlib, deap).
Simply run the script by typing python <main.py> on a command line