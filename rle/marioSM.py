#!/usr/bin/env python
# marioQLearning.py
# Author: Fabrício Olivetti de França
#
# A simple Q-Learning Agent for Super Mario World
# using RLE

import sys
from rle_python_interface.rle_python_interface import RLEInterface
import numpy as np
from numpy.random import uniform, choice, random
import neat

import time
from rominfo import *
from utils import *

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    for runs in range(runs_per_net):
        # Run the given simulation for up to num_steps time steps.
        state, x, y = getInputs(rle.getRAM()) #features
        fitness = 0.0
        while not rle.game_over():
            state_aggr = np.sum(state)
            print(state_aggr)
            #state = list(state)
            inputs = (state_aggr, x, y)
            result = net.activate(inputs) #Roda na net do NEAT

            print(result)
            if result[0] > 0.5:
                action = 64
            else:
                action = 128
            print(action)
            # Testa
            reward = performAction(action, rle)
            state, x, y = getInputs(rle.getRAM())
            fitness = x
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config_path = 'config'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    #pe = neat.ParallelEvaluator(4, eval_genomes)
    winner = pop.run(eval_genomes, 1000)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

runs_per_net = 1 #DEFINIR DEPOIS
rle = loadInterface(False)
run()
