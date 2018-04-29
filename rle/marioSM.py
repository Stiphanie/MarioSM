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

def choose_action(result):
    '''
    result = result[0]
    if result < 0.2:
        return 66
    elif result < 0.4:
        return 130
    elif result < 0.6:
        return 128
    elif result < 0.8:
        return 131
    else:
        return 386
    '''
   
    action = 0
    #print(result)
    #print(len([x for x in result if x > 0.4]))
    for i in range(len(main_actions)):        
        if result[i] > 0.5:
            #print(main_actions[i])
            action += main_actions[i]
    #print(action)
    return action


def eval_genome(genome, config):
    TIMEOUT = 100
    runs_per_net = 1 #DEFINIR DEPOIS
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    rle = loadInterface(True)
    timeout = TIMEOUT
    #print("-------ENTROU GENOMA---------")
    #print(genome)
    for runs in range(runs_per_net):
        # Run the given simulation for up to num_steps time steps.
        state, xi, y, l1x, l1y = getInputs(rle.getRAM()) #features
        x = xi
        rightmost = xi
        fitness = 0.0
        while not rle.game_over() and timeout > 0:
            #print(x, y)
            print(l1x, l1y)
            state_aggr = list(state)
            state_aggr.append(y)
            #print(state_aggr)
            #state = list(state)
            inputs = (state_aggr)
            result = net.activate(inputs) #Roda na net do NEAT

            #print(result)
            action = choose_action(result)

            #print(action)
            # Testa
            reward = performAction(action, rle)
            state, x, y, l1x, l1y = getInputs(rle.getRAM())

            timeout -= 1
            if(x > rightmost):
                rightmost = x
                timeout = TIMEOUT

            #print("O VALOR DE X É "+str(x))
            fitness = float(x - xi)
            #print("mama meus ovo")
        fitnesses.append(fitness)
        #print("morreu")
    # The genome's fitness is its worst performance across all runs.
    #print("-------SAIU GENOMA---------")
    fitness = max(fitnesses)
    #print(fitness)
    #print(type(fitness))
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print("\n")
        print(genome.fitness)
        print("\n")

def run(i):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config_path = 'config'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    check_point = neat.Checkpointer(generation_interval=1)    
    #pop = neat.Population(config)
    #stats = neat.StatisticsReporter()
    #pop.add_reporter(stats)
    
    pop = None
    if(i == 0):
        pop = neat.Population(config)
    else:
        filename = 'neat-checkpoint-' + str(i)
        pop = check_point.restore_checkpoint(filename)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(check_point)
    winner = pop.run(eval_genomes)
        

    #pe = neat.ParallelEvaluator(4, eval_genomes)
    #print("------------ENTROU WINNER--------------")
    
    #print("------------SAIU WINNER--------------")
    # Save the winner.
    #with open('winner-feedforward', 'wb') as f:
        #pickle.dump(winner, f)

    #return winner

if __name__ == '__main__':
    i = 0
    while(True):
        run(i)
        i += 1
    #print(r)
    print("-----------CABO SABOSTA-------")










