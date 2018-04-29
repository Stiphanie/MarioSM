#!/usr/bin/env python

import sys
import neat
import time
from rominfo import *
from utils import *
#from memory_profiler import profile


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
    
#@profile
def eval_genome(genome, config):
    global rle
    TIMEOUT = 100
    runs_per_net = 1 #DEFINIR DEPOIS
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []    
    timeout = TIMEOUT

    for runs in range(runs_per_net):
        rle.saveState()
        state, xi, y, l1x, l1y = getInputs(rle.getRAM()) #features
        x = xi
        rightmost = xi
        fitness = 0.0
        while not rle.game_over() and timeout > 0:

            state_aggr = list(state)
            inputs = (state_aggr)
            result = net.activate(inputs) #Roda na net do NEAT

            action = choose_action(result)

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
        #rle.loadROM('super_mario_world.smc', 'snes') -> É mais rápido, mas consome memória do mesmo jeito
        rle.loadState()
    fitness = max(fitnesses)
    return fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        #print("\n")
        print(genome.fitness)
        #print("\n")


def run(generation = 0, numIterations = 10000):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    global rle
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    #config_path = 'config'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    check_point = neat.Checkpointer(generation_interval=5)    
    #stats = neat.StatisticsReporter()
    #pop.add_reporter(stats)
    rle = loadInterface(True)
    pop = None
    if(generation == 0):
        # Inicia nova população
        pop = neat.Population(config)
    else:
        # Restaura população de um checkpoint
        filename = 'neat-checkpoint-' + str(generation)
        pop = check_point.restore_checkpoint(filename)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(check_point)
    winner = pop.run(eval_genomes, numIterations)        

    #pe = neat.ParallelEvaluator(4, eval_genomes)
    #print("------------ENTROU WINNER--------------")
    
    #print("------------SAIU WINNER--------------")
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)
        print("Winner salvo")

    #return winner

if __name__ == '__main__':
    run()











