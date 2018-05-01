#!/usr/bin/env python

import sys
import neat
import time
from rominfo import *
from utils import *
from multiprocessing import Process, Pipe
#from memory_profiler import profile


def choose_action(result):
    action = 0
    for i in range(len(main_actions)):        
        if result[i] > 0.5:
            action += main_actions[i]
    return action
    
#@profile
def eval_genome(genome, config, connection):
    TIMEOUT = 100
    runs_per_net = 1 #DEFINIR DEPOIS
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []    
    timeout = TIMEOUT

    for runs in range(runs_per_net):
        rle = loadInterface(False)
        state, xi, y = getInputs(rle.getRAM()) #features
        x = xi
        rightmost = xi
        fitness = 0.0
        while not rle.game_over() and timeout > 0:
            #state_aggr = list(state)
            inputs = state
            result = net.activate(inputs) #Roda na net do NEAT

            action = choose_action(result)

            reward = performAction(action, rle)
            state, x, y = getInputs(rle.getRAM())

            timeout -= 1
            if(x > rightmost):
                rightmost = x
                timeout = TIMEOUT
            fitness = float(x - xi)
        fitnesses.append(fitness)
    fitness = min(fitnesses)
    connection.send(fitness)
    connection.close()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Avalia cada genoma em um processo separado para evitar problemas de memória
        parent_conn, child_conn = Pipe(False)
        p = Process(target=eval_genome, args=(genome, config, child_conn))
        p.start()
        genome.fitness = parent_conn.recv()


def run(generation = 0, num_iterations = None):
    # Carrega o config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    check_point = neat.Checkpointer(generation_interval=5)    
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
    winner = pop.run(eval_genomes, num_iterations)        

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)
        print("Winner salvo")

    # Salva o config usado
    with open('winner_config', 'wb') as f:
        pickle.dump(config, f)
        print("winner_config salvo")

if __name__ == '__main__':
    run()











