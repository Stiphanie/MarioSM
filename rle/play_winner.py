from rominfo import *
from utils import *
import pickle
import neat

def choose_action(result):
    action = 0
    for i in range(len(main_actions)):        
        if result[i] > 0.5:
            action += main_actions[i]
    return action

def play(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rle = loadInterface(True)
    state, xi, y = getInputs(rle.getRAM())
    while not rle.game_over():
        result = net.activate(state) #Roda na net do NEAT
        action = choose_action(result)
        reward = performAction(action, rle)
        state, x, y = getInputs(rle.getRAM())
    print(x - xi)

def main():
    with open('winner-feedforward', 'rb') as f:
        winner_genome = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config')
        print(winner_genome)
        play(winner_genome, config)

if __name__ == '__main__':
    main()
