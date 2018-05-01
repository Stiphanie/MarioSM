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
    print("Distance:", x - xi)

def main():
    winner_genome = None
    config = None
    with open('winner', 'rb') as f:
        winner_genome = pickle.load(f)
    
    with open('winner_config', 'rb') as f:
        config = pickle.load(f)

    play(winner_genome, config)

if __name__ == '__main__':
    main()
