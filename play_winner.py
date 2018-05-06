from rominfo import *
from utils import *
import pickle
import neat

def choose_action(result, actions):
    action = 0
    for i in range(len(actions)):        
        if result[i] > 0.5:
            action += actions[i]
    return action

def play(genome, config, actions):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rle = loadInterface(True)
    state, xi, y = getInputs(rle.getRAM())
    while not rle.game_over():
        result = net.activate(state) #Roda na net do NEAT
        action = choose_action(result, actions)
        reward = performAction(action, rle)
        state, x, y = getInputs(rle.getRAM())
    print("Distance:", x - xi)

def main():
    with open('winner', 'rb') as f:
        winner_genome, config, actions = pickle.load(f)    
        play(winner_genome, config, actions)

if __name__ == '__main__':
    main()
