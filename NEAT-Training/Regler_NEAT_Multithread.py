# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:02:27 2020

@author: cpommer
"""
#region Init

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import neat.config
import multiprocessing, time, ctypes, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import scipy
import matplotlib.pyplot as plt
#import tensorflow as tf
import pickle
import copy

import visualize
from itertools import cycle
from datetime import datetime
import multiprocessing
import time
import Regler_Backend
import sys
import random

Processes = []


locklist = []
Modellist = []




cycol = cycle('bgrcmk')

queue_in    = multiprocessing.Queue()
queue_out   = multiprocessing.Queue()
drawqueue   = multiprocessing.Queue()

numofprocesses = 3

#Lets start with 4 Threads!
def Keras_Model_Process(queue_input, queue_output, drawqueue):
    import tensorflow as tf
    sys.stdout.flush()
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    PreTrainedModel = tf.keras.models.load_model("Best_model0")
    while(True):
        #if queue_input.not_empty:
        id, net, Done, speed = queue_input.get()
        queue_output.put([id, Regler_Backend.Trainingloop(PreTrainedModel, net, drawqueue, True, speed), False])


now = datetime.now()

Datatime = ""

# axen 0-x; 1-y; 2-z
generations = 1000
numberofentities = 100
numofslices = 50
minslicedistance = 1.0 / numofslices


#tf.keras.backend.clear_session()
#PreTrainedModel =tf.keras.models.load_model("Best_model0")

initlifepoints = 1
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

print("Model loaded")
Generation = 0

def eval_genomes(genomes, config):

    global Generation
    i=0
    speed = random.uniform(0.5,1.5)
    #genomes = pool.imap_unordered(Engine, genomes)
    for genome_id, genome in genomes:
        genome.fitness = -1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        queue_in.put([i, net, False, speed])
        i+=1
    print("Everything sorted!")
    anythingleft = True
    for i in range(len(genomes)):
        
        ID, Fitness, Done = queue_out.get(True)
        genomes[ID][1].fitness = Fitness
        printProgressBar(i, 250, prefix="Recieving: ")
    drawqueue.put([4, Generation, 1])
    Generation += 1

def eval_genome(genome, config):
    i=0

    #genomes = pool.imap_unordered(Engine, genomes)


    fitness = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = Regler_Backend.Trainingloop(Regler_Backend.PreTrainedModel, net, 0.1, False, 'grey', False)
    return fitness

########################################################################################################################
#Neat Implementation
########################################################################################################################



#winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
if __name__ == '__main__':

    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    dirstring = "Winningspree/"

    multiprocessing.freeze_support()

    for i in range(numofprocesses):
        Processes.append(multiprocessing.Process(target=Keras_Model_Process, args=(queue_in, queue_out, drawqueue)))
        Processes[i].daemon = True
        Processes[i].start()
    p = multiprocessing.Process(target=Regler_Backend.Drawmodels,args=(drawqueue, dirstring))
    p.start()
    neatconfig = config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                      "Neatconfig_Config.ini")
    #p = neat.Population(config)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-270')

    p.config.no_fitness_termination = True
    p.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()

    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    try
    os.mkdir(dirstring)
    except
    os.mkdir(dirstring + "draw_net")
    os.mkdir(dirstring + "Evalresults")
    for i in range(1):

        winner = p.run(eval_genomes, 1)

        node_names = {-1: 'D1',
                      -2: 'D2',
                      -3: 'D3',
                      -4: 'D4',
                      -5: 'D5',
                      -6: 'D6',
                      -7: 'D7',
                      -8: 'D8',
                      -9: 'D9',
                      -10: 'D10',
                      -11: 'D11',
                      -12: 'D12',
                      -13: 'D13',
                      -14: 'D14',
                      -15: 'D15',
                      -16: 'D16',
                      -17: 'D17',
                      -18: 'D18',
                      -19: 'D19',
                      -20: 'VIN',
                      0: 'Vout'}

        visualize.draw_net(config, winner, False,filename=dirstring+"draw_net/"+str(i), node_names=node_names,fmt='svg')
        visualize.plot_stats(stats, ylog=False, view=False, filename=dirstring + "structures.svg")
        visualize.plot_species(stats, view=False, filename=dirstring + "species.svg")
    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')

    winner = []
    with open("Best_winner_290521.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

#Trainingloop(winner_net,0.1, False, 'red')

#node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
#visualize.draw_net(config, winner, True, node_names=node_names)
#visualize.plot_stats(stats, ylog=False, view=True)
#visualize.plot_species(stats, view=True)

#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
#p.run(eval_genomes, 10)
########################################################################################################################
#End Neat Implementation
########################################################################################################################