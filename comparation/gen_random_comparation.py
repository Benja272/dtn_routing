import findspark
findspark.init()
import sys
sys.path.append('../brufn-0.0.2/brufn/')
sys.path.append('../')
import os
from brufn.network import Net, Contact
from copy import copy
import random
import json
from brufn.brufspark import BRUFSpark
from brufn.utils import print_str_to_file
from pyspark import SparkContext, SparkConf
from experiment_generator import generate_omnet_ini_file, generate_bash_script
from brufn.helper_ibrufn_function_generator6 import IBRUFNFunctionGenerator
import ruting_morucop as net


DTNSIM_PATH = "/home/benja/Documents/facu/tesis/project/dtnsim/dtnsim/src"
SEED = 10
PATH_TO_RESULT = 'results'
SOURCES = range(8) # Nodes that will send packages to every target (except themselves)
TARGETS = range(8) # Nodes that will recieve packages from every sources
COPIES_RNG = [2] #[1,2,3]
NET_RNG = [0,1] #range(10)
TS_DURATION_IN_SECONDS = 10 # Duration of a single time stamp
random.seed(SEED) # Seed to generate random networks
NUM_OF_REPS = 1000 # Number of simulations that will be runned in OMNET++ for BRUF/IBRUF simulation
TRAFIC = dict((s+1, [t+1 for t in TARGETS if t != s]) for s in SOURCES)




            # exp_commands.append(f'echo && echo [Running] {working_dir} && echo '
            #                     f'&& cd {os.path.relpath(working_dir, PATH_TO_RESULT)} && bash run_simulation.sh && rm results/*.out && cd "$base_dir" '
            #                     # f'&& cd ../'
            #                     f'&& pwd'
            #                     f'&& python -OO /home/benja/Documents/facu/tesis/project/comparation/utils/parametric_compute_metrics.py . net{net} {os.path.join(*working_dir.split("/")[2:])} {NUM_OF_REPS} '
            #                     f'&& bash /home/experiment/rucopvsdss/utils/delete_results.sh . net{net}/copies={copies} {f"IRUCoPn-{copies}"}'
            #                     f'&& cd "$base_dir"'
            #                     )

    # with open(os.path.join(PATH_TO_RESULT, 'run_experiment.sh'), 'w') as f:
    #     f.write('#!/bin/bash \n')
    #     f.write(' && \n'.join(exp_commands))


def generate_comparation_script(routing_algorithms):
    with open(os.path.join(PATH_TO_RESULT, 'run_experiment.sh'), 'w') as f:
        f.write('#!/bin/bash \nbase_dir="$PWD"\n')
        for routig_algorithm in routing_algorithms:
            for net in NET_RNG:
                for copies in COPIES_RNG:
                    path_to_sim = os.path.join(f'net{net}', f'copies={copies}', f"{routig_algorithm}")
                    f.write(f'echo [Running] {path_to_sim}'
                            f'&& cd {path_to_sim} && bash run_simulation.sh && rm results/*.out && cd "$base_dir" '
                            f'&& cd ../'
                            f'&& pwd'
                            f'&& python -OO /home/benja/Documents/facu/tesis/project/comparation/utils/parametric_compute_metrics.py results net{net} copies={copies}/{routig_algorithm} {NUM_OF_REPS} '
                            f'&& bash /home/benja/Documents/facu/tesis/project/comparation/utils/delete_results.sh results net{net}/copies={copies} {routig_algorithm}'
                            f'&& cd "$base_dir" \n'
                            )


# generate_random_networks()
# rucop()
# morucop()
# generate_comparation_script(['MORUCOP', 'IRUCoPn'])