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

def generate_random_networks():
    '''
    Generate a network with full duplex connection and at most 1 bidireccional contact per node.
    The network is obtained from one of the 10 random networks used in ICC
    '''
    for net in NET_RNG:
        path = os.path.join(PATH_TO_RESULT, f'net{net}'); os.makedirs(path, exist_ok=True)
        path_to_net = os.path.join(path, f'net-{net}-seed={SEED}.py')
        dtnsim_cp_path = os.path.join(path, f'0.2_{net}_seed={SEED}_reflexive.dtnsim')
        if not os.path.isfile(path_to_net):
            path = os.path.join(PATH_TO_RESULT, f'net{net}')
            net_obj = Net.get_net_from_file('10NetICC/net0/net.py', contact_pf_required=False)
            contacts_by_ts = dict((ts, []) for ts in range(net_obj.num_of_ts))
            print("Number of contacts: ", len(net_obj.contacts))
            for c in net_obj.contacts:
                contacts_by_ts[c.ts].append(copy(c))
                c_simetric = Contact(c.to, c.from_, c.ts, pf=c.pf, begin_time=c.begin_time, end_time=c.end_time)
                if c_simetric not in net_obj.contacts:
                    print(f"add {c_simetric}")
                    contacts_by_ts[c.ts].append(c_simetric)

            contacts = []
            for ts in range(net_obj.num_of_ts):
                nodes = list(range(net_obj.num_of_nodes))
                while len(nodes) > 0:
                    node = nodes.pop(0)
                    snd_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.from_], key=lambda c: c.to)
                    rcv_contacts = sorted([c for c in contacts_by_ts[ts] if node == c.to], key=lambda c: c.from_)
                    # According to cp2modest.py all contacts must be full-duplex (reflexive) and only one full-duplex communication is allowed
                    if len(snd_contacts) == len(rcv_contacts):
                        if len(snd_contacts) > 0:
                            for i in random.sample(range(len(snd_contacts)), len(snd_contacts)):
                                if snd_contacts[i].to in nodes:
                                    contacts.append(snd_contacts[i])
                                    contacts.append(rcv_contacts[i])
                                    nodes.remove(snd_contacts[i].to)
                                    assert snd_contacts[i].to == rcv_contacts[i].from_, "must add a simetric contact"
                                    break
                    else:
                        assert False, f"node {node} has unidirectional contact in slot {ts}"

            sorted(contacts, key=lambda x: (x.ts, x.from_, x.to))
            net_obj = Net(net_obj.num_of_nodes, contacts)
            head, tail = os.path.split(path_to_net)
            net_obj.print_to_file(head, file_name=tail)
        else:
            net_obj = Net.get_net_from_file(path_to_net, contact_pf_required=False)

        net_obj.print_dtnsim_cp_to_file(TS_DURATION_IN_SECONDS, 100, dtnsim_cp_path)

def rucop():
    # exp_commands = ['base_dir="$PWD"']
    for net in NET_RNG:
        net_path = os.path.join(PATH_TO_RESULT, f'net{net}', f'net-{net}-seed={SEED}.py')
        net_obj = Net.get_net_from_file(net_path, contact_pf_required=False)
        cp_path = os.path.join(PATH_TO_RESULT, f'net{net}', f'0.2_{net}_seed={SEED}_reflexive.dtnsim')
        for copies in COPIES_RNG:
            for target in TARGETS:
                working_dir = os.path.join(PATH_TO_RESULT, f'net{net}', f'copies={copies}', f"BRUF-{copies}","states_files", f"to-{target}")
                os.makedirs(working_dir, exist_ok=True)
                conf = SparkConf().setAppName("BRUF-Spark")
                conf = (conf.setMaster('local[2]')
                        .set('spark.executor.memory', '2G')
                        .set('spark.driver.memory', '4G')
                        .set('spark.driver.maxResultSize', '8G'))
                sc = SparkContext(conf=conf)
                bruf = BRUFSpark(net_obj, [x for x in range(8) if x != target], target, copies,
                                    [x/100. for x in range(0,110,10)], working_dir)

                bruf.compute_bruf(sc)
                sc.stop()
            print("aca")
            #IBRUF
            #Generate link to BRUF-x with x<copies bc it is required to compute IBRUF that BRUF-x be all in the same directory
            working_dir = os.path.join(PATH_TO_RESULT, f'net{net}', f'copies={copies}', f'IRUCoPn')
            os.makedirs(working_dir, exist_ok=True)
            for c in range(1, copies):
                link_source = os.path.join("..", f'copies={c}', f'BRUF-{c}')
                link_target = os.path.join(PATH_TO_RESULT, f'net{net}', f'copies={copies}', f'BRUF-{c}')
                if os.path.islink(link_target):
                    os.unlink(link_target)
                os.symlink(link_source, link_target)

            for target in TARGETS:
                routing_files_path = os.path.join(working_dir, 'routing_files')
                ibruf = IBRUFNFunctionGenerator(net_obj, target, copies, working_dir[:working_dir.rindex('/')], [x/100. for x in range(0,110,10)])
                func = ibruf.generate()
                for c in range(1, copies+1):
                    for pf in [i / 100 for i in range(0, 110, 10)]:
                        pf_dir = os.path.join(routing_files_path, f'pf={pf:.2f}');
                        os.makedirs(pf_dir, exist_ok=True)
                        try:
                            print_str_to_file(json.dumps(func[c][str(pf)]),
                                              os.path.join(pf_dir, f'todtnsim-{target}-{c}-{pf:.2f}.json'))
                        except BaseException as e:
                            print(f"[Exception] {e}")

            ini_path = os.path.join(working_dir, 'run.ini')
            frouting_path = 'routing_files/'
            generate_omnet_ini_file(net_obj.num_of_nodes, TRAFIC, f'IRUCoPn-{copies}', ini_path,
                                    os.path.relpath(cp_path, working_dir), frouting_path=frouting_path,
                                    ts_duration=TS_DURATION_IN_SECONDS, repeats=NUM_OF_REPS)

            generate_bash_script('run.ini', os.path.join(working_dir, 'run_simulation.sh'), DTNSIM_PATH)
            # exp_commands.append(f'echo && echo [Running] {working_dir} && echo '
            #                     f'&& cd {os.path.relpath(working_dir, PATH_TO_RESULT)} && bash run_simulation.sh && rm results/*.out && cd "$base_dir" '
            #                     # f'&& cd ../'
            #                     f'&& pwd'
            #                     f'&& python -OO /home/benja/Documents/facu/tesis/project/random_comparation/utils/parametric_compute_metrics.py . net{net} {os.path.join(*working_dir.split("/")[2:])} {NUM_OF_REPS} '
            #                     f'&& bash /home/experiment/rucopvsdss/utils/delete_results.sh . net{net}/copies={copies} {f"IRUCoPn-{copies}"}'
            #                     f'&& cd "$base_dir"'
            #                     )

    # with open(os.path.join(PATH_TO_RESULT, 'run_experiment.sh'), 'w') as f:
    #     f.write('#!/bin/bash \n')
    #     f.write(' && \n'.join(exp_commands))


def morucop():
    for n in NET_RNG:
        path_to_net = f"./results/net{n}/"
        path_to_cp = path_to_net + f'0.2_{n}_seed={SEED}_reflexive.dtnsim'
        network = net.Network.from_contact_plan(path_to_cp, ts_duration=TS_DURATION_IN_SECONDS)
        for copies in COPIES_RNG:
            path_to_morucop_folder = path_to_net + f'copies={copies}/MORUCOP/'
            for pf in [i / 100 for i in range(0, 110, 10)]:
                network.set_pf(pf)
                print(network.contacts[0])
                network.run_multiobjective_derivation(1, max_copies=copies)
                network.export_rute_table(TARGETS, path=path_to_morucop_folder+ 'routing_files', pf=pf)
            ini_file = path_to_morucop_folder + 'run.ini'
            generate_omnet_ini_file(network.node_number, TRAFIC, f'IRUCoPn-{copies}',
                                    ini_file, f'../../0.2_{n}_seed={SEED}_reflexive.dtnsim', frouting_path='routing_files/',
                                    ts_duration=TS_DURATION_IN_SECONDS, repeats=NUM_OF_REPS)
            generate_bash_script('run.ini', path_to_morucop_folder + 'run_simulation.sh', DTNSIM_PATH)


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
                            f'&& python -OO /home/benja/Documents/facu/tesis/project/random_comparation/utils/parametric_compute_metrics.py results net{net} copies={copies}/{routig_algorithm} {NUM_OF_REPS} '
                            f'&& bash /home/benja/Documents/facu/tesis/project/random_comparation/utils/delete_results.sh results net{net}/copies={copies} {routig_algorithm}'
                            f'&& cd "$base_dir" \n'
                            )

# generate_random_networks()
# rucop()
# morucop()
generate_comparation_script(['MORUCOP', 'IRUCoPn'])