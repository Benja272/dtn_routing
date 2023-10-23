import sys
sys.path.append('../')
import os
from settings import *
from experiment_generator import generate_omnet_ini_file, generate_omnetpp_script, generate_exec_script
from ruting_morucop import Network
import ipdb

def morucop(net_path, dtnsim_cp_path, ts_duration, traffic, targets, copies_rng, f_output_name):
    network = Network.from_contact_plan(dtnsim_cp_path, ts_duration=ts_duration, priorities=[0,2,1])
    for pf in PF_RNG:
        network.set_pf(pf)
        print(network.contacts[0])
        print(targets)
        network.run_multiobjective_derivation(targets, bundle_size=1, max_copies=max(copies_rng))
        for copies in copies_rng:
            working_dir = os.path.join(net_path, f'copies={copies}', 'MORUCOP')
            path_to_morucop_folder = net_path + f'/copies={copies}/MORUCOP/'
            ini_file = path_to_morucop_folder + 'run.ini'
            network.export_rute_table(targets, copies, path=path_to_morucop_folder+ 'routing_files', pf=pf)
            generate_omnet_ini_file(network.node_number, traffic, f'IRUCoPn-{copies}',
                                    ini_file, os.path.relpath(dtnsim_cp_path, working_dir), frouting_path='routing_files/',
                                    ts_duration=ts_duration, repeats=NUM_OF_REPS_OMNET)
            generate_omnetpp_script(['run.ini'], path_to_morucop_folder + 'run_simulation.sh', DTNSIM_PATH,
                                    os.path.join(net_path, f'copies={copies}', 'MORUCOP'))
            generate_exec_script(working_dir, net_path, copies, 'MORUCOP', f_output_name)

    # for n in NET_RNG:
    # path_to_net = f"./results/net{n}/"
    # path_to_cp = path_to_net + f'0.2_{n}_seed={SEED}_reflexive.dtnsim'
    # f'../../0.2_{n}_seed={SEED}_reflexive.dtnsim'