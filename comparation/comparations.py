import os
from settings import *
from irucop import rucop, irucop
from morucop import morucop
import time
import ipdb

def exec_with_time(fun, *args):
    time_start = time.perf_counter()
    fun(*args)
    return time.perf_counter() - time_start

def get_net_path(network_name):
    return os.path.join(PATH_TO_RESULT, f'networks/{network_name}')

def RRN_comparation():
    params = [(1,7200,10800),  #1h
                (1,10800,14400),  #1h
                (1, 7200, 14400),  # 2h
                (7,21600,28800),  #2h
                (7, 0, 10800),  # 3h
                (15,21600,32400),  #3h
                ]
    irucop_time = 0.
    morucop_time = 0.
    for source, starting_time, end_time in params:
        duration_in_hours = (end_time - starting_time) // 3600
        starting_hour = starting_time // 3600
        startt = starting_hour * 3600
        endt = (starting_hour + duration_in_hours) * 3600
        net_path = get_net_path(f'rrn_start_t:{startt},end_t:{endt}')
        cp_path = os.path.join(net_path, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
        sources = [F_RENAME_DTNSIM[gt] - 1 for gt in GROUND_TARGETS_EIDS]
        target = F_RENAME_DTNSIM[GS_EID] - 1
        probabilities_rng = [x / 100. for x in range(0, 110, 10)]

        # RUCoP
        for copies in COPIES_RNG:
            irucop_time += exec_with_time(
                rucop,net_path, copies, sources, [target], probabilities_rng, cp_path=cp_path)
    ####################################################################################
    sims_commands = []
    for source, starting_time, end_time in params:
        target = 38
        duration_in_hours = (end_time - starting_time) // 3600
        starting_hour = starting_time // 3600
        startt = starting_hour * 3600
        endt = (starting_hour + duration_in_hours) * 3600
        net_path = get_net_path(f'rrn_start_t:{startt},end_t:{endt}')
        cp_path = os.path.join(net_path, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
        dtnsim_cp_path = os.path.join(net_path, f'RRN_A_with_ISL_seed={SEED}_reflexive.dtnsim')
        traffic = {source + 1: [target + 1]}

        f_output_name = f'run_{startt},{endt},IRUCOP.sh'
        sims_commands.append("bash " + f_output_name)
        # IRUCoP
        for copies in COPIES_RNG:
            irucop_time += exec_with_time(
                irucop,net_path, dtnsim_cp_path, 60, traffic, [target],
                    copies, f_output_name, RRN_NUM_OF_REPS_OMNET, PF_RNG, cp_path)
        # MORUCOP
        f_output_name = f'run_{startt},{endt},MORUCOP.sh'
        morucop_time += exec_with_time(
            morucop,net_path, dtnsim_cp_path, 60, traffic, [target], COPIES_RNG,
                PF_RNG, f_output_name, RRN_NUM_OF_REPS_OMNET)
        sims_commands.append("bash " + f_output_name)


    with open(os.path.join(PATH_TO_RESULT,"run_sims.sh"), 'w') as f:
        f.write("&&\n".join(sims_commands))
    print("IRUCOP Time: ", irucop_time)
    print("MORUCOP Time: ", morucop_time)

def random_comparation():
    sims_commands = []
    irucop_time = 0.
    morucop_time = 0.
    for net in NET_RNG:
        net_path = get_net_path(f'net{net}')
        cp_path = os.path.join(net_path, f'net-{net}-seed={SEED}.py')
        # RUCoP
        dtnsim_cp_path = os.path.join(net_path, f'0.2_{net}_seed={SEED}_reflexive.dtnsim')
        for copies in COPIES_RNG:
            for target in TARGETS:
                irucop_time += exec_with_time(
                    rucop,net_path, copies, SOURCES, [target], PF_RNG, cp_path)
        # IRUCoP
        f_output_name = f'run_net{net},IRUCOP.sh'
        sims_commands.append("bash " + f_output_name)
        for copies in COPIES_RNG:
            irucop_time += exec_with_time(
                irucop, net_path, dtnsim_cp_path, RANDOM_TS_DURATION_IN_SECONDS,
                    RANDOM_TRAFIC, TARGETS, copies, f_output_name, RANDOM_NUM_OF_REPS,
                    PF_RNG, cp_path)

        # MORUCOP
        f_output_name = f'run_net{net},MORUCOP.sh'
        sims_commands.append("bash " + f_output_name)
        morucop_time += exec_with_time(
            morucop, net_path, dtnsim_cp_path, RANDOM_TS_DURATION_IN_SECONDS,
                RANDOM_TRAFIC, TARGETS, COPIES_RNG, PF_RNG, f_output_name, RANDOM_NUM_OF_REPS)

    with open(os.path.join(PATH_TO_RESULT,"run_sims.sh"), 'w') as f:
        f.write("&&\n".join(sims_commands))
    print("IRUCOP Time: ", irucop_time)
    print("MORUCOP Time: ", morucop_time)

def morucop_case():
    sims_commands = []
    net_path = get_net_path('morucop_case')
    dtnsim_cp_path = "../use_cases/morucop_case.txt"
    traffic = {1: [5]}
    targets = [4]
    rucop(net_path, 1, [0], targets, [0.5], ts_duration=1, dtnsim_cp_path=dtnsim_cp_path)
    irucop(net_path, dtnsim_cp_path, 1, traffic, targets, 1,
        "run_morucop_case,IRUCOP.sh", 1, [0.5]) #cp_path=os.path.join(net_path, 'net.py')
    sims_commands.append("bash run_morucop_case,IRUCOP.sh")

    morucop(net_path, dtnsim_cp_path, 1, traffic, targets, [1], [0.5],
        "run_morucop_case,MORUCOP.sh", 1)
    sims_commands.append("bash run_morucop_case,MORUCOP.sh")

    with open(os.path.join(PATH_TO_RESULT,"run_sims.sh"), 'w') as f:
        f.write("&&\n".join(sims_commands))

morucop_case()
# random_comparation()
# RRN_comparation()


