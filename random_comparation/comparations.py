import os
from settings import *
from irucop import rucop, irucop
from morucop import morucop
import time
import ipdb

def RRN_comparation(copies_rng):
    params = [(1,7200,10800),  #1h
                (1,10800,14400),  #1h
                (1, 7200, 14400),  # 2h
                (7,21600,28800),  #2h
                # (7, 0, 10800),  # 3h
                # (15,21600,32400),  #3h
                ]
    irucop_time = 0.
    morucop_time = 0.
    # time_start = time.perf_counter()

    # for source, starting_time, end_time in params:
    #     duration_in_hours = (end_time - starting_time) // 3600
    #     starting_hour = starting_time // 3600
    #     startt = starting_hour * 3600
    #     endt = (starting_hour + duration_in_hours) * 3600
    #     net_path = os.path.join(PATH_TO_RESULT, f'networks/rrn_start_t:{startt},end_t:{endt}')
    #     cp_path = os.path.join(net_path, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
    #     sources = [F_RENAME_DTNSIM[gt] - 1 for gt in GROUND_TARGETS_EIDS]
    #     target = F_RENAME_DTNSIM[GS_EID] - 1
    #     probabilities_rng = [x / 100. for x in range(0, 110, 10)]

    #     # RUCoP
    #     for copies in copies_rng:
    #         rucop(net_path, cp_path, copies, sources, [target], probabilities_rng)
    # irucop_time = time.perf_counter() - time_start
    ####################################################################################
    irucop_commands = []
    for source, starting_time, end_time in params:
        target = 38
        duration_in_hours = (end_time - starting_time) // 3600
        starting_hour = starting_time // 3600
        startt = starting_hour * 3600
        endt = (starting_hour + duration_in_hours) * 3600
        net_path = os.path.join(PATH_TO_RESULT, f'networks/rrn_start_t:{startt},end_t:{endt}')
        cp_path = os.path.join(net_path, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
        dtnsim_cp_path = os.path.join(net_path, f'RRN_A_with_ISL_seed={SEED}_reflexive.dtnsim')
        traffic = {source + 1: [target + 1]}

        f_output_name = f'run_{startt},{endt},IRUCOP.sh'
        irucop_commands.append("bash " + f_output_name)
        for copies in copies_rng:
            # IRUCoP
            time_start = time.perf_counter()
            # irucop(net_path, cp_path, dtnsim_cp_path, 60, traffic, [target], copies, f_output_name=f_output_name)
            irucop_time += time.perf_counter() - time_start
            # MORUCOP
        f_output_name = f'run_{startt},{endt},MORUCOP.sh'
        time_start = time.perf_counter()
        # morucop(net_path, dtnsim_cp_path, 60, traffic, [target], copies_rng, f_output_name=f_output_name)
        morucop_time += time.perf_counter() - time_start
        irucop_commands.append("bash " + f_output_name)


    with open(os.path.join(PATH_TO_RESULT,"run_sims.sh"), 'w') as f:
        f.write("&&\n".join(irucop_commands))
    print("IRUCOP Time: ", irucop_time)
    print("MORUCOP Time: ", morucop_time)

RRN_comparation([1,2])