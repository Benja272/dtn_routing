import os
from settings import *
from irucop import rucop, irucop
from morucop import morucop
import time
import ipdb

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
    time_start = time.perf_counter()

    for source, starting_time, end_time in params:
        duration_in_hours = (end_time - starting_time) // 3600
        starting_hour = starting_time // 3600
        startt = starting_hour * 3600
        endt = (starting_hour + duration_in_hours) * 3600
        net_path = os.path.join(PATH_TO_RESULT, f'networks/rrn_start_t:{startt},end_t:{endt}')
        cp_path = os.path.join(net_path, f'cp_start_t:{startt},end_t:{endt}-seed={SEED}.py')
        sources = [F_RENAME_DTNSIM[gt] - 1 for gt in GROUND_TARGETS_EIDS]
        target = F_RENAME_DTNSIM[GS_EID] - 1
        probabilities_rng = [x / 100. for x in range(0, 110, 10)]

        # RUCoP
        for copies in COPIES_RNG:
            rucop(net_path, cp_path, copies, sources, [target], probabilities_rng)
    irucop_time = time.perf_counter() - time_start
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
        # IRUCoP
        for copies in COPIES_RNG:
            time_start = time.perf_counter()
            irucop(net_path, cp_path, dtnsim_cp_path, 60, traffic, [target], copies, f_output_name=f_output_name)
            irucop_time += time.perf_counter() - time_start
        # MORUCOP
        f_output_name = f'run_{startt},{endt},MORUCOP.sh'
        time_start = time.perf_counter()
        morucop(net_path, dtnsim_cp_path, 60, traffic, [target], COPIES_RNG, f_output_name=f_output_name)
        morucop_time += time.perf_counter() - time_start
        irucop_commands.append("bash " + f_output_name)


    with open(os.path.join(PATH_TO_RESULT,"run_sims.sh"), 'w') as f:
        f.write("&&\n".join(irucop_commands))
    print("IRUCOP Time: ", irucop_time)
    print("MORUCOP Time: ", morucop_time)

def random_comparation():
    for net in NET_RNG:
        net_path = os.path.join(PATH_TO_RESULT, f'net{net}')
        cp_path = os.path.join(net_path, f'net-{net}-seed={SEED}.py')
        # RUCoP
        dtnsim_cp_path = os.path.join(net_path, f'0.2_{net}_seed={SEED}_reflexive.dtnsim')
        for copies in COPIES_RNG:
            for target in TARGETS:
                rucop(net_path, cp_path, copies, SOURCES, [target], PF_RNG)

        # IRUCoP
        f_output_name = f'run_net{net},IRUCOP.sh'
        for copies in COPIES_RNG:
            irucop(net_path, cp_path, dtnsim_cp_path,
                RANDOM_TS_DURATION_IN_SECONDS, RANDOM_TRAFIC, TARGETS, copies, f_output_name=f_output_name)

        # MORUCOP
        f_output_name = f'run_net{net},MORUCOP.sh'
        morucop(net_path, dtnsim_cp_path,
                RANDOM_TS_DURATION_IN_SECONDS, RANDOM_TRAFIC, TARGETS, COPIES_RNG, f_output_name=f_output_name)
            #IBRUF
            #Generate link to BRUF-x with x<copies bc it is required to compute IBRUF that BRUF-x be all in the same directory
            # working_dir = os.path.join(PATH_TO_RESULT, f'net{net}', f'copies={copies}', f'IRUCoPn')
            # os.makedirs(working_dir, exist_ok=True)
            # for c in range(1, copies):
            #     link_source = os.path.join("..", f'copies={c}', f'BRUF-{c}')
            #     link_target = os.path.join(PATH_TO_RESULT, f'net{net}', f'copies={copies}', f'BRUF-{c}')
            #     if os.path.islink(link_target):
            #         os.unlink(link_target)
            #     os.symlink(link_source, link_target)

            # for target in TARGETS:
            #     routing_files_path = os.path.join(working_dir, 'routing_files')
            #     ibruf = IBRUFNFunctionGenerator(net_obj, target, copies, working_dir[:working_dir.rindex('/')], [x/100. for x in range(0,110,10)])
            #     func = ibruf.generate()
            #     for c in range(1, copies+1):
            #         for pf in [i / 100 for i in range(0, 110, 10)]:
            #             pf_dir = os.path.join(routing_files_path, f'pf={pf:.2f}');
            #             os.makedirs(pf_dir, exist_ok=True)
            #             try:
            #                 print_str_to_file(json.dumps(func[c][str(pf)]),
            #                                   os.path.join(pf_dir, f'todtnsim-{target}-{c}-{pf:.2f}.json'))
            #             except BaseException as e:
            #                 print(f"[Exception] {e}")

            # ini_path = os.path.join(working_dir, 'run.ini')
            # frouting_path = 'routing_files/'
            # generate_omnet_ini_file(net_obj.num_of_nodes, TRAFIC, f'IRUCoPn-{copies}', ini_path,
            #                         os.path.relpath(cp_path, working_dir), frouting_path=frouting_path,
            #                         ts_duration=TS_DURATION_IN_SECONDS, repeats=NUM_OF_REPS)

            # generate_bash_script('run.ini', os.path.join(working_dir, 'run_simulation.sh'), DTNSIM_PATH)

random_comparation()
RRN_comparation()

