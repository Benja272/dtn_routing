from typing import List, Dict
import os
from settings import *

CGR_FA = 'cgr-fa'
CGR_MODEL350 = 'cgrModel350'
CGR_HOPS = 'cgrModel350_3'
CGR_2COPIES = 'cgrModel350_Proactive'
BRUF_1 = 'BRUF-1'
BRUF_2 = 'BRUF-2'
BRUF_3 = 'BRUF-3'
BRUF_4 = 'BRUF-4'
CGR_BRUF_POWERED = 'CGR_BRUFPowered'
SPRAY_AND_WAIT_2 = 'sprayAndWait-2'
SPRAY_AND_WAIT_3 = 'sprayAndWait-3'
SPRAY_AND_WAIT_4 = 'sprayAndWait-4'
SPRAY_AND_WAIT_5 = 'sprayAndWait-5'
SPRAY_AND_WAIT_6 = 'sprayAndWait-6'
SPRAY_AND_WAIT_7 = 'sprayAndWait-7'
SPRAY_AND_WAIT_8 = 'sprayAndWait-8'
BINARY_SPRAY_AND_WAIT_2 = 'binarySprayAndWait-2'
BINARY_SPRAY_AND_WAIT_3 = 'binarySprayAndWait-3'
BINARY_SPRAY_AND_WAIT_4 = 'binarySprayAndWait-4'
BINARY_SPRAY_AND_WAIT_5 = 'binarySprayAndWait-5'
BINARY_SPRAY_AND_WAIT_6 = 'binarySprayAndWait-6'
BINARY_SPRAY_AND_WAIT_7 = 'binarySprayAndWait-7'
BINARY_SPRAY_AND_WAIT_8 = 'binarySprayAndWait-8'
IBRUF_1 = 'IRUCoPn-1'
IBRUF_2 = 'IRUCoPn-2'
IBRUF_3 = 'IRUCoPn-3'
IBRUF_4 = 'IRUCoPn-4'
NEW_IBRUF_1 = 'NEW_IRUCoPn-1'
NEW_IBRUF_2 = 'NEW_IRUCoPn-2'
NEW_IBRUF_3 = 'NEW_IRUCoPn-3'
NEW_IBRUF_4 = 'NEW_IRUCoPn-4'

# traffic = Dict(source -> [target])
# traffic_startt = Dict(source -> Dict(target -> start_time))

def generate_omnet_traffic(traffic:Dict[int, List[int]], traffic_ttls:Dict[int, int]={}, traffic_startt: Dict[int, Dict[int, int]]={}) -> str:
    assert all(k > 0 for k in traffic.keys()) and all(k > 0 for l in traffic.values() for k in l), 'Nodes source and target must have eid > 0'

    traffic_setting = ""
    for source_eid in sorted(traffic.keys()):

        start_times = []
        for target in traffic[source_eid]:
            if source_eid in traffic_startt.keys() and target in traffic_startt[source_eid]:
                start_times.append(str(traffic_startt[source_eid][target]))
            else:
                start_times.append('0')

        traffic_setting += f'dtnsim.node[{source_eid}].app.enable=true\n'
        traffic_setting += f'dtnsim.node[{source_eid}].app.bundlesNumber="{",".join(["1"] * len(traffic[source_eid]))}"\n'
        traffic_setting += f'dtnsim.node[{source_eid}].app.start="{",".join(start_times) }"\n'
        traffic_setting += f'dtnsim.node[{source_eid}].app.destinationEid="{",".join(map(str,traffic[source_eid])) }"\n'
        traffic_setting += f'dtnsim.node[{source_eid}].app.size="{",".join(["1"] * len(traffic[source_eid]))}"\n'
        if source_eid in traffic_ttls.keys() :
            traffic_setting += f'dtnsim.node[{source_eid}].app.ttl={traffic_ttls[source_eid]}\n'
        traffic_setting += '\n'

    return traffic_setting


def generate_omnet_ini_file(nodes_number: int, traffic: Dict[int, List[int]], routing_algorithm: str, output_file: str,
                            cp_path: str, frouting_path: str = None, repeats=100, traffic_ttls: Dict[int, Dict[int, int]]={}, traffic_startts: Dict[int, Dict[int, int]]={}, ts_duration:int=-1, ts_start_times:List[int]=None):
    '''

    :return:
    '''

    fault_aware = routing_algorithm == CGR_FA
    routing_algorithm = CGR_MODEL350  if fault_aware else routing_algorithm
    if 'BRUF-' in routing_algorithm:
        if frouting_path is None:
            raise ValueError("If routing algorithm is bruf, frouting_path can't be None")
        num_of_copies = int(routing_algorithm[5:])
        routing_algorithm_ini_setting = (
                                            f'dtnsim.node[*].dtn.routing = "BRUFNCopies"\n'
                                            f'dtnsim.node[*].dtn.bundlesCopies = {num_of_copies}\n'
                                            f'dtnsim.node[*].dtn.frouting = "{frouting_path}"\n'
                                        )
    elif routing_algorithm == CGR_BRUF_POWERED:
        if frouting_path is None:
            raise ValueError("If routing algorithm is CGR_BRUFPowered, frouting_path can't be None")
        if (ts_duration > 0 and ts_start_times is not None):
            raise ValueError("If routing algorithm is CGR_BRUFPowered, ts_duration or ts_start_time must be setted but not both")

        routing_algorithm_ini_setting = f'dtnsim.node[*].dtn.routing = "CGR_BRUFPowered"\n'
        routing_algorithm_ini_setting += f'dtnsim.node[*].dtn.frouting = "{frouting_path}"\n'
        if ts_duration > 0:
            routing_algorithm_ini_setting += f'dtnsim.node[*].dtn.ts_duration = {ts_duration}\n'
        elif len(ts_start_times) > 0:
            routing_algorithm_ini_setting += f'dtnsim.node[*].dtn.ts_start_times = "{",".join([str(t) for t in ts_start_times])}"\n'
        else:
            raise ValueError("If routing algorithm is CGR_BRUFPowered, ts_duration or ts_start_time must be setted but not both")

    elif 'binarySprayAndWait' in routing_algorithm:
        num_of_copies = int(routing_algorithm[19:])
        routing_algorithm_ini_setting = f'dtnsim.node[*].dtn.routing = "binarySprayAndWait"\n'
        routing_algorithm_ini_setting += f'dtnsim.node[*].dtn.bundlesCopies = {num_of_copies}\n'
    elif 'sprayAndWait' in routing_algorithm:
        num_of_copies = int(routing_algorithm[13:])
        routing_algorithm_ini_setting = f'dtnsim.node[*].dtn.routing = "sprayAndWait"\n'
        routing_algorithm_ini_setting += f'dtnsim.node[*].dtn.bundlesCopies = {num_of_copies}\n'
    elif 'IRUCoPn' in routing_algorithm:
        if frouting_path is None:
            raise ValueError("If routing algorithm is IRUCoPn, frouting_path can't be None")
        #if (ts_duration == -1):
        #    raise ValueError("If routing algorithm is IRUCoPn, ts_duration must be setted")

        #num_of_copies = int(routing_algorithm[8:])
        num_of_copies = int(routing_algorithm[routing_algorithm.index('-')+1:])
        routing_algorithm_ini_setting = (
                                            f'dtnsim.node[*].dtn.routing = "IRUCoPn2"\n'
                                            f'dtnsim.node[*].dtn.bundlesCopies = {num_of_copies}\n'
                                            f'dtnsim.node[*].dtn.frouting = "{frouting_path}"\n'
                                            f'dtnsim.node[*].dtn.ts_duration = {ts_duration}\n'
                                        )
    else:
        routing_algorithm_ini_setting = f'dtnsim.node[*].dtn.routing = "{routing_algorithm}"\n'

    routing_algorithm_ini_setting += 'dtnsim.node[*].app.returnToSender = false'

    ini_file = f""" 
[General]
outputvectormanager-class="omnetpp::envir::SqliteOutputVectorManager"
outputscalarmanager-class="omnetpp::envir::SqliteOutputScalarManager"

[Config dtnsim]
network = dtnsim.dtnsim
dtnsim.node[*].**.result-recording-modes = -vector

repeat = {repeats}				
dtnsim.nodesNumber = {nodes_number}	

num-rngs = 1
seed-0-mt = ${{repetition}}
dtnsim.central.rng-0 = 0

dtnsim.central.contactsFile = "{cp_path}"


dtnsim.central.faultsAware = ${{faultsAware={str(fault_aware).lower()}}}
dtnsim.central.useCentrality = false
dtnsim.central.failureProbability = ${{failureProbability=0..1 step 0.1}}

{routing_algorithm_ini_setting}

{generate_omnet_traffic(traffic, traffic_startt=traffic_startts, traffic_ttls=traffic_ttls)}
"""

    # print(ini_file)
    with open(output_file, 'w') as f:
        f.write(ini_file)

# def generate_bash_script(ini_name, output_path, dtnsim_path):
#     ini_file = '#!/bin/bash\n'
#     ini_file += f'DTNSIM_PATH="{dtnsim_path}";\n'
#     ini_file += f'opp_runall -j4 $DTNSIM_PATH/dtnsim {ini_name} -n $DTNSIM_PATH -u Cmdenv -c dtnsim;\n'
#     with open(output_path, 'w') as f:
#         f.write(ini_file)


def generate_omnetpp_script(ini_names: List[str], output_path, dtnsim_path, net_path, num_of_reps,
                         parametric_compute_metrics_fpath=UTILS_PATH):
    ini_file = '#!/bin/bash\n'
    ini_file += f'DTNSIM_PATH="{dtnsim_path}";\n\n'
    for ini_name in ini_names:
        # f'run-source={source}-target={target}.ini'
        print(ini_name)
        ini_file += f'opp_runall -j2 $DTNSIM_PATH/dtnsim {ini_name} -n $DTNSIM_PATH -u Cmdenv -c dtnsim && \n'
        ini_file += f'current_dir="$PWD" && \n'
        ini_file += f'cd {parametric_compute_metrics_fpath} && \n'
        ini_file += f'python -OO parametric_compute_metrics.py "{os.path.dirname(os.path.abspath(__file__))}" "{net_path}" "" {num_of_reps} && \n'
        ini_file += f'cd "$current_dir"'

    if ini_file.endswith("&& \n\n"):
        ini_file = ini_file[:-5]

    with open(output_path, 'w') as f:
        print(f.name)
        f.write(ini_file)

def generate_exec_script(working_dir, net_path, copies, algorithm, f_output_name):
    exp_commands = ['base_dir="$PWD"']
    exp_commands.append(f'echo && echo [Running] {working_dir} && echo '
                    f'&& cd {os.path.relpath(working_dir, PATH_TO_RESULT)}'
                    f'&& bash run_simulation.sh && rm -f results/*.out && cd "$base_dir" '
                    f'&& pwd'
                    f'&& bash {UTILS_PATH}/delete_results.sh "{os.path.dirname(os.path.abspath(__file__))}" "{net_path}/copies={copies}" {f"{algorithm}"}'
                    f'&& cd "$base_dir"'
                    # f'&& bash delete_results.sh {PATH_TO_RESULT} net{net}/copies={copies} {f"IRUCoPn-{copies}"}'
                    f'&& cd "$base_dir"'
                    )

    with open(os.path.join(PATH_TO_RESULT, f_output_name), 'w') as f:
        f.write('#!/bin/bash \n')
        f.write(' && \n'.join(exp_commands))