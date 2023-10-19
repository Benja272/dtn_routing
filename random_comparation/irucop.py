import findspark
findspark.init()
import sys
sys.path.append('../brufn-0.0.2/brufn/')
sys.path.append('../')
import os
from brufn.network import Net

from brufn.brufspark import BRUFSpark
from brufn.utils import print_str_to_file
from pyspark import SparkContext, SparkConf
from experiment_generator import generate_omnet_ini_file, generate_omnetpp_script, generate_exec_script
from brufn.helper_ibrufn_function_generator6 import IBRUFNFunctionGenerator
import simplejson as json
import os.path
from brufn.utils import *
# from utils.result_analysis_utilities import *
from brufn.experiment_generator import *
from settings import *


def rucop(net_path, cp_path, copies, sources, targets, probabilities_rng):
    net = Net.get_net_from_file(cp_path, contact_pf_required=False)
    rc = json.load(open(os.path.join(net_path, 'transitive_closure.json')),
                                     object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v
                                                            for k, v in d.items()})
    try:
        for target in targets:
            working_dir = os.path.join(net_path, f'copies={copies}', f'BRUF-{copies}', f"to-{target}")
            os.makedirs(working_dir, exist_ok=True)
            conf = SparkConf().setAppName("BRUF-Spark")
            conf = (conf.setMaster('local[2]')
                    .set('spark.executor.memory', '2G')
                    .set('spark.driver.memory', '4G')
                    .set('spark.driver.maxResultSize', '8G'))
            sc = SparkContext(conf=conf)
            bruf = BRUFSpark(net, sources, target, copies, probabilities_rng, working_dir)
            bruf.compute_bruf(sc, reachability_closure=rc)
    except Exception as e:
        import pdb; pdb.set_trace()
        print(e)
    finally:
        try:
            sc.stop()
        except:
            pass

def irucop(net_path, cp_path, dtnsim_cp_path, ts_duration, traffic, targets, copies, f_output_name):
    net = Net.get_net_from_file(cp_path, contact_pf_required=False)
    # IBRUF
    # Generate link to BRUF-x with x<copies bc it is required to compute IBRUF that BRUF-x be all in the same directory
    working_dir = os.path.join(net_path, f'copies={copies}', 'IRUCoPn')
    os.makedirs(working_dir, exist_ok=True)
    for c in range(1, copies):
        link_source = os.path.join(net_path, f'copies={c}', f'BRUF-{c}')
        link_target = os.path.join(net_path, f'copies={copies}', f'BRUF-{c}')
        if os.path.islink(link_target):
            os.unlink(link_target)
        os.symlink(link_source, link_target)

    routing_files_path = os.path.join(working_dir, 'routing_files')
    path_to_load_bruf_states = [os.path.join(net_path, f'copies={c}', f'BRUF-{c}') for c in range(1, copies + 1)]
    for target in targets:
        ibruf = IBRUFNFunctionGenerator(net, target, copies, working_dir[:working_dir.rindex('/')],
                                        PF_RNG, path_to_load_bruf_states=path_to_load_bruf_states)
        func = ibruf.generate()
        for c in range(1, copies + 1):
            for pf in PF_RNG:
                pf_dir = os.path.join(routing_files_path, f'pf={pf:.2f}');
                os.makedirs(pf_dir, exist_ok=True)
                try:
                    print_str_to_file(json.dumps(func[c][str(pf)]),
                                    os.path.join(pf_dir, f'todtnsim-{target}-{c}-{pf:.2f}.json'))
                except BaseException as e:
                    print(f"[Exception] {e}")

    ini_path = os.path.join(working_dir, 'run.ini')
    generate_omnet_ini_file(net.num_of_nodes, traffic, f'IRUCoPn-{copies}', ini_path,
                            os.path.relpath(dtnsim_cp_path, working_dir), frouting_path='routing_files/',
                            ts_duration=ts_duration, repeats=NUM_OF_REPS_OMNET)

    generate_omnetpp_script(['run.ini'], os.path.join(working_dir, 'run_simulation.sh'), DTNSIM_PATH,
                         os.path.join(net_path, f'copies={copies}', 'IRUCoPn'))
    generate_exec_script(working_dir, net_path, copies, 'IRUCoPn', f_output_name)




