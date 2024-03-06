'''
BAD RESULTS
'''
from brufn.network import SoftState, Net
from brufn.brufspark import BRUFSpark
from typing import List, Dict
import os
import simplejson as json
from pyspark import SparkContext, SparkConf
import time

class IBRUFNFunctionGenerator:

    def __init__(self, net:Net, target:int, copies:int, working_dir:str, probability_range:List[float]=[], path_to_load_bruf:str=None, report_folder=None):
        self.net = net
        self._target = target
        self._working_dir = working_dir
        self._num_of_nodes = net.num_of_nodes
        self._num_of_ts = net.num_of_ts
        self.copies = copies
        self._failure_probability_rng = probability_range
        self._bruf_list = {}
        self.path_to_load_bruf = path_to_load_bruf
        for c in range(1, copies + 1):
            bruf_local_path = os.path.join(working_dir, f'BRUF-{c}/states_files', f'to-{target}'); os.makedirs(bruf_local_path, exist_ok=True);
            self._bruf_list[c] = BRUFSpark(net, [n for n in range(self._num_of_nodes) if n != target], target, c, probability_range, bruf_local_path)
        self.should_be_states = dict((c, dict((ts, {}) for ts in range(self._num_of_ts))) for c in range(1, copies+1))
        self.should_be_states_not_preset = 0
        report_folder = os.path.join(working_dir, f'IRUCoPn-{copies}') if report_folder is None else os.path.join(working_dir, report_folder)
        self.report_file = open(os.path.join(report_folder, f'IBRUFNFunctionGenerator-report-{copies}copies.txt'), 'w')

    def print_to_report(self, msg):
        self.report_file.write(msg + '\n')

    def __generate(self, copies):
        #print("[Warning] Remember NOT to generate BRUF model using transitive closure") # I don't think it will be necessary
        function = dict((ts, {}) for ts in range(self._num_of_ts))
        for ts in range(self._num_of_ts, -1, -1):
            start_time = time.time()
            for node in range(self._num_of_nodes):
                key = f"{node+1}:{copies}"
                print('[Generating] from: %d - ts %d'%(node, ts))
                #routing_decision = self.getRoutingDecision(node, ts, copies)
                routing_decision = self.getRoutingDecision(self._gen_list0_except_n(node, copies), node, ts)
                if len(routing_decision) > 0:
                    function[ts][key] = routing_decision
            print(f'[Info] IBRUFNFunctionGenerator.__generate_rng: Solve ts {ts} takes {time.time() - start_time} seconds.')
            self.print_to_report(f'[Info] IBRUFNFunctionGenerator.__generate_rng: Solve ts {ts} takes {time.time() - start_time} seconds.')

        return function

    def __generate_rng(self, copies):
        #print("[Warning] Remember NOT to generate BRUF model using transitive closure") # I don't think it will be necessary

        function = dict((str(pf), dict((ts, {}) for ts in range(self._num_of_ts))) for pf in self._failure_probability_rng)
        for ts in range(self._num_of_ts, -1, -1):
            start_time = time.time()
            for node in range(self._num_of_nodes):
                key = f"{node+1}:{copies}"
                print('[Generating] from: %d - ts %d'%(node, ts))
                routing_decision = self.getRoutingDecisionRng(self._gen_list0_except_n(node, copies), node, ts, function)
                for pf in self._failure_probability_rng:
                    if len(routing_decision[str(pf)]) > 0:
                        function[str(pf)][ts][key] = routing_decision[str(pf)]
            print(f'[Info] IBRUFNFunctionGenerator.__generate_rng: Solve ts {ts} takes {time.time() - start_time} seconds.')
            self.print_to_report(f'[Info] IBRUFNFunctionGenerator.__generate_rng: Solve ts {ts} takes {time.time() - start_time} seconds.')

        return function

    def generate(self):
        res = {}
        for copies in range(1, self.copies + 1):
            if self._failure_probability_rng == []:
                res[copies] = self.__generate(copies)
            else:
                if self.path_to_load_bruf is not None:
                    loaded_bruf_copies = {}
                    try:
                        for pf in self._failure_probability_rng:
                            file_path = os.path.join(self.path_to_load_bruf, f'IRUCoPn-{copies}', 'routing_files', f'pf={pf:.2f}', f'todtnsim-{self._target}-{copies}-{pf:.2f}.json')
                            with open(file_path, 'r') as f:
                                loaded_bruf_copies[str(pf)] = dict((int(k), v) for k, v in json.load(f).items())

                        res[copies] = loaded_bruf_copies
                        continue
                    except FileNotFoundError:
                        print("[Warning] File {file_path} not found so It will compute BRUF-{copies}")

                res[copies] = self.__generate_rng(copies)

        return res


    def getRoutingDecisionRng(self, state, node, ts, function) -> Dict[str, List[int]]:
        copies = sum(state)
        assert copies <= self.copies, "getRoutingDecisionRng: More copies than the availables"
        id = SoftState.get_identifier(state, ts)
        c_ids = dict((str(pf), []) for pf in self._failure_probability_rng)
        #print('[Info] Looking for State %s - %d (id: %d)' % (state, ts, id))
        try:
            loaded_state = self._bruf_list[copies].get_state_by_id_and_ts2(id, ts)
        except KeyError as e:
            print('[Info] State %s - %d (id: %d) was not found' % (state, ts, id))
            return c_ids

        routing_decisions = dict((str(pf), loaded_state[f'best_t_pf={pf}']) for pf in self._failure_probability_rng)
        for pf in self._failure_probability_rng:
            is_next_present = False
            for r_decision in routing_decisions[str(pf)] if routing_decisions[str(pf)] is not None else []:
                if r_decision['name'] != 'next':
                    copies_decision = r_decision['copies']
                    c_ids[str(pf)].append({'copies': copies_decision, 'route': r_decision['contact_ids']})
                else:
                    is_next_present = True

            if not is_next_present and 0 < sum(rd['copies'] for rd in c_ids[str(pf)]) < state[node]:
                # If we won't make any routing decision fr the remaning copies then send it to some other node.
                # It won't make any difference in the delivery ratio but it should make that any copies get stored in
                # bag data structure. (Just for DTNSim)
                c_ids[str(pf)][0]['copies'] += state[node] - sum(rd['copies'] for rd in c_ids[str(pf)])

        return c_ids

    def getRoutingDecision(self, state, node, ts, pf=-1) -> List[int]:
        copies = sum(state)
        assert copies <= self.copies, "getRoutingDecision: More copies than the availables"
        id = SoftState.get_identifier(state, ts)
        #print('[Info] Looking for State %s - %d (id: %d)' % (state, ts, id))
        try:
            loaded_state = self._bruf_list[copies].get_state_by_id_and_ts2(id, ts)
        except KeyError as e:
            print('[Info] State %s - %d (id: %d) was not found' % (state, ts, id))
            return []

        routing_decisions = loaded_state[f'best_t_pf={pf}']

        is_next_present = False
        c_ids = []
        for r_decision in routing_decisions if routing_decisions is not None else []:
            if r_decision['name'] != 'next':
                copies_decision = r_decision['copies']
                c_ids.append({'copies': copies_decision, 'route': r_decision['contact_ids']})
            else:
                is_next_present = True
        if not is_next_present and 0 < sum(rd['copies'] for rd in c_ids) < state[node]:
            # If we won't make any routing decision fr the remaning copies then send it to some other node.
            # It won't make any difference in the delivery ratio but it should make that any copies get stored in
            # bag data structure. (Just for DTNSim)
            c_ids[0]['copies'] += state[node] - sum(rd['copies'] for rd in c_ids)

        return c_ids

    def _gen_list0_except_n(self, n: int, copies:int):
        n += 1
        return [0] * (n - 1) + [copies] + [0] * (self._num_of_nodes - n)

'''
# import os
# from pprint import pprint
# working_dir = '/home/fraverta/development/BRUF-WithCopies19/examples/25-9-19/results2'; os.makedirs(working_dir, exist_ok=True)
# S=0; A=1; B=2; C=3; E=4; D=5;
#
# net = Net.get_net_from_file('/home/fraverta/development/BRUF-WithCopies19/examples/25-9-19/net.py', contact_pf_required=True)
# pprint(IBRUFNFunctionGenerator(net, D, 2, working_dir, 0.5).generate())

import os
from pprint import pprint
from pyspark import SparkContext, SparkConf

working_dir = '/home/fraverta/development/BRUF-WithCopies19/examples/21-2-19/'
copies = 2
net = Net.get_net_from_file('/home/fraverta/development/BRUF-WithCopies19/examples/21-2-19/10NetICC/net0/net.py', contact_pf_required=False)
# conf = SparkConf().setAppName("BRUF-Spark")
# conf = (conf.setMaster('local[1]')
#         .set('spark.executor.memory', '3G')
#         .set('spark.driver.memory', '16G')
#         .set('spark.driver.maxResultSize', '12G'))
# sc = SparkContext(conf=conf)
os.makedirs(os.path.join(working_dir, 'pf=0.50'), exist_ok=True)
for c in range(1,copies+1):
    wd_copies = os.path.join(working_dir, f'BRUF-{c}')
    os.makedirs(wd_copies, exist_ok=True)
    # brufn = BRUFSpark(net, [0], 7, c, [0.5], wd_copies,debug=True)
    # brufn.compute_bruf(sc)
func = IBRUFNFunctionGenerator(net, 7, copies, working_dir, 0.5).generate()
for c in range(1, copies + 1):
    with open(os.path.join(working_dir, f'pf=0.50/todtnsim-0-7-{c}-0.50.json'), 'w') as f:
        f.write(json.dumps(func[c]))


working_dir = '/home/fraverta/development/BRUF-WithCopies19/examples/tests_helper_ibrufn/test1'
copies = 2
net = Net.get_net_from_file('/home/fraverta/development/BRUF-WithCopies19/examples/tests_helper_ibrufn/test1/test_next.py', contact_pf_required=False)
# conf = SparkConf().setAppName("BRUF-Spark")
# conf = (conf.setMaster('local[1]')
#         .set('spark.executor.memory', '3G')
#         .set('spark.driver.memory', '16G')
#         .set('spark.driver.maxResultSize', '12G'))
# sc = SparkContext(conf=conf)
# os.makedirs(os.path.join(working_dir, 'pf=0.50'), exist_ok=True)
# for c in range(1, copies+1):
#     wd_copies = os.path.join(working_dir, f'BRUF-{c}')
#     os.makedirs(wd_copies, exist_ok=True)
#     brufn = BRUFSpark(net, [0], 3, c, [0.5], wd_copies,debug=True)
#     brufn.compute_bruf(sc)
#     brufn.generate_mc_to_dtnsim_all_sources_all_pf(wd_copies, partially_save_ts_number=1)

# func = IBRUFNFunctionGenerator(net, 3, copies, working_dir, 0.5).generate()
# for c in range(1, copies+1):
#     with open(os.path.join(working_dir, f'pf=0.50/0-3-{c}.json'), 'w') as f:
#         f.write(json.dumps(func[c]))
#pprint(IBRUFNFunctionGenerator(net, 3, copies, working_dir, 0.5).generate())
'''
