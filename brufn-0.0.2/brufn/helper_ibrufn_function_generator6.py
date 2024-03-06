from brufn.network import SoftState, Net
from brufn.brufspark import BRUFSpark
from typing import List, Dict
import os
import simplejson as json
from pyspark import SparkContext, SparkConf
import time
from brufn.helper_ibrufn_function_generator3 import IBRUFNFunctionGenerator as IBRUFNFunctionGenerator_v3

class IBRUFNFunctionGenerator:

    def __init__(self, net:Net, target:int, copies:int, working_dir:str, probability_range:List[float]=[], path_to_load_ibruf:str=None, report_folder=None, path_to_load_bruf_states:List[str]=None):
        self.net = net
        self._target = target
        self._working_dir = working_dir
        self._num_of_nodes = net.num_of_nodes
        self._num_of_ts = net.num_of_ts
        self.copies = copies
        self._failure_probability_rng = probability_range
        self._bruf_list = {}
        self.path_to_load_ibruf = path_to_load_ibruf
        for c in range(1, copies + 1):
            if path_to_load_bruf_states is None:
                path_to_load_bruf = os.path.join(working_dir, f'BRUF-{c}/states_files', f'to-{target}'); os.makedirs(path_to_load_bruf, exist_ok=True);
            else:
                path_to_load_bruf = path_to_load_bruf_states[c-1]
            self._bruf_list[c] = BRUFSpark(net, [n for n in range(self._num_of_nodes) if n != target], target, c, probability_range, path_to_load_bruf)
        report_folder = os.path.join(working_dir, f'IRUCoPn') if report_folder is None else report_folder
        self.report_file = open(os.path.join(report_folder, f'IBRUFNFunctionGenerator-report-{copies}copies.txt'), 'w')

    def print_to_report(self, msg):
        self.report_file.write(msg + '\n')

    def generate(self):
        start_time = time.time()
        res = {}
        for copies in range(1, self.copies + 1):
            if self._failure_probability_rng == []:
                res[copies] = self.__generate(copies)
            else:
                if self.path_to_load_ibruf is not None:
                    loaded_bruf_copies = {}
                    try:
                        for pf in self._failure_probability_rng:
                            file_path = os.path.join(self.path_to_load_ibruf, f'IRUCoPn', 'routing_files', f'pf={pf:.2f}', f'todtnsim-{self._target}-{copies}-{pf:.2f}.json')
                            with open(file_path, 'r') as f:
                                loaded_bruf_copies[str(pf)] = dict((int(k), v) for k, v in json.load(f).items())

                        res[copies] = loaded_bruf_copies
                        continue
                    except FileNotFoundError:
                        print("[Warning] File {file_path} not found so It will compute BRUF-{copies}")

                self.print_to_report(f'[Info] IBRUFNFunctionGenerator.generate: Will compute IRUCoPn-{copies}.')
                res[copies] = self.__generate_rng(copies)

        self.print_to_report(f'[Info] IBRUFNFunctionGenerator.generate: takes {time.time() - start_time} seconds.')
        return res

    def __generate(self, copies):
        #print("[Warning] Remember NOT to generate BRUF model using transitive closure") # I don't think it will be necessary
        function = dict((ts, {}) for ts in range(self._num_of_ts + 1))
        auxiliary_function = dict((ts, dict((c, {}) for c in range(1, copies+1))) for ts in range(self._num_of_ts + 1))
        start_time = time.time()
        for ts in range(0, self._num_of_ts + 1):
            ts_start_time = time.time()
            for node in range(self._num_of_nodes):
                key = f"{node+1}:{copies}"
                print('[Generating] from: %d - ts %d'%(node, ts))
                '''
                Assuming that nodes carries all copies available in the network at ts, it should compute the 
                routing decisions it should make for each bundle of data. In order to do that, it is necessary a 
                2 steps process:
                    1) Compute the routing decisions it will make in current timestamp
                    2) Assuming that the first hop in those ts is successful, it should compute the future routing decisions.
                    why? Because this way the node will make the best routing decisions according the information it has.
                Having this 2-step process is necessary to improve memory access because we need to access states ordered by
                timestamps in order to dimisish the number of disks loads.
                '''
                present_routing_decision, future_routing_decisions = self.getRoutingDecision(SoftState.get_identifier(self._gen_list0_except_n(node, copies), ts), node, ts, copies)
                function[ts][key] = [present_routing_decision, future_routing_decisions] # future_routing_decisions[0]-> state_id,  future_routing_decisions[1]->copies
                if future_routing_decisions != -1:
                    if future_routing_decisions[0] not in auxiliary_function[ts + 1][future_routing_decisions[1]].keys():
                        auxiliary_function[ts + 1][future_routing_decisions[1]][future_routing_decisions[0]] = {node: None}
                    else:
                        auxiliary_function[ts + 1][future_routing_decisions[1]][future_routing_decisions[0]][node] = None

            for ibruf_copies in range(1, copies + 1):
                for state_id in auxiliary_function[ts][ibruf_copies].keys():
                    for node in auxiliary_function[ts][ibruf_copies][state_id].keys():
                        present_routing_decision, future_routing_decisions = self.getRoutingDecision(state_id, node, ts, ibruf_copies)
                        auxiliary_function[ts][ibruf_copies][state_id][node] = [present_routing_decision, future_routing_decisions]
                        if future_routing_decisions != -1:
                            if future_routing_decisions[0] not in auxiliary_function[ts + 1][future_routing_decisions[1]].keys():
                                auxiliary_function[ts + 1][future_routing_decisions[1]][future_routing_decisions[0]] = {node: None}
                            else:
                                auxiliary_function[ts + 1][future_routing_decisions[1]][future_routing_decisions[0]][node] = None

        function_final = dict((ts, {}) for ts in range(self._num_of_ts))
        for ts in range(self._num_of_ts, -1, -1):
            for ibruf_copies in range(1, copies + 1):
                for k in auxiliary_function[ts][ibruf_copies].keys():
                    for node, v in auxiliary_function[ts][ibruf_copies][k].items():
                        if v[1] == -1:
                            auxiliary_function[ts][ibruf_copies][k][node] = v[0]
                        else:
                            auxiliary_function[ts][ibruf_copies][k][node] = v[0] + auxiliary_function[ts+1][v[1][1]][v[1][0]][node]

            for k, v in function[ts].items():
                if v[1] == -1:
                    if len(v[0]) > 0:
                        function_final[ts][k] = v[0]
                else:
                    res = v[0] + auxiliary_function[ts+1][v[1][1]][v[1][0]][int(k.split(":")[0]) - 1]
                    if len(res)>0:
                        function_final[ts][k] = res

            print(f'[Info] IBRUFNFunctionGenerator.__generate: Solve ts {ts} takes {time.time() - ts_start_time} seconds.')
            #self.print_to_report(f'[Info] IBRUFNFunctionGenerator.__generate: Solve ts {ts} takes {time.time() - ts_start_time} seconds.')

        #self.print_to_report(f'[Info] IBRUFNFunctionGenerator.__generate: takes {time.time() - start_time} seconds.')
        return function_final

    def __generate_rng(self, copies):
        function = dict((str(pf), dict((ts, {}) for ts in range(self._num_of_ts + 1))) for pf in self._failure_probability_rng)
        auxiliary_function = dict(dict((ts, dict((c, {}) for c in range(1, copies+1))) for ts in range(self._num_of_ts + 1)))
        for ts in range(0, self._num_of_ts+1):
            for node in range(self._num_of_nodes):
                key = f"{node + 1}:{copies}"
                print('[Generating] from: %d - ts %d' % (node, ts))
                present_routing_decision, future_routing_decisions = self.getRoutingDecisionRng(SoftState.get_identifier(self._gen_list0_except_n(node, copies), ts), node, ts, copies)
                for pf in self._failure_probability_rng:
                    function[str(pf)][ts][key] = [present_routing_decision[str(pf)], future_routing_decisions[str(pf)]]
                    if future_routing_decisions[str(pf)] != -1:
                        if future_routing_decisions[str(pf)][0] not in auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]].keys():
                            auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]][future_routing_decisions[str(pf)][0]] = {node: {}}
                        else:
                            auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]][future_routing_decisions[str(pf)][0]][node] = {}

            for ibruf_copies in range(1, copies + 1):
                for state_id in auxiliary_function[ts][ibruf_copies].keys():
                    for node in auxiliary_function[ts][ibruf_copies][state_id].keys():
                        present_routing_decision, future_routing_decisions = self.getRoutingDecisionRng(state_id, node, ts, ibruf_copies)
                        for pf in self._failure_probability_rng:
                            auxiliary_function[ts][ibruf_copies][state_id][node][str(pf)] = [present_routing_decision[str(pf)], future_routing_decisions[str(pf)]]
                            if future_routing_decisions[str(pf)] != -1:
                                if future_routing_decisions[str(pf)][0] not in auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]].keys():
                                    auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]][future_routing_decisions[str(pf)][0]] = {node: {}}
                                else:
                                    auxiliary_function[ts + 1][future_routing_decisions[str(pf)][1]][future_routing_decisions[str(pf)][0]][node] = {}

        function_final = dict((str(pf), dict((ts, {}) for ts in range(self._num_of_ts))) for pf in self._failure_probability_rng)
        for ts in range(self._num_of_ts, -1, -1):
            for ibruf_copies in range(1, copies + 1):
                for k in auxiliary_function[ts][ibruf_copies].keys():
                    for node in auxiliary_function[ts][ibruf_copies][k].keys():
                        for pf, v in auxiliary_function[ts][ibruf_copies][k][node].items():
                            if v[1] == -1:
                                auxiliary_function[ts][ibruf_copies][k][node][str(pf)] = v[0]
                            else:
                                auxiliary_function[ts][ibruf_copies][k][node][str(pf)] = v[0] + auxiliary_function[ts + 1][v[1][1]][v[1][0]][node][str(pf)]

            for pf in self._failure_probability_rng:
                for k, v in function[str(pf)][ts].items():
                    if v[1] == -1:
                        if len(v[0]) > 0:
                            function_final[str(pf)][ts][k] = v[0]
                    else:
                        res = v[0] + auxiliary_function[ts + 1][v[1][1]][v[1][0]][int(k.split(":")[0])-1][str(pf)]
                        if len(res) > 0:
                            function_final[str(pf)][ts][k] = res


        return function_final

    def getRoutingDecision(self, state_id, node, ts, copies, pf=-1) -> List[int]:
        #print('[Info] Looking for State %d - ts %d' % (id, ts))
        try:
            loaded_state = self._bruf_list[copies].get_state_by_id_and_ts2(state_id, ts)
        except KeyError as e:
            print(f'[Info] State {state_id} - ts {ts} was not found')
            return [], -1

        routing_decisions = loaded_state[f'best_t_pf={pf}']
        state = loaded_state['states']

        next_known_state = [0] * self._num_of_nodes
        is_next_present = False
        c_ids = []
        for r_decision in routing_decisions if routing_decisions is not None else []:
            if r_decision['source_node'] != node + 1: # The annoying diference between nodes ids and the ids used in OMNET
                continue
            copies_decision = r_decision['copies']
            if r_decision['name'] == 'next':
                is_next_present = True
                next_known_state[node] += copies_decision
            else:
                c_id = r_decision['contact_ids'][0]
                c_ids.append({'copies': copies_decision, 'route': r_decision['contact_ids']})
                next_known_state[self.net.get_contact_by_id(c_id-1).to] += copies_decision

        if is_next_present:
            copies_next_state = sum(next_known_state)
            return c_ids, (SoftState.get_identifier(next_known_state, ts + 1), copies_next_state)
        elif 0 < sum(rd['copies'] for rd in c_ids) < state[node]:
            # If we won't make any routing decision fr the remaning copies then send it to some other node.
            # It won't make any difference in the delivery ratio but it should make that any copies get stored in
            # bag data structure. (Just for DTNSim)
            c_ids[0]['copies'] += state[node] - sum(rd['copies'] for rd in c_ids)

        return c_ids, -1

    #(self, state_id, node, ts, copies, pf=-1)
    def getRoutingDecisionRng(self, state_id, node, ts, copies) -> Dict[str, List[int]]:
        c_ids = dict((str(pf), []) for pf in self._failure_probability_rng)
        future_routing_decisions = dict((str(pf), -1) for pf in self._failure_probability_rng)
        #print('[Info] Looking for State %s - %d (id: %d)' % (state, ts, id))
        try:
            loaded_state = self._bruf_list[copies].get_state_by_id_and_ts2(state_id, ts)
        except KeyError as e:
            print(f'[Info] State {state_id} - ts {ts} was not found')
            return c_ids, future_routing_decisions

        state = loaded_state['states']
        routing_decisions = dict((str(pf), loaded_state[f'best_t_pf={pf}']) for pf in self._failure_probability_rng)
        for pf in self._failure_probability_rng:
            next_known_state = [0] * self._num_of_nodes
            is_next_present = False
            for r_decision in routing_decisions[str(pf)] if routing_decisions[str(pf)] is not None else []:
                if r_decision['source_node'] != node + 1: # The annoying diference between nodes ids and the ids used in OMNET
                    continue
                copies_decision = r_decision['copies']
                if r_decision['name'] == 'next':
                    is_next_present = True
                    next_known_state[node] += copies_decision
                else:
                    c_id = r_decision['contact_ids'][0]
                    c_ids[str(pf)].append({'copies': copies_decision, 'route': r_decision['contact_ids']})
                    next_known_state[self.net.get_contact_by_id(c_id-1).to] += copies_decision
            if is_next_present:
                copies_next_state = sum(next_known_state)
                future_routing_decisions[str(pf)] = (SoftState.get_identifier(next_known_state, ts + 1), copies_next_state)
            elif 0 < sum(rd['copies'] for rd in c_ids[str(pf)]) < state[node]:
                # If we won't make any routing decision fr the remaning copies then send it to some other node.
                # It won't make any difference in the delivery ratio but it should make that any copies get stored in
                # bag data structure. (Just for DTNSim)
                c_ids[str(pf)][0]['copies'] += state[node] - sum(rd['copies'] for rd in c_ids[str(pf)])

        return c_ids, future_routing_decisions

    def _gen_list0_except_n(self, n: int, copies:int):
        n += 1
        return [0] * (n - 1) + [copies] + [0] * (self._num_of_nodes - n)

'''
import os
from pprint import pprint
working_dir = '/home/fraverta/development/BRUF-WithCopies19/examples/11-06-20'; os.makedirs(working_dir, exist_ok=True)
S=0; A=1; B=2; C=3; E=4; D=5;
net = Net.get_net_from_file(os.path.join(working_dir,'net.py'), contact_pf_required=True)
copies = 2

conf = SparkConf().setAppName("BRUF-Spark")
conf = (conf.setMaster('local[1]')
         .set('spark.executor.memory', '3G')
         .set('spark.driver.memory', '16G')
         .set('spark.driver.maxResultSize', '12G'))
sc = SparkContext(conf=conf)
os.makedirs(working_dir, exist_ok=True)
for c in range(1,copies+1):
    wd_copies = os.path.join(working_dir, f'BRUF-{c}', 'states_files', f'to-{D}')
    os.makedirs(wd_copies, exist_ok=True)
    #brufn = BRUFSpark(net, [S], D, c, [.5], wd_copies)
    brufn = BRUFSpark(net, [S], D, c,[], wd_copies)
    brufn.compute_bruf(sc)


os.makedirs(os.path.join(working_dir, 'IRUCoPn-2'), exist_ok=True)
pprint(IBRUFNFunctionGenerator(net, D, 2, working_dir).generate())
v6 = IBRUFNFunctionGenerator(net, D, 2, working_dir).generate()
v3 = IBRUFNFunctionGenerator_v3(net, D, 2, working_dir).generate()
#v6 = IBRUFNFunctionGenerator(net, D, 2, working_dir, probability_range=[0.5]).generate()
#v3 = IBRUFNFunctionGenerator_v3(net, D, 2, working_dir, probability_range=[0.5]).generate()
pprint(v3)
print("\n"*5)
pprint(v6)
'''