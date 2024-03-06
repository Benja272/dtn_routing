from brufn.network import SoftState, Net
from brufn.brufspark import BRUFSpark
from typing import List
import os
import simplejson as json
from pyspark import SparkContext, SparkConf

class IBRUFNFunctionGenerator:

    def __init__(self, net:Net, target:int, copies:int, working_dir:str, failure_probability:float = None):
        self.net = net
        self._target = target
        self._working_dir = working_dir
        self._num_of_nodes = net.num_of_nodes
        self._num_of_ts = net.num_of_ts
        self.copies = copies
        self._failure_probability = failure_probability
        self._bruf_list = {}
        for c in range(1, copies + 1):
            bruf_local_path = os.path.join(working_dir, f'BRUF-{c}/states_files', f'to-{target}'); os.makedirs(bruf_local_path, exist_ok=True);
            self._bruf_list[c] = BRUFSpark(net, [n for n in range(self._num_of_nodes) if n != target], target, c, [failure_probability], bruf_local_path)
        self.should_be_states = dict((c, dict((ts, {}) for ts in range(self._num_of_ts))) for c in range(1, copies+1))
        self.should_be_states_not_preset = 0


    def __generate(self, copies):
        print("[Warning] Remember NOT to generate BRUF model using transitive closure")

        function = dict((ts, {}) for ts in range(self._num_of_ts))
        for ts in range(self._num_of_ts):
            for node in range(self._num_of_nodes):
                key = f"{node+1}:{copies}"
                print('[Generating] from: %d - ts %d'%(node, ts))
                #routing_decision = self.getRoutingDecision(node, ts, copies)
                routing_decision = self.getRoutingDecision(self._gen_list0_except_n(node, copies), node, ts)
                if len(routing_decision) > 0:
                    function[ts][key] = routing_decision

        return function

    def generate(self):
        res = {}
        for copies in range(1, self.copies + 1):
            res[copies] = self.__generate(copies)
        for copies in range(1, self.copies + 1):
            for ts in range(self._num_of_ts):
                #Here I may replace other routing decisions for the last I have stored in should_be_states
                res[copies][ts].update(self.should_be_states[copies][ts])
                #Here I respect the decisions taken by BRUF-copies
                #res[copies][ts].update(dict((k, v) for k,v in self.should_be_states[copies][ts].items() if k not in res[copies][ts].keys()))
                #self.should_be_states_not_preset += sum(1 for k,v in self.should_be_states[copies][ts].items() if k not in res[copies][ts].keys())
                #print(f"With {copies} copies, we found {sum(1 for k,v in self.should_be_states[copies][ts] .items() if k in res[copies][ts].keys() and self.should_be_states[copies][ts][k] != res[copies][ts][k])} diferences")
        print(f"Number of states that should be present but not: {self.should_be_states_not_preset}")
        return res

    # def getRoutingDecision(self, node, ts, copies):
    #     copies_decision = self._gen_list0_except_n(node, copies)
    #     id = SoftState.get_identifier(copies_decision, ts)
    #     print('[Info] Looking for State %s - %d (id: %d)' % (copies_decision, ts, id))
    #     try:
    #         state = self._bruf_list[copies].get_state_by_id(id)
    #     except KeyError as e:
    #         print('[Info] State %s - %d (id: %d) was not found' % (copies_decision, ts, id))
    #         return []
    #
    #     if self._failure_probability is None:
    #         routing_decisions = state['best_t_pf=-1']
    #     else:
    #         routing_decisions = state[f'best_t_pf={self._failure_probability}']
    #
    #     c_ids = []
    #     is_next_present = False
    #     for r_decision in routing_decisions if routing_decisions is not None else []:
    #         assert r_decision['source_node'] == node + 1 # The annoying diference between nodes ids and the ids used in OMNET
    #         copies_decision = r_decision['copies']
    #         c_ids.extend(r_decision['contact_ids'][0:1] * copies_decision)
    #         if len(r_decision['contact_ids']) > 1:
    #             for c_id in r_decision['contact_ids'][1:]:
    #                 contact_source = self.net.get_contact_by_id(c_id-1).from_ + 1
    #                 self.should_be_states[copies_decision][ts][contact_source] = c_id





            #elif r_decision['name'] == 'next':
                # Tengo que seguir explorando al cadena de markov para ver que hace con la copia que le queda
                # Voy a tomar la decision de consultar a la cadena de markov que tiene esas copias que hacer a partir del siguiente ts
                #c_ids.extend(self.getRoutingDecision(node, ts+1, copies_decision))

        return c_ids

    def getRoutingDecision(self, state, node, ts) -> List[int]:
        copies = sum(state)
        id = SoftState.get_identifier(state, ts)
        print('[Info] Looking for State %s - %d (id: %d)' % (state, ts, id))
        try:
            loaded_state = self._bruf_list[copies].get_state_by_id(id)
        except KeyError as e:
            print('[Info] State %s - %d (id: %d) was not found' % (state, ts, id))
            return []

        if self._failure_probability is None:
            routing_decisions = loaded_state['best_t_pf=-1']
        else:
            routing_decisions = loaded_state[f'best_t_pf={self._failure_probability}']

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
                c_ids.extend([c_id] * copies_decision)
                next_known_state[self.net.get_contact_by_id(c_id-1).to] += copies_decision
                # To assure that a route belonging to a timestamp happen
                if len(r_decision['contact_ids']) > 1:
                     for c_id in r_decision['contact_ids'][1:]:
                         contact_source = self.net.get_contact_by_id(c_id-1).from_ + 1
                         self.should_be_states[copies_decision][ts][f'{contact_source}:{copies_decision}'] = [c_id] * copies_decision

        if is_next_present:
            return c_ids + self.getRoutingDecision(next_known_state, node, ts+1)
        elif 0<len(c_ids)<state[node]:
            # If we won't make any routing decision for the remaning copies then send it to some other node.
            # It won't make any difference in the delivery ratio but it should make that any copies get stored in
            # bag data structure. (Just for DTNSim)
            return c_ids + [c_ids[0]] * (state[node] - len(c_ids))
        else:
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
