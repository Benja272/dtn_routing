import itertools
from brufn.network import  Transition, SoftState, Route, SoftNextRule, Rule, bounded_iterator
from typing import List, Set, Dict, Tuple
import os
import gc
from collections import OrderedDict
import simplejson as json
from brufn.utils import print_str_to_file
import time
import os
import psutil
import statistics
import pandas as pd
from copy import copy
from datetime import datetime
from itertools import groupby
from functools import reduce
from operator import itemgetter
#from memory_profiler import profile


def get_process_current_memory_consumption():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss # in bytes

import timeit
class CodeTimer:
    def __init__(self, f_logger, name=None):
        self.name = " '"  + name + "'" if name else ''
        self.f_logger = f_logger

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        took = (timeit.default_timer() - self.start)
        memory_consumption = get_process_current_memory_consumption()
        report = []
        report.append(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]')
        report.append(str(took) + ' seconds')
        report.append(str(memory_consumption) + ' bytes')
        self.f_logger(','.join(report))


class BRUFSpark:
    '''
    Memory efficient BRUF

    Given a net, solve the problem of computing the Best Routing Under Uncertainties
    given n copies for probabilities varying in a given range
    '''

    def __init__(self, net, source:List[int], target:int, num_of_copies, probabilities_rng, working_dir:str, debug=False):
        self.net = net

        self.source = tuple(source)
        self.target = target
        self.num_of_copies = num_of_copies

        self.probabilities_rng = probabilities_rng[:]
        self.working_dir = working_dir;

        #self._transitions = []

        #statistical information
        self._transitions_number = {}
        self.__memory_consumption = {}
        self._states_number = {}
        self._time_consumption = {}

        self.__debug = debug
        self.__f_perf_metric = os.path.join(self.working_dir, 'log_performance.txt');
        # When a level is loaded from disk it is stored in this variable. The first value indicates which timestamp
        # the second is a list of loaded dicts which has information about the states dumped to disk.
        self.loaded_level = (-1, None)


    def add_to_perf_report(self, msg):
        with open(self.__f_perf_metric, 'a') as f:
            f.write(msg + '\n')

    @property
    def transitions_number(self):
        return sum(self._transitions_number.values())

    @property
    def states_number(self):
        return sum(self._states_number.values())

    def compute_previous_transitions(self, bmap_broad, to_state: SoftState):
        carrier_nodes = to_state.get_carrier_nodes()
        simple_paths_by_carrier = {}
        for carrier in carrier_nodes:
            simple_paths_by_carrier[carrier]: List[Route] = []
            for other in [x for x in range(to_state.num_of_nodes) if carrier != x and self.target != x]:
                simple_paths_by_carrier[carrier].extend(self.net.compute_routes(other, carrier, to_state.ts - 1))

        rules_generators = [bounded_iterator(len(simple_paths_by_carrier[c]) + 1, to_state.num_of_carrying_copies(c)) for c in carrier_nodes]

        previous_states = []
        for indexes in itertools.product(*rules_generators):
            applied_rule = [Rule(copies, simple_paths_by_carrier[c][sp]) if sp < len(simple_paths_by_carrier[c]) else SoftNextRule(c, copies)
                            for i,c in enumerate(carrier_nodes) for sp, copies in indexes[i]]

            previous_state = tuple(to_state.gen_previous_state(applied_rule))
            if to_state.ts - 1 > 0 or (any(previous_state[source] == self.num_of_copies for source in self.source)):
                t = Transition(SoftState(previous_state, to_state.ts - 1), to_state, applied_rule)
                t.compute_probability(bmap_broad)
                t = t.to_SoftTransition()
                previous_states.append((previous_state, {'-1': t}))
                self._current_level_transition_number += 1
        return previous_states

    def compute_previous_transitions_rng(self, bmap_broad, to_state: SoftState):
        carrier_nodes = to_state.get_carrier_nodes()
        simple_paths_by_carrier = {}
        for carrier in carrier_nodes:
            simple_paths_by_carrier[carrier]: List[Route] = []
            for other in [x for x in range(to_state.num_of_nodes) if carrier != x and self.target != x]:
                simple_paths_by_carrier[carrier].extend(self.net.compute_routes(other, carrier, to_state.ts - 1))

        rules_generators = [bounded_iterator(len(simple_paths_by_carrier[c]) + 1, to_state.num_of_carrying_copies(c)) for c in carrier_nodes]

        previous_states = []
        probabilities_rng = self.probabilities_rng if len(self.probabilities_rng) > 0 else [-1]
        for indexes in itertools.product(*rules_generators):
            applied_rule = [Rule(copies, simple_paths_by_carrier[c][sp]) if sp < len(simple_paths_by_carrier[c]) else SoftNextRule(c, copies)
                            for i,c in enumerate(carrier_nodes) for sp, copies in indexes[i]]

            previous_state = tuple(to_state.gen_previous_state(applied_rule))
            if to_state.ts - 1 > 0 or (any(previous_state[source] == self.num_of_copies for source in self.source)):
                t = Transition(SoftState(previous_state, to_state.ts - 1), to_state, applied_rule)
                t.compute_probability_rng(bmap_broad, probabilities_rng)
                t = t.to_SoftTransition()
                previous_states.append((previous_state, dict((str(k), t) for k in probabilities_rng)))
                self._current_level_transition_number += 1
        return previous_states


    def compute_previous_transitions_rng_rc(self, bmap_broad, reachability_closure, to_state: SoftState):
        carrier_nodes = to_state.get_carrier_nodes()
        simple_paths_by_carrier = {}
        for carrier in carrier_nodes:
            simple_paths_by_carrier[carrier]: List[Route] = []
            for other in [x for x in range(to_state.num_of_nodes) if carrier != x and self.target != x]:
                simple_paths_by_carrier[carrier].extend(self.net.compute_routes(other, carrier, to_state.ts - 1))

        rules_generators = [bounded_iterator(len(simple_paths_by_carrier[c]) + 1, to_state.num_of_carrying_copies(c)) for c in carrier_nodes]

        previous_states = []
        probabilities_rng = self.probabilities_rng if len(self.probabilities_rng) > 0 else [-1]
        for indexes in itertools.product(*rules_generators):
            applied_rule = [Rule(copies, simple_paths_by_carrier[c][sp]) if sp < len(simple_paths_by_carrier[c]) else SoftNextRule(c, copies)
                            for i,c in enumerate(carrier_nodes) for sp, copies in indexes[i]]

            previous_state = tuple(to_state.gen_previous_state(applied_rule))
            if any(all(carrier in reachability_closure[source][to_state.ts - 2] for carrier in [i for i,c in enumerate(previous_state) if c>0]) for source in self.source) \
                and (to_state.ts - 1 > 0 or (any(previous_state[source] == self.num_of_copies for source in self.source))):
                t = Transition(SoftState(previous_state, to_state.ts - 1), to_state, applied_rule)
                t.compute_probability_rng(bmap_broad, probabilities_rng)
                t = t.to_SoftTransition()
                previous_states.append((previous_state, dict((str(k), t) for k in probabilities_rng)))
                self._current_level_transition_number += 1
        return previous_states

    def get_best_transition(self, t1, t2):
        t1_sdp = t1['-1'].get_probability()
        t2_sdp = t2['-1'].get_probability()
        t1_cost = t1['-1'].get_cost()
        t2_cost = t2['-1'].get_cost()

        if t1_sdp > t2_sdp or (t1_sdp == t2_sdp and t1_cost < t2_cost):
           return t1
        else:
           return t2

    def get_best_transition_rng(self, t1, t2):
        res = {}
        for pr in self.probabilities_rng:
            pr = str(pr)
            t1_sdp = t1[pr].get_probability(pf=pr)
            t2_sdp = t2[pr].get_probability(pf=pr)
            t1_cost = t1[pr].get_cost(pf=pr)
            t2_cost = t2[pr].get_cost(pf=pr)

            if t1_sdp > t2_sdp or (t1_sdp == t2_sdp and t1_cost < t2_cost):
                res[pr] = t1[pr].to_SoftTransition()
            else:
                res[pr] = t2[pr].to_SoftTransition()

        return res

    def generate_final_states(self) -> Dict[int, SoftState]:
        final_states = {}
        if len(self.probabilities_rng) > 0:
            max_success_pr = dict((str(pr), 1.) for pr in self.probabilities_rng)
            max_success_transition =  dict((str(pr), None) for pr in self.probabilities_rng)
            max_success_transition_cost = dict((str(pr), 0) for pr in self.probabilities_rng)
        else:
            max_success_pr = {'-1': 1.}
            max_success_transition = {'-1': None}
            max_success_transition_cost = {'-1':0}

        for c in range(1, self.num_of_copies + 1):
            c_states = [SoftState(s, self.net.num_of_ts, max_success_transition=copy(max_success_transition), max_success_pr=copy(max_success_pr), max_success_transition_cost=max_success_transition_cost) for s in self.gen_final_states(self.num_of_copies - c, self.target, c)]
            for s in c_states:
                final_states[s.id] = s

        return dict(final_states)

    def load_level(self, ts) -> Dict[int, SoftState]:
        data = pd.read_csv(os.path.join(self.working_dir, f'ts={ts}.csv'), converters={'id': int}).to_dict(orient='records')
        level = {}
        for state in data:
            assert ts == state['ts']
            max_success_pr = dict((str(pf), state[f'sdp_pf={pf}']) for pf in (self.probabilities_rng if len(self.probabilities_rng) else [-1]))
            max_success_cost = dict((str(pf), state[f't_cost_pf={pf}']) for pf in (self.probabilities_rng if len(self.probabilities_rng) else [-1]))
            level[state['id']] = SoftState(eval(state['states']), ts, id=state['id'], max_success_pr=max_success_pr, max_success_transition_cost=max_success_cost)

        return level

    #@profile
    def compute_bruf(self, sc, starting_ts:int = None, num_states_per_slice=1000, reachability_closure=None):
        '''

        :param starting_ts: first time stamp to be computed
        :return:
        '''
        if reachability_closure is not None:
            sc.broadcast(reachability_closure)
        f = open(self.__f_perf_metric, 'w');  f.close()
        start_time = time.time()
        if starting_ts is None:
            starting_ts = self.net.num_of_ts

        if starting_ts == self.net.num_of_ts:
            next_level = self.generate_final_states()
        else:
            # Load previous computed timestamp
            next_level = self.load_level(starting_ts)
            nro_slices = max(1, len(next_level.keys()) / num_states_per_slice)
            rdd = sc.parallelize(next_level.values(), nro_slices)
            print(f"***It has loaded {len(next_level.keys())} states***")
            start_time_ts = time.time()
            bmap_broad = sc.broadcast(next_level)
            del next_level
            self._current_level_transition_number = sc.accumulator(0)
            next_level = self.solve_ts(starting_ts, rdd, bmap_broad)
            self._transitions_number[starting_ts] = self._current_level_transition_number.value
            elapsed_time = time.time() - start_time_ts; self._time_consumption[starting_ts]=elapsed_time
            print(f"\n***Ts {starting_ts} takes {elapsed_time}***\n")
            starting_ts -= 1

        self._states_number[starting_ts] = len(next_level)
        for level in range(starting_ts, 0, -1):
            start_time_ts = time.time()
            with CodeTimer(self.add_to_perf_report, 'compute_bruf:time_to_solve_ts'):
                #bulding level i-1
                nro_slices = max(1, len(next_level.keys()) / num_states_per_slice)
                rdd = sc.parallelize(next_level.values(), nro_slices)
                print(f"***States Number = {len(next_level.keys())}***")
                print(f"***Num of slices = {nro_slices}***")
                with CodeTimer(self.add_to_perf_report, 'compute_bruf dumping data'):
                    dicts_from_soft_states = [s.to_dict() for s in next_level.values()];
                    pd.DataFrame(dicts_from_soft_states).set_index('id').to_csv(os.path.join(self.working_dir, f'ts={level}.csv'));
                for s in next_level.values():
                    s.drop_information()
                bmap_broad = sc.broadcast(next_level)
                del next_level; gc.collect()
                self._current_level_transition_number = sc.accumulator(0)
                next_level = self.solve_ts(level, rdd, bmap_broad, reachability_closure)
                self._transitions_number[level-1] = self._current_level_transition_number.value

                self._states_number[level-1] = len(next_level)
                elapsed_time = time.time() - start_time_ts; self._time_consumption[level-1] = elapsed_time
                print(f"\n***Ts {level-1} takes {elapsed_time}***\n")

        with CodeTimer(self.add_to_perf_report, 'compute_bruf:time_to_generate_mc'):
            pd.DataFrame([s.to_dict() for s in next_level.values()]).set_index('id').to_csv(os.path.join(self.working_dir, f'ts={0}.csv'))
            print("There are %d Initial States"%(len(next_level)))
            for s in self.source:
                copies = [0] * self.net.num_of_nodes; copies[s] = self.num_of_copies
                id_initial_state = SoftState.get_identifier(copies, 0)
                if id_initial_state in next_level.keys():
                    initial_state = next_level[id_initial_state]
                    if len(self.probabilities_rng) > 0:
                        computed_probabilities_rng = [(pf, initial_state.get_probability(pf=pf)) for pf in self.probabilities_rng]
                        print("SDP from:%d - to:%d : %s" %(s, self.target, computed_probabilities_rng))
                        print_str_to_file(str(computed_probabilities_rng), os.path.join(self.working_dir, 'f-from:%d-to:%d.txt' %(s,self.target)))
                    else:
                        print("SDP from:%d - to:%d : %s" %(s, self.target, initial_state.get_probability()))
                        print_str_to_file(str(initial_state.get_probability()), os.path.join(self.working_dir, 'f-from:%d-to:%d.txt' %(s,self.target)))
                else:
                    print("[WARNING] There is not initial state for traffic from %d to %d"%(s, self.target))

        elapsed_time = time.time() - start_time
        self.print_resources_report(elapsed_time)

    def solve_ts(self, ts:int, rdd:Dict[int, SoftState], bmap_broad, reachability_closure) -> Dict[int, Transition]:
        if len(self.probabilities_rng) == 0 :
            rdd = rdd.flatMap(lambda s: self.compute_previous_transitions(bmap_broad, s))
            rdd = rdd.reduceByKey(lambda t1, t2: self.get_best_transition(t1, t2))
        else:
            if reachability_closure is None:
                rdd = rdd.flatMap(lambda s: self.compute_previous_transitions_rng(bmap_broad, s))
            else:
                rdd = rdd.flatMap(lambda s: self.compute_previous_transitions_rng_rc(bmap_broad, reachability_closure, s))

            rdd = rdd.reduceByKey(lambda t1, t2: self.get_best_transition_rng(t1, t2))

        self.__memory_consumption[ts] = get_process_current_memory_consumption()
        previous_level = rdd.collect()
        bmap_broad.destroy()
        return dict((s.id, s) for s in [SoftState(s, ts-1, max_success_transition=t) for s, t in previous_level])


    def get_state_by_id_and_ts(self, id: int, ts:int, pf:float) -> dict:
        if self.loaded_level[0] != ts:
            df = self.loaded_level = (ts, pd.read_csv(os.path.join(self.working_dir, f'ts={ts}.csv'), converters={'id':int}).set_index('id'))

        state = self.loaded_level[1].loc[id].to_dict()
        assert ts == state['ts']

        state[f't_changes_pf={pf}'] = None if pd.isna(state[f't_changes_pf={pf}']) else eval(state[f't_changes_pf={pf}'])
        state[f'best_t_pf={pf}'] =  None if pd.isna(state[f'best_t_pf={pf}']) else eval(state[f'best_t_pf={pf}'])
        state[f't_changes_proba_pf={pf}'] = None if pd.isna(state[f't_changes_proba_pf={pf}']) else eval(state[f't_changes_proba_pf={pf}'])

        state['states'] = eval(state['states'])
        state['id'] = id
        state['ts'] = ts
        return state

    def get_state_by_id_and_ts2(self, id: int, ts:int) -> dict:
        if self.loaded_level[0] != ts:
            df =  self.loaded_level = (ts, pd.read_csv(os.path.join(self.working_dir, f'ts={ts}.csv'), converters={'id':str}).set_index('id'))

        state = self.loaded_level[1].loc[str(id)].to_dict()
        assert ts == state['ts']

        for pf in self.probabilities_rng if len(self.probabilities_rng) > 0 else [-1]:
            state[f't_changes_pf={pf}'] = None if pd.isna(state[f't_changes_pf={pf}']) else eval(state[f't_changes_pf={pf}'])
            state[f'best_t_pf={pf}'] =  None if pd.isna(state[f'best_t_pf={pf}']) else eval(state[f'best_t_pf={pf}'])
            state[f't_changes_proba_pf={pf}'] = None if pd.isna(state[f't_changes_proba_pf={pf}']) else eval(state[f't_changes_proba_pf={pf}'])

        state['states'] = tuple(eval(state['states']))
        state['id'] = id
        state['ts'] = ts
        return state

    def get_state_by_id(self, id: int) -> dict:
        ts = 1
        while ts * (self.num_of_copies + 1)**self.net.num_of_nodes <= id:
           ts+=1
        ts= ts-1
        if self.loaded_level[0] != ts:
            df = self.loaded_level = (ts, pd.read_csv(os.path.join(self.working_dir, f'ts={ts}.csv'), converters={'id':int}).set_index('id'))

        state = self.loaded_level[1].loc[id].to_dict()
        assert ts == state['ts']

        for pf in self.probabilities_rng if len(self.probabilities_rng) > 0 else [-1]:
            state[f't_changes_pf={pf}'] = None if pd.isna(state[f't_changes_pf={pf}']) else eval(state[f't_changes_pf={pf}'])
            state[f'best_t_pf={pf}'] =  None if pd.isna(state[f'best_t_pf={pf}']) else eval(state[f'best_t_pf={pf}'])
            state[f't_changes_proba_pf={pf}'] = None if pd.isna(state[f't_changes_proba_pf={pf}']) else eval(state[f't_changes_proba_pf={pf}'])

        state['states'] = tuple(eval(state['states']))
        state['id'] = id
        state['ts'] = ts
        return state

    def gen_final_states(self, copies, target_node, target_copies, prefix=None):
        '''
        Given a number of copies that will be send, it generates all possible final states,
        in which target_node gets target_copies of the traffic.

        :param copies: Number of copies that will be send apart from those gotten for the target node
        :param target_node: Identifier of target node
        :param target_copies: An int, which tells the number of copies that will arrive to target_node
        :param prefix: It is an internal parameter. So it must be call without setting this parameter.
        :return:
        '''
        if prefix is None:
            prefix = []
        if len(prefix) == target_node:
            prefix += [target_copies]
            return self.gen_final_states(copies, target_node, target_copies, prefix=prefix)
        elif len(prefix) == self.net.num_of_nodes - 1:
            return [prefix + [copies]]
        elif len(prefix) + 2 == self.net.num_of_nodes and target_node == self.net.num_of_nodes - 1:
            #Case target_node is the one with highest id
            return [prefix + [copies, target_copies]]
        else:
            result = []
            for i in range(copies + 1):
                new_prefix = prefix + [i]
                result += self.gen_final_states(copies - i, target_node, target_copies, prefix=new_prefix)

            return result

    def generate_mc_to_dtnsim_all_sources_all_pf(self, output_folder, partially_save_ts_number=0):
        print(f"generate_mc_to_dtnsim_all_sources_all_pf")
        start_time = time.time()
        probabilities_rng = self.probabilities_rng if len(self.probabilities_rng) > 0 else [-1]
        os.makedirs(output_folder, exist_ok=True)
        number_slices=0; ts_to_save = [self.net.num_of_ts + 1] if partially_save_ts_number == 0 else sorted([x for x in range(partially_save_ts_number, self.net.num_of_ts)] + [self.net.num_of_ts + 1])

        to_visit = dict((source, dict((str(pf), []) for pf in probabilities_rng)) for source in self.source)
        for source in self.source:
            root = SoftState.get_identifier([0 if i != source else self.num_of_copies for i in range(self.net.num_of_nodes)], 0)
            for pf in probabilities_rng:
                to_visit[source][str(pf)] = [(root, 0)]  # (id,ts)

        dtnsim_states = {}
        for source in self.source:
            dtnsim_states[source] = {}
            for pf in probabilities_rng:
                pf = str(pf)
                first_dtnsim_state = OrderedDict()
                first_dtnsim_state['id_python'] = -1
                first_dtnsim_state['ts'] = -1
                first_dtnsim_state['copies_by_node'] = [0] * self.net.num_of_nodes
                first_dtnsim_state['actions'] = []
                first_dtnsim_state['children'] = [SoftState.get_identifier([0 if i != source else self.num_of_copies for i in range(self.net.num_of_nodes)], 0)]  # Then it is solved to int which means the position of state in list
                first_dtnsim_state['solved_copies'] = 0  # next copies
                first_dtnsim_state['id_dtnsim'] = 0
                dtnsim_states[source][pf] = [first_dtnsim_state]

        mapping_stateid_to_dtnsimid = dict((source, dict((str(pf), {}) for pf in probabilities_rng)) for source in self.source) # state id is the SoftState id whereas dtnsim id is the position of the state in the json array
        mapping_stateid_to_num_saved_states = dict((source, dict((str(pf), 0) for pf in probabilities_rng)) for source in self.source) # Keep the number of dumped states in order to generate the appropiate dtnsim id for following states
        for current_ts in range(self.net.num_of_ts + 2):
            # It iterates 1 time stamp beyond the end to be sure of saving all states into the routing file
            # Notice that current ts means "At the start of ts"
            start_t = time.time()
            for source in self.source:
                for pf in probabilities_rng:
                    pf = str(pf)
                    visited = []
                    print(f'generate_mc_to_dtnsim: Source: {source} - pf: {pf} - Ts: {current_ts}')
                    while len(to_visit[source][pf]) > 0 and to_visit[source][pf][0][1] <= current_ts:
                        current: SoftState = self.get_state_by_id_and_ts(*(to_visit[source][pf].pop(0)), pf)
                        #print(f'generate_mc_to_dtnsim: Source: {source} - pf: {pf} - State: {current["id"]} - Ts: {current["ts"]}')
                        # Add state to states list
                        dtnsim_state = OrderedDict()
                        dtnsim_state['id_python'] = current['id']
                        dtnsim_state['ts'] = current['ts']
                        dtnsim_state['copies_by_node'] = current['states']
                        dtnsim_state['actions'] = []
                        dtnsim_state['children'] = []  # Then it is solved to int which means the position of state in list
                        dtnsim_state['solved_copies'] = 0  # next copies

                        mapping_stateid_to_dtnsimid[source][pf][current['id']] = len(dtnsim_states[source][pf]) + mapping_stateid_to_num_saved_states[source][pf]
                        dtnsim_states[source][pf].append(dtnsim_state)

                        if current['states'][self.target] == 0:
                            # If it isn't a final state, explore children
                            if current[f'best_t_pf={pf}'] is not None:
                                dtnsim_state['actions'] = current[f'best_t_pf={pf}']
                                dtnsim_state['solved_copies'] = self.num_of_copies - sum([action['copies'] for action in dtnsim_state['actions'] if action['name'] != 'next'])
                                for change in current[f't_changes_pf={pf}']:
                                    dtnsim_state['children'].append(change)
                                    if change not in visited:
                                        to_visit[source][pf].append((change, current['ts'] + 1))
                                        visited.append(change)
            print(f"\nTS {current_ts} takes {time.time() - start_t} seconds\n")
            if current_ts in ts_to_save:
                #Save partial routing file to free memory usage
                for source in self.source:
                    for pf in probabilities_rng:
                        to_dump = []
                        while len(dtnsim_states[source][str(pf)]) > 0 and dtnsim_states[source][str(pf)][0]['ts'] < current_ts:
                            dtnsim_state = dtnsim_states[source][str(pf)].pop(0)
                            dtnsim_state['children'] = [mapping_stateid_to_dtnsimid[source][str(pf)][state_id] for state_id in dtnsim_state['children']]
                            to_dump.append(dtnsim_state)

                        print_str_to_file(json.dumps(to_dump), os.path.join(output_folder, f'mc-dtnsim-from:%d-to:%d-%0.2f-part:{number_slices}.json' % (source, self.target, pf)))
                        mapping_stateid_to_num_saved_states[source][str(pf)] += len(to_dump)

                del mapping_stateid_to_dtnsimid; gc.collect()
                mapping_stateid_to_dtnsimid = dict((source, dict((str(pf), {}) for pf in probabilities_rng)) for source in self.source)
                number_slices+=1

        #Agregate slices into one routing file
        for source in self.source:
            for pf in probabilities_rng:
                output_routing_pf = os.path.join(output_folder, 'pf=%.2f'%pf)
                os.makedirs(output_routing_pf, exist_ok=True)
                l = []
                for slice in range(number_slices):
                    with open(os.path.join(output_folder, f'mc-dtnsim-from:%d-to:%d-%0.2f-part:{slice}.json' % (source, self.target, pf)),'r') as f:
                        l.extend(json.load(f))

                print_str_to_file(json.dumps(l), os.path.join(output_routing_pf, f'todtnsim-%d-%d-%0.2f.json' % (source, self.target, pf)))
                # Remove temporal created slices.
                for slice in range(number_slices):
                    os.remove(os.path.join(output_folder, f'mc-dtnsim-from:%d-to:%d-%0.2f-part:{slice}.json' % (source, self.target, pf)))

        elapsed_time = time.time() - start_time
        with open(os.path.join(self.working_dir, 'generate_mc_to_dtnsim_all_sources_all_pf_log.txt'), 'w') as f:
            f.write(f'It takes {elapsed_time} seconds\n')

    def print_resources_report(self, elapsed_time:int):
        with open(os.path.join(self.working_dir, 'memory_consumption_log.txt'), 'w') as f:
            f.write('ts,memory_consumption(bytes)\n')
            f.write('\n'.join([f'{k},{self.__memory_consumption[k]}' for k in sorted(self.__memory_consumption.keys())]))

        with open(os.path.join(self.working_dir, 'states_log.txt'), 'w') as f:
            f.write('ts,nro_states\n')
            f.write('\n'.join([f'{k},{self._states_number[k]}' for k in sorted(self._states_number.keys())]))

        with open(os.path.join(self.working_dir, 'transitions_log.txt'), 'w') as f:
            f.write('ts,nro_transitions\n')
            f.write('\n'.join([f'{k},{self._transitions_number[k]}' for k in sorted(self._transitions_number.keys())]))

        with open(os.path.join(self.working_dir, 'time_log.txt'), 'w') as f:
            f.write('ts,time(seconds)\n')
            f.write('\n'.join([f'{k},{self._time_consumption[k]}' for k in sorted(self._time_consumption.keys())]))

        with open(os.path.join(self.working_dir, 'resources_report.txt'),'w') as f:
            f.write(f'Compute BRUF time: {elapsed_time} seconds\n')
            f.write(f'Total States number: {self.states_number}\n')
            f.write(f'Total Transitions number: {self.transitions_number}\n')
            f.write(f'Average Memory Consumption: {statistics.mean(self.__memory_consumption.values())}\n')
            f.write(f'Max Memory Consumption: {max(self.__memory_consumption.values(), default=-1)} bytes\n')
            f.write(f'Min Memory Consumption: {min(self.__memory_consumption.values(), default=-1)} bytes\n')


    def is_routing_implementable(self, source, transitive_closure, pf=-1):
        root_id = SoftState.get_identifier(self._gen_list0_except_n(source, self.num_of_copies), 0)
        to_visit:List[Tuple[int, int]] = [(root_id, 0)] #id,ts
        reachables_states_at_ts = []
        reachables_states_at_ts_plus_1 = [] #the reachables states for pf (it depends on decisions nodes make)
        current_ts = 0
        while to_visit:
            last_ts = current_ts
            current_id, current_ts = to_visit.pop(0)
            loaded_state:dict = self.get_state_by_id_and_ts2(current_id, current_ts)

            if current_ts > last_ts:
                reachables_states_at_ts = reachables_states_at_ts_plus_1
                reachables_states_at_ts_plus_1 = []

            if loaded_state[f'sdp_pf={pf}'] == 0.:
                continue #It is not significant to consider this state for implementability (to consider case in which pf=1.)


            assert loaded_state["ts"] == current_ts
            if loaded_state['states'][self.target] > 0:
                # if it is a succesful state, then routing decisions made here are not important
                continue
            #action = loaded_state[f"best_t_pf={pf}"]

            for carrier_node, c in [(n, loaded_state['states'][n]) for n in range(self.net.num_of_nodes) if loaded_state['states'][n] > 0]:
                # Check all states at this ts in which node n has c copies and not bundle has been successfully delivered
                states_copies_to_check = filter(lambda s: s[self.target] == 0, self.gen_final_states(self.num_of_copies - c, carrier_node, c))
                states_to_check = []
                for s in states_copies_to_check:
                    if self._is_state_reachable_from_source_at_the_begining_of_ts(source, s, current_ts, transitive_closure):
                        s_id = SoftState.get_identifier(s, current_ts)
                        if s_id in reachables_states_at_ts:
                            states_to_check.append(s_id)
                        else:
                            print(f"[State {s} - {current_ts}] is not reachable for pf={pf}")
                    else:
                        print(f"[State {s} - {current_ts}] is not reachable")

                #states_to_check = [SoftState.get_identifier(s, current_ts) for s in states_copies_to_check]

                for state_ in states_to_check:
                    try:
                        loaded_state_2: dict = self.get_state_by_id_and_ts2(state_, current_ts)
                    except KeyError:
                        print(f"[State id {state_}] was not found. It must not be reachable")
                        continue

                    assert loaded_state_2["ts"] == current_ts
                    action_relevant_to_carrier = list(filter(lambda rule: rule['source_node'] - 1 == carrier_node, loaded_state[f"best_t_pf={pf}"])) #minus 1 because saved rules work with dtnsim ids
                    action2_relevant_to_carrier = list(filter(lambda rule: rule['source_node'] - 1 == carrier_node, loaded_state_2[f"best_t_pf={pf}"])) #minus 1 because saved rules work with dtnsim ids
                    if len(action_relevant_to_carrier) != len(action2_relevant_to_carrier):
                        print(f"BRUFSpark:is_routing_implementable source={source} pf={pf} not implementable")
                        print(f'\t At state {loaded_state["states"]} - {loaded_state["ts"]} the following routing decision must be made:')
                        for a in action_relevant_to_carrier:
                            print(f'\t\t {a}')
                        print(f'\t But, at state {loaded_state_2["states"]} - {loaded_state_2["ts"]} the following routing decision must be made:')
                        for a in action2_relevant_to_carrier:
                            print(f'\t\t {a}')
                        return False, loaded_state, loaded_state_2,

                    for rule in action2_relevant_to_carrier:
                        #if rule['source_node'] - 1 == carrier_node:
                        if not any(rule == r for r in action_relevant_to_carrier):
                            print(f"BRUFSpark:is_routing_implementable source={source} pf={pf} not implementable")
                            print(f'\t At state {loaded_state["states"]} - {loaded_state["ts"]} the following routing decision must be made:')
                            for a in action_relevant_to_carrier:
                                print(f'\t\t {a}')
                            print(f'\t But, at state {loaded_state_2["states"]} - {loaded_state_2["ts"]} the following routing decision must be made:')
                            for a in action2_relevant_to_carrier:
                                print(f'\t\t {a}')
                            return False, loaded_state, loaded_state_2,

            for to_state, pr in zip(loaded_state[f"t_changes_pf={pf}"], loaded_state[f"t_changes_proba_pf={pf}"]):
                if float(pr) > 0 and (to_state, current_ts+1) not in to_visit:
                    to_visit.append((to_state, current_ts+1))
                    reachables_states_at_ts_plus_1.append(to_state)


        return True, None, None


    def _gen_list0_except_n(self, n: int, copies:int):
        n += 1
        return [0] * (n - 1) + [copies] + [0] * (self.net.num_of_nodes - n)


    def _is_state_reachable_from_source_at_the_begining_of_ts(self, source, states:List[int], ts:int, reachability_closure):
        '''
        Is state reachable at the BEGINING of ts number ts if bundles start in source?
        :param states:
        :param ts:
        :param reachability_closure:
        :return:
        '''
        if ts==0:
            return states[source] == self.num_of_copies and all(states[x] == 0 for x in range(self.net.num_of_nodes) if x!=source)

        return all(any(n in reachability_closure[x][ts-1] for x in range(self.net.num_of_nodes) if x != n or x==source)
                   for n in range(self.net.num_of_nodes) if states[n] > 0)

    def mc_to_dot(self, root:int, output_file, pf=-1):
        output_file = open(output_file, 'w')
        output_file.write("digraph s { \n")
        output_file.write('size="8,5"\n')
        output_file.write( 'node [shape=box];\n')
        states = ""

        to_visit: List[int] = [(root, 0)]
        visited: Set[int] = set(to_visit)
        while len(to_visit) > 0:
            current: Dict = self.get_state_by_id_and_ts(*to_visit.pop(0), pf)
            copies = "(" + ", ".join(["n%d: %d"%(n, c) for n, c in enumerate(current['states']) if c>0]) + ")"

            if current['states'][self.target] > 0:
                states += ('%d [label="%d\\n %s - %d "] [peripheries=2];\n' % (current['id'], current['id'], copies, current['ts']))
            else:
                states += ('%d [label="%d\\n %s - %d "];\n' % (current['id'], current['id'], copies, current['ts']))
                if current[f'best_t_pf={pf}'] is not None:
                    rules = [rule['name'] for rule in current[f'best_t_pf={pf}']]
                    output_file.write( '%d -> n%d_0[arrowhead=none, label = "%s"];\n' %(current['id'], current['id'], ", ".join(rules)))
                    output_file.write( 'n%d_0[shape = point, width = 0.1, height = 0.1, label = ""];\n' % (current['id']))
                    for change_id, change_pr in zip(current[f't_changes_pf={pf}'], current[f't_changes_proba_pf={pf}']):
                        output_file.write('n%d_0 -> %d[label = "%.2f"];\n' % (current['id'], change_id, change_pr))
                        if change_id not in visited:
                            to_visit.append((change_id, current['ts'] + 1))
                            visited.add(change_id)

        output_file.write(states)
        output_file.write('}')
        output_file.close()