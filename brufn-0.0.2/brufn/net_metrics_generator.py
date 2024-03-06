from brufn.network import Net, bounded_iterator
from typing import List
import csv
from functools import reduce
import operator
import sys
import simplejson as json
import os
from brufn.utils import print_str_to_file
from itertools import chain

def get_transition_number(num_simple_path, copies):
    prefix = [[]]
    finish_prefix = 0
    for sp in range(0, num_simple_path + 1): # +1 because it considers next rule
        new_prefixs = []
        for p in prefix:
            new_prefixs.append(p[:])
            for c in range(1, copies - sum([x[1] for x in p])):
                new_prefixs.append(p + [(sp, c)])
            finish_prefix += 1
            prefix = new_prefixs

    return finish_prefix

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def pritn_to_csv(columns, data, fpath):
    try:
        with open(fpath, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    except IOError:
        print("I/O error")


class NetMetricGenerator:

    def __init__(self, net:Net, interesting_nodes:List[int], copies_range:List[int], sources, output_dir):
        self.net = net
        self.nodes = interesting_nodes
        self.copies_range = copies_range
        self.sources = sources
        self.output_dir = output_dir
        self.compute_metrics()

    def compute_metrics(self):
        node_data = []; ts_data = [];
        num_of_contacts_by_ts = dict((ts, len([c for c in self.net.contacts if c.ts==ts])) for ts in range(self.net.num_of_ts))
        simple_path_number_arriving_to_a_node_at_ts = dict((node, {}) for node in self.nodes)
        max_hop_path_arriving_to_a_node_at_ts = dict((node, {}) for node in self.nodes)
        pred_at_ts = dict((node, {}) for node in self.nodes)
        reachability_closure = dict((node, dict((ts,set([node])) for ts in range(-1, self.net.num_of_ts))) for node in self.nodes)
        transition_by_state = dict((node, {}) for node in self.nodes) #It is an indicative number not necessarly the maximum one
        non_reachable_states_count = dict((ts, {}) for ts in range(self.net.num_of_ts))
        non_reachable_states = dict((ts, {}) for ts in range(self.net.num_of_ts))
        for ts in range(self.net.num_of_ts):
            simple_path_arriving_to_a_node_at_current_ts = {}
            for node in self.nodes:
                print(f'Ts: {ts} - Node: {node}')
                s_paths_by_source_node = dict((source, self.net.compute_routes(source, node, ts)) for source in range(self.net.num_of_nodes))
                simple_path_arriving_to_a_node_at_current_ts[node] = list(chain.from_iterable(s_paths_by_source_node.values()))
                pred_at_ts[node][ts] = [s for s in range(self.net.num_of_nodes) if len(s_paths_by_source_node[s]) > 0]
                simple_path_number_arriving_to_a_node_at_ts[node][ts] = sum([len(v) for v in s_paths_by_source_node.values()]) # Chech what happend when node == source
                max_hop_path_arriving_to_a_node_at_ts[node][ts] = max(max(map(lambda sp: sp.hop_count(), s_paths_by_source_node[source]), default=0) for source in range(self.net.num_of_nodes))

                # I am not sure if i am computing the reflexo-transitivite closure properly
                for source in range(self.net.num_of_nodes):
                    # Node will be reachable from source, at the end of current ts if:
                    #  _ node was already a reachable node
                    #  _ or source has a path toward node at current ts
                    #  _ or if source has a path toward other node n arriving at the start of current ts or before, and n has a path to node at current ts
                    if node in reachability_closure[source][ts-1] \
                            or len(s_paths_by_source_node[source]) > 0 \
                            or any(len(s_paths_by_source_node[other]) > 0 for other in range(self.net.num_of_nodes) if other in reachability_closure[source][ts-1]):
                            reachability_closure[source][ts].add(node)


                row = {'ts': ts, 'contacts': num_of_contacts_by_ts[ts], 'node': node, 'pred+': pred_at_ts[node][ts], 'sp': simple_path_number_arriving_to_a_node_at_ts[node][ts]}
                transition_by_state[node][ts] = {}
                for copies in self.copies_range:
                    transition_by_state[node][ts][copies] = get_transition_number(simple_path_number_arriving_to_a_node_at_ts[node][ts], copies)
                    row[f't_{copies}copies'] = transition_by_state[node][ts][copies]
                node_data.append(row)

            for copies in self.copies_range:
                num_max_transition = 0
                num_min_transitions = sys.maxsize
                acum_num_transitions = 0 #Notice some states could be not explored by the algorithm
                explored_states = 0
                non_reachable_states_count[ts][copies] = 0
                non_reachable_states[ts][copies]=[]
                dif_contacts_upper_bound = 0
                for state in bounded_iterator(self.net.num_of_nodes, copies):
                    carriers = [i for i, c in state]; state = dict(state)
                    num_of_transitions = prod(get_transition_number(simple_path_number_arriving_to_a_node_at_ts[carrier][ts], state[carrier]) for carrier in carriers)
                    acum_num_transitions += num_of_transitions; explored_states+=1
                    if num_of_transitions > num_max_transition:
                        num_max_transition = num_of_transitions
                        state_max_transition = state
                    if num_of_transitions < num_min_transitions:
                        num_min_transitions = num_of_transitions
                        state_min_transition = state
                    if all(any(carrier not in reachability_closure[source][ts-1] for carrier in carriers) for source in self.sources):
                        # It counts the number of states which are not reachable at the BEGINNING of current ts
                        non_reachable_states_count[ts][copies] += 1
                        non_reachable_states[ts][copies].append(state)

                    state_dif_contacts_upper_bound = sum(sp.hop_count() for sp in chain.from_iterable([simple_path_arriving_to_a_node_at_current_ts[node][:state[node]] for node in carriers]))
                    if state_dif_contacts_upper_bound > dif_contacts_upper_bound:
                        dif_contacts_upper_bound = state_dif_contacts_upper_bound


                ts_data.append({'ts': ts, 'contacts':num_of_contacts_by_ts[ts], 'copies': copies,
                                'max_t': num_max_transition, 'min_t':num_min_transitions,
                                'avg_t': acum_num_transitions/explored_states, 'acum_t':acum_num_transitions,
                                'max_hop_sp': max(max_hop_path_arriving_to_a_node_at_ts[node][ts] for node in range(self.net.num_of_nodes)),
                                #'dif_contacts_upper_bound': sum(sorted((max_hop_path_arriving_to_a_node_at_ts[node][ts] for node in range(self.net.num_of_nodes)), reverse=True)[:copies]),
                                'dif_contacts_upper_bound': dif_contacts_upper_bound,
                                'state_max_t':state_max_transition,'state_min_t': state_min_transition,
                                'explored_states':explored_states,
                                'non_reachable_states_count': non_reachable_states_count[ts][copies]})


            # node_with_most_sp = max(self.nodes, key=lambda x: simple_path_number_arriving_to_a_node_at_ts[x][ts])
            # for copies in self.copies_range:
            #     transition_by_state[ts][copies] = get_transition_number(simple_path_number_arriving_to_a_node_at_ts[node_with_most_sp][ts], copies)
            #     row[f't_{copies}copies'] = transition_by_state[ts][copies]
        pritn_to_csv(['ts', 'contacts', 'node', 'pred+', 'sp'] + [f't_{c}copies' for c in self.copies_range], node_data,  os.path.join(self.output_dir,'node_data.csv'))
        pritn_to_csv(['ts', 'contacts', 'copies', 'max_t', 'min_t', 'avg_t', 'acum_t', 'max_hop_sp','dif_contacts_upper_bound', 'state_max_t', 'state_min_t', 'explored_states', 'non_reachable_states_count'], ts_data, os.path.join(self.output_dir,'ts_data.csv'))
        transitive_closure_json = json.dumps(reachability_closure, iterable_as_array=True)
        print_str_to_file(transitive_closure_json, os.path.join(self.output_dir, 'transitive_closure.json'))
        print()
        print(transitive_closure_json)
        print(json.dumps(non_reachable_states_count))
        print()
        #print(json.dumps(non_reachable_states))


    def compute_path(self):
        paths = {}
        for ts in range(self.net.num_of_ts):
            paths[ts] = {}
            for source in range(self.net.num_of_nodes):
                paths[ts][source] = {}
                for target in [n for n in range(self.net.num_of_nodes) if source != target]:
                    paths[ts][source][target] = self.net.compute_routes(source, target, ts)



#net = Net.get_net_from_file('/home/fraverta/development/BRUF-WithCopies19/examples/25-9-19/net.py', contact_pf_required=True)
#net = Net.get_net_from_file('/home/fraverta/development/BRUF-WithCopies19/examples/RRN-SLICING/exp-02-11-2019/start_t:0,end_t:21600/FROM-NET-start_t:0,end_t:21600.py', contact_pf_required=False)
#NetMetricGenerator(net, range(net.num_of_nodes), range(3,4), [x for x in range(25)]).compute_metrics()
