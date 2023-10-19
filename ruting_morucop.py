
from contact_plan import ContactPlan
from collections import Counter
from typing import List, Any
from math import ceil
import numpy as np
from itertools import combinations_with_replacement, combinations, product, chain
import json
import os
import ipdb

NEXT = 0

SUCCESS = 0
FAIL = 1

T = 0
SOURCE = 1
PROBABILITY= 2
ENERGY = 3

SDP_INDEX = 0
ENERGY_INDEX = 1
DELAY_INDEX = 2

class Decision:
    @staticmethod
    def is_worse_desicion(c1, c2, priorities):
        c1, c2 = list(c1), list(c2)
        if c1[1] == 0 and c2[1] > 0: return True
        c1[1], c2[1] = -c1[1], -c2[1]
        for i in priorities:
            if (c1[i] > c2[i]):
                return True
            elif c1[i] < c2[i]:
                return False
        return False

    @staticmethod
    def estimate_sdp_energy(prob, future_desicions, energy=1):
        return np.dot(prob, (future_desicions[SDP_INDEX], (energy + future_desicions[ENERGY_INDEX])))


class Contact:
    def __init__(self, id: int, from_n: int, to_n: int, t_range: (int, int), pf: float, data_rate: int):
        self.id = id
        self.from_n: int = from_n
        self.to: int = to_n
        self.pf: float = pf
        self.data_rate: int = data_rate
        self.t_since: int = t_range[0]
        self.t_until: int = t_range[1]
        assert self.t_since < self.t_until

    def set_delay(self, data_size: int):
        self.delay = ceil(data_size / self.data_rate)

    def __str__(self):
        return 'send_to:%d_from:%d_since:%d_until:%d_pf:%f'%(self.to, self.from_n, self.t_since, self.t_until, self.pf)

    def to_dict(self) -> dict:
        d = {'from':self.from_, 'to': self.to, 'ts': self.ts}
        if self.__begin_time != -1:
            d['begin_time'] = self.__begin_time
            d['end_time'] = self.__end_time
        return d

    @staticmethod
    def useful_contacts(contacts: List['Contact'], t: int, endtime: int) -> List['Contact']:
        return [c for c in contacts if t+c.delay <= endtime]

class State:
    def __init__(self, costs: List[float], transition: List[Any]=[], contact_ids: List[int]=[]) -> None:
        self.contact_ids = contact_ids
        self.costs = costs
        self.transition = transition

    def __str__(self) -> str:
        return 'contact_ids:%s costs(SDP: %f, ENERGY: %f, DELAY: %f)'%(self.contact_ids, self.costs[0], self.costs[1], self.costs[2])

    def set(self, costs: List[float], transition: List[Any]=[], contact_ids: List[int]=[]) -> None:
        self.contact_ids = contact_ids
        self.costs = costs
        self.transition = transition


class Network:
    def __init__(self, contacts: List[Contact], start_time, end_time, node_number, priorities, ts_duration: int = 1):
        self.start_time = start_time // ts_duration
        self.end_time = end_time // ts_duration
        self.slot_range = self.end_time - self.start_time+1
        self.node_number = node_number
        self.priorities = priorities
        self.contacts = contacts
        self.contacts.sort(key=lambda c: c.t_until, reverse=True)
        self.rute_dict = {}
        assert len(priorities) == 3 and 0 in priorities and 1 in priorities and 2 in priorities
        print("Network created with %d nodes, %d slots, %d contacts and priorities %s"%(node_number, self.slot_range, len(contacts), priorities))

    def set_pf(self, pf):
        for c in self.contacts:
            c.pf = pf

    @staticmethod
    def from_contact_plan(cp_path: str, pf: float = 0.5, priorities = [0,1,2], ts_duration: int = 1):
        contacts = []
        cp = ContactPlan.parse_contact_plan(cp_path)
        for c in cp.contacts:
            if c.end_t - c.start_t >= ts_duration:
                contacts.append(Contact(c.id, c.source, c.target, (c.start_t//ts_duration, c.end_t//ts_duration), pf, c.data_rate))
                # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), pf, c.data_rate))

        return Network(contacts, cp.start_time, cp.end_time, cp.node_number, priorities, ts_duration)

    def to_dict(self) -> dict:
        return {'nodes': [n.to_dict() for n in self.nodes]}

    def is_worst(self, c1, c2):
        c1, c2 = c1.copy(), c2.copy()
        if c1[0] == 0 and c2[0] > 0: return True
        c1[0], c2[0] = -c1[0], -c2[0]
        for i in self.priorities:
            if (c1[i] > c2[i]):
                return True
            elif c1[i] < c2[i]:
                return False
        return False

    def estimate_delay(self, success_cases):
        delay = 0
        total = sum([c[0] for c in success_cases])
        for case in success_cases:
            delay += (case[0]/total) * case[1]
        return delay

    def contacts_in_slot(self, t: int) -> List[Contact]:
        contacts = []
        while self.index < len(self.contacts)-1 and self.contacts[self.index].t_since > t:
            self.index += 1
        i = self.index
        while i < len(self.contacts):
            c = self.contacts[i]
            if c.t_until <= t:
                break
            if(c.t_since <= t and t+c.delay <= self.end_time):
                contacts.append(self.contacts[i])
            i += 1
        return contacts

    def transitions_by_target(self, contacts: List[Contact]):
        contacts_by_target = {node: [(node,)] for node in range(self.node_number)}
        for c in contacts:
            contacts_by_target[c.to-1].append(c.id)
        return contacts_by_target

    def contacts_by_source(self, contacts: List[Contact]):
        contacts_by_source = {node: [(node,)] for node in range(self.node_number)}
        for c in contacts:
            contacts_by_source[c.from_n-1].append(c.id)
        return contacts_by_source

    def setup(self, case):
        counter_case = Counter(case)
        case_without_next = list(filter(lambda x: type(x) != tuple, case))
        case_without_next_or_repeats = list(set(case_without_next))
        costs = np.zeros(3, dtype=float)
        return costs, counter_case, case_without_next_or_repeats


    def fail_case_info(self, transition_counter, failed, copies):
        energy = 0
        fail_case_prob = 1
        fail_state = ()
        copies += 1
        already_in_target = 0
        for id, copies_through in transition_counter.items():
            if type(id) != tuple:
                energy += 1
                contact = self.contacts_by_id[id]
                if id in failed:
                    fail_state += (contact.from_n-1,)*copies_through
                    fail_case_prob *= contact.pf
                else:
                    fail_state += (contact.to-1,)*copies_through
                    fail_case_prob *= (1-contact.pf)
            else:
                fail_state += id*copies_through
                if id[0] == self.target:
                    already_in_target = copies_through

        fail_state = tuple(sorted(fail_state))
        delay = (copies - already_in_target)/copies
        return fail_state, fail_case_prob, energy, delay

    def less_copies_state_costs(self, fail_state, copies): #recalcular delay
        for i in range(copies-1, -1, -1):
            j = 0
            searching = True
            states_keys = list(self.states[self.t+1][self.target][i].keys())
            while searching and j < len(states_keys):
                searching = not all(id in fail_state for id in states_keys[j])
                j += 1
            if not searching:
                return self.states[self.t+1][self.target][i][states_keys[j-1]].costs

    def estimate_costs(self, transition, copies):
        contact_fail_count = 0
        i_failed_cases = [()]
        success_cases = []
        costs, transition_counter, case_without_next_or_repeats = self.setup(transition)

        while contact_fail_count < len(case_without_next_or_repeats)+1:
            for failed in i_failed_cases:
                fail_state, fail_case_prob, energy, delay = self.fail_case_info(transition_counter, failed, copies)
                fail_state_info = self.states[self.t+1][self.target][copies].get(fail_state)
                if fail_state_info is not None:
                    fail_state_costs = fail_state_info.costs
                else:
                    fail_state_costs = self.less_copies_state_costs(fail_state, copies)
                if fail_state_costs is not None:
                    costs[SDP_INDEX] += fail_case_prob * fail_state_costs[SDP_INDEX]
                    costs[ENERGY_INDEX] +=  fail_case_prob * (energy + fail_state_costs[ENERGY_INDEX])
                    success_cases.append((fail_case_prob, delay + fail_state_costs[DELAY_INDEX]))
                else:
                    costs[ENERGY_INDEX] +=  fail_case_prob * energy

            contact_fail_count += 1
            i_failed_cases = combinations(case_without_next_or_repeats, contact_fail_count)
        if costs[SDP_INDEX] > 0:
            costs[DELAY_INDEX] = self.estimate_delay(success_cases)
        return costs

    def new_state_key(self, transition):
        new_state = ()
        for id in transition:
            if type(id) != tuple:
                new_state += (self.contacts_by_id[id].from_n-1,)
            else:
                new_state += id
        new_state = tuple(sorted(new_state))
        return new_state

    def transition_contact_ids(self, transition):
        contact_ids = []
        next_transitions = 0
        for id in transition:
            if type(id) != tuple:
                contact_ids.append(id)
            else:
                node = id
                next_transitions += 1
        if next_transitions > 0:
            state = node * next_transitions
            state_info = self.states[self.t+1][self.target][next_transitions-1].get(state)
            if state_info is not None:
                contact_ids += state_info.contact_ids
        return contact_ids

    def lrucop_transitions(self, copies, contacts_by_source):
        comb_sum = list(combinationSum(copies+1))
        transitions = []
        if copies > 0:
            for comb in comb_sum:
                counter = Counter(comb)
                transitions_set = []
                for c, amount in counter.items():
                    trans = list(map(lambda s: s.transition, self.states[self.t, self.target, c-1].values()))
                    transitions_set.append(list(combinations_with_replacement(trans, amount)))
                prod = list(product(*transitions_set))
                comb_transitions = [list(chain(*t[0])) for t in prod]
                source_transitions = [list(combinations_with_replacement(contacts_by_source[source], copies+1)) for source in range(self.node_number)]
                comb_transitions += list(filter(lambda trans: trans not in comb_transitions, [list(t) for t in chain(*source_transitions)]))
                transitions += comb_transitions
        return transitions

    def rucop_transitions(self, copies, transitions_by_target):
        transitions = []
        for state_key in self.states[self.t+1][self.target][copies].keys():
            state_counter = Counter(state_key)
            to_target_transitions = []
            for target, copies_in in state_counter.items():
                to_target_transitions.append(list(combinations_with_replacement(transitions_by_target[target], copies_in)))
            prod = list(product(*to_target_transitions))
            transitions += [list(chain(*t)) for t in prod]
        return transitions


    def set_best_desicions(self, contacts_in_slot) -> None:
        transitions_by_target = self.transitions_by_target(contacts_in_slot)
        if self.max_copies > 1:
            contacts_by_source = self.contacts_by_source(contacts_in_slot)
        for copies in range(self.max_copies): #revisar si se envio o no en la copia anterior
            for self.target in range(self.node_number):
                if copies > 0:
                    transitions = self.lrucop_transitions(copies, contacts_by_source)
                else:
                    transitions = self.rucop_transitions(copies, transitions_by_target)

                for transition in transitions:
                    # ipdb.set_trace()
                    costs = self.estimate_costs(transition, copies)
                    if costs[SDP_INDEX] > 0:
                        contact_ids = self.transition_contact_ids(transition)
                        new_state_key = self.new_state_key(transition)
                        new_state_info = self.states[self.t][self.target][copies].get(new_state_key)
                        if new_state_info is None:
                            self.states[self.t][self.target][copies][new_state_key] = State(costs, transition, contact_ids)
                        elif self.is_worst(new_state_info.costs, costs):
                            new_state_info.set(costs, transition, contact_ids)

    def by_id(self):
        self.contacts_by_id = {}
        for contact in self.contacts:
            self.contacts_by_id[contact.id] = contact

    def set_delays(self, bundle_size):
        for contact in self.contacts:
            contact.set_delay(bundle_size)

    def run_multiobjective_derivation(self, bundle_size=1, max_copies = 1):
        # self.start_time = 81000
        self.index = 0
        self.max_copies = max_copies
        self.states = np.empty((self.slot_range, self.node_number, self.max_copies), dtype=object)
        for t in range(self.end_time+1):
            for node in range(self.node_number):
                for c in range(self.max_copies):
                    if t == self.end_time:
                        state = tuple([node] * (c+1))
                        self.states[self.end_time][node][c] = {state: State([1, 0, 0])}
                    else:
                        self.states[t][node][c] = {}
        self.by_id()
        self.set_delays(bundle_size)
        for self.t in range(self.end_time-1, self.start_time -1, -1):
            contacts_in_slot = self.contacts_in_slot(self.t)
            self.set_best_desicions(contacts_in_slot)


    def print_table(self):
        for c in range(self.max_copies):
            print("----------------------------------------------------------------")
            for t in range(self.end_time -1, self.start_time -1, -1):
                for target in range(self.node_number):
                    for state, state_info in self.states[t, target, c].items():
                        source = state[0]
                        if all(n == source for n in state) and source != target:
                            print("En t=",t," desde ", source +1, " hasta ", target +1, " con ", state_info)

    def dump_rute_table(self, folder, rute_dict, targets, copies_str, pf_str):
        for target in targets:
            with open(folder + "/todtnsim-" + str(target) + "-" + copies_str + "-" + pf_str + ".json", "w") as file:
                json.dump(rute_dict[target], file)

    def routes(self, rute_dict, target, t, copies, copies_str):
        for state, state_info in self.states[t, target, copies].items():
            source = state[0]
            if all(n == source for n in state) and source != target:
                str_source = str(source+1)
                key = str_source + ":" + copies_str
                routes = Counter(state_info.contact_ids)
                send_to = []
                for to in routes.keys():
                    send_to.append({'copies': routes[to], 'route': [to]})
                rute_dict[target][str(t)][key] = send_to


    def export_rute_table(self, targets, copies, path=".", pf=0.5):
        assert copies <= self.max_copies
        pf_str = f'{pf:.2f}'
        rute_dict = {}
        for i in range(copies):
            copies_str = str(i+1)
            for target in targets:
                rute_dict[target] = {}
                for t in range(self.start_time, self.end_time):
                    rute_dict[target][str(t)] = {}
                    self.routes(rute_dict, target, t, i, copies_str)
            folder = path + "/pf="+ pf_str
            create_folder(folder)
            self.dump_rute_table(folder, rute_dict, targets, copies_str, pf_str)


def f(N, acumulate, combination):
    if acumulate == N:
        yield combination
    for i in range(min(combination[-1], N-acumulate), 0, -1):
        yield from f(N, acumulate + i, combination + [i])


def combinationSum(N):
    for i in range(1,N):
        yield from f(N,i,[i])

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# init_value = np.empty((), dtype=Decision_np)
#         init_value[()]= (0, 0, slot_range, slot_range)
#         self.rute_table = np.full((slot_range, self.node_number, self.node_number, self.max_copies), init_value, dtype=Decision_np)