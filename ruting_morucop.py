
from contact_plan import ContactPlan
from collections import Counter
from typing import List
import numpy as np
import itertools
import json
import os
# import ipdb

NEXT = 0

SUCCESS = 0
FAIL = 1

T = 0
SOURCE = 1
PROBABILITY= 2
ENERGY = 3

CONTACTS_ID_INDEX = 0
SDP_INDEX = 1
ENERGY_INDEX = 2
DELAY_INDEX = 3

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
        self.delay = int(data_size / self.data_rate)

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

class Network:
    def __init__(self, contacts: List[Contact], start_time, end_time, node_number, priorities):
        self.index = 0
        self.start_time = start_time
        self.end_time = end_time
        self.slot_range = end_time - start_time+1
        self.node_number = node_number
        self.priorities = priorities
        self.contacts = contacts
        self.contacts.sort(key=lambda c: c.t_until, reverse=True)
        assert len(priorities) == 3 and 0 in priorities and 1 in priorities and 2 in priorities
        print("Network created with %d nodes, %d slots, %d contacts and priorities %s"%(node_number, self.slot_range, len(contacts), priorities))

    def set_pf(self, pf):
        for c in self.contacts:
            c.pf = pf

    @staticmethod
    def from_contact_plan(cp_path: str, pf: float = 0.5, priorities = [0,1,2]):
        contacts = []
        cp = ContactPlan.parse_contact_plan(cp_path)
        for c in cp.contacts:
            contacts.append(Contact(c.id, c.source, c.target, (c.start_t, c.end_t), pf, c.data_rate))
            # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), pf, c.data_rate))

        return Network(contacts, cp.start_time, cp.end_time, cp.node_number, priorities)

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

    def contacts_by_source(self, contacts):
        contacts_by_source = {}
        for c in contacts:
            if c.from_n-1 not in contacts_by_source.keys():
                contacts_by_source[c.from_n-1] = {}
            contacts_by_source[c.from_n-1][c.id] = c
        return contacts_by_source

    def useful_contacts_ids(self, contacts):
        contacts_keys = []
        for id in contacts.keys():
            if self.rute_table[self.t+1, contacts[id].to-1, self.target, 0, SDP_INDEX] > 0:
                contacts_keys.append(id)
        return contacts_keys

    def setup(self, case):
        counter_case = Counter(case)
        case_without_next = list(filter(lambda x: x != 0, case))
        case_without_next_or_repeats = list(set(case_without_next))
        desicion = np.zeros(4, dtype=object)
        if counter_case[NEXT] > 0:
            desicion[CONTACTS_ID_INDEX] = case_without_next + self.rute_table[self.t+1, self.source, self.target, counter_case[NEXT]-1, CONTACTS_ID_INDEX]
        else:
            desicion[CONTACTS_ID_INDEX] = case_without_next
        return desicion, counter_case, case_without_next_or_repeats


    def fail_case_info(self, case, failed, contacts_possible_states):
        energy = 0
        fail_case_prob = 1
        copies_and_prob = {}
        for id in case.keys():
            if id in failed:
                state_prob_energy = contacts_possible_states[id][FAIL]
            else:
                state_prob_energy = contacts_possible_states[id][SUCCESS]
            state_key = tuple(state_prob_energy[:2])
            if state_key not in copies_and_prob.keys():#separar caso next del resto
                copies_and_prob[state_key] = [-1, 1]
            fail_case_prob *= state_prob_energy[PROBABILITY]
            energy += state_prob_energy[ENERGY]
            copies_and_prob[state_key][0] += case[id]
            copies_and_prob[state_key][1] *= state_prob_energy[PROBABILITY]
        return copies_and_prob, fail_case_prob, energy

    def estimate_desicion(self, case, contacts_possible_states):
        contact_fail_count = 0
        delay_cases = []
        i_failed_cases = [()]
        desicion, case, case_without_next_or_repeats = self.setup(case)
        # if self.source==0 and self.target == 3 and self.t == 0:ipdb.set_trace()

        while contact_fail_count < len(case_without_next_or_repeats)+1:
            for failed in i_failed_cases:
                sdp = 0
                copies_and_prob, fail_case_prob, energy = self.fail_case_info(case, failed, contacts_possible_states)
                for state in copies_and_prob.keys():
                    state_costs = self.rute_table[state[T], state[SOURCE], self.target, copies_and_prob[state][0]].copy()
                    if state_costs[SDP_INDEX] > 0 and copies_and_prob[state][1] > 0:
                        delay_cases.append((copies_and_prob[state][1], state_costs[DELAY_INDEX] + state[T] - self.t))
                    desicion[ENERGY_INDEX] +=  fail_case_prob * state_costs[ENERGY_INDEX]
                    sdp += state_costs[SDP_INDEX]
                if sdp > 1:
                    sdp = 1
                desicion[ENERGY_INDEX] += fail_case_prob * energy
                desicion[SDP_INDEX] += fail_case_prob * sdp
            contact_fail_count += 1
            i_failed_cases = itertools.combinations(case_without_next_or_repeats, contact_fail_count)
        desicion[DELAY_INDEX] = self.estimate_delay(delay_cases)
        return desicion

    def cases(self, next_sdp, contacts, copies):
        if next_sdp == 0:
            desicions = self.useful_contacts_ids(contacts)
            start = 0
        else:
            desicions = [0] + self.useful_contacts_ids(contacts)
            start = 1
        cases = list(itertools.combinations_with_replacement(desicions, copies+1))[start:]
        return cases

    def set_best_desicions(self, max_copies, contacts) -> None:
        contacts_by_source = self.contacts_by_source(contacts)
        for i in range(max_copies): #revisar si se envio o no en la copia anterior
            for self.source in contacts_by_source.keys():
                desicions_possible_states = {0: [[self.t+1, self.source, 1, 0]]}
                for c in contacts_by_source[self.source].values():
                    desicions_possible_states[c.id] = [[self.t+c.delay, c.to-1, 1-c.pf, 1], [self.t+c.delay, self.source, c.pf, 1]]
                for self.target in range(self.node_number):
                    if self.target == self.source: continue
                    best_desicion = self.rute_table[self.t][self.source][self.target][i].copy()
                    next_sdp = best_desicion[SDP_INDEX]
                    if i > 0:
                        desicion_with_less_copies = self.rute_table[self.t][self.source][self.target][i-1].copy()
                        if self.is_worst(best_desicion[1:], desicion_with_less_copies[1:]):
                            best_desicion = desicion_with_less_copies
                    cases = self.cases(next_sdp, contacts_by_source[self.source], i)
                    for case in cases:
                        desicion = self.estimate_desicion(case, desicions_possible_states)
                        if desicion[SDP_INDEX] > 0 and self.is_worst(best_desicion[1:], desicion[1:]):
                            best_desicion = desicion
                    self.rute_table[self.t][self.source][self.target][i] = best_desicion



    def set_delays(self, bundle_size):
        for contact in self.contacts:
            contact.set_delay(bundle_size)

    def run_multiobjective_derivation(self, bundle_size=1, max_copies = 1):
        # self.start_time = 81000
        self.rute_table = np.zeros((self.slot_range, self.node_number, self.node_number, max_copies, 4), dtype=object)
        for node in range(self.node_number):
            self.rute_table[self.end_time, node, node, :] = [[0], 1, 0, 0]
        self.set_delays(bundle_size)
        for self.t in range(self.end_time-1, self.start_time -1, -1):
            for node in range(self.node_number):
                self.rute_table[self.t, node]= self.rute_table[self.t+1][node].copy()
                self.rute_table[self.t, node, :node, :, DELAY_INDEX] += 1
                self.rute_table[self.t, node, node+1:, :, DELAY_INDEX] += 1

            contacts = self.contacts_in_slot(self.t)
            self.set_best_desicions(max_copies, contacts)


    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", *d)

    def create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def dump_rute_table(self, folder, rute_dict, targets, copies_str, pf_str):
        for target in targets:
            with open(folder + "/todtnsim-" + str(target) + "-" + copies_str + "-" + pf_str + ".json", "w") as file:
                json.dump(rute_dict[target], file)

    def routes(self, rute_dict, source, target, t, copies, copies_str):
        if source == target: return
        send_to = []
        str_source = str(source+1)
        if self.rute_table[t][source][target][0][SDP_INDEX] > 0:
            key = str_source + ":" + copies_str
            routes = Counter(self.rute_table[t][source][target][copies][CONTACTS_ID_INDEX])
            for to in routes.keys():
                send_to.append({'copies': routes[to], 'route': [to]})
            rute_dict[target][str(t)][key] = send_to

    def export_rute_table(self, targets, path="", pf=0.5):
        copies = len(self.rute_table[0][0][0])
        pf_str = f'{pf:.2f}'
        rute_dict = {}
        targets = [t-1 for t in targets]
        for i in range(copies):
            copies_str = str(i+1)
            for target in targets:
                rute_dict[target] = {}
                for t in range(self.start_time, self.end_time):
                    rute_dict[target][str(t)] = {}
                    for source in range(self.node_number):
                        self.routes(rute_dict, source, target, t, i, copies_str)
            folder = path + "/pf="+ pf_str
            self.create_folder(folder)
            self.dump_rute_table(folder, rute_dict, targets, copies_str, pf_str)



# init_value = np.empty((), dtype=Decision_np)
#         init_value[()]= (0, 0, slot_range, slot_range)
#         self.rute_table = np.full((slot_range, self.node_number, self.node_number, max_copies), init_value, dtype=Decision_np)