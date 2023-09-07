
from contact_plan import ContactPlan
from collections import Counter
from typing import List
import numpy as np
import itertools
import json
import os
import ipdb

SUCCESS = 0
FAIL = 1

T_INDEX = 0
SOURCE_INDEX = 1
PROB_INDEX = 2

CONTACT_ID_INDEX = 0
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
        assert len(priorities) == 3 and 1 in priorities and 2 in priorities and 3 in priorities
        print("Network created with %d nodes, %d slots, %d contacts and priorities %s"%(node_number, self.slot_range, len(contacts), priorities))

    @staticmethod
    def from_contact_plan(cp_path: str, pf: float = 0.5, priorities = [1,2,3]):
        contacts = []
        cp = ContactPlan.parse_contact_plan(cp_path)
        for c in cp.contacts:
            contacts.append(Contact(c.id, c.source, c.target, (c.start_t, c.end_t), pf, c.data_rate))
            # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), pf, c.data_rate))

        return Network(contacts, cp.start_time, cp.end_time, cp.node_number, priorities)

    def to_dict(self) -> dict:
        return {'nodes': [n.to_dict() for n in self.nodes]}

    def cost_less(self, c1, c2):
        if c1[0] == 0 and c2[0] > 0: return True
        c1[0], c2[0] = -c1[0], -c2[0]
        for i in self.priorities:
            if (c1[i] > c2[i]):
                return True
            elif c1[i] < c2[i]:
                return False
        return False

    def coincidences_prob(self, coincidences, prob, case):
        for c in coincidences:
            if c in case:
                prob *= c[1]
            else:
                prob *= (1 - c[1])
        return prob

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


    def case_cost(self, coincidences_base, coincidences, next_t, prob, delay, energy=1):
        i = 0
        success_cases = []
        sdp_energy_sum = np.zeros(2)
        i_failed_cases = [()]
        while i < len(coincidences)+1:
            case_i = self.rute_table[next_t][self.source][self.target][coincidences_base + i].copy()
            for failed in i_failed_cases:
                coin_prob = self.coincidences_prob(coincidences, prob, failed)
                sdp_energy_cost = Decision.estimate_sdp_energy(coin_prob, case_i, energy)
                if sdp_energy_cost[0] > 0:
                    success_cases.append((coin_prob, delay + case_i[DELAY_INDEX]))
                sdp_energy_sum = np.add(sdp_energy_sum, sdp_energy_cost)
            i += 1
            i_failed_cases = itertools.combinations(coincidences, i)

        return sdp_energy_sum, success_cases




    def send_decision(self, succes_coincidences, fail_coincidences, contact, next_t):
        success_case = self.rute_table[next_t][contact.to-1][self.target][succes_coincidences].copy()
        if success_case[SDP_INDEX] > 0:
            success_sdp_energy_sum = np.array([success_case[1], success_case[2]])
            if succes_coincidences == 0:
                success_sdp_energy_sum = Decision.estimate_sdp_energy((1 - contact.pf), success_case)
            fail_sdp_energy_sum, success_cases = self.case_cost(0, fail_coincidences, next_t, contact.pf, contact.delay)
            success_cases.append((1 - contact.pf, contact.delay + success_case[DELAY_INDEX]))
            sdp_energy_sum = np.add(success_sdp_energy_sum, fail_sdp_energy_sum)
            return np.concatenate(([contact.id], sdp_energy_sum, [self.estimate_delay(success_cases)]))
        else:
            return [0, 0, 0, 0]



    def coincidences(self, sended_copies, next_t, target):
        success_coincidences = 0
        fail_coincidences = []
        for s in sended_copies.keys():
            if s[1] == next_t and s[0] == target: #usa mismo contacto
                success_coincidences += 1
            elif s[1] == next_t:
                fail_coincidences.append((s[0], sended_copies[s])) #coincidentes en el tiempo en caso de fallar
        return success_coincidences, fail_coincidences

    def next_decision(self, t, sended_copies):
        success_coincidences, fail_coincidences = self.coincidences(sended_copies, t+1, self.source + 1)
        sdp_energy_sum, success_cases = self.case_cost(success_coincidences, fail_coincidences, t+1, 1, 1, energy=0)
        return [0] + list(sdp_energy_sum) + [self.estimate_delay(success_cases)]

    def init_state(self, t, sended_copies, targets, best_desicions, copies):
        if self.source not in sended_copies.keys():
            sended_copies[self.source] = [{} for i in range(self.node_number)]
            targets[self.source] = [i for i in range(self.node_number)]
        if self.source not in best_desicions.keys():
            best_desicions[self.source] = [(self.rute_table[t][self.source][target][copies].copy(), (self.source + 1, t+1), 1) for target in range(self.node_number)]

    def update_state(self, t, sended_copies, targets, best_desicions, copies):
        for source in best_desicions.keys():
            nodes = targets[source].copy()
            for target in nodes:
                best_desicion, best_send_pair, pf = best_desicions[source][target]
                if best_desicion[SDP_INDEX] > 0:
                    sended_copies[source][target][best_send_pair] = pf
                    self.rute_table[t][source][target][copies] = best_desicion
                else:
                    targets[source].remove(target)

    def contacts_by_source(self, contacts):
        contacts_by_source = {}
        for c in contacts:
            if c.from_n not in contacts_by_source.keys():
                contacts_by_source[c.from_n-1] = {}
            contacts_by_source[c.from_n-1][c] = c
        return contacts_by_source

    def costs(self, case, contacts_possible_states):
        i = 0
        case_without_next = list(filter(lambda x: x != 0, case))
        costs = np.zeros(3)
        delay_cases = []
        i_failed_cases = [()]
        for failed in i_failed_cases:
            copies_and_probs = {}
            for id in case:
                if id in failed:
                    state_and_probability = contacts_possible_states[id][FAIL]
                else:
                    state_and_probability = contacts_possible_states[id][SUCCESS]
                if state_and_probability[:2] not in copies_and_probs.keys():
                    copies_and_probs[state_and_probability[:2]] = [1,0]
                copies_and_probs[state_and_probability[:2]][0] *= state_and_probability[PROB_INDEX]
                copies_and_probs[state_and_probability[:2]][1] += 1
            for state in copies_and_probs.keys():
                state_costs = self.rute_table[state[T_INDEX], state[SOURCE_INDEX], self.target, copies_and_probs[state][1]]
                if state_costs[SDP_INDEX] > 0:
                    delay_cases.append((copies_and_probs[state][0], state_costs[DELAY_INDEX] + state[T_INDEX] - self.t))
                costs += np.append(state_costs[1:3]* copies_and_probs[state][0], 0)
            i += 1
            i_failed_cases = itertools.combinations(case_without_next, i)
        costs[2] = self.estimate_delay(delay_cases)
        return costs

    def set_best_desicions(self, max_copies, contacts) -> None:
        contacts_by_source = self.contacts_by_source(contacts)
        for i in range(max_copies): #revisar si se envio o no en la copia anterior
            for self.source in contacts_by_source.keys():
                desicions_possible_states = {0: [self.t+1, self.source, 1]}
                for c in contacts_by_source[self.source]:
                    desicions_possible_states[c.id] = [[self.t+c.delay, c.to-1, 1-c.pf], [self.t+c.delay, self.source, c.pf]]
                for self.target in range(self.node_number):
                    best_desicion = None
                    desicions = list(contacts_by_source[self.source].keys()) + [0]
                    cases = itertools.combinations_with_replacement(desicions, i)
                    for case in cases:
                        costs = self.costs(case, desicions_possible_states)
                        if best_desicion is None or self.cost_less(costs, best_desicion):
                            best_desicion = [list(case)] +  costs.tolist()
                    self.rute_table[self.t][self.source][self.target][i] = best_desicion
            # for c in contacts:
            #     next_t = t+c.delay
            #     self.source = c.from_n - 1
            #     self.init_state(t, sended_copies, targets, best_desicions, i)
            #     for self.target in targets[self.source]:
            #         if i > 0:
            #             best_desicions[self.source][self.target] = [self.next_decision(t, sended_copies[self.source][self.target]), (self.source + 1, t+1), 1]
            #         success_coincidences, fail_coincidences = self.coincidences(sended_copies[self.source][self.target], next_t, c.to)
            #         decision = self.send_decision(success_coincidences, fail_coincidences, c, next_t)
            #         if decision[SDP_INDEX] > 0 and Decision.is_worse_desicion(best_desicions[self.source][self.target][0], decision, self.priorities):
            #             best_desicions[self.source][self.target] = [decision, (c.to, next_t), c.pf]
            # self.update_state(t, sended_copies, targets, best_desicions, i)



    def set_delays(self, bundle_size):
        for contact in self.contacts:
            contact.set_delay(bundle_size)

    def run_multiobjective_derivation(self, bundle_size=1, max_copies = 1):
        # self.start_time = 81000
        # copies_array = [[[], 0., 0., 0.] for i in range(1,max_copies+1)]
        self.rute_table = np.zeros((self.slot_range, self.node_number, self.node_number, max_copies, 4), dtype=object)
        # self.rute_table[:][:][:][:] = copies_array
        for t in range(self.start_time, self.end_time):
            for node in range(self.node_number):
                self.rute_table[self.end_time][node][node][:] = [0, 1, 0, 0]
        self.set_delays(bundle_size)
        for self.t in range(self.end_time-1, self.start_time -1, -1):
            # for node in range(self.node_number):
            #     self.rute_table[self.t][node]= self.rute_table[self.t+1][node] + np.array([0, 0, 0, 1])
            #     # self.rute_table[t][node][node][0] = [0, 1, 0, 0]
            #     self.rute_table[self.t][node][node+1:] = self.rute_table[self.t+1][node][node+1:] + np.array([0, 0, 0, 1])
            #     # self.rute_table[t][node][:,:,0] = node + 1
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

    def routes(self, rute_dict, source, target, t, copies_str):
        if source == target: return
        send_to = []
        str_source = str(source+1)
        if self.rute_table[t][source][target][0][SDP_INDEX] > 0:
            key = str_source + ":" + copies_str
            routes = Counter(map(lambda x: int(x[CONTACT_ID_INDEX]), filter(lambda x: x[SDP_INDEX] > 0, self.rute_table[t][source][target])))
            for to in routes.keys():
                send_to.append({'copies': routes[to], 'route': [to]})
            rute_dict[target][str(t)][key] = send_to

    def export_rute_table(self, targets, copies=1, pf=0.5):
        assert len(self.rute_table[0][0][0]) >= copies
        copies_str = str(copies)
        pf_str = f'{pf:.2f}'
        rute_dict = {}
        targets = [t-1 for t in targets]
        for target in targets:
            rute_dict[target] = {}
            for t in range(self.start_time, self.end_time):
                rute_dict[target][str(t)] = {}
                for source in range(self.node_number):
                    self.routes(rute_dict, source, target, t, copies_str)
        folder = "pf="+ pf_str
        self.create_folder(folder)
        self.dump_rute_table(folder, rute_dict, targets, copies_str, pf_str)



# init_value = np.empty((), dtype=Decision_np)
#         init_value[()]= (0, 0, slot_range, slot_range)
#         self.rute_table = np.full((slot_range, self.node_number, self.node_number, max_copies), init_value, dtype=Decision_np)