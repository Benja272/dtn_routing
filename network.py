
from contact_plan import ContactPlan
from typing import List
import numpy as np
import itertools
import numpy.typing as npt
import ipdb

node_id_index = 0
sdp_index = 1
energy_index = 2
delay_index = 3

class Decision:
    @staticmethod
    def is_worse_desicion(d1, d2, priorities):
        d1, d2 = list(d1), list(d2)
        if d1[1] == 0 and d2[1] > 0: return True
        d1[1], d2[1] = -d1[1], -d2[1]
        for i in priorities:
            if (d1[i] > d2[i]):
                return True
            elif d1[i] < d2[i]:
                return False
        return False

    @staticmethod
    def estimate_sdp_energy(prob, future_desicions, energy=1):
        return np.dot(prob, (future_desicions[sdp_index], (energy + future_desicions[energy_index])))


class Contact:
    def __init__(self, from_n: int, to_n: int, t_range: (int, int), pf: float, data_rate: int):
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

# class Node:
#     def __init__(self, id: int, contacts: List[Contact]):
#         self.id = id
#         self.index = 0
#         self.contacts = contacts
#         self.contacts.sort(key=lambda c: c.t_until, reverse=True)

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
            contacts.append(Contact(c.source, c.target, (c.start_t, c.end_t), pf, c.data_rate))
            # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), pf, c.data_rate))

        return Network(contacts, cp.start_time, cp.end_time, cp.node_number, priorities)

    def to_dict(self) -> dict:
        return {'nodes': [n.to_dict() for n in self.nodes]}

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
                    success_cases.append((coin_prob, delay + case_i[delay_index]))
                sdp_energy_sum = np.add(sdp_energy_sum, sdp_energy_cost)
            i += 1
            i_failed_cases = itertools.combinations(coincidences, i)

        return sdp_energy_sum, success_cases




    def get_costs(self, succes_coincidences, fail_coincidences, contact, next_t):
        success_case = self.rute_table[next_t][contact.to-1][self.target][succes_coincidences].copy()
        if success_case[sdp_index] > 0:
            success_sdp_energy_sum = np.array([success_case[1], success_case[2]])
            if succes_coincidences == 0:
                success_sdp_energy_sum = Decision.estimate_sdp_energy((1 - contact.pf), success_case)
            fail_sdp_energy_sum, success_cases = self.case_cost(0, fail_coincidences, next_t, contact.pf, contact.delay)
            success_cases.append((1 - contact.pf, contact.delay + success_case[delay_index]))
            sdp_energy_sum = np.add(success_sdp_energy_sum, fail_sdp_energy_sum)
            return np.append(sdp_energy_sum, self.estimate_delay(success_cases))
        else:
            return [0, 0, 0]



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
        # if self.rute_table[t+1][self.source][self.target][success_coincidences]['sdp'] == 0:
        #     return (self.source + 1, 0, self.slot_range, self.slot_range)
        sdp_energy_sum, success_cases = self.case_cost(success_coincidences, fail_coincidences, t+1, 1, 1, energy=0)
        return [self.source + 1] + list(sdp_energy_sum) + [self.estimate_delay(success_cases)]

    def set_best_desicions(self, max_copies, contacts, t) -> None:
        sended_copies = {}
        nodes = {}
        for i in range(max_copies): #revisar si se envio o no en la copia anterior
            best_desicions = {}
            for c in contacts:
                next_t = t+c.delay
                self.source = c.from_n - 1
                if self.source not in sended_copies.keys():
                    sended_copies[self.source] = [{} for i in range(self.node_number)]
                    nodes[self.source] = [i for i in range(self.node_number)]
                if self.source not in best_desicions.keys():
                    best_desicions[self.source] = [(self.rute_table[t][self.source][target][i].copy(), (self.source + 1, t+1), 1) for target in range(self.node_number)]
                for self.target in nodes[self.source]:
                    if i > 0:
                        best_desicions[self.source][self.target] = [self.next_decision(t, sended_copies[self.source][self.target]), (self.source + 1, t+1), 1]
                    success_coincidences, fail_coincidences = self.coincidences(sended_copies[self.source][self.target], next_t, c.to)

                    decision = np.append(c.to, self.get_costs(success_coincidences, fail_coincidences, c, next_t))
                    if decision[sdp_index] > 0 and Decision.is_worse_desicion(best_desicions[self.source][self.target][0], decision, self.priorities):
                        best_desicions[self.source][self.target] = [decision, (c.to, next_t), c.pf]

            for source in best_desicions.keys():
                targets = nodes[source].copy()
                for target in targets:
                    best_desicion, best_send_pair, pf = best_desicions[source][target]
                    if best_desicion[sdp_index] > 0:
                        sended_copies[source][target][best_send_pair] = pf
                        self.rute_table[t][source][target][i] = best_desicion
                    else:
                        nodes[source].remove(target)


    def set_delays(self, bundle_size):
        for contact in self.contacts:
            contact.set_delay(bundle_size)

    def run_multiobjective_derivation(self, bundle_size=1, max_copies = 1):
        self.rute_table = np.zeros((self.slot_range, self.node_number, self.node_number, max_copies, 4), dtype=np.float)
        for node in range(self.node_number):
            self.rute_table[self.end_time][node][node][0] = (node + 1, 1, 0, 0)
        self.set_delays(bundle_size)
        for t in range(self.end_time-1, self.start_time -1, -1):
            self.rute_table[t] = self.rute_table[t+1].copy() + np.array([0, 0, 0, 1])
            for source in range(self.node_number):
                self.rute_table[t][source][:,:,0] = source + 1
            contacts = self.contacts_in_slot(t)
            self.set_best_desicions(max_copies, contacts, t)


    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", *d)


# init_value = np.empty((), dtype=Decision_np)
#         init_value[()]= (0, 0, slot_range, slot_range)
#         self.rute_table = np.full((slot_range, self.node_number, self.node_number, max_copies), init_value, dtype=Decision_np)