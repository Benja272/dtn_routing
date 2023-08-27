
from contact_plan import ContactPlan
from typing import List
import numpy as np
import itertools
import numpy.typing as npt
import ipdb

Decision_np = np.dtype([("node_id", np.int32), ("sdp", np.float32), ("energy", np.float32), ("delay", np.float32)]) # (node_id, sdp, energy, delay)

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
        return np.dot(prob, (future_desicions['sdp'], (energy + future_desicions['energy'])))


class Contact:
    def __init__(self, to: int, t_range: (int, int), pf: float, data_rate: int):
        self.to: int = to
        self.pf: float = pf
        self.data_rate: int = data_rate
        self.t_since: int = t_range[0]
        self.t_until: int = t_range[1]
        assert self.t_since < self.t_until

    def set_delay(self, data_size: int):
        self.delay = int(data_size / self.data_rate)

    def __str__(self):
        return 'send_to:%d_since:%d_until:%d_pf:%f'%(self.to, self.t_since, self.t_until, self.pf)

    def to_dict(self) -> dict:
        d = {'from':self.from_, 'to': self.to, 'ts': self.ts}
        if self.__begin_time != -1:
            d['begin_time'] = self.__begin_time
            d['end_time'] = self.__end_time
        return d

    @staticmethod
    def useful_contacts(contacts: List['Contact'], t: int, endtime: int) -> List['Contact']:
        return [c for c in contacts if t+c.delay <= endtime]

class Node:
    def __init__(self, id: int, contacts: List[Contact]):
        self.id = id
        self.index = 0
        self.contacts = contacts
        self.contacts.sort(key=lambda c: c.t_until, reverse=True)

    def contacts_in_slot(self, t: int) -> List[Contact]:
        contacts = []
        while self.index < len(self.contacts)-1 and self.contacts[self.index].t_since > t:
            self.index += 1
        i = self.index
        while i < len(self.contacts):
            if self.contacts[i].t_until <= t:
                break
            if(self.contacts[i].t_since <= t):
                contacts.append(self.contacts[i])
            i += 1
        return contacts

class Network:
    def __init__(self, nodes: List[Node], start_time, end_time, node_number, priorities):
        self.nodes = nodes
        self.start_time = start_time
        self.end_time = end_time
        self.slot_range = end_time - start_time+1
        self.node_number = node_number
        self.priorities = priorities
        assert len(priorities) == 3 and 1 in priorities and 2 in priorities and 3 in priorities
        print("Network created with %d nodes, %d slots, %d contacts and priorities %s"%(node_number, self.slot_range, sum([len(n.contacts) for n in nodes]), priorities))

    @staticmethod
    def from_contact_plan(cp_path: str, pf: float = 0.5, priorities = [1,2,3]):
        nodes = []
        nodes_contacts = {}
        cp = ContactPlan.parse_contact_plan(cp_path)
        for n in range(1, cp.node_number+1):
            nodes_contacts[n] = []
        for c in cp.contacts:
            nodes_contacts[c.source].append(Contact(c.target, (c.start_t, c.end_t), pf, c.data_rate))
            # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), pf, c.data_rate))

        for n in range(1, cp.node_number+1):
            nodes.append(Node(n, nodes_contacts[n]))

        return Network(nodes, cp.start_time, cp.end_time, cp.node_number, priorities)

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
                    success_cases.append((coin_prob, delay + case_i['delay']))
                sdp_energy_sum = np.add(sdp_energy_sum, sdp_energy_cost)
            i += 1
            i_failed_cases = itertools.combinations(coincidences, i)

        return sdp_energy_sum, success_cases




    def get_costs(self, succes_coincidences, fail_coincidences, contact, next_t):
        success_case = self.rute_table[next_t][contact.to-1][self.target][succes_coincidences].copy()
        if success_case[1] > 0:
            success_sdp_energy_sum = np.array([success_case[1], success_case[2]])
            if succes_coincidences == 0:
                success_sdp_energy_sum = Decision.estimate_sdp_energy((1 - contact.pf), success_case)
            fail_sdp_energy_sum, success_cases = self.case_cost(0, fail_coincidences, next_t, contact.pf, contact.delay)
            success_cases.append((1 - contact.pf, contact.delay + success_case['delay']))
            sdp_energy_sum = np.add(success_sdp_energy_sum, fail_sdp_energy_sum)
            return tuple(np.append(sdp_energy_sum, self.estimate_delay(success_cases)))
        else:
            return (0, 0, 0)



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
        return (self.source + 1,) + tuple(sdp_energy_sum) + (self.estimate_delay(success_cases),)

    def get_best_desicions(self, max_copies, contacts, t) -> Decision_np:
        sended_copies = {}
        best_desicion = np.zeros(max_copies, dtype=Decision_np)
        contacts = Contact.useful_contacts(contacts, t, self.end_time)
        for i in range(max_copies):
            if t + 1 < self.end_time:
                best_send_pair = (self.source + 1, t+1)
                best_desicion[i] = self.next_decision(t, sended_copies)
                pf = 1 #porque la probabilidad de que "falle" y termine ocurriendo la coincidencia es 1
            for c in contacts:
                next_t = t+c.delay
                success_coincidences, fail_coincidences = self.coincidences(sended_copies, next_t, c.to)
                decision = (c.to,) + self.get_costs(success_coincidences, fail_coincidences, c, next_t)
                if decision[1] > 0 and Decision.is_worse_desicion(best_desicion[i], decision, self.priorities):
                    best_desicion[i] = decision
                    best_send_pair = (c.to, next_t)
                    pf = c.pf
            if best_desicion[i]['sdp'] > 0:
                sended_copies[best_send_pair] = pf
            else:
                break
        return best_desicion


    def set_delays(self, bundle_size):
        for node in self.nodes:
            for contact in node.contacts:
                contact.set_delay(bundle_size)

    def run_multiobjective_derivation(self, bundle_size=1, max_copies = 1):
        self.rute_table = np.zeros((self.slot_range, self.node_number, self.node_number, max_copies), dtype=Decision_np)
        self.set_delays(bundle_size)
        for t in range(self.end_time, self.start_time -1, -1):
            print("t=", t)
            for self.source in range(self.node_number):
                contacts = self.nodes[self.source].contacts_in_slot(t + self.start_time) # creo que esta mal sumar el start_time
                # contacts.append(Contact(self.source + 1, (t, t+1), 0, bundle_size))
                for self.target in range(self.node_number):
                    if self.source == self.target:
                        self.rute_table[t][self.source][self.target][0] = (self.source + 1, 1, 0, 0)
                        continue
                    self.rute_table[t][self.source][self.target] = self.get_best_desicions(max_copies, contacts, t)


    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", d)


# init_value = np.empty((), dtype=Decision_np)
#         init_value[()]= (0, 0, slot_range, slot_range)
#         self.rute_table = np.full((slot_range, self.node_number, self.node_number, max_copies), init_value, dtype=Decision_np)