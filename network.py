
from contact_plan import ContactPlan
from typing import List
import numpy as np
import itertools
import numpy.typing as npt
import ipdb

Decision = np.dtype('int, float, float, float') # (node_id, sdp, energy, delay)

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
        self.node_number = node_number
        self.priorities = priorities
        assert len(priorities) == 3 and 1 in priorities and 2 in priorities and 3 in priorities

    def to_dict(self) -> dict:
        return {'nodes': [n.to_dict() for n in self.nodes]}

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

        # ipdb.set_trace()
        for n in range(1, cp.node_number+1):
            nodes.append(Node(n, nodes_contacts[n]))

        return Network(nodes, cp.start_time, cp.end_time, cp.node_number, priorities)

    def costs(self, desicion):
        return list(desicion)[1:]

    def is_worse_desicion(self, d1, d2):
        d1, d2 = list(d1), list(d2)
        d1[1], d2[1] = (-1)*d1[1], (-1)*d2[1]
        for i in self.priorities:
            if (d1[i] > d2[i]):
                return True
            elif d1[i] < d2[i]:
                return False
        return False

    def estimate_costs(self, prob, future_desicions, delay, energy=1):
        future_costs = list(future_desicions)[1:]
        return np.dot(prob, (future_costs[0], (energy + future_costs[1]), (delay + future_costs[2])))

    def estimate_coincidences(self, coincidences, fail_case, cost_sum):
        for c in coincidences:
            if c in fail_case:#c[1] = pf, asumo que coincide porque fallo el contacto anterior
                cost_sum = np.dot(c[1], cost_sum)
            else: # asumo que no hay coincidencia porque no fallo el contacto anterior
                cost_sum = np.dot(1 - c[1], cost_sum)
        return cost_sum

    def case_cost(self, coincidences_base, coincidences, next_t, prob, delay, energy=1):
        case_cost = self.estimate_costs(prob, self.rute_table[next_t][self.source][self.target][coincidences_base].copy(), delay, energy)
        fail_case_sum = self.estimate_coincidences(coincidences, (), case_cost)
        for i in range(1, len(coincidences)):
            # en caso de que no llegue al target deberia castigarse el delay y la energia
            case_cost = self.estimate_costs(prob, self.rute_table[next_t][self.source][self.target][coincidences_base + i].copy(), delay, energy)
            for failed in itertools.combinations(coincidences, i):
                case_cost = self.estimate_coincidences(coincidences, failed, case_cost)
                fail_case_sum = np.add(fail_case_sum, case_cost)
        return fail_case_sum


    def get_costs(self, succes_coincidences, fail_coincidences, contact, next_t):
        # if self.source == 0 and self.target == 2 and next_t == 1: ipdb.set_trace()
        success_case = self.rute_table[next_t][contact.to-1][self.target][succes_coincidences].copy()
        success_case_sum = np.array([success_case[1], success_case[2], success_case[3]])
        if succes_coincidences == 0 and success_case[1] > 0:
            success_case_sum = self.estimate_costs((1 - contact.pf), success_case, contact.delay)
        fail_case_sum = self.case_cost(0, fail_coincidences, next_t, contact.pf, contact.delay)
        return tuple(np.add(success_case_sum, fail_case_sum))



    def coincidences(self, sended_copies, next_t, target):
        success_coincidences = 0
        fail_coincidences = []
        for s in sended_copies.keys():
            if s[1] == next_t and s[0] == target: #usa mismo contacto
                success_coincidences += 1
            elif s[1] == next_t:
                fail_coincidences.append((s[0], sended_copies[s])) #coincidentes en el tiempo en caso de fallar
        return success_coincidences, fail_coincidences

    def get_best_desicions(self, max_copies, contacts, t):
        sended_copies = {}
        best_desicion = np.zeros(max_copies, dtype=Decision)
        for i in range(max_copies):
            if t + 1 < self.end_time:
                # if t == 2 and self.source == 3 and self.target == 5: ipdb.set_trace()
                best_send_pair = (self.source + 1, t+1)
                success_coincidences, fail_coincidences = self.coincidences(sended_copies, t+1, self.source + 1)
                next_cost = tuple(self.case_cost(success_coincidences, fail_coincidences, t+1, 1, 1, energy=0))
                best_desicion[i] = (self.source + 1,) + next_cost
                pf = 1 #porque la probabilidad de que "falle" y termine ocurriendo la coincidencia es 1
            for c in contacts:
                next_t = t+c.delay
                if next_t > self.end_time:
                    continue
                success_coincidences, fail_coincidences = self.coincidences(sended_copies, next_t, c.to)
                decision = (c.to,) + self.get_costs(success_coincidences, fail_coincidences, c, next_t)
                if self.is_worse_desicion(best_desicion[i], decision):
                    best_desicion[i] = decision
                    best_send_pair = (c.to, next_t)
                    pf = c.pf
            if best_desicion[i][1] > 0:
                sended_copies[best_send_pair] = pf
            else:
                break
        return best_desicion


    def set_delays(self, bundle_size):
        for node in self.nodes:
            for contact in node.contacts:
                contact.set_delay(bundle_size)

    def rucop(self, bundle_size=1, max_copies = 1):
        slot_range = self.end_time - self.start_time+1
        init_value = np.empty((), dtype=Decision)
        init_value[()]= (0, 0, slot_range, slot_range)
        self.rute_table = np.full((slot_range, self.node_number, self.node_number, max_copies), init_value, dtype=Decision)
        print(self.rute_table[0])
        self.set_delays(bundle_size)
        print(self.rute_table[0])
        for t in range(self.end_time, self.start_time -1, -1):
            for self.source  in range(self.node_number):
                contacts = self.nodes[self.source].contacts_in_slot(t + self.start_time)
                # contacts.append(Contact(self.source + 1, (t, t+1), 0, bundle_size))
                for self.target in range(self.node_number):
                    if self.source == self.target:
                        self.rute_table[t][self.source][self.target][0] = (self.source + 1, 1, 0, 0)
                        continue
                    best_d = self.get_best_desicions(max_copies, contacts, t)
                    self.rute_table[t][self.source][self.target] = best_d
            # if (self.source == 5 and 5== self.target): ipdb.set_trace()


    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", d)