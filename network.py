
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
        return 'send_%d_%d_%d'%(self.to, self.t_since, self.t_until)

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

    def is_worse_desicion(self, d1, d2):
        for i in self.priorities:
            if ((i == 1 and d1[i] < d2[i]) or (i != 0 and  d1[i] > d2[i])):
                return True
            elif d1[i] < d2[i]:
                return False
        return False

    def insert_if_better(self, decisions, d):
        print("inserting ", d, " in ", decisions)
        last_index = len(decisions) - 1
        index = last_index
        is_worse = self.is_worse_desicion(decisions[index], d)
        while index >= 0 and is_worse:
            decisions[index] = decisions[index-1]
            index -= 1
            is_worse = self.is_worse_desicion(decisions[index], d)
        if is_worse:
            decisions[index+1] = d
        print("result ", decisions)
        return decisions

    def get_costs(self, succes_coincidences, fail_coincidences, contact, next_t):
        success_case = self.rute_table[next_t][self.source][contact.to-1][succes_coincidences]
        success_case_pf, success_case_delay, success_case_energy = success_case[1], success_case[2], success_case[3]
        if succes_coincidences == 0:
            success_case_pf = (1 - contact.pf) * success_case[1]
            success_case_energy = (1 - contact.pf)*(1 + success_case[2])
            success_case_delay = (1 - contact.pf)*(contact.delay + success_case[3])
        fail_case_pf_sum, fail_case_delay_sum, fail_case_energy_sum = 0, 0, 0
        for i in range(len(fail_coincidences)):
            fail_case = self.rute_table[next_t][self.source][self.target][i]
            fail_case_pf = contact.pf * fail_case[1]
            fail_case_energy = contact.pf*(1 + fail_case[2])
            fail_case_delay = contact.pf*(contact.delay + fail_case[3])
            for failed in itertools.combinations(fail_coincidences, i):
                for c in fail_coincidences:
                    if c in failed: #c[1] = pfa, asumo que coincide porque fallo el contacto anterior
                        fail_case_pf *= c[1]
                        fail_case_energy *= c[1]
                        fail_case_delay *= c[1]
                    else: #asumo que no hay coincidencia porque no fallo el contacto anterior
                        fail_case_pf *= (1 - c[1])
                        fail_case_energy *= (1 - c[1])
                        fail_case_delay *= (1 - c[1])
                fail_case_pf_sum += fail_case_pf
                fail_case_energy_sum += fail_case_energy
                fail_case_delay_sum += fail_case_delay
        return (success_case_pf + fail_case_pf_sum, success_case_energy + fail_case_energy_sum, success_case_delay + fail_case_delay_sum)



    def coincidences(self, sended_copies, next_t, source, contact):
        success_coincidences = 0
        fail_coincidences = []
        for s in sended_copies.keys():
            if s[1] == next_t and s[0] == contact.to: #usa mismo contacto
                success_coincidences += 1
            elif s[1] == next_t:
                fail_coincidences.append((s[0], sended_copies[s])) #coincidentes en el tiempo en caso de fallar
        return success_coincidences, fail_coincidences

    def get_best_desicions(bundle_size, self, max_copies, contacts, t):
        n = len(contacts)
        sended_copies = {}
        contacts :[Contact]= []
        best_desicion = np.zeros(max_copies, dtype=Decision)
        for i in range(max_copies):
            best_desicion[i] = (self.source + 1,) + tuple(self.rute_table[t+1][self.source][self.target][0])[1:]
            for c in contacts:
                c.set_delay(bundle_size) #mejorar
                next_t = t+c.delay
                if next_t > self.end_time:
                    continue
                success_coincidences, fail_coincidences = self.coincidences(sended_copies, next_t, self.source, c)
                cost = self.get_costs(success_coincidences, fail_coincidences, c, next_t)
                best_desicion = self.insert_if_better(best_desicion, (c.to,) + cost)




    def rucop(self, bundle_size=1, max_copies = 1):
        self.rute_table = np.zeros((self.end_time - self.start_time+1, self.node_number, self.node_number, max_copies), dtype=Decision)
        for t in range(self.end_time, self.start_time -1, -1):
            for source in range(self.node_number):
                self.source = source
                contacts = self.nodes[source].contacts_in_slot(t + self.start_time)
                for target in range(self.node_number):
                    self.target = target
                    if source == target:
                        self.rute_table[t][source][target][0] = (source + 1, 1, 0, 0)
                        continue
                    best_desicion = self.get_best_desicions(bundle_size, max_copies, contacts, t)
                    self.rute_table[t][source][target] = best_desicion

    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", d)