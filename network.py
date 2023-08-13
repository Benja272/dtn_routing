
from contact_plan import ContactPlan
from typing import List
import numpy as np
import numpy.typing as npt
import ipdb

decision = np.dtype('int, float') # (node_id, sdp)

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
    def __init__(self, nodes: List[Node], start_time, end_time, node_number):
        self.nodes = nodes
        self.start_time = start_time
        self.end_time = end_time
        self.node_number = node_number

    def to_dict(self) -> dict:
        return {'nodes': [n.to_dict() for n in self.nodes]}

    @staticmethod
    def from_contact_plan(cp_path: str):
        nodes = []
        nodes_contacts = {}
        cp = ContactPlan.parse_contact_plan(cp_path)
        for n in range(1, cp.node_number+1):
            nodes_contacts[n] = []
        for c in cp.contacts:
            nodes_contacts[c.source].append(Contact(c.target, (c.start_t, c.end_t), 0.5, c.data_rate))
            # nodes_contacts[c.target].append(Contact(c.source, (c.start_t, c.end_t), 0.5, c.data_rate))

        # ipdb.set_trace()
        for n in range(1, cp.node_number+1):
            nodes.append(Node(n, nodes_contacts[n]))

        return Network(nodes, cp.start_time, cp.end_time, cp.node_number)

    def insert_if_better(self, decisions, d):
        last_index = len(decisions) - 1
        if decisions[last_index][1] < d[1]:
            decisions[last_index] = d
        decisions = sorted(decisions, key=lambda d: d[1], reverse=True)
        return decisions

    def get_best_desicions(self, max_copies, contacts, rute_table, bundle_size, source, target, t):
        best_desicion = np.zeros(max_copies, dtype=decision)
        if (t != self.end_time):
            best_desicion[0] = (source + 1, rute_table[t+1][source][target][0][1])
        for c in contacts:
            c.set_delay(bundle_size) #mejorar
            if t + c.delay > self.end_time:
                continue
            for i in range(max_copies):
                sdp = rute_table[t+c.delay][c.to-1][target][i][1] * (1-c.pf)
                if t + 2*c.delay <= self.end_time:
                    sdp += rute_table[t+2*c.delay][source][target][0][1] * c.pf
                best_desicion = self.insert_if_better(best_desicion, (c.to, sdp))
        return best_desicion


    def rucop(self, bundle_size, max_copies = 1):
        rute_table = np.zeros((self.end_time - self.start_time+1, self.node_number, self.node_number, max_copies), dtype=decision)
        for t in range(self.end_time, self.start_time -1, -1):
            for source in range(self.node_number):
                contacts = self.nodes[source].contacts_in_slot(t + self.start_time)
                for target in range(self.node_number):
                    if source == target:
                        rute_table[t][source][target][0] = (source + 1, 1)
                        continue
                    best_desicion = self.get_best_desicions(max_copies, contacts, rute_table, bundle_size, source, target, t)
                    rute_table[t][source][target] = best_desicion
        self.rute_table = rute_table

    def print_table(self):
        for t in range(self.end_time -1, self.start_time -1, -1):
            for source in range(self.node_number):
                for target in range(self.node_number):
                    if self.rute_table[t][source][target][0][1] > 0 and source != target:
                        d = self.rute_table[t][source][target]
                        print("En t=",t," desde ", source +1, " hasta ", target +1, " con desiciones ", d)