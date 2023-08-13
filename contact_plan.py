from typing import *
from statistics import mean
from copy import copy


class CP_Contact:

    def __init__(self, source, target, start_t, end_t, data_rate, id=-1):
        '''
        :param start_t: seconds
        :param end_t: seconds
        :param data_rate: bytes per second
        :param id:
        :return:
        '''
        assert source > 0 and target > 0 and 0 <= start_t < end_t
        assert id!=0, 'Valid id starts from 1. Besides, -1 is valid to represent a unsetted identifier'
        self.source = source
        self.target = target
        self.start_t = start_t
        self.end_t = end_t
        self.data_rate = data_rate
        self.id = id
        self.duration = self.end_t - self.start_t

    @staticmethod
    def get_contact_from_str(line:str, id=-1) -> 'CP_Contact':
        c = line.split(' ')
        return CP_Contact(int(c[4]), int(c[5]), int(c[2][1:]), int(c[3][1:]), int(c[6]), id=id)

    def __str__(self):
        return f"a contact {self.start_t:+08} {self.end_t:+08} {self.source} {self.target} {self.data_rate}"

    def __eq__(self, other):
        if type(other) is CP_Contact:
            return all(getattr(self, attr) == getattr(other, attr) for attr in self.__dict__.keys())
        else:
            raise ValueError(f"{CP_Contact}:__eq__ other must be a {CP_Contact} but it is a {type(other)}")

    def __hash__(self):
        return hash(str(self))



class GraphTS:
    def __init__(self, ts_begin:int, ts_end:int):
        self.ts_begin = ts_begin
        self.ts_end = ts_end
        self.contacts = []

    def add_contact(self, c:int):
        self.contacts.append(c)

    def __eq__(self, other):
        if type(other) is GraphTS:
            return self.contacts == other.contacts
        else:
            raise ValueError(f"{GraphTS}:__eq__ other must be a {GraphTS} but it is a {type(other)}")


class ContactPlan:

    def __init__(self, contacts):
        '''
        In order to keep compatibility with DTNSim:
            EIDs must be greater than 0
            Contacts id must be greater than 0 or -1
        When a contact plan is translated to a Net (network abstraction) node EIDs and contact ids are decreased by one
        :param contacts:
        '''
        assert all(c.id > 0 or c.id == -1 for c in contacts),'All contact id must be -1 or > 0'
        assert all(c.source > 0 and c.target > 0 for c in contacts), 'All eid must be > 0'

        self.contacts = contacts
        self.start_time = min((c.start_t for c in self.contacts), default=-1)
        self.end_time = max((c.end_t for c in self.contacts), default=-1)
        self.node_number = max([c.source for c in self.contacts] + [c.target for c in self.contacts], default=0)

    @staticmethod
    def parse_contact_plan(cp_path) -> 'ContactPlan':
        contacts = []
        with open(cp_path, 'r') as cp:
            id = 1
            for line in cp.readlines():
                if line.startswith('a contact'):
                    contacts.append(CP_Contact.get_contact_from_str(line, id=id))
                    id += 1

        return ContactPlan(contacts)

    def rename_eids(self, f_rename:Dict[int,int], allow_multiple_same_id=False):
        assert len(f_rename.values()) == len(set(f_rename.values())) or allow_multiple_same_id
        assert all(n > 0 for n in list(f_rename.keys()) + list(f_rename.values())), "EIDs must be possitive ang greater than 0"

        for c in self.contacts:
            if c.source in f_rename.keys():
                c.source = f_rename[c.source]
            if c.target in f_rename.keys():
                c.target = f_rename[c.target]

        self.node_number = max([c.source for c in self.contacts] + [c.target for c in self.contacts])

    def slice_cp(self, start_time:int, end_time:int) -> 'ContactPlan':
        assert 0 <= start_time < end_time
        slice_contact = []
        id = 1
        for c in self.contacts:
            if c.end_t > start_time and c.start_t < end_time:
                c_copy = copy(c)
                c_copy.start_t = max(c.start_t, start_time)
                c_copy.end_t = min(c.end_t, end_time)
                c_copy.id = id
                slice_contact.append(c_copy)
                id += 1

        return ContactPlan(slice_contact)

    def filter_contact_by_endpoints(self, valid_sources:List[int], valid_targets: List[int]):
        valid_contacts = []
        for c in self.contacts:
            if c.source in valid_sources and c.target in valid_targets:
                valid_contacts.append(copy(c))

        return ContactPlan(valid_contacts)

    def print_to_file(self, output_path):
        with open(output_path, 'w') as f:
            for c in self.contacts:
                f.write(str(c) + '\n')


    def generate_statistics(self):
        statistics = dict()

        statistics['contact_duration-avg'] = mean([c.duration for c in self.contacts]) if len(self.contacts) > 0 else -1
        statistics['contact_duration-min'] = min([c.duration for c in self.contacts], default=-1)
        statistics['contact_duration-max'] = max([c.duration for c in self.contacts], default=-1)

        return statistics

    def generate_slicing_abstraction(self, slicing_time, min_contact_duration):
        def f(ts_duration:int, min_contact_duration:int, graph_by_slice: List[GraphTS]):
            nets = []
            id = 1
            for t in range(0, len(graph_by_slice)):
                graph = graph_by_slice[t]
                t_graph = GraphTS(t*ts_duration, (t+1) * ts_duration)
                for c in graph.contacts:
                    if min((t+1) * ts_duration, c.end_t) - max(t * ts_duration, c.start_t) >= min_contact_duration:
                        start_t = max(t * ts_duration, c.start_t)
                        end_t = min((t+1) * ts_duration, c.end_t)
                        t_graph.add_contact(CP_Contact(c.source, c.target, start_t, end_t, c.data_rate, id=id))
                        id+=1

                nets.append(t_graph)
            return nets

        return self._get_net(lambda graph_by_slice: f(slicing_time, min_contact_duration, graph_by_slice), delta_t=slicing_time)

    def generate_all_contact_abstraction(self):
        def generate_net(graph_by_slice: List[GraphTS]):
            nets = []
            graph = graph_by_slice[0]
            for t in range(1, len(graph_by_slice)):
                if graph == graph_by_slice[t]:
                    print(f"[Ts {t}]: Graph are equals")
                    graph.ts_end += 1
                elif len(graph.contacts) > 0 and len([c for c in graph.contacts if c in graph_by_slice[t].contacts]) == 0:
                    print(f"[Ts {t}]: Empty interseccion -> SPLIT GRAPHS")
                    nets.append(graph)
                    graph = graph_by_slice[t]
                elif all(c in graph_by_slice[t].contacts for c in graph.contacts):
                    print(f"[Ts {t}]: Add contacts")
                    graph_by_slice[t].ts_begin = graph.ts_begin
                    graph = graph_by_slice[t]
                else:
                    print(f"[Ts {t}]: Add and Erase contacts")
                    graph.contacts = list(set(graph.contacts + graph_by_slice[t].contacts))
                    graph.ts_end += 1
            nets.append(graph)

            return nets

        return self._get_net(generate_net)

    def shift_cp_in_time(self, delta_t):
        '''

        :param delta_t: > 0 to shift to future, < 0 to shift to past
        :return:
        '''
        new_contacts = []
        for c in self.contacts:
            c_new = copy(c)
            c_new.start_t += delta_t #To begin from t=0
            c_new.end_t += delta_t  # To begin from t=0
            new_contacts.append(c_new)

        return ContactPlan(new_contacts)
