import networkx as nx
from functools import reduce
from abc import ABC
from typing import List, Tuple, Dict
import os
from collections import OrderedDict
import itertools
from copy import copy
from brufn.utils import average
from brufn.net_reachability_closure import reachability_clousure
from datetime import date
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from brufn.state_factory import StateFactory


class Contact:

    def __init__(self, from_: int, to: int, ts: int, pf: float = 0, identifier: int = -1, begin_time: int = -1, end_time: int = 0):
        self._from: int = from_
        self._to: int = to
        self._ts: int = ts
        self.pf: float = pf
        self._id: int = identifier

        self.__begin_time: int = begin_time
        self.__end_time: int = end_time

    def __str__(self):
        return 'send_%d_%d_%d'%(self.from_, self.to, self.ts)

    def to_dict(self) -> dict:
        d = {'from':self.from_, 'to': self.to, 'ts': self.ts}
        if self.__begin_time != -1:
            d['begin_time'] = self.__begin_time
            d['end_time'] = self.__end_time
        return d

    @property
    def from_(self) -> int:
        return self._from

    @property
    def to(self) -> int:
        return self._to

    @property
    def ts(self) -> int:
        return self._ts

    @property
    def id(self) -> int:
        return self._id

    @property
    def begin_time(self):
        return self.__begin_time

    @property
    def end_time(self):
        return self.__end_time

    def __eq__(self, other):
        """Overrides the default implementation. Id is ignored"""
        if isinstance(other, Contact):
            return self.from_ == other.from_ and self.to == other.to and self.ts == other.ts and self.pf == other.pf and self.begin_time == other.begin_time and self.end_time == other.end_time

        return False

    def __hash__(self):
        """Overrides the default implementation. Id is ignored"""
        return hash((self.from_, self.to, self.ts, self.pf, self.begin_time, self.end_time))

class Route:

    def __init__(self, contacts):
        assert all(contacts[i].to == contacts[i + 1].from_ for i in range(0,len(contacts) - 1)), "It isn't a valid route"
        assert all(contacts[i].ts <= contacts[i + 1].ts for i in range(0, len(contacts) - 1)), "It isn't a valid route"
        assert all(contacts[i].begin_time < contacts[i + 1].end_time for i in range(0, len(contacts) - 1)), "It isn't a valid route"

        self._contacts: tuple[Contact] = tuple(contacts)

    def __str__(self):
        return '_'.join([str(c) for c in self.contacts])

    @property
    def contacts(self) -> Tuple[Contact]:
        return self._contacts

    @property
    def contacts_ids(self) -> Tuple[Contact]:
        return tuple(c.id for c in self._contacts)

    @property
    def sender_node(self) -> int:
        return self.contacts[0].from_

    @property
    def receiver_node(self) -> int:
        return self.contacts[-1].to

    def to_softroute(self):
        return SoftRoute.make_from_Route(self)

    def hop_count(self):
        return len(self._contacts)

    @staticmethod
    def is_valid_route(contacts):
        valid = all(contacts[i].to == contacts[i + 1].from_ for i in range(0, len(contacts) - 1))
        valid = valid and all(contacts[i].ts <= contacts[i + 1].ts for i in range(0, len(contacts) - 1))
        valid = valid and all(contacts[i].begin_time < contacts[i + 1].end_time for i in range(0, len(contacts) - 1))

        return valid


class SoftRoute:

    def __init__(self, contacts, sender_node=None, receiver_node=None):
        self._contacts_: tuple[int] = tuple(contacts)
        if sender_node is not None and receiver_node is not None:
            self._sender_node: int = sender_node
            self._receiver_node: int = receiver_node

    @staticmethod
    def make_from_Route(route:Route):
        if len(route.contacts) == 0:
            return SoftRoute(route.contacts_ids)
        else:
            return SoftRoute(route.contacts_ids, sender_node=route.sender_node, receiver_node=route.receiver_node)

    def __str__(self):
        return '_'.join(["i%d"%c for c in self.contacts])

    @property
    def contacts(self) -> Tuple[int]:
        return self._contacts_

    @property
    def contacts_ids(self) -> Tuple[int]:
        return self._contacts_

    @property
    def sender_node(self) -> int:
        return self._sender_node

    @property
    def receiver_node(self) -> int:
        return self._receiver_node

    def __str__(self):
        return '_'.join([str(c) for c in self.contacts])



class Net:

    def __init__(self, num_of_nodes: int, contacts: List[Contact], traffic:dict = None):
        self.num_of_nodes: int = num_of_nodes
        self.contacts: List[Contact] = []
        for id, c in enumerate(contacts):
            self.contacts.append(Contact(c.from_, c.to, c.ts, pf=c.pf, identifier=id, begin_time=c.begin_time, end_time=c.end_time))
        self.traffic: Dict = traffic
        self.num_of_ts: int = max([c.ts for c in self.contacts]) + 1

        self.networkX_graph: List[nx.DiGraph] = [None] * self.num_of_ts

    '''
    Check is a contact is valid, returns true or false
    '''
    @staticmethod
    def contact_is_valid(c, num_of_nodes, pf_is_required=True):
        if type(c) == dict and all(k in c.keys() for k in ['from', 'to', 'ts']) and \
                type(c['from']) == int and \
                type(c['to']) == int and \
                type(c['ts']) == int and \
                0 <= c['from'] < num_of_nodes and \
                0 <= c['to'] < num_of_nodes and \
                c['from'] != c['to'] and \
                c['ts'] >= 0 and \
                (not pf_is_required or
                 ('pf' in c.keys() and type(c['pf']) == float and 0. <= c['pf'] <= 1.)):
            return True
        return False

    '''
    Get Net from file, if traffic required it returns an error if it does not exist.
    returns
        {'NUM_OF_NODES':int, 'CONTACTS':[{'from':int,'to':int,'ts':int}] (,'TRAFFIC':{'from':int,'to':int, 'ts':int})}
    '''
    @staticmethod
    def get_net_from_file(path_to_net, traffic_required=False, contact_pf_required=True):
        input = {}
        file = open(path_to_net, 'r')
        f = exec(file.read(), input)
        file.close()
        # CHECK INPUT FILE HAS THE REQUIRED FIELDS
        if 'NUM_OF_NODES' in input.keys():
            if type(input['NUM_OF_NODES']) != int or input['NUM_OF_NODES'] <= 1:
                TypeError("[ERROR] NUM_OF_NODES must be an integer greater than 1")
            else:
                NUM_OF_NODES = input['NUM_OF_NODES']
        else:
            TypeError("[ERROR] The input network must contain NUM_OF_NODES")

        if 'CONTACTS' in input.keys():
            if type(input['CONTACTS']) != list or len(input['CONTACTS']) < 1:
                TypeError("[ERROR] CONTACTS must be a list with at least 1 element")
            else:
                # Check if each contact is write in the correct way
                for c in input['CONTACTS']:
                    if not Net.contact_is_valid(c, NUM_OF_NODES, contact_pf_required):
                        err = "[ERROR] Contact must described for a dict: {'from':int,'to':int,'ts':int (,'pf':float)} where: \n"
                        err += "\t to, from are different and to,from in [0,NUM_OF_NODES)\n"
                        err += "\t ts >= 0 \n"
                        err += "\t pf in [0.,1.] (pay attention to write the dots!)\n"
                        err += "\t %s does not satisfy the above properties.\n" % str(c)
                        raise TypeError(err)

                contacts = []
                for i in range(len(input['CONTACTS'])):
                    c = input['CONTACTS'][i]
                    contacts.append(Contact(c['from'], c['to'], c['ts'], pf= c['pf'] if 'pf' in c.keys() else None,
                                            identifier=i))


        else:
            raise TypeError("[ERROR] The input network must contain CONTACTS:[{'from':int,'to':int,'ts':int, 'pf':float}]")

        # Traffic is readed if it exists. If traffic_required, it reports an error when traffic does not exist
        if 'TRAFFIC' in input.keys():
            t = input['TRAFFIC']
            if type(t) == dict and all(k in t.keys() for k in ['from', 'to', 'ts']) and \
                    type(t['from']) == int and \
                    type(t['to']) == int and \
                    type(t['ts']) == int and \
                    0 <= t['from'] < NUM_OF_NODES and \
                    0 <= t['to'] < NUM_OF_NODES and \
                    t['from'] != t['to'] \
                    and t['ts'] >= 0:
                return Net(NUM_OF_NODES, contacts,  traffic = t)
            else:
                err = "[ERROR] TRAFFIC must be a dict: {'from':int,'to':int,'ts':int} where: \n"
                err += "\t to, from are different and to,from in [0,NUM_OF_NODES)\n"
                err += "\t ts >= 0\n"
                raise TypeError(err)

        elif traffic_required:
            print("[ERROR] The input network must contain TRAFFIC:{'from':int,'to':int,'ts':int}")
            return {}

        return Net(NUM_OF_NODES, contacts)

    def compute_routes(self, source: int, target: int,  ts: int) -> List[Route]:
        assert 0 <= ts < self.num_of_ts, 'ts must be in [%d, %d) but ts = %d'%(0, self.num_of_ts, ts)

        if self.networkX_graph[ts] is None:
            self.networkX_graph[ts] = nx.DiGraph()
            self.networkX_graph[ts].add_nodes_from(range(self.num_of_nodes))
            for c in self.contacts:
                if c.ts == ts:
                    self.networkX_graph[ts].add_edge(c.from_, c.to, object=c)

        routes = []
        for path in nx.all_simple_paths(self.networkX_graph[ts], source, target):
            contacts = [self.networkX_graph[ts].edges[path[i],path[i+1]]['object'] for i in range(len(path) - 1)]
            if Route.is_valid_route(contacts):
                routes.append(Route(contacts))

        return routes

    def get_dtnsim_cp(self, ts_duration: int, capacity: int) -> str:
        '''
            It builds a DTNSim Contact Plan and return it as string
        :param ts_duration: Duration of any contact in the network
        :param capacity:
        :return: A string with the requierd contact_plan
        '''

        cp = ""
        for c in self.contacts:
            cp += "a contact +%d +%d %d %d %d\n" % (
            c.ts * ts_duration, (c.ts + 1) * ts_duration, c.from_ + 1, c.to + 1, capacity)

        return cp

    def print_dtnsim_cp_to_file(self, ts_duration: int, capacity: int, output_file_path: str):
        '''
            It builds a DTNSim Contact Plan and return it as string
        :param ts_duration: Duration of any contact in the network
        :param capacity:
        :param output_file_path:
        :return: A string with the required contact_plan
        '''
        with open(output_file_path,'w') as f:
            for c in self.contacts:
                c_start = max(c.ts * ts_duration, c.begin_time)
                if c.end_time > 0:
                    c_end = min((c.ts + 1) * ts_duration, c.end_time)
                else:
                    c_end = (c.ts + 1) * ts_duration
                f.write(f"a contact {c_start:+08} {c_end:+08} {c.from_ + 1} {c.to + 1} {capacity}\n")


    @staticmethod
    def get_net_from_dtnsim_cp(cp_path: str, ts_duration: int) -> 'Net':
        contacts = []
        with open(cp_path) as fin:
            for line in fin:
                if 'a contact' in line:
                    line = line.split(' ')
                    ts_from = int(line[2][1:]) / ts_duration
                    ts_to = int(line[3][1:]) / ts_duration
                    if ts_to - ts_from != 1:
                        err = '''[ERROR] It can not translate networks which has contact with
                                more than 1 ts duration. ts_duration was setted to %d''' % ts_duration
                        raise TypeError(err)
                    node_from = int(line[4]) - 1
                    node_to = int(line[5]) - 1
                    contacts.append(Contact(node_from, node_to, int(ts_from)))

        return Net(max([c.from_ for c in contacts] + [c.to for c in contacts]) + 1, contacts)

    def to_dot(self, output_path: str, file_name: str = 'net.dot'):
        '''
        Save file with network's dot representation

        :param output_path: path to destination folder
        :param file_name: name under which the file will be saved
        :return: None
        '''
        with open(os.path.join(output_path, file_name), 'w') as f:
            f.write("digraph G { \n\n")
            f.write("rank=same;\n")
            f.write("ranksep=equally;\n")
            f.write("nodesep=equally;\n")

            for ts in range(self.num_of_ts):
                f.write("\n// TS = %d\n" % ts)
                for n in range(self.num_of_nodes):
                    f.write("%d.%d[label=L%d];\n" % (n, ts, n))

                f.write(str(self.num_of_nodes) + "." + str(ts) + "[shape=box,fontsize=16,label=\"TS " + str(ts) + "\"];\n")

                for n in range(self.num_of_nodes):
                    f.write("%d.%d -> %d.%d[style=\"invis\"];\n" % (n, ts, n + 1, ts))

                for c in self.contacts:
                    if c.ts == ts:
                        f.write("%d.%d -> %d.%d%s\n" % (c.from_, ts, c.to, ts,
                                                           "[color=green,fontcolor=green,penwidth=2]" if "FLAG" in c.__dict__.keys() else ""))

            f.write("\n\n// Ranks\n")
            for n in range(self.num_of_nodes):
                f.write("{ rank = same;")
                for ts in range(self.num_of_ts):
                    f.write(" %d.%d;" % (n, ts))
                f.write("}\n")
            f.write(" \n}")

    def print_to_file(self, output_path: str, file_name: str ='net.py'):
        '''
        This method prints a file at output_path/file_name with a python representation of current network.

        :param output_path:
        :param file_name:
        :return: None
        '''
        f = open(output_path + (lambda s: "/" if s[-1] != '/' else '')(output_path) + file_name, 'w')
        f.write("NUM_OF_NODES = %d\n" % self.num_of_nodes)
        f.write("CONTACTS = [%s] \n" %','.join([str(c.to_dict()) for c in self.contacts]))
        if self.traffic is not None:
            # Traffic is optional
            f.write("TRAFFIC = %s\n" % self.traffic)

        f.close()

    def get_contact_by_id(self, id):
        for c in self.contacts:
            if c.id == id:
                return c

    def generate_statictiscs(self):
        statistics = dict()

        net_contacts_by_ts = dict((ts, {'contacts': [], 'min_tstart': -1,'max_tend': -1}) for ts in range(self.num_of_ts))
        for c in self.contacts:
            net_contacts_by_ts[c.ts]['contacts'].append(c)
            if net_contacts_by_ts[c.ts]['min_tstart'] > c.begin_time or net_contacts_by_ts[c.ts]['min_tstart'] < 0:
                net_contacts_by_ts[c.ts]['min_tstart'] = c.begin_time
            if net_contacts_by_ts[c.ts]['max_tend'] < c.end_time:
                net_contacts_by_ts[c.ts]['max_tend'] = c.end_time

        l = [net_contacts_by_ts[c.ts]['max_tend'] - c.end_time for c in self.contacts]
        statistics['distance_contact_end_to_ts_end-avg'] = average(l)
        statistics['distance_contact_end_to_ts_end-min'] = min(l)
        statistics['distance_contact_end_to_ts_end-max'] = max(l)

        l = [c.begin_time - net_contacts_by_ts[c.ts]['min_tstart']  for c in self.contacts]
        statistics['distance_contact_start_to_ts_start-avg'] = average(l)
        statistics['distance_contact_start_to_ts_start-min'] = min(l)
        statistics['distance_contact_start_to_ts_start-max'] = max(l)

        l = [net_contacts_by_ts[ts]['max_tend'] - net_contacts_by_ts[ts]['min_tstart'] for ts in range(self.num_of_ts)]
        statistics['ts_duration-avg'] = average(l)
        statistics['ts_duration-min'] = min(l)
        statistics['ts_duration-max'] = max(l)

        l = [len(net_contacts_by_ts[ts]['contacts']) for ts in range(self.num_of_ts)]
        statistics['ts_number_of_contacts-avg'] = average(l)
        statistics['ts_number_of_contacts-min'] = min(l)
        statistics['ts_number_of_contacts-max'] = max(l)

        return statistics

    def get_all_time_varying_paths(self, source:int, target:int, ts:int=0) -> List[List[Contact]]:
        '''
        Compute simple time varying path in network from source to target starting at ts' >= ts
        :param source: Path starting point
        :param target: Path endpoint
        :param ts: starting ts
        :return: A list of paths (List[Contact])
        '''

        if ts >= self.num_of_ts or source==target: #Check if the starting_ts belong to the net
            return []

        contacts_by_ts = dict((ts, [c for c in self.contacts if c.ts == ts]) for ts in range(self.num_of_ts))
        all_paths = []
        path = []
        visited_nodes = [(source,ts)] # To avoid loops
        contacts = [c for c in contacts_by_ts[ts] if c.from_ == source] #Contacts to explore
        to_visit = [(c.to, ts) for c in contacts] #Nodes to explore
        if ts + 1 < self.num_of_ts:
            contacts = [Contact(source, source, ts)] + contacts
            to_visit = [(source, ts + 1)] + to_visit
        levels = [0] * len(to_visit)

        while len(to_visit) > 0:
            #print(len(to_visit))
            node = to_visit.pop(-1)
            c = contacts.pop(-1)
            level = levels.pop(-1)

            #while len(path) > 0 and c.from_ != path[-1].to: #path with at least one hop, check if this node is a next hop or if the DPS already ends and a path reset is nedeed
            #    path.pop(-1)
            #    visited_nodes.pop(-1)
            while len(path) > level:
                path.pop(-1)
                visited_nodes.pop(-1)

            #if c.from_ != c.to:
            #    path.append(c)
            #    assert Route.is_valid_route(path)
            path.append(c)
            assert Route.is_valid_route(path)

            if node[0] == target and len(path) > 0: #Empty path can happen when source==target
                all_paths.append([c for c in path if c.from_ != c.to])
                path.pop(-1)
            else:
                visited_nodes.append(node)
                if node[1] + 1 < self.num_of_ts:
                    to_visit.append((node[0], node[1] + 1))
                    contacts.append(Contact(node[0], node[0], node[1]))
                    levels.append(len(path))

                to_visit += [(c.to, c.ts) for c in contacts_by_ts[node[1]] if c.from_ == node[0] and (c.to, node[1]) not in visited_nodes]
                contacts.extend([c for c in contacts_by_ts[node[1]] if c.from_ == node[0] and (c.to, node[1]) not in visited_nodes])
                levels.extend([len(path)] * len([c for c in contacts_by_ts[node[1]] if c.from_ == node[0] and (c.to, node[1]) not in visited_nodes]))
        return all_paths

    def get_all_time_varying_paths_non_revising(self, source:int, target:int, ts:int=0) -> List[List[Contact]]:
        '''
        Compute simple time varying path in network from source to target starting at ts' >= ts
        that do not visit a single node more than once (If node 1 send the data at ts i then it
         can not receive the bundle back in any ts. That is allowed by get_all_time_varying_paths)

        :param source: Path starting point
        :param target: Path endpoint
        :param ts: starting ts
        :return: A list of paths (List[Contact])
        '''

        if ts >= self.num_of_ts or source==target: #Check if the starting_ts belong to the net
            return []

        contacts_by_ts = dict((ts, [c for c in self.contacts if c.ts == ts]) for ts in range(self.num_of_ts))
        all_paths = []
        path = []
        visited_nodes = [(source,ts)] # To avoid loops
        contacts = [c for c in contacts_by_ts[ts] if c.from_ == source] #Contacts to explore
        to_visit = [(c.to, ts) for c in contacts] #Nodes to explore
        if ts + 1 < self.num_of_ts:
            contacts = [Contact(source, source, ts)] + contacts
            to_visit = [(source, ts + 1)] + to_visit
        levels = [0] * len(to_visit)

        while len(to_visit) > 0:
            #print(len(to_visit))
            node = to_visit.pop(-1)
            c = contacts.pop(-1)
            level = levels.pop(-1)

            while len(path) > level:
                path.pop(-1)
                visited_nodes.pop(-1)

            path.append(c)
            assert Route.is_valid_route(path)

            if node[0] == target and len(path) > 0: #Empty path can happen when source==target
                all_paths.append([c for c in path if c.from_ != c.to])
                path.pop(-1)
            else:
                visited_nodes.append(node)
                if node[1] + 1 < self.num_of_ts:
                    to_visit.append((node[0], node[1] + 1))
                    contacts.append(Contact(node[0], node[0], node[1]))
                    levels.append(len(path))

                contacs_to_add = [c for c in contacts_by_ts[node[1]] if c.from_ == node[0]
                                  and all((c.to, ts) not in visited_nodes for ts in range(node[1] + 1))]
                to_visit += [(c.to, c.ts) for c in contacs_to_add]
                contacts.extend([c for c in contacs_to_add])
                levels.extend([len(path)] * len(contacs_to_add))
        return all_paths

    def gen_modest_model(self, source, target, pf, copies, ack=True) -> str:
        if ack:
            return self.__gen_modest_model_with_ack(source, target, pf, copies)
        else:
            return self.__gen_modest_model_without_ack(source, target, pf, copies)

    def __gen_modest_model_with_ack(self, source, target, pf, copies) -> str:
        rc = reachability_clousure(self)
        rc_from_source = self.reachability_clousure_from_source(source)
        meaningful_contacts = [c for c in self.contacts if c.to==target or (c.ts + 1 < self.num_of_ts and target in rc[c.to][c.ts + 1])]
        meaningful_contacts = [c for c in meaningful_contacts if c.from_ in rc_from_source[c.ts-1]]
        #meaningful_contacts.sort(key=lambda c: c.ts)
        meaningful_ts = sorted(list(set(c.ts for c in meaningful_contacts)))
        processes = {}
        for node in range(self.num_of_nodes):
            node_process = f"process Node{node}(int(0..{copies}) copies)\n"
            node_process += "{\n"
            for ts in meaningful_ts:
                snd_contact = list(filter(lambda c: c.from_==node and c.ts == ts, meaningful_contacts))
                rcv_contact = list(filter(lambda c: c.to==node and c.ts == ts, meaningful_contacts))
                assert len(snd_contact) <= 1, "No more than 1 outgoing transmission is allowed"
                assert len(rcv_contact) <= 1, "No more than 1 ingoing transmission is allowed"
                assert not snd_contact or not rcv_contact or snd_contact[0].to == rcv_contact[0].from_, "If there are a snd and rcv, it has to be with the same node (full duplex communication)"

                snd_contact = snd_contact[0] if snd_contact else None
                rcv_contact = rcv_contact[0] if rcv_contact else None
                nop_snd = ts + 1 < self.num_of_ts and target in rc[node][ts + 1] if snd_contact else False
                nop_rcv = ts + 1 < self.num_of_ts and target in rc[rcv_contact.from_][ts + 1] if rcv_contact else False
                if snd_contact and rcv_contact:
                    snd = []
                    if nop_snd:
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies >= {c}) tau; sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, 2: copies -= ack{snd_contact.to} == {node} ? {c} : 0 =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")

                        rcv = f":: tau; sync {{= 1: copies += dest{rcv_contact.from_} == {node} ? data{rcv_contact.from_} : 0, 1: ack{node} = {rcv_contact.from_} =}}\n"
                        node_process += f"\talt {{ //slot communicate with {rcv_contact.from_} at ts {rcv_contact.ts}\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t\t" + rcv + "\n"
                        node_process += "\t};\n"
                    else:
                        assert False, "There must not be a contact incoming to a node that can not reach the target"
                elif snd_contact:
                    snd = []
                    if nop_snd:
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies >= {c}) tau; sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, 2: copies -= ack{snd_contact.to} == {node} ? {c} : 0 =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")

                        node_process += f"\talt {{ //slot communicate with {snd_contact.to} at ts {snd_contact.ts}\n"
                        node_process += f"\t\t:: tau; sync //do nothing in this slot\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t};\n"
                    else:
                        snd = [":: when(copies == 0) sync\n"]
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies == {c}) sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, 2: copies -= ack{snd_contact.to} == {node} ? {c} : 0 =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")
                        node_process += f"\talt {{ //slot communicate with {snd_contact.to} at ts {snd_contact.ts}\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t};\n"

                elif rcv_contact:
                    #I think that it has to have the nop option always when
                    rcv = f":: sync {{= 1: copies += dest{rcv_contact.from_} == {node} ? data{rcv_contact.from_} : 0, 1: ack{node} = {rcv_contact.from_} =}}"
                    node_process += f"\t//slot communicate with {rcv_contact.from_} at ts {rcv_contact.ts}\n"
                    node_process += "\t" + rcv[3:] + ";\n"
                else:
                    node_process += "\tsync;\n"

            node_process = node_process[:node_process.rindex(";")] + node_process[node_process.rindex(";")+1:] + "}"
            if node == target:
                lines = node_process.split("\n")
                body = lines[2:-1]
                body = map(lambda l: "\t" + l, body)
                node_process = lines[0] + "\n{\n" + "\twith(success == (copies != 0))\n" + "\t{\n" + "\n".join(body) + ";\n\t\tstop\n" + "\t}" + "\n}"
            processes[node] = node_process

        #Delete nodes that do not have any transmission
        deleted_nodes = 0
        for node in range(self.num_of_nodes):
            if "//slot communicate with" not in processes[node]:
                del processes[node]
                deleted_nodes += 1



        model = f"// Modest model ({copies} copies) created with brufn library in {date.today()}\n"
        model += f"// {len(self.contacts) - len(meaningful_contacts)} contacts have been deleted by considering reachability closure\n\n"
        model += f"// {deleted_nodes} nodes have been deleted\n\n"
        model += f"// {self.num_of_ts - len(meaningful_ts)} ts have been deleted\n\n"

        model += f"const int NODES = {self.num_of_nodes};\n"
        model += f"const int COPIES = {copies};\n\n"
        model += "action sync;\n"
        model += 'transient int(0..NODES) ' + ", ".join([f'dest{n}' for n in sorted(processes.keys())]) + ";\n"
        model += 'transient int(0..COPIES) ' + ", ".join([f'data{n}' for n in sorted(processes.keys())]) + ";\n"
        model += 'transient int(0..NODES) ' + ", ".join([f'ack{n}' for n in sorted(processes.keys())]) + ";\n\n"
        model += 'action ' + ", ".join([f'nop{n}' for n in sorted(processes.keys())]) + ";\n\n"
        model += 'transient bool success;\n'
        model += 'property PmaxSuccess = Pmax(<> success);\n\n'
        model += "\n\n".join([processes[n] for n in sorted(processes.keys())]) + "\n\n"
        model += "par {\n"
        model += "\n".join([f":: Node{n}({copies if n==source else 0})" for n in sorted(processes.keys())]) + "\n"
        model += "}"
        return model

    def __gen_modest_model_without_ack(self, source, target, pf, copies) -> str:
        rc = reachability_clousure(self)
        rc_from_source = self.reachability_clousure_from_source(source)
        meaningful_contacts = [c for c in self.contacts if c.to==target or (c.ts + 1 < self.num_of_ts and target in rc[c.to][c.ts + 1])]
        meaningful_contacts = [c for c in meaningful_contacts if c.from_ in rc_from_source[c.ts-1]]
        #meaningful_contacts.sort(key=lambda c: c.ts)
        meaningful_ts = sorted(list(set(c.ts for c in meaningful_contacts)))
        processes = {}
        for node in range(self.num_of_nodes):
            node_process = f"process Node{node}(int(0..{copies}) copies)\n"
            node_process += "{\n"
            for ts in meaningful_ts:
                snd_contact = list(filter(lambda c: c.from_==node and c.ts == ts, meaningful_contacts))
                rcv_contact = list(filter(lambda c: c.to==node and c.ts == ts, meaningful_contacts))
                assert len(snd_contact) <= 1, "No more than 1 outgoing transmission is allowed"
                assert len(rcv_contact) <= 1, "No more than 1 ingoing transmission is allowed"
                assert not snd_contact or not rcv_contact or snd_contact[0].to == rcv_contact[0].from_, "If there are a snd and rcv, it has to be with the same node (full duplex communication)"

                snd_contact = snd_contact[0] if snd_contact else None
                rcv_contact = rcv_contact[0] if rcv_contact else None
                nop_snd = ts + 1 < self.num_of_ts and target in rc[node][ts + 1] if snd_contact else False
                nop_rcv = ts + 1 < self.num_of_ts and target in rc[rcv_contact.from_][ts + 1] if rcv_contact else False
                if snd_contact and rcv_contact:
                    snd = []
                    if nop_snd:
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies >= {c}) tau; sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, copies -= {c} =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")

                        rcv = f":: tau; sync {{= 1: copies += dest{rcv_contact.from_} == {node} ? data{rcv_contact.from_} : 0=}}\n"
                        node_process += f"\talt {{ //slot communicate with {rcv_contact.from_} at ts {rcv_contact.ts}\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t\t" + rcv + "\n"
                        node_process += "\t};\n"
                    else:
                        assert False, "There must not be a contact incoming to a node that can not reach the target"
                elif snd_contact:
                    snd = []
                    if nop_snd:
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies >= {c}) tau; sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, copies -= {c} =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")

                        node_process += f"\talt {{ //slot communicate with {snd_contact.to} at ts {snd_contact.ts}\n"
                        node_process += f"\t\t:: tau; sync //do nothing in this slot\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t};\n"
                    else:
                        snd = [":: when(copies == 0) sync\n"]
                        for c in range(1, copies + 1):
                            snd.append(f":: when(copies == {c}) sync palt{{ :{1-pf:.2f}: {{= data{node} = {c}, dest{node} = {snd_contact.to}, copies -= {c} =}} :{pf:.2f}: {{= /* lost */ =}}}}\n")
                        node_process += f"\talt {{ //slot communicate with {snd_contact.to} at ts {snd_contact.ts}\n"
                        node_process += "\t\t" + "\t\t".join(snd)
                        node_process += "\t};\n"

                elif rcv_contact:
                    #I think that it has to have the nop option always when
                    rcv = f":: sync {{= 1: copies += dest{rcv_contact.from_} == {node} ? data{rcv_contact.from_} : 0 =}}"
                    node_process += f"\t//slot communicate with {rcv_contact.from_} at ts {rcv_contact.ts}\n"
                    node_process += "\t" + rcv[3:] + ";\n"
                else:
                    node_process += "\tsync;\n"

            node_process = node_process[:node_process.rindex(";")] + node_process[node_process.rindex(";")+1:] + "}"
            if node == target:
                lines = node_process.split("\n")
                body = lines[2:-1]
                body = map(lambda l: "\t" + l, body)
                node_process = lines[0] + "\n{\n" + "\twith(success == (copies != 0))\n" + "\t{\n" + "\n".join(body) + ";\n\t\tstop\n" + "\t}" + "\n}"
            processes[node] = node_process

        #Delete nodes that do not have any transmission
        deleted_nodes = 0
        for node in range(self.num_of_nodes):
            if "//slot communicate with" not in processes[node]:
                del processes[node]
                deleted_nodes += 1



        model = f"// Modest model ({copies} copies) created with brufn library in {date.today()}\n"
        model += f"// {len(self.contacts) - len(meaningful_contacts)} contacts have been deleted by considering reachability closure\n\n"
        model += f"// {deleted_nodes} nodes have been deleted\n\n"
        model += f"// {self.num_of_ts - len(meaningful_ts)} ts have been deleted\n\n"

        model += f"const int NODES = {self.num_of_nodes};\n"
        model += f"const int COPIES = {copies};\n\n"
        model += "action sync;\n"
        model += 'transient int(0..NODES) ' + ", ".join([f'dest{n}' for n in sorted(processes.keys())]) + ";\n"
        model += 'transient int(0..COPIES) ' + ", ".join([f'data{n}' for n in sorted(processes.keys())]) + ";\n"
        #model += 'transient int(0..NODES) ' + ", ".join([f'ack{n}' for n in sorted(processes.keys())]) + ";\n\n"
        #model += 'action ' + ", ".join([f'nop{n}' for n in sorted(processes.keys())]) + ";\n\n"
        model += 'transient bool success;\n'
        model += 'property PmaxSuccess = Pmax(<> success);\n\n'
        model += "\n\n".join([processes[n] for n in sorted(processes.keys())]) + "\n\n"
        model += "par {\n"
        model += "\n".join([f":: Node{n}({copies if n==source else 0})" for n in sorted(processes.keys())]) + "\n"
        model += "}"
        return model


    def reachability_clousure_from_source(self, source:int)->Dict:
        '''
        computes f:ts -> {int} st f(t) indicates the set of nodes reachables by source at the end of ts t

        :param source:
        :return:
        '''
        f = {-1:{source}}
        for ts in range(0, self.num_of_ts):
            f[ts] = copy(f[ts-1])
            neighbors = [n for n in f[ts]]
            ts_contacts = list(filter(lambda c: c.ts==ts, self.contacts))
            while neighbors:
                n = neighbors.pop(0)
                for c in filter(lambda c: c.from_ == n,ts_contacts):
                    if c.to not in f[ts]:
                        f[ts].add(c.to)
                        neighbors.append(c.to)
            if all(n in f[ts] for n in range(self.num_of_nodes)):
                break
        if ts < self.num_of_ts - 1:
            for ts in range(ts+1, self.num_of_ts):
                f[ts] = set(range(self.num_of_nodes))

        return f

class SoftState:

    def __init__(self, states, ts,  id=None, max_success_transition:Dict[str, 'SoftTransition']=None, max_success_pr=None, max_success_transition_cost=None):
        self._states_: Tuple[int] = states
        self._ts_: int = ts

        self._transitions:List[SoftTransition] = None

        if max_success_transition is not None:
            self._max_success_transition_ = dict((k, max_success_transition[k] if max_success_transition[k] is not None else None) for k in max_success_transition.keys())
        else:
            max_success_transition = None

        if id is None:
            self._id_ = SoftState.get_identifier(states, ts)
        else:
            self._id_ = id

        if max_success_pr is None and max_success_transition is not None:
            self._max_success_pr_ = dict((k, max_success_transition[k].get_probability(pf=k)) for k in max_success_transition.keys())
        else:
            self._max_success_pr_ = max_success_pr

        if max_success_transition_cost is None and max_success_transition is not None:
            self._cost_ = dict((k,  max_success_transition[k].get_cost(pf=k)) for k in max_success_transition.keys())
        else:
            self._cost_ = max_success_transition_cost

    def drop_information(self):
        del self._max_success_transition_

    def get_carrier_nodes(self) -> List[int]:
        return [x for x in range(len(self._states_)) if self._states_[x] > 0]

    def num_of_carrying_copies(self, node: int) -> int:
        return self._states_[node]

    def gen_previous_state(self, rules: List['Rule']) -> Tuple[int]:
        previous = [x for x in self._states_]
        for rule in rules:
            previous[rule.sender_node] += rule.copies
            previous[rule.receiver_node] -= rule.copies

        return previous

    @property
    def transitions(self):
        return self._transitions_

    @property
    def num_of_nodes(self):
        return len(self._states_)

    @property
    def ts(self):
        return self._ts_

    @property
    def states(self):
        return self._states_

    @property
    def id(self):
        return self._id_

    def max_success_transition(self, pf: float = -1) -> 'SoftTransition':
        return self._max_success_transition_[str(pf)]

    def get_probability(self, pf:float = -1) -> float:
        return self._max_success_pr_[str(pf)]

    def get_cost(self, pf=-1):
        return self._cost_[str(pf)]

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        if isinstance(other, SoftState):
            return self._id_ == other.id
        raise TypeError('Comparing SoftState with %s'%(type(other)))

    def __lt__(self, other) -> bool:
        """Overrides the default implementation"""
        if isinstance(other, SoftState):
            return self._id_ < other.id
        raise TypeError('Comparing SoftState with %s'%(type(other)))

    def to_dict(self, debug=False):
        res = {}
        res['states'] = self._states_
        res['id'] = self._id_
        res['ts'] = self._ts_
        if debug:
            res['transitions'] = self._transitions_

        for pf in self._max_success_transition_.keys():
            res[f'sdp_pf={pf}'] = self._max_success_pr_[pf]
            res[f'best_t_pf={pf}'] = (lambda x: x.to_dtnsim_action() if x is not None else None)(self._max_success_transition_[pf])
            res[f'best_t_cost={pf}'] = self._cost_[pf]
            res[f't_changes_pf={pf}'] = (lambda x: [change.to_state for change in x.change_list] if x is not None else None)(self._max_success_transition_[pf])
            res[f't_changes_proba_pf={pf}'] = (lambda x: [change.get_probability(pf=pf) for change in x.change_list] if x is not None else None)(self._max_success_transition_[pf])
            res[f't_cost_pf={pf}'] = self.get_cost(pf=pf)

        return res

    @staticmethod
    def get_identifier(copies: List[int], ts: int) -> int:
        num_of_copies = sum(copies)
        l = [ts * (num_of_copies + 1) ** len(copies)]
        l += [copies[i] * (num_of_copies + 1) ** (len(copies) - i - 1) for i in range(len(copies))]
        return sum(l)

    @staticmethod
    def get_new_state_by_identifier(id: int, ts_number: int, nodes_number: int, num_of_copies: int) -> 'SoftState':
        it = bounded_iterator(nodes_number, num_of_copies)
        for values in it:
            values = dict(values); values = tuple(0 if x not in values.keys() else values[x] for x in range(nodes_number))
            for ts in range(ts_number):
                if SoftState.get_identifier(values, ts) == id:
                    return SoftState(values, ts, id=id)

        raise ValueError('Identifier %d was not find in the states set generated by: Nodes: %d - Time Stamps: %d - Num of copies: %d' % (id, nodes_number, ts_number, num_of_copies))


class Rule(ABC):

    def __init__(self, num_of_copies:int, route:Route):
        self._copies = num_of_copies
        self._route = route

    @property
    def sender_node(self) -> int:
        return self._route.sender_node

    @property
    def receiver_node(self) -> int:
        return self._route.receiver_node

    @property
    def copies(self) -> int:
        return self._copies

    def get_contacts(self):
        return self._route.contacts

    def get_contacts_ids(self):
        return (c.id for c in self.get_contacts())

    @property
    def route(self):
        return self._route

    def to_SoftRule(self):
       return SoftRule.make_from_rule(self)

    def hop_count(self):
        return self._route.hop_count()


class SoftRule:

    def __init__(self, copies, route:SoftRoute):
        self._copies_:int = copies
        self._route_: SoftRoute = route

    @staticmethod
    def make_from_rule(rule: Rule):
        return SoftRule(rule.copies, rule.route.to_softroute())

    def to_tuple(self) -> Tuple[int,Tuple[int]]:
        res = tuple( zip(self._copies_, self._route_) )
        return [(x[0], x[1].contacts) for x in res]

    def to_dtnsim_rule(self) -> OrderedDict:
        dtnsim_rule = OrderedDict()
        dtnsim_rule['copies'] = self._copies_
        dtnsim_rule['name'] = 'send%d_%d-toward:%s' % (self.sender_node, self.receiver_node, self._route_.contacts)
        dtnsim_rule['contact_ids'] = [c + 1 for c in self._route_.contacts]
        dtnsim_rule['source_node'] = self._route_.sender_node + 1

        return dtnsim_rule

    @property
    def sender_node(self) -> int:
        return self._route_.sender_node

    @property
    def receiver_node(self) -> int:
        return self._route_.receiver_node

    @property
    def copies(self) -> int:
        return self._copies_

    def get_SoftRule_from_dict(self, save_dict):
        self._copies_:int = save_dict['copies']
        self._route_: SoftRoute = rule.route.to_softroute()


class SoftNextRule(SoftRule):

    def __init__(self, node: int, num_of_copies: int):
        self._node_ = node
        self._route_ = None
        self._copies_ = num_of_copies

    @property
    def sender_node(self) -> int:
        return self._node_

    @property
    def receiver_node(self) -> int:
        return self._node_

    def to_dtnsim_rule(self) -> OrderedDict:
        dtnsim_rule = OrderedDict()
        dtnsim_rule['copies'] = self.copies
        dtnsim_rule['name'] = 'next'
        dtnsim_rule['contact_ids'] = []
        dtnsim_rule['source_node'] = self._node_ + 1

        return dtnsim_rule

    def __str__(self):
        return "Next"

    def hop_count(self):
        return 0

    @property
    def sender_node(self) -> int:
        return self._node_

    @property
    def receiver_node(self) -> int:
        return self._node_

    def get_contacts(self):
        return []

    def to_SoftRule(self):
        return self

class ChangeCase:
    '''
    A change case is described by contacts and when it fails or work:
        _case[i] stored 0 if _contacts[i] is considered to fail in this case or
        1 otherwise.
    '''

    def __init__(self, contacts, case):
        self._contacts: Tuple[Contact] = contacts
        self._case: Tuple[int] = case

        self._pr_result: Dict[str, float] = None #Store the probability of this ChangeCase happen. Dict or float

    def get_probability(self, pf:float = -1):
        if self._pr_result is None:
            raise TypeError("ChangeCase:get_probability was called but it has not been computed yet.")

        return self._pr_result[str(pf)]

    def compute_probability(self) -> float:
        prs = [1 - self._contacts[i].pf if self._case[i] else self._contacts[i].pf for i in range(len(self._contacts))]
        self._pr_result = {'-1': reduce(lambda x, y: x * y, prs, 1.)}

        return self._pr_result['-1']

    def compute_probability_rng(self, pr_rng) -> Dict:
        '''
        :param pr_rng: A range in which each individual value is the failure probability for all links
        '''
        n_of_working_links = sum(self._case)
        self._pr_result = dict([(str(pf), pf ** (len(self._case) - n_of_working_links) * (1-pf) ** n_of_working_links) for pf in pr_rng])

        return self._pr_result

    def get_failure_links(self) -> Tuple[Contact]:
        return tuple(self._contacts[i] for i in range(len(self._contacts)) if not self._case[i])


class Change:
    '''
    A change is a possible update in a transition. It is described by the update,
    it means which state will be gotten and for a list of cases that stored all cases in which
    this change is performed. The Change probability is computed by summing all cases probability
    '''

    def __init__(self, to_state):
        self._cases: List[ChangeCase] = []
        self._to_state: SoftState = to_state

        self._pr_result = None

    def add_case(self, case):
        self._cases.append(case)

    def get_probability(self, pf: float = -1):
        if self._pr_result is None:
            raise TypeError('Change:get_probability was called but it has not been computed yet')
        return self._pr_result[str(pf)]

    def compute_probability(self) -> float:
        self._pr_result = {'-1': sum([case.compute_probability() for case in self._cases])}
        return self._pr_result ['-1']

    def compute_probability_rng(self, pr_rng) -> Dict:
        '''
        :param pr_rng:
        :return:
        '''
        self._pr_result = dict([(str(pf), 0) for pf in pr_rng])
        for cc in self._cases:
            cc_pr = cc.compute_probability_rng(pr_rng)
            for pf in pr_rng:
                self._pr_result[str(pf)] += cc_pr[str(pf)]

        return self._pr_result

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Change):
            return self.id == other.id
        return False

    @property
    def id(self):
        return self._to_state.id

    @property
    def to_state(self):
        return self._to_state

    @property
    def cases(self):
        return tuple(self._cases)

    def to_SoftChange(self):
        return SoftChange(self)


class SoftChange:
    def __init__(self, change):
        self._to_state_: int = change.to_state.id
        self._pr_result_: Dict[str, float] = change._pr_result #it could be float or dict

    def get_probability(self, pf: float = -1):
        return self._pr_result_[str(pf)]

    @property
    def to_state(self):
        return self._to_state_


class Transition:
    '''
    A transition is a tuple of Rules
    '''

    def __init__(self, from_, to, rules):
        self._from: SoftState = from_ #from_.to_softstate(solved_transition_required=False)
        self._to: SoftState = to
        self._rules: Tuple[Rule] = tuple(rules)
        self._changes_list: Dict[int, Change] = {} #SortedList(key=lambda change: change.id)
        self._cost: Dict[str, int] = None
        self._pr_result: Dict[str, float] = None

    def get_cost(self, pf:float = -1) -> int:
        if self._cost is None:
            raise TypeError('Transition:get_cost was called but it has not been computed yet')

        return self._cost[str(pf)]

    def _add_change_case(self, change_case, reachable_states):
        to_state = self._compute_tostate(change_case, reachable_states)
        if to_state is not None:
            change = Change(to_state)
            if change.id in self._changes_list.keys():
                change = self._changes_list[change.id]
            else:
                self._changes_list[change.id] = change

            change.add_case(change_case)

    def _compute_tostate(self, change_case, reachable_states):
        '''
        Returns the state that will be reached if the change_case happen

        :param change_case:
        :return: The State that will be reached if the change_case happen
        '''

        to_status = [i for i in self.from_.states]
        failure_set = [c.id for c in change_case.get_failure_links()]

        for rule in self._rules:
            for c in rule.get_contacts():
                if c.id in failure_set:
                    break
                else:
                    to_status[c.from_] -= rule.copies
                    to_status[c.to] += rule.copies

        to_status_id = SoftState.get_identifier(to_status, self.to.ts)
        if to_status_id in reachable_states.value:
            return reachable_states.value[to_status_id]
        else:
            return None

        return state_factory.create_state(to_status, self.to.ts, allow_creation=False)

    def compute_changes(self, reachable_states):
        '''
        Compute all reachable states by considering all possible failures

        :param reachable_states: A list with states from which successful states can be reached
        :param state_factory:
        :return: None. Its save the information inside transition object
        '''

        contacts_used = []
        contacts_used_ids = []
        for rule in self._rules:
            for c in rule.get_contacts():
                if c.id not in contacts_used_ids:
                    contacts_used_ids.append(c.id)
                    contacts_used.append(c)

        #generate all diferent cases
        change_contacts = tuple(contacts_used)
        for fs in itertools.product(*[[0,1] for i in range(len(contacts_used_ids))]):
            case = ChangeCase(change_contacts, fs)
            self._add_change_case(case, reachable_states)

    def get_probability(self, pf = -1):
        if self._pr_result is None:
            raise TypeError('Transition:get_probability was called but it has not been computed yet')
        return self._pr_result[str(pf)]

    def compute_or_get_probability(self, reachable_states, pf=-1):
        if self._pr_result is None:
            self.compute_probability(reachable_states)
        return self._pr_result[str(pf)]

    def compute_probability(self, reachable_states) -> float:
        self.compute_changes(reachable_states)
        cost = sum(rule.hop_count() for rule in self._rules)
        pr_result = 0.
        for change in self._changes_list.values():
            pr_result += change.compute_probability() * change.to_state.get_probability()
            cost += change.compute_probability() * change.to_state.get_cost()

        self._cost = {'-1': cost}
        self._pr_result = {'-1': pr_result}
        return pr_result

    def compute_probability_rng(self, reachable_states, pr_rng) -> Dict:
        self.compute_changes(reachable_states)
        t_cost = sum(rule.hop_count() for rule in self._rules)
        self._pr_result = dict([(str(pf),0.) for pf in pr_rng])
        self._cost = dict([(str(pf), t_cost) for pf in pr_rng])

        for change in self._changes_list.values():
            change_pf_rng = change.compute_probability_rng(pr_rng)
            for pf in pr_rng:
                self._pr_result[str(pf)] += change_pf_rng[str(pf)] * change.to_state.get_probability(pf=pf)
                self._cost[str(pf)] += change_pf_rng[str(pf)] * change.to_state.get_cost(pf)

        return self._pr_result

    @property
    def from_(self):
        return self._from

    @property
    def to(self):
        return self._to

    @property
    def rule(self) -> Tuple[Rule]:
        return self._rules

    def to_SoftTransition(self):
        return SoftTransition(self)


class SoftTransition:

    def __init__(self, transition: Transition):
        self._from_: int = transition.from_.id
        self._to_: int = transition.to.id
        self._rules_: Tuple[SoftRule] = tuple([r.to_SoftRule() for r in transition.rule])
        self._changes_list_: tuple[SoftChange] = tuple([r.to_SoftChange() for r in transition._changes_list.values()])
        self._cost_ = copy(transition._cost)
        self._pr_result = copy(transition._pr_result)

    @property
    def rules(self) -> Tuple[SoftRule]:
        return self._rules_

    @property
    def change_list(self) -> Tuple[SoftChange]:
       return self._changes_list_

    def to_tuple_rules(self):
        l = []
        for rule in self._rules_:
            if type(rule) != SoftNextRule:
                l.append(rule.to_tuple())

        return tuple(itertools.chain.from_iterable(l))

    def to_dtnsim_action(self) -> List[str]:
        return list([rule.to_dtnsim_rule() for rule in self._rules_])

    def get_cost(self, pf: float = -1):
        return self._cost_[str(pf)]

    def get_probability(self, pf = -1):
        if self._pr_result is None:
            raise TypeError('Transition:get_probability was called but it has not been computed yet')
        return self._pr_result[str(pf)]

    def to_SoftTransition(self):
        return self

def bounded_iterator(bins, bound):
    assert bins > 0 and bound > 0, f'bins and bound must be grether than 0'
    prefix = [[]]
    for sp in range(0, bins):
        # print(num_simple_path - i)
        new_prefixs = []
        for p in prefix:
            assigned:int = sum([x[1] for x in p])
            new_prefixs.append(p[:])
            for c in range(1, bound - assigned):
                new_prefixs.append(p + [(sp, c)])
            yield p + [(sp, bound - assigned)]
        prefix = new_prefixs