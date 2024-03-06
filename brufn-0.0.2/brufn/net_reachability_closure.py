'''
This class is designed to compute the reachability clousure of nodes in the network.
It is different from the one computed by net_metrics_generators in the following sense:

    f(n, t, l): Node -> Ts -> [Nodes]
        The set of nodes that are reachable from Node n at ts <= t

    This script computes

    f(n,t,l): Node->Ts-> [Nodes]
        The set of nodes that are reachable from Node n at ts >= t
'''
#from brufn.network import Net
from copy import copy
from collections import OrderedDict

def reachability_clousure(net:'Net', max_hops_route_in_ts=-1):
    rc = OrderedDict((source, OrderedDict()) for source in range(net.num_of_nodes))
    for ts in range(net.num_of_ts - 1, -1, -1):
        for source in range(net.num_of_nodes):
            rc[source][ts] = {source}
            # The cost could be reduced bc we are getting all simple path and just one is needed
            single_paths_to = dict([(target, net.compute_routes(source, target, ts)) for target in range(net.num_of_nodes) if target != source])
            if max_hops_route_in_ts>=0:
                for target in [x for x in range(net.num_of_nodes) if x != source]:
                    single_paths_to[target] = list(filter(lambda r: r.hop_count()<=max_hops_route_in_ts, single_paths_to[target]))

            for target in single_paths_to.keys():
                if len(single_paths_to[target]) > 0:
                    rc[source][ts].add(target)

            if ts+1 < net.num_of_ts:
                previous_rc_nodes = copy(rc[source][ts])
                for x in previous_rc_nodes:
                    rc[source][ts].update(rc[x][ts+1])


    return rc


