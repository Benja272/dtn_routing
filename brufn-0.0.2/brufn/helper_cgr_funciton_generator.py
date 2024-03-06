from brufn.network import SoftState, Net
from brufn.brufspark import BRUFSpark
from typing import List

class OneCopyHelperFunctionGenerator:

    def __init__(self, net:Net, target:int, carriers_nodes:List[int], num_of_nodes: int, num_of_ts:int, working_dir:str, failure_probability:float = None):
        self._target = target
        self._working_dir = working_dir
        self._carrier_nodes = tuple(carriers_nodes)
        self._num_of_nodes = num_of_nodes
        self._num_of_ts = num_of_ts
        self._failure_probability = failure_probability
        self._brufn = BRUFSpark(net, [], target, 1, [failure_probability], working_dir)


    def _future_delivery_probability(self, node:int, from_ts:int):
        '''' This function computes the delivery probability of node for t' >= time '''
        try:
            copies = self._gen_list0_except_n(node)
            state = self._brufn.get_state_by_id(SoftState.get_identifier(copies, from_ts))
            print('[Open] State %s - %d'%(copies, from_ts))
            if self._failure_probability is None:
                return state['sdp_pf=-1']
            else:
                return state[f'sdp_pf={self._failure_probability}']
        except Exception as e:
            print("_future_delivery_probability: State was not found. It will consider 0 as SDP " + self._working_dir)

        return 0


    def generate(self):
        print("[Warning] Remember NOT to generate BRUF model using transitive closure")

        function = dict((ts, {}) for ts in range(self._num_of_ts))
        for ts in range(self._num_of_ts):
            for node in self._carrier_nodes:
                print('[Generating] from: %d - ts %d'%(node, ts))
                function[ts][node + 1] = self._future_delivery_probability(node, ts)
            function[ts][self._target + 1] = 1.

        return function


    def _gen_list0_except_n(self, n: int):
        n += 1
        return [0] * (n - 1) + [1] + [0] * (self._num_of_nodes - n)