import unittest
import tempfile
from brufn.network import Net
import os
from pprint import pprint
from pyspark import SparkContext, SparkConf
from brufn.brufspark import BRUFSpark
from brufn.helper_ibrufn_function_generator import IBRUFNFunctionGenerator
import simplejson as json
from copy import deepcopy
from brufn.utils import print_str_to_file

class TestIBRUFNFuntionGenerator(unittest.TestCase):

    def setUp(self) -> None:
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[1]')
                 .set('spark.executor.memory', '3G')
                 .set('spark.driver.memory', '16G')
                 .set('spark.driver.maxResultSize', '12G'))
        self.sc = SparkContext(conf=conf)

    def tearDown(self) -> None:
        self.sc.stop()

    def test_IBRUFNFuntionGenerator1(self):
        net = '''
NUM_OF_NODES = 4
CONTACTS = [
    {'from': 0, 'to': 1, 'ts': 0, 'pf':0.5},

    {'from': 0, 'to': 1, 'ts': 1, 'pf': 0.5},
    {'from': 0, 'to': 2, 'ts': 1, 'pf': 0.5},

    {'from': 1, 'to': 3, 'ts': 2, 'pf': 0.5},
    {'from': 2, 'to': 3, 'ts': 2, 'pf': 0.5}
]
        '''
        expected_func_1 = {1:
                           {0: {'1:1':[1], '2:1':[4], '3:1':[5]},
                            1: {'1:1':[3], '2:1':[4], '3:1':[5]},
                            2: {'2:1':[4], '3:1':[5]}},
                         2:
                           {0: {'1:2': [1,3], '2:2': [4,4], '3:2': [5,5]},
                            1: {'1:2': [2, 3], '2:2': [4, 4], '3:2': [5, 5]},
                            2: {'2:2': [4, 4], '3:2': [5, 5]}}
                         }
        expected_func_2 = deepcopy(expected_func_1)[1][1]['1:1'] = [2] #Both function could be a correct answer

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        working_dir = 'aux'
        copies = 2
        os.makedirs(os.path.join(working_dir, 'pf=0.50'), exist_ok=True)
        for c in range(1, copies+1):
            wd_copies = os.path.join(working_dir, f'to-{3}', f'BRUF-{c}', 'states_files')
            os.makedirs(wd_copies, exist_ok=True)
            brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != 3], 3, c, [0.5], wd_copies)
            brufn.compute_bruf(self.sc)

        func = IBRUFNFunctionGenerator(net, 3, copies, working_dir, 0.5).generate()
        self.assertTrue(func == expected_func_1 or func == expected_func_2)


    def test_IBRUFNFuntionGenerator2(self):
        net = '''
NUM_OF_NODES = 5
CONTACTS = [
    {'from': 0, 'to': 1, 'ts': 0, 'pf':0.5},

    {'from': 0, 'to': 1, 'ts': 1, 'pf': 0.5},
    {'from': 0, 'to': 2, 'ts': 1, 'pf': 0.5},

    {'from': 1, 'to': 3, 'ts': 2, 'pf': 0.5},
    {'from': 2, 'to': 3, 'ts': 2, 'pf': 0.5},
    {'from': 3, 'to': 4, 'ts': 2, 'pf': 0.5},
]
        '''
        targets = [3, 4]
        expected_func_1 = {}
        expected_func_1[3] = {1:
                                {0: {'1:1':[1], '2:1':[4], '3:1':[5]},
                                1: {'1:1':[3], '2:1':[4], '3:1':[5]},
                                2: {'2:1':[4], '3:1':[5]}},
                         2:
                               {0: {'1:2': [1,3], '2:2': [4,4], '3:2': [5,5]},
                                1: {'1:2': [2, 3], '2:2': [4, 4], '3:2': [5, 5]},
                                2: {'2:2': [4, 4], '3:2': [5, 5]}}
                         }
        expected_func_1[4] = {1:
                                   {0: {'1:1': [1], '2:1': [4], '3:1': [5], '4:1':[6]},
                                    1: {'1:1': [3], '2:1': [4], '3:1': [5], '4:1':[6]},
                                    2: {'2:1': [4], '3:1': [5], '4:1':[6]}},
                               2:
                                   {0: {'1:2': [1, 3], '2:2': [4, 4], '3:2': [5, 5], '4:2':[6,6]},
                                    1: {'1:2': [2, 3], '2:2': [4, 4], '3:2': [5, 5], '4:2':[6,6]},
                                    2: {'2:2': [4, 4], '3:2': [5, 5], '4:2':[6,6]}}
                               }

        expected_func_2 = {}
        expected_func_2[3] = deepcopy(expected_func_1[3])[1][1]['1:1'] = [2] #Both function could be a correct answer
        expected_func_2[4] = deepcopy(expected_func_1[4])[1][1]['1:1'] = [2] #Both function could be a correct answer


        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        working_dir = 'aux'
        copies = 2
        os.makedirs(os.path.join(working_dir, 'pf=0.50'), exist_ok=True)
        for target in targets:
            for c in range(1, copies+1):
                wd_copies = os.path.join(working_dir, f'to-{target}', f'BRUF-{c}', 'states_files')
                os.makedirs(wd_copies, exist_ok=True)
                brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != target], target, c, [0.5], wd_copies)
                brufn.compute_bruf(self.sc)

        for target in targets:
            func = IBRUFNFunctionGenerator(net, target, copies, working_dir, 0.5).generate()
            for c in range(1, copies+1):
                print_str_to_file(json.dumps(func[c]), os.path.join(working_dir, f'todtnsim-{target}-{c}-0.00.json'))
            self.assertTrue(func == expected_func_1[target] or func == expected_func_2[target])

if __name__ == '__main__':
    unittest.main()
