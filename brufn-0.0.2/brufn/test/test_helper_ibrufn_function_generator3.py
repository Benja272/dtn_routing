import unittest
import tempfile
from brufn.network import Net
import os
from pprint import pprint
from pyspark import SparkContext, SparkConf
from brufn.brufspark import BRUFSpark
from brufn.helper_ibrufn_function_generator3 import IBRUFNFunctionGenerator
import simplejson as json
from copy import deepcopy
from brufn.utils import print_str_to_file
import shutil
from pprint import pprint

class TestIBRUFNFuntionGenerator3(unittest.TestCase):

    WORKING_DIR = 'aux'

    def setUp(self) -> None:
        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[1]')
                 .set('spark.executor.memory', '3G')
                 .set('spark.driver.memory', '16G')
                 .set('spark.driver.maxResultSize', '12G'))
        self.sc = SparkContext(conf=conf)
        os.makedirs(os.path.join(self.WORKING_DIR), exist_ok=True)

    def tearDown(self) -> None:
        self.sc.stop()
        shutil.rmtree(self.WORKING_DIR)

    def test_IBRUFNFuntionGeneratorRng1(self):
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
        expected_func_1 = {1:{'0.5':
                               {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}}},
                         2:{'0.5':
                           {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}}
                         }
        expected_func_2 = deepcopy(expected_func_1)[1]['0.5'][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for c in range(1, copies+1):
            wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{3}')
            os.makedirs(wd_copies, exist_ok=True)
            brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != 3], 3, c, [0.5], wd_copies)
            brufn.compute_bruf(self.sc)

        func = IBRUFNFunctionGenerator(net, 3, copies, self.WORKING_DIR, [0.5], report_folder='').generate()
        pprint(func)
        self.assertTrue(func == expected_func_1 or func == expected_func_2)


    def test_IBRUFNFuntionGeneratorRng2(self):
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
        expected_func_1[3] = {1:{'0.5':
                                       {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                        1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                        2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}}},
                         2:{'0.5':
                               {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                                1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                                2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}}
                         }
        expected_func_1[4] = {1:{'0.5':
                                       {0: {'1:1': [{'copies':1, 'route':[1]}], '2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]},
                                        1: {'1:1': [{'copies':1, 'route':[3]}], '2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]},
                                        2: {'2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]}}},
                               2:{'0.5':
                                       {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]},
                                        1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]},
                                        2: {'2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]}}}
                               }

        expected_func_2 = {}
        expected_func_2[3] = deepcopy(expected_func_1[3])[1]['0.5'][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer
        expected_func_2[4] = deepcopy(expected_func_1[4])[1]['0.5'][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer


        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for target in targets:
            for c in range(1, copies+1):
                wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{target}')
                os.makedirs(wd_copies, exist_ok=True)
                brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != target], target, c, [0.5], wd_copies)
                brufn.compute_bruf(self.sc)

        for target in targets:
            func = IBRUFNFunctionGenerator(net, target, copies, self.WORKING_DIR, [0.5], report_folder='').generate()
            for c in range(1, copies+1):
                print_str_to_file(json.dumps(func[c]), os.path.join(self.WORKING_DIR, f'todtnsim-{target}-{c}-0.00.json'))
            self.assertTrue(func == expected_func_1[target] or func == expected_func_2[target], f"Differs in routing to target node {target}")


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
                               {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}},
                         2:
                           {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}
                         }
        expected_func_2 = deepcopy(expected_func_1)[1][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for c in range(1, copies+1):
            wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{3}')
            os.makedirs(wd_copies, exist_ok=True)
            brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != 3], 3, c, [], wd_copies)
            brufn.compute_bruf(self.sc)

        func = IBRUFNFunctionGenerator(net, 3, copies, self.WORKING_DIR, report_folder='').generate()
        pprint(func)
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
                               {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}},
                            2:
                               {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                                1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                                2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}
                            }
        expected_func_1[4] = {1:
                                   {0: {'1:1': [{'copies':1, 'route':[1]}], '2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]},
                                    1: {'1:1': [{'copies':1, 'route':[3]}], '2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]},
                                    2: {'2:1': [{'copies':1, 'route':[4,6]}], '3:1': [{'copies':1, 'route':[5,6]}], '4:1':[{'copies':1, 'route':[6]}]}},
                               2:
                                   {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]},
                                    1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]},
                                    2: {'2:2': [{'copies':2, 'route':[4,6]}], '3:2': [{'copies':2, 'route':[5,6]}], '4:2':[{'copies':2, 'route':[6]}]}}
                               }

        expected_func_2 = {}
        expected_func_2[3] = deepcopy(expected_func_1[3])[1][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer
        expected_func_2[4] = deepcopy(expected_func_1[4])[1][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer


        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for target in targets:
            for c in range(1, copies+1):
                wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{target}')
                os.makedirs(wd_copies, exist_ok=True)
                brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != target], target, c, [], wd_copies)
                brufn.compute_bruf(self.sc)

        for target in targets:
            func = IBRUFNFunctionGenerator(net, target, copies, self.WORKING_DIR, report_folder='').generate()
            for c in range(1, copies+1):
                print_str_to_file(json.dumps(func[c]), os.path.join(self.WORKING_DIR, f'todtnsim-{target}-{c}-0.00.json'))
            self.assertTrue(func == expected_func_1[target] or func == expected_func_2[target], f"Differs in routing to target node {target}")

    def test_IBRUFNFuntionGeneratorRng_load_previously_computed_bruf(self):
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
        expected_func_1 = {1:{'0.5':
                               {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}}},
                         2:{'0.5':
                           {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}}
                         }
        expected_func_2 = deepcopy(expected_func_1)[1]['0.5'][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for c in range(1, copies+1):
            wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{3}')
            os.makedirs(wd_copies, exist_ok=True)
            brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != 3], 3, c, [0.5], wd_copies)
            brufn.compute_bruf(self.sc)

        func = IBRUFNFunctionGenerator(net, 3, 1, self.WORKING_DIR, [0.5], report_folder='').generate()
        #self.assertTrue(func == expected_func_1 or func == expected_func_2)
        os.makedirs(os.path.join(self.WORKING_DIR, 'IRUCoPn-1/routing_files/pf=0.50'), exist_ok=True)
        print_str_to_file(json.dumps(func[1]['0.5']), os.path.join(self.WORKING_DIR, 'IRUCoPn-1/routing_files/pf=0.50/todtnsim-3-1-0.50.json'))
        func = IBRUFNFunctionGenerator(net, 3, 2, self.WORKING_DIR, [0.5], path_to_load_bruf=self.WORKING_DIR, report_folder='').generate()
        self.assertTrue(func == expected_func_1 or func == expected_func_2)


    def test_IBRUFNFuntionGeneratorRng_load_previously_computed_bruf(self):
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
        expected_func_1 = {1:{'0.5':
                               {0: {'1:1':[{'copies':1, 'route':[1]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                1: {'1:1':[{'copies':1, 'route':[3]}], '2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]},
                                2: {'2:1':[{'copies':1, 'route':[4]}], '3:1':[{'copies':1, 'route':[5]}]}}},
                         2:{'0.5':
                           {0: {'1:2': [{'copies':1, 'route':[1]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            1: {'1:2': [{'copies':1, 'route':[2]}, {'copies':1, 'route':[3]}], '2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]},
                            2: {'2:2': [{'copies':2, 'route':[4]}], '3:2': [{'copies':2, 'route':[5]}]}}}
                         }
        expected_func_2 = deepcopy(expected_func_1)[1]['0.5'][1]['1:1'] = [{'copies':1, 'route':[2]}] #Both function could be a correct answer

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(net); f.flush()
            net = Net.get_net_from_file(f.name)

        copies = 2
        for c in range(1, copies+1):
            wd_copies = os.path.join(self.WORKING_DIR, f'BRUF-{c}', 'states_files', f'to-{3}')
            os.makedirs(wd_copies, exist_ok=True)
            brufn = BRUFSpark(net, [n for n in range(net.num_of_nodes) if n != 3], 3, c, [0.5], wd_copies)
            brufn.compute_bruf(self.sc)

        func = IBRUFNFunctionGenerator(net, 3, 1, self.WORKING_DIR, [0.5], report_folder='').generate()
        #self.assertTrue(func == expected_func_1 or func == expected_func_2)
        os.makedirs(os.path.join(self.WORKING_DIR, 'IRUCoPn-1/routing_files/pf=0.50'), exist_ok=True)
        print_str_to_file(json.dumps(func[1]['0.5']), os.path.join(self.WORKING_DIR, 'IRUCoPn-1/routing_files/pf=0.50/todtnsim-3-1-0.50.json'))
        func = IBRUFNFunctionGenerator(net, 3, 2, self.WORKING_DIR, [0.5], path_to_load_bruf=self.WORKING_DIR, report_folder='').generate()
        self.assertTrue(func == expected_func_1 or func == expected_func_2)


if __name__ == '__main__':
    unittest.main()
