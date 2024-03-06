import unittest
from brufn.brufspark import BRUFSpark
from brufn.network import Net, SoftState
from brufn.test.test_utils import gen_binomial_net_level_by_ts
from brufn.net_metrics_generator import NetMetricGenerator
from brufn.brufspark import BRUFSpark
from pyspark import SparkContext, SparkConf
import os
import json

class Test_MC_To_Dot(unittest.TestCase):


    def test_1(self):
        copies = 2; levels=4
        #net = gen_binomial_net_level_by_ts(levels)
        rucop_dir = os.path.join('Test_MC_To_Dot', f'levels={levels}', f'BRUF-{copies}')
        os.makedirs(rucop_dir, exist_ok=True)
        #net.print_to_file(os.path.join('Test_MC_To_Dot', f'levels={levels}'))
        net = Net.get_net_from_file(os.path.join('Test_MC_To_Dot', f'levels={levels}', 'net.py'), contact_pf_required=False)
        NetMetricGenerator(net, range(net.num_of_nodes), [1, 2, 3], [0], rucop_dir)

        reachability_closure = json.load(open(os.path.join(rucop_dir, 'transitive_closure.json')),
                                         object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in d.items()})

        conf = SparkConf().setAppName("BRUF-Spark")
        conf = (conf.setMaster('local[2]')
                .set('spark.executor.memory', '2G')
                .set('spark.driver.memory', '4G')
                .set('spark.driver.maxResultSize', '8G'))
        sc = SparkContext(conf=conf)
        bruf = BRUFSpark(net, [0], net.num_of_nodes - 1, copies,
                          [x / 100. for x in range(0, 110, 10)], rucop_dir)
        bruf.compute_bruf(sc, reachability_closure=reachability_closure)
        sc.stop()
        root = SoftState.get_identifier([copies] + [0] * (net.num_of_nodes-1), 0)
        bruf.mc_to_dot(root, os.path.join(rucop_dir,'mc.dot'), pf=.1)





